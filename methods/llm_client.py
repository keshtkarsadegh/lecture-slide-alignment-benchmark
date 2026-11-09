"""The LLM endpoint, its configuration and the response cache.

The model and the transport settings are in config/llm.toml (in git). The endpoint's
address and the API key are not: the address is the environment variable LLM_API_BASE,
the key LLM_API_KEY or, failing that, the untracked file config/llm_credentials.toml
(one line: api_key = "..."). Both are set in the untracked cluster/env.local
(ACTIVATE.md), so the repository names no host. The address is not in the cache key
(the payload is), so where it points changes no cached answer.

THE CACHE KEY IS THE WHOLE REQUEST. Every answer is stored under the sha256 of the exact
payload sent to the server: model, both messages (so every text cap, packet size and
pacing sentence), temperature, max_tokens, response format and the thinking switch. Any
change to any setting is therefore a cache miss and a new call, never a replay of an
answer to a different question.

The text caps and max_tokens are in the key on purpose: without them, changing a cap
or a token budget would replay an earlier (possibly truncated) answer.

THE COST OF A CALL IS ITS SUCCESSFUL ATTEMPT (docs/TUNING_DESIGN.md 6). `OpenAICompatClient`
times each attempt from just before the request to just after the answer is read and
parsed; the rate-limit wait and the backoff sleep lie outside that window, and failed
attempts and failed calls are counted apart and never enter the cost. The duration is
stored with the answer, so a cached answer still carries what it cost.

Nothing in this module calls the API unless an `LLM` is built with a client; the tests
use a fake client or a fake transport.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config" / "llm.toml"
CREDENTIALS_PATH = REPO / "config" / "llm_credentials.toml"


class LLMCallFailed(RuntimeError):
    """The call never produced an answer (network, HTTP error, timeout after retries).

    Not cached: the model said nothing, so there is nothing to replay. An answer that is
    unusable (prose, wrong JSON, cut off) IS an answer: it is cached and the method falls
    back to the matrix argmax, recording `parse_failed`.

    `attempts` and `errors` (one error kind per failed attempt) say what was tried; they
    are counted as failures, never as cost.
    """

    def __init__(self, message: str, *, attempts: int = 0, errors: Sequence[str] = ()):
        super().__init__(message)
        self.attempts = attempts
        self.errors = list(errors)


class LLMUnavailable(RuntimeError):
    """The answer is not in the cache and no client was given (offline run)."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    temperature: float = 0.0
    disable_thinking: bool = True
    timeout_s: float = 300.0
    max_retries: int = 4
    requests_per_minute: float = 20.0
    cache_path: str = "data/llm_cache.sqlite3"


def load_settings(path: Path = CONFIG_PATH) -> LLMSettings:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    # The address may be unset: an offline replay reads only the cache; a client refuses
    # to start without it.
    return LLMSettings(base_url=os.environ.get("LLM_API_BASE", "").strip(), **raw["llm"],
                       cache_path=raw.get("cache", {}).get("path", LLMSettings.cache_path))


def api_base() -> str:
    """LLM_API_BASE from the environment: the OpenAI-compatible endpoint's address."""
    base = os.environ.get("LLM_API_BASE", "").strip()
    if not base:
        raise LLMUnavailable("no endpoint: set LLM_API_BASE (cluster/env.local, see "
                             "cluster/env.local.example)")
    return base


def api_key(path: Path = CREDENTIALS_PATH) -> str:
    """LLM_API_KEY from the environment, else from the untracked credentials file."""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if key:
        return key
    if path.exists():
        with open(path, "rb") as fh:
            key = str(tomllib.load(fh).get("api_key", "")).strip()
    if not key:
        raise LLMUnavailable(f"no API key: set LLM_API_KEY or write api_key = \"...\" "
                             f"into {path} (not in git)")
    return key


class ChatClient(Protocol):
    def chat(self, payload: dict) -> dict:
        """Send one chat-completions payload; return {"content", "finish_reason", "usage"}
        and, from a timing client, "duration_s", "attempts", "failed_attempts", "errors"."""


def http_post(request: urllib.request.Request, timeout: float) -> bytes:
    """The real transport: send the request, return the response body."""
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


class OpenAICompatClient:
    """An OpenAI-compatible /chat/completions endpoint, standard library only.

    Retries on 429, 5xx and network errors with exponential backoff; a client-side
    throttle keeps under `requests_per_minute` (the endpoint limits per account, so
    parallel jobs must share it). Honours https_proxy (compute nodes need the cluster's
    proxy).

    TIMING: each attempt is timed with `clock` from just before the request is sent to
    just after the answer is read and parsed. Only the successful attempt's time is
    returned as `duration_s`; the throttle and backoff sleeps are outside every attempt's
    window and are returned apart (`wait_s`), for information only. `transport`, `sleep`
    and `clock` are the real ones unless a test replaces them.
    """

    def __init__(self, settings: LLMSettings, key: str, *,
                 transport: Callable[[urllib.request.Request, float], bytes] = http_post,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter):
        if not settings.base_url:
            api_base()           # raises LLMUnavailable, naming where the address is set
        self.url = settings.base_url.rstrip("/") + "/chat/completions"
        self.key = key
        self.timeout = settings.timeout_s
        self.max_retries = settings.max_retries
        self.min_interval = 60.0 / settings.requests_per_minute if settings.requests_per_minute > 0 else 0.0
        self._last = 0.0
        self._transport, self._sleep, self._clock = transport, sleep, clock

    def _throttle(self) -> float:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        return max(wait, 0.0)

    def chat(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        last_error = "no attempt"
        errors: list[str] = []
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            if attempt:
                backoff = min(60.0, 2.0 * 1.6 ** attempt)
                self._sleep(backoff)
                waited += backoff
            waited += self._throttle()
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            t0 = self._clock()
            try:
                body = json.loads(self._transport(req, self.timeout).decode("utf-8"))
            except urllib.error.HTTPError as e:
                errors.append(f"HTTP {e.code}")
                last_error = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code == 429 or e.code >= 500:
                    continue
                raise LLMCallFailed(last_error, attempts=attempt + 1, errors=errors) from e
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                errors.append(type(e).__name__)
                last_error = f"{type(e).__name__}: {e}"
                continue
            try:
                choice = body["choices"][0]
                content = choice["message"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                errors.append("malformed response")
                raise LLMCallFailed(f"malformed response: {str(body)[:300]}",
                                    attempts=attempt + 1, errors=errors) from e
            duration = self._clock() - t0
            return {"content": content, "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage") or {}, "duration_s": duration,
                    "attempts": attempt + 1, "failed_attempts": attempt, "errors": errors,
                    "wait_s": waited}
        raise LLMCallFailed(f"no answer after {self.max_retries + 1} attempts: {last_error}",
                            attempts=self.max_retries + 1, errors=errors)


class ResponseCache:
    """Answers by request hash, in one SQLite file (one inode; safe for parallel jobs)."""

    def __init__(self, path: Path | str):
        p = Path(path)
        self.path = p if p.is_absolute() else REPO / p
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self, create: bool) -> Optional[sqlite3.Connection]:
        if self._conn is None:
            if not create and not self.path.exists():
                return None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, "
                               "method TEXT NOT NULL, record TEXT NOT NULL)")
        return self._conn

    def get(self, key: str) -> Optional[dict]:
        conn = self._connect(create=False)
        row = conn.execute("SELECT record FROM answers WHERE key=?", (key,)).fetchone() \
            if conn is not None else None
        return json.loads(row[0]) if row else None

    def put(self, key: str, method: str, record: dict) -> None:
        self._connect(create=True).execute(
            "INSERT OR REPLACE INTO answers (key, method, record) VALUES (?, ?, ?)",
            (key, method, json.dumps(record, ensure_ascii=False)))


def request_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


class LLM:
    """What the LLM methods talk to: builds a request, answers it from cache or client.

    Counts, in three groups that are never mixed (docs/TUNING_DESIGN.md 6):
    - paid in this run: `calls` (successful calls made now), `call_seconds` (the sum of
      their successful attempts' durations), `prompt_tokens`, `completion_tokens`;
    - the cost of the answers used, fresh or cached (`used_calls`, `used_seconds`,
      `used_prompt_tokens`, `used_completion_tokens`): a cached answer brings its stored
      duration and tokens, so a method's cost does not depend on what was cached before
      (`used_untimed` counts answers stored without a duration);
    - failures, never cost: `failed_attempts` (attempts that failed before a call
      succeeded), `failed_calls` (calls that raised LLMCallFailed) and `errors` (failed
      attempts by error kind, of both).
    Also `cache_hits` and `truncated` (answers that stopped at max_tokens).
    """

    def __init__(self, *, model: str, client: Optional[ChatClient] = None,
                 cache: Optional[ResponseCache] = None, temperature: float = 0.0,
                 disable_thinking: bool = True):
        self.model = model
        self.client = client
        self.cache = cache
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.calls = self.cache_hits = self.truncated = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.call_seconds = 0.0
        self.used_calls = self.used_prompt_tokens = self.used_completion_tokens = 0
        self.used_seconds = 0.0
        self.used_untimed = 0
        self.failed_attempts = self.failed_calls = 0
        self.errors: dict[str, int] = {}

    @classmethod
    def from_config(cls, *, offline: bool = False, model: Optional[str] = None,
                    config: Path = CONFIG_PATH) -> "LLM":
        """The configured endpoint; `offline=True` answers from the cache only."""
        s = load_settings(config)
        client = None if offline else OpenAICompatClient(s, api_key())
        return cls(model=model or s.model, client=client, cache=ResponseCache(s.cache_path),
                   temperature=s.temperature, disable_thinking=s.disable_thinking)

    def payload(self, system: str, user: str, *, max_tokens: int, json_mode: bool = False,
                response_format: Optional[dict] = None) -> dict:
        p: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
        }
        if response_format is not None:
            p["response_format"] = response_format
        elif json_mode:
            p["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        return p

    def _count_errors(self, kinds: Sequence[str]) -> None:
        for k in kinds:
            self.errors[k] = self.errors.get(k, 0) + 1

    def _use(self, rec: dict) -> None:
        """Add an answer's stored cost to the cost of the answers used."""
        self.used_calls += 1
        self.used_prompt_tokens += int(rec.get("prompt_tokens") or 0)
        self.used_completion_tokens += int(rec.get("completion_tokens") or 0)
        if rec.get("duration_s") is None:
            self.used_untimed += 1
        else:
            self.used_seconds += float(rec["duration_s"])

    def ask(self, method: str, system: str, user: str, *, max_tokens: int,
            json_mode: bool = False, response_format: Optional[dict] = None) -> dict:
        """The answer record: content, finish_reason, tokens, duration_s (the successful
        attempt), attempts, cache_hit."""
        payload = self.payload(system, user, max_tokens=max_tokens, json_mode=json_mode,
                               response_format=response_format)
        key = request_key(payload)
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is not None:
            self.cache_hits += 1
            self.truncated += hit.get("finish_reason") == "length"
            self._use(hit)
            return {**hit, "cache_hit": True}
        if self.client is None:
            raise LLMUnavailable(f"{method}: answer not cached and no client (offline)")
        try:
            r = self.client.chat(payload)
        except LLMCallFailed as e:
            self.failed_calls += 1
            self._count_errors(e.errors)
            raise
        usage = r.get("usage") or {}
        duration = r.get("duration_s")
        rec = {"content": str(r.get("content") or ""), "finish_reason": r.get("finish_reason"),
               "prompt_tokens": int(usage.get("prompt_tokens") or 0),
               "completion_tokens": int(usage.get("completion_tokens") or 0),
               "duration_s": None if duration is None else float(duration),
               "attempts": int(r.get("attempts") or 1),
               "method": method, "model": self.model,
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if self.cache is not None:
            self.cache.put(key, method, rec)
        self.calls += 1
        self.prompt_tokens += rec["prompt_tokens"]
        self.completion_tokens += rec["completion_tokens"]
        self.call_seconds += rec["duration_s"] or 0.0
        self.failed_attempts += int(r.get("failed_attempts") or 0)
        self._count_errors(r.get("errors") or ())
        self.truncated += rec["finish_reason"] == "length"
        self._use(rec)
        return {**rec, "cache_hit": False}

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "truncated": self.truncated,
                "call_seconds": self.call_seconds,
                "used_calls": self.used_calls, "used_seconds": self.used_seconds,
                "used_prompt_tokens": self.used_prompt_tokens,
                "used_completion_tokens": self.used_completion_tokens,
                "used_untimed": self.used_untimed,
                "failed_attempts": self.failed_attempts, "failed_calls": self.failed_calls,
                "errors": dict(self.errors)}


__all__ = ["LLMCallFailed", "LLMUnavailable", "LLMSettings", "load_settings", "api_base",
           "api_key", "http_post", "OpenAICompatClient", "ResponseCache", "request_key", "LLM"]
