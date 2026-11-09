"""The models and the endpoint behind the scorers: what produces vectors and scores.

  * `EmbeddingClient`: the OpenAI-compatible /embeddings endpoint (the `embed` scorer),
    standard library only, with the same key mechanism as the LLM client.
  * `SpladeEncoder`, `CrossEncoderScorer`, `ColQwenEncoder`: local models through torch,
    transformers and colpali-engine (the `models` extra in pyproject.toml). They are
    imported only when a model is first used, so the package, the cached matrices and
    the tests need none of them.

Each class exposes the one small interface its scorer calls (see registry.py); the
tests replace them with tiny fakes of the same interface. Nothing here is called unless
a matrix is missing from the cache and the run is not offline.

DETERMINISM. The local models run in eval mode under inference_mode, with the seed set
before the weights load, a fixed batch size and a fixed input order: padding is masked,
but the batch composition still decides the kernel shapes, so the batch size is part of
every cache key. The device is in the key too. torch's CPU thread count can change the
last bits of a score; it is left to the job (not pinned here).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

import numpy as np

# torch 2.13 sends some eager CUDA ops to Triton kernels it compiles on first use (its
# "native" op registry, torch/_native): in ColQwen2 the rotary embedding's outer product
# (inv_freq @ position_ids) goes to `bmm_outer_product`, and the compile needs a C
# compiler, which the image (cluster/similarity.def) does not have, so every image
# matrix failed on the GPU. TORCH_DISABLE_NATIVE_JIT=1 keeps every op on torch's
# standard ATen/cuBLAS kernels - the same arithmetic, the path torch took before 2.13.
# Set here, at import, because the registry reads it once and torch may be imported
# before a model loads; a value already in the environment is kept.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

#: Seed set before any local model is loaded (a head missing from a checkpoint would be
#: initialised from the global RNG).
SEED = 0


class ModelDependencyMissing(ImportError):
    """torch / transformers / colpali-engine are not installed (the `models` extra)."""


class EmbeddingCallFailed(RuntimeError):
    """The embedding endpoint gave no usable answer after the retries."""


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise ModelDependencyMissing(
            "this scorer runs a local model: pip install -e '.[models]' "
            "(torch, transformers, colpali-engine; see pyproject.toml)") from e
    torch.manual_seed(SEED)
    return torch


def _transformers():
    try:
        import transformers  # noqa: PLC0415
    except ImportError as e:
        raise ModelDependencyMissing("this scorer needs transformers: "
                                     "pip install -e '.[models]'") from e
    return transformers


# --------------------------------------------------------------------------- #
# The OpenAI-compatible embedding endpoint
# --------------------------------------------------------------------------- #

def http_post(request: urllib.request.Request, timeout: float) -> bytes:
    """The real transport: send the request, return the response body."""
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


class EmbeddingClient:
    """POST {base_url}/embeddings, in batches, throttled and retried.

    Batches are cut at `batch` texts or `batch_chars` characters, whichever comes first.
    The server embeds each text on its own (the same vector at 16 to 512 texts per
    request, measured), so the batch size is not part of the cache key.
    Retries on 429, 5xx and network errors; the endpoint limits requests per ACCOUNT, so
    parallel jobs must share `requests_per_minute`. Honours https_proxy.

    TIMING (the cost rule of docs/TUNING_DESIGN.md 6, as for the LLM client): each
    attempt is timed with `clock` from just before the request is sent to just after the
    answer is read and parsed. Only successful requests count: `requests` and
    `request_seconds` (their successful attempts' durations) are the cost. The throttle
    and backoff sleeps lie outside every attempt's window and are counted apart
    (`wait_seconds`), and so are failed attempts (`failed_attempts`, `failed_seconds`,
    `errors` by kind) and requests that failed for good (`failed_calls`): information,
    never cost. `stats()` returns all counters, so a caller can take differences.
    `transport`, `sleep` and `clock` are the real ones unless a test replaces them.
    """

    def __init__(self, *, base_url: str, model: str, key: str, batch: int = 512,
                 batch_chars: int = 1_000_000, timeout_s: float = 300.0,
                 max_retries: int = 6, requests_per_minute: float = 20.0,
                 transport: Callable[[urllib.request.Request, float], bytes] = http_post,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.key = key
        self.batch = int(batch)
        self.batch_chars = int(batch_chars)
        self.timeout = float(timeout_s)
        self.max_retries = int(max_retries)
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last = 0.0
        self._transport, self._sleep, self._clock = transport, sleep, clock
        self.requests = 0
        self.request_seconds = 0.0
        self.texts_sent = 0
        self.wait_seconds = 0.0
        self.failed_attempts = 0
        self.failed_seconds = 0.0
        self.failed_calls = 0
        self.errors: dict[str, int] = {}

    def stats(self) -> dict:
        return {"requests": self.requests, "request_seconds": self.request_seconds,
                "texts_sent": self.texts_sent, "wait_seconds": self.wait_seconds,
                "failed_attempts": self.failed_attempts,
                "failed_seconds": self.failed_seconds, "failed_calls": self.failed_calls,
                "errors": dict(self.errors)}

    def _throttle(self) -> float:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        return max(wait, 0.0)

    def _failed(self, t0: float, kind: str) -> None:
        self.failed_attempts += 1
        self.failed_seconds += self._clock() - t0
        self.errors[kind] = self.errors.get(kind, 0) + 1

    def _chunks(self, texts: list[str]):
        cur: list[str] = []
        size = 0
        for t in texts:
            if cur and (len(cur) >= self.batch or size + len(t) > self.batch_chars):
                yield cur
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        if cur:
            yield cur

    def _post(self, texts: list[str]) -> list[list[float]]:
        data = json.dumps({"model": self.model, "input": texts,
                           "encoding_format": "float"}).encode("utf-8")
        last_error = "no attempt"
        for attempt in range(self.max_retries + 1):
            if attempt:
                backoff = min(60.0, 2.0 * 1.6 ** attempt)
                self._sleep(backoff)
                self.wait_seconds += backoff
            self.wait_seconds += self._throttle()
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            t0 = self._clock()
            try:
                body = json.loads(self._transport(req, self.timeout).decode("utf-8"))
            except urllib.error.HTTPError as e:
                self._failed(t0, f"HTTP {e.code}")
                last_error = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code == 429 or e.code >= 500:
                    continue
                self.failed_calls += 1
                raise EmbeddingCallFailed(last_error) from e
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                self._failed(t0, type(e).__name__)
                last_error = f"{type(e).__name__}: {e}"
                continue
            rows = body.get("data") if isinstance(body, dict) else None
            if not isinstance(rows, list) or len(rows) != len(texts):
                self._failed(t0, "malformed response")
                self.failed_calls += 1
                raise EmbeddingCallFailed(f"expected {len(texts)} embeddings, got: "
                                          f"{str(body)[:300]}")
            # The OpenAI format numbers each vector; order by it rather than trusting
            # the list order.
            rows = sorted(rows, key=lambda r: int(r.get("index", 0)))
            self.requests += 1
            self.request_seconds += self._clock() - t0
            self.texts_sent += len(texts)
            return [r["embedding"] for r in rows]
        self.failed_calls += 1
        raise EmbeddingCallFailed(f"no answer after {self.max_retries + 1} attempts: "
                                  f"{last_error}")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One vector per text, in order, as a (len(texts), dim) float32 array."""
        out: list[list[float]] = []
        for chunk in self._chunks(list(texts)):
            out.extend(self._post(chunk))
        return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# SPLADE
# --------------------------------------------------------------------------- #

class SpladeEncoder:
    """A SPLADE checkpoint's MLM head, max-pooled into one sparse vector per text.

    weight(term) = max over the text's tokens of log(1 + relu(logit)) (Formal et al.
    2021; SPLADE-v3, Lassance et al. 2024). Terms at or below `min_weight` are dropped.
    """

    def __init__(self, *, model: str, max_length: int, batch: int, min_weight: float,
                 device: str = "cpu"):
        self.model_id, self.device = model, device
        self.max_length, self.batch = int(max_length), int(batch)
        self.min_weight = float(min_weight)
        self._tok = self._model = self._t = None
        self.vocab_size = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        self._t = _torch()
        tf = _transformers()
        self._tok = tf.AutoTokenizer.from_pretrained(self.model_id)
        model = tf.AutoModelForMaskedLM.from_pretrained(self.model_id)
        self._model = model.eval().to(self.device)
        self.vocab_size = int(model.config.vocab_size)

    def encode(self, texts: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        self._load()
        torch = self._t
        out: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch):
                enc = self._tok(list(texts[i:i + self.batch]), padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                weights = torch.log1p(torch.relu(self._model(**enc).logits))
                mask = enc["attention_mask"].unsqueeze(-1)
                # padding must not win the max; the weights are >= 0, so 0 is neutral
                rep = torch.max(weights * mask, dim=1).values.float().cpu().numpy()
                for r in rep:
                    keep = np.nonzero(r > self.min_weight)[0]
                    out.append((keep.astype(np.int64), r[keep].astype(np.float64)))
        return out


# --------------------------------------------------------------------------- #
# Cross-encoder
# --------------------------------------------------------------------------- #

class CrossEncoderScorer:
    """One relevance logit per (unit text, slide text) pair, from a MS MARCO cross-encoder.

    The pair is truncated together to `max_length` tokens (longest side first).
    """

    def __init__(self, *, model: str, max_length: int, batch: int, device: str = "cpu"):
        self.model_id, self.device = model, device
        self.max_length, self.batch = int(max_length), int(batch)
        self._tok = self._model = self._t = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._t = _torch()
        tf = _transformers()
        self._tok = tf.AutoTokenizer.from_pretrained(self.model_id)
        model = tf.AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model = model.eval().to(self.device)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = np.zeros(len(pairs), dtype=np.float64)
        if not pairs:
            return out
        self._load()
        with self._t.inference_mode():
            for i in range(0, len(pairs), self.batch):
                chunk = pairs[i:i + self.batch]
                enc = self._tok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                if logits.shape[-1] != 1:
                    raise ValueError(f"{self.model_id} has a {logits.shape[-1]}-way head; "
                                     f"a MS MARCO cross-encoder has one relevance logit")
                out[i:i + len(chunk)] = logits.squeeze(-1).float().cpu().numpy()
        return out


# --------------------------------------------------------------------------- #
# ColQwen2 (page images)
# --------------------------------------------------------------------------- #

class ColQwenEncoder:
    """ColQwen2 multi-vectors: one (n_patches, 128) array per slide image, one
    (n_tokens, 128) array per unit text (Faysse et al. 2025).

    Padding positions are dropped from every output, so they never take part in MaxSim.
    Queries are length-sorted inside fixed blocks of `query_block` consecutive texts to
    keep padding small; blocks and sort are functions of the input alone, so every run
    forms the same batches.
    """

    def __init__(self, *, model: str, image_batch: int, query_batch: int, query_block: int,
                 dtype: str = "float32", device: str = "cpu"):
        self.model_id, self.device, self.dtype = model, device, dtype
        self.image_batch, self.query_batch = int(image_batch), int(query_batch)
        self.query_block = int(query_block)
        self._model = self._proc = self._t = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._t = _torch()
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor  # noqa: PLC0415
        except ImportError as e:
            raise ModelDependencyMissing("the image scorer needs colpali-engine: "
                                         "pip install -e '.[models]'") from e
        self._model = ColQwen2.from_pretrained(
            self.model_id, torch_dtype=getattr(self._t, self.dtype),
            device_map=self.device).eval()
        self._proc = ColQwen2Processor.from_pretrained(self.model_id)

    def _forward(self, batch) -> list[np.ndarray]:
        torch = self._t
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            out = self._model(**batch)
        arr = out.to(torch.float32).cpu().numpy()
        mask = batch.get("attention_mask")
        if mask is None:
            return [np.ascontiguousarray(a) for a in arr]
        keep = mask.cpu().numpy().astype(bool)
        return [np.ascontiguousarray(a[k]) for a, k in zip(arr, keep)]

    def encode_images(self, images: Sequence[bytes]) -> list[np.ndarray]:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415  (pillow comes with the models extra)

        self._load()
        out: list[np.ndarray] = []
        for i in range(0, len(images), self.image_batch):
            pil = [Image.open(io.BytesIO(raw)).convert("RGB")
                   for raw in images[i:i + self.image_batch]]
            out.extend(self._forward(self._proc.process_images(pil)))
        return out

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.query_block):
            chunk = list(texts[start:start + self.query_block])
            order = sorted(range(len(chunk)), key=lambda k: (
                len(self._proc.tokenizer(chunk[k])["input_ids"]), k))
            got: dict[int, np.ndarray] = {}
            for i in range(0, len(order), self.query_batch):
                idx = order[i:i + self.query_batch]
                for k, v in zip(idx, self._forward(
                        self._proc.process_queries([chunk[k] for k in idx]))):
                    got[k] = v
            out.extend(got[k] for k in range(len(chunk)))
        return out


__all__ = ["ModelDependencyMissing", "EmbeddingCallFailed", "EmbeddingClient", "http_post",
           "SpladeEncoder", "CrossEncoderScorer", "ColQwenEncoder", "SEED"]
