#!/usr/bin/env python3
"""Step 5: give every kept lecture one discipline, an OECD FOS 2007 field, with the LLM.

The fixed list is the second level of the OECD Fields of Science and Technology (FOS
2007): 42 fields (config/oecd_fos2007.csv, source in config/oecd_fos2007.md). The corpora
carry no field metadata, so the LLM reads what there is about each lecture:

  * its title, if the corpus gives one (none of the five does at present);
  * the M3AV venue (the prefix of the talk id: CHI, IPP, MLS, NIH, Ubi), as its full name
    and what the venue is (config/attributes.toml [discipline.venues], from the M3AV
    paper and README): the bare code alone left the model guessing, and a check found 16
    of 34 CHI talks (HCI papers) labelled Nano-technology or an engineering field. A code
    without an entry goes in bare;
  * the text of its first 3 slides, and of 3 more spread evenly over the rest of the deck
    (each cut to 1200 characters);
  * the first 150 words of its transcript,

and picks exactly one field code from the full list, answering in JSON. The first slides
alone are often a title, a biography, a CME notice or nearly empty, so the spread slides
and the transcript opening give the LLM the content itself; the counts are settings in
config/attributes.toml [discipline]. The prompt asks for the discipline of the work
itself (its methods and contribution), not the area it is applied to: a check of 50
labels found HCI work set in health labelled as a health science, and CHI fabrication
and materials talks labelled Nano-technology. The endpoint and
the model are config/llm.toml's (temperature 0); the client, its cache and its API key
handling are methods/llm_client.py's: every answer is cached in data/llm_cache.sqlite3
under the sha256 of the full request (method name "discipline"), so a rerun replays and
any change to the prompt is a new call.

An answer is valid if its JSON has a "code" in the list (or, failing that, a "field"
equal to one of the names, ignoring case). An invalid answer gets one retry, with a
prompt that adds what was wrong (so it is a different request, not a cached replay); a
second invalid answer makes the lecture "unassigned".

Output, in data/attributes/:
  discipline.csv            corpus, lecture_id, field_code, field, attempts, raw_answer
                            (the last answer's text)
  discipline_raw.jsonl.gz   every answer of every attempt, with tokens and cache hit
  discipline.json           model, temperature, the prompts, counts

The endpoint: LLM_API_BASE; the API key: LLM_API_KEY, or the untracked
config/llm_credentials.toml, and nowhere else (cluster/env.local sets both). Without a key
the script writes nothing, prints "discipline blocked: no API key" and exits with status
3 (without an address, "discipline blocked: no endpoint"); scripts/attributes.py then
leaves the discipline column empty. discipline.json does not record the address.

Usage (a compute node needs the proxy):
    python scripts/assign_discipline.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import tomllib
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.llm import extract_json  # noqa: E402
from methods.llm_client import LLM, LLMCallFailed, LLMUnavailable, load_settings  # noqa: E402

CONFIG = REPO_ROOT / "config" / "attributes.toml"
PREPARED = REPO_ROOT / "data" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "attributes"
CORPORA = ("mavils", "m3av_a", "m3av_b", "mcd", "slide_highlight")
METHOD = "discipline"
BLOCKED_EXIT = 3
UNASSIGNED = "unassigned"

SYSTEM = ("You classify recorded lectures and conference talks by academic discipline. "
          "You answer with JSON only.")

USER_TEMPLATE = """Assign this lecture to exactly ONE field of the OECD Fields of Science and Technology (FOS 2007). The fields, as "code: name":

{fields}

The lecture:
{lecture}

Choose the single field that best describes the subject of the lecture. Label the discipline of the work itself (its methods and contribution), not the application area it is applied to. Answer with JSON only, in this form:
{{"code": "<one code from the list>", "field": "<its name>"}}"""

RETRY_TEMPLATE = """{first}

Your previous answer was not valid: {problem}
Answer again with exactly one code from the list above, as JSON only."""


def load_fos(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len({r["code"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate codes")
    return rows


def spread_positions(m: int, k: int) -> list[int]:
    """k of the positions 0..m-1, one at the centre of each of k equal parts.

    Fixed arithmetic, no sampling: the same deck always gives the same slides, so the
    request, and with it the cache key, is stable across reruns.
    """
    k = min(k, m)
    return [(2 * j + 1) * m // (2 * k) for j in range(k)]


def select_slides(slides: Sequence[dict], n_slides: int, n_spread: int) -> list[dict]:
    """The first n_slides slides, then n_spread slides spread evenly over the rest."""
    rest = slides[n_slides:]
    return list(slides[:n_slides]) + [rest[i] for i in spread_positions(len(rest), n_spread)]


def transcript_opening(units: Sequence[dict], n_words: int) -> str:
    """The first n_words whitespace-separated words of the kept transcript units."""
    words: list[str] = []
    for u in units:
        words.extend((u.get("text") or "").split())
        if len(words) >= n_words:
            break
    return " ".join(words[:n_words])


def lecture_text(rec: dict, *, n_slides: int, n_spread: int, slide_chars: int,
                 transcript_words: int, venues: Optional[Mapping[str, str]] = None) -> str:
    """What the LLM sees of one lecture: title, venue, the selected slides' text and the
    opening of the transcript.

    venues: venue code -> its name and description ([discipline.venues]). Only the venue
    is described, never a field: the field is the LLM's to choose. A code with no entry
    is sent bare, so its request (and cache key) is the same as without descriptions.
    """
    parts = []
    if rec.get("title"):
        parts.append(f"Title: {rec['title']}")
    venue = (rec.get("meta") or {}).get("venue_prefix")
    if venue and venue in (venues or {}):
        parts.append(f"Venue (from the corpus): {venues[venue]}")
    elif venue:
        parts.append(f"Venue code (from the corpus): {venue}")
    for s in select_slides(rec["slides"], n_slides, n_spread):
        text = " ".join((s.get("text") or "").split())[:slide_chars].rstrip()
        parts.append(f"Slide {s['index']}: {text if text else '(no text)'}")
    opening = transcript_opening(rec.get("units") or [], transcript_words)
    if opening:
        parts.append(f"Transcript (first {transcript_words} words): {opening}")
    return "\n".join(parts)


def build_prompt(rec: dict, fos: Sequence[dict], **select) -> str:
    """select: lecture_text's keyword arguments, the [discipline] input settings."""
    fields = "\n".join(f"{r['code']}: {r['field']}" for r in fos)
    return USER_TEMPLATE.format(fields=fields, lecture=lecture_text(rec, **select))


def parse_answer(content: str, fos: Sequence[dict]) -> tuple[Optional[dict], str]:
    """(the FOS row the answer names, or None; what was wrong with it)."""
    data = extract_json(content)
    if not isinstance(data, dict):
        return None, "no JSON object"
    by_code = {r["code"]: r for r in fos}
    code = str(data.get("code", "")).strip()
    if code in by_code:
        return by_code[code], ""
    by_name = {r["field"].casefold(): r for r in fos}
    name = str(data.get("field", "")).strip().casefold()
    if name in by_name:
        return by_name[name], ""
    return None, f"code {code!r} is not in the list"


def assign(llm: LLM, rec: dict, fos: Sequence[dict], *, max_tokens: int, **select) -> dict:
    """One lecture: ask, validate, retry once on an invalid answer, else unassigned."""
    first = build_prompt(rec, fos, **select)
    prompt, answers = first, []
    for attempt in (1, 2):
        ans = llm.ask(METHOD, SYSTEM, prompt, max_tokens=max_tokens, json_mode=True)
        answers.append({k: ans.get(k) for k in ("content", "finish_reason", "prompt_tokens",
                                                "completion_tokens", "cache_hit")})
        row, problem = parse_answer(ans["content"], fos)
        if row is not None:
            return {"field_code": row["code"], "field": row["field"], "attempts": attempt,
                    "raw_answer": ans["content"], "answers": answers}
        prompt = RETRY_TEMPLATE.format(first=first, problem=problem)
    return {"field_code": "", "field": UNASSIGNED, "attempts": 2,
            "raw_answer": answers[-1]["content"], "answers": answers}


def read_prepared(corpus: str, prepared: Path = PREPARED) -> list[dict]:
    with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run(llm: LLM, cfg: dict, *, prepared: Path = PREPARED, out_dir: Path = OUT_DIR,
        limit: Optional[int] = None) -> list[dict]:
    d = cfg["discipline"]
    fos = load_fos(REPO_ROOT / d["fos_list"])
    select = {k: int(d[k]) for k in ("n_slides", "n_spread", "slide_chars", "transcript_words")}
    select["venues"] = dict(d.get("venues", {}))
    results = []
    for corpus in CORPORA:
        recs = read_prepared(corpus, prepared)
        for rec in recs[:limit] if limit else recs:
            r = assign(llm, rec, fos, max_tokens=int(d["max_tokens"]), **select)
            results.append({"corpus": corpus, "lecture_id": rec["lecture_id"], **r})
        print(f"{corpus}: {len(recs[:limit] if limit else recs)} lectures  {llm.stats()}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "discipline.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "lecture_id", "field_code", "field", "attempts", "raw_answer"])
        for r in results:
            w.writerow([r["corpus"], r["lecture_id"], r["field_code"], r["field"],
                        r["attempts"], r["raw_answer"]])
    with gzip.open(out_dir / "discipline_raw.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({"corpus": r["corpus"], "lecture_id": r["lecture_id"],
                                 "answers": r["answers"]}, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["field"]] = counts.get(r["field"], 0) + 1
    s = load_settings()
    with open(out_dir / "discipline.json", "w") as fh:
        json.dump({
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model": llm.model, "temperature": llm.temperature,
            "disable_thinking": llm.disable_thinking, "max_tokens": int(d["max_tokens"]),
            **select,
            "fos_list": d["fos_list"], "system": SYSTEM, "user_template": USER_TEMPLATE,
            "retry_template": RETRY_TEMPLATE, "cache": s.cache_path, "method": METHOD,
            "lectures": len(results), "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "retried": sum(r["attempts"] > 1 for r in results),
            "llm_stats": llm.stats(),
        }, fh, indent=1)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, help="first N lectures per corpus (a trial run)")
    args = ap.parse_args(argv)
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    try:
        llm = LLM.from_config()
    except LLMUnavailable as e:      # no API key, or no endpoint address
        print(f"discipline blocked: {e}", file=sys.stderr)
        return BLOCKED_EXIT
    try:
        results = run(llm, cfg, limit=args.limit)
    except LLMCallFailed as e:
        print(f"discipline: the endpoint failed: {e}", file=sys.stderr)
        return 1
    n_un = sum(r["field"] == UNASSIGNED for r in results)
    print(f"{len(results)} lectures, {n_un} unassigned -> {OUT_DIR / 'discipline.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
