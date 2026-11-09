"""The LLM methods: three that read the texts, three that also use the similarity matrix.

LLM reads the text:    llm_single_call, llm_monotone, hierarchical
LLM + similarity:      llm_rerank_kbest, fusion, llm_monotone_constrained

Every method takes the matrix `sim` (n_units x n_slides), the slide texts and the unit
texts in order, and an `llm.LLM` (methods/llm_client.py), and returns one 0-based slide
per unit. The model numbers slides and units from 0, as the matrix does. Temperature,
model and thinking switch come from the `LLM`; everything a method puts into a request
is part of the cache key (see llm_client.py).

NOTHING HERE CRASHES ON A BAD ANSWER. Prose, wrong JSON, a cut-off answer or slide 97 of
a 20-slide deck is recorded (`parse_failed`, `repairs`, `truncated`) and the labels fall
back to the matrix argmax (single-call methods) or to an even spread (a hierarchical
leaf). A call that never completed raises `LLMCallFailed` and is not cached.

`llm_monotone_constrained` and `llm_single_call`/`llm_monotone` do not read the matrix
except for the fallback; `llm_rerank_kbest` and `fusion` do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from methods.common import (DPConfig, Member, check_matrix, check_member_names,
                            prepare_emissions, resolve_member, transition_scores)
from methods.llm_client import LLM
from methods.soft_order import viterbi


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

# The answer shapes show letters, not numbers: with numbers the model copied them
# (12 of 60 llm_monotone answers in the first tuning run began 0, 12, 30). The one
# digit left is the 0 that MONOTONE_TEMPLATE requires as the first start.

SYSTEM_PROMPT = (
    "You align lecture transcripts to slide decks. You answer with JSON and nothing "
    "else: no explanation, no code fence, no prose before or after the object."
)

SINGLE_CALL_TEMPLATE = """The deck has {n_slides} slides, numbered 0 to {last_slide}.
The transcript has {n_rows} rows, numbered 0 to {last_row}.

SLIDES
{slides}

TRANSCRIPT
{rows}

Assign every transcript row to the slide it is about. Rows about the same slide are
usually next to each other, but you are not required to move through the deck in order,
and slides nobody discussed may be left out.

Answer with a JSON object of exactly this shape:

  {{"blocks": [{{"from": a, "to": b, "slide": j}}, {{"from": c, "to": d, "slide": k}}, ...]}}

"from" and "to" are INCLUSIVE row numbers, "slide" is a slide number in
0..{last_slide}, and the blocks together must cover every row from 0 to {last_row}
exactly once, listed in increasing order of "from"."""

MONOTONE_TEMPLATE = """The deck has {n_slides} slides, numbered 0 to {last_slide}.
The transcript has {n_rows} rows, numbered 0 to {last_row}.

SLIDES
{slides}

TRANSCRIPT
{rows}

The lecturer moves through the deck in order, from slide 0 to slide {last_slide},
without going back. Say where each slide STARTS: the row number at which the lecturer
begins talking about it.

Answer with a JSON object of exactly this shape:

  {{"starts": [0, s1, s2, ...]}}

The list has exactly {n_slides} numbers, one per slide in deck order. It must be
non-decreasing, the first number must be 0, and every number must be in
0..{last_row}. A slide the lecturer skipped gets the same start as the slide after it."""

RERANK_TEMPLATE = """The deck has {n_slides} slides, numbered 0 to {last_slide}.
The transcript has {n_rows} rows, numbered 0 to {last_row}.

SLIDES
{slides}

TRANSCRIPT
{rows}

Below are {k} candidate alignments of this transcript to this deck. Each was produced
by a different alignment method and is written as the row ranges on which each slide is
being discussed (a slide may appear more than once if the lecturer returns to it).

{candidates}

Exactly one of them is to be chosen. Pick the candidate whose slide ranges best match
what the transcript actually says, and answer with a JSON object of exactly this shape:

  {{"choice": 1}}

where the number is between 1 and {k}."""

MONOTONE_CONSTRAINED_TEMPLATE = """The deck has {n_slides} slides, numbered 0 to {last_slide}.
The transcript has {n_rows} rows, numbered 0 to {last_row}.

SLIDES
{slides}

TRANSCRIPT
{rows}

The lecturer moves through the deck in order, from slide 0 to slide {last_slide},
without ever going back. Say how LONG the lecturer spends on each slide: for each slide
in deck order, the number of consecutive transcript rows spent on it.

Answer with a JSON object of exactly this shape:

  {{"gaps": [g0, g1, g2, ...]}}

The list has exactly {n_slides} numbers, one per slide in deck order, each of them 0 or
more. A slide the lecturer skipped gets 0. The numbers should add up to about {n_rows},
the length of the transcript.

Slide 0 starts at row 0. Slide k starts at the row given by the sum of the first k
numbers, so the alignment always moves forward through the deck."""

HIER_SPLIT_SYSTEM_PROMPT = """You are aligning lecture transcript packets to slide-page ranges.

The slides and transcript are from the SAME lecture.
The spoken transcript does NOT need exact wording overlap with the slide text.
A transcript packet can belong to a page range even when it:
- paraphrases slide content
- expands with examples
- explains figures/tables indirectly
- continues the instructor's explanation without repeating slide words

You are given:
- a LEFT consecutive page range
- a RIGHT consecutive page range
- consecutive transcript packets from one continuous lecture segment

Your job:
decide where the lecture moves from the LEFT page range to the RIGHT page range.

Return strict JSON only."""

HIER_LEAF_SYSTEM_PROMPT = """You are aligning transcript sections to a small consecutive range of slide pages.

The slides and transcript are from the SAME lecture.
A section may belong to a page even if the wording differs. It may paraphrase, expand, explain examples, or continue the instructor's explanation of the page.

Return a monotonic section-to-page assignment:
- section assignments must be non-decreasing in page index
- pages may receive zero or more sections
- use exact provided page indices only
Return strict JSON only."""

HIER_BOUNDARY_SYSTEM_PROMPT = """You are refining a local boundary between two adjacent slide pages in a lecture transcript.

The slides and transcript are from the SAME lecture.
The transcript may paraphrase, expand, or indirectly explain a page.

Return strict JSON only."""

HIER_SPLIT_USER_TEMPLATE = """LEFT PAGE RANGE:
{left_pages}

RIGHT PAGE RANGE:
{right_pages}

TRANSCRIPT PACKETS IN CHRONOLOGICAL ORDER:
{packets}

Return strict JSON with this exact schema:
{{
  "transition_after_local_packet_pos": <int or null>,
  "summary_reason": "<short summary>"
}}

Interpretation:
- if transition_after_local_packet_pos is X, packets up to local position X belong mainly to the LEFT page range,
  and packets after X belong mainly to the RIGHT page range
- local packet positions are 0-based within the packet list shown above
- if ALL packets belong mainly to the RIGHT page range, return null
- if ALL packets belong mainly to the LEFT page range, return the last local packet position shown

Rules:
- preserve chronology
- be conservative
- choose the most plausible transition in lecture flow, not literal text overlap
- do not output any text outside the JSON object"""

HIER_LEAF_USER_TEMPLATE = """PAGES:
{pages}

TRANSCRIPT SECTIONS:
{sections}

Return strict JSON with this exact schema:
{{
  "assignments": [
    {{
      "section_idx": <int>,
      "page_index": <int>,
      "reason": "<short reason>"
    }}
  ],
  "summary_reason": "<short summary>"
}}

Rules:
- assign every shown section exactly one page_index from the provided pages
- assignments must be monotonic non-decreasing in page_index
- use lecture-flow reasoning, not literal text matching only
- do not output any text outside the JSON object"""

HIER_BOUNDARY_USER_TEMPLATE = """PREVIOUS PAGE INDEX:
{prev_page_index}

PREVIOUS PAGE TEXT:
<<<
{prev_page_text}
>>>

NEXT PAGE INDEX:
{next_page_index}

NEXT PAGE TEXT:
<<<
{next_page_text}
>>>

TRANSCRIPT SECTIONS IN CHRONOLOGICAL ORDER:
{section_block}

Return strict JSON with this exact schema:
{{
  "assignments": [
    {{
      "section_idx": <int>,
      "belongs_to": "prev" | "next" | "ambiguous",
      "reason": "<short reason>"
    }}
  ],
  "summary_reason": "<short summary>"
}}

Rules:
- assign each shown section to prev, next, or ambiguous
- use "ambiguous" sparingly
- preserve chronology
- do not output any text outside the JSON object"""

HIER_SPLIT_USER_TEMPLATE_FAST = """LEFT PAGE RANGE:
{left_pages}

RIGHT PAGE RANGE:
{right_pages}

TRANSCRIPT PACKETS IN CHRONOLOGICAL ORDER:
{packets}

Return strict JSON with this exact schema:
{{
  "transition_after_local_packet_pos": <int or null>
}}

Interpretation:
- if transition_after_local_packet_pos is X, packets up to local position X belong mainly to the LEFT page range,
  and packets after X belong mainly to the RIGHT page range
- local packet positions are 0-based within the packet list shown above
- if ALL packets belong mainly to the RIGHT page range, return null
- if ALL packets belong mainly to the LEFT page range, return the last local packet position shown

Rules:
- preserve chronology
- be conservative
- choose the most plausible transition in lecture flow, not literal text overlap
- do not explain your choice and do not output any text outside the JSON object"""

HIER_LEAF_USER_TEMPLATE_FAST = """PAGES:
{pages}

TRANSCRIPT SECTIONS:
{sections}

Return strict JSON with this exact schema:
{{
  "assignments": [
    {{"section_idx": <int>, "page_index": <int>}}
  ]
}}

Rules:
- assign every shown section exactly one page_index from the provided pages
- assignments must be monotonic non-decreasing in page_index
- use lecture-flow reasoning, not literal text matching only
- do not give reasons and do not output any text outside the JSON object"""

HIER_BOUNDARY_USER_TEMPLATE_FAST = """PREVIOUS PAGE INDEX:
{prev_page_index}

PREVIOUS PAGE TEXT:
<<<
{prev_page_text}
>>>

NEXT PAGE INDEX:
{next_page_index}

NEXT PAGE TEXT:
<<<
{next_page_text}
>>>

TRANSCRIPT SECTIONS IN CHRONOLOGICAL ORDER:
{section_block}

Return strict JSON with this exact schema:
{{
  "assignments": [
    {{"section_idx": <int>, "belongs_to": "prev" | "next" | "ambiguous"}}
  ]
}}

Rules:
- assign each shown section to prev, next, or ambiguous
- use "ambiguous" sparingly
- preserve chronology
- do not give reasons and do not output any text outside the JSON object"""

HIER_PACE_LEAF_TEMPLATE = (
    '\n\nPacing guidance: in this lecture the instructor spends on average about '
    '{spp} transcript sections per slide. Distribute the sections across the pages '
    'accordingly; do NOT lump most sections onto the first page. Move to the next '
    'page once the discussion clearly shifts to it, and keep progressing in order.')

HIER_PACE_SPLIT_TEMPLATE = (
    '\n\nPacing guidance: on average about {spp} sections correspond to one slide, so '
    'a page range of P pages should receive roughly P*{spp} sections. Prefer a '
    'transition that reflects this pacing over dumping all packets on one side.')


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _check_inputs(sim, slide_texts: Sequence[str], unit_texts: Sequence[str]) -> np.ndarray:
    S = check_matrix(sim)
    n, m = S.shape
    if len(slide_texts) != m or len(unit_texts) != n:
        raise ValueError(f"matrix is {n} x {m} but there are {len(unit_texts)} unit texts "
                         f"and {len(slide_texts)} slide texts")
    return S


def argmax_fallback(S: np.ndarray) -> list[int]:
    """The back-fill for an unusable answer: each unit's most similar slide. Not another
    ordered method, so a failed LLM row never silently copies one."""
    return [int(j) for j in S.argmax(axis=1)]


def extract_json(text: Any) -> Optional[dict]:
    """The first {...} object in an answer, tolerating a code fence; no repair."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        out = json.loads(s[a:b + 1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _clamp(x, lo, hi) -> int:
    return int(max(lo, min(hi, x)))


def _numbered(texts: Sequence[str], limit: int) -> str:
    return "\n".join(f"[{i}] {(t or '').strip()[:limit]}" for i, t in enumerate(texts))


def common_fields(slide_texts: Sequence[str], unit_texts: Sequence[str], slide_chars: int,
                  row_chars: int) -> dict:
    """The template fields shared by the single-call prompts: counts and numbered texts,
    each text cut to `slide_chars` / `row_chars` characters."""
    return {"n_slides": len(slide_texts), "last_slide": len(slide_texts) - 1,
            "n_rows": len(unit_texts), "last_row": len(unit_texts) - 1,
            "slides": _numbered(slide_texts, slide_chars),
            "rows": _numbered(unit_texts, row_chars)}


def _delta(before: dict, after: dict) -> dict:
    """after - before for every count; a dict of counts (the error kinds) per key, with
    the kinds that did not change left out."""
    out: dict = {}
    for k, v in after.items():
        if isinstance(v, dict):
            out[k] = {e: c - before[k].get(e, 0) for e, c in v.items()
                      if c != before[k].get(e, 0)}
        else:
            out[k] = v - before[k]
    return out


def _one_call(llm: LLM, method: str, prompt: str, max_tokens: int,
              response_format: Optional[dict] = None) -> dict:
    return llm.ask(method, SYSTEM_PROMPT, prompt, max_tokens=max_tokens, json_mode=True,
                   response_format=response_format)


# --------------------------------------------------------------------------- #
# Parsers: (labels, parse_failed), never raising
# --------------------------------------------------------------------------- #

def parse_blocks(payload, n_rows: int, n_slides: int) -> tuple[list[int], bool]:
    """{"blocks": [{"from": a, "to": b, "slide": j}, ...]} -> one label per row.

    Out-of-range numbers are clamped; rows no block covers take the previous row's slide
    (leading rows: the first block's). `parse_failed` only when no block is usable.
    """
    obj = extract_json(payload)
    blocks = obj.get("blocks") if isinstance(obj, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return [], True
    labels = [-1] * n_rows
    for b in blocks:
        if not isinstance(b, dict):
            continue
        try:
            lo, hi, j = int(b["from"]), int(b["to"]), int(b["slide"])
        except (KeyError, TypeError, ValueError):
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        j = _clamp(j, 0, n_slides - 1)
        for i in range(_clamp(lo, 0, n_rows - 1), _clamp(hi, 0, n_rows - 1) + 1):
            labels[i] = j
    if all(x < 0 for x in labels):
        return [], True
    last = next(x for x in labels if x >= 0)
    out = []
    for x in labels:
        last = x if x >= 0 else last
        out.append(last)
    return out, False


def starts_to_labels(starts: Sequence[int], n_rows: int, n_slides: int) -> list[int]:
    """Non-decreasing slide starts -> labels; slides sharing a start are skipped."""
    labels, j = [], 0
    for i in range(n_rows):
        while j + 1 < n_slides and starts[j + 1] <= i:
            j += 1
        labels.append(j)
    return labels


def parse_starts(payload, n_rows: int, n_slides: int) -> tuple[list[int], bool]:
    """{"starts": [...]} -> labels. Monotonicity is enforced here (running maximum,
    clamped, first start 0); too few starts leave the tail slides unreached, extra
    starts are dropped."""
    obj = extract_json(payload)
    starts = obj.get("starts") if isinstance(obj, dict) else None
    if not isinstance(starts, list) or not starts:
        return [], True
    vals = []
    for s in starts:
        try:
            vals.append(_clamp(int(s), 0, max(n_rows - 1, 0)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return [], True
    vals = vals[:n_slides] + [n_rows] * max(0, n_slides - len(vals))
    run, cur = [], 0
    for v in vals:
        cur = max(cur, v)
        run.append(cur)
    run[0] = 0
    return starts_to_labels(run, n_rows, n_slides), False


def rescale_counts(counts: Sequence[int], n_rows: int) -> list[int]:
    """Non-negative counts rescaled to add up to exactly `n_rows`.

    Rule (largest remainder): each count becomes c * n_rows / sum, rounded down; the
    units still missing go one each to the slides with the largest fractional parts,
    ties to the lower slide. All-zero counts become an even split by the same rule.
    """
    c = np.asarray(counts, dtype=float)
    if c.sum() <= 0:
        c = np.ones(len(c))
    x = c * n_rows / c.sum()
    out = np.floor(x).astype(int)
    rest = n_rows - int(out.sum())
    order = sorted(range(len(c)), key=lambda k: (-(x[k] - out[k]), k))
    for k in order[:rest]:
        out[k] += 1
    return [int(v) for v in out]


def parse_gaps(payload, n_rows: int, n_slides: int) -> tuple[list[int], bool, Optional[int]]:
    """{"gaps": [...]} (units per slide) -> (labels, parse_failed, raw sum of the counts).

    Missing counts are 0, extra ones dropped; the counts are then rescaled to add up to
    n_rows (`rescale_counts`) and slide k starts at the sum of the first k.
    """
    obj = extract_json(payload)
    gaps = obj.get("gaps") if isinstance(obj, dict) else None
    if not isinstance(gaps, list) or not gaps:
        return [], True, None
    vals = []
    for g in gaps:
        try:
            vals.append(max(0, int(g)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return [], True, None
    vals = vals[:n_slides] + [0] * max(0, n_slides - len(vals))
    scaled = rescale_counts(vals, n_rows)
    starts = np.concatenate(([0], np.cumsum(scaled[:-1]))).astype(int)
    return starts_to_labels(starts, n_rows, n_slides), False, int(sum(vals))


def parse_choice(payload, k: int) -> tuple[int, bool]:
    """{"choice": i} (1-based) -> a 0-based candidate index; 0 when unusable."""
    obj = extract_json(payload)
    if not isinstance(obj, dict):
        return 0, True
    for key in ("choice", "candidate", "answer", "pick"):
        if key in obj:
            try:
                return _clamp(int(obj[key]) - 1, 0, k - 1), False
            except (TypeError, ValueError):
                return 0, True
    return 0, True


def path_ranges(path: Sequence[int]) -> str:
    """A path as 'slide 0: rows 0-11; slide 1: rows 12-29' - what the model is shown."""
    if not len(path):
        return "(empty)"
    out, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or int(path[i]) != int(path[start]):
            out.append(f"slide {int(path[start])}: rows {start}-{i - 1}")
            start = i
    return "; ".join(out)


# --------------------------------------------------------------------------- #
# The single-call methods
# --------------------------------------------------------------------------- #

def _finish(labels, failed, S, rec, llm, before, diagnostics, **extra) -> list[int]:
    if failed or len(labels) != S.shape[0]:
        labels, failed = argmax_fallback(S), True
    if diagnostics is not None:
        diagnostics.update({**_delta(before, llm.stats()), "parse_failed": bool(failed),
                            "finish_reason": rec.get("finish_reason"), **extra})
    return [int(x) for x in labels]


def align_llm_single_call(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *,
                          llm: LLM, slide_chars: int = 3000, row_chars: int = 800,
                          max_tokens: Optional[int] = None,
                          diagnostics: Optional[dict] = None) -> list[int]:
    """One call holding the whole deck and the whole transcript; the model assigns every
    unit a slide, as row-range blocks, in any order.

    `slide_chars`/`row_chars` cut each text; `max_tokens` defaults to
    min(4000, 256 + 12 per unit).
    """
    S = _check_inputs(sim, slide_texts, unit_texts)
    n, m = S.shape
    if n == 0:
        return []
    before = llm.stats()
    prompt = SINGLE_CALL_TEMPLATE.format(**common_fields(slide_texts, unit_texts,
                                                         slide_chars, row_chars))
    rec = _one_call(llm, "llm_single_call", prompt,
                    max_tokens or min(4000, 256 + 12 * n))
    labels, failed = parse_blocks(rec["content"], n, m)
    return _finish(labels, failed, S, rec, llm, before, diagnostics)


def align_llm_monotone(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *,
                       llm: LLM, slide_chars: int = 3000, row_chars: int = 800,
                       max_tokens: Optional[int] = None,
                       diagnostics: Optional[dict] = None) -> list[int]:
    """One call; the model gives the unit at which each slide STARTS, and the code makes
    the starts non-decreasing (`parse_starts`). `max_tokens` defaults to
    min(1500, 64 + 8 per slide)."""
    S = _check_inputs(sim, slide_texts, unit_texts)
    n, m = S.shape
    if n == 0:
        return []
    before = llm.stats()
    prompt = MONOTONE_TEMPLATE.format(**common_fields(slide_texts, unit_texts,
                                                      slide_chars, row_chars))
    rec = _one_call(llm, "llm_monotone", prompt, max_tokens or min(1500, 64 + 8 * m))
    labels, failed = parse_starts(rec["content"], n, m)
    return _finish(labels, failed, S, rec, llm, before, diagnostics)


def gaps_response_format(n_slides: int, n_rows: int) -> dict:
    """The JSON schema the server enforces: exactly n_slides integers in 0..n_rows."""
    return {"type": "json_schema", "json_schema": {"name": "slide_gaps", "strict": True,
            "schema": {"type": "object",
                       "properties": {"gaps": {"type": "array", "minItems": int(n_slides),
                                               "maxItems": int(n_slides),
                                               "items": {"type": "integer", "minimum": 0,
                                                         "maximum": int(n_rows)}}},
                       "required": ["gaps"], "additionalProperties": False}}}


def align_llm_monotone_constrained(sim, slide_texts: Sequence[str],
                                   unit_texts: Sequence[str], *, llm: LLM,
                                   slide_chars: int = 3000, row_chars: int = 800,
                                   max_tokens: int = 900,
                                   diagnostics: Optional[dict] = None) -> list[int]:
    """One call under a server-enforced JSON schema (Willard and Louf 2023's guided
    generation): the model gives the number of units spent on each slide, and the
    starts are the running sum, so every admissible answer is monotone.

    DECIDED: the counts are rescaled to add up to the number of units
    (`rescale_counts`; the schema cannot constrain their sum, and counts that overrun
    the transcript would leave the last slides unreached). `diagnostics` records the
    model's raw sum (`raw_sum`) and its difference from the number of units (`sum_off`).
    """
    S = _check_inputs(sim, slide_texts, unit_texts)
    n, m = S.shape
    if n == 0:
        return []
    before = llm.stats()
    prompt = MONOTONE_CONSTRAINED_TEMPLATE.format(**common_fields(slide_texts, unit_texts,
                                                                  slide_chars, row_chars))
    rec = _one_call(llm, "llm_monotone_constrained", prompt, max_tokens,
                    response_format=gaps_response_format(m, n))
    labels, failed, raw_sum = parse_gaps(rec["content"], n, m)
    return _finish(labels, failed, S, rec, llm, before, diagnostics, raw_sum=raw_sum,
                   sum_off=None if raw_sum is None else raw_sum - n)


#: The methods whose alignments llm_rerank_kbest offers, in this fixed order.
RERANK_METHODS = ("dtw", "dp", "pelt", "needleman_wunsch", "cohesion_dp")


def rerank_candidates(S: np.ndarray, cfg: Optional[DPConfig],
                      member_inputs: Optional[Mapping[str, Member]] = None,
                      ) -> tuple[list[list[int]], list[str]]:
    """The candidates in RERANK_METHODS order, each method at its own defaults on `S`
    under `cfg` unless `member_inputs` gives it its own matrix, cfg and settings (its
    tuned ones, docs/TUNING_DESIGN.md 2.6); a candidate identical to an earlier one is
    dropped (its name is kept with the first)."""
    from methods.boundaries import align_pelt
    from methods.monotone import align_dtw, align_needleman_wunsch
    from methods.soft_order import align_cohesion_dp, align_dp
    fns = {"dtw": align_dtw, "dp": align_dp, "pelt": align_pelt,
           "needleman_wunsch": align_needleman_wunsch, "cohesion_dp": align_cohesion_dp}
    inputs = check_member_names(member_inputs, RERANK_METHODS, "llm_rerank_kbest")
    cands: list[list[int]] = []
    names: list[str] = []
    for name in RERANK_METHODS:
        S_m, cfg_m, kw = resolve_member(name, inputs.get(name), S, cfg, fns[name])
        lab = [int(x) for x in fns[name](S_m, cfg=cfg_m, **kw)]
        if lab in cands:
            names[cands.index(lab)] += "+" + name
        else:
            cands.append(lab)
            names.append(name)
    return cands, names


def align_llm_rerank_kbest(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *,
                           llm: LLM, cfg: Optional[DPConfig] = None,
                           slide_chars: int = 3000, row_chars: int = 800,
                           max_tokens: int = 64,
                           member_inputs: Optional[Mapping[str, Member]] = None,
                           diagnostics: Optional[dict] = None) -> list[int]:
    """The alignments of five different methods (RERANK_METHODS: dtw, dp, pelt,
    needleman_wunsch, cohesion_dp, each at its defaults), duplicates removed, shown as
    slide row ranges; the model picks one (Kumar and Byrne 2004's reranking idea).
    Candidate 1 is dtw's; an unusable answer also gives dtw's.

    DECIDED: the candidates come from five different methods, not from dtw's exact k
    best paths, which are near-duplicates (they typically differ in one or two units).

    `member_inputs`: each candidate method's own matrix, cfg and settings (see
    `rerank_candidates`).
    """
    S = _check_inputs(sim, slide_texts, unit_texts)
    n, m = S.shape
    if n == 0:
        return []
    before = llm.stats()
    cands, names = rerank_candidates(S, cfg, member_inputs)
    listing = "\n".join(f"CANDIDATE {i + 1}: {path_ranges(p)}" for i, p in enumerate(cands))
    prompt = RERANK_TEMPLATE.format(k=len(cands), candidates=listing,
                                    **common_fields(slide_texts, unit_texts,
                                                    slide_chars, row_chars))
    rec = _one_call(llm, "llm_rerank_kbest", prompt, max_tokens)
    idx, failed = parse_choice(rec["content"], len(cands))
    if diagnostics is not None:
        diagnostics.update({**_delta(before, llm.stats()), "parse_failed": bool(failed),
                            "finish_reason": rec.get("finish_reason"), "chosen": names[idx],
                            "candidates": names})
    return list(cands[idx])


# --------------------------------------------------------------------------- #
# The hierarchical aligner
# --------------------------------------------------------------------------- #
#
# The transcript is cut into PACKETS of consecutive units. A SPLIT call asks, for a page
# range and a packet range, where the lecture moves from the left half of the pages to the
# right half; the two halves recurse. A node with one page, or few pages and few packets,
# is a LEAF: one call assigns every unit in it a page. A BOUNDARY call then re-asks the
# few units either side of each page change. Monotonicity, coverage and range are enforced
# in code and every such repair is counted.

@dataclass(frozen=True)
class HierConfig:
    """The tree's shape, the text budgets per prompt, and the answer budgets."""

    #: ask for a `reason` per assignment (longer answers; the parser ignores them)
    ask_for_reason: bool = False
    #: units per packet (the split calls move packets around)
    packet_size: int = 5
    #: a node with at most this many pages AND packets becomes a leaf
    leaf_max_pages: int = 3
    leaf_max_packets: int = 5
    max_depth: int = 32
    #: hard ceiling on paid calls per lecture
    max_calls: int = 600
    page_preview_chars: int = 1200
    page_full_chars: int = 8000
    packet_chars: int = 1200
    section_chars: int = 900
    use_boundary_refinement: bool = True
    boundary_context_before: int = 2
    boundary_context_after: int = 2
    #: answer budget per assigned unit in leaf and boundary calls, and the ceiling
    tokens_per_unit: int = 40
    tokens_per_unit_with_reason: int = 120
    max_answer_tokens: int = 8192

    def split_tokens(self) -> int:
        return 1200 if self.ask_for_reason else 192

    def unit_tokens(self, n_units: int) -> int:
        """Budget for a leaf or boundary answer listing `n_units` assignments.

        A compact entry {"section_idx": 123, "page_index": 12} is about 18-22 tokens
        with the digit-wise number tokenisation of the Qwen models, and a leaf holds up to
        8 packets x 20 units, so the budget is 64 + tokens_per_unit per unit, up to
        max_answer_tokens: a smaller one (16 per unit, capped at 1200) cuts long leaves off
        mid-answer, and they fall back to an even spread.
        """
        per = self.tokens_per_unit_with_reason if self.ask_for_reason else self.tokens_per_unit
        return min(self.max_answer_tokens, 64 + per * max(n_units, 1))

    def templates(self) -> tuple[str, str, str]:
        if self.ask_for_reason:
            return HIER_SPLIT_USER_TEMPLATE, HIER_LEAF_USER_TEMPLATE, HIER_BOUNDARY_USER_TEMPLATE
        return (HIER_SPLIT_USER_TEMPLATE_FAST, HIER_LEAF_USER_TEMPLATE_FAST,
                HIER_BOUNDARY_USER_TEMPLATE_FAST)


def hier_config(n_slides: int, n_units: int, **overrides) -> HierConfig:
    """Sized from the lecture: one packet is about one slide's worth of units
    (clamped to 3..20), a leaf about 12 slides' worth of units (3..8 packets)."""
    spp = n_units / max(n_slides, 1)
    packet = _clamp(round(spp), 3, 20)
    sized = {"packet_size": packet,
             "leaf_max_packets": _clamp(round(12 / max(packet, 1)) + 2, 3, 8),
             "packet_chars": max(1200, packet * 140)}
    return HierConfig(**{**sized, **overrides})


@dataclass(frozen=True)
class _Packet:
    idx: int
    start: int
    end: int
    text: str


def build_packets(units: Sequence[str], size: int) -> list[_Packet]:
    size = max(1, int(size))
    return [_Packet(k, s, min(s + size, len(units)) - 1,
                    " ".join(t.strip() for t in units[s:s + size] if t and t.strip()))
            for k, s in enumerate(range(0, len(units), size))]


def _trunc(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


class HierarchicalAligner:
    """The recursion, one `LLM.ask` per split, leaf and boundary call."""

    def __init__(self, pages: Sequence[str], units: Sequence[str], *, llm: LLM,
                 cfg: HierConfig):
        self.pages = [str(t or "") for t in pages]
        self.units = [str(t or "") for t in units]
        self.llm = llm
        self.cfg = cfg
        spp = f"{len(self.units) / max(len(self.pages), 1):.1f}"
        split_t, leaf_t, self.boundary_t = cfg.templates()
        self.leaf_t = leaf_t + HIER_PACE_LEAF_TEMPLATE.format(spp=spp)
        self.split_t = split_t + HIER_PACE_SPLIT_TEMPLATE.format(spp=spp)
        self.calls_before = llm.calls
        self.parse_failed = self.repairs = 0
        self.split_calls = self.leaf_calls = self.boundary_calls = 0

    def _ask_json(self, system: str, user: str, max_tokens: int) -> Optional[dict]:
        if self.llm.calls - self.calls_before >= self.cfg.max_calls:
            raise RuntimeError(f"hierarchical: the max_calls ceiling ({self.cfg.max_calls} "
                               f"paid calls for one lecture) was reached")
        rec = self.llm.ask("hierarchical", system, user, max_tokens=max_tokens)
        obj = extract_json(rec["content"])
        if obj is None:
            self.parse_failed += 1
        return obj

    def _split(self, packets, p0, p1, k0, k1) -> Optional[int]:
        """Position of the last packet on the LEFT half of pages p0..p1, or None for
        'all on the right'; an unusable answer falls back to the middle packet."""
        left_end = (p0 + p1) // 2
        prev = lambda a, b: "\n\n".join(
            f"- page_index={p}\n{_trunc(self.pages[p], self.cfg.page_preview_chars)}"
            for p in range(a, b + 1))
        packets_txt = "\n".join(
            f"[local_packet_pos={pos}] [global_packet_idx={pk.idx}] "
            f"(sections {pk.start}-{pk.end}) {_trunc(pk.text, self.cfg.packet_chars)}"
            for pos, pk in enumerate(packets[k0:k1 + 1]))
        prompt = self.split_t.format(left_pages=prev(p0, left_end),
                                     right_pages=prev(left_end + 1, p1), packets=packets_txt)
        self.split_calls += 1
        obj = self._ask_json(HIER_SPLIT_SYSTEM_PROMPT, prompt, self.cfg.split_tokens())
        if obj is None:
            self.repairs += 1
            return (k0 + k1) // 2
        raw = obj.get("transition_after_local_packet_pos")
        if raw is None:
            return None
        if not isinstance(raw, int) or isinstance(raw, bool):
            self.parse_failed += 1
            self.repairs += 1
            return (k0 + k1) // 2
        return None if raw < 0 else k0 + min(raw, k1 - k0)

    def _segment(self, packets, p0, p1, k0, k1, depth=0) -> list[tuple[int, int, int, int]]:
        """Leaves as (first page, last page, first unit, last unit)."""
        if k0 > k1:
            return []
        if p0 == p1 or depth >= self.cfg.max_depth or (
                p1 - p0 + 1 <= self.cfg.leaf_max_pages
                and k1 - k0 + 1 <= self.cfg.leaf_max_packets):
            return [(p0, p1, packets[k0].start, packets[k1].end)]
        mid = (p0 + p1) // 2
        pos = self._split(packets, p0, p1, k0, k1)
        if pos is None:
            return self._segment(packets, mid + 1, p1, k0, k1, depth + 1)
        pos = _clamp(pos, k0, k1)
        if pos >= k1:
            return self._segment(packets, p0, mid, k0, k1, depth + 1)
        return (self._segment(packets, p0, mid, k0, pos, depth + 1)
                + self._segment(packets, mid + 1, p1, pos + 1, k1, depth + 1))

    def _spread(self, s0, s1, p0, p1) -> dict[int, int]:
        """The leaf fallback: units spread evenly and in order over the pages."""
        n_s, n_p = s1 - s0 + 1, p1 - p0 + 1
        if n_s == 1 or p0 == p1:
            return {i: p0 for i in range(s0, s1 + 1)}
        return {s0 + k: p0 + min(n_p - 1, int(k / (n_s - 1) * n_p)) for k in range(n_s)}

    def _leaf(self, p0, p1, s0, s1) -> dict[int, int]:
        if p0 == p1:
            return {i: p0 for i in range(s0, s1 + 1)}
        pages = "\n\n".join(f"PAGE {p}:\n<<<\n{_trunc(self.pages[p], self.cfg.page_full_chars)}\n>>>"
                            for p in range(p0, p1 + 1))
        sections = "\n".join(f"[local_section_pos={pos}] [section_idx={s0 + pos}] "
                             f"{_trunc(t, self.cfg.section_chars)}"
                             for pos, t in enumerate(self.units[s0:s1 + 1]))
        self.leaf_calls += 1
        obj = self._ask_json(HIER_LEAF_SYSTEM_PROMPT,
                             self.leaf_t.format(pages=pages, sections=sections),
                             self.cfg.unit_tokens(s1 - s0 + 1))
        if obj is None:
            self.repairs += 1
            return self._spread(s0, s1, p0, p1)
        said: dict[int, int] = {}
        items = obj.get("assignments", [])
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            sec, page = item.get("section_idx"), item.get("page_index")
            if (isinstance(sec, int) and not isinstance(sec, bool) and isinstance(page, int)
                    and not isinstance(page, bool) and s0 <= sec <= s1 and p0 <= page <= p1):
                said[sec] = page
        out, last = {}, p0
        for i in range(s0, s1 + 1):          # missing, out-of-range or backward: previous
            page = said.get(i)
            if page is None or page < last:
                self.repairs += 1
                page = last
            out[i] = last = page
        return out

    def _refine(self, assign: list[int]) -> list[int]:
        """Re-ask the units around each page change: prev / next / ambiguous."""
        if not self.cfg.use_boundary_refinement:
            return assign
        out = list(assign)
        i = 0
        while i < len(out) - 1:
            a, b = out[i], out[i + 1]
            if a == b:
                i += 1
                continue
            lo = max(0, i - self.cfg.boundary_context_before)
            hi = min(len(out) - 1, i + 1 + self.cfg.boundary_context_after)
            prompt = self.boundary_t.format(
                prev_page_index=a, prev_page_text=_trunc(self.pages[a], self.cfg.page_full_chars),
                next_page_index=b, next_page_text=_trunc(self.pages[b], self.cfg.page_full_chars),
                section_block="\n".join(f"[section_idx={k}] {_trunc(self.units[k], self.cfg.section_chars)}"
                                        for k in range(lo, hi + 1)))
            self.boundary_calls += 1
            obj = self._ask_json(HIER_BOUNDARY_SYSTEM_PROMPT, prompt,
                                 self.cfg.unit_tokens(hi - lo + 1))
            i = hi + 1
            items = obj.get("assignments") if obj is not None else None
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                sec, side = item.get("section_idx"), item.get("belongs_to")
                if isinstance(sec, int) and not isinstance(sec, bool) and lo <= sec <= hi:
                    if side == "prev":
                        out[sec] = a
                    elif side == "next":
                        out[sec] = b
            for pos in range(lo + 1, hi + 1):   # keep the window non-decreasing
                out[pos] = max(out[pos], out[pos - 1])
        return out

    def run(self) -> list[int]:
        packets = build_packets(self.units, self.cfg.packet_size)
        assign: list[Optional[int]] = [None] * len(self.units)
        for p0, p1, s0, s1 in self._segment(packets, 0, len(self.pages) - 1, 0,
                                            len(packets) - 1):
            for sec, page in self._leaf(p0, p1, s0, s1).items():
                assign[sec] = page
        # every packet reaches exactly one leaf, so every unit is assigned; the fill is a
        # guard, not a repair the method relies on
        first = next((x for x in assign if x is not None), 0)
        filled, last = [], first
        for x in assign:
            last = x if x is not None else last
            filled.append(last)
        return [_clamp(x, 0, len(self.pages) - 1) for x in self._refine(filled)]

    def diagnostics(self) -> dict:
        return {"parse_failed": self.parse_failed, "repairs": self.repairs,
                "split_calls": self.split_calls, "leaf_calls": self.leaf_calls,
                "boundary_calls": self.boundary_calls}


def align_hierarchical(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *,
                       llm: LLM, diagnostics: Optional[dict] = None,
                       **config) -> list[int]:
    """The recursive LLM aligner (split / leaf / boundary calls; see the section comment).

    `config` overrides `HierConfig` fields; packet and leaf sizes are otherwise sized from
    the lecture (`hier_config`). Does not read the matrix.
    """
    S = _check_inputs(sim, slide_texts, unit_texts)
    if S.shape[0] == 0:
        return []
    cfg = hier_config(len(slide_texts), len(unit_texts), **config)
    before = llm.stats()
    aligner = HierarchicalAligner(slide_texts, unit_texts, llm=llm, cfg=cfg)
    labels = aligner.run()
    if diagnostics is not None:
        diagnostics.update({**_delta(before, llm.stats()), **aligner.diagnostics()})
    return labels


def align_fusion(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *, llm: LLM,
                 cfg: Optional[DPConfig] = None, pos_weight: float = 10.0,
                 diagnostics: Optional[dict] = None, **config) -> list[int]:
    """dp with the hierarchical answer as its positional prior.

    The hierarchical labels are the expected slide per unit (the pacing curve); the
    prior charges pos_weight * |slide - curve| / m (instead of the straight diagonal at
    cfg.pos_weight), and dp's transitions decide. `config`: hierarchical's settings.
    """
    cfg = cfg or DPConfig()
    S = _check_inputs(sim, slide_texts, unit_texts)
    n, m = S.shape
    if n == 0:
        return []
    curve = np.asarray(align_hierarchical(S, slide_texts, unit_texts, llm=llm,
                                          diagnostics=diagnostics, **config), dtype=float)
    E = prepare_emissions(S, replace(cfg, pos_weight=pos_weight), expected=curve)
    return viterbi(E, transition_scores(m, cfg))


__all__ = ["SYSTEM_PROMPT", "SINGLE_CALL_TEMPLATE", "MONOTONE_TEMPLATE", "RERANK_TEMPLATE",
           "MONOTONE_CONSTRAINED_TEMPLATE", "extract_json", "parse_blocks", "parse_starts",
           "parse_gaps", "rescale_counts", "parse_choice", "RERANK_METHODS",
           "rerank_candidates", "path_ranges", "gaps_response_format",
           "align_llm_single_call", "align_llm_monotone", "align_llm_monotone_constrained",
           "align_llm_rerank_kbest", "HierConfig", "hier_config", "HierarchicalAligner",
           "align_hierarchical", "align_fusion"]
