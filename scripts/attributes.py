#!/usr/bin/env python3
"""Step 5: the attributes table, one row per kept lecture.

It joins the prepared lectures (data/prepared/, step 2), the layout boxes
(data/attributes/layout_boxes.jsonl.gz, scripts/detect_layout.py) and the disciplines
(data/attributes/discipline.csv, scripts/assign_discipline.py) into
data/attributes/document_attributes.csv. The columns, in plain words (docs/ATTRIBUTES.md
has the full definitions):

  corpus, lecture_id      which lecture
  text_source             where its slide text comes from (pdf_text_layer, tesseract_ocr,
                          corpus_corrected_ocr)
  n_slides                kept slides (at most 40)
  n_units                 kept transcript units
  n_transcript_words      whitespace-separated words in the kept units
  image_pct               % of the lecture's slide area covered by detected pictures:
                          the area of the union of the kept picture boxes, summed over
                          its slides, divided by the summed full pixel area of its slide
                          images, x 100. The union merges overlapping boxes, so no pixel
                          counts twice. A picture box is not kept if a picture box in the
                          same place is on at least half the lecture's slides (logos,
                          speaker insets, player bars) or if it is smaller than 1 % of the
                          slide (icons, glyphs); config/attributes.toml [picture_filter].
                          M3AV's slide images are video frames.
  table_pct               the same for table boxes
  noise                   % of the slide-text word tokens that are not in the fixed English
                          word list (config/wordlist_en_scowl70.txt.gz, SCOWL). A token is a
                          maximal run of the letters a-z/A-Z, lowercased, of length >= 2;
                          the share is over all the lecture's slides together
  text_density            mean slide-text words per slide: the whitespace-separated
                          tokens holding at least one letter or digit, summed over the
                          slides and divided by n_slides
  discipline              one OECD FOS 2007 field (config/oecd_fos2007.csv), or
                          "unassigned"; empty when the discipline step was blocked

The run stops if a lecture has no layout line for one of its slides, a lecture appears
twice, a value is missing (except the discipline when blocked) or a percentage is
outside 0..100. A record next to the table, document_attributes.json, holds the
definitions, the inputs' hashes, the discipline status, per-corpus summaries and the
picture boxes each filter rule dropped per corpus.

Usage:
    python scripts/attributes.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "attributes.toml"
PREPARED = REPO_ROOT / "data" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "attributes"
CORPORA = ("mavils", "m3av_a", "m3av_b", "mcd", "slide_highlight")

COLUMNS = ["corpus", "lecture_id", "text_source", "n_slides", "n_units",
           "n_transcript_words", "image_pct", "table_pct", "noise", "text_density",
           "discipline"]
PCT_COLUMNS = ("image_pct", "table_pct", "noise")
SUMMARY_COLUMNS = ("image_pct", "table_pct", "noise", "text_density", "n_slides", "n_units",
                   "n_transcript_words")

NOISE_TOKEN = re.compile(r"[A-Za-z]+")
MIN_TOKEN_LEN = 2


# --------------------------------------------------------------------------- definitions

def load_wordlist(path: Path) -> frozenset[str]:
    """The fixed English word list: one word per line, '#' lines are the header."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as fh:
        return frozenset(w for line in fh if (w := line.strip()) and not w.startswith("#"))


def noise_tokens(text: str) -> list[str]:
    """Word tokens for the noise share: letter runs, lowercased, length >= 2.

    Digits, punctuation and single letters are not tokens at all: a slide full of
    numbers or bullet glyphs is not "noisy" text, and single letters (variables, list
    markers, OCR specks) would be in any word list or in none, depending on the list.
    """
    return [t.lower() for t in NOISE_TOKEN.findall(text) if len(t) >= MIN_TOKEN_LEN]


def noise_counts(texts: Iterable[str], words: frozenset[str]) -> tuple[int, int]:
    """(tokens not in the list, all tokens) over the given slide texts."""
    out = total = 0
    for text in texts:
        for t in noise_tokens(text):
            total += 1
            out += t not in words
    return out, total


def noise_pct(texts: Iterable[str], words: frozenset[str]) -> float:
    """% of the tokens not in the list, pooled over all the texts (0 if no token)."""
    out, total = noise_counts(texts, words)
    return 100.0 * out / total if total else 0.0


def slide_words(text: str) -> int:
    """Slide-text words: whitespace-separated tokens with at least one letter or digit.

    The condition drops what OCR and PDF text layers emit as bare symbols (bullets,
    dashes, '|'), which are not words a reader would count.
    """
    return sum(any(ch.isalnum() for ch in tok) for tok in text.split())


def text_density(texts: Sequence[str]) -> float:
    """Mean slide-text words per slide."""
    return sum(slide_words(t) for t in texts) / len(texts) if texts else 0.0


def union_area(boxes: Iterable[Sequence[float]], width: float, height: float) -> float:
    """Area covered by the union of axis-aligned boxes [x0, y0, x1, y1], clipped to the image.

    Overlapping boxes are merged, so a pixel under two boxes counts once (the brief: merge
    overlapping boxes of the same class before summing). Exact: a sweep over the distinct
    x edges, merging the y intervals of the boxes that span each strip.
    """
    clipped = []
    for x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, min(float(x0), width)), max(0.0, min(float(x1), width))
        y0, y1 = max(0.0, min(float(y0), height)), max(0.0, min(float(y1), height))
        if x1 > x0 and y1 > y0:
            clipped.append((x0, y0, x1, y1))
    if not clipped:
        return 0.0
    xs = sorted({x for b in clipped for x in (b[0], b[2])})
    area = 0.0
    for xa, xb in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in clipped if b[0] <= xa and b[2] >= xb)
        covered, start, end = 0.0, None, None
        for s, e in spans:
            if end is None or s > end:
                if end is not None:
                    covered += end - start
                start, end = s, e
            else:
                end = max(end, e)
        if end is not None:
            covered += end - start
        area += covered * (xb - xa)
    return area


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes [x0, y0, x1, y1] (0 if both are empty)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clipped(box: Sequence[float], width: float, height: float) -> tuple[float, ...]:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
    y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
    return x0, y0, max(x0, x1), max(y0, y1)


def picture_drops(slides: Sequence[dict], image_class: str, *, repeat_share: float,
                  repeat_iou: float, tiny_pct: float) -> list[list[Optional[str]]]:
    """For each box of each slide of one lecture: None (kept, or not a picture), "repeat"
    or "tiny".

    The layout model boxes much that is not a slide picture: a check of 100 slides by eye
    found the picture boxes wrong on 72 of them, mostly logos and watermarks, the
    speaker's camera inset in M3AV frames, small icons and glyphs, and video-player or
    beamer control bars, against 4 missed real figures (docs/ATTRIBUTES.md). Two rules
    remove most of them:

    - repeat: a picture box is dropped if a picture box in the same place (IoU >=
      repeat_iou in page coordinates normalised to 0..1, so slides of different pixel
      sizes compare) is on at least repeat_share of the lecture's slides. Each slide
      counts once however many boxes match on it, and the box's own slide counts. Logos,
      insets and bars sit in one place on every slide; a lecture's figures rarely do.
    - tiny: a remaining picture box covering less than tiny_pct % of its slide (after
      clipping to the image) is dropped.

    The repeat rule is decided on all picture boxes and comes first, so a small repeated
    logo is labelled "repeat"; "tiny" counts only boxes the repeat rule kept.
    """
    norm = []
    for s in slides:
        w, h = float(s["width"]), float(s["height"])
        norm.append([tuple(v / d for v, d in zip(_clipped(b["box"], w, h), (w, h, w, h)))
                     for b in s["boxes"] if b["class"] == image_class])
    n = len(slides)
    out: list[list[Optional[str]]] = []
    for i, s in enumerate(slides):
        w, h = float(s["width"]), float(s["height"])
        marks: list[Optional[str]] = []
        for b in s["boxes"]:
            if b["class"] != image_class:
                marks.append(None)
                continue
            nb = tuple(v / d for v, d in zip(_clipped(b["box"], w, h), (w, h, w, h)))
            seen = 1 + sum(any(box_iou(nb, p) >= repeat_iou for p in norm[k])
                           for k in range(n) if k != i)
            x0, y0, x1, y1 = _clipped(b["box"], w, h)
            if seen >= repeat_share * n:
                marks.append("repeat")
            elif 100.0 * (x1 - x0) * (y1 - y0) < tiny_pct * w * h:
                marks.append("tiny")
            else:
                marks.append(None)
        out.append(marks)
    return out


def keep_pictures(slides: Sequence[dict], image_class: str,
                  rules: dict) -> tuple[list[dict], dict[str, int]]:
    """The slides without their dropped picture boxes, and the boxes each rule dropped."""
    marks = picture_drops(slides, image_class, repeat_share=float(rules["repeat_share"]),
                          repeat_iou=float(rules["repeat_iou"]),
                          tiny_pct=float(rules["tiny_pct"]))
    kept = [{**s, "boxes": [b for b, m in zip(s["boxes"], ms) if m is None]}
            for s, ms in zip(slides, marks)]
    flat = [m for ms in marks for m in ms]
    return kept, {"repeat": flat.count("repeat"), "tiny": flat.count("tiny")}


def area_pct(slides: Sequence[dict], cls: str) -> float:
    """% of the summed slide area covered by boxes of one class (union per slide)."""
    covered = sum(union_area([b["box"] for b in s["boxes"] if b["class"] == cls],
                             s["width"], s["height"]) for s in slides)
    total = sum(s["width"] * s["height"] for s in slides)
    if total <= 0:
        raise ValueError("slides with no pixel area")
    return 100.0 * covered / total


# --------------------------------------------------------------------------- inputs

def read_prepared(corpus: str, prepared: Path = PREPARED) -> list[dict]:
    with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def read_layout(path: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """(corpus, lecture) -> slide index -> layout line."""
    out: dict[tuple[str, str], dict[int, dict]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            slides = out.setdefault((r["corpus"], r["lecture"]), {})
            if r["slide"] in slides:
                raise ValueError(f"layout: {r['corpus']}/{r['lecture']} slide {r['slide']} twice")
            slides[r["slide"]] = r
    return out


def read_discipline(path: Path) -> Optional[dict[tuple[str, str], str]]:
    """(corpus, lecture_id) -> field name, or None when the step did not run (blocked)."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["corpus"], r["lecture_id"]): r["field"] for r in csv.DictReader(fh)}


# --------------------------------------------------------------------------- the table

def lecture_row(rec: dict, layout: dict[int, dict], words: frozenset[str],
                image_class: str, table_class: str, discipline: Optional[str],
                picture_rules: dict) -> tuple[dict, dict[str, int]]:
    """The lecture's row, and the picture boxes each filter rule dropped (keep_pictures)."""
    slides = rec["slides"]
    texts = [s.get("text") or "" for s in slides]
    missing = [s["index"] for s in slides if s["index"] not in layout]
    if missing:
        raise ValueError(f"{rec['corpus']}/{rec['lecture_id']}: no layout for slides {missing}")
    lay = [layout[s["index"]] for s in slides]
    kept, dropped = keep_pictures(lay, image_class, picture_rules)
    return {
        "corpus": rec["corpus"],
        "lecture_id": rec["lecture_id"],
        "text_source": rec["text_source"],
        "n_slides": len(slides),
        "n_units": len(rec["units"]),
        "n_transcript_words": sum(len((u.get("text") or "").split()) for u in rec["units"]),
        "image_pct": round(area_pct(kept, image_class), 4),
        "table_pct": round(area_pct(lay, table_class), 4),
        "noise": round(noise_pct(texts, words), 4),
        "text_density": round(text_density(texts), 2),
        "discipline": discipline,
    }, dropped


def validate(rows: Sequence[dict], expected: Iterable[tuple[str, str]],
             discipline_blocked: bool) -> None:
    """One row per kept lecture, no missing value, percentages in 0..100."""
    keys = [(r["corpus"], r["lecture_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("a lecture appears more than once")
    exp = set(expected)
    if set(keys) != exp:
        raise ValueError(f"rows != kept lectures: {len(set(keys) - exp)} extra, "
                         f"{len(exp - set(keys))} missing")
    for r in rows:
        if list(r) != COLUMNS:
            raise ValueError(f"columns {list(r)} != {COLUMNS}")
        for c in COLUMNS:
            v = r[c]
            if c == "discipline" and discipline_blocked:
                if v not in (None, ""):
                    raise ValueError("discipline set although the step was blocked")
                continue
            if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
                raise ValueError(f"{r['corpus']}/{r['lecture_id']}: {c} missing")
        for c in PCT_COLUMNS:
            if not 0.0 <= r[c] <= 100.0:
                raise ValueError(f"{r['corpus']}/{r['lecture_id']}: {c}={r[c]} outside 0..100")
        if r["n_slides"] < 1 or r["text_density"] < 0:
            raise ValueError(f"{r['corpus']}/{r['lecture_id']}: bad counts")


def summarise(rows: Sequence[dict]) -> dict:
    """Per corpus: lectures, and median/min/max of each numeric attribute."""
    out = {}
    for c in dict.fromkeys(r["corpus"] for r in rows):
        sub = [r for r in rows if r["corpus"] == c]
        out[c] = {"lectures": len(sub)}
        for col in SUMMARY_COLUMNS:
            vals = [r[col] for r in sub]
            out[c][col] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        d: dict[str, int] = {}
        for r in sub:
            d[r["discipline"] or ""] = d.get(r["discipline"] or "", 0) + 1
        out[c]["discipline"] = dict(sorted(d.items(), key=lambda kv: -kv[1]))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build(cfg: dict, prepared: Path = PREPARED, out_dir: Path = OUT_DIR,
          repo_root: Path = REPO_ROOT) -> tuple[list[dict], dict]:
    lay_cfg = cfg["layout"]
    words = load_wordlist(repo_root / cfg["noise"]["wordlist"])
    layout_path = out_dir / "layout_boxes.jsonl.gz"
    disc_path = out_dir / "discipline.csv"
    layout = read_layout(layout_path)
    disc = read_discipline(disc_path)
    blocked = disc is None
    rows, expected = [], []
    dropped: dict[str, dict[str, int]] = {}
    for corpus in CORPORA:
        d = dropped.setdefault(corpus, {"repeat": 0, "tiny": 0})
        for rec in read_prepared(corpus, prepared):
            key = (corpus, rec["lecture_id"])
            expected.append(key)
            field = None if blocked else disc.get(key)
            if not blocked and field is None:
                raise ValueError(f"{corpus}/{rec['lecture_id']}: not in {disc_path.name}")
            row, n = lecture_row(rec, layout.get(key, {}), words, lay_cfg["image_class"],
                                 lay_cfg["table_class"], field, cfg["picture_filter"])
            rows.append(row)
            for rule in d:
                d[rule] += n[rule]
    extra = set(layout) - set(expected)
    if extra:
        raise ValueError(f"layout has lectures that are not kept: {sorted(extra)[:5]}")
    validate(rows, expected, blocked)
    record = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "columns": COLUMNS,
        "definitions": {
            "image_pct": "100 * sum over slides of the area of the union of the kept "
                         f"'{lay_cfg['image_class']}' boxes / sum over slides of width*height "
                         "(full image); M3AV images are video frames. Dropped: a picture box "
                         "with a picture box at IoU >= repeat_iou (normalised coordinates) on "
                         ">= repeat_share of the lecture's slides (its own slide included), "
                         "then a picture box under tiny_pct % of its slide's area",
            "table_pct": f"the same for '{lay_cfg['table_class']}' boxes",
            "noise": "100 * tokens not in the word list / all tokens, pooled over the "
                     "lecture's slides; token = maximal run of A-Za-z, lowercased, "
                     f"length >= {MIN_TOKEN_LEN}",
            "text_density": "sum over slides of whitespace tokens with a letter or digit, "
                            "/ n_slides",
            "n_transcript_words": "sum over kept units of whitespace-separated tokens",
            "discipline": "OECD FOS 2007 second-level field (scripts/assign_discipline.py)",
        },
        "layout_threshold": lay_cfg["threshold"],
        "picture_filter": dict(cfg["picture_filter"]),
        "pictures_dropped": dropped,
        "wordlist": {"path": cfg["noise"]["wordlist"], "words": len(words),
                     "sha256": sha256_file(repo_root / cfg["noise"]["wordlist"])},
        "inputs": {"layout_boxes": sha256_file(layout_path),
                   "discipline": None if blocked else sha256_file(disc_path),
                   **{f"prepared/{c}.jsonl.gz": sha256_file(prepared / f"{c}.jsonl.gz")
                      for c in CORPORA}},
        "discipline_status": "blocked: no API key" if blocked else "assigned",
        "rows": len(rows),
        "summary": summarise(rows),
    }
    return rows, record


def write(rows: Sequence[dict], record: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "document_attributes.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    with open(out_dir / "document_attributes.json", "w") as fh:
        json.dump(record, fh, indent=1)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    rows, record = build(cfg)
    path = write(rows, record)
    print(f"{len(rows)} lectures -> {path}  (discipline: {record['discipline_status']})")
    for c, s in record["summary"].items():
        parts = [f"{col} {s[col]['median']:.4g} [{s[col]['min']:.4g}, {s[col]['max']:.4g}]"
                 for col in SUMMARY_COLUMNS]
        print(f"  {c} ({s['lectures']}): " + "; ".join(parts))
    print(f"picture boxes dropped (repeat, tiny): {record['pictures_dropped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
