#!/usr/bin/env python3
"""Step 2: prepare every lecture of the four corpora, with the 40-slide cap.

For each lecture (one talk, one deck) the script writes the slides (index, text, text
source, image where the corpus has one), the transcript units in the corpus's own
granularity (with times where the corpus gives them) and the true slide of every unit,
taken from the corpus's ground truth. It reads the archives in data/raw/ (step 1)
directly, without unpacking them, and writes to data/prepared/:

    <corpus>.jsonl.gz        one JSON record per kept lecture
    <corpus>_images.zip      slide images, <lecture>/<slide>.png|jpg
    <corpus>_dropped.csv     lectures not kept, with the reason
    summary.csv              per-corpus counts

The prepared corpora are mavils, m3av_a, m3av_b, mcd and slide_highlight. M3AV is split
into two independent datasets with different lectures (data/splits/m3av_split.csv,
written by scripts/split_m3av.py), m3av_a and m3av_b; both are prepared like any corpus,
with the corpus's hand-corrected OCR as slide text. MCD, Slide-Highlight and the scanned
MaViLS deck take their slide text from our Tesseract OCR (data/ocr/, scripts/ocr.py),
since the corpus gives none; their text source is "tesseract_ocr".

The rules (decided):
  * the cap: slides 1-40 are kept; the transcript is kept from its start up to, not
    including, the first unit whose true slide is > 40. Everything after that point is
    cut, even if the lecture later returns to slides <= 40;
  * lectures whose deck has fewer than 5 slides are dropped;
  * in a corpus with slide images, a lecture is dropped if any kept slide (1-40) has no
    image (reason "slide_without_image"), so every kept lecture can be scored on its
    images. This rule is applied here, after the split: scripts/split_m3av.py keeps its
    lectures (prepare.decide), so the split file does not change; the dropped M3AV
    lectures stay in the split and are listed in their version's dropped file.

Units the ground truth gives no slide (MaViLS '-1', speech outside every annotated
slide segment in M3AV) and units with empty text are not kept: the record counts them,
and every kept unit carries its position in the corpus's own unit list, so the gaps
stay visible. What a "slide", a "unit" and the ground truth are in each corpus is
documented at the loader of that corpus and in README.md.

Usage:
    python scripts/prepare.py --corpus all
    python scripts/prepare.py --corpus m3av_b [--raw-root data/raw] [--ocr-root data/ocr]
        [--split data/splits/m3av_split.csv] [--out-root data/prepared]
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import io
import json
import os
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OCR = REPO_ROOT / "data" / "ocr"
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "m3av_split.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "prepared"

CAP = 40  # slides kept per lecture (a design decision)
MIN_SLIDES = 5  # lectures with fewer slides in the deck are dropped (a design decision)
PDF_DPI = 150  # MaViLS slide images are rendered from the PDF pages at this resolution
# Two M3AV segments with the same slide text are one slide only if their images also
# agree: mean absolute grey-level difference (0-255) of 32x24 thumbnails at most this.
SAME_IMAGE_MAX_DIFF = 10.0
MIN_WORDS_FOR_SAME_TEXT = 3  # shorter texts (a bare title) are too weak to merge on
ZIP_DATE = (2020, 1, 1, 0, 0, 0)  # fixed member date: identical inputs, identical zip

NO_IMAGE = "slide_without_image"  # drop reason: a kept slide has no image (a design decision)
TESSERACT = "tesseract_ocr"  # text source of slides whose text is our OCR (scripts/ocr.py)
M3AV_VERSIONS = {"m3av_a": "A", "m3av_b": "B"}  # prepared corpus -> version in the split


class PrepareError(Exception):
    """The data or the result breaks a rule: stop loudly rather than write it."""


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------- data model


@dataclass
class Slide:
    """One slide of the deck, in deck order. source_ids names what it came from in the
    corpus (PDF page, segment image, slide file); more than one id means several video
    segments were identified as the same slide. image_id is the source id of the image
    the slide shows, which is also the key of its OCR in data/ocr/; ocr is filled when the
    text comes from that OCR (mean word confidence, word count, blank-image flag)."""

    text: str
    source_ids: list[str]
    image: bytes | None = None
    image_ext: str | None = None  # "png" | "jpg"
    image_id: str | None = None
    ocr: dict | None = None


@dataclass
class Unit:
    """One transcript unit in the corpus's own granularity. true_slide is the 1-based
    deck index from the ground truth, or None when the ground truth gives no slide."""

    text: str
    source_index: int  # position in the corpus's full unit list for this lecture
    true_slide: int | None
    start: float | None = None
    end: float | None = None


@dataclass
class Lecture:
    corpus: str
    lecture_id: str
    title: str | None
    text_source: str
    unit_type: str
    time_base: str | None  # "lecture" | "slide_segment" | None (no times)
    slides: list[Slide]
    units: list[Unit]
    meta: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class Dropped:
    corpus: str
    lecture_id: str
    reason: str
    n_slides: int | None
    detail: str = ""


# ------------------------------------------------------------------- the rules


def cap_point(units: list[Unit], cap: int = CAP) -> int | None:
    """Index of the first unit whose true slide is > cap, or None if there is none.
    The transcript is kept up to, not including, this unit."""
    for i, u in enumerate(units):
        if u.true_slide is not None and u.true_slide > cap:
            return i
    return None


def drop_reason(lec: Lecture) -> str | None:
    if not lec.slides:
        return "no_slides"
    if len(lec.slides) < MIN_SLIDES:
        return f"fewer_than_{MIN_SLIDES}_slides"
    return None


def decide(corpus: str, lec: Lecture) -> tuple[dict, dict[str, bytes]] | Dropped:
    """Whether a loaded lecture is kept: (record, images) after the cap, or why not.
    scripts/split_m3av.py calls this too, so the split covers exactly the kept lectures."""
    reason = drop_reason(lec)
    if reason:
        return Dropped(corpus, lec.lecture_id, reason, len(lec.slides))
    record, images = build_record(lec)
    if not record["units"]:
        return Dropped(corpus, lec.lecture_id, "no_labelled_units", len(lec.slides))
    return record, images


def normalise_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def identify_slides(
    segments: list[tuple[str, str]],
    same_image: Callable[[int, int], bool] | None = None,
) -> tuple[list[list[int]], list[int], int]:
    """Group video segments (id, text) in time order into slides.

    A segment is the same slide as an earlier one when their normalised texts are equal
    and have at least MIN_WORDS_FOR_SAME_TEXT words, and (when same_image is given) the
    images agree. Slides are numbered by first appearance. Returns (members of each
    slide as segment positions, the slide index (0-based) of every segment, the number
    of text-equal pairs kept apart because the images differ)."""
    groups: list[list[int]] = []
    by_text: dict[str, list[int]] = collections.defaultdict(list)
    seg_slide: list[int] = []
    kept_apart = 0
    for i, (_, text) in enumerate(segments):
        key = normalise_text(text)
        target = None
        if len(key.split()) >= MIN_WORDS_FOR_SAME_TEXT:
            for g in by_text[key]:
                if same_image is None or same_image(groups[g][0], i):
                    target = g
                    break
                kept_apart += 1
        if target is None:
            target = len(groups)
            groups.append([])
            if len(key.split()) >= MIN_WORDS_FOR_SAME_TEXT:
                by_text[key].append(target)
        groups[target].append(i)
        seg_slide.append(target)
    return groups, seg_slide, kept_apart


def assign_by_time(intervals: list[tuple[float, float]], word_times: list[tuple[float, float]]) -> int | None:
    """The interval holding most of a unit's words (by word midpoint); ties go to the
    earlier interval. None when no word falls in any interval. Intervals are
    half-open [start, end)."""
    votes: collections.Counter[int] = collections.Counter()
    for s, e in word_times:
        mid = (s + e) / 2
        for k, (a, b) in enumerate(intervals):
            if a <= mid < b:
                votes[k] += 1
                break
    if not votes:
        return None
    best = max(votes.values())
    return min(k for k, v in votes.items() if v == best)


def build_record(lec: Lecture) -> tuple[dict, dict[str, bytes]]:
    """Apply the cap and return (the JSON record, {zip member: image bytes})."""
    n_slides = len(lec.slides)
    kept_slides = lec.slides[:CAP]
    cut = cap_point(lec.units)
    window = lec.units if cut is None else lec.units[:cut]
    labelled_all = [u for u in lec.units if u.true_slide is not None and u.text.strip()]
    kept_units = [u for u in window if u.true_slide is not None and u.text.strip()]
    images: dict[str, bytes] = {}
    slides_out = []
    for i, s in enumerate(kept_slides, start=1):
        member = None
        if s.image is not None:
            member = f"{lec.lecture_id}/{i:02d}.{s.image_ext}"
            images[member] = s.image
        slides_out.append({"index": i, "text": s.text, "image": member, "source_ids": s.source_ids, "ocr": s.ocr})
    units_out = [
        {
            "index": j,
            "source_index": u.source_index,
            "text": u.text,
            "start": u.start,
            "end": u.end,
            "true_slide": u.true_slide,
        }
        for j, u in enumerate(kept_units)
    ]
    n_unlab = sum(1 for u in lec.units if u.true_slide is None)
    n_empty = sum(1 for u in lec.units if u.true_slide is not None and not u.text.strip())
    record = {
        "corpus": lec.corpus,
        "lecture_id": lec.lecture_id,
        "title": lec.title,
        "text_source": lec.text_source,
        "unit_type": lec.unit_type,
        "time_base": lec.time_base,
        "n_slides_before_cap": n_slides,
        "n_slides_after_cap": len(kept_slides),
        "n_units_before_cap": len(labelled_all),
        "n_units_after_cap": len(kept_units),
        "cut_applied": n_slides > CAP or cut is not None,
        "transcript_cut_at_source_index": None if cut is None else lec.units[cut].source_index,
        "n_units_raw": len(lec.units),
        "n_units_without_slide": n_unlab,
        "n_units_empty_text": n_empty,
        "n_slides_with_text": sum(1 for s in kept_slides if s.text.strip()),
        "n_slides_ocr": sum(1 for s in kept_slides if s.ocr is not None),
        "n_slides_ocr_empty_text": sum(1 for s in kept_slides if s.ocr is not None and not s.text.strip()),
        "n_images": len(images),
        "meta": lec.meta,
        "diagnostics": lec.diagnostics,
        "slides": slides_out,
        "units": units_out,
    }
    return record, images


def missing_images(record: dict) -> list[int]:
    """The kept slides (1-based index) that have no image."""
    return [s["index"] for s in record["slides"] if s["image"] is None]


def validate(record: dict, images: dict[str, bytes], corpus_has_images: bool) -> None:
    """The checks decided for the data; any failure stops the run."""
    lid = f"{record['corpus']}/{record['lecture_id']}"
    slides, units = record["slides"], record["units"]
    n = len(slides)
    if n == 0:
        raise PrepareError(f"{lid}: empty slide list")
    if n > CAP or record["n_slides_after_cap"] != n:
        raise PrepareError(f"{lid}: {n} slides kept, cap is {CAP}")
    if [s["index"] for s in slides] != list(range(1, n + 1)):
        raise PrepareError(f"{lid}: slide indices are not 1..{n}")
    if not units:
        raise PrepareError(f"{lid}: no transcript unit kept")
    for u in units:
        t = u["true_slide"]
        if not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= n:
            raise PrepareError(f"{lid}: unit {u['source_index']} has true slide {t!r}, not in 1..{n}")
        if not u["text"].strip():
            raise PrepareError(f"{lid}: unit {u['source_index']} has empty text")
    src = [u["source_index"] for u in units]
    if any(b <= a for a, b in zip(src, src[1:])):
        raise PrepareError(f"{lid}: unit order not preserved")
    if record["n_units_after_cap"] != len(units):
        raise PrepareError(f"{lid}: unit count mismatch")
    with_image = [s["image"] for s in slides if s["image"] is not None]
    if sorted(with_image) != sorted(images) or record["n_images"] != len(images):
        raise PrepareError(f"{lid}: {len(images)} images for {len(with_image)} slides that name one")
    if corpus_has_images:
        if missing_images(record):
            # such lectures are dropped (NO_IMAGE) before they get here
            raise PrepareError(f"{lid}: slides {missing_images(record)} have no image")
    elif images:
        raise PrepareError(f"{lid}: images written for a corpus without images")
    ocrd = [s for s in slides if s["ocr"] is not None]
    if record["text_source"] == TESSERACT:
        for s in slides:
            if s["image"] is not None and s["ocr"] is None:
                raise PrepareError(f"{lid}: slide {s['index']} has an image but no OCR")
            # Empty OCR text is accepted only where there is nothing to read.
            if s["ocr"] is not None and not s["text"].strip() and not s["ocr"]["blank_image"]:
                raise PrepareError(f"{lid}: slide {s['index']} has empty OCR text but its image is not blank")
    elif ocrd:
        raise PrepareError(f"{lid}: OCR on slides of a lecture whose text source is {record['text_source']}")


# ----------------------------------------------------------------- archive access


@dataclass
class Sources:
    """What the loaders read: the raw archives (step 1), our OCR (scripts/ocr.py) and the
    M3AV split (scripts/split_m3av.py)."""

    raw: Path
    ocr: Path = DEFAULT_OCR
    split: Path = DEFAULT_SPLIT


def one_archive(raw: Path, corpus: str) -> Path:
    src = json.loads((raw / corpus / "SOURCE.json").read_text())
    path = raw / corpus / src["verification"]["archive"]
    if not path.is_file():
        raise PrepareError(f"{path} missing: run scripts/download.py --corpus {corpus}")
    return path


def read_tar(path: Path, wanted: Callable[[str], bool]) -> dict[str, bytes]:
    """Members matching `wanted`, read in one streaming pass (works for .tar and .tar.gz)."""
    out: dict[str, bytes] = {}
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if m.isfile() and wanted(m.name):
                out[m.name] = tf.extractfile(m).read()
    return out


def load_ocr(src: Sources, name: str) -> tuple[dict[tuple[str, str], dict], dict]:
    """Our OCR of one corpus: ({(lecture, image id): OCR line}, the engine record)."""
    jl, rec = src.ocr / f"{name}.jsonl.gz", src.ocr / f"{name}.json"
    if not jl.is_file() or not rec.is_file():
        raise PrepareError(f"{jl} or {rec.name} missing: run scripts/ocr.py --corpus {name}")
    record = json.loads(rec.read_text())
    out: dict[tuple[str, str], dict] = {}
    with gzip.open(jl, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["lecture"], r["slide"])
            if key in out:
                raise PrepareError(f"{jl}: {key} twice")
            out[key] = r
    return out, record


def ocr_meta(name: str, record: dict) -> dict:
    """The short form of the OCR record kept in every lecture that uses it."""
    return {
        "record": f"data/ocr/{name}.json",
        "engine": record["engine"],
        "version": record["tesseract_version"],
        "language_data": record["language_data"]["package"],
        "settings": record["settings"],
    }


def apply_ocr(slides: list[Slide], lecture_id: str, ocr: dict[tuple[str, str], dict], what: str) -> None:
    """Replace the text of every slide that has an image by our OCR of that image."""
    for s in slides:
        if s.image_id is None:
            s.text = ""
            continue
        hit = ocr.get((lecture_id, s.image_id))
        if hit is None:
            raise PrepareError(f"{what} {lecture_id}: no OCR for image {s.image_id}; rerun scripts/ocr.py")
        s.text = hit["text"]
        s.ocr = {"mean_conf": hit["mean_conf"], "n_words": hit["n_words"], "blank_image": hit["blank_image"]}


def read_split(path: Path) -> dict[str, str]:
    """{lecture id: 'A' | 'B'} from data/splits/m3av_split.csv."""
    if not path.is_file():
        raise PrepareError(f"{path} missing: run scripts/split_m3av.py")
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["lecture_id", "version", "stratum"]:
            raise PrepareError(f"{path}: header {reader.fieldnames}, expected lecture_id,version,stratum")
        for r in reader:
            if r["version"] not in ("A", "B"):
                raise PrepareError(f"{path}: version {r['version']!r} for {r['lecture_id']}")
            if r["lecture_id"] in out:
                raise PrepareError(f"{path}: {r['lecture_id']} is listed twice (in both versions?)")
            out[r["lecture_id"]] = r["version"]
    return out


# ----------------------------------------------------------------------- MaViLS

# MaViLS ships 21 slide PDFs, 20 faster-whisper SRT files and 24 ground-truth xlsx files.
# 20 xlsx files are the main labels and 4 are control re-labelings by a second rater.
# The authors' own evaluation script (evaluation/evaluate_recall_precision.py at the
# pinned commit) reads, for cognitive robotics, ground_truth_cognitive_robotics_MIT_control.xlsx;
# the file without the suffix has the 'Minutes' column of the other three control files
# (physics/team_dynamics/theory_of_computation ..._labeled_control.xlsx). So the names are
# swapped for that lecture, and the main labels are the 20 files below. Each xlsx row is
# one sentence of the lecturer (made from the SRT by the authors' sentence splitter) with
# its time and the human slide label; '-1' means no slide visible or not identifiable.
# PDF pairing: by name, confirmed by word overlap with every PDF (checked at run time).
# Slide text: the PDF text layer; a deck without one (the scanned image_processing.pdf)
# gets our Tesseract OCR of the 150-dpi page renders, the same images as in the zip.
MAVILS_LECTURES = {
    # lecture id: (ground-truth xlsx, slide PDF, faster-whisper SRT the sentences come from)
    "ML_for_health_MIT": ("ground_truth_ML_for_health_MIT.xlsx", "ML_for_health_care_MIT.pdf", "ML_for_health_MIT.srt"),
    "cities_and_decarbonization": ("ground_truth_cities_and_decarbonization.xlsx", "cities&decarbonization.pdf", "cities_and_decarbonization.srt"),
    "climate_and_cities": ("ground_truth_climate_and_cities.xlsx", "cities&climate.pdf", "climate_and_cities.srt"),
    "climate_science_policy_MIT": ("ground_truth_climate_science_policy_MIT2.xlsx", "climate_science_policy_MIT.pdf", "climate_science_policy_MIT.srt"),
    "cognitive_robotics_MIT": ("ground_truth_cognitive_robotics_MIT_control.xlsx", "cognitive_robotics_MIT.pdf", "cognitive_robotics_MIT.srt"),
    "computer_vision_2_2": ("ground_truth_computer_vision_2_2.xlsx", "computer_vision_2_2_geiger.pdf", "computer_vision_2_2_high_res.srt"),
    "creating_breakthrough_products_MIT": ("ground_truth_creating_breakthrough_products_MIT.xlsx", "creating_breakthrough_products.pdf", "creating_breakthrough_products_MIT.srt"),
    "cryptocurrency_MIT": ("ground_truth_cryptocurrency_MIT.xlsx", "cryptocurrency.pdf", "cryptocurrency.srt"),
    "deeplearning_goodfellow": ("ground_truth_deeplearning.xlsx", "deeplearning_goodfellow.pdf", "deeplearning_goodfellow.srt"),
    "image_processing": ("ground_truth_image_processing.xlsx", "image_processing.pdf", "image_processing_high_res.srt"),
    "numerics": ("ground_truth_numerics.xlsx", "numerics.pdf", "numerics_hennig.srt"),
    "phonetics": ("ground_truth_phonetics.xlsx", "phonetics.pdf", "phonetics.srt"),
    "physics_intro_02": ("ground_truth_physics.xlsx", "physics_intro_02.pdf", "physics_intro_02.srt"),
    "psychology_MIT": ("ground_truth_psychology.xlsx", "psychology.pdf", "psychology_MIT.srt"),
    "reinforcement_learning_silver": ("ground_truth_reinforcement_learning.xlsx", "reinforcement_learning.pdf", "reinforcement_learning_silver.srt"),
    "sensory_systems_MIT": ("ground_truth_sensory_systems.xlsx", "sensory_systems.pdf", "visual_system_MIT.srt"),
    "short_range_mit": ("ground_truth_short_range.xlsx", "short_range_mit.pdf", "short_range_mit.srt"),
    "solar_resource": ("ground_truth_solar_resource.xlsx", "solar_resource.pdf", "solar_resource.srt"),
    "team_dynamics_game_design_MIT": ("ground_truth_team_dynamics_game_design_MIT.xlsx", "team_dynamics_game_design_mit.pdf", "team_dynamics_game_design_MIT.srt"),
    "theory_of_computation_MIT": ("ground_truth_theory_of_computation.xlsx", "theory_of_computation.pdf", "theory_of_computation_MIT.srt"),
}
MAVILS_CONTROL = (
    "ground_truth_cognitive_robotics_MIT.xlsx",
    "physics_ground_truth_labeled_control.xlsx",
    "team_dynamics_game_design_MIT_ground_truth_labeled_control.xlsx",
    "theory_of_computation_MIT_ground_truth_labeled_control.xlsx",
)


def mavils_page_id(pdf: str, k: int) -> str:
    """Source id of page k (0-based) of a MaViLS PDF: also the image's OCR key."""
    return f"{pdf}#page={k + 1}"


def render_page(page) -> bytes:
    """The slide image of a PDF page; scripts/ocr.py OCRs exactly these bytes."""
    return page.get_pixmap(dpi=PDF_DPI).tobytes("png")


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{5,}", s.lower()))


def read_xlsx_rows(data: bytes) -> list[tuple]:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    wb.close()
    return rows


def mavils_units(rows: list[tuple], n_pages: int) -> tuple[list[Unit], dict]:
    """Units from a MaViLS ground-truth sheet: one row per sentence, columns located by
    header ('Time' or 'Key' = start second, 'Slidenumber', 'Value' = sentence)."""
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    try:
        t_col = header.index("Time") if "Time" in header else header.index("Key")
        s_col, v_col = header.index("Slidenumber"), header.index("Value")
    except ValueError as e:
        raise PrepareError(f"unexpected MaViLS header {header}") from e
    units, labels = [], collections.Counter()
    for i, r in enumerate(rows[1:]):
        lab = r[s_col]
        if isinstance(lab, float) and lab.is_integer():
            lab = int(lab)
        if not isinstance(lab, int):
            raise PrepareError(f"MaViLS row {i + 2}: slide label {lab!r} is not an integer")
        if lab > n_pages:
            raise PrepareError(f"MaViLS row {i + 2}: slide {lab} > {n_pages} PDF pages")
        labels["valid" if lab >= 1 else str(lab)] += 1
        t = r[t_col]
        units.append(
            Unit(
                text=str(r[v_col]).strip() if r[v_col] is not None else "",
                source_index=i,
                true_slide=lab if lab >= 1 else None,
                start=float(t) if isinstance(t, (int, float)) else None,
            )
        )
    return units, {"gt_labels": dict(labels)}


def mavils_files(raw: Path) -> dict[str, bytes]:
    """{file name: bytes} of the MaViLS PDFs, xlsx and SRT files."""
    files = read_tar(one_archive(raw, "mavils"), lambda n: n.endswith((".pdf", ".xlsx", ".srt")))
    return {Path(n).name: b for n, b in files.items()}


def load_mavils(src: Sources) -> Iterator[Lecture | Dropped]:
    import fitz  # PyMuPDF

    by_name = mavils_files(src.raw)
    pdf_words = {}
    for name, b in by_name.items():
        if name.endswith(".pdf"):
            with fitz.open(stream=b, filetype="pdf") as doc:
                pdf_words[name] = _words(" ".join(p.get_text() for p in doc))
    used = {v[1] for v in MAVILS_LECTURES.values()}
    for name in sorted(pdf_words):
        if name not in used:
            yield Dropped("mavils", Path(name).stem, "no_ground_truth", None, f"{name} has no ground-truth xlsx")
    ocr = None
    for lid, (gt, pdf, srt) in sorted(MAVILS_LECTURES.items()):
        for f in (gt, pdf, srt):
            if f not in by_name:
                raise PrepareError(f"MaViLS: {f} missing from the archive")
        with fitz.open(stream=by_name[pdf], filetype="pdf") as doc:
            n_pages = doc.page_count
            texts = [p.get_text().strip() for p in doc]
            slides = []
            for k, page in enumerate(doc):
                png = render_page(page) if k < CAP else None
                sid = mavils_page_id(pdf, k)
                slides.append(Slide(text=texts[k], source_ids=[sid], image=png, image_ext="png" if png else None, image_id=sid))
        units, diag = mavils_units(read_xlsx_rows(by_name[gt]), n_pages)
        meta = {"gt_file": gt, "pdf": pdf, "srt": srt, "image_dpi": PDF_DPI, "control_relabelings_not_used": list(MAVILS_CONTROL)}
        if any(texts):
            # The pairing check: the sentences must share more words with this PDF than
            # with any other (a PDF without a text layer cannot be checked).
            gw = _words(" ".join(u.text for u in units))
            score = {p: len(gw & w) / max(1, len(w)) for p, w in pdf_words.items()}
            if max(score, key=score.get) != pdf:
                raise PrepareError(f"MaViLS {lid}: transcript matches {max(score, key=score.get)} better than {pdf}")
            text_source = "pdf_text_layer"
        else:
            # A scanned deck (no text layer at all): our OCR of the page renders.
            if ocr is None:
                ocr = load_ocr(src, "mavils")
            apply_ocr(slides, lid, ocr[0], "MaViLS")
            text_source = TESSERACT
            meta["ocr"] = ocr_meta("mavils", ocr[1])
        diag["pdf_pages_without_text"] = sum(1 for t in texts if not t)
        yield Lecture(
            corpus="mavils",
            lecture_id=lid,
            title=None,
            text_source=text_source,
            unit_type="sentence",
            time_base="lecture",
            slides=slides,
            units=units,
            meta=meta,
            diagnostics=diag,
        )


# ------------------------------------------------------------------------- M3AV

# M3AV v1.0: per talk, ocr/seg_speech_ocr.json lists the slide segments of the video in
# time order (start, end, image name, merged OCR text), ocr/seg_imgs/ has one image per
# segment, speech/final+timestamps.json has the speech units (<=10 s, written form, with
# word times). The OCR is PaddleOCR (Mathpix for formulae) corrected by hand, with
# adjacent frames already merged by the authors. The same slide can still appear in
# several, non-adjacent segments (the speaker goes back). A slide here is a group of
# segments with identical text whose images also agree (identify_slides); slides are
# numbered by first appearance. The true slide of a speech unit is the slide of the
# segment holding most of its words (word midpoints); words outside every segment have
# no slide.
#
# The kept lectures are split in two independent datasets, m3av_a and m3av_b, with
# different lectures (scripts/split_m3av.py); both are prepared the same way.

M3AV_RX = re.compile(
    r"^\./dataset_v1\.0_noaudio/([^/]+)/(ocr/seg_speech_ocr\.json|speech/final\+timestamps\.json|ocr/seg_imgs/[^/]+\.jpg|raw/[^/]+\.ytbUrl)$"
)


def _thumb(jpg: bytes) -> bytes:
    import fitz

    with fitz.open(stream=jpg, filetype="jpg") as d:
        r = d[0].rect
        return d[0].get_pixmap(matrix=fitz.Matrix(32 / r.width, 24 / r.height), colorspace=fitz.csGRAY).samples


def image_diff(a: bytes, b: bytes) -> float:
    ta, tb = _thumb(a), _thumb(b)
    if len(ta) != len(tb):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(ta, tb)) / len(ta)


def m3av_lecture(
    talk: str,
    seg: list[dict],
    speech: list[dict],
    jpgs: dict[str, bytes],
    url: str | None,
    corpus: str = "m3av",
) -> Lecture:
    """One M3AV talk, with the corpus's corrected OCR as slide text."""
    names = [s["name"] for s in seg]
    if len(set(names)) != len(names):
        raise PrepareError(f"M3AV {talk}: duplicate segment names")

    def same_image(i: int, j: int) -> bool:
        a, b = jpgs.get(names[i]), jpgs.get(names[j])
        return a is not None and b is not None and image_diff(a, b) <= SAME_IMAGE_MAX_DIFF

    groups, seg_slide, kept_apart = identify_slides([(s["name"], s["ocr_text"]) for s in seg], same_image)
    slides = []
    for g in groups:
        img_name = next((names[i] for i in g if names[i] in jpgs), None)
        slides.append(
            Slide(
                text=seg[g[0]]["ocr_text"].strip(),
                source_ids=[names[i] for i in g],
                image=jpgs.get(img_name) if img_name else None,
                image_ext="jpg" if img_name else None,
                image_id=img_name,
            )
        )
    intervals = [(float(s["start"]), float(s["end"])) for s in seg]
    units, straddle = [], 0
    for i, u in enumerate(speech):
        words = [(w["start"], w["end"]) for w in u.get("words_spoken", [])]
        k = assign_by_time(intervals, words)
        if k is not None and len({assign_by_time(intervals, [w]) for w in words} - {None}) > 1:
            straddle += 1
        a, b = u["timestr"].split("_")
        units.append(
            Unit(
                text=(u.get("final_written") or "").strip(),
                source_index=i,
                true_slide=None if k is None else seg_slide[k] + 1,
                start=int(a) / 1000,
                end=int(b) / 1000,
            )
        )
    missing = [i + 1 for i, s in enumerate(slides[:CAP]) if s.image is None]
    return Lecture(
        corpus=corpus,
        lecture_id=talk,
        title=None,
        text_source="corpus_corrected_ocr",
        unit_type="speech_segment",
        time_base="lecture",
        slides=slides,
        units=units,
        meta={"venue_prefix": talk.split("-")[0], "youtube_url": url, "n_segments": len(seg)},
        diagnostics={
            "segments_merged_into_earlier_slide": len(seg) - len(groups),
            "same_text_kept_apart_by_image": kept_apart,
            "units_straddling_segments": straddle,
            "segment_images_missing": sum(1 for n in names if n not in jpgs),
            "images_without_segment": sum(1 for n in jpgs if n not in set(names)),
            "slides_without_image_in_corpus": missing,
        },
    )


def load_m3av(src: Sources, version: str | None = None) -> Iterator[Lecture | Dropped]:
    """The M3AV talks of one split version ('A' | 'B'), or all of them (version None:
    what scripts/split_m3av.py splits). Talks in no version are the
    ones the rules drop; they are loaded in both versions, so each version's dropped list
    is complete, and prepare_corpus stops if one of them would be kept."""
    split = read_split(src.split) if version else {}
    corpus = "m3av" if version is None else f"m3av_{version.lower()}"

    def wanted(talk: str) -> bool:
        return version is None or split.get(talk, version) == version

    talks: dict[str, dict] = collections.defaultdict(lambda: {"jpg": {}})
    with tarfile.open(one_archive(src.raw, "m3av"), "r|gz") as tf:
        for m in tf:
            hit = M3AV_RX.match(m.name) if m.isfile() else None
            if not hit or not wanted(hit[1]):
                continue
            talk, rel = hit.groups()
            data = tf.extractfile(m).read()
            if rel.endswith(".jpg"):
                talks[talk]["jpg"][Path(rel).name] = data
            elif rel.endswith(".ytbUrl"):
                talks[talk]["url"] = data.decode().strip()
            else:
                talks[talk]["seg" if "seg_speech" in rel else "speech"] = json.loads(data)
    for talk in sorted(talks):
        t = talks.pop(talk)  # free the images as we go
        if "seg" not in t or "speech" not in t:
            raise PrepareError(f"M3AV {talk}: segment or speech file missing")
        if not t["seg"]:
            yield Dropped(corpus, talk, "no_slides", 0, "seg_speech_ocr.json is empty")
            continue
        lec = m3av_lecture(talk, t["seg"], t["speech"], t["jpg"], t.get("url"), corpus)
        if version is not None:
            lec.meta["split_version"] = split.get(talk)
        yield lec


# -------------------------------------------------------------------------- MCD

# MCD presentation set (Dataset/PRES): per talk, vid-seg/slides/sp_X/sp_X_k.png is a
# slide and vid-seg/trans/sp_X/sp_X_k.txt the WhisperX transcript spoken over it, one
# line per sentence (no times). The ground truth is this file-name pairing. The authors
# removed title/outline/thank-you slides and kept only the final state of animated
# slides, so k has gaps; the slides are renumbered 1..n in k order. No slide text is
# distributed: the text is our Tesseract OCR of the slide image. The 'test' split has
# the 20 talks of the paper, 'eg' 5 more (few-shot examples); both are kept and the
# split is recorded.

MCD_RX = re.compile(r"^Dataset/PRES/(test|eg)/input/vid-seg/(slides|trans)/(sp_\d+)/\3_(\d+)\.(png|txt)$")


def mcd_slide_id(sp: str, k: int) -> str:
    return f"{sp}_{k:03d}.png"


def load_mcd(src: Sources) -> Iterator[Lecture | Dropped]:
    files = read_tar(one_archive(src.raw, "mcd"), lambda n: bool(MCD_RX.match(n)))
    ocr, ocr_record = load_ocr(src, "mcd")
    talks: dict[tuple[str, str], dict] = collections.defaultdict(lambda: {"slides": {}, "trans": {}})
    for n, b in files.items():
        split, kind, sp, k, _ = MCD_RX.match(n).groups()
        talks[(split, sp)][kind][int(k)] = b
    ids = collections.Counter(sp for _, sp in talks)
    for (split, sp), t in sorted(talks.items(), key=lambda kv: kv[0][1]):
        if ids[sp] > 1:
            raise PrepareError(f"MCD {sp} is in more than one split")
        if sorted(t["slides"]) != sorted(t["trans"]):
            raise PrepareError(f"MCD {sp}: slide and transcript ids differ")
        ks = sorted(t["slides"])
        slides = [
            Slide(text="", source_ids=[mcd_slide_id(sp, k)], image=t["slides"][k], image_ext="png", image_id=mcd_slide_id(sp, k))
            for k in ks
        ]
        apply_ocr(slides, sp, ocr, "MCD")
        units = []
        for idx, k in enumerate(ks, start=1):
            for line in t["trans"][k].decode("utf-8").splitlines():
                if line.strip():
                    units.append(Unit(text=line.strip(), source_index=len(units), true_slide=idx))
        yield Lecture(
            corpus="mcd",
            lecture_id=sp,
            title=None,
            text_source=TESSERACT,
            unit_type="transcript_line",
            time_base=None,
            slides=slides,
            units=units,
            meta={"split": split, "slide_numbers_in_corpus": ks, "ocr": ocr_meta("mcd", ocr_record)},
            diagnostics={"slides_without_image_in_corpus": []},
        )


# --------------------------------------------------------------- Slide-Highlight

# Slide-Highlight: per talk (ICML/NeurIPS vid_N), Dataset/slides/.../PART_<k>_<id>.png is
# slide k and Dataset/transcripts/.../PART_<k>_<id>.srt the WhisperX subtitle cues
# spoken over it; the ground truth is this file-name pairing. Cue times restart in every
# PART, so they are slide-segment relative (time_base 'slide_segment'). NeurIPS vid_0054
# repeats its 11 images in a subfolder 38992648/ (byte-identical; ignored) and GT has a
# second copy of vid_0113 as vid_113 (highlight boxes only; not used here). The slide
# text is our Tesseract OCR of the slide image.

SH_RX = re.compile(r"^Dataset/(slides|transcripts)/(ICML|NeurIPS)_(slides|trans)/(vid_\d+)/PART_(\d+)_([^/.]+)\.(png|srt)$")


def sh_lecture_id(conf: str, vid: str) -> str:
    return f"{conf}_{vid}"


def sh_slide_id(k: int, sid: str) -> str:
    return f"PART_{k:04d}_{sid}"


def sh_wanted(n: str) -> bool:
    m = SH_RX.match(n)
    return bool(m) and (m[1], m[7]) in {("slides", "png"), ("transcripts", "srt")}


SRT_CUE = re.compile(
    r"(\d+):(\d\d):(\d\d)[,.](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{3})[^\n]*\n(.*?)(?:\n\s*\n|\Z)", re.S
)


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    out = []
    for m in SRT_CUE.finditer(text.replace("\r\n", "\n")):
        g = m.groups()
        s = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        e = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        out.append((s, e, " ".join(g[8].split())))
    return out


def load_slide_highlight(src: Sources) -> Iterator[Lecture | Dropped]:
    files = read_tar(one_archive(src.raw, "slide_highlight"), sh_wanted)
    ocr, ocr_record = load_ocr(src, "slide_highlight")
    talks: dict[tuple[str, str], dict] = collections.defaultdict(lambda: {"png": {}, "srt": {}})
    for n, b in files.items():
        m = SH_RX.match(n)
        talks[(m[2], m[4])][m[7]][(int(m[5]), m[6])] = b
    for (conf, vid), t in sorted(talks.items()):
        lid = sh_lecture_id(conf, vid)
        parts = sorted(t["png"])
        if parts != sorted(t["srt"]):
            raise PrepareError(f"Slide-Highlight {lid}: slide and transcript parts differ")
        if [k for k, _ in parts] != list(range(len(parts))):
            raise PrepareError(f"Slide-Highlight {lid}: PART numbers are not 0..{len(parts) - 1}")
        if len({i for _, i in parts}) != len(parts):
            raise PrepareError(f"Slide-Highlight {lid}: a slide id appears in two PARTs")
        slides = [
            Slide(text="", source_ids=[sh_slide_id(*p)], image=t["png"][p], image_ext="png", image_id=sh_slide_id(*p))
            for p in parts
        ]
        apply_ocr(slides, lid, ocr, "Slide-Highlight")
        units = []
        for idx, p in enumerate(parts, start=1):
            for s, e, txt in parse_srt(t["srt"][p].decode("utf-8")):
                units.append(Unit(text=txt, source_index=len(units), true_slide=idx, start=s, end=e))
        yield Lecture(
            corpus="slide_highlight",
            lecture_id=lid,
            title=None,
            text_source=TESSERACT,
            unit_type="subtitle_cue",
            time_base="slide_segment",
            slides=slides,
            units=units,
            meta={"conference": conf, "video": vid, "ocr": ocr_meta("slide_highlight", ocr_record)},
            diagnostics={"slides_without_image_in_corpus": []},
        )


# ---------------------------------------------------------------------- outputs

M3AV_GT = "speech-to-slide-segment times (seg_speech_ocr.json + word times)"
LOADERS: dict[str, tuple[Callable[[Sources], Iterable[Lecture | Dropped]], bool, str]] = {
    # corpus: (loader, has images, ground-truth source)
    "mavils": (load_mavils, True, "human sentence-to-slide labels (xlsx, main file per lecture)"),
    "m3av_a": (lambda src: load_m3av(src, "A"), True, M3AV_GT),
    "m3av_b": (lambda src: load_m3av(src, "B"), True, M3AV_GT),
    "mcd": (load_mcd, True, "file-name pairing slides/sp_X_k.png <-> trans/sp_X_k.txt"),
    "slide_highlight": (load_slide_highlight, True, "file-name pairing PART_k_id.png <-> PART_k_id.srt"),
}


def atomic_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def prepare_corpus(name: str, src: Sources, out: Path) -> dict:
    loader, has_images, gt_source = LOADERS[name]
    out.mkdir(parents=True, exist_ok=True)
    jl, zp, dc = out / f"{name}.jsonl.gz", out / f"{name}_images.zip", out / f"{name}_dropped.csv"
    dropped: list[Dropped] = []
    totals = collections.Counter()
    sources, unit_types = collections.Counter(), collections.Counter()
    conf_sum, conf_n = 0.0, 0
    seen: set[str] = set()
    parts = [atomic_path(p) for p in (jl, zp, dc)]
    zf = zipfile.ZipFile(atomic_path(zp), "w", zipfile.ZIP_STORED) if has_images else None
    try:
        with open(atomic_path(jl), "wb") as raw_f, gzip.GzipFile(filename="", fileobj=raw_f, mode="wb", mtime=0) as gz:
            for item in loader(src):
                totals["available"] += 1
                if isinstance(item, Dropped):
                    dropped.append(item)
                    continue
                if item.lecture_id in seen:
                    raise PrepareError(f"{name}: lecture id {item.lecture_id} twice")
                seen.add(item.lecture_id)
                result = decide(name, item)
                if isinstance(result, Dropped):
                    dropped.append(result)
                    continue
                record, images = result
                if "split_version" in record["meta"] and record["meta"]["split_version"] is None:
                    raise PrepareError(f"{name}: lecture {item.lecture_id} is kept but is in no split version; rerun scripts/split_m3av.py")
                if has_images and missing_images(record):
                    dropped.append(Dropped(name, item.lecture_id, NO_IMAGE, len(item.slides),
                                           f"slides {missing_images(record)} of the kept slides have no image"))
                    continue
                validate(record, images, has_images)
                gz.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                for member in sorted(images):
                    zf.writestr(zipfile.ZipInfo(member, ZIP_DATE), images[member])
                totals["kept"] += 1
                totals["slides_before_cap"] += record["n_slides_before_cap"]
                totals["slides_after_cap"] += record["n_slides_after_cap"]
                totals["units_before_cap"] += record["n_units_before_cap"]
                totals["units_after_cap"] += record["n_units_after_cap"]
                totals["units_without_slide"] += record["n_units_without_slide"]
                totals["units_empty_text"] += record["n_units_empty_text"]
                totals["lectures_cut"] += record["cut_applied"]
                totals["transcripts_cut"] += record["transcript_cut_at_source_index"] is not None
                totals["images"] += record["n_images"]
                totals["slides_without_text"] += record["n_slides_after_cap"] - record["n_slides_with_text"]
                totals["slides_ocr"] += record["n_slides_ocr"]
                totals["slides_ocr_empty_text"] += record["n_slides_ocr_empty_text"]
                for s in record["slides"]:
                    if s["ocr"] is not None and s["ocr"]["mean_conf"] is not None:
                        conf_sum += s["ocr"]["mean_conf"]
                        conf_n += 1
                sources[record["text_source"]] += 1
                unit_types[record["unit_type"]] += 1
                for k, v in record["diagnostics"].items():
                    if isinstance(v, bool):
                        totals["diag_" + k] += v
                    elif isinstance(v, int):
                        totals["diag_" + k] += v
                    elif isinstance(v, list):
                        totals["diag_" + k] += len(v)
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            totals[f"diag_{k}_{kk}"] += vv
            if name in M3AV_VERSIONS:
                # The version holds exactly the split's lectures of that version, less the
                # ones dropped for a slide without an image (a rule applied after the split).
                want = {k for k, v in read_split(src.split).items() if v == M3AV_VERSIONS[name]}
                kept = seen - {d.lecture_id for d in dropped}
                no_image = {d.lecture_id for d in dropped if d.reason == NO_IMAGE}
                if kept != want - no_image:
                    raise PrepareError(
                        f"{name}: kept lectures differ from the split ({len(kept - want)} extra, "
                        f"{len(want - kept)} missing); rerun scripts/split_m3av.py"
                    )
    except BaseException:
        # Leave no half-written output behind; the previous complete outputs stay.
        if zf is not None:
            zf.close()
            zf = None
        for part in parts:
            part.unlink(missing_ok=True)
        raise
    finally:
        if zf is not None:
            zf.close()
    with open(atomic_path(dc), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "lecture_id", "reason", "n_slides", "detail"])
        for d in dropped:
            w.writerow([d.corpus, d.lecture_id, d.reason, "" if d.n_slides is None else d.n_slides, d.detail])
    os.replace(atomic_path(jl), jl)
    os.replace(atomic_path(dc), dc)
    if has_images:
        os.replace(atomic_path(zp), zp)
    reasons = collections.Counter(d.reason for d in dropped)
    row = {
        "corpus": name,
        "text_source": ";".join(f"{k}={v}" for k, v in sorted(sources.items())),
        "unit_type": ";".join(sorted(unit_types)),
        "ground_truth": gt_source,
        "images": "yes" if has_images else "no",
        "lectures_available": totals["available"],
        "lectures_kept": totals["kept"],
        "lectures_dropped": len(dropped),
        "drop_reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
        "slides_before_cap": totals["slides_before_cap"],
        "slides_after_cap": totals["slides_after_cap"],
        "units_before_cap": totals["units_before_cap"],
        "units_after_cap": totals["units_after_cap"],
        "lectures_cut": totals["lectures_cut"],
        "transcripts_cut": totals["transcripts_cut"],
        "units_without_slide": totals["units_without_slide"],
        "units_empty_text": totals["units_empty_text"],
        "slides_without_text": totals["slides_without_text"],
        "images_written": totals["images"],
        "slides_ocr": totals["slides_ocr"],
        "slides_ocr_empty_text": totals["slides_ocr_empty_text"],
        "ocr_mean_conf": round(conf_sum / conf_n, 2) if conf_n else "",
    }
    log(f"  {json.dumps(row)}")
    log(f"  diagnostics: {json.dumps({k[5:]: v for k, v in sorted(totals.items()) if k.startswith('diag_')})}")
    return row


def check_m3av_disjoint(out: Path) -> None:
    """No lecture may be in both M3AV versions (checked on the written outputs)."""
    ids = {}
    for name in M3AV_VERSIONS:
        path = out / f"{name}.jsonl.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                ids[name] = {json.loads(line)["lecture_id"] for line in f}
    if len(ids) == 2:
        both = ids["m3av_a"] & ids["m3av_b"]
        if both:
            raise PrepareError(f"{len(both)} lectures in both m3av_a and m3av_b, e.g. {sorted(both)[:3]}")
        log(f"  m3av_a and m3av_b are disjoint ({len(ids['m3av_a'])} + {len(ids['m3av_b'])} lectures)")


SUMMARY_FIELDS = [
    "corpus", "text_source", "unit_type", "ground_truth", "images", "lectures_available", "lectures_kept",
    "lectures_dropped", "drop_reasons", "slides_before_cap", "slides_after_cap", "units_before_cap",
    "units_after_cap", "lectures_cut", "transcripts_cut", "units_without_slide", "units_empty_text",
    "slides_without_text", "images_written", "slides_ocr", "slides_ocr_empty_text", "ocr_mean_conf",
]


def write_summary(out: Path, rows: list[dict]) -> None:
    """Merge the rows into summary.csv, keeping rows of corpora not rerun now (rows of
    corpora no longer prepared are not carried over)."""
    path = out / "summary.csv"
    old = {}
    if path.exists():
        with open(path, newline="") as f:
            old = {r["corpus"]: r for r in csv.DictReader(f)}
    old.update({r["corpus"]: r for r in rows})
    with open(atomic_path(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for name in LOADERS:
            if name in old:
                w.writerow({k: old[name].get(k, "") for k in SUMMARY_FIELDS})
    os.replace(atomic_path(path), path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="all", help=f"one of {', '.join(LOADERS)}, or all")
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    key = args.corpus.lower().replace("-", "_")
    if key != "all" and key not in LOADERS:
        ap.error(f"unknown corpus {args.corpus!r}; choose from {', '.join(LOADERS)}, all")
    names = list(LOADERS) if key == "all" else [key]
    src = Sources(args.raw_root, args.ocr_root, args.split)
    rows = []
    for name in names:
        log(f"== {name}")
        rows.append(prepare_corpus(name, src, args.out_root))
    check_m3av_disjoint(args.out_root)
    write_summary(args.out_root, rows)
    log("== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
