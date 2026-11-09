#!/usr/bin/env python3
"""Step 2b: OCR the slide images that need it, with a fixed Tesseract.

Which images (by decision), by OCR corpus name:

    mavils            the MaViLS deck without a PDF text layer (the scanned
                      image_processing.pdf), its pages rendered at 150 dpi exactly as
                      scripts/prepare.py renders them for the image zip
    mcd               every MCD slide image (the corpus distributes no slide text)
    slide_highlight   every Slide-Highlight slide image (likewise)

M3AV is not OCRed: both of its datasets, m3av_a and m3av_b, use the corpus's own
hand-corrected slide OCR.

The engine is Tesseract 5.3.4 from the Apptainer image built from cluster/tesseract.def
(Ubuntu packages pinned to exact versions, English model tesseract-ocr-eng 1:4.1.0-2,
i.e. tessdata_fast 4.1.0). The settings are fixed and few: --oem 1 (the LSTM engine),
--psm 3 (fully automatic page segmentation, Tesseract's default), -l eng, TSV output,
one thread per process. Each image goes through `tesseract stdin stdout`, so nothing is
written per image.

From the TSV the slide text is rebuilt line by line (words of one (block, paragraph,
line) joined by spaces, lines by newlines) and mean_conf is the mean of the word
confidences (0-100) of the words kept; it is null when no word is found. Every image
also gets a blank_image flag, so that prepare.py can tell "nothing to read" from "OCR
found nothing": blank means at most BLANK_MAX_FRACTION of the pixels differ from the
most common grey level by more than BLANK_GREY_DIFF.

Output in data/ocr/ (not in git), per OCR corpus name:

    <name>.jsonl.gz   one line per image: lecture, slide (the image id prepare.py looks
                      up), text, mean_conf, n_words, blank_image, width, height
    <name>.json       the record: engine, version, packages, language data and its
                      sha256, settings, image file and its sha256, inputs, counts

Usage (compute node, with `module load apptainer/1.4.3`):
    python scripts/ocr.py --corpus mcd [--workers 32] [--image .../tesseract.sif]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import prepare
from project_root import project_path

REPO_ROOT = prepare.REPO_ROOT
DEFAULT_OUT = prepare.DEFAULT_OCR
# Built by cluster/00_tesseract_image.sbatch; under PROJECT_ROOT (scripts/project_root.py).
DEFAULT_IMAGE = project_path("envs/lsab-tesseract/tesseract.sif")

# The fixed Tesseract settings (see the module docstring).
SETTINGS = {"oem": 1, "psm": 3, "lang": "eng", "output": "tsv", "threads_per_process": 1}
TESS_ARGS = ["stdin", "stdout", "--oem", str(SETTINGS["oem"]), "--psm", str(SETTINGS["psm"]), "-l", SETTINGS["lang"], "tsv"]
BLANK_GREY_DIFF = 32  # grey levels (0-255) from the most common one
BLANK_MAX_FRACTION = 0.001  # share of pixels allowed that far off for a blank image
LANGUAGE_DATA = {
    "file": "/usr/share/tesseract-ocr/5/tessdata/eng.traineddata",
    "package": "tesseract-ocr-eng 1:4.1.0-2 (Ubuntu noble)",
    "source": "tessdata_fast 4.1.0, https://github.com/tesseract-ocr/tessdata_fast",
}

ENGINE_CMD: list[str] = []  # set per process by init_worker


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- the engine


def engine_cmd(image: Path) -> list[str]:
    # --containall: no host environment, home or /tmp inside; the image is the whole engine.
    return ["apptainer", "exec", "--containall", str(image)]


def init_worker(cmd: list[str]) -> None:
    global ENGINE_CMD
    ENGINE_CMD = cmd


def run_tesseract(data: bytes) -> str:
    """Tesseract's TSV for one image, given as bytes on stdin."""
    r = subprocess.run([*ENGINE_CMD, "tesseract", *TESS_ARGS], input=data, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"tesseract exited {r.returncode}: {r.stderr.decode(errors='replace')[-500:]}")
    return r.stdout.decode("utf-8")


def parse_tsv(tsv: str) -> tuple[str, float | None, int]:
    """(text, mean word confidence, number of words) from Tesseract's TSV output."""
    lines: dict[tuple[str, ...], list[str]] = {}
    confs: list[float] = []
    for row in tsv.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        word, conf = parts[11].strip(), float(parts[10])
        if not word or conf < 0:
            continue
        lines.setdefault(tuple(parts[1:5]), []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for ws in lines.values())
    return text, (round(sum(confs) / len(confs), 3) if confs else None), len(confs)


def image_stats(data: bytes) -> tuple[bool, int, int]:
    """(blank, width, height) of an image."""
    import fitz

    pix = fitz.Pixmap(data)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 1:
        pix = fitz.Pixmap(fitz.csGRAY, pix)
    hist = collections.Counter(pix.samples)
    mode = hist.most_common(1)[0][0]
    off = sum(c for v, c in hist.items() if abs(v - mode) > BLANK_GREY_DIFF)
    return off <= BLANK_MAX_FRACTION * pix.width * pix.height, pix.width, pix.height


def ocr_image(job: tuple[str, str, bytes]) -> dict:
    lecture, slide, data = job
    blank, w, h = image_stats(data)
    text, conf, n = parse_tsv(run_tesseract(data))
    return {"lecture": lecture, "slide": slide, "text": text, "mean_conf": conf, "n_words": n, "blank_image": blank, "width": w, "height": h}


def engine_info(image: Path) -> dict:
    """What the image holds: tesseract --version, the pinned packages, the model hash."""
    cmd = engine_cmd(image)

    def out(*args: str) -> str:
        return subprocess.run([*cmd, *args], capture_output=True, check=True, text=True).stdout

    version = out("tesseract", "--version")
    pkgs = dict(line.split("\t", 1) for line in out("cat", "/opt/dpkg-versions.tsv").splitlines() if "\t" in line)
    wanted = ("tesseract-ocr", "libtesseract5", "tesseract-ocr-eng", "tesseract-ocr-osd", "liblept5")
    return {
        "tesseract_version": version.splitlines()[0].strip(),
        "tesseract_version_full": version.strip(),
        "packages": {k: pkgs.get(k) for k in wanted},
        "eng_traineddata_sha256": out("cat", "/opt/eng.traineddata.sha256").split()[0],
        "image": str(image),
        "image_sha256": sha256_file(image),
        "image_definition": "cluster/tesseract.def",
    }


# ---------------------------------------------------------------- what to OCR


def jobs_mavils(src: prepare.Sources) -> Iterator[tuple[str, str, bytes]]:
    """Every page of every MaViLS deck that has no text layer, as prepare.py renders it."""
    import fitz

    by_name = prepare.mavils_files(src.raw)
    for lid, (_, pdf, _) in sorted(prepare.MAVILS_LECTURES.items()):
        with fitz.open(stream=by_name[pdf], filetype="pdf") as doc:
            if any(p.get_text().strip() for p in doc):
                continue
            log(f"  {pdf}: no text layer, {doc.page_count} pages")
            for k, page in enumerate(doc):
                yield lid, prepare.mavils_page_id(pdf, k), prepare.render_page(page)


def jobs_mcd(src: prepare.Sources) -> Iterator[tuple[str, str, bytes]]:
    files = prepare.read_tar(prepare.one_archive(src.raw, "mcd"), lambda n: bool(prepare.MCD_RX.match(n)) and n.endswith(".png"))
    for n in sorted(files):
        _, _, sp, k, _ = prepare.MCD_RX.match(n).groups()
        yield sp, prepare.mcd_slide_id(sp, int(k)), files[n]


def jobs_slide_highlight(src: prepare.Sources) -> Iterator[tuple[str, str, bytes]]:
    files = prepare.read_tar(prepare.one_archive(src.raw, "slide_highlight"), prepare.sh_wanted)
    for n in sorted(files):
        m = prepare.SH_RX.match(n)
        if m[7] == "png":
            yield prepare.sh_lecture_id(m[2], m[4]), prepare.sh_slide_id(int(m[5]), m[6]), files[n]


CORPORA = {
    # OCR corpus name: (image source, raw corpus it reads)
    "mavils": (jobs_mavils, "mavils"),
    "mcd": (jobs_mcd, "mcd"),
    "slide_highlight": (jobs_slide_highlight, "slide_highlight"),
}


# ------------------------------------------------------------------------ run


def ocr_corpus(name: str, src: prepare.Sources, out: Path, image: Path, workers: int) -> dict:
    jobs_fn, raw_name = CORPORA[name]
    out.mkdir(parents=True, exist_ok=True)
    jl, rec = out / f"{name}.jsonl.gz", out / f"{name}.json"
    info = engine_info(image)
    log(f"  {info['tesseract_version']}, eng.traineddata {info['eng_traineddata_sha256'][:12]}, {workers} workers")
    t0 = time.time()
    cmd = engine_cmd(image)
    if workers == 1:
        init_worker(cmd)
        results = [ocr_image(j) for j in jobs_fn(src)]
    else:
        with multiprocessing.get_context("fork").Pool(workers, initializer=init_worker, initargs=(cmd,)) as pool:
            results = []
            for r in pool.imap_unordered(ocr_image, jobs_fn(src), chunksize=4):
                results.append(r)
                if len(results) % 500 == 0:
                    log(f"  {len(results)} images")
    results.sort(key=lambda r: (r["lecture"], r["slide"]))
    keys = [(r["lecture"], r["slide"]) for r in results]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"{name}: an image id appears twice")
    if not results:
        raise RuntimeError(f"{name}: no image to OCR")
    part = jl.with_name(jl.name + ".part")
    with open(part, "wb") as raw_f, gzip.GzipFile(filename="", fileobj=raw_f, mode="wb", mtime=0) as gz:
        for r in results:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(part, jl)
    empty = [r for r in results if not r["text"].strip()]
    confs = [r["mean_conf"] for r in results if r["mean_conf"] is not None]
    counts = {
        "lectures": len({r["lecture"] for r in results}),
        "images": len(results),
        "images_blank": sum(r["blank_image"] for r in results),
        "empty_text": len(empty),
        "empty_text_blank_image": sum(r["blank_image"] for r in empty),
        "empty_text_nonblank_image": sum(not r["blank_image"] for r in empty),
        "mean_conf_over_images": round(sum(confs) / len(confs), 2) if confs else None,
        "words": sum(r["n_words"] for r in results),
    }
    raw_src = json.loads((src.raw / raw_name / "SOURCE.json").read_text())
    inputs = {"corpus": raw_name, "archive": raw_src["verification"]["archive"], "archive_sha256": raw_src["files"][0]["sha256"]}
    if name == "mavils":
        inputs["render_dpi"] = prepare.PDF_DPI
    record = {
        "name": name,
        "engine": "tesseract",
        **info,
        "language_data": {**LANGUAGE_DATA, "sha256": info["eng_traineddata_sha256"]},
        "settings": SETTINGS,
        "command": [*cmd, "tesseract", *TESS_ARGS],
        "text": "words of one (block, paragraph, line) joined by spaces, lines by newlines; "
        "words with empty text or confidence -1 are left out",
        "mean_conf": "mean of the kept words' confidences (0-100); null when no word",
        "blank_image": f"at most {BLANK_MAX_FRACTION} of the pixels differ from the most common grey level by more than {BLANK_GREY_DIFF}",
        "inputs": inputs,
        "counts": counts,
        "empty_text_nonblank_images": [f"{r['lecture']}/{r['slide']}" for r in empty if not r["blank_image"]],
        "seconds": round(time.time() - t0, 1),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/ocr.py",
    }
    part = rec.with_name(rec.name + ".part")
    part.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, rec)
    log(f"  {json.dumps(counts)}")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help=f"one of {', '.join(CORPORA)}, or all")
    ap.add_argument("--raw-root", type=Path, default=prepare.DEFAULT_RAW)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="the Tesseract Apptainer image")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = ap.parse_args(argv)
    key = args.corpus.lower().replace("-", "_")
    if key != "all" and key not in CORPORA:
        ap.error(f"unknown corpus {args.corpus!r}; choose from {', '.join(CORPORA)}, all")
    if shutil.which("apptainer") is None:
        ap.error("apptainer not found: module load apptainer/1.4.3")
    if not args.image.is_file():
        ap.error(f"{args.image} missing: sbatch cluster/00_tesseract_image.sbatch")
    src = prepare.Sources(args.raw_root, args.out_root)
    for name in list(CORPORA) if key == "all" else [key]:
        log(f"== {name}")
        ocr_corpus(name, src, args.out_root, args.image, max(1, args.workers))
    log("== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
