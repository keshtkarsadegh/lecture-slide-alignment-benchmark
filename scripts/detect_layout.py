#!/usr/bin/env python3
"""Step 5: find the pictures and tables on every slide image with a layout-detection model.

The model is Docling's "heron" layout detector (config/attributes.toml, [layout]): an
RT-DETRv2 object detector trained on DocLayNet and further layout data, Apache-2.0, with
17 classes among which `picture` and `table`. Livathinos et al. (2025), "Advanced Layout
Analysis Models for Docling", arXiv:2509.11720, report 0.776 mAP@[.50:.95] on their
canonical DocLayNet test set. It was chosen over DocLayout-YOLO (whose code is AGPL-3.0
and needs its own fork of ultralytics) because it runs with transformers alone, which
the `models` extra already pins.

The weights are pinned to one commit of the Hugging Face repository and every file is
checked by sha256; they live outside the repository in $PROJECT_ROOT/models_lsab/ (the
shared hf_cache is not used). torch and transformers run from one Apptainer image
(cluster/layout.def), not from the virtual environment: the project space is close to
its inode quota and an image is one file.

Three commands:

    fetch   download the pinned weight files (needs the proxy on a compute node)
    detect  run the model on every kept slide image of every prepared corpus (GPU, image)
    sheet   draw a seeded, stratified sample of 100 slides with their boxes into one PDF
            for a check by eye, plus the CSV template filled in by the checker (PyMuPDF,
            the virtual environment)
    sheet_v2  the same 100 slides (validation_100_key.csv, S001-S100 in order) with the
            picture boxes that scripts/attributes.py keeps (blue) and drops (grey,
            dashed, "repeat" or "tiny"): validation_100_v2.pdf

`detect` writes data/attributes/layout_boxes.jsonl.gz, one line per slide:

    {"corpus", "lecture", "slide", "width", "height",
     "boxes": [{"class", "box": [x0, y0, x1, y1], "score"}, ...]}

with every box of every class at or above the threshold (pixels of the full image,
origin top left), and data/attributes/layout_boxes.json, the record: model, revision,
file hashes, threshold, package versions, GPU, counts. M3AV's slide images are video
frames (the first frame of each slide segment, as the corpus gives them); the other
corpora's are the slides themselves (MaViLS: PDF pages rendered at 150 dpi).

Usage:
    python scripts/detect_layout.py fetch
    apptainer exec --nv ... python scripts/detect_layout.py detect     (all corpora, one file)
    python scripts/detect_layout.py sheet
    python scripts/detect_layout.py sheet_v2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

from project_root import project_path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "attributes.toml"
PREPARED = REPO_ROOT / "data" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "attributes"
BOXES = OUT_DIR / "layout_boxes.jsonl.gz"
CORPORA = ("mavils", "m3av_a", "m3av_b", "mcd", "slide_highlight")
VIDEO_FRAME_CORPORA = ("m3av_a", "m3av_b")  # slide images that are video frames


def load_config(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- fetch

def fetch(cfg: dict) -> Path:
    """Download each pinned file once; a file already there must match its sha256."""
    lay = cfg["layout"]
    out = project_path(lay["weights_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, want in lay["files"].items():
        dest = out / name
        if dest.exists():
            got = sha256_file(dest)
            if got != want:
                raise SystemExit(f"{dest}: sha256 {got} != pinned {want}; delete it and rerun")
            print(f"ok      {name}")
            continue
        url = f"https://huggingface.co/{lay['repo']}/resolve/{lay['revision']}/{name}"
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url, timeout=600) as resp, open(part, "wb") as fh:
            while chunk := resp.read(1 << 22):
                fh.write(chunk)
        got = sha256_file(part)
        if got != want:
            part.unlink()
            raise SystemExit(f"{url}: sha256 {got} != pinned {want}")
        part.rename(dest)
        print(f"fetched {name}  {dest.stat().st_size} bytes")
    return out


def verify_weights(cfg: dict) -> Path:
    lay = cfg["layout"]
    out = project_path(lay["weights_dir"])
    for name, want in lay["files"].items():
        p = out / name
        if not p.exists() or sha256_file(p) != want:
            raise SystemExit(f"{p}: missing or not the pinned file; run `fetch` first")
    return out


# --------------------------------------------------------------------------- slides

def iter_slides(corpus: str, prepared: Path = PREPARED) -> Iterator[tuple[str, int, str]]:
    """(lecture_id, slide index, image member) of every kept slide, in file order."""
    with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for s in rec["slides"]:
                if not s.get("image"):
                    raise SystemExit(f"{corpus}/{rec['lecture_id']} slide {s['index']}: no image")
                yield rec["lecture_id"], int(s["index"]), s["image"]


# --------------------------------------------------------------------------- detect

def detect(cfg: dict, corpora: tuple[str, ...] = CORPORA, prepared: Path = PREPARED,
           out_dir: Path = OUT_DIR) -> dict:
    """Run the detector on every kept slide; write the boxes and the record."""
    import numpy as np
    import PIL
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    lay = cfg["layout"]
    wdir = verify_weights(cfg)
    threshold = float(lay["threshold"])
    bs = int(lay["batch_size"])
    # Reproducible numbers: no TF32, no autotuned kernels.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(wdir)
    model = AutoModelForObjectDetection.from_pretrained(wdir).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    for cls in (lay["image_class"], lay["table_class"]):
        if cls not in id2label.values():
            raise SystemExit(f"class {cls!r} not in the model's labels {sorted(id2label.values())}")

    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / (BOXES.name + ".part")
    counts: dict[str, dict] = {}
    t0 = time.time()
    with gzip.open(part, "wt", encoding="utf-8", compresslevel=6) as out:
        for corpus in corpora:
            slides = list(iter_slides(corpus, prepared))
            c = {"slides": len(slides), "boxes": {}, "slides_with": {}, "sizes": {}}
            with zipfile.ZipFile(prepared / f"{corpus}_images.zip") as zf:
                for i in range(0, len(slides), bs):
                    batch = slides[i:i + bs]
                    imgs = []
                    for _, _, member in batch:
                        with Image.open(io.BytesIO(zf.read(member))) as im:
                            imgs.append(im.convert("RGB"))
                    sizes = [(im.height, im.width) for im in imgs]
                    inputs = processor(images=imgs, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    results = processor.post_process_object_detection(
                        outputs, target_sizes=torch.tensor(sizes), threshold=threshold)
                    for (lecture, slide, _), (h, w), res in zip(batch, sizes, results):
                        boxes = []
                        for score, label, box in zip(res["scores"].tolist(),
                                                     res["labels"].tolist(),
                                                     res["boxes"].tolist()):
                            cls = id2label[int(label)]
                            boxes.append({"class": cls,
                                          "box": [round(float(v), 1) for v in box],
                                          "score": round(float(score), 4)})
                            c["boxes"][cls] = c["boxes"].get(cls, 0) + 1
                        for cls in {b["class"] for b in boxes}:
                            c["slides_with"][cls] = c["slides_with"].get(cls, 0) + 1
                        key = f"{w}x{h}"
                        c["sizes"][key] = c["sizes"].get(key, 0) + 1
                        out.write(json.dumps({"corpus": corpus, "lecture": lecture,
                                              "slide": slide, "width": w, "height": h,
                                              "boxes": boxes}) + "\n")
            # Keep only the most common image sizes in the record.
            c["sizes"] = dict(sorted(c["sizes"].items(), key=lambda kv: -kv[1])[:10])
            counts[corpus] = c
            print(f"{corpus}: {c['slides']} slides, boxes {c['boxes']}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    part.rename(out_dir / BOXES.name)

    record = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {"repo": lay["repo"], "revision": lay["revision"], "files": lay["files"],
                  "architecture": type(model).__name__,
                  "image_processor": type(processor).__name__,
                  "labels": id2label, "licence": "Apache-2.0",
                  "paper": "Livathinos et al. 2025, arXiv:2509.11720"},
        "threshold": threshold, "batch_size": bs,
        "image_class": lay["image_class"], "table_class": lay["table_class"],
        "video_frame_corpora": list(VIDEO_FRAME_CORPORA),
        "box_format": "[x0, y0, x1, y1] in pixels of the full image, origin top left",
        "packages": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "pillow": PIL.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "pip_freeze": _pip_freeze(),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seconds": round(time.time() - t0, 1),
        "counts": counts,
    }
    with open(out_dir / "layout_boxes.json", "w") as fh:
        json.dump(record, fh, indent=1)
    return record


def _pip_freeze() -> list[str]:
    """The image's full package list, written at build time (cluster/layout.def)."""
    p = Path("/opt/pip-freeze.txt")
    return p.read_text().split() if p.exists() else []


# --------------------------------------------------------------------------- sheet

def read_boxes(path: Path = BOXES) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def quotas(sizes: dict[str, int], n: int, minimum: int) -> dict[str, int]:
    """Proportional allocation of n over the corpora, at least `minimum` each.

    Corpora whose proportional share falls below the minimum get the minimum; the rest of
    n is shared among the others in proportion to their size (largest remainder, ties by
    corpus order). A corpus never gets more than it has.
    """
    fixed: dict[str, int] = {}
    while True:
        free = [c for c in sizes if c not in fixed]
        budget = n - sum(fixed.values())
        total = sum(sizes[c] for c in free)
        share = {c: budget * sizes[c] / total for c in free} if total else {}
        low = [c for c in free if share[c] < minimum or sizes[c] <= minimum]
        if not low:
            break
        for c in low:
            fixed[c] = min(minimum, sizes[c])
    q = {c: int(math.floor(share[c])) for c in free}
    left = budget - sum(q.values())
    order = sorted(free, key=lambda c: (-(share[c] - q[c]), list(sizes).index(c)))
    for c in order[:left]:
        q[c] += 1
    q.update(fixed)
    return {c: min(q[c], sizes[c]) for c in sizes}


def sample_slides(lines: list[dict], *, n: int, seed: int, minimum: int,
                  table_share: float, table_class: str) -> list[dict]:
    """The validation sample: stratified by corpus, over-sampling slides with a table."""
    import numpy as np

    by_corpus: dict[str, list[dict]] = {}
    for ln in lines:
        by_corpus.setdefault(ln["corpus"], []).append(ln)
    order = [c for c in CORPORA if c in by_corpus] + sorted(set(by_corpus) - set(CORPORA))
    q = quotas({c: len(by_corpus[c]) for c in order}, n, minimum)
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for c in order:
        pool = sorted(by_corpus[c], key=lambda r: (r["lecture"], r["slide"]))
        with_t = [r for r in pool if any(b["class"] == table_class for b in r["boxes"])]
        without = [r for r in pool if not any(b["class"] == table_class for b in r["boxes"])]
        k_t = min(len(with_t), math.ceil(table_share * q[c]))
        k_o = min(len(without), q[c] - k_t)
        k_t = min(len(with_t), q[c] - k_o)  # too few table-free slides: take more tables
        for stratum, src, k in (("table", with_t, k_t), ("no_table", without, k_o)):
            idx = sorted(rng.choice(len(src), size=k, replace=False).tolist()) if k else []
            picked.extend({**src[i], "stratum": stratum} for i in idx)
    picked.sort(key=lambda r: (order.index(r["corpus"]), r["lecture"], r["slide"]))
    return picked


COLOURS = {"picture": (0.10, 0.35, 0.85), "table": (0.85, 0.15, 0.15)}
DROPPED = (0.45, 0.45, 0.45)
PAGE_W, BAND = 842.0, 46.0


def _slide_page(doc, image: bytes, r: dict):
    """A page with the slide image under a text band; returns (page, pixel -> point scale)."""
    import fitz  # PyMuPDF

    pix = fitz.Pixmap(image)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    while pix.width > 1400:  # keeps the PDF small; the boxes are scaled, not redrawn
        pix.shrink(1)
    scale = PAGE_W / r["width"]
    img_h = r["height"] * scale
    page = doc.new_page(width=PAGE_W, height=img_h + BAND)
    page.insert_image(fitz.Rect(0, BAND, PAGE_W, BAND + img_h),
                      stream=pix.tobytes("jpeg", jpg_quality=85))
    return page, scale


def _draw_box(page, box, scale: float, col, label: str, dashes=None) -> None:
    import fitz  # PyMuPDF

    x0, y0, x1, y1 = (v * scale for v in box)
    rect = fitz.Rect(x0, y0 + BAND, x1, y1 + BAND)
    page.draw_rect(rect, color=col, width=2, dashes=dashes)
    tw = fitz.get_text_length(label, fontsize=8) + 4
    ty = max(BAND, rect.y0 - 10)
    page.draw_rect(fitz.Rect(rect.x0, ty, rect.x0 + tw, ty + 10), color=None, fill=col)
    page.insert_text((rect.x0 + 2, ty + 8), label, fontsize=8, color=(1, 1, 1))


def sheet(cfg: dict, prepared: Path = PREPARED, out_dir: Path = OUT_DIR) -> list[dict]:
    """validation_100.pdf (one page per slide), its CSV template and the key."""
    import fitz  # PyMuPDF

    lay, val = cfg["layout"], cfg["validation"]
    classes = (lay["image_class"], lay["table_class"])
    lines = read_boxes(out_dir / BOXES.name)
    picked = sample_slides(lines, n=int(val["n"]), seed=int(val["seed"]),
                           minimum=int(val["min_per_corpus"]),
                           table_share=float(val["table_share"]), table_class=lay["table_class"])
    members = {}
    for c in {r["corpus"] for r in picked}:
        for lecture, slide, member in iter_slides(c, prepared):
            members[(c, lecture, slide)] = member

    doc = fitz.open()
    zips: dict[str, zipfile.ZipFile] = {}
    for k, r in enumerate(picked, 1):
        r["slide_id"] = f"S{k:03d}"
        zf = zips.setdefault(r["corpus"], zipfile.ZipFile(prepared / f"{r['corpus']}_images.zip"))
        page, scale = _slide_page(doc, zf.read(members[(r["corpus"], r["lecture"], r["slide"])]),
                                  r)
        shown = [b for b in r["boxes"] if b["class"] in classes]
        n_pic = sum(b["class"] == classes[0] for b in shown)
        n_tab = sum(b["class"] == classes[1] for b in shown)
        frame = "  (video frame)" if r["corpus"] in VIDEO_FRAME_CORPORA else ""
        page.insert_text((8, 18), f"{r['slide_id']}   {r['corpus']} / {r['lecture']} / slide "
                         f"{r['slide']}   {r['width']}x{r['height']} px{frame}",
                         fontsize=11, color=(0, 0, 0))
        page.insert_text((8, 36), f"blue = {classes[0]} ({n_pic}), red = {classes[1]} ({n_tab}); "
                         f"score >= {lay['threshold']}", fontsize=9, color=(0.3, 0.3, 0.3))
        for b in shown:
            _draw_box(page, b["box"], scale, COLOURS.get(b["class"], (0.2, 0.6, 0.2)),
                      f"{b['class']} {b['score']:.2f}")
    for zf in zips.values():
        zf.close()
    doc.save(out_dir / "validation_100.pdf", garbage=3, deflate=True)

    with open(out_dir / "validation_100.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "user_images_ok", "user_tables_ok", "comment"])
        for r in picked:
            w.writerow([r["slide_id"], "", "", ""])
    with open(out_dir / "validation_100_key.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "corpus", "lecture_id", "slide", "stratum",
                    f"n_{classes[0]}", f"n_{classes[1]}"])
        for r in picked:
            w.writerow([r["slide_id"], r["corpus"], r["lecture"], r["slide"], r["stratum"],
                        sum(b["class"] == classes[0] for b in r["boxes"]),
                        sum(b["class"] == classes[1] for b in r["boxes"])])
    return picked


def sheet_v2(cfg: dict, prepared: Path = PREPARED, out_dir: Path = OUT_DIR) -> list[dict]:
    """validation_100_v2.pdf: the slides of validation_100_key.csv, same ids and order,
    with the picture filter of scripts/attributes.py applied.

    The slides are read from the key, not drawn again, so the check compares the same 100
    slides before and after the filter. The filter needs the whole lecture (the repeat
    rule), so each sampled slide's lecture is filtered as attributes.py filters it: its
    kept slides in prepared order.
    """
    import fitz  # PyMuPDF

    import attributes as at

    lay = cfg["layout"]
    pic, tab = lay["image_class"], lay["table_class"]
    with open(out_dir / "validation_100_key.csv", newline="") as fh:
        key = list(csv.DictReader(fh))
    want = {(r["corpus"], r["lecture_id"]) for r in key}
    lines = {(ln["corpus"], ln["lecture"], ln["slide"]): ln
             for ln in read_boxes(out_dir / BOXES.name) if (ln["corpus"], ln["lecture"]) in want}
    members: dict[tuple[str, str, int], str] = {}
    order: dict[tuple[str, str], list[int]] = {}
    for c in dict.fromkeys(r["corpus"] for r in key):
        for lecture, slide, member in iter_slides(c, prepared):
            if (c, lecture) in want:
                members[(c, lecture, slide)] = member
                order.setdefault((c, lecture), []).append(slide)
    rules = cfg["picture_filter"]
    marks: dict[tuple[str, str, int], list] = {}
    for (c, lecture), slides in order.items():
        lec = [lines[(c, lecture, s)] for s in slides]
        for s, m in zip(slides, at.picture_drops(lec, pic,
                                                 repeat_share=float(rules["repeat_share"]),
                                                 repeat_iou=float(rules["repeat_iou"]),
                                                 tiny_pct=float(rules["tiny_pct"]))):
            marks[(c, lecture, s)] = m

    doc = fitz.open()
    zips: dict[str, zipfile.ZipFile] = {}
    out = []
    for k in key:
        sid = (k["corpus"], k["lecture_id"], int(k["slide"]))
        r = lines[sid]
        zf = zips.setdefault(r["corpus"], zipfile.ZipFile(prepared / f"{r['corpus']}_images.zip"))
        page, scale = _slide_page(doc, zf.read(members[sid]), r)
        boxes = list(zip(r["boxes"], marks[sid]))
        n = {"kept": sum(b["class"] == pic and m is None for b, m in boxes),
             "repeat": sum(m == "repeat" for _, m in boxes),
             "tiny": sum(m == "tiny" for _, m in boxes),
             "table": sum(b["class"] == tab for b, _ in boxes)}
        frame = "  (video frame)" if r["corpus"] in VIDEO_FRAME_CORPORA else ""
        page.insert_text((8, 18), f"{k['slide_id']}   {r['corpus']} / {r['lecture']} / slide "
                         f"{r['slide']}   {r['width']}x{r['height']} px{frame}",
                         fontsize=11, color=(0, 0, 0))
        page.insert_text((8, 36), f"blue = {pic} kept ({n['kept']}), grey dashed = {pic} "
                         f"dropped: repeat ({n['repeat']}), tiny ({n['tiny']}); red = {tab} "
                         f"({n['table']}); score >= {lay['threshold']}",
                         fontsize=9, color=(0.3, 0.3, 0.3))
        # Dropped boxes first, so a kept box drawn over one stays visible.
        for b, m in boxes:
            if m is not None:
                _draw_box(page, b["box"], scale, DROPPED, m, dashes="[5 3] 0")
        for b, m in boxes:
            if b["class"] == pic and m is None:
                _draw_box(page, b["box"], scale, COLOURS[pic], f"{pic} {b['score']:.2f}")
            elif b["class"] == tab:
                _draw_box(page, b["box"], scale, COLOURS[tab], f"{tab} {b['score']:.2f}")
        out.append({"slide_id": k["slide_id"], **n})
    for zf in zips.values():
        zf.close()
    doc.save(out_dir / "validation_100_v2.pdf", garbage=3, deflate=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["fetch", "detect", "sheet", "sheet_v2"])
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.command == "fetch":
        fetch(cfg)
    elif args.command == "detect":
        detect(cfg)
    elif args.command == "sheet_v2":
        pages = sheet_v2(cfg)
        tot = {k: sum(p[k] for p in pages) for k in ("kept", "repeat", "tiny", "table")}
        print(f"validation sheet v2: {len(pages)} slides, picture boxes kept {tot['kept']}, "
              f"dropped repeat {tot['repeat']} / tiny {tot['tiny']}, tables {tot['table']} "
              f"-> {OUT_DIR / 'validation_100_v2.pdf'}")
    else:
        picked = sheet(cfg)
        by = {}
        for r in picked:
            by.setdefault(r["corpus"], {}).setdefault(r["stratum"], 0)
            by[r["corpus"]][r["stratum"]] += 1
        print(f"validation sheet: {len(picked)} slides {by} -> {OUT_DIR / 'validation_100.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
