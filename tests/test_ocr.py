"""Tests for scripts/ocr.py: which images are OCRed, the TSV parsing, the blank-image
flag, and the output schema and engine record.

Tesseract itself is never run: run_tesseract and engine_info are replaced by fakes, and
the archives are tiny synthetic ones in each corpus's own layout.
"""

from __future__ import annotations

import gzip
import json

import pytest

import ocr
import prepare as p
from test_prepare import jpg, make_tar, pdf, png, write_raw, xlsx

TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t640\t480\t-1\t\n"
    "4\t1\t1\t1\t1\t0\t10\t10\t200\t20\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t10\t10\t50\t20\t90.5\tHello\n"
    "5\t1\t1\t1\t1\t2\t70\t10\t50\t20\t80.5\tworld\n"
    "5\t1\t1\t1\t1\t3\t130\t10\t50\t20\t95.0\t \n"
    "5\t1\t1\t1\t2\t1\t10\t40\t50\t20\t70.0\tsecond\n"
    "5\t1\t2\t1\t1\t1\t10\t90\t50\t20\t-1\tghost\n"
)
OUT_KEYS = {"lecture", "slide", "text", "mean_conf", "n_words", "blank_image", "width", "height"}


def fake_engine(monkeypatch, tsv: str = TSV) -> list[bytes]:
    seen: list[bytes] = []

    def run(data: bytes) -> str:
        seen.append(data)
        return tsv

    info = {
        "tesseract_version": "tesseract 5.3.4",
        "tesseract_version_full": "tesseract 5.3.4\n leptonica-1.82.0",
        "packages": {"tesseract-ocr": "5.3.4-1build5"},
        "eng_traineddata_sha256": "ab" * 32,
        "image": "/x/tesseract.sif",
        "image_sha256": "cd" * 32,
        "image_definition": "cluster/tesseract.def",
    }
    monkeypatch.setattr(ocr, "run_tesseract", run)
    monkeypatch.setattr(ocr, "engine_info", lambda image: info)
    return seen


def read_jsonl(path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def test_parse_tsv_lines_confidence_and_skipped_words():
    text, conf, n = ocr.parse_tsv(TSV)
    assert text == "Hello world\nsecond"  # blank word and conf -1 word left out
    assert n == 3 and conf == pytest.approx((90.5 + 80.5 + 70.0) / 3, abs=1e-3)
    assert ocr.parse_tsv(TSV.splitlines()[0] + "\n") == ("", None, 0)


def test_fixed_settings_in_the_command():
    assert ocr.TESS_ARGS == ["stdin", "stdout", "--oem", "1", "--psm", "3", "-l", "eng", "tsv"]
    assert ocr.engine_cmd(ocr.DEFAULT_IMAGE)[:3] == ["apptainer", "exec", "--containall"]


def test_blank_image_flag():
    import fitz

    assert ocr.image_stats(png(3)) == (True, 64, 48)  # one flat colour
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 48), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(fitz.IRect(5, 5, 30, 12), (0, 0, 0))  # a line of "text"
    assert ocr.image_stats(pix.tobytes("png"))[0] is False
    assert ocr.image_stats(jpg(128))[0] is True


def test_ocr_corpus_mcd_schema_and_record(tmp_path, monkeypatch):
    files = {}
    for k in (2, 3):
        files[f"Dataset/PRES/test/input/vid-seg/slides/sp_0001/sp_0001_{k:03d}.png"] = png(k)
        files[f"Dataset/PRES/test/input/vid-seg/trans/sp_0001/sp_0001_{k:03d}.txt"] = b"x\n"
    write_raw(tmp_path / "raw", "mcd", "m.tar", make_tar(files))
    (tmp_path / "raw" / "mcd" / "SOURCE.json").write_text(json.dumps({"verification": {"archive": "m.tar"}, "files": [{"sha256": "00"}]}))
    seen = fake_engine(monkeypatch)
    rec = ocr.ocr_corpus("mcd", p.Sources(tmp_path / "raw"), tmp_path / "ocr", tmp_path / "t.sif", workers=1)
    rows = read_jsonl(tmp_path / "ocr" / "mcd.jsonl.gz")
    assert len(seen) == 2 and all(set(r) == OUT_KEYS for r in rows)
    assert [(r["lecture"], r["slide"]) for r in rows] == [("sp_0001", "sp_0001_002.png"), ("sp_0001", "sp_0001_003.png")]
    assert rows[0]["text"] == "Hello world\nsecond" and rows[0]["n_words"] == 3 and rows[0]["blank_image"] is True
    on_disk = json.loads((tmp_path / "ocr" / "mcd.json").read_text())
    assert on_disk == rec
    assert rec["engine"] == "tesseract" and rec["tesseract_version"] == "tesseract 5.3.4"
    assert rec["settings"] == {"oem": 1, "psm": 3, "lang": "eng", "output": "tsv", "threads_per_process": 1}
    assert rec["language_data"]["sha256"] == "ab" * 32 and "tessdata_fast 4.1.0" in rec["language_data"]["source"]
    assert rec["command"][-9:] == ocr.TESS_ARGS and rec["counts"]["images"] == 2
    # The OCR output is what prepare.py reads.
    hits, record = p.load_ocr(p.Sources(tmp_path / "raw", tmp_path / "ocr"), "mcd")
    assert set(hits) == {("sp_0001", "sp_0001_002.png"), ("sp_0001", "sp_0001_003.png")}
    assert p.ocr_meta("mcd", record)["version"] == "tesseract 5.3.4"


def test_mavils_ocrs_only_decks_without_text_layer_from_the_same_renders(tmp_path, monkeypatch):
    import fitz

    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=200, height=150)
    scan = doc.tobytes()
    files = {
        "data/lectures/scan.pdf": scan,
        "data/lectures/text.pdf": pdf(["has a text layer"] * 3),
        "data/audioscripts/a.srt": b"",
        "data/ground_truth_files/g.xlsx": xlsx([("Time", "Slidenumber", "Value")]),
    }
    write_raw(tmp_path / "raw", "mavils", "m.tar", make_tar(files))
    monkeypatch.setattr(p, "MAVILS_LECTURES", {"scan": ("g.xlsx", "scan.pdf", "a.srt"), "text": ("g.xlsx", "text.pdf", "a.srt")})
    jobs = list(ocr.jobs_mavils(p.Sources(tmp_path / "raw")))
    assert [(lec, sid) for lec, sid, _ in jobs] == [("scan", f"scan.pdf#page={k}") for k in (1, 2, 3)]
    with fitz.open(stream=scan, filetype="pdf") as d:
        assert jobs[0][2] == p.render_page(d[0])  # the very bytes prepare.py puts in the zip


def test_slide_highlight_ids_match_prepare(tmp_path):
    files = {
        "Dataset/slides/ICML_slides/vid_0001/PART_0000_aaa.png": png(1),
        "Dataset/transcripts/ICML_trans/vid_0001/PART_0000_aaa.srt": b"",
        "Dataset/slides/NeurIPS_slides/vid_0054/38992648/aaa.png": png(2),
    }
    write_raw(tmp_path / "raw", "slide_highlight", "s.tar", make_tar(files))
    jobs = list(ocr.jobs_slide_highlight(p.Sources(tmp_path / "raw")))
    assert [(lec, sid) for lec, sid, _ in jobs] == [("ICML_vid_0001", "PART_0000_aaa")]
