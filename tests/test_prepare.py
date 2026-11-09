"""Tests for scripts/prepare.py: the cap rule, the drop rule, the ground-truth mapping of
every corpus, the slide text from our OCR, the two M3AV versions and the record schema.

Everything runs on tiny synthetic archives built here in the corpus's own layout, with
synthetic OCR files in the layout of scripts/ocr.py; the real data is never read.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

import download
import ocr
import prepare as p


# ---------------------------------------------------------------- small builders


def lecture(n_slides: int, labels: list[int | None], corpus: str = "fake", images: bool = False) -> p.Lecture:
    slides = [
        p.Slide(text=f"slide {i}", source_ids=[str(i)], image=b"img" if images else None, image_ext="png" if images else None)
        for i in range(1, n_slides + 1)
    ]
    units = [p.Unit(text=f"unit {i}", source_index=i, true_slide=t, start=float(i)) for i, t in enumerate(labels)]
    return p.Lecture(corpus, "L1", None, "none", "sentence", "lecture", slides, units)


def write_raw(root: Path, corpus: str, archive: str, data: bytes) -> None:
    d = root / corpus
    d.mkdir(parents=True)
    (d / archive).write_bytes(data)
    (d / "SOURCE.json").write_text(json.dumps({"verification": {"archive": archive}}))


def make_tar(files: dict[str, bytes], gz: bool = False) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if gz else "w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def png(seed: int = 0) -> bytes:
    import fitz

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 48), False)
    pix.set_rect(pix.irect, (seed % 256, 100, 200))
    return pix.tobytes("png")


def jpg(shade: int) -> bytes:
    import fitz

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 48), False)
    pix.set_rect(pix.irect, (shade, shade, shade))
    return pix.tobytes("jpg")


def write_ocr(root: Path, name: str, rows: list[tuple[str, str, str, bool]]) -> None:
    """data/ocr/<name>.jsonl.gz and .json as scripts/ocr.py writes them;
    rows: (lecture, image id, text, blank image)."""
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / f"{name}.jsonl.gz", "wt") as f:
        for lec, sid, text, blank in rows:
            n = len(text.split())
            f.write(json.dumps({"lecture": lec, "slide": sid, "text": text, "mean_conf": 90.0 if n else None, "n_words": n, "blank_image": blank, "width": 64, "height": 48}) + "\n")
    record = {"engine": "tesseract", "tesseract_version": "tesseract 5.3.4", "language_data": {"package": "eng"}, "settings": ocr.SETTINGS}
    (root / f"{name}.json").write_text(json.dumps(record))


def src(tmp_path: Path) -> p.Sources:
    return p.Sources(tmp_path / "raw", tmp_path / "ocr", tmp_path / "split.csv")


def read_out(out: Path, corpus: str) -> tuple[list[dict], list[dict]]:
    with gzip.open(out / f"{corpus}.jsonl.gz", "rt") as f:
        recs = [json.loads(line) for line in f]
    with open(out / f"{corpus}_dropped.csv", newline="") as f:
        dropped = list(csv.DictReader(f))
    return recs, dropped


# ------------------------------------------------------------------- the cap rule


def test_cap_cuts_at_first_unit_beyond_40_even_if_lecture_returns():
    labels = [1, 2, None, 40, 41, 3, 2, 42]
    rec, _ = p.build_record(lecture(45, labels))
    assert [u["true_slide"] for u in rec["units"]] == [1, 2, 40]  # the None unit is not kept
    assert [u["source_index"] for u in rec["units"]] == [0, 1, 3]
    assert rec["transcript_cut_at_source_index"] == 4
    assert rec["n_slides_before_cap"] == 45 and rec["n_slides_after_cap"] == 40
    assert rec["n_units_before_cap"] == 7 and rec["n_units_after_cap"] == 3
    assert rec["cut_applied"] is True and rec["n_units_without_slide"] == 1


def test_cap_keeps_whole_transcript_when_no_unit_beyond_40():
    rec, _ = p.build_record(lecture(50, [1, 5, 40, 2]))
    assert rec["transcript_cut_at_source_index"] is None
    assert rec["n_units_after_cap"] == 4 and rec["n_slides_after_cap"] == 40
    assert rec["cut_applied"] is True  # slides 41-50 were removed


def test_short_lecture_is_not_cut():
    rec, _ = p.build_record(lecture(12, [1, 2, 12]))
    assert rec["cut_applied"] is False and rec["n_slides_after_cap"] == 12


def test_cap_point_first_unit_only():
    assert p.cap_point(lecture(50, [None, 41, 1]).units) == 1
    assert p.cap_point(lecture(50, [1, 40]).units) is None


def test_empty_text_units_are_not_kept():
    lec = lecture(6, [1, 2])
    lec.units[1].text = "  "
    rec, _ = p.build_record(lec)
    assert rec["n_units_after_cap"] == 1 and rec["n_units_empty_text"] == 1


# ------------------------------------------------------------------ the drop rule


def test_drop_rule_fewer_than_five_slides():
    assert p.drop_reason(lecture(4, [1])) == "fewer_than_5_slides"
    assert p.drop_reason(lecture(5, [1])) is None
    assert p.drop_reason(lecture(0, [])) == "no_slides"


def test_prepare_corpus_writes_dropped_and_kept(tmp_path, monkeypatch):
    def loader(raw):
        yield lecture(4, [1, 2])
        big = lecture(41, [1, 41])
        big.lecture_id = "L2"
        yield big
        none = lecture(6, [None, None])
        none.lecture_id = "L3"
        yield none
        yield p.Dropped("fake", "L4", "no_ground_truth", None, "detail")

    monkeypatch.setitem(p.LOADERS, "fake", (loader, False, "gt"))
    row = p.prepare_corpus("fake", src(tmp_path), tmp_path / "out")
    recs, dropped = read_out(tmp_path / "out", "fake")
    assert [r["lecture_id"] for r in recs] == ["L2"]
    assert {(d["lecture_id"], d["reason"]) for d in dropped} == {
        ("L1", "fewer_than_5_slides"),
        ("L3", "no_labelled_units"),
        ("L4", "no_ground_truth"),
    }
    assert row["lectures_available"] == 4 and row["lectures_kept"] == 1 and row["lectures_dropped"] == 3
    assert not list((tmp_path / "out").glob("*.part"))


def test_failed_validation_leaves_no_partial_output(tmp_path, monkeypatch):
    def loader(raw):
        lec = lecture(6, [1, 2])
        lec.units[1].source_index = 0  # order broken
        yield lec

    monkeypatch.setitem(p.LOADERS, "fake", (loader, False, "gt"))
    with pytest.raises(p.PrepareError, match="order"):
        p.prepare_corpus("fake", src(tmp_path), tmp_path / "out")
    assert not list((tmp_path / "out").iterdir())


# ---------------------------------------------------------------- record schema


SCHEMA_KEYS = {
    "corpus", "lecture_id", "title", "text_source", "unit_type", "time_base", "n_slides_before_cap",
    "n_slides_after_cap", "n_units_before_cap", "n_units_after_cap", "cut_applied",
    "transcript_cut_at_source_index", "n_units_raw", "n_units_without_slide", "n_units_empty_text",
    "n_slides_with_text", "n_slides_ocr", "n_slides_ocr_empty_text", "n_images", "meta", "diagnostics",
    "slides", "units",
}


def test_record_schema_and_images():
    rec, images = p.build_record(lecture(6, [1, 6], images=True))
    assert set(rec) == SCHEMA_KEYS
    assert set(rec["slides"][0]) == {"index", "text", "image", "source_ids", "ocr"}
    assert set(rec["units"][0]) == {"index", "source_index", "text", "start", "end", "true_slide"}
    assert rec["slides"][0]["image"] == "L1/01.png" and sorted(images) == [f"L1/{i:02d}.png" for i in range(1, 7)]
    p.validate(rec, images, corpus_has_images=True)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda r, i: r["units"][0].update(true_slide=0), "true slide"),
        (lambda r, i: r["units"][0].update(true_slide=7), "true slide"),
        (lambda r, i: r["units"][0].update(true_slide="1"), "true slide"),
        (lambda r, i: r["units"].reverse(), "order"),
        (lambda r, i: r.update(slides=[]), "empty slide list"),
        (lambda r, i: i.pop("L1/02.png"), "images"),
        (lambda r, i: r["slides"][1].update(image=None) or r.update(n_images=5) or i.pop("L1/02.png"), "no image"),
    ],
)
def test_validate_fails_loudly(mutate, msg):
    rec, images = p.build_record(lecture(6, [1, 6], images=True))
    mutate(rec, images)
    with pytest.raises(p.PrepareError, match=msg):
        p.validate(rec, images, corpus_has_images=True)


def test_validate_rejects_more_than_40_slides():
    rec, images = p.build_record(lecture(41, [1]))
    rec["slides"].append({"index": 41, "text": "", "image": None, "source_ids": [], "ocr": None})
    rec["n_slides_after_cap"] = 41
    with pytest.raises(p.PrepareError, match="cap"):
        p.validate(rec, images, corpus_has_images=False)


# ----------------------------------------------------------- shared GT mechanics


def test_identify_slides_merges_repeated_text_only():
    segs = [("a", "Intro to the topic"), ("b", "Methods and data here"), ("c", "intro  to the TOPIC"), ("d", "Results"), ("e", "Results")]
    groups, seg_slide, apart = p.identify_slides(segs)
    assert groups == [[0, 2], [1], [3], [4]]  # 'Results' is too short to merge on
    assert seg_slide == [0, 1, 0, 2, 3] and apart == 0
    groups, seg_slide, apart = p.identify_slides(segs, same_image=lambda i, j: False)
    assert len(groups) == 5 and apart == 1


def test_assign_by_time_majority_tie_and_outside():
    iv = [(0.0, 10.0), (10.0, 20.0), (25.0, 30.0)]
    assert p.assign_by_time(iv, [(8, 9), (11, 12), (12, 13)]) == 1
    assert p.assign_by_time(iv, [(8, 9), (11, 12)]) == 0  # tie -> earlier
    assert p.assign_by_time(iv, [(21, 22)]) is None  # in a gap
    assert p.assign_by_time(iv, [(9.5, 10.5)]) == 1  # midpoint 10.0 is in [10, 20)


def test_srt_parser():
    srt = "1\n00:00:01,210 --> 00:00:06,476\nHello there.\n\n2\n00:01:07,011 --> 00:01:09,672\nSecond\nline.\n"
    assert p.parse_srt(srt) == [(1.21, 6.476, "Hello there."), (67.011, 69.672, "Second line.")]


# --------------------------------------------------------------------- MaViLS


def xlsx(rows: list[tuple]) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    for r in rows:
        wb.active.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def pdf(pages: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for t in pages:
        doc.new_page(width=200, height=150).insert_text((10, 20), t, fontsize=8)
    return doc.tobytes()


def test_mavils_units_labels():
    rows = [("Key", "Slidenumber", "Value"), (1.5, -1, " intro"), (2.0, 2, " two"), (3.0, -2, " odd"), (4.0, 3.0, None)]
    units, diag = p.mavils_units(rows, n_pages=5)
    assert [u.true_slide for u in units] == [None, 2, None, 3]
    assert [u.start for u in units] == [1.5, 2.0, 3.0, 4.0] and units[3].text == ""
    assert diag["gt_labels"] == {"-1": 1, "valid": 2, "-2": 1}
    with pytest.raises(p.PrepareError, match="PDF pages"):
        p.mavils_units([("Time", "Slidenumber", "Value"), (1, 9, "x")], n_pages=5)


def test_mavils_end_to_end(tmp_path, monkeypatch):
    pages = [f"photosynthesis chlorophyll page{i}" for i in range(1, 7)]
    other = ["quantum entanglement superposition"] * 6
    rows = [("Time", "Slidenumber", "Value"), (1.0, -1, " Welcome"), (2.0, 1, " photosynthesis chlorophyll"), (3.0, 6, " page six")]
    files = {
        "data/lectures/bio.pdf": pdf(pages),
        "data/lectures/extra.pdf": pdf(other),
        "data/audioscripts/bio.srt": b"",
        "data/ground_truth_files/ground_truth_bio.xlsx": xlsx(rows),
    }
    write_raw(tmp_path / "raw", "mavils", "m.tar", make_tar(files))
    monkeypatch.setattr(p, "MAVILS_LECTURES", {"bio": ("ground_truth_bio.xlsx", "bio.pdf", "bio.srt")})
    row = p.prepare_corpus("mavils", src(tmp_path), tmp_path / "out")
    recs, dropped = read_out(tmp_path / "out", "mavils")
    (r,) = recs
    assert r["text_source"] == "pdf_text_layer" and r["n_slides_after_cap"] == 6
    assert [u["true_slide"] for u in r["units"]] == [1, 6] and r["n_units_without_slide"] == 1
    assert "photosynthesis" in r["slides"][0]["text"]
    assert [(d["lecture_id"], d["reason"]) for d in dropped] == [("extra", "no_ground_truth")]
    with zipfile.ZipFile(tmp_path / "out" / "mavils_images.zip") as z:
        assert sorted(z.namelist()) == [f"bio/{i:02d}.png" for i in range(1, 7)]
    assert row["images_written"] == 6


def test_mavils_scanned_deck_takes_our_ocr(tmp_path, monkeypatch):
    import fitz

    doc = fitz.open()
    for _ in range(5):
        doc.new_page(width=200, height=150)  # no text layer: a scanned deck
    rows = [("Time", "Slidenumber", "Value"), (1.0, 2, " the second scan")]
    files = {
        "data/lectures/scan.pdf": doc.tobytes(),
        "data/audioscripts/scan.srt": b"",
        "data/ground_truth_files/ground_truth_scan.xlsx": xlsx(rows),
    }
    write_raw(tmp_path / "raw", "mavils", "m.tar", make_tar(files))
    write_ocr(tmp_path / "ocr", "mavils", [("scan", f"scan.pdf#page={k}", f"ocr page {k}", False) for k in range(1, 6)])
    monkeypatch.setattr(p, "MAVILS_LECTURES", {"scan": ("ground_truth_scan.xlsx", "scan.pdf", "scan.srt")})
    p.prepare_corpus("mavils", src(tmp_path), tmp_path / "out")
    (r,), _ = read_out(tmp_path / "out", "mavils")
    assert r["text_source"] == "tesseract_ocr" and r["slides"][1]["text"] == "ocr page 2"
    assert r["meta"]["ocr"]["record"] == "data/ocr/mavils.json" and r["n_slides_ocr"] == 5


def test_mavils_wrong_pdf_pairing_fails(tmp_path, monkeypatch):
    files = {
        "data/lectures/bio.pdf": pdf(["quantum entanglement superposition"] * 5),
        "data/lectures/extra.pdf": pdf(["photosynthesis chlorophyll"] * 5),
        "data/audioscripts/bio.srt": b"",
        "data/ground_truth_files/ground_truth_bio.xlsx": xlsx([("Time", "Slidenumber", "Value"), (1.0, 1, "photosynthesis chlorophyll")]),
    }
    write_raw(tmp_path / "raw", "mavils", "m.tar", make_tar(files))
    monkeypatch.setattr(p, "MAVILS_LECTURES", {"bio": ("ground_truth_bio.xlsx", "bio.pdf", "bio.srt")})
    with pytest.raises(p.PrepareError, match="better"):
        p.prepare_corpus("mavils", src(tmp_path), tmp_path / "out")


# ----------------------------------------------------------------------- M3AV

M3AV_TOP = "./dataset_v1.0_noaudio"


def m3av_chi1(files: dict[str, bytes]) -> None:
    top = f"{M3AV_TOP}/CHI-1"
    texts = ["Title of the talk", "Our method in detail", "Results on benchmarks", "Title of the talk", "Discussion of limits", "Future work items"]
    seg = []
    for i, t in enumerate(texts):
        name = f"CHI-1-{(i + 1) * 10000:07d}.jpg"
        seg.append({"check": "paddleocr+check", "name": name, "ocr_text": t, "speech_text": "", "start": i * 10.0, "end": (i + 1) * 10.0})
        files[f"{top}/ocr/seg_imgs/{name}"] = jpg(50 if i == 3 else i * 40)  # segment 4 repeats slide 1
    speech = [
        {"timestr": "0000500_0001500", "final_written": "Hello.", "words_spoken": [{"start": 0.5, "end": 1.0}]},
        {"timestr": "0009000_0012000", "final_written": "Across.", "words_spoken": [{"start": 9.0, "end": 9.5}, {"start": 10.5, "end": 11.0}, {"start": 11.0, "end": 12.0}]},
        {"timestr": "0031000_0032000", "final_written": "Back to title.", "words_spoken": [{"start": 31.0, "end": 32.0}]},
        {"timestr": "0070000_0071000", "final_written": "After the end.", "words_spoken": [{"start": 70.0, "end": 71.0}]},
    ]
    files[f"{top}/ocr/seg_speech_ocr.json"] = json.dumps(seg).encode()
    files[f"{top}/speech/final+timestamps.json"] = json.dumps(speech).encode()
    files[f"{top}/raw/CHI-1.mp4.ytbUrl"] = b"https://youtu.be/x"


def m3av_simple(files: dict[str, bytes], talk: str, n: int) -> list[str]:
    """A talk of n segments with distinct text and images, one speech unit per segment."""
    top = f"{M3AV_TOP}/{talk}"
    seg, speech, names = [], [], []
    for i in range(n):
        name = f"{talk}-{(i + 1) * 10000:07d}.jpg"
        names.append(name)
        seg.append({"name": name, "ocr_text": f"Corpus text of slide {i}", "start": i * 10.0, "end": (i + 1) * 10.0})
        speech.append({"timestr": f"{i * 10000 + 1000:07d}_{i * 10000 + 2000:07d}", "final_written": f"Unit {i}.", "words_spoken": [{"start": i * 10.0 + 1, "end": i * 10.0 + 2}]})
        files[f"{top}/ocr/seg_imgs/{name}"] = jpg(i * 40)
    files[f"{top}/ocr/seg_speech_ocr.json"] = json.dumps(seg).encode()
    files[f"{top}/speech/final+timestamps.json"] = json.dumps(speech).encode()
    return names


def m3av_setup(tmp_path: Path, split_rows: list[tuple[str, str]]) -> list[str]:
    files: dict[str, bytes] = {}
    m3av_chi1(files)
    names = m3av_simple(files, "CHI-3", 5)
    files[f"{M3AV_TOP}/CHI-2/ocr/seg_speech_ocr.json"] = b"[]"
    files[f"{M3AV_TOP}/CHI-2/speech/final+timestamps.json"] = b"[]"
    write_raw(tmp_path / "raw", "m3av", "d.tar.gz", make_tar(files, gz=True))
    (tmp_path / "split.csv").write_text("lecture_id,version,stratum\n" + "".join(f"{lid},{v},CHI|short\n" for lid, v in split_rows))
    return names


def test_m3av_versions_end_to_end(tmp_path):
    m3av_setup(tmp_path, [("CHI-1", "A"), ("CHI-3", "B")])  # no OCR files: M3AV is not OCRed
    p.prepare_corpus("m3av_a", src(tmp_path), tmp_path / "out")
    p.prepare_corpus("m3av_b", src(tmp_path), tmp_path / "out")
    (a,), dropped_a = read_out(tmp_path / "out", "m3av_a")
    (b,), dropped_b = read_out(tmp_path / "out", "m3av_b")
    # The 4th segment has the text of the 1st, but its image
    # differs, so it stays its own slide.
    assert a["lecture_id"] == "CHI-1" and a["corpus"] == "m3av_a" and a["text_source"] == "corpus_corrected_ocr"
    assert a["n_slides_before_cap"] == 6 and a["diagnostics"]["same_text_kept_apart_by_image"] == 1
    assert [u["true_slide"] for u in a["units"]] == [1, 2, 4]
    assert a["n_units_without_slide"] == 1 and a["diagnostics"]["units_straddling_segments"] == 1
    assert a["units"][0]["start"] == 0.5 and a["meta"]["youtube_url"] == "https://youtu.be/x"
    assert a["slides"][0]["text"] == "Title of the talk" and a["slides"][0]["ocr"] is None
    # B is an independent dataset prepared the same way: the corpus's own slide text.
    assert b["lecture_id"] == "CHI-3" and b["corpus"] == "m3av_b" and b["text_source"] == "corpus_corrected_ocr"
    assert [s["text"] for s in b["slides"]] == [f"Corpus text of slide {i}" for i in range(5)]
    assert b["n_slides_ocr"] == 0 and all(s["ocr"] is None for s in b["slides"]) and "ocr" not in b["meta"]
    assert [u["true_slide"] for u in b["units"]] == [1, 2, 3, 4, 5]
    # The talk the rules drop is in no version and listed in both dropped files.
    assert [(d["lecture_id"], d["reason"]) for d in dropped_a] == [("CHI-2", "no_slides")] == [(d["lecture_id"], d["reason"]) for d in dropped_b]
    p.check_m3av_disjoint(tmp_path / "out")


def test_m3av_kept_lecture_outside_the_split_fails(tmp_path):
    m3av_setup(tmp_path, [("CHI-1", "A")])
    with pytest.raises(p.PrepareError, match="no split version"):
        p.prepare_corpus("m3av_a", src(tmp_path), tmp_path / "out")


def test_m3av_split_lecture_missing_from_the_data_fails(tmp_path):
    m3av_setup(tmp_path, [("CHI-1", "A"), ("CHI-3", "B"), ("CHI-9", "A")])
    with pytest.raises(p.PrepareError, match="differ from the split"):
        p.prepare_corpus("m3av_a", src(tmp_path), tmp_path / "out")


def test_split_file_with_a_lecture_in_both_versions_is_rejected(tmp_path):
    (tmp_path / "split.csv").write_text("lecture_id,version,stratum\nX,A,s\nX,B,s\n")
    with pytest.raises(p.PrepareError, match="twice"):
        p.read_split(tmp_path / "split.csv")


def test_m3av_versions_must_be_disjoint(tmp_path):
    for name in ("m3av_a", "m3av_b"):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt") as f:
            f.write(json.dumps({"lecture_id": "CHI-1"}) + "\n")
    with pytest.raises(p.PrepareError, match="both"):
        p.check_m3av_disjoint(tmp_path)


def test_m3av_same_image_merges_repeat():
    seg = [{"name": f"T-{i}.jpg", "ocr_text": t, "start": i * 10.0, "end": i * 10.0 + 10} for i, t in enumerate(["Alpha beta gamma", "Delta eps zeta", "Alpha beta gamma", "x y z w", "q r s t", "u v w x"])]
    jp = {s["name"]: jpg(100 if i in (0, 2) else 20 * i) for i, s in enumerate(seg)}
    speech = [{"timestr": "0021000_0022000", "final_written": "Again.", "words_spoken": [{"start": 21.0, "end": 22.0}]}]
    lec = p.m3av_lecture("T", seg, speech, jp, None)
    assert len(lec.slides) == 5 and lec.slides[0].source_ids == ["T-0.jpg", "T-2.jpg"]
    assert lec.slides[0].image_id == "T-0.jpg" and lec.units[0].true_slide == 1


# ------------------------------------------------------------------------ MCD


def test_mcd_end_to_end(tmp_path):
    files, ocr_rows = {}, []
    for split, sp, ks in [("test", "sp_0001", [2, 3, 5, 6, 7]), ("eg", "sp_0002", [2, 3, 4])]:
        for k in ks:
            files[f"Dataset/PRES/{split}/input/vid-seg/slides/{sp}/{sp}_{k:03d}.png"] = png(k)
            files[f"Dataset/PRES/{split}/input/vid-seg/trans/{sp}/{sp}_{k:03d}.txt"] = f"Sentence a{k}.\n\nSentence b{k}.\n".encode()
            ocr_rows.append((sp, f"{sp}_{k:03d}.png", f"Slide {k} words", False))
    write_raw(tmp_path / "raw", "mcd", "m.tar", make_tar(files))
    write_ocr(tmp_path / "ocr", "mcd", ocr_rows)
    row = p.prepare_corpus("mcd", src(tmp_path), tmp_path / "out")
    (r,), dropped = read_out(tmp_path / "out", "mcd")
    assert r["lecture_id"] == "sp_0001" and r["meta"]["slide_numbers_in_corpus"] == [2, 3, 5, 6, 7]
    assert [u["true_slide"] for u in r["units"]] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert r["units"][2]["text"] == "Sentence a3." and r["time_base"] is None and r["text_source"] == "tesseract_ocr"
    assert r["slides"][2]["text"] == "Slide 5 words" and r["slides"][2]["ocr"]["mean_conf"] == 90.0
    assert [(d["lecture_id"], d["reason"]) for d in dropped] == [("sp_0002", "fewer_than_5_slides")]
    assert row["slides_ocr"] == 5 and row["slides_ocr_empty_text"] == 0 and row["ocr_mean_conf"] == 90.0


def test_mcd_without_ocr_fails(tmp_path):
    write_raw(tmp_path / "raw", "mcd", "m.tar", make_tar({}))
    with pytest.raises(p.PrepareError, match="ocr.py"):
        p.prepare_corpus("mcd", src(tmp_path), tmp_path / "out")


# ------------------------------------------------------------ Slide-Highlight


def sh_files() -> dict[str, bytes]:
    files = {}
    for k, sid in enumerate(["aaa", "bbb", "ccc", "ddd", "eee"]):
        files[f"Dataset/slides/ICML_slides/vid_0001/PART_{k:04d}_{sid}.png"] = png(k)
        files[f"Dataset/transcripts/ICML_trans/vid_0001/PART_{k:04d}_{sid}.srt"] = (
            f"1\n00:00:00,500 --> 00:00:02,000\nFirst {k}.\n\n2\n00:00:02,100 --> 00:00:04,000\nSecond {k}.\n"
        ).encode()
    files["Dataset/slides/NeurIPS_slides/vid_0054/38992648/aaa.png"] = png(9)  # duplicate folder: ignored
    return files


def test_slide_highlight_end_to_end(tmp_path):
    write_raw(tmp_path / "raw", "slide_highlight", "s.tar", make_tar(sh_files()))
    write_ocr(tmp_path / "ocr", "slide_highlight", [("ICML_vid_0001", f"PART_{k:04d}_{sid}", f"Line {k}", False) for k, sid in enumerate(["aaa", "bbb", "ccc", "ddd", "eee"])])
    p.prepare_corpus("slide_highlight", src(tmp_path), tmp_path / "out")
    (r,), _ = read_out(tmp_path / "out", "slide_highlight")
    assert r["lecture_id"] == "ICML_vid_0001" and r["time_base"] == "slide_segment"
    assert [u["true_slide"] for u in r["units"]] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert r["units"][1] | {} == {"index": 1, "source_index": 1, "text": "Second 0.", "start": 2.1, "end": 4.0, "true_slide": 1}
    assert r["text_source"] == "tesseract_ocr" and r["slides"][2]["text"] == "Line 2"


def test_empty_ocr_text_on_a_non_blank_image_fails(tmp_path):
    write_raw(tmp_path / "raw", "slide_highlight", "s.tar", make_tar(sh_files()))
    rows = [("ICML_vid_0001", f"PART_{k:04d}_{sid}", "" if k == 1 else "x", False) for k, sid in enumerate(["aaa", "bbb", "ccc", "ddd", "eee"])]
    write_ocr(tmp_path / "ocr", "slide_highlight", rows)
    with pytest.raises(p.PrepareError, match="not blank"):
        p.prepare_corpus("slide_highlight", src(tmp_path), tmp_path / "out")


# -------------------------------------------------------------------- summary


def test_summary_merges_rows(tmp_path):
    p.write_summary(tmp_path, [{"corpus": "mcd", "lectures_kept": 1}])
    p.write_summary(tmp_path, [{"corpus": "m3av_a", "lectures_kept": 2}])
    with open(tmp_path / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["corpus"], r["lectures_kept"]) for r in rows] == [("m3av_a", "2"), ("mcd", "1")]


def test_summary_drops_rows_of_corpora_no_longer_prepared(tmp_path):
    (tmp_path / "summary.csv").write_text("corpus,lectures_kept\nlpm,320\nm3av,1083\n")
    p.write_summary(tmp_path, [{"corpus": "mcd", "lectures_kept": 1}])
    with open(tmp_path / "summary.csv", newline="") as f:
        assert [r["corpus"] for r in csv.DictReader(f)] == ["mcd"]


# ------------------------------------------------------------- LPM is gone


def test_lpm_is_removed_from_every_corpus_list():
    assert list(download.CORPORA) == ["mavils", "m3av", "mcd", "slide_highlight"]
    assert list(p.LOADERS) == ["mavils", "m3av_a", "m3av_b", "mcd", "slide_highlight"]
    assert set(ocr.CORPORA) == {"mavils", "mcd", "slide_highlight"}  # M3AV is not OCRed
    assert not [n for n in dir(p) if "lpm" in n.lower()]


# ------------------------------------------------- slides without an image (step 4)


def test_lecture_with_a_kept_slide_without_image_is_dropped(tmp_path, monkeypatch):
    """DECIDED in step 4: such a lecture is dropped with the reason slide_without_image.
    A slide without an image beyond the 40-slide cap does not count."""

    def loader(raw):
        lec = lecture(6, [1, 2], images=True)
        lec.slides[2].image, lec.slides[2].image_ext = None, None
        yield lec
        ok = lecture(6, [1, 2], images=True)
        ok.lecture_id = "L2"
        yield ok
        long = lecture(41, [1, 2], images=True)
        long.lecture_id = "L3"
        long.slides[40].image, long.slides[40].image_ext = None, None
        yield long

    monkeypatch.setitem(p.LOADERS, "fake", (loader, True, "gt"))
    row = p.prepare_corpus("fake", src(tmp_path), tmp_path / "out")
    recs, dropped = read_out(tmp_path / "out", "fake")
    assert [r["lecture_id"] for r in recs] == ["L2", "L3"]
    assert [(d["lecture_id"], d["reason"], d["detail"]) for d in dropped] == [
        ("L1", p.NO_IMAGE, "slides [3] of the kept slides have no image")]
    assert row["drop_reasons"] == f"{p.NO_IMAGE}=1"


def test_m3av_lecture_without_an_image_is_dropped_and_the_split_holds(tmp_path):
    """The M3AV versions: a split lecture with a slide without an image is dropped (it
    stays in the split file)."""
    files: dict[str, bytes] = {}
    m3av_chi1(files)
    names = m3av_simple(files, "CHI-3", 5)
    del files[f"{M3AV_TOP}/CHI-3/ocr/seg_imgs/{names[2]}"]
    write_raw(tmp_path / "raw", "m3av", "d.tar.gz", make_tar(files, gz=True))
    (tmp_path / "split.csv").write_text("lecture_id,version,stratum\nCHI-1,A,CHI|short\nCHI-3,B,CHI|short\n")
    p.prepare_corpus("m3av_b", src(tmp_path), tmp_path / "out")
    recs, dropped = read_out(tmp_path / "out", "m3av_b")
    assert recs == []
    assert ("CHI-3", p.NO_IMAGE) in {(d["lecture_id"], d["reason"]) for d in dropped}
    p.prepare_corpus("m3av_a", src(tmp_path), tmp_path / "out")
    assert [r["lecture_id"] for r in read_out(tmp_path / "out", "m3av_a")[0]] == ["CHI-1"]
