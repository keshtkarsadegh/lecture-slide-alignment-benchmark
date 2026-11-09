"""Step 15: the importance packs, labels and check sample (scripts/importance_pack.py,
scripts/importance_merge.py, scripts/importance_sample.py).

Nothing here reads the real data: two made-up test lectures (one long enough to be
split into parts) and one tuning lecture in a made-up prepared corpus file, and label
CSVs written by hand, so every count and every kappa can be worked out on paper.
"""

from __future__ import annotations

import csv
import gzip
import json

import pytest

import importance_merge as M
import importance_pack as P
import importance_sample as S


# --------------------------------------------------------------------------- #
# Synthetic prepared data
# --------------------------------------------------------------------------- #

def record(lid: str, n_slides: int, n_units: int, corpus: str = "c") -> dict:
    """Units spread evenly over the slides, in order."""
    slides = [{"index": k, "text": f"Title {k}\n  bullet   one\n\nbullet two"}
              for k in range(1, n_slides + 1)]
    units = [{"index": i, "text": f"unit {i}\n says  something",
              "true_slide": 1 + i * n_slides // n_units} for i in range(n_units)]
    return {"corpus": corpus, "lecture_id": lid, "slides": slides, "units": units}


@pytest.fixture
def data(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    recs = [record("L1", 3, 6), record("TUNE", 2, 4), record("L2", 10, 260)]
    with gzip.open(prepared / "c.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    selection = tmp_path / "selected_lectures.csv"
    selection.write_text("corpus,lecture_id,set\nc,L2,test\nc,TUNE,tuning\nc,L1,test\n",
                         encoding="utf-8")
    return P.load_test_lectures(selection, prepared)


def write_labels(path, rows, header="corpus,lecture_id,unit_index,label"):
    path.write_text("\n".join([header] + [",".join(map(str, r)) for r in rows]) + "\n",
                    encoding="utf-8")


def all_rows(lectures, label=lambda i: i % 3):
    return [(L.corpus, L.lecture_id, i, label(i)) for L in lectures for i, _, _ in L.units]


# --------------------------------------------------------------------------- #
# Packs
# --------------------------------------------------------------------------- #

def test_load_test_lectures_only_the_test_set_in_selection_order(data):
    assert [L.lecture_id for L in data] == ["L2", "L1"]
    assert [L.n_units for L in data] == [260, 6]
    assert data[1].units[0] == (0, 1, "unit 0\n says  something")


def test_record_with_a_gap_in_the_unit_indices_stops():
    r = record("X", 2, 4)
    r["units"][2]["index"] = 7
    with pytest.raises(SystemExit, match="unit indices"):
        P.lecture_from_record(r)


def test_slide_line_joins_lines_and_cuts_at_the_limit():
    assert P.slide_line("Title\n  a   b \n\n c") == "Title / a b / c"
    assert P.slide_line(" \n ") == "(no text)"
    long = P.slide_line("x" * 800)
    assert long == "x" * 700 + " [...]"
    assert P.slide_line("y" * 700) == "y" * 700


def test_split_parts_are_equal_and_cover_every_unit():
    assert P.split_parts(250) == [(0, 250)]
    assert P.split_parts(260) == [(0, 130), (130, 260)]
    assert P.split_parts(1002) == [(0, 201), (201, 402), (402, 602), (602, 802), (802, 1002)]
    assert P.split_parts(0) == [(0, 0)]


def test_small_lecture_is_one_pack_with_its_true_slides(data):
    [(name, text)] = P.pack_files(data[1])
    assert name == "L1.txt"
    assert text.splitlines()[:9] == [
        "lecture: L1", "corpus: c", "n_slides: 3", "n_units: 6", "",
        "SLIDES", "slide 1: Title 1 / bullet one / bullet two",
        "slide 2: Title 2 / bullet one / bullet two",
        "slide 3: Title 3 / bullet one / bullet two"]
    units = text.split("UNITS\n")[1].splitlines()
    assert units[0] == "u0 [slide 1] unit 0 says something"
    assert units[5] == "u5 [slide 3] unit 5 says something"
    assert len(units) == 6


def test_long_lecture_is_split_into_parts_with_their_slides_only(data):
    files = P.pack_files(data[0])
    assert [n for n, _ in files] == ["L2.part1.txt", "L2.part2.txt"]
    seen = []
    for (_, text), (a, b) in zip(files, [(0, 130), (130, 260)]):
        head, units = text.split("UNITS\n")
        assert "n_units: 260" in head
        assert f"units u{a} to u{b - 1} ({b - a} units" in head
        lines = units.splitlines()
        seen += [int(x.split()[0][1:]) for x in lines]
        shown = {int(x.split(":")[0].split()[1]) for x in head.splitlines()
                 if x.startswith("slide ")}
        used = {int(x.split("[slide ")[1].split("]")[0]) for x in lines}
        assert shown == used
    assert seen == list(range(260))


def test_batches_balance_and_keep_a_big_lecture_alone():
    sizes = {("c", "big"): 1000, ("c", "a"): 500, ("c", "b"): 400, ("c", "d"): 300,
             ("c", "e"): 300, ("c", "f"): 200}
    bins = P.make_batches(sizes, 900)
    assert bins[0] == [("c", "big")]
    # 1700 units in ceil(1700/900) = 2 batches: 500+300 | 400+300+200 by largest first
    assert bins[1:] == [[("c", "a"), ("c", "e")], [("c", "b"), ("c", "d"), ("c", "f")]]
    assert sorted(k for b in bins for k in b) == sorted(sizes)


def test_export_writes_packs_manifest_and_batches(data, tmp_path):
    out = tmp_path / "packs"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    batches = P.export(data, out, target=200)
    assert sorted(p.name for p in out.iterdir()) == [
        "BATCHES.md", "L1.txt", "L2.part1.txt", "L2.part2.txt", "MANIFEST.csv"]
    with open(out / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
        man = list(csv.DictReader(fh))
    assert man == [
        {"corpus": "c", "lecture_id": "L2", "n_units": "260", "n_slides": "10", "batch": "b01"},
        {"corpus": "c", "lecture_id": "L1", "n_units": "6", "n_slides": "3", "batch": "b02"}]
    assert batches == {"b01": [("c", "L2", ["L2.part1.txt", "L2.part2.txt"], 260)],
                       "b02": [("c", "L1", ["L1.txt"], 6)]}
    md = (out / "BATCHES.md").read_text(encoding="utf-8")
    assert "| c | L2 | L2.part1.txt, L2.part2.txt | 260 |" in md
    assert "2 lectures, 266 units, 2 batches" in md


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def test_merge_writes_the_annotation(data, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = all_rows(data)
    write_labels(raw / "b01.csv", rows[:100])
    write_labels(raw / "b02.csv", rows[100:])
    guide = tmp_path / "IMPORTANCE.md"
    guide.write_text("guide\n", encoding="utf-8")
    lines, problems = M.merge(raw, data, tmp_path / "out", date="2026-10-14", guideline=guide)
    assert problems == []
    with gzip.open(tmp_path / "out" / "labels.csv.gz", "rt", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 266
    assert got[0] == {"corpus": "c", "lecture_id": "L2", "unit_index": "0",
                      "true_slide": "1", "label": "0"}
    assert got[-1] == {"corpus": "c", "lecture_id": "L1", "unit_index": "5",
                       "true_slide": "3", "label": "2"}
    meta = json.loads((tmp_path / "out" / "labels_meta.json").read_text(encoding="utf-8"))
    assert meta["labeller"] == "Claude Opus 5.5"
    assert meta["date"] == "2026-10-14" and meta["n_units"] == 266
    assert len(meta["raw_files"]) == 2 and meta["guideline_sha256"]
    assert sum(meta["label_counts"].values()) == 266
    assert any("L1" in x and "33.3 % core" in x for x in lines)
    # the same labels give the same bytes
    first = (tmp_path / "out" / "labels.csv.gz").read_bytes()
    M.merge(raw, data, tmp_path / "out", date="2026-10-14", guideline=guide)
    assert (tmp_path / "out" / "labels.csv.gz").read_bytes() == first


@pytest.mark.parametrize("change, expect", [
    (lambda r: r[:-1], "not labelled: ('c', 'L1', 5)"),
    (lambda r: r + [("c", "L1", 5, 2)], "labelled more than once"),
    (lambda r: r + [("c", "L1", 6, 1)], "not a test unit: ('c', 'L1', 6)"),
    (lambda r: r + [("c", "TUNE", 0, 1)], "not a test unit: ('c', 'TUNE', 0)"),
    (lambda r: r[:-1] + [("c", "L1", 5, 3)], "label '3' is not 0, 1 or 2"),
    (lambda r: r[:-1] + [("c", "L1", "five", 1)], "is not an integer"),
])
def test_merge_refuses_bad_labels_and_writes_nothing(data, tmp_path, change, expect):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_labels(raw / "b.csv", change(all_rows(data)))
    _, problems = M.merge(raw, data, tmp_path / "out", date="2026-10-14")
    assert any(expect in p for p in problems), problems
    assert not (tmp_path / "out").exists()


def test_wrong_header_is_a_problem(tmp_path):
    write_labels(tmp_path / "b.csv", [("c", "L1", 0, 1)], header="corpus,lecture,unit,label")
    labels, problems = M.read_labels(tmp_path / "b.csv")
    assert labels == {} and "the header is not" in problems[0]


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

def test_weighted_kappa_by_hand():
    a, b = [0, 0, 1, 2], [0, 1, 1, 2]
    # linear: observed 0.125, expected 0.4375 -> 1 - 2/7
    assert M.weighted_kappa(a, b, 1) == pytest.approx(5 / 7)
    # quadratic: observed 0.0625, expected 0.3125 -> 0.8
    assert M.weighted_kappa(a, b, 2) == pytest.approx(0.8)
    assert M.weighted_kappa([1, 1], [1, 1], 1) is None
    assert M.weighted_kappa([0, 2], [0, 2], 2) == pytest.approx(1.0)


def test_agreement_per_lecture_and_overall(tmp_path):
    write_labels(tmp_path / "a.csv", [("c", "L1", 0, 0), ("c", "L1", 1, 0), ("c", "L1", 2, 1),
                                      ("c", "L1", 3, 2), ("c", "L9", 0, 1)])
    b = tmp_path / "b"
    b.mkdir()
    write_labels(b / "x.csv", [("c", "L1", 0, 0), ("c", "L1", 1, 1), ("c", "L1", 2, 1),
                               ("c", "L1", 3, 0)])
    rows, problems = M.agreement(M.read_labels(tmp_path / "a.csv")[0], M.read_labels(b)[0])
    assert problems == []
    assert [r["lecture_id"] for r in rows] == ["L1", ""]
    r = rows[0]
    assert r["n_units"] == 4 and r["exact_pct"] == 50 and r["within_one_pct"] == 75
    assert rows[1]["corpus"] == "ALL" and rows[1]["exact_pct"] == 50
    M.write_agreement(rows, tmp_path / "agreement.csv")
    with open(tmp_path / "agreement.csv", newline="", encoding="utf-8") as fh:
        assert [x["corpus"] for x in csv.DictReader(fh)] == ["c", "ALL"]


def test_agreement_refuses_different_units(tmp_path):
    rows, problems = M.agreement({("c", "L1", 0): 1, ("c", "L1", 1): 1}, {("c", "L1", 0): 1})
    assert rows == [] and "different units" in problems[0]


# --------------------------------------------------------------------------- #
# The blind check sample (scripts/importance_sample.py)
# --------------------------------------------------------------------------- #

def test_draw_is_by_lecture_first_distinct_and_repeatable(data):
    units = [(L.corpus, L.lecture_id, i) for L in data for i, _, _ in L.units]
    sample = S.draw(units, 10, seed=20261012)
    assert len(set(sample)) == 10 and set(sample) <= set(units)
    assert S.draw(units, 10, seed=20261012) == sample
    # L1 holds 6 of 266 units; drawn by units it would get none of 10, by lecture about half
    assert sum(1 for u in sample if u[1] == "L1") >= 3
    # every unit of the lectures can be drawn: all 266 come out
    assert sorted(S.draw(units, 266, seed=1)) == sorted(units)
    with pytest.raises(SystemExit, match="cannot draw"):
        S.draw(units, 267)


def test_sample_markdown_has_no_label_and_the_key_has_them(data, tmp_path):
    labels = {(L.corpus, L.lecture_id, i): 2 if i == 0 else 1 for L in data
              for i, _, _ in L.units}
    data[1].slides[0] = (1, "s" * 500)
    sample = [("c", "L1", 0), ("c", "L2", 130)]
    gz = tmp_path / "labels.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write("corpus,lecture_id,unit_index,true_slide,label\n")
        fh.writelines(f"{c},{l},{i},1,{x}\n" for (c, l, i), x in labels.items())
    assert S.read_labels(gz) == labels
    S.write_sample(sample, labels, data, tmp_path / "out", seed=7)
    md = (tmp_path / "out" / "local" / "check_sample.md").read_text(encoding="utf-8")
    one, two = md.split("## ")[1:]
    assert one.splitlines()[0] == "1. c / L1"
    # the true slide cut at 400 characters; u0 has no units before it
    assert f"**True slide 1:** {'s' * 400} [...]" in one
    assert [x for x in one.splitlines() if x.startswith("- ")] == [
        "- **u0 [slide 1] unit 0 says something**",
        "- u1 [slide 1] unit 1 says something",
        "- u2 [slide 2] unit 2 says something"]
    assert [x.split("]")[0] for x in two.splitlines() if x.startswith("- ")] == [
        "- u128 [slide 5", "- u129 [slide 5", "- **u130 [slide 6",
        "- u131 [slide 6", "- u132 [slide 6"]
    assert md.split("\n\n")[2].startswith("## 1. c / L1")
    assert "label" not in md.split("\n\n", 2)[2]
    assert "seed 7" in md
    with open(tmp_path / "out" / "check_sample_key.csv", newline="", encoding="utf-8") as fh:
        key = list(csv.DictReader(fh))
    assert key == [
        {"item": "1", "corpus": "c", "lecture_id": "L1", "unit_index": "0", "label": "2"},
        {"item": "2", "corpus": "c", "lecture_id": "L2", "unit_index": "130", "label": "1"}]


def test_check_compares_by_unit_with_check_rows_and_agent_columns(tmp_path, capsys):
    key, manual = tmp_path / "key.csv", tmp_path / "manual.csv"
    key.write_text("item,corpus,lecture_id,unit_index,label\n"
                   "1,c,L1,0,2\n2,c,L1,1,1\n3,c,L2,4,0\n4,c,L2,5,1\n", encoding="utf-8")
    # another order: matched by unit, not by row; one 0-vs-2 and one 2-vs-1 disagreement
    manual.write_text("item,corpus,lecture_id,unit_index,label_check\n"
                      "4,c,L2,5,2\n3,c,L2,4,2\n2,c,L1,1,1\n1,c,L1,0,2\n", encoding="utf-8")
    n, exact, within, conf = S.compare(key, manual)
    assert (n, exact, within) == (4, 2, 3)
    assert conf == [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
    assert S.main(["--check", str(key), str(manual)]) == 0
    assert "n 4  exact 50%  within one 75%" in capsys.readouterr().out
    manual.write_text("item,corpus,lecture_id,unit_index,label_check\n1,c,L1,0,2\n",
                      encoding="utf-8")
    with pytest.raises(SystemExit, match="different units"):
        S.compare(key, manual)
