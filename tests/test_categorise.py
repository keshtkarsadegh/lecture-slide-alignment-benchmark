"""Tests for step 6: the tercile cut points and their edges, tables none/some, the
discipline areas and the merge into "other", the level counts and determinism.

Every input is a tiny synthetic attributes table; the FOS list is the repository's.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import categorise as ct

REPO = Path(__file__).resolve().parent.parent
LEVELS = ("low", "medium", "high")


@pytest.fixture(scope="module")
def cfg():
    with open(REPO / "config" / "categories.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def fos(cfg):
    return ct.read_fos(REPO / cfg["discipline"]["fos_list"])


def _row(i, corpus="mavils", source="pdf_text_layer", n_slides=10, image_pct=1.0,
         table_pct=0.0, noise=5.0, text_density=30.0, discipline="Mathematics"):
    return {"corpus": corpus, "lecture_id": f"L{i:03d}", "text_source": source,
            "n_slides": str(n_slides), "n_units": "50", "n_transcript_words": "900",
            "image_pct": str(image_pct), "table_pct": str(table_pct), "noise": str(noise),
            "text_density": str(text_density), "discipline": discipline}


# --------------------------------------------------------------------------- terciles

def test_tercile_cuts_are_the_values_at_one_and_two_thirds():
    assert ct.tercile_cuts(list(range(1, 10))) == (3, 6)
    # n = 10: positions ceil(10/3) = 4 and ceil(20/3) = 7 (1-based).
    assert ct.tercile_cuts([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == (4, 7)


def test_tercile_cuts_do_not_depend_on_the_input_order():
    vals = [3.5, 0.0, 12.25, 7.0, 7.0, 1.5, 40.0]
    assert ct.tercile_cuts(vals) == ct.tercile_cuts(sorted(vals, reverse=True))


def test_tercile_cuts_reject_no_values():
    with pytest.raises(ValueError):
        ct.tercile_cuts([])


def test_tercile_level_edges():
    cuts = (3.0, 6.0)
    assert ct.tercile_level(2.9, cuts, LEVELS) == "low"
    assert ct.tercile_level(3.0, cuts, LEVELS) == "low"          # value == q1
    assert ct.tercile_level(3.0001, cuts, LEVELS) == "medium"
    assert ct.tercile_level(6.0, cuts, LEVELS) == "medium"       # value == q2
    assert ct.tercile_level(6.0001, cuts, LEVELS) == "high"


def test_ties_at_a_cut_point_all_go_low(cfg, fos):
    # Discrete values with a heavy tie at the top, as n_slides has at 40.
    slides = [5, 10, 40, 40, 40, 40, 40, 40, 40]
    rows = [_row(i, n_slides=s) for i, s in enumerate(slides)]
    cats, rec = ct.categorise(rows, cfg, fos)
    assert (rec["cut_points"]["length"]["q1"], rec["cut_points"]["length"]["q2"]) == (40, 40)
    assert [c["length"] for c in cats] == ["short"] * 9


def test_categorise_tercile_axes(cfg, fos):
    rows = [_row(i, n_slides=i + 1, image_pct=float(i), noise=float(10 - i),
                 text_density=float(i % 3)) for i in range(9)]
    cats, rec = ct.categorise(rows, cfg, fos)
    assert [c["length"] for c in cats] == ["short"] * 3 + ["medium"] * 3 + ["long"] * 3
    assert [c["images"] for c in cats] == ["low"] * 3 + ["medium"] * 3 + ["high"] * 3
    assert [c["noise"] for c in cats] == ["high"] * 3 + ["medium"] * 3 + ["low"] * 3
    # text_density 0,1,2 repeated: sorted 0,0,0,1,1,1,2,2,2 -> q1 0, q2 1.
    assert [c["text_density"] for c in cats] == ["low", "medium", "high"] * 3
    assert rec["cut_points"]["images"] == {"column": "image_pct", "q1": 2.0, "q2": 5.0,
                                           "levels": list(LEVELS)}


# --------------------------------------------------------------------------- tables

def test_table_level_none_only_at_zero():
    assert ct.table_level(0.0) == "none"
    assert ct.table_level(0.0001) == "some"
    assert ct.table_level(37.5) == "some"


def test_categorise_tables_from_the_csv_strings(cfg, fos):
    rows = [_row(0, table_pct="0.0"), _row(1, table_pct="0"), _row(2, table_pct="0.0102")]
    cats, _ = ct.categorise(rows, cfg, fos)
    assert [c["tables"] for c in cats] == ["none", "none", "some"]


# --------------------------------------------------------------------------- discipline

def test_field_level_is_the_first_digit_of_the_code_without_own_levels(cfg, fos):
    areas = cfg["discipline"]["areas"]
    assert ct.field_level("Computer and information sciences", fos, areas) == "Natural sciences"
    assert ct.field_level("Nano-technology", fos, areas) == "Engineering and technology"
    assert ct.field_level("Health sciences", fos, areas) == "Medical and health sciences"
    assert ct.field_level("Psychology", fos, areas) == "Social sciences"
    assert ct.field_level("Other humanities", fos, areas) == "Humanities"


def test_own_level_splits_its_area_only(cfg, fos):
    areas, own = cfg["discipline"]["areas"], cfg["discipline"]["own_level"]
    assert own == ["1.2"]
    cs = "Computer and information sciences"
    assert ct.field_level(cs, fos, areas, own) == cs
    assert ct.field_level("Mathematics", fos, areas, own) == "Natural sciences (other)"
    assert ct.field_level("Biological sciences", fos, areas, own) == "Natural sciences (other)"
    assert ct.field_level("Nano-technology", fos, areas, own) == "Engineering and technology"
    assert ct.field_level("Psychology", fos, areas, own) == "Social sciences"


def test_discipline_levels_in_code_order(cfg, fos):
    areas = cfg["discipline"]["areas"]
    levels = ct.discipline_levels(fos, areas, ["1.2"])
    assert levels[:3] == ["Computer and information sciences", "Natural sciences (other)",
                          "Engineering and technology"]
    assert len(levels) == 7
    assert ct.discipline_levels(fos, areas) == list(areas.values())


def test_own_level_rejects_an_unknown_code(cfg, fos):
    with pytest.raises(ValueError):
        ct.field_level("Mathematics", fos, cfg["discipline"]["areas"], ["9.9"])


@pytest.mark.parametrize("field", ["", "unassigned", "Astrology"])
def test_field_level_rejects_what_is_not_a_fos_field(cfg, fos, field):
    with pytest.raises(ValueError):
        ct.field_level(field, fos, cfg["discipline"]["areas"], cfg["discipline"]["own_level"])


def test_merge_small_areas_into_other():
    labels = ["A"] * 3 + ["B"] * 2 + ["C"]
    out, merged = ct.merge_small(labels, 2)
    assert out == ["A"] * 3 + ["B"] * 2 + ["other"]
    assert merged == {"C": 1}
    out, merged = ct.merge_small(labels, 3)
    assert out == ["A"] * 3 + ["other"] * 3
    assert merged == {"B": 2, "C": 1}


def test_categorise_discipline_merge(cfg, fos):
    n = cfg["discipline"]["min_count"]
    cs = "Computer and information sciences"
    rows = ([_row(i, discipline=cs) for i in range(n)]
            + [_row(n + i, discipline="Mathematics") for i in range(n)]
            + [_row(2 * n + i, discipline="Psychology") for i in range(n - 1)]
            + [_row(3 * n, discipline="Clinical medicine")])
    cats, rec = ct.categorise(rows, cfg, fos)
    assert [c["discipline"] for c in cats] == ([cs] * n + ["Natural sciences (other)"] * n
                                               + ["other"] * n)
    d = rec["discipline"]
    assert d["merged_into_other"] == {"Social sciences": n - 1,
                                      "Medical and health sciences": 1}
    assert d["level_counts_before_merge"][cs] == n
    assert d["level_counts_before_merge"]["Natural sciences (other)"] == n
    assert d["level_counts_before_merge"]["Humanities"] == 0
    counts = [r for r in ct.level_counts(cats, cfg, fos) if r["axis"] == "discipline"]
    assert [(r["level"], r["n"]) for r in counts] == [
        (cs, n), ("Natural sciences (other)", n), ("other", n)]


# --------------------------------------------------------------------------- the table

def test_text_source_kept_as_is_and_checked(cfg, fos):
    rows = [_row(0), _row(1, source="tesseract_ocr"), _row(2, corpus="m3av_a",
                                                           source="corpus_corrected_ocr")]
    cats, _ = ct.categorise(rows, cfg, fos)
    assert [c["text_source"] for c in cats] == ["pdf_text_layer", "tesseract_ocr",
                                                "corpus_corrected_ocr"]
    with pytest.raises(ValueError):
        ct.categorise([_row(0, source="manual")], cfg, fos)


def test_duplicate_lecture_stops_the_run(cfg, fos):
    with pytest.raises(ValueError):
        ct.categorise([_row(0), _row(0)], cfg, fos)


def test_level_counts_crosstab(cfg, fos):
    rows = [_row(0, corpus="mavils", table_pct=1.0), _row(1, corpus="mcd"),
            _row(2, corpus="mcd")]
    cats, _ = ct.categorise(rows, cfg, fos)
    counts = ct.level_counts(cats, cfg, fos)
    by = {(r["axis"], r["level"]): r for r in counts}
    assert by[("tables", "none")]["n"] == 2 and by[("tables", "none")]["mcd"] == 2
    assert by[("tables", "some")]["mavils"] == 1
    assert by[("corpus", "slide_highlight")]["n"] == 0
    # Every axis's counts add up to the number of lectures.
    for axis in ct.AXES:
        assert sum(r["n"] for r in counts if r["axis"] == axis) == len(rows)


def _write_attributes(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_build_and_write_are_deterministic(cfg, tmp_path):
    rows = [_row(i, corpus=("mavils", "mcd", "m3av_a")[i % 3], n_slides=5 + i % 7,
                 image_pct=(i * 7) % 11, table_pct=(i % 4 == 0) * 2.5, noise=(i * 3) % 13,
                 text_density=10 + i % 5,
                 discipline=("Mathematics", "Psychology", "Basic medicine")[i % 3])
            for i in range(30)]
    src = tmp_path / "document_attributes.csv"
    _write_attributes(src, rows)
    outs = []
    for k in range(2):
        cats, record, counts = ct.build(cfg, in_path=src, repo_root=REPO)
        out = tmp_path / f"run{k}"
        ct.write(cats, record, counts, cfg, out_dir=out)
        outs.append({p.name: p.read_bytes() for p in sorted(out.iterdir())})
    assert set(outs[0]) == {"document_categories.csv", "cut_points.json", "level_counts.csv"}
    assert outs[0] == outs[1]
    header = outs[0]["document_categories.csv"].decode().splitlines()[0]
    assert header == ("corpus,lecture_id,text_source,length,images,tables,noise,"
                      "text_density,discipline")
    assert record["input"]["sha256"] == ct.sha256_file(src)
