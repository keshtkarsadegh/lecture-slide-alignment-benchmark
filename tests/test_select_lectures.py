"""Tests for step 7: disjoint sets of the exact sizes, the per-level minima, the small
corpora taken whole with the exact split, the m3av balance, determinism, and the stop
when a minimum cannot be met.

Every pool is a small synthetic categories table; the levels are step 6's
(categorise.axis_levels on the repository's config/categories.toml).
"""

from __future__ import annotations

import csv
import random
import tomllib
from pathlib import Path

import pytest

import select_lectures as sl

REPO = Path(__file__).resolve().parent.parent
T3 = ("low", "medium", "high")
LENGTH = ("short", "medium", "long")
SOURCE = {"mavils": "pdf_text_layer", "m3av_a": "corpus_corrected_ocr",
          "m3av_b": "corpus_corrected_ocr", "mcd": "tesseract_ocr",
          "slide_highlight": "tesseract_ocr"}
SIZES = {"mavils": 6, "mcd": 6, "slide_highlight": 3, "m3av_a": 40, "m3av_b": 40}
CS, SOC = "Computer and information sciences", "Social sciences"


def _cfg(seed=7, tuning=(12, 2), test=(24, 3)):
    """Tuning 12 (min 2 per level), test 24 (min 3): the small corpora give tuning
    mavils 2, mcd 2, slide_highlight 1 and test 4, 4, 2; m3av fills 4/3 and 7/7."""
    return {"seed": seed,
            "sets": {"tuning": {"size": tuning[0], "min_per_level": tuning[1]},
                     "test": {"size": test[0], "min_per_level": test[1]}},
            "small_corpora": {"corpora": ["mavils", "mcd", "slide_highlight"],
                              "tuning_share": [1, 3]},
            "fill_corpora": {"corpora": ["m3av_a", "m3av_b"]}}


def _pool(rare=tuple(("m3av_b", k) for k in range(6))):
    """Every level of every axis present; Social sciences only on the `rare` lectures
    (default 6 m3av_b lectures, one more than the 2 + 3 the minima need)."""
    rare = set(rare)
    rows, i = [], 0
    for corpus, n in SIZES.items():
        for k in range(n):
            rows.append({"corpus": corpus, "lecture_id": f"{corpus}_{k:03d}",
                         "text_source": SOURCE[corpus], "length": LENGTH[i % 3],
                         "images": T3[(i // 3) % 3], "tables": ("none", "some")[i % 2],
                         "noise": T3[(i // 2) % 3], "text_density": T3[(i + 1) % 3],
                         "discipline": SOC if (corpus, k) in rare else CS})
            i += 1
    return rows


@pytest.fixture(scope="module")
def selected():
    pool = _pool()
    return sl.select(pool, _cfg(), sl.pool_levels(pool, REPO))


def _by_set(selected, s):
    return [r for r in selected if r["set"] == s]


# --------------------------------------------------------------------------- the counts

def test_round_half_up():
    assert [sl.round_half_up(n, 1, 3) for n in (20, 23, 14, 3, 6)] == [7, 8, 5, 1, 2]
    assert sl.round_half_up(1, 1, 2) == 1          # 0.5 -> 1, where round() gives 0
    assert sl.round_half_up(5, 1, 2) == 3          # 2.5 -> 3, where round() gives 2


def test_corpus_counts_with_the_repository_settings():
    with open(REPO / "config" / "selection.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    pool_sizes = {"mavils": 20, "m3av_a": 538, "m3av_b": 536, "mcd": 23,
                  "slide_highlight": 14}
    counts = sl.corpus_counts(pool_sizes, cfg)
    assert counts["tuning"] == {"mavils": 7, "mcd": 8, "slide_highlight": 5,
                                "m3av_a": 20, "m3av_b": 20}
    assert counts["test"] == {"mavils": 13, "mcd": 15, "slide_highlight": 9,
                              "m3av_a": 52, "m3av_b": 51}


# --------------------------------------------------------------------------- the sets

def test_sets_are_disjoint_with_the_exact_sizes(selected):
    tuning = {(r["corpus"], r["lecture_id"]) for r in _by_set(selected, "tuning")}
    test = {(r["corpus"], r["lecture_id"]) for r in _by_set(selected, "test")}
    assert len(tuning) == len(_by_set(selected, "tuning")) == 12
    assert len(test) == len(_by_set(selected, "test")) == 24
    assert not tuning & test


def test_minima_are_met_on_every_level_but_corpus(selected):
    pool = _pool()
    levels = sl.pool_levels(pool, REPO)
    counts = sl.selection_counts(selected, pool, levels)
    for r in counts:
        if r["axis"] == "corpus":
            continue
        assert r["tuning"] >= 2, r
        assert r["test"] >= 3, r
    # The rare level needs the greedy fill: 5 of its 6 lectures are taken.
    soc = next(r for r in counts if r["level"] == SOC)
    assert soc["pool"] == 6 and soc["tuning"] + soc["test"] >= 5


def test_small_corpora_whole_with_the_exact_split(selected):
    for corpus, (n_tuning, n_test) in {"mavils": (2, 4), "mcd": (2, 4),
                                       "slide_highlight": (1, 2)}.items():
        t = [r for r in _by_set(selected, "tuning") if r["corpus"] == corpus]
        s = [r for r in _by_set(selected, "test") if r["corpus"] == corpus]
        assert (len(t), len(s)) == (n_tuning, n_test)
        assert len({r["lecture_id"] for r in t + s}) == SIZES[corpus]


def test_m3av_balance(selected):
    for s, (a, b) in {"tuning": (4, 3), "test": (7, 7)}.items():
        sub = _by_set(selected, s)
        assert sum(r["corpus"] == "m3av_a" for r in sub) == a
        assert sum(r["corpus"] == "m3av_b" for r in sub) == b


# --------------------------------------------------------------------------- determinism

def test_select_is_deterministic_and_ignores_the_input_order(selected):
    pool = _pool()
    levels = sl.pool_levels(pool, REPO)
    assert sl.select(pool, _cfg(), levels) == selected
    shuffled = pool[:]
    random.Random(1).shuffle(shuffled)
    assert sl.select(shuffled, _cfg(), levels) == selected
    # The seed matters.
    assert sl.select(pool, _cfg(seed=8), levels) != selected


def _write_categories(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_build_and_write_are_deterministic(tmp_path):
    src = tmp_path / "document_categories.csv"
    _write_categories(src, _pool())
    outs = []
    for k in range(2):
        selected, record, counts = sl.build(_cfg(), in_path=src, repo_root=REPO)
        out = tmp_path / f"run{k}"
        sl.write(selected, record, counts, out_dir=out)
        outs.append({p.name: p.read_bytes() for p in sorted(out.iterdir())})
    assert set(outs[0]) == {"selected_lectures.csv", "selection_counts.csv",
                            "selection.json"}
    assert outs[0] == outs[1]
    header = outs[0]["selected_lectures.csv"].decode().splitlines()[0]
    assert header == ("corpus,lecture_id,set,text_source,length,images,tables,noise,"
                      "text_density,discipline")
    assert outs[0]["selection_counts.csv"].decode().splitlines()[0] == \
        "axis,level,tuning,test,pool"
    assert record["seed"] == 7
    assert record["input"]["sha256"] == sl.ct.sha256_file(src)


# --------------------------------------------------------------------------- the stops

def test_a_level_too_small_for_the_minima_stops_the_run():
    pool = _pool(rare=[("m3av_b", k) for k in range(4)])     # 4 < 2 + 3
    with pytest.raises(ValueError, match="not attainable.*Social sciences"):
        sl.select(pool, _cfg(), sl.pool_levels(pool, REPO))


def test_a_minimum_the_fill_cannot_reach_stops_the_run():
    # 4 lectures carry the level, enough for the minima 1 + 3, but all in m3av_b, and a
    # tuning set of 6 has room for 5 small-corpus lectures, 1 m3av_a and no m3av_b.
    pool = _pool(rare=[("m3av_b", k) for k in range(4)])
    with pytest.raises(ValueError, match="set tuning: minimum 1 .*Social sciences"):
        sl.select(pool, _cfg(tuning=(6, 1)), sl.pool_levels(pool, REPO))


def test_a_duplicate_lecture_stops_the_run():
    pool = _pool()
    with pytest.raises(ValueError, match="more than once"):
        sl.select(pool + pool[:1], _cfg(), sl.pool_levels(pool, REPO))
