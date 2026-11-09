"""Tests for scripts/split_m3av.py: the split is seeded, stratified by topic and length,
and puts every kept lecture in exactly one version.

The lectures are synthetic; the real M3AV archive is never read.
"""

from __future__ import annotations

import collections
import csv
import json
import random

import prepare as p
import split_m3av as s


def lectures(n: int = 120, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        deck = rng.randint(5, 80)
        out.append({"lecture_id": f"{rng.choice(['CHI', 'NIH', 'Ubi'])}-{i:05X}", "n_slides_deck": deck, "n_slides": min(deck, 40), "n_units": rng.randint(10, 200)})
        out[-1]["topic"] = out[-1]["lecture_id"].split("-")[0]
    return out


def test_tercile_cuts_and_classes():
    assert s.tercile_cuts(list(range(1, 10))) == (3, 6)
    assert [s.length_class(x, (3, 6)) for x in (1, 3, 4, 6, 7)] == ["short", "short", "medium", "medium", "long"]


def test_split_is_seeded():
    lec = lectures()
    assert s.split_lectures(lec, 7) == s.split_lectures(list(reversed(lec)), 7)  # input order does not matter
    assert s.split_lectures(lec, 7)[0] != s.split_lectures(lec, 8)[0]


def test_split_is_disjoint_complete_and_stratified():
    lec = lectures(121)
    rows, cuts = s.split_lectures(lec)
    ids = [r["lecture_id"] for r in rows]
    assert sorted(ids) == sorted(x["lecture_id"] for x in lec) and len(set(ids)) == len(ids)
    assert {r["version"] for r in rows} == {"A", "B"}
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["stratum"]][r["version"]] += 1
    assert all(abs(c["A"] - c["B"]) <= 1 for c in per.values())
    total = collections.Counter(r["version"] for r in rows)
    assert abs(total["A"] - total["B"]) <= 1
    by_id = {x["lecture_id"]: x for x in lec}
    for r in rows:
        topic, length = r["stratum"].split("|")
        assert topic == by_id[r["lecture_id"]]["topic"]
        assert length == s.length_class(by_id[r["lecture_id"]]["n_slides_deck"], cuts)


def test_balance_report():
    lec = lectures(10)
    rows, _ = s.split_lectures(lec)
    bal = s.balance(rows, lec)
    assert sum(v["lectures"] for v in bal["per_version"].values()) == 10
    assert sum(c["A"] + c["B"] for c in bal["per_stratum"].values()) == 10


def test_main_writes_the_definition(tmp_path, monkeypatch):
    def fake_load(src, version):
        assert version is None
        yield p.Dropped("m3av", "CHI-00000", "no_slides", 0)
        for i in range(1, 9):
            n = 4 if i == 1 else 5 + i  # CHI-00001 has too few slides
            slides = [p.Slide(text=f"s{k}", source_ids=[str(k)]) for k in range(n)]
            units = [p.Unit(text="u", source_index=0, true_slide=1)]
            yield p.Lecture("m3av", f"CHI-{i:05d}", None, "corpus_corrected_ocr", "speech_segment", "lecture", slides, units, meta={"venue_prefix": "CHI"})

    monkeypatch.setattr(p, "load_m3av", fake_load)
    out = tmp_path / "splits" / "m3av_split.csv"
    s.main(["--raw-root", str(tmp_path), "--out", str(out)])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["lecture_id", "version", "stratum"]
    assert [r["lecture_id"] for r in rows] == [f"CHI-{i:05d}" for i in range(2, 9)]
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["seed"] == s.SEED and meta["lectures"] == 7 and meta["talks_dropped_by_the_rules"] == 2
    # prepare.py reads it back.
    assert set(p.read_split(out).values()) == {"A", "B"}
