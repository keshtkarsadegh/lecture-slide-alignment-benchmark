"""Step 16: the per-unit predictions (scripts/final_predictions.py) and Q4, the important
units placed on the wrong slide (scripts/analysis_q4.py).

Nothing here reads the real data or calls the API: two made-up methods on three
made-up lectures with made-up labels, small enough to work out the percentages by hand;
the replay runs on made-up matrices with a fake LLM whose answers are put in a cache in
tmp_path first.
"""

from __future__ import annotations

import pytest

import analysis_q4 as Q
import final_predictions as FP
import tune as T

from test_tuning import TimedGold, fake_lectures, fake_llms  # noqa: F401


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

#: Per lecture its units' (true slide, label).
LECTURES = {"L0": [(1, 2), (1, 2), (2, 1), (2, 0)],
            "L1": [(1, 2), (2, 2)],
            "L2": [(1, 2), (1, 1), (2, 2), (2, 2)]}
#: Per lecture its (length, images, tables); no lecture has medium length.
CATS = {"L0": ("short", "low", "none"), "L1": ("long", "high", "some"),
        "L2": ("long", "medium", "none")}
#: Per method the units it puts on a wrong slide.
WRONG = {"x": {("L0", 0), ("L0", 2), ("L2", 3)},          # 2 important wrong
         "y": {("L0", 0), ("L0", 1), ("L1", 1), ("L2", 0)}}  # 4 important wrong


def label_rows():
    return [{"corpus": "c", "lecture_id": lec, "unit_index": str(i),
             "true_slide": str(s), "label": str(lab)}
            for lec, units in LECTURES.items() for i, (s, lab) in enumerate(units)]


def pred_rows(m):
    return [{"corpus": "c", "lecture_id": lec, "unit_index": str(i), "true_slide": str(s),
             "pred_slide": str(-1 if (lec, i) in WRONG[m] else s)}
            for lec, units in LECTURES.items() for i, (s, _) in enumerate(units)]


def cat_rows():
    return [{"corpus": "c", "lecture_id": lec, "length": ln, "images": im, "tables": tb}
            for lec, (ln, im, tb) in CATS.items()]


def step13_rows(m):
    return [{"corpus": "c", "lecture_id": lec, "n_units": str(len(units)),
             "n_correct": str(len(units) - sum(1 for k in WRONG[m] if k[0] == lec))}
            for lec, units in LECTURES.items()]


def run(**change):
    args = {"preds": {m: pred_rows(m) for m in WRONG}, "labels_rows": label_rows(),
            "cats": cat_rows(), "step13": {m: step13_rows(m) for m in WRONG},
            "expected": sorted(WRONG)}
    args.update(change)
    return Q.analyse(**args)


# --------------------------------------------------------------------------- #
# Q4
# --------------------------------------------------------------------------- #

def test_the_percentage_is_important_wrong_over_all_units_pooled():
    rows, units, _, problems = run()
    assert problems == []
    x = next(r for r in rows if r["method"] == "x")
    # x: L0 unit 0 (core) and L2 unit 3 (core) count; L0 unit 2 (supporting) does not.
    assert units["all"] == 10 and x["all_wrong_important"] == 2
    assert x["all_pct"] == pytest.approx(100 * 2 / 10)
    # long = L1 + L2, 6 units, one important wrong: pooled 1/6, not the lecture mean.
    assert x["length_long_units"] == 6 and x["length_long_wrong_important"] == 1
    assert x["length_long_pct"] == pytest.approx(100 / 6)
    assert x["tables_none_pct"] == pytest.approx(100 * 2 / 8)


def test_the_rows_are_sorted_ascending_by_all_documents():
    rows, _, _, _ = run()
    assert [r["method"] for r in rows] == ["x", "y"]
    flipped = Q.sort_rows([{"method": "a", "all_pct": 30.0}, {"method": "b", "all_pct": 5.0},
                           {"method": "c", "all_pct": 5.0}])
    assert [r["method"] for r in flipped] == ["b", "c", "a"]


def test_a_column_without_units_does_not_crash():
    rows, units, lines, problems = run()
    assert problems == [] and units["length_medium"] == 0
    assert all(r["length_medium_pct"] is None for r in rows)
    md = Q.markdown(rows, units)
    assert "length medium (0)" in md and "| - |" in md
    assert any(line.startswith("3. length") and line.endswith("ok") for line in lines)


def test_a_missing_label_raises():
    labels = Q.label_table(label_rows()[1:])
    with pytest.raises(ValueError, match="no label"):
        Q.wrong_units(pred_rows("x"), labels)
    rows, _, _, problems = run(labels_rows=label_rows()[1:])
    assert rows == [] and problems and problems[0].startswith("1.")


# --------------------------------------------------------------------------- #
# The replay of the final run (scripts/final_predictions.py)
# --------------------------------------------------------------------------- #

def test_run_one_keeps_the_scored_labels_only_when_asked():
    grids = T.load_grid()
    lecs, mats = fake_lectures(((12, 4),))
    c = grids["dtw"].combos()[0]
    plain = T.run_one(c, lecs[0], mats)
    kept = T.run_one(c, lecs[0], mats, keep_pred=True)
    assert "pred" not in plain
    assert T.accuracy(kept["pred"], lecs[0].gold)[0] == kept["n_correct"] == plain["n_correct"]


def test_the_offline_llms_never_read_the_key(monkeypatch):
    def no_key(*a, **k):
        raise AssertionError("the API key was read")
    monkeypatch.setattr(T, "api_key", no_key)
    tuned = T.load_tuned()
    grids = T.tuned_grids(tuned, T.load_grid())
    llms = FP.offline_llms(grids, tuned)
    assert llms and all(llm.client is None for llm in llms.values())


def test_the_replay_answers_from_the_cache_and_stops_on_a_miss(tmp_path):
    tuned = T.load_tuned()
    grid = T.tuned_grids(tuned, T.load_grid())["llm_monotone"]
    lecs, mats = fake_lectures(((12, 4), (9, 3)))
    L, other = lecs
    live = fake_llms(tmp_path, TimedGold(L.gold), models=grid.models)
    paid = T.run_one(grid.combos()[0], L, mats, llms=live)
    assert paid["calls"] == 1
    offline = fake_llms(tmp_path, None, models=grid.models)
    row = FP.predict("llm_monotone", grid, L, mats, offline, tuned)
    assert row["calls"] == 0 and row["cache_hits"] == 1
    assert T.accuracy(row["pred"], L.gold)[0] == paid["n_correct"]
    with pytest.raises(SystemExit, match="crash.*not cached"):
        FP.predict("llm_monotone", grid, other, mats, offline, tuned)


def test_the_written_predictions_recount_to_the_run(tmp_path):
    lecs, _ = fake_lectures(((5, 3),))
    L = lecs[0]
    pred = [int(g) for g in L.gold]
    pred[1], pred[2] = -1, (pred[2] + 1) % 3
    rows = FP.unit_rows(L, pred)
    assert rows[0] == ("c", "L0", 0, int(L.gold[0]) + 1, int(L.gold[0]) + 1)
    assert rows[1][4] == -1                                   # no slide stays -1
    path = tmp_path / "m.csv.gz"
    FP.write_predictions(path, rows)
    counts = FP.recount(FP.read_predictions(path))
    assert counts == {("c", "L0"): (3, 5)}
    ok = [{"corpus": "c", "lecture_id": "L0", "n_correct": "3", "n_units": "5"}]
    assert FP.compare("m", counts, ok) == []
    assert len(FP.compare("m", counts, [{**ok[0], "n_correct": "4"}])) == 1
    assert len(FP.compare("m", counts, ok + [{**ok[0], "lecture_id": "L9"}])) == 1
