"""Step 14: the analysis (scripts/analysis.py).

Nothing here reads the real data: three made-up methods on six made-up lectures, with
categories on the step-6 axes, so the ranks, the ties and Q3's tie rules can be worked
out by hand.
"""

from __future__ import annotations

import pytest

import analysis as A


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

LECTURES = [("c", f"L{i}") for i in range(6)]
#: n_correct out of 10 units per lecture: x and y have the same total (a tie overall).
CORRECT = {"x": [10, 10, 0, 0, 5, 5],
           "y": [0, 0, 10, 10, 5, 5],
           "z": [5, 5, 5, 5, 0, 0]}
KINDS = {"x": "non-LLM", "y": "LLM", "z": "mixed"}
#: Every axis splits the lectures the same way: L0, L1 / L2, L3 / L4, L5.
LEVEL_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
ORDER = {a: [f"{a}_{i}" for i in range(3)] for a in A.AXES}


def lecture_row(method: str, i: int) -> dict:
    r = {"corpus": "c", "lecture_id": f"L{i}", "status": "ok",
         "n_correct": str(CORRECT[method][i]), "n_units": "10",
         "accuracy": str(CORRECT[method][i] / 10), "compute_s": "0.5",
         "build_cost_s": "2.0"}
    if KINDS[method] != "non-LLM":
        r.update(used_calls="1", used_seconds="3.0", used_prompt_tokens="100",
                 used_completion_tokens="10")
    return r


def inputs():
    rows = {m: [lecture_row(m, i) for i in range(6)] for m in CORRECT}
    cats = [{"corpus": c, "lecture_id": l,
             **{a: ORDER[a][LEVEL_OF[i]] for a in A.AXES}}
            for i, (c, l) in enumerate(LECTURES)]
    load = {"x": 1.0, "y": 0.0, "z": 0.0}
    summary = {}
    for m in CORRECT:
        acc = 100 * sum(CORRECT[m]) / 60
        s = {"method": m, "test_accuracy": repr(acc),
             "compute_s_total": "3.0", "compute_s_per_lecture": "0.5",
             "build_cost_s_total": repr(12.0 + load[m]),
             "build_cost_s_per_lecture": repr((12.0 + load[m]) / 6)}
        for c, v in (("used_calls", 1), ("used_seconds", 3.0), ("used_prompt_tokens", 100),
                     ("used_completion_tokens", 10)):
            tot = 6 * v if KINDS[m] != "non-LLM" else 0
            s[f"{c}_total"], s[f"{c}_per_lecture"] = repr(float(tot)), repr(tot / 6)
        summary[m] = s
    return rows, KINDS, load, summary, cats, ORDER, sorted(CORRECT)


# --------------------------------------------------------------------------- #
# Ranks
# --------------------------------------------------------------------------- #

def test_ranks_highest_first_and_ties_share_the_rank():
    assert A.ranks({"a": 50.0, "b": 70.0, "c": 70.0, "d": 10.0}) == \
        {"b": 1, "c": 1, "a": 3, "d": 4}
    assert A.ranks({"a": 1.0}) == {"a": 1}


def test_a_last_bit_difference_does_not_split_a_tie():
    a = (0.1 + 0.2) * 100
    b = 0.3 * 100
    assert a != b
    assert A.ranks({"a": a, "b": b}) == {"a": 1, "b": 1}


# --------------------------------------------------------------------------- #
# Q1
# --------------------------------------------------------------------------- #

def test_q1_ranks_and_costs_by_kind():
    rows, kinds, load, *_ = inputs()
    q1 = {r["method"]: r for r in A.q1_rows(rows, kinds, load)}
    assert q1["x"]["accuracy"] == pytest.approx(50.0)
    assert q1["y"]["accuracy"] == pytest.approx(50.0)
    assert q1["z"]["accuracy"] == pytest.approx(100 * 20 / 60)
    assert (q1["x"]["rank"], q1["y"]["rank"], q1["z"]["rank"]) == (1, 1, 3)
    # non-LLM: compute and build (with the model load once), no LLM cost
    assert q1["x"]["compute_s_total"] == pytest.approx(3.0)
    assert q1["x"]["build_cost_s_total"] == pytest.approx(13.0)
    assert q1["x"]["build_cost_s_per_lecture"] == pytest.approx(13.0 / 6)
    assert q1["x"]["used_calls_total"] is None and q1["x"]["used_tokens_total"] is None
    # LLM: calls, seconds, tokens and build; no compute
    assert q1["y"]["compute_s_total"] is None
    assert q1["y"]["used_calls_total"] == 6 and q1["y"]["used_tokens_total"] == 660
    assert q1["y"]["used_tokens_per_lecture"] == pytest.approx(110)
    assert q1["y"]["build_cost_s_total"] == pytest.approx(12.0)
    # mixed: all
    assert all(q1["z"][f"{c}_total"] is not None for c, _ in A.COSTS)


def test_q1_is_sorted_by_rank_then_name():
    rows, kinds, load, *_ = inputs()
    assert [r["method"] for r in A.q1_rows(rows, kinds, load)] == ["x", "y", "z"]


# --------------------------------------------------------------------------- #
# Q2
# --------------------------------------------------------------------------- #

def test_q2_means_and_ranks_within_each_level():
    rows, kinds, load, summary, cats, order, expected = inputs()
    tables, _, problems = A.analyse(rows, kinds, load, summary, cats, order, expected)
    assert problems == []
    long = tables["q2"]["length"]
    cell = {(r["level"], r["method"]): r for r in long}
    assert {r["n"] for r in long} == {2}
    # level 0: x 100, z 50, y 0
    assert [cell[("length_0", m)]["rank"] for m in "xyz"] == [1, 3, 2]
    assert cell[("length_0", "x")]["accuracy"] == pytest.approx(100.0)
    # level 2: x and y tie at 50, z 0
    assert [cell[("length_2", m)]["rank"] for m in "xyz"] == [1, 1, 3]
    wide, cols = A.q2_wide(long, tables["q1"])
    assert cols[:4] == ["method", "kind", "overall_rank", "overall_accuracy"]
    assert "length_1 rank" in cols and wide[0]["method"] == "x"
    assert wide[1]["length_1 rank"] == 1          # y is first on level 1


def test_q2_a_level_without_test_lectures_fails_loudly():
    rows, kinds, load, summary, cats, order, expected = inputs()
    order = {**order, "noise": order["noise"] + ["noise_empty"]}
    _, _, problems = A.analyse(rows, kinds, load, summary, cats, order, expected)
    assert any("noise_empty" in p for p in problems)


def test_a_lecture_without_one_level_per_axis_fails_loudly():
    rows, kinds, load, summary, cats, order, expected = inputs()
    _, _, problems = A.analyse(rows, kinds, load, summary, cats[1:] + cats[:1] * 2,
                               order, expected)
    assert any("has 2 rows" in p for p in problems)
    cats2 = [dict(c) for c in cats]
    cats2[0]["tables"] = "unknown"
    _, _, problems = A.analyse(rows, kinds, load, summary, cats2, order, expected)
    assert any("not a step-6 level" in p for p in problems)
    # the lecture without a level breaks the weighted-mean check too
    assert any("weighted" in p for p in problems)


def test_q1_against_the_summary_fails_loudly():
    rows, kinds, load, summary, cats, order, expected = inputs()
    summary = {**summary, "z": {**summary["z"], "test_accuracy": "33.0"}}
    _, _, problems = A.analyse(rows, kinds, load, summary, cats, order, expected)
    assert any("Q1 z: accuracy" in p for p in problems)
    rows2, *_ = inputs()
    _, _, problems = A.analyse(rows2, kinds, {}, inputs()[3], cats, order, expected)
    assert any("Q1 x: build_cost_s_total" in p for p in problems)


def test_missing_methods_or_lectures_fail_loudly():
    rows, kinds, load, summary, cats, order, expected = inputs()
    _, _, problems = A.analyse({m: r for m, r in rows.items() if m != "z"}, kinds, load,
                               summary, cats, order, expected)
    assert any("missing ['z']" in p for p in problems)
    rows["y"] = rows["y"][:-1]
    _, _, problems = A.analyse(rows, kinds, load, summary, cats, order, expected)
    assert any("y differs" in p for p in problems)


# --------------------------------------------------------------------------- #
# Q3
# --------------------------------------------------------------------------- #

def E(axis, level, rank, acc):
    return {"axis": axis, "level": level, "rank": rank, "accuracy": acc}


def test_q3_best_is_the_smallest_rank_and_worst_the_largest():
    best, worst = A.best_worst([E("a", "1", 3, 60.0), E("a", "2", 1, 40.0),
                                E("b", "1", 7, 70.0)])
    assert [e["level"] for e in best] == ["2"] and best[0]["accuracy"] == 40.0
    assert [(e["axis"], e["level"]) for e in worst] == [("b", "1")]


def test_q3_rank_ties_go_to_higher_accuracy_for_best_and_lower_for_worst():
    entries = [E("a", "1", 1, 60.0), E("a", "2", 1, 80.0),
               E("b", "1", 9, 30.0), E("b", "2", 9, 20.0)]
    best, worst = A.best_worst(entries)
    assert [(e["axis"], e["level"]) for e in best] == [("a", "2")]
    assert [(e["axis"], e["level"]) for e in worst] == [("b", "2")]


def test_q3_a_tie_in_rank_and_accuracy_lists_both():
    entries = [E("a", "1", 2, 55.0), E("b", "1", 2, 55.0), E("c", "1", 5, 10.0),
               E("c", "2", 5, 10.0)]
    best, worst = A.best_worst(entries)
    assert [e["axis"] for e in best] == ["a", "b"]
    assert [e["level"] for e in worst] == ["1", "2"]


def test_q3_rows_from_the_synthetic_run():
    rows, kinds, load, summary, cats, order, expected = inputs()
    tables, _, _ = A.analyse(rows, kinds, load, summary, cats, order, expected)
    q3 = {r["method"]: r for r in tables["q3"]}
    # x: rank 1 on every axis's levels 0 (100 %) and 2 (50 %): best is level 0 of each
    # axis (higher accuracy), all listed; worst: rank 3 on every level 1 (0 %)
    assert q3["x"]["best_rank"] == 1 and q3["x"]["best_accuracy"] == pytest.approx(100.0)
    assert q3["x"]["best_level"].split("; ") == [f"{a}: {a}_0" for a in A.AXES]
    assert q3["x"]["worst_rank"] == 3 and q3["x"]["worst_accuracy"] == pytest.approx(0.0)
    assert q3["x"]["worst_level"].split("; ") == [f"{a}: {a}_1" for a in A.AXES]
    assert (q3["z"]["overall_rank"], q3["z"]["worst_rank"]) == (3, 3)


def test_the_tables_are_written(tmp_path):
    rows, kinds, load, summary, cats, order, expected = inputs()
    tables, lines, problems = A.analyse(rows, kinds, load, summary, cats, order, expected)
    assert problems == [] and len(lines) == 6
    A.write(tables, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"q1_ranking.csv", "q1_ranking.md", "q2_levels.md", "q3_best_worst.csv",
            "q3_best_worst.md"} <= names
    assert {f"q2_{a}_{f}.csv" for a in A.AXES for f in ("long", "wide")} <= names
    q1 = A.read_csv(tmp_path / "q1_ranking.csv")
    assert q1[0]["used_calls_total"] == "" and q1[1]["compute_s_total"] == ""
    assert "| 1 | x | non-LLM | 50.00 |" in (tmp_path / "q1_ranking.md").read_text()
