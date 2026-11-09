"""Step 11: the tuning runner (scripts/tune.py) and the selector (scripts/select_settings.py).

Nothing here reads the real data or calls the API: lectures and matrices are made up,
the selection file and a prepared corpus are written to tmp_path, and the LLM is a fake
client that answers from the true labels.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pytest

import select_settings as SEL
import tune as T
from methods import DPConfig, Member, run_method
from methods.llm_client import LLMCallFailed

from test_llm_methods import GoldClient

REPO = Path(__file__).resolve().parents[1]
DESIGN = REPO / "docs" / "TUNING_DESIGN.md"


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

def design_counts() -> dict[str, int]:
    """The combinations column of docs/TUNING_DESIGN.md's section 5.1 table."""
    text = DESIGN.read_text(encoding="utf-8")
    section = text.split("### 5.1 Combinations")[1].split("### 5.2")[0]
    out = {}
    for name, combos in re.findall(r"^\| (\*\*.+?\*\*|[a-z_0-9]+) \| (\*\*[\d,]+\*\*|[\d,]+) \|",
                                   section, flags=re.M):
        out[name.strip("*")] = int(combos.strip("*").replace(",", ""))
    return out


def test_grid_counts_equal_the_design_table():
    grids = T.load_grid()
    want = design_counts()
    per_method = {k: v for k, v in want.items() if not k.startswith(("non-LLM", "LLM", "all"))}
    assert len(per_method) == 31
    assert {m: g.n_combos() for m, g in grids.items()} == per_method
    llm_like = {"llm_single_call", "llm_monotone", "llm_monotone_constrained", "hierarchical",
                "llm_rerank_kbest", "fusion", "ensemble"}
    assert sum(g.n_combos() for m, g in grids.items() if m not in llm_like) \
        == want["non-LLM total (24 methods)"] == 25115
    assert sum(g.n_combos() for m, g in grids.items() if m in llm_like) \
        == want["LLM and mixed total (7 methods)"] == 658
    assert sum(g.n_combos() for g in grids.values()) == want["all 31"] == 25773
    for g in grids.values():
        keys = [c.key for c in g.combos()]
        assert len(keys) == len(set(keys)) == g.n_combos()


def test_grid_stages_and_the_llm_order():
    grids = T.load_grid()
    assert T.stage_methods(grids, "llm") == ["llm_single_call", "llm_monotone",
                                             "llm_monotone_constrained", "hierarchical"]
    assert set(T.stage_methods(grids, "consensus")) == {"score_fusion", "ensemble",
                                                        "llm_rerank_kbest", "fusion"}
    assert grids["mbr"].stage == "matrix"
    assert len(T.stage_methods(grids, "matrix")) == 23
    # hierarchical: refinement on before off, for every model and leaf size
    hier = [(c.model, dict(c.settings)) for c in grids["hierarchical"].combos()]
    for i in range(0, len(hier), 2):
        assert hier[i][1]["use_boundary_refinement"] is True
        assert hier[i + 1][1]["use_boundary_refinement"] is False
        assert hier[i][0] == hier[i + 1][0]
    assert grids["mbr"].settings["temperature"] == (None, 1.0)
    # decided 2026-10-12: one model for every LLM and mixed method
    for name, g in grids.items():
        assert g.models in ((), ("qwen3-30b-a3b-instruct-2507",)), name
        assert g.models or name not in ("llm_single_call", "llm_monotone",
                                        "llm_monotone_constrained", "hierarchical",
                                        "llm_rerank_kbest", "fusion")
    assert T.default_model() == "qwen3-30b-a3b-instruct-2507"


def _grid_text(**replace):
    text = T.GRID_PATH.read_text(encoding="utf-8")
    for old, new in replace.items():
        assert old in text
        text = text.replace(old, new)
    return text


@pytest.mark.parametrize("old,new", [("radius = [2, 4, 8]", "radios = [2, 4, 8]"),
                                     ("radius = [2, 4, 8]", "radius = [2, 8]"),
                                     ("max_jump = [1, 3, 8]", "max_jump = [1, 3, 3]"),
                                     ('[argmax]\nstage = "matrix"', '[argmax]\nstage = "x"')])
def test_grid_rejects_a_bad_setting_a_missing_default_or_a_repeat(tmp_path, old, new):
    p = tmp_path / "grid.toml"
    p.write_text(_grid_text(**{old: new}), encoding="utf-8")
    with pytest.raises(ValueError):
        T.load_grid(p)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #

def test_accuracy_counts_equal_slides_and_minus_one_is_always_wrong():
    assert T.accuracy([0, 1, -1, 2], [0, 1, 2, 2]) == (3, 4)
    assert T.accuracy([-1, -1], [0, 0]) == (0, 2)
    assert T.accuracy([], []) == (0, 0)
    with pytest.raises(ValueError):
        T.accuracy([0, 1], [0, 1, 2])


# --------------------------------------------------------------------------- #
# The tuning-set guard
# --------------------------------------------------------------------------- #

def _record(corpus, lid, n=6, m=3):
    return {"corpus": corpus, "lecture_id": lid,
            "slides": [{"index": j + 1, "text": f"slide {j}"} for j in range(m)],
            "units": [{"index": i, "text": f"unit {i}", "true_slide": 1 + i * m // n}
                      for i in range(n)]}


@pytest.fixture
def tiny_set(tmp_path):
    sel = tmp_path / "selected.csv"
    sel.write_text("corpus,lecture_id,set\nc,A,tuning\nc,B,tuning\nc,T1,test\n",
                   encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with gzip.open(prepared / "c.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_record("c", "B")) + "\n")
        # a test lecture whose line is not even JSON: it must be skipped unread
        fh.write('{"corpus": "c", "lecture_id": "T1", THIS IS NOT JSON\n')
        fh.write(json.dumps(_record("c", "A")) + "\n")
    return sel, prepared


def test_only_the_tuning_set_is_read(tiny_set):
    sel, prepared = tiny_set
    with pytest.raises(SystemExit, match="tuning"):
        T.set_lectures("test", sel)
    with pytest.raises(SystemExit):
        T.load_lectures("test", selection=sel, prepared=prepared)
    with pytest.raises(SystemExit, match="refused"):
        T.load_lectures(only=["c/T1"], selection=sel, prepared=prepared)
    lecs = T.load_lectures(selection=sel, prepared=prepared)       # T1's line is never parsed
    assert [L.name for L in lecs] == ["c/A", "c/B"]                 # selection order
    assert list(lecs[0].gold) == [0, 0, 1, 1, 2, 2]
    assert [L.name for L in T.load_lectures(only=["c/B"], selection=sel,
                                            prepared=prepared)] == ["c/B"]


def test_the_selector_refuses_a_run_outside_the_tuning_set():
    row = {"method": "argmax", "combo": "{}", "corpus": "c", "lecture_id": "T1",
           "n_units": 1, "n_correct": 1, "status": "ok"}
    with pytest.raises(SystemExit, match="not a tuning lecture"):
        SEL.combo_stats([row], {("c", "A")})


# --------------------------------------------------------------------------- #
# The runner: rows, crashes, resume, merge, LLM cost, members
# --------------------------------------------------------------------------- #

def fake_lectures(sizes=((24, 6), (30, 8))):
    lecs, mats = [], {}
    for k, (n, m) in enumerate(sizes):
        gold = np.asarray([i * m // n for i in range(n)])
        L = T.Lecture(k, "c", f"L{k}", gold, [f"slide {j}" for j in range(m)],
                      [f"unit {i}" for i in range(n)])
        lecs.append(L)
        for s, meas in enumerate(T.MEASURES):
            rng = np.random.default_rng(100 * k + s)
            S = rng.normal(scale=0.6 + 0.1 * s, size=(n, m))
            S[np.arange(n), gold] += 1.5
            mats[(meas, L.name)] = S
    return lecs, mats


def test_a_row_carries_accuracy_and_compute_time():
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    c = grids["dtw"].combos()[0]
    row = T.run_one(c, lecs[0], mats)
    labels = run_method("dtw", mats[(c.measure, lecs[0].name)], cfg=DPConfig(**dict(c.cfg)))
    assert row["n_correct"] == T.accuracy(labels, lecs[0].gold)[0]
    assert row["accuracy"] == row["n_correct"] / 24
    assert row["status"] == "ok" and row["compute_s"] > 0 and row["calls"] == 0
    assert set(T.COLUMN_NAMES) == set(row)


def test_a_crash_is_recorded_and_tried_again(tmp_path, monkeypatch):
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    real = T.run_method

    def flaky(name, S, **kw):
        if S.shape[0] == 30:
            raise RuntimeError("boom")
        return real(name, S, **kw)

    monkeypatch.setattr(T, "run_method", flaky)
    shard = T.Shard(tmp_path / "shards" / "s.sqlite3")
    combos = {"argmax": grids["argmax"].combos()[:2]}
    T.run(["argmax"], lecs, mats, shard, grids=grids, runs_dir=tmp_path / "runs",
          combos=combos, log=lambda *_: None)
    rows = T.read_shard(shard.path)
    bad = [r for r in rows if r["status"] != "ok"]
    assert len(rows) == 4 and len(bad) == 2
    assert all(r["status"] == "crash" and "boom" in r["error"] and r["n_correct"] == 0
               for r in bad)
    monkeypatch.setattr(T, "run_method", real)
    made = T.run(["argmax"], lecs, mats, shard, grids=grids, runs_dir=tmp_path / "runs",
                 combos=combos, log=lambda *_: None)
    assert made == {"argmax": 2}                     # only the crashed ones again
    assert all(r["status"] == "ok" for r in T.read_shard(shard.path))


def test_resume_and_merge_are_idempotent(tmp_path, monkeypatch):
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    combos = {"dtw": grids["dtw"].combos()[:5], "pelt": grids["pelt"].combos()[:3]}
    out = tmp_path / "tuning"
    shard = T.Shard(out / "shards" / "a.sqlite3")
    kw = dict(grids=grids, runs_dir=out / "runs", combos=combos, log=lambda *_: None)
    assert T.run(["dtw", "pelt"], lecs, mats, shard, **kw) == {"dtw": 10, "pelt": 6}
    first = {(r["method"], r["combo"], r["lecture_id"]): r for r in T.read_shard(shard.path)}

    calls = []
    monkeypatch.setattr(T, "run_one", lambda *a, **k: calls.append(a) or {})
    assert T.run(["dtw", "pelt"], lecs, mats, shard, **kw) == {"dtw": 0, "pelt": 0}
    assert calls == []
    shard.close()

    T.merge(out, log=lambda *_: None)
    assert not list((out / "shards").glob("*"))      # the shards are folded and removed
    merged = T.read_runs("dtw", out / "runs") + T.read_runs("pelt", out / "runs")
    assert len(merged) == 16
    for r in merged:                                  # nothing lost on the way
        f = first[(r["method"], r["combo"], r["lecture_id"])]
        assert {k: r[k] for k in T.COLUMN_NAMES} == {k: f[k] for k in T.COLUMN_NAMES}
    raw = (out / "runs" / "dtw.csv.gz").read_bytes()
    T.merge(out, log=lambda *_: None)                 # nothing new: the same bytes
    assert (out / "runs" / "dtw.csv.gz").read_bytes() == raw

    shard2 = T.Shard(out / "shards" / "b.sqlite3")    # a new writer skips the merged runs
    assert T.run(["dtw", "pelt"], lecs, mats, shard2, **kw) == {"dtw": 0, "pelt": 0}


def test_runs_are_deterministic():
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    for name in ("wbs2", "mbr", "bocpd"):
        c = grids[name].combos()[7]
        a, b = T.run_one(c, lecs[1], mats), T.run_one(c, lecs[1], mats)
        assert (a["n_correct"], a["status"]) == (b["n_correct"], b["status"])


class TimedGold(GoldClient):
    """The gold fake with a successful attempt's duration, as OpenAICompatClient returns it."""

    def chat(self, payload):
        return {**super().chat(payload), "duration_s": 0.25, "attempts": 1,
                "failed_attempts": 0, "errors": []}


class Failing:
    def chat(self, payload):
        raise LLMCallFailed("down", attempts=5, errors=["HTTP 503"] * 5)


def fake_llms(tmp_path, client, models=("qwen3-30b-a3b-instruct-2507",)):
    cache = T.MemoCache(tmp_path / "llm.sqlite3")
    return {m: T.TimedLLM(model=m, client=client, cache=cache) for m in models}


def test_llm_cost_paid_and_used_are_kept_apart(tmp_path):
    grids = T.load_grid()
    lecs, mats = fake_lectures(((12, 4),))
    L = lecs[0]
    llms = fake_llms(tmp_path, TimedGold(L.gold))
    c = grids["llm_single_call"].combos()[0]
    first = T.run_one(c, L, mats, llms=llms)
    assert first["status"] == "ok" and first["n_correct"] == 12
    assert (first["calls"], first["used_calls"], first["cache_hits"]) == (1, 1, 0)
    assert first["call_seconds"] == first["used_seconds"] == 0.25
    assert first["prompt_tokens"] == first["used_prompt_tokens"] > 0
    again = T.run_one(c, L, mats, llms=llms)          # answered from the cache
    assert (again["calls"], again["call_seconds"], again["prompt_tokens"]) == (0, 0.0, 0)
    assert (again["used_calls"], again["used_seconds"], again["cache_hits"]) == (1, 0.25, 1)
    assert again["compute_s"] <= again["run_s"]


def test_a_failed_call_is_not_cost_and_is_recorded(tmp_path):
    grids = T.load_grid()
    lecs, mats = fake_lectures(((12, 4),))
    llms = fake_llms(tmp_path, Failing())
    row = T.run_one(grids["llm_monotone"].combos()[0], lecs[0], mats, llms=llms)
    assert row["status"] == "llm_failed" and row["n_correct"] == 0
    assert (row["calls"], row["used_calls"], row["call_seconds"]) == (0, 0, 0.0)
    assert row["failed_calls"] == 1 and json.loads(row["errors"]) == {"HTTP 503": 5}


def _tuned_members():
    return {
        "softdtw": T.Tuned("bm25", None, {"stay_bonus": 0.4, "pos_weight": 4.0},
                           {"gamma": 1.0, "mode": "path"}),
        "cohesion_dp": T.Tuned("embed", None, {"backward_penalty": 1.0},
                               {"window": 6, "strength": 2.0}),
        "needleman_wunsch": T.Tuned("image", None, {"pos_weight": 0.0},
                                    {"gap_segment": 0.8, "gap_slide": 0.3}),
    }


def test_consensus_members_run_at_their_tuned_settings_and_measures():
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    L = lecs[1]
    tuned = _tuned_members()
    c = grids["score_fusion"].combos()[0]
    row = T.run_one(c, L, mats, tuned=tuned)
    members = {n: Member(sim=mats[(t.measure, L.name)], cfg=DPConfig(**t.cfg),
                         settings=t.settings) for n, t in tuned.items()}
    want = run_method("score_fusion", mats[("hybrid", L.name)], member_inputs=members,
                      **dict(c.settings))
    assert row["n_correct"] == T.accuracy(want, L.gold)[0]
    with pytest.raises(SystemExit, match="tuned settings"):
        T.run_one(c, L, mats, tuned={})


def test_tuned_settings_round_trip(tmp_path):
    rows = []
    lect = {("c", "A")}
    g = T.MethodGrid("mbr", "matrix", ("hybrid",), (), {"stay_bonus": (0.1, 0.2)},
                     {"k": (20, 100), "temperature": (None, 1.0)})
    for c in g.combos():
        rows.append(_row("mbr", c, "A", 50 + c.index))
    choices, _ = SEL.select({"mbr": g}, lambda m: rows, lect, "q")
    p = tmp_path / "tuned.toml"
    p.write_text(SEL.tuned_toml(choices), encoding="utf-8")
    t = T.load_tuned(p)["mbr"]
    ch = choices["mbr"].chosen.values
    assert t.measure == "hybrid" and t.cfg == {"stay_bonus": ch[("cfg", "stay_bonus")]}
    assert t.settings == {"k": ch[("settings", "k")],
                          "temperature": ch[("settings", "temperature")]}


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #

def _row(method, combo, lid, n_correct, *, n=1000, status="ok", used_calls=0,
         used_seconds=0.0):
    return {"method": method, "combo": combo.key, "combo_index": combo.index, "corpus": "c",
            "lecture_id": lid, "n_units": n, "n_correct": n_correct, "status": status,
            "used_calls": used_calls, "used_seconds": used_seconds, "used_prompt_tokens": 0,
            "used_completion_tokens": 0, "compute_s": 0.01}


LECS = {("c", "A"), ("c", "B")}


def _select(grid, acc, cost=None):
    """acc: combo index -> accuracy in % (a tenth of a point resolves: 1,000 units);
    both lectures get it."""
    rows = []
    for c in grid.combos():
        a = acc.get(c.index, 10.0)
        extra = (cost or {}).get(c.index, {})
        rows += [_row(grid.name, c, lid, int(round(10 * a)), **extra) for lid in ("A", "B")]
    choices, report = SEL.select({grid.name: grid}, lambda m: rows, LECS, "qwen-default")
    return choices.get(grid.name), report[0]


def _index(grid, **want):
    for c in grid.combos():
        v = {"measure": c.measure, "model": c.model, **dict(c.cfg), **dict(c.settings)}
        if all(v[k] == x for k, x in want.items()):
            return c.index
    raise KeyError(want)


def test_the_tie_rule_counts_combinations_within_half_a_point():
    g = T.MethodGrid("dtw", "matrix", ("hybrid", "bm25"), (),
                     {"stay_bonus": (0.1, 0.2, 0.4)}, {})
    rows = []
    acc = {0: 80.0, 1: 79.6, 2: 79.4, 3: 79.5, 4: 50.0, 5: 79.9}
    for c in g.combos():
        rows += [_row("dtw", c, lid, int(round(acc[c.index] * 10))) for lid in ("A", "B")]
    choices, report = SEL.select({"dtw": g}, lambda m: rows, LECS, "q")
    assert report[0]["n_tied"] == 4                   # 80.0, 79.6, 79.5, 79.9; not 79.4
    assert report[0]["tuning_acc"] in (80.0, 79.6, 79.5, 79.9)


def test_the_cheaper_work_setting_wins_a_tie():
    g = T.MethodGrid("mbr", "matrix", ("hybrid",), (), {}, {"k": (20, 100, 300)})
    ch, rep = _select(g, {0: 70.0, 1: 70.0, 2: 70.4})
    assert ch.chosen.values[("settings", "k")] == 20   # 0.4 points better is a tie
    ch, rep = _select(g, {0: 69.0, 1: 70.0, 2: 70.4})
    assert ch.chosen.values[("settings", "k")] == 100  # 20 is 1.4 points behind


def test_fewer_llm_calls_then_less_time_win_a_tie():
    g = T.MethodGrid("hierarchical", "llm", ("none",), ("qwen-default", "big", "glm"), {},
                     {"use_boundary_refinement": (True, False)})
    on, off = _index(g, model="big", use_boundary_refinement=True), \
        _index(g, model="big", use_boundary_refinement=False)
    cost = {c.index: {"used_calls": 10, "used_seconds": 5.0} for c in g.combos()}
    cost[on] = {"used_calls": 10, "used_seconds": 5.0}
    cost[off] = {"used_calls": 6, "used_seconds": 9.0}
    ch, _ = _select(g, {on: 60.0, off: 59.6}, cost)
    assert ch.chosen.index == off                     # fewer calls, though slower
    cost[off] = {"used_calls": 10, "used_seconds": 4.0}
    ch, _ = _select(g, {on: 60.0, off: 59.6}, cost)
    assert ch.chosen.index == off                     # same calls, less time
    cost[off] = {"used_calls": 10, "used_seconds": 6.0}
    ch, _ = _select(g, {on: 60.0, off: 59.6}, cost)
    assert ch.chosen.index == on
    assert set(ch.per_model) == {"qwen-default", "big", "glm"}


def test_then_the_combination_closest_to_the_defaults_then_the_higher_accuracy():
    g = T.MethodGrid("dtw", "matrix", ("bm25", "hybrid"), (),
                     {"stay_bonus": (0.1, 0.2), "pos_weight": (4.0, 8.0)}, {})
    far = _index(g, measure="bm25", stay_bonus=0.1, pos_weight=4.0)       # 3 away
    near = _index(g, measure="hybrid", stay_bonus=0.1, pos_weight=8.0)    # 1 away
    ch, rep = _select(g, {far: 70.0, near: 69.6})
    assert ch.chosen.index == near and rep["n_tied"] == 2
    one_a = _index(g, measure="bm25", stay_bonus=0.2, pos_weight=8.0)     # 1 away
    ch, _ = _select(g, {far: 70.0, near: 69.6, one_a: 69.8})
    assert ch.chosen.index == one_a                   # equally near: the higher accuracy
    default = _index(g, measure="hybrid", stay_bonus=0.2, pos_weight=8.0)
    ch, rep = _select(g, {far: 70.0, default: 69.6})
    assert ch.chosen.index == default
    assert rep["default_acc"] == rep["tuning_acc"] == 69.6 and rep["gain"] == 0


def test_a_crash_counts_as_zero_and_an_incomplete_grid_is_not_chosen():
    g = T.MethodGrid("dtw", "matrix", ("hybrid", "bm25"), (), {}, {})
    rows = [_row("dtw", c, lid, 900 if c.measure == "bm25" else 800) for c in g.combos()
            for lid in ("A", "B")]
    rows[2] = {**rows[2], "status": "crash", "n_correct": 900}            # bm25 on A
    choices, rep = SEL.select({"dtw": g}, lambda m: rows, LECS, "q")
    assert choices["dtw"].chosen.values[("measure", "measure")] == "hybrid"
    assert rep[0]["not_ok_runs_grid"] == 1
    choices, rep = SEL.select({"dtw": g}, lambda m: rows[:-1], LECS, "q")
    assert "dtw" not in choices and rep[0]["status"].startswith("incomplete")


def test_edge_flags_only_numeric_settings_at_an_end():
    g = T.MethodGrid("drop_dtw", "matrix", ("hybrid", "bm25"), (),
                     {"stay_bonus": (0.1, 0.2, 0.4), "pos_weight": (0.0, 8.0)},
                     {"drop_percentile": (40.0, 60.0, 80.0), "drop_fill": ("carry", "argmax")})
    i = _index(g, measure="bm25", stay_bonus=0.4, pos_weight=0.0, drop_percentile=60.0,
               drop_fill="argmax")
    ch, rep = _select(g, {i: 90.0})
    assert rep["edge"] == "EDGE: stay_bonus=0.4 (largest); pos_weight=0.0 (smallest)"
    i = _index(g, measure="hybrid", stay_bonus=0.2, pos_weight=8.0, drop_percentile=60.0,
               drop_fill="carry")
    ch, rep = _select(g, {i: 90.0})
    assert rep["edge"] == "EDGE: pos_weight=8.0 (largest)"
    g2 = T.MethodGrid("mbr", "matrix", ("hybrid",), (), {}, {"temperature": (None, 1.0)})
    ch, rep = _select(g2, {1: 90.0})
    assert rep["edge"] == ""                           # none / 1.0 is categorical
