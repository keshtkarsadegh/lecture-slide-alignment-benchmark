"""Step 13: the final run (scripts/tune.py --set test --settings ...) and its report
(scripts/final_report.py).

Nothing here reads the real data or calls the API: lectures and matrices are made up,
the selection file and a prepared corpus are written to tmp_path, and the LLM is a fake
client that answers from the true labels. The committed config/tuned_settings.toml is
read, because the final run must be able to cut every grid down to it.
"""

from __future__ import annotations

import math

import pytest

import final_report as F
import select_settings as SEL
import tune as T
from methods import MEMBERS, METHODS

from test_tuning import TimedGold, _record, _row, fake_lectures, fake_llms, tiny_set  # noqa: F401


# --------------------------------------------------------------------------- #
# The test set: only at the tuned settings
# --------------------------------------------------------------------------- #

def test_the_test_set_is_read_only_at_the_tuned_settings(tiny_set):  # noqa: F811
    sel, prepared = tiny_set
    with pytest.raises(SystemExit, match="tuned settings"):
        T.set_lectures("test", sel)
    with pytest.raises(SystemExit):
        T.set_lectures("other", sel, tuned_only=True)
    assert T.set_lectures("test", sel, tuned_only=True) == [("c", "T1")]
    # tuning is unchanged by the flag: still its own lectures only
    assert T.set_lectures("tuning", sel, tuned_only=True) == [("c", "A"), ("c", "B")]
    with pytest.raises(SystemExit, match="refused"):
        T.load_lectures("test", only=["c/A"], selection=sel, prepared=prepared,
                        tuned_only=True)


def test_set_test_without_settings_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)                       # main moves to the repository root
    for cmd in (["run", "--stage", "matrix"], ["plan"], ["merge"], ["status"]):
        with pytest.raises(SystemExit) as e:
            T.main(cmd + ["--set", "test"])
        assert e.value.code == 2
    assert "--settings" in capsys.readouterr().err


def test_each_set_has_its_own_tables_and_output():
    assert T.tables_path("tuning") == T.TABLES
    assert T.tables_path("test") == T.REPO_ROOT / "data" / "similarity" / "test" / "tables.sqlite3"
    assert T.OUT_DIRS == {"tuning": T.OUT, "test": T.REPO_ROOT / "data" / "final"}


# --------------------------------------------------------------------------- #
# One combination per method: the tuned one
# --------------------------------------------------------------------------- #

def test_the_committed_tuned_settings_cut_every_grid_to_one_combination():
    grids = T.load_grid()
    one = T.tuned_grids(T.load_tuned(), grids)
    assert set(one) == set(grids) and len(one) == 31
    for name, g in one.items():
        assert g.n_combos() == 1, name
        assert g.stage == grids[name].stage
        # the one combination is a combination of the tuning grid
        assert g.combos()[0].key in {c.key for c in grids[name].combos()}, name


def test_the_tuned_combination_is_the_one_the_selector_chose(tmp_path):
    g = T.MethodGrid("mbr", "matrix", ("hybrid", "image"), (), {"stay_bonus": (0.1, 0.2)},
                     {"k": (20, 100), "temperature": (None, 1.0)})
    h = T.MethodGrid("hierarchical", "llm", ("none",), ("q",), {},
                     {"use_boundary_refinement": (True, False)})
    rows = {"mbr": [_row("mbr", c, "A", 50 + 7 * c.index % 13) for c in g.combos()],
            "hierarchical": [_row("hierarchical", c, "A", 40 + c.index) for c in h.combos()]}
    choices, _ = SEL.select({"mbr": g, "hierarchical": h}, lambda m: rows[m], {("c", "A")}, "q")
    p = tmp_path / "tuned.toml"
    p.write_text(SEL.tuned_toml(choices), encoding="utf-8")
    one = T.tuned_grids(T.load_tuned(p), {"mbr": g, "hierarchical": h})
    for name in ("mbr", "hierarchical"):
        assert [c.key for c in one[name].combos()] == [choices[name].chosen.key]


def test_tuned_settings_that_do_not_fit_the_grid_are_refused():
    g = T.MethodGrid("dtw", "matrix", ("hybrid", "image"), (), {"stay_bonus": (0.1, 0.2)}, {})
    ok = T.Tuned("image", None, {"stay_bonus": 0.2}, {})
    assert T.tuned_grids({"dtw": ok}, {"dtw": g})["dtw"].combos()[0].key == \
        T.combo_key("image", None, {"stay_bonus": 0.2}, {})
    with pytest.raises(SystemExit, match="no tuned setting"):
        T.tuned_grids({}, {"dtw": g})
    with pytest.raises(SystemExit, match="not among"):
        T.tuned_grids({"dtw": T.Tuned("image", None, {"stay_bonus": 0.3}, {})}, {"dtw": g})
    with pytest.raises(SystemExit, match="not among"):
        T.tuned_grids({"dtw": T.Tuned("bm25", None, {"stay_bonus": 0.2}, {})}, {"dtw": g})
    with pytest.raises(SystemExit, match="axes"):
        T.tuned_grids({"dtw": T.Tuned("image", None, {}, {})}, {"dtw": g})


def test_a_one_combination_run_is_the_tuning_run_of_that_combination(tmp_path):
    """The same `run`: the row of the tuned combination equals tuning's row of it."""
    grids = T.load_grid()
    lecs, mats = fake_lectures()
    c = grids["drop_dtw"].combos()[17]
    t = T.Tuned(c.measure, c.model, dict(c.cfg), dict(c.settings))
    one = T.tuned_grids({"drop_dtw": t}, {"drop_dtw": grids["drop_dtw"]})
    shard = T.Shard(tmp_path / "shards" / "s.sqlite3")
    T.run(["drop_dtw"], lecs, mats, shard, grids=one, runs_dir=tmp_path / "runs",
          log=lambda *_: None)
    got = T.read_shard(shard.path)
    assert [r["combo"] for r in got] == [c.key, c.key]
    for r, lec in zip(sorted(got, key=lambda r: r["lecture_id"]), lecs):
        assert r["n_correct"] == T.run_one(c, lec, mats)["n_correct"]


# --------------------------------------------------------------------------- #
# The report: kinds, measures, build cost
# --------------------------------------------------------------------------- #

def test_kinds():
    kinds = {m: F.kind(m) for m in METHODS}
    assert sorted(m for m, k in kinds.items() if k == "mixed") == \
        ["ensemble", "fusion", "llm_rerank_kbest"]
    assert sorted(m for m, k in kinds.items() if k == "LLM") == \
        ["hierarchical", "llm_monotone", "llm_monotone_constrained", "llm_single_call"]
    assert sum(k == "non-LLM" for k in kinds.values()) == 24


def _tuned_for_measures():
    t = {m: T.Tuned("image", None, {}, {}) for m in METHODS}
    t["hierarchical"] = T.Tuned("none", "q", {}, {})
    for m in ("llm_single_call", "llm_monotone", "llm_monotone_constrained"):
        t[m] = T.Tuned("hybrid", "q", {}, {})
    for m in ("score_fusion", "ensemble", "llm_rerank_kbest"):
        t[m] = T.Tuned("members", None, {}, {})
    t["cohesion_dp"] = T.Tuned("bm25", None, {}, {})
    t["fusion"] = T.Tuned("hybrid_image", "q", {}, {})
    return t


def test_a_method_counts_its_members_measures():
    t = _tuned_for_measures()
    assert F.method_measures("dtw", t) == {"image"}
    assert F.method_measures("hierarchical", t) == set()
    assert F.method_measures("llm_monotone", t) == {"hybrid"}     # the fallback reads it
    assert F.method_measures("score_fusion", t) == {"image", "bm25"}
    assert F.method_measures("ensemble", t) == {"image", "bm25", "hybrid"}
    assert "llm_single_call" in MEMBERS["ensemble"]
    assert F.method_measures("llm_rerank_kbest", t) == {"image", "bm25"}
    assert F.method_measures("fusion", t) == {"hybrid_image"}      # hierarchical: none


def test_build_cost_counts_each_measure_and_its_deps_once():
    deps = {"hybrid": ["embed", "bm25"], "hybrid_image": ["embed", "bm25", "image"],
            "cross_encoder": ["hybrid"]}
    assert F.with_deps({"cross_encoder"}, deps) == {"cross_encoder", "hybrid", "embed", "bm25"}
    measures = F.with_deps({"hybrid_image", "hybrid", "image"}, deps)
    assert measures == {"hybrid_image", "hybrid", "embed", "bm25", "image"}
    costs = {(m, "c", "A"): float(i + 1) for i, m in
             enumerate(["hybrid_image", "hybrid", "embed", "bm25", "image"])}
    rows = [{"corpus": "c", "lecture_id": "A", "accuracy": 0.5, "n_correct": 1, "n_units": 2,
             "status": "ok", "error": "", "compute_s": 0.1,
             **{src: 0 for _, src in F.LLM_COLUMNS}}]
    out = F.lecture_rows(rows, measures, costs, {("c", "A"): 0})
    assert out[0]["build_cost_s"] == 1 + 2 + 3 + 4 + 5


# --------------------------------------------------------------------------- #
# The report: rows, summary, checks
# --------------------------------------------------------------------------- #

def _runner_row(lid, n_correct, *, n=10, status="ok", calls=0, used=0, seconds=0.0,
                combo="k"):
    return {"method": "m", "combo": combo, "corpus": "c", "lecture_id": lid, "n_units": n,
            "n_correct": n_correct, "accuracy": n_correct / n, "status": status,
            "error": "" if status == "ok" else "boom", "compute_s": 0.5,
            "calls": calls, "call_seconds": seconds * (calls > 0), "prompt_tokens": 100 * calls,
            "completion_tokens": 10 * calls, "used_calls": used, "used_seconds": seconds,
            "used_prompt_tokens": 100 * used, "used_completion_tokens": 10 * used,
            "cache_hits": used - calls, "failed_attempts": 1, "failed_calls": 0,
            "truncated": 0, "parse_failed": 0}


def test_the_summary_keeps_paid_and_used_apart_and_scores_like_tuning():
    rows = [_runner_row("A", 5, calls=1, used=2, seconds=3.0),
            _runner_row("B", 10, calls=0, used=2, seconds=4.0),
            _runner_row("C", 9, status="crash")]
    costs = {("image", "c", lid): 2.0 for lid in "ABC"}
    out = F.lecture_rows(rows, {"image"}, costs, {("c", "C"): 0, ("c", "A"): 1, ("c", "B"): 2})
    assert [r["lecture_id"] for r in out] == ["C", "A", "B"]          # selection order
    s = F.summary_row("ensemble", "consensus", {"image"}, out, 1.5, 0.6)
    assert s["kind"] == "mixed" and s["n_lectures"] == 3 and s["n_ok"] == 2
    assert s["test_accuracy"] == pytest.approx(100 * (0.5 + 1.0 + 0.0) / 3)  # crash = 0
    assert s["tuning_accuracy"] == pytest.approx(60.0)
    assert s["build_cost_s_total"] == 3 * 2.0 + 1.5
    assert s["used_calls_total"] == 4 and s["paid_calls_total"] == 1
    assert s["used_seconds_total"] == 7.0 and s["paid_seconds_total"] == 3.0
    assert s["used_calls_per_lecture"] == pytest.approx(4 / 3)
    assert s["failed_attempts_total"] == 3
    assert F.output_columns("dtw") == F.BASE_COLUMNS
    assert "paid_calls" in F.output_columns("ensemble") and "calls" not in F.output_columns("ensemble")


def test_check_complete_finds_every_gap():
    lectures = [("c", "A"), ("c", "B")]
    good = [_runner_row("A", 5), _runner_row("B", 5)]
    assert F.check_complete("m", good, lectures, "k", "k") == []
    assert any("missing" in p for p in F.check_complete("m", good[:1], lectures, "k"))
    assert any("outside" in p for p in
               F.check_complete("m", good + [_runner_row("Z", 1)], lectures, "k"))
    assert any("repeated" in p for p in F.check_complete("m", good + good[:1], lectures, "k"))
    assert any("not ok" in p for p in
               F.check_complete("m", [good[0], _runner_row("B", 0, status="crash")],
                                lectures, "k"))
    nan = [good[0], {**good[1], "accuracy": math.nan}]
    assert any("accuracy" in p for p in F.check_complete("m", nan, lectures, "k"))
    assert any("ran" in p for p in F.check_complete("m", good, lectures, "other"))
    assert any("chosen in tuning" in p for p in F.check_complete("m", good, lectures, "k", "x"))


@pytest.mark.parametrize("method,content,old,echo", [
    ("llm_monotone", '{"starts": [0, 12, 30, 41]}', True, False),
    ("llm_monotone", '{"starts": [0, 3, 9, 12]}', False, False),
    ("llm_monotone", '{"starts": [0, s1, s2, ...]}', False, True),
    ("llm_monotone_constrained", '{"gaps": [12, 18, 0, 7]}', True, False),
    ("llm_monotone_constrained", '{"gaps": [4, 12, 18]}', False, False),
    ("llm_monotone_constrained", '{"gaps": [g0, g1, g2]}', False, True),
    ("llm_single_call", '{"blocks": [{"from": 0, "to": 11, "slide": 0}]}', True, False),
    ("llm_single_call", '{"blocks": [{"from": 0, "to": 4, "slide": 0}, '
                        '{"from": 5, "to": 11, "slide": 2}]}', False, False),
    ("llm_single_call", '{"blocks": [{"from": a, "to": b, "slide": j}, ...]}', False, True),
    ("llm_single_call", '{"blocks": [{"from": 0, "to": 3, "slide": 0}, ...]}', False, True),
    ("hierarchical", '{"section_idx": 2, "page_index": 4}', False, False),
    ("hierarchical", '{"section_idx": <int>, "page_index": <int>}', False, True),
    ("hierarchical", '{"section_idx": 1, "belongs_to": "prev" | "next"}', False, True),
    ("hierarchical", '{"section_idx": 1, "belongs_to": "prev"}', False, False),
    ("llm_rerank_kbest", '{"choice": 1}', False, False),
    ("llm_rerank_kbest", "I think candidate 2 is best", False, False),
])
def test_copy_flags(method, content, old, echo):
    assert F.old_example(method, content) is old
    assert F.echoed(content) is echo


def test_the_replay_reads_the_answers_back_from_the_cache(tmp_path):
    grids = T.load_grid()
    lecs, mats = fake_lectures(((12, 4), (16, 5)))
    llms = fake_llms(tmp_path, None)
    paid = {}
    for L in lecs:                                  # the run: the gold client pays
        live = fake_llms(tmp_path, TimedGold(L.gold))
        c = grids["llm_monotone"].combos()[0]
        paid[("llm_monotone", L.corpus, L.lecture_id)] = T.run_one(c, L, mats, llms=live)
    rec = {m: F.RecordingLLM(model=m, client=None, cache=l.cache) for m, l in llms.items()}
    one = {"llm_monotone": grids["llm_monotone"]}
    answers, problems = F.replay(["llm_monotone"], lecs, mats, one, {}, paid, rec,
                                 log=lambda *_: None)
    assert problems == []
    assert len(answers) == 2 and not any(a["old_example"] or a["echo"] for a in answers)
    # a lecture whose answer is not cached cannot be replayed: a problem, never a call
    lecs2, mats2 = fake_lectures(((9, 3),))
    lecs2[0].lecture_id = "other"
    mats2 = {(m, lecs2[0].name): v for (m, _), v in mats2.items()}
    _, problems = F.replay(["llm_monotone"], lecs2, mats2, one, {}, paid, rec,
                           log=lambda *_: None)
    assert len(problems) == 1 and "crash" in problems[0]
