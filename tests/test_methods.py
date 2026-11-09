"""Step 3: the matrix methods and the registry, on small made-up matrices.

Three kinds of test:
  * the contract, for every method: right length, labels in range, deterministic, an
    easy case solved, and no crash on edge cases (one slide, more slides than units,
    one unit, no units, NaN);
  * exactness: on matrices small enough to enumerate every alignment, the path
    decoders return the optimum of the score their docstring states, and the posterior
    decoders the exact marginals;
  * one test per pitfall of a method (the number in some docstrings is what the method
    gives when it falls into it).
The LLM methods are in test_llm_methods.py; the contract tests here include them,
answered by a fake client.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from methods import METHODS, DPConfig, gold_labels, lecture_texts, run_method, to_slide_index
from methods.common import depth_scores, prepare_emissions, transition_scores
from methods.common import fill_drops
from methods.monotone import (duration_logpmf, gotoh_alignment, kbest_dtw,
                              needleman_wunsch_alignment, nw_posterior)
from methods.soft_order import forward_backward, rank_transform, softdtw_posterior

from test_llm_methods import gold_llm  # a fake LLM that answers from the gold labels

CFG = DPConfig()

BRIEF = {  # the 31 methods by group (step 3; wbs dropped by decision)
    "independent": ["argmax"],
    "monotone": ["dtw", "multiscale_dtw", "needleman_wunsch", "gotoh", "drop_dtw",
                 "anchored_dp", "semimarkov", "adaptive_prior"],
    "back": ["dp", "rank_dp", "cohesion_dp", "jump_dtw", "forward_backward", "softdtw"],
    "boundaries": ["pelt", "bocpd", "wbs2", "texttiling", "texttiling_dp"],
    "transport": ["opw", "ot_unbalanced"],
    "consensus": ["ensemble", "score_fusion", "mbr"],
    "llm_text": ["llm_single_call", "llm_monotone", "hierarchical"],
    "llm_matrix": ["llm_rerank_kbest", "fusion", "llm_monotone_constrained"],
}
ALL = list(METHODS)
MONOTONE = ["dtw", "multiscale_dtw", "needleman_wunsch", "gotoh", "drop_dtw", "anchored_dp",
            "semimarkov", "adaptive_prior", "pelt", "bocpd", "wbs2", "texttiling",
            "texttiling_dp", "llm_monotone", "llm_monotone_constrained", "hierarchical"]


def block(n_per, m, noise, seed=0):
    """A lecture whose unit i is about slide i // n_per: that column scores +1."""
    rng = np.random.default_rng(seed)
    gold = np.repeat(np.arange(m), n_per)
    S = rng.normal(0, noise, (len(gold), m))
    S[np.arange(len(gold)), gold] += 1.0
    return S, gold


def run(name, S, gold=None, **kw):
    """Run any method; the LLM ones get a fake model that knows `gold`."""
    if METHODS[name].uses_llm:
        g = gold if gold is not None else np.minimum(np.arange(S.shape[0]) * S.shape[1]
                                                     // max(S.shape[0], 1), S.shape[1] - 1)
        return run_method(name, S, slide_texts=[f"slide {j}" for j in range(S.shape[1])],
                          unit_texts=[f"unit {i}" for i in range(S.shape[0])],
                          llm=gold_llm(g), **kw)
    return run_method(name, S, **kw)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

def test_registry_has_the_31_methods_in_their_groups():
    assert len(METHODS) == 31
    assert {g: [n for n, m in METHODS.items() if m.group == g] for g in BRIEF} == BRIEF


def test_settings_are_the_function_defaults_and_unknown_settings_are_rejected():
    assert METHODS["dtw"].settings == {}
    assert METHODS["needleman_wunsch"].settings == {"gap_segment": 0.4, "gap_slide": 0.1}
    S, _ = block(4, 3, 0.1)
    with pytest.raises(ValueError, match="no setting"):
        run_method("dtw", S, radius=3)
    assert run_method("needleman_wunsch", S, gap_slide=0.5) == run_method("needleman_wunsch", S)


def test_prepared_record_helpers():
    rec = {"lecture_id": "x",
           "slides": [{"index": 2, "text": "b"}, {"index": 1, "text": "a"}],
           "units": [{"index": 1, "text": "u1", "true_slide": 2},
                     {"index": 0, "text": "u0", "true_slide": 1}]}
    assert lecture_texts(rec) == (["a", "b"], ["u0", "u1"])
    assert gold_labels(rec) == [0, 1]
    assert to_slide_index([0, 1, -1]) == [1, 2, -1]


# --------------------------------------------------------------------------- #
# The contract, every method
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL)
def test_easy_case_is_solved(name):
    S, gold = block(8, 5, 0.1, seed=1)
    acc = np.mean(np.asarray(run(name, S, gold)) == gold)
    assert acc >= 0.9, f"{name}: {acc:.2f}"


@pytest.mark.parametrize("name", ALL)
def test_length_range_and_determinism(name):
    S = np.random.default_rng(2).normal(size=(17, 6))
    a, b = run(name, S), run(name, S)
    assert a == b
    assert len(a) == 17 and all(isinstance(x, int) and -1 <= x < 6 for x in a)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", [(9, 1), (3, 10), (1, 4), (1, 1), (2, 2)])
def test_edge_shapes(name, shape):
    S = np.random.default_rng(3).normal(size=shape)
    out = run(name, S)
    assert len(out) == shape[0] and all(-1 <= x < shape[1] for x in out)
    if shape[1] == 1:
        assert set(out) <= {0, -1}


@pytest.mark.parametrize("name", ALL)
def test_no_units_and_bad_input(name):
    assert run(name, np.zeros((0, 4))) == []
    bad = np.ones((4, 3))
    bad[1, 2] = np.nan
    with pytest.raises(ValueError):
        run(name, bad)
    with pytest.raises(ValueError):
        run(name, np.ones((4, 0)))


@pytest.mark.parametrize("name", MONOTONE)
def test_monotone_methods_never_go_back(name):
    for seed in range(4):
        out = [x for x in run(name, np.random.default_rng(seed).normal(size=(25, 7))) if x >= 0]
        assert all(b >= a for a, b in zip(out, out[1:])), name


# --------------------------------------------------------------------------- #
# Exactness on enumerable cases
# --------------------------------------------------------------------------- #

def small_cases(count, seed, n_max=5, m_max=4):
    for t in range(count):
        rng = np.random.default_rng(seed + t)
        n, m = int(rng.integers(1, n_max + 1)), int(rng.integers(1, m_max + 1))
        yield rng.normal(size=(n, m)) * 2


def path_score(E, lab, step):
    """sum of emissions + step(prev, cur) per transition; None if a step is illegal."""
    s = E[0, lab[0]]
    for i in range(1, len(lab)):
        t = step(lab[i - 1], lab[i])
        if t is None:
            return None
        s += E[i, lab[i]] + t
    return s


def dtw_step(a, b):
    d = b - a
    return CFG.stay_bonus if d == 0 else (-CFG.forward_penalty * d if d in (1, 2) else None)


def brute(E, scorer):
    n, m = E.shape
    vals = [(scorer(lab), lab) for lab in itertools.product(range(m), repeat=n)]
    return max(v for v, _ in vals if v is not None)


def test_dtw_is_optimal():
    for S in small_cases(40, 100):
        E = prepare_emissions(S, CFG)
        sc = lambda lab: (None if path_score(E, lab, dtw_step) is None
                          else path_score(E, lab, dtw_step) - CFG.forward_penalty * lab[0])
        assert sc(tuple(run_method("dtw", S))) == pytest.approx(brute(E, sc))


def test_dp_and_jump_dtw_are_optimal():
    for S in small_cases(40, 200):
        E = prepare_emissions(S, CFG)
        T = transition_scores(S.shape[1], CFG)
        sc = lambda lab: path_score(E, lab, lambda a, b: T[a, b])
        assert sc(tuple(run_method("dp", S))) == pytest.approx(brute(E, sc))

        def jstep(a, b):
            return dtw_step(a, b) if b >= a else (-CFG.backward_penalty * (a - b) if a - b <= 3 else None)
        sj = lambda lab: (None if path_score(E, lab, jstep) is None
                          else path_score(E, lab, jstep) - CFG.forward_penalty * lab[0])
        assert sj(tuple(run_method("jump_dtw", S))) == pytest.approx(brute(E, sj))


def test_pelt_is_optimal():
    for S in small_cases(40, 300):
        E = prepare_emissions(S, CFG)
        sc = lambda lab: path_score(E, lab, lambda a, b: None if b < a else (0.0 if a == b else -1.0))
        assert sc(tuple(run_method("pelt", S))) == pytest.approx(brute(E, sc))


def test_semimarkov_is_optimal():
    for S in small_cases(40, 400):
        n, m = S.shape
        E = prepare_emissions(S, CFG)
        max_run = int(min(n, max(4, np.ceil(6 * n / m))))

        def sc(lab):
            if any(b < a for a, b in zip(lab, lab[1:])):
                return None
            runs = [len(list(g)) for _, g in itertools.groupby(lab)]
            if max(runs) > max_run:
                return None
            slides = [k for k, _ in itertools.groupby(lab)]
            skipped = slides[0] + sum(b - a - 1 for a, b in zip(slides, slides[1:]))
            logdur = duration_logpmf(max_run, n / m)
            return (sum(E[i, l] for i, l in enumerate(lab)) + CFG.stay_bonus * (n - len(runs))
                    - CFG.forward_penalty * skipped + sum(logdur[r - 1] for r in runs))
        assert sc(tuple(run_method("semimarkov", S))) == pytest.approx(brute(E, sc))


def nw_score(E, lab, slide_gap_cost, gap_segment=0.4):
    """The stated score of a gapped alignment: emissions + stay_bonus for every unit
    that is not the first on its slide - gap_segment per unassigned unit - the cost of
    the slides no unit is on."""
    s, used, prev = 0.0, set(), -1
    for i, l in enumerate(lab):
        if l < 0:
            s -= gap_segment
            continue
        if l < prev:
            return None
        s += E[i, l] + (CFG.stay_bonus if l in used else 0.0)
        used.add(l)
        prev = l
    return s - slide_gap_cost(sorted(set(range(E.shape[1])) - used))


def affine(open_, extend):
    def cost(unused):
        runs = [len(list(g)) for _, g in itertools.groupby(
            enumerate(unused), key=lambda t: t[1] - t[0])]
        return sum(open_ + extend * (r - 1) for r in runs)
    return cost


@pytest.mark.parametrize("name,cost", [("needleman_wunsch", lambda u: 0.1 * len(u)),
                                       ("gotoh", affine(0.4, 0.05))])
def test_gap_aligners_are_optimal(name, cost):
    raw = {"needleman_wunsch": needleman_wunsch_alignment, "gotoh": gotoh_alignment}[name]
    """A STAY may not follow a slide gap (the first unit on a slide must not collect
    the stay bonus); a recursion with one score per cell misses the optimum on 4 of these
    30 cases (needleman_wunsch) and on 1 (gotoh)."""
    for t in list(range(30)):
        rng = np.random.default_rng(1000 + t)
        n, m = int(rng.integers(2, 6)), int(rng.integers(2, 5))
        S = rng.normal(size=(n, m)) * 2
        E = prepare_emissions(S, CFG)
        best = max(v for lab in itertools.product(range(-1, m), repeat=n)
                   if (v := nw_score(E, lab, cost)) is not None)
        assert nw_score(E, raw(S), cost) == pytest.approx(best), (name, t)


@pytest.mark.parametrize("name,raw", [("needleman_wunsch", needleman_wunsch_alignment),
                                      ("gotoh", gotoh_alignment)])
def test_unassigned_units_are_filled_from_their_neighbours(name, raw):
    """DECIDED: a unit gap takes the previous unit's slide (a leading one the first
    assigned slide), never -1."""
    S = np.random.default_rng(0).normal(size=(12, 5))
    gaps = raw(S)
    assert -1 in gaps
    out = run_method(name, S)
    assert -1 not in out
    assert out == fill_drops(gaps, prepare_emissions(S, CFG), "carry")


def test_gotoh_with_flat_gaps_is_needleman_wunsch():
    for S in small_cases(30, 500, n_max=7, m_max=5):
        assert (run_method("gotoh", S, gap_open=0.1, gap_extend=0.1)
                == run_method("needleman_wunsch", S))


def enumerate_nw_paths(E, gs=0.4, gsl=0.1):
    n, m = E.shape
    out = []

    def rec(i, j, state, score, lab):
        if i == n and j == m:
            out.append((score, lab))
            return
        if i < n and j < m:
            rec(i + 1, j + 1, "O", score + E[i, j], lab + [j])
        if i < n and j >= 1 and state == "O":
            rec(i + 1, j, "O", score + E[i, j - 1] + CFG.stay_bonus, lab + [j - 1])
        if i < n:
            rec(i + 1, j, state, score - gs, lab + [-1])
        if j < m:
            rec(i, j + 1, "C", score - gsl, lab)
    rec(0, 0, "C", 0.0, [])
    return out


def marginals(weighted, n, m):
    A = np.zeros((n, m))
    top = max(s for s, _ in weighted)
    for s, lab in weighted:
        for i, l in enumerate(lab):
            if l >= 0:
                A[i, l] += np.exp(s - top)
    return A / A.sum(axis=1, keepdims=True)


def test_nw_posterior_is_exact():
    """The backward pass scores a STAY with the slide's own emission; with the next
    slide's, the marginals are off by up to 0.53."""
    for S in small_cases(25, 600, n_max=4, m_max=3):
        E = prepare_emissions(S, CFG)
        paths = [(s / 0.3, lab) for s, lab in enumerate_nw_paths(E)]
        np.testing.assert_allclose(nw_posterior(S), marginals(paths, *S.shape), atol=1e-9)


def test_softdtw_and_forward_backward_posteriors_are_exact():
    for S in small_cases(25, 700, n_max=4, m_max=3):
        n, m = S.shape
        E = prepare_emissions(S, CFG)
        paths = []
        for lab in itertools.product(range(m), repeat=n):
            sc = path_score(E, lab, dtw_step)
            if sc is not None:
                paths.append(((sc - CFG.forward_penalty * lab[0]) / 0.2, lab))
        np.testing.assert_allclose(softdtw_posterior(S), marginals(paths, n, m), atol=1e-9)
        # the HMM: softmax emissions, row-normalised exp(transition scores), uniform start
        B = np.exp(E - E.max(axis=1, keepdims=True))
        B /= B.sum(axis=1, keepdims=True)
        T = np.exp(transition_scores(m, CFG))
        T /= T.sum(axis=1, keepdims=True)
        A = np.zeros((n, m))
        for lab in itertools.product(range(m), repeat=n):
            p = np.prod([B[i, l] for i, l in enumerate(lab)]) * np.prod(
                [T[a, b] for a, b in zip(lab, lab[1:])])
            A[np.arange(n), lab] += p
        np.testing.assert_allclose(forward_backward(S), A / A.sum(axis=1, keepdims=True),
                                   atol=1e-9)


def test_kbest_paths_are_the_exact_top_k():
    for S in small_cases(20, 800, n_max=5, m_max=4):
        n, m = S.shape
        E = prepare_emissions(S, CFG)
        allsc = []
        for lab in itertools.product(range(m), repeat=n):
            v = path_score(E, lab, dtw_step)
            if v is not None:
                allsc.append(v - CFG.forward_penalty * lab[0])
        allsc.sort(reverse=True)
        paths, scores = kbest_dtw(S, 7)
        np.testing.assert_allclose(scores, allsc[:7], atol=1e-9)
        assert list(paths[0]) == run_method("dtw", S) or scores[0] == pytest.approx(scores[1])


def test_kbest_beyond_255_paths():
    """k > 255: uint8 back-pointers would give wrong paths (here 72 of 400 paths
    without their reported score, and 72 duplicates)."""
    S = np.random.default_rng(9).normal(size=(20, 5))
    E = prepare_emissions(S, CFG)
    paths, scores = kbest_dtw(S, 400)
    assert len({tuple(p) for p in paths}) == 400
    for p, s in zip(paths, scores):
        assert path_score(E, tuple(p), dtw_step) - CFG.forward_penalty * p[0] == pytest.approx(s)


# --------------------------------------------------------------------------- #
# The other fixed bugs
# --------------------------------------------------------------------------- #

def acc(name, S, gold, **kw):
    return float(np.mean(np.asarray(run_method(name, S, **kw)) == gold))


def test_multiscale_dtw_refines_with_dtw_not_a_greedy_ratchet():
    """A per-row greedy fine pass that never goes back gives 6 % here."""
    S, gold = block(10, 40, 0.3)
    assert acc("multiscale_dtw", S, gold) >= 0.95
    assert acc("multiscale_dtw", S, gold) == pytest.approx(acc("dtw", S, gold), abs=0.02)


def test_wbs2_finds_boundaries_in_both_halves():
    """Prior-free contrast + breadth-first recursion (with the prior and depth-first:
    6 %, 7 %)."""
    for seed, noise in ((0, 0.3), (1, 0.6)):
        S, gold = block(10 if seed == 0 else 25, 40 if seed == 0 else 20, noise, seed)
        out = np.asarray(run_method("wbs2", S))
        assert np.mean(out == gold) >= 0.9
        assert np.mean((out == gold)[len(gold) // 2:]) >= 0.9


@pytest.mark.parametrize("name", ["texttiling", "texttiling_dp"])
def test_texttiling_boundaries_follow_the_transcript_not_the_prior(name):
    """Cohesion on prior-free rows, Hearst's nearest-peak depth (with the prior and the
    global-maximum depth: 48 % and 10 % here)."""
    S, gold = block(10, 40, 0.2)
    assert acc(name, S, gold) >= 0.95


def test_boundary_statistics_ignore_the_positional_prior():
    from methods.boundaries import _tiling_depth
    S = np.random.default_rng(4).normal(size=(30, 6))
    np.testing.assert_allclose(_tiling_depth(S, DPConfig(pos_weight=0.0), 3),
                               _tiling_depth(S, DPConfig(pos_weight=50.0), 3))


def test_depth_climbs_to_the_nearest_peaks():
    """Hearst: gap 1 lies between peaks 1.0 (left) and 0.8 (right): depth 0.5 + 0.3.
    Measured against the global maximum it would be 0.5 + 0.5."""
    np.testing.assert_allclose(depth_scores([1.0, 0.5, 0.8, 0.2, 1.0]),
                               [0.0, 0.8, 0.0, 1.4, 0.0])


def test_rank_transform_gives_ties_the_same_rank():
    """Argsort ranks would turn a row of equal scores into a ramp favouring the last
    slide; rank_dp on an all-zero matrix would then differ from dp ([0,1,2,3,3,4] vs
    [0,1,2,2,3,4])."""
    np.testing.assert_allclose(rank_transform(np.array([[0.0, 0, 0, 0], [1, 3, 3, 2]])),
                               [[0.5, 0.5, 0.5, 0.5], [0, 5 / 6, 5 / 6, 1 / 3]])
    Z = np.zeros((6, 5))
    assert run_method("rank_dp", Z) == run_method("dp", Z)


def test_cohesion_strength_outside_its_range_is_rejected():
    S, _ = block(5, 4, 0.1)
    with pytest.raises(ValueError, match="strength"):
        run_method("cohesion_dp", S, strength=2.5)


# --------------------------------------------------------------------------- #
# The decisions on the open questions (docs/METHODS.md, "decided")
# --------------------------------------------------------------------------- #

def test_argmax_has_no_threshold_setting():
    """DECIDED: no no_slide_threshold setting (it could only produce -1)."""
    S, _ = block(5, 4, 0.3)
    with pytest.raises(ValueError, match="no setting"):
        run_method("argmax", S, no_slide_threshold=0.5)


def test_semimarkov_dur_weight_now_matters():
    """DECIDED: a fixed log-normal duration shape, so dur_weight has an effect."""
    S = np.random.default_rng(5).normal(size=(30, 5))
    outs = {tuple(run_method("semimarkov", S, dur_weight=w)) for w in (0.0, 1.0, 10.0)}
    assert len(outs) == 3
    p = np.exp(duration_logpmf(20, 4.0))
    assert p.sum() == pytest.approx(1.0) and 2 <= int(p.argmax()) + 1 <= 4


def test_bocpd_labels_never_step_back():
    """DECIDED: trace back the best final hypothesis; an online read-out steps back on
    this matrix ([..., 2, 4, 3, ...])."""
    out = run_method("bocpd", np.random.default_rng(1).normal(size=(20, 6)))
    assert all(b >= a for a, b in zip(out, out[1:]))


def test_wbs_is_dropped():
    assert "wbs" not in METHODS and "wbs2" in METHODS
