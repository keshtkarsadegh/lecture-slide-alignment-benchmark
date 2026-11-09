"""Monotone-path methods: the alignment never goes back to an earlier slide.

dtw, multiscale_dtw, needleman_wunsch, gotoh, drop_dtw, anchored_dp, semimarkov,
adaptive_prior; plus `kbest_dtw`, the exact top-k DTW paths that `mbr` and
`llm_rerank_kbest` consume.

All of them score a path by adding up prepared emissions (`common.prepare_emissions`:
z-scored rows minus a positional prior) and the shared transition values of `DPConfig`
(stay bonus, forward penalty), and differ in which paths they allow and what extra moves
cost.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from methods.common import (NEG, NO_SLIDE, DPConfig, check_matrix, fill_drops,
                            prepare_emissions)


# --------------------------------------------------------------------------- #
# DTW and its banded core
# --------------------------------------------------------------------------- #

def dtw_path(E: np.ndarray, cfg: DPConfig, lo: Optional[np.ndarray] = None,
             hi: Optional[np.ndarray] = None) -> Optional[list[int]]:
    """The best monotone path through emission matrix E, inside an optional band.

    Moves from one unit to the next: stay (+stay_bonus), advance one slide
    (-forward_penalty) or two (-2 * forward_penalty). The path may start on any slide at
    -forward_penalty per slide skipped before it, and may end on any slide. With `lo`/`hi`
    only slides lo[i]..hi[i] are allowed at unit i; None when the band admits no path.
    Ties: stay before +1 before +2; the lowest end slide.
    """
    n, m = E.shape
    j = np.arange(m)
    banded = lo is not None
    if banded:
        allowed = (j[None, :] >= np.asarray(lo)[:, None]) & (j[None, :] <= np.asarray(hi)[:, None])
    D = np.full((n, m), NEG)
    back = np.zeros((n, m), dtype=np.int8)
    D[0] = E[0] - cfg.forward_penalty * j
    if banded:
        D[0, ~allowed[0]] = NEG
    a1 = np.full(m, NEG)
    a2 = np.full(m, NEG)
    for i in range(1, n):
        prev = D[i - 1]
        a1[1:] = prev[:-1] - cfg.forward_penalty
        a2[2:] = prev[:-2] - 2 * cfg.forward_penalty
        C = np.stack([prev + cfg.stay_bonus, a1, a2])
        arg = C.argmax(axis=0)
        D[i] = E[i] + C[arg, j]
        back[i] = arg
        if banded:
            D[i, ~allowed[i]] = NEG
    if D[n - 1].max() <= NEG / 2:
        return None
    k = int(D[n - 1].argmax())
    path = [k]
    for i in range(n - 1, 0, -1):
        k -= int(back[i, k])
        path.append(k)
    path.reverse()
    return path


def align_dtw(sim, *, cfg: Optional[DPConfig] = None) -> list[int]:
    """Dynamic time warping (Sakoe and Chiba 1978): the best monotone path.

    Steps: stay, +1 or +2 slides per unit; free start (charged per slide skipped) and
    free end. Reads forward_penalty, stay_bonus, standardize, pos_weight.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    if S.shape[0] == 0:
        return []
    return dtw_path(prepare_emissions(S, cfg), cfg)


def align_multiscale_dtw(sim, *, cfg: Optional[DPConfig] = None, min_size: int = 32,
                         radius: int = 4) -> list[int]:
    """FastDTW (Salvador and Chan 2007): coarsen, solve, project, refine in a band.

    The units are averaged in pairs until at most `min_size` remain; DTW solves that
    coarse matrix; the coarse path is projected to the next finer level, and DTW is solved
    again there, restricted to `radius` slides either side of the projection.

    The refinement is the DTW recursion inside the band, as in FastDTW, not a row-by-row
    choice of the best slide in the band: such a greedy pass never goes back, so one noisy
    row that jumps ahead drags every later row with it (6 % on a clean 400 x 40
    block-diagonal matrix).
    """
    cfg = cfg or DPConfig()
    if min_size < 2:
        raise ValueError(f"min_size must be at least 2, got {min_size}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if n <= min_size:
        return align_dtw(S, cfg=cfg)
    half = n // 2
    coarse = (S[0:2 * half:2] + S[1:2 * half:2]) / 2.0
    coarse_path = align_multiscale_dtw(coarse, cfg=cfg, min_size=min_size, radius=radius)
    proj = np.asarray([coarse_path[min(i // 2, half - 1)] for i in range(n)])
    lo = np.maximum(proj - radius, 0)
    hi = np.minimum(proj + radius, m - 1)
    # the projected coarse path itself moves by at most 2 slides between consecutive fine
    # units, so the band always contains a legal DTW path
    return dtw_path(prepare_emissions(S, cfg), cfg, lo, hi)


# --------------------------------------------------------------------------- #
# Anchored DP (Bertalign's two passes)
# --------------------------------------------------------------------------- #

def _anchor_rows(E: np.ndarray, margin_q: float) -> list[tuple[int, int]]:
    """Pass 1: confident, one-per-slide, mutually reachable anchors (unit, slide).

    A unit qualifies when its best slide beats its runner-up by at least the `margin_q`
    quantile of all units' margins; per slide only the most confident unit is kept; then
    consecutive anchors must be connectable by DTW's moves (never back, at most +2 slides
    per unit) - of two incompatible anchors the less confident one is dropped.
    """
    n, m = E.shape
    if m < 2 or margin_q >= 1.0:
        return []
    top2 = np.argpartition(-E, 1, axis=1)[:, :2]
    v = np.take_along_axis(E, top2, axis=1)
    best = np.where(v[:, 0] >= v[:, 1], top2[:, 0], top2[:, 1])
    margin = np.abs(v[:, 0] - v[:, 1])
    thr = float(np.quantile(margin, margin_q))
    per_slide: dict[int, tuple[float, int]] = {}
    for i in range(n):
        if margin[i] < thr:
            continue
        j = int(best[i])
        if j not in per_slide or margin[i] > per_slide[j][0]:
            per_slide[j] = (float(margin[i]), i)
    anchors = sorted((i, j) for j, (_, i) in per_slide.items())
    kept: list[tuple[int, int]] = []
    for i, j in anchors:
        while kept:
            pi, pj = kept[-1]
            if j >= pj and (j - pj) <= 2 * (i - pi):
                break
            if margin[i] <= margin[pi]:
                i = -1
                break
            kept.pop()
        if i >= 0:
            kept.append((i, j))
    return kept


def align_anchored_dp(sim, *, cfg: Optional[DPConfig] = None,
                      margin_q: float = 0.75) -> list[int]:
    """Bertalign-style anchoring (Liu and Zhu 2023): fix the sure units, DTW between them.

    An anchor unit may only take its own slide; a unit between two anchors only the
    slides between theirs. `margin_q` = how confident a unit must be to anchor (a
    quantile of the top1 - top2 margins); 1.0 means no anchors, i.e. plain dtw. Falls
    back to plain dtw if the band admits no path.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    E = prepare_emissions(S, cfg)
    anchors = _anchor_rows(E, margin_q)
    lo = np.zeros(n, dtype=int)
    hi = np.full(n, m - 1, dtype=int)
    for k, (i, j) in enumerate(anchors):
        lo[i] = hi[i] = j
        nxt_i, nxt_j = anchors[k + 1] if k + 1 < len(anchors) else (n, m - 1)
        lo[i + 1:nxt_i] = j
        hi[i + 1:nxt_i] = nxt_j
    if anchors:
        first_i, first_j = anchors[0]
        hi[:first_i] = first_j
    path = dtw_path(E, cfg, lo, hi)
    return path if path is not None else dtw_path(E, cfg)


# --------------------------------------------------------------------------- #
# Adaptive prior: dtw under a pacing curve read off the matrix
# --------------------------------------------------------------------------- #

def pava(y, w=None) -> np.ndarray:
    """Pool-adjacent-violators (Ayer et al. 1955): nearest non-decreasing sequence (L2)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnt = [], [], []
    for i in range(n):
        vals.append(y[i]); wts.append(w[i]); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2); cnt.append(c1 + c2)
    out = np.empty(n)
    k = 0
    for v, c in zip(vals, cnt):
        out[k:k + c] = v
        k += c
    return out


def pacing_curve(S: np.ndarray, tau: float = 0.05, alpha: float = 0.5,
                 smooth: int = 9) -> np.ndarray:
    """Which slide the talk is on at each unit, estimated from the matrix itself.

    Per row a soft-argmax (temperature `tau`, on z-scored rows) gives the expected slide;
    a moving average of width `smooth` and PAVA make it a smooth non-decreasing curve,
    stretched to span slides 0..m-1; `alpha` blends it with the straight diagonal
    (0 = diagonal only, 1 = curve only).
    """
    n, m = S.shape
    Z = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
    P = np.exp((Z - Z.max(axis=1, keepdims=True)) / max(tau, 1e-6))
    P /= P.sum(axis=1, keepdims=True) + 1e-300
    c = P @ np.arange(m)
    if smooth > 1 and n > smooth:
        k = np.ones(smooth) / smooth
        c = np.convolve(np.pad(c, (smooth // 2, smooth // 2), mode="edge"), k,
                        mode="valid")[:n]
    c = pava(c)
    linear = np.arange(n) * (m - 1) / max(n - 1, 1)
    c = (c - c.min()) / (c.max() - c.min()) * (m - 1) if c.max() > c.min() else linear
    return (1.0 - alpha) * linear + alpha * c


def align_adaptive_prior(sim, *, cfg: Optional[DPConfig] = None, tau: float = 0.05,
                         alpha: float = 0.5, smooth: int = 9) -> list[int]:
    """dtw with the constant-pace positional prior replaced by `pacing_curve`.

    The prior still weighs `cfg.pos_weight`; with pos_weight = 0 the curve has no effect
    and this is plain dtw.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    if S.shape[0] == 0:
        return []
    E = prepare_emissions(S, cfg, expected=pacing_curve(S, tau=tau, alpha=alpha, smooth=smooth))
    return dtw_path(E, cfg)


# --------------------------------------------------------------------------- #
# Needleman-Wunsch and Gotoh: monotone alignment with explicit gaps
# --------------------------------------------------------------------------- #
#
# Both align the unit sequence against the slide sequence. A unit is MATCHED (the first
# unit on a slide), STAYS (another unit on the slide it is on), or is a UNIT GAP (belongs
# to no slide, label NO_SLIDE, -gap_segment); a slide nobody speaks to is a SLIDE GAP.
# The path runs from (0 units, 0 slides) to (all units, all slides): skipped slides at
# the start and the end are paid for like any other.
#
# Both decoders keep a state per cell: whether its slide already has a unit (state O) or
# not (C), and only O may STAY. With one score per cell a STAY could follow a slide gap -
# "skip slide j, then stay on slide j" - and the first unit on a slide would collect the
# stay bonus whenever stay_bonus > gap_slide (0.2 > 0.1 by default). With the states, the
# score of an alignment is exactly what the docstrings say.

_O, _C, _G = 0, 1, 2   # slide has a unit / slide has none / inside a run of slide gaps
_MATCH, _STAY, _UGAP, _SGAP = 1, 2, 3, 4


def align_needleman_wunsch(sim, *, cfg: Optional[DPConfig] = None,
                           gap_segment: float = 0.4, gap_slide: float = 0.1) -> list[int]:
    """needleman_wunsch with its unassigned units filled from their neighbours.

    `needleman_wunsch_alignment` gives the raw alignment; a unit gap (NO_SLIDE) then
    takes the previous unit's slide, a leading run the first assigned slide - the
    "carry" rule of drop_dtw. DECIDED: every unit has a true slide, so a NO_SLIDE label
    could only be wrong.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    raw = needleman_wunsch_alignment(S, cfg=cfg, gap_segment=gap_segment, gap_slide=gap_slide)
    return fill_drops(raw, prepare_emissions(S, cfg), "carry") if raw else raw


def needleman_wunsch_alignment(sim, *, cfg: Optional[DPConfig] = None,
                               gap_segment: float = 0.4, gap_slide: float = 0.1) -> list[int]:
    """Needleman and Wunsch (1970), many-to-one: global alignment with gaps.

    The raw alignment, with NO_SLIDE for unit gaps.
    `gap_segment`: cost of leaving a unit unassigned (label NO_SLIDE); `gap_slide`: cost
    per slide nobody speaks to. Reads stay_bonus, standardize, pos_weight (no
    forward_penalty: a slide change costs nothing, a skipped slide costs gap_slide).
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    E = prepare_emissions(S, cfg)
    V = np.full((2, n + 1, m + 1), NEG)
    ptr = np.zeros((2, n + 1, m + 1), dtype=np.int8)     # move into the cell
    src = np.zeros((2, n + 1, m + 1), dtype=np.int8)     # state it came from
    V[_C, 0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            # state O: slide j has a unit among the first i
            if i >= 1 and j >= 1:
                e = E[i - 1, j - 1]
                pd = (V[_O, i - 1, j - 1], V[_C, i - 1, j - 1])
                d_state = _O if pd[0] >= pd[1] else _C
                cands = ((max(pd) + e, _MATCH, d_state),
                         (V[_O, i - 1, j] + e + cfg.stay_bonus, _STAY, _O),
                         (V[_O, i - 1, j] - gap_segment, _UGAP, _O))
                v, mv, st = max(cands, key=lambda c: c[0])
                V[_O, i, j], ptr[_O, i, j], src[_O, i, j] = v, mv, st
            # state C: slide j has no unit (j = 0 counts as C)
            cands = []
            if j >= 1:
                pl = (V[_O, i, j - 1], V[_C, i, j - 1])
                cands.append((max(pl) - gap_slide, _SGAP, _O if pl[0] >= pl[1] else _C))
            if i >= 1:
                cands.append((V[_C, i - 1, j] - gap_segment, _UGAP, _C))
            v, mv, st = max(cands, key=lambda c: c[0])
            V[_C, i, j], ptr[_C, i, j], src[_C, i, j] = v, mv, st
    out = [NO_SLIDE] * n
    state = _O if V[_O, n, m] >= V[_C, n, m] else _C
    i, j = n, m
    while i > 0 or j > 0:
        mv, nxt = int(ptr[state, i, j]), int(src[state, i, j])
        if mv in (_MATCH, _STAY):
            out[i - 1] = j - 1
        if mv == _MATCH:
            i, j = i - 1, j - 1
        elif mv in (_STAY, _UGAP):
            i -= 1
        else:
            j -= 1
        state = nxt
    return out


def _smax(vals, gamma: float) -> float:
    """gamma * log(sum(exp(v / gamma))): the smoothed max over alternative paths."""
    mx = max(vals)
    if mx <= NEG / 2:
        return NEG
    return mx + gamma * float(np.log(sum(np.exp((v - mx) / gamma) for v in vals)))


def nw_posterior(sim, *, cfg: Optional[DPConfig] = None, gap_segment: float = 0.4,
                 gap_slide: float = 0.1, gamma: float = 0.3) -> np.ndarray:
    """Soft needleman_wunsch: P(unit i on slide j) over all its alignments, each weighted
    by exp(score / gamma), normalised over the slides (the unassigned mass is left out).

    Forward and backward passes over the same two-state lattice as the hard decoder.

    The backward pass scores a STAY from cell (i, j) with the emission of slide j itself,
    so no posterior mass shifts to the next slide (the posteriors feed score_fusion).
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return np.zeros((0, m))
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    E = prepare_emissions(S, cfg)
    sb, g = cfg.stay_bonus, gamma
    FO = np.full((n + 1, m + 1), NEG)
    FC = np.full((n + 1, m + 1), NEG)
    FC[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i >= 1 and j >= 1:
                e = E[i - 1, j - 1]
                FO[i, j] = _smax([_smax([FO[i - 1, j - 1], FC[i - 1, j - 1]], g) + e,
                                  FO[i - 1, j] + e + sb, FO[i - 1, j] - gap_segment], g)
            cands = []
            if j >= 1:
                cands.append(_smax([FO[i, j - 1], FC[i, j - 1]], g) - gap_slide)
            if i >= 1:
                cands.append(FC[i - 1, j] - gap_segment)
            FC[i, j] = _smax(cands, g)
    BO = np.full((n + 1, m + 1), NEG)
    BC = np.full((n + 1, m + 1), NEG)
    BO[n, m] = BC[n, m] = 0.0
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            o, c = [], []
            if i < n and j < m:                       # MATCH: unit i+1 -> slide j+1
                o.append(BO[i + 1, j + 1] + E[i, j])
                c.append(BO[i + 1, j + 1] + E[i, j])
            if i < n and j >= 1:                      # STAY: unit i+1 -> slide j (O only)
                o.append(BO[i + 1, j] + E[i, j - 1] + sb)
            if i < n:                                 # unit gap
                o.append(BO[i + 1, j] - gap_segment)
                c.append(BC[i + 1, j] - gap_segment)
            if j < m:                                 # slide gap
                o.append(BC[i, j + 1] - gap_slide)
                c.append(BC[i, j + 1] - gap_slide)
            BO[i, j] = _smax(o, g)
            BC[i, j] = _smax(c, g)
    L = np.full((n, m), NEG)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e = E[i - 1, j - 1]
            match = _smax([FO[i - 1, j - 1], FC[i - 1, j - 1]], g) + e + BO[i, j]
            stay = FO[i - 1, j] + e + sb + BO[i, j]
            L[i - 1, j - 1] = _smax([match, stay], g)
    L = L / g
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / (P.sum(axis=1, keepdims=True) + 1e-300)


def align_gotoh(sim, *, cfg: Optional[DPConfig] = None, gap_segment: float = 0.4,
                gap_open: float = 0.4, gap_extend: float = 0.05) -> list[int]:
    """gotoh with its unassigned units filled from their neighbours (as
    align_needleman_wunsch; DECIDED for the same reason)."""
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    raw = gotoh_alignment(S, cfg=cfg, gap_segment=gap_segment, gap_open=gap_open,
                          gap_extend=gap_extend)
    return fill_drops(raw, prepare_emissions(S, cfg), "carry") if raw else raw


def gotoh_alignment(sim, *, cfg: Optional[DPConfig] = None, gap_segment: float = 0.4,
                    gap_open: float = 0.4, gap_extend: float = 0.05) -> list[int]:
    """Gotoh (1982): needleman_wunsch with AFFINE slide gaps.

    A run of L consecutive skipped slides costs gap_open + (L - 1) * gap_extend, so
    skipping a whole block (an appendix) is cheap compared with L separate skips. Unit
    gaps stay linear (gap_segment). gap_open == gap_extend == gap_slide gives
    needleman_wunsch.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    E = prepare_emissions(S, cfg)
    V = np.full((3, n + 1, m + 1), NEG)
    ptr = np.zeros((3, n + 1, m + 1), dtype=np.int8)
    src = np.zeros((3, n + 1, m + 1), dtype=np.int8)
    V[_C, 0, 0] = 0.0

    def best_of(states, i, j):
        vals = [V[s, i, j] for s in states]
        k = int(np.argmax(vals))
        return vals[k], states[k]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i >= 1 and j >= 1:
                e = E[i - 1, j - 1]
                pv, ps = best_of((_O, _G, _C), i - 1, j - 1)
                cands = ((pv + e, _MATCH, ps),
                         (V[_O, i - 1, j] + e + cfg.stay_bonus, _STAY, _O),
                         (V[_O, i - 1, j] - gap_segment, _UGAP, _O))
                V[_O, i, j], ptr[_O, i, j], src[_O, i, j] = max(cands, key=lambda c: c[0])[:3]
            if j >= 1:
                ov, os_ = best_of((_O, _C, _G), i, j - 1)
                cands = ((ov - gap_open, _SGAP, os_),               # open (ties: open)
                         (V[_G, i, j - 1] - gap_extend, _SGAP, _G))  # extend
                V[_G, i, j], ptr[_G, i, j], src[_G, i, j] = max(cands, key=lambda c: c[0])[:3]
            if i >= 1:
                cv, cs = best_of((_G, _C), i - 1, j)
                V[_C, i, j], ptr[_C, i, j], src[_C, i, j] = cv - gap_segment, _UGAP, cs
    out = [NO_SLIDE] * n
    _, state = best_of((_O, _G, _C), n, m)
    i, j = n, m
    while i > 0 or j > 0:
        mv, nxt = int(ptr[state, i, j]), int(src[state, i, j])
        if mv in (_MATCH, _STAY):
            out[i - 1] = j - 1
        if mv == _MATCH:
            i, j = i - 1, j - 1
        elif mv in (_STAY, _UGAP):
            i -= 1
        else:
            j -= 1
        state = nxt
    return out


# --------------------------------------------------------------------------- #
# Drop-DTW
# --------------------------------------------------------------------------- #

def align_drop_dtw(sim, *, cfg: Optional[DPConfig] = None, drop_percentile: float = 60.0,
                   slide_drop: Optional[float] = None, drop_fill: str = "carry") -> list[int]:
    """Drop-DTW (Dvornik et al. 2021): DTW in which a unit may be DROPPED.

    A unit is dropped for a score d_i = the `drop_percentile` percentile of its own row
    of prepared emissions (the paper takes one percentile over the whole cost matrix; a
    per-row value needs no common scale across rows). A drop leaves the path where it is.
    Advancing over k slides costs forward_penalty + (k - 1) * slide_drop, with no limit
    on k (slide_drop defaults to forward_penalty). `drop_fill` labels the dropped units:
    "carry" (the slide the path is on), "argmax" (the unit's own best slide) or "none"
    (NO_SLIDE). Free start and end, as dtw.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if drop_fill not in ("carry", "argmax", "none"):
        raise ValueError(f"unknown drop_fill {drop_fill!r}")
    E = prepare_emissions(S, cfg)
    if m == 1:
        return [0] * n
    sd = cfg.forward_penalty if slide_drop is None else float(slide_drop)
    drop_score = np.percentile(E, drop_percentile, axis=1)

    Dm = np.full((n, m), NEG)
    Dd = np.full((n, m), NEG)
    bm = np.zeros((n, m, 2), dtype=np.int32)
    bd = np.zeros((n, m, 2), dtype=np.int32)
    j_idx = np.arange(m)
    Dm[0] = E[0] - cfg.forward_penalty * j_idx
    Dd[0] = drop_score[0] - cfg.forward_penalty * j_idx
    for i in range(1, n):
        prev_best = np.maximum(Dm[i - 1], Dd[i - 1])
        prev_from_m = Dm[i - 1] >= Dd[i - 1]
        stay_m = Dm[i - 1] + cfg.stay_bonus
        stay_d = Dd[i - 1]
        # advance from any j' < j: prev_best[j'] - forward_penalty - sd * (j - j' - 1),
        # a running prefix maximum of prev_best[j'] + sd * j'
        shifted = prev_best + sd * j_idx
        run_max = np.full(m, NEG)
        run_arg = np.zeros(m, dtype=np.int32)
        best, arg = NEG, 0
        for j in range(1, m):
            if shifted[j - 1] > best:
                best, arg = float(shifted[j - 1]), j - 1
            run_max[j] = best
            run_arg[j] = arg
        adv = np.full(m, NEG)
        adv[1:] = run_max[1:] - sd * j_idx[1:] + sd - cfg.forward_penalty
        for j in range(m):
            best_v, best_t, best_j = max(((stay_m[j], 0, j), (stay_d[j], 1, j)))
            if adv[j] > best_v:
                jp = int(run_arg[j])
                best_v, best_t, best_j = float(adv[j]), (0 if prev_from_m[jp] else 1), jp
            Dm[i, j] = E[i, j] + best_v
            bm[i, j] = (best_t, best_j)
            if Dm[i - 1, j] >= Dd[i - 1, j]:
                Dd[i, j] = drop_score[i] + Dm[i - 1, j]
                bd[i, j] = (0, j)
            else:
                Dd[i, j] = drop_score[i] + Dd[i - 1, j]
                bd[i, j] = (1, j)
    end = np.stack([Dm[n - 1], Dd[n - 1]])
    t, j = (int(x) for x in np.unravel_index(int(end.argmax()), end.shape))
    out = [NO_SLIDE] * n
    for i in range(n - 1, -1, -1):
        out[i] = NO_SLIDE if t == 1 else j
        if i == 0:
            break
        t, j = (int(x) for x in (bd if t == 1 else bm)[i, j])
    return fill_drops(out, E, drop_fill)


# --------------------------------------------------------------------------- #
# Semi-Markov (explicit-duration) segmentation
# --------------------------------------------------------------------------- #

#: The width (sigma of log L) of the fixed duration shape: 0.5 puts its mode at 0.78 x
#: the median (sigma 1 would put it near one unit). A constant, not a setting:
#: the decision was a fixed parametric shape with nothing fitted.
DUR_SIGMA = 0.5


def duration_logpmf(max_run: int, ratio: float, sigma: float = DUR_SIGMA) -> np.ndarray:
    """log P(run length = L), L = 1..max_run: a log-normal whose median is `ratio`
    (the lecture's units per slide), discretised and normalised over 1..max_run."""
    L = np.arange(1, max_run + 1, dtype=float)
    logpdf = -np.log(L) - (np.log(L) - np.log(max(ratio, 1e-6))) ** 2 / (2 * sigma ** 2)
    return logpdf - (logpdf.max() + np.log(np.exp(logpdf - logpdf.max()).sum()))


def align_semimarkov(sim, *, cfg: Optional[DPConfig] = None, dur_weight: float = 1.0,
                     max_run: Optional[int] = None,
                     skip_penalty: Optional[float] = None) -> list[int]:
    """Semi-Markov segmentation (Sarawagi and Cohen 2004): runs of units, one slide each.

    The transcript is cut into contiguous runs with strictly increasing slides. A run of
    L units on slide j scores its summed emissions + stay_bonus * (L - 1) +
    dur_weight * log P(L), where P is a fixed log-normal duration shape whose median is
    the lecture's units per slide (n / m, `duration_logpmf`; nothing is fitted). Moving to
    the next slide is free; every slide skipped costs `skip_penalty` (default
    forward_penalty), also before the first run. `max_run` caps a run's length (default
    max(4, ceil(6 * units per slide)), at most n). dur_weight = 0 drops the duration term.

    DECIDED: the duration shape is fixed and comes from the lecture itself, so no model is
    fitted on other lectures and dur_weight always has an effect.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    E = prepare_emissions(S, cfg)
    ratio = n / m
    max_run = int(max_run or min(n, max(4, np.ceil(6 * ratio))))
    if max_run < 1:
        raise ValueError(f"max_run must be at least 1, got {max_run}")
    sp = cfg.forward_penalty if skip_penalty is None else float(skip_penalty)
    cum = np.vstack([np.zeros(m), np.cumsum(E, axis=0)])
    Ls = np.arange(1, max_run + 1)
    logdur = dur_weight * duration_logpmf(max_run, ratio) + cfg.stay_bonus * (Ls - 1)
    jr = np.arange(m)
    R = np.full((n + 1, m), NEG)
    bl = np.zeros((n + 1, m), dtype=int)
    bj = np.full((n + 1, m), -1, dtype=int)
    for i in range(1, n + 1):
        best = np.full(m, NEG)
        best_L = np.zeros(m, dtype=int)
        best_j = np.full(m, -1, dtype=int)
        for li, L in enumerate(Ls):
            a = i - L
            if a < 0:
                break
            emit = cum[i] - cum[a]
            if a == 0:
                cand = emit + logdur[li] - sp * jr
                prev = np.full(m, -1, dtype=int)
            else:
                G = R[a] + sp * jr
                pm = np.maximum.accumulate(G)
                pa = np.maximum.accumulate(np.where(G == pm, jr, -1))
                prev = np.full(m, -1, dtype=int)
                prev[1:] = pa[:-1]
                shifted = np.full(m, NEG)
                shifted[1:] = pm[:-1]
                cand = emit + logdur[li] + shifted - sp * (jr - 1)
                cand = np.where(shifted <= NEG / 2, NEG, cand)
            upd = cand > best
            best = np.where(upd, cand, best)
            best_L = np.where(upd, L, best_L)
            best_j = np.where(upd, prev, best_j)
        R[i], bl[i], bj[i] = best, best_L, best_j
    out = [0] * n
    i, j = n, int(R[n].argmax())
    while i > 0 and j >= 0:
        L = int(bl[i, j]) or 1
        for t in range(max(0, i - L), i):
            out[t] = j
        nj = int(bj[i, j])
        i -= L
        j = nj
    return out


# --------------------------------------------------------------------------- #
# The k best DTW paths
# --------------------------------------------------------------------------- #

def kbest_dtw(sim, k: int, cfg: Optional[DPConfig] = None) -> tuple[np.ndarray, np.ndarray]:
    """The exact top-k paths under dtw's scoring, best first: (paths (K x n), scores).

    paths[0] is dtw's own answer (up to ties). K < k when fewer paths exist. Every cell
    keeps its k best partial paths (stay / +1 / +2 from the k best of each predecessor).

    The back-pointers are int32: uint8 ones would silently give wrong paths for k > 255.
    """
    cfg = cfg or DPConfig()
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros(0)
    E = prepare_emissions(S, cfg)
    stay, fwd = cfg.stay_bonus, cfg.forward_penalty
    D = np.full((m, k), NEG)
    D[:, 0] = E[0] - fwd * np.arange(m)
    MOVE = np.zeros((n, m, k), dtype=np.int32)
    SLOT = np.zeros((n, m, k), dtype=np.int32)
    pad = np.full((1, k), NEG)
    for i in range(1, n):
        c_stay = D + stay
        c_adv = np.concatenate([pad, D[:-1]], axis=0) - fwd
        c_skip = (np.concatenate([pad, pad, D[:-2]], axis=0)[:m] - 2 * fwd
                  if m >= 2 else np.full((m, k), NEG))
        cand = np.concatenate([c_stay, c_adv, c_skip], axis=1)
        part = np.argpartition(-cand, kth=min(k - 1, cand.shape[1] - 1), axis=1)[:, :k]
        vals = np.take_along_axis(cand, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        part = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        MOVE[i] = part // k
        SLOT[i] = part % k
        D = E[i][:, None] + vals
    flat = D.reshape(-1)
    paths, scores = [], []
    for t in np.argsort(-flat, kind="stable")[:k]:
        if flat[t] <= NEG / 2:
            break
        j, s = int(t // k), int(t % k)
        lab = np.empty(n, dtype=np.int32)
        lab[n - 1] = j
        for i in range(n - 1, 0, -1):
            mv, s = int(MOVE[i, j, s]), int(SLOT[i, j, s])
            j -= mv
            lab[i - 1] = j
        paths.append(lab)
        scores.append(float(flat[t]))
    return np.stack(paths), np.asarray(scores)


__all__ = ["dtw_path", "align_dtw", "align_multiscale_dtw", "align_anchored_dp",
           "pava", "pacing_curve", "align_adaptive_prior", "align_needleman_wunsch",
           "nw_posterior", "needleman_wunsch_alignment", "align_gotoh", "gotoh_alignment",
           "align_drop_dtw", "DUR_SIGMA", "duration_logpmf",
           "align_semimarkov", "kbest_dtw"]
