"""Boundary-first methods: find where the slide changes, then label the runs in order.

pelt, bocpd, wbs2, texttiling, texttiling_dp. (wbs, the one-pass version of wbs2, was
dropped by decision: each random interval proposed only one split.)

pelt and bocpd decide boundaries and labels together, unit by unit. The other three first
place boundaries from a statistic of the transcript (cohesion dips, change-point
contrasts) and then give each run a slide, never going back.

wbs2, texttiling, texttiling_dp: the boundary statistic uses the z-scored rows alone,
not the prepared emissions, which include the positional prior. The prior depends only
on the row index, so it would dominate the statistic - every split of a long interval
would look best near its middle, every block pair cohesive - and the boundaries would
follow the prior instead of the transcript (6-16 % on a clean 400 x 40 block-diagonal
matrix); the runs are still labelled with the prepared
emissions, prior included.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from methods.common import (NEG, DPConfig, block_cohesion, check_matrix, depth_scores,
                            prepare_emissions, standardize_rows)
from methods.monotone import align_dtw


# --------------------------------------------------------------------------- #
# Labelling runs
# --------------------------------------------------------------------------- #

def label_runs_greedy(E: np.ndarray, bounds: list[int]) -> list[int]:
    """Runs labelled left to right: each takes its best slide (by mean emission) at or
    after the slide following the previous run's (the last slide once the deck is used
    up)."""
    n, m = E.shape
    out = [0] * n
    slide = 0
    for s, e in zip([0] + bounds, bounds + [n]):
        j = int(E[s:e].mean(axis=0)[slide:].argmax()) + slide
        out[s:e] = [j] * (e - s)
        slide = min(j + 1, m - 1)
    return out


def label_runs_dp(E: np.ndarray, bounds: list[int]) -> list[int]:
    """Runs labelled with strictly increasing slides (skips allowed), maximising the sum
    of the runs' mean emissions. Needs at most m runs."""
    n, m = E.shape
    starts, ends = [0] + bounds, bounds + [n]
    segs = [E[s:e].mean(axis=0) for s, e in zip(starts, ends)]
    ns = len(segs)
    f = np.full((ns, m), NEG)
    back = np.full((ns, m), -1, dtype=int)
    f[0] = segs[0]
    for i in range(1, ns):
        run = np.maximum.accumulate(f[i - 1])
        arg = np.zeros(m, dtype=int)
        best, bi = NEG, 0
        for j in range(m):
            if f[i - 1][j] > best:
                best, bi = f[i - 1][j], j
            arg[j] = bi
        f[i, 1:] = run[:-1] + segs[i][1:]
        back[i, 1:] = arg[:-1]
    out = [0] * n
    j = int(f[ns - 1].argmax())
    for i in range(ns - 1, -1, -1):
        out[starts[i]:ends[i]] = [j] * (ends[i] - starts[i])
        j = int(max(0, back[i][j]))
    return out


def best_k_boundaries(score: np.ndarray, k: int, offset: int, n: int,
                      min_gap: int) -> list[int]:
    """Exactly k (or as many as fit) boundaries maximising the summed score, at least
    `min_gap` apart; score[c] belongs to boundary c + offset."""
    c = len(score)
    if k <= 0 or c == 0:
        return []
    k = min(k, c)
    f = np.full((k + 1, c), NEG)
    back = np.full((k + 1, c), -1, dtype=int)
    f[1] = score
    for j in range(2, k + 1):
        for i in range(c):
            lo = i - min_gap
            if lo < 0:
                continue
            b = int(f[j - 1][:lo + 1].argmax())
            if f[j - 1][b] <= NEG / 2:
                continue
            f[j][i] = f[j - 1][b] + score[i]
            back[j][i] = b
    j = k
    while j > 0 and f[j].max() <= NEG / 2:
        j -= 1
    out: list[int] = []
    i = int(f[j].argmax()) if j > 0 else -1
    while j > 0 and i >= 0:
        out.append(i + offset)
        i = back[j][i]
        j -= 1
    return sorted(b for b in out if 0 < b < n)


# --------------------------------------------------------------------------- #
# PELT and BOCPD
# --------------------------------------------------------------------------- #

def align_pelt(sim, *, cfg: Optional[DPConfig] = None, change_penalty: float = 1.0) -> list[int]:
    """Optimal segmentation with a cost per change (Killick et al. 2012), monotone.

    Maximises the summed prepared emissions minus `change_penalty` per slide change; a
    change may jump any number of slides forward at the same price (dp charges per slide
    instead). Exact DP; PELT's pruning is not needed at this size.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    E = prepare_emissions(S, cfg)
    D = np.full((n, m), NEG)
    back = np.zeros((n, m), dtype=int)
    D[0] = E[0]
    for i in range(1, n):
        best_prev, best_idx = NEG, 0
        for j in range(m):
            if D[i - 1, j] > best_prev:
                best_prev, best_idx = D[i - 1, j], j
            stay = D[i - 1, j]
            change = best_prev - change_penalty
            if stay >= change:
                D[i, j], back[i, j] = E[i, j] + stay, j
            else:
                D[i, j], back[i, j] = E[i, j] + change, best_idx
    path = [int(D[n - 1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return path


def align_bocpd(sim, *, cfg: Optional[DPConfig] = None, hazard: float = 0.05,
                max_hypotheses: int = 64) -> list[int]:
    """Bayesian online change-point detection (Adams and MacKay 2007), adapted.

    Run-length hypotheses, each committed to one slide. At every unit each hypothesis
    either grows (log(1 - hazard) + the unit's emission on its slide) or, from the best
    hypothesis only, a change starts a new run (log hazard + emission) on the unit's best
    slide at or after the current one - the next slide when that best is the current
    one. The `max_hypotheses` most probable are kept. The labels are read by tracing
    the most probable FINAL hypothesis back to the first unit, so they never go back.

    DECIDED: the labels are not an online read-out (the most probable hypothesis at each
    unit), which steps back whenever a younger hypothesis overtakes an older one.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if not 0.0 < hazard < 1.0:
        raise ValueError(f"hazard must be in (0, 1), got {hazard}")
    E = prepare_emissions(S, cfg)
    lgrow, lchange = np.log(1.0 - hazard), np.log(hazard)
    logp = {0: 0.0}
    slide_of = {0: 0}
    parents: list[dict[int, tuple[int, int]]] = []   # per unit: hypothesis -> (parent, slide)
    for i in range(n):
        new_logp: dict[int, float] = {}
        new_slide: dict[int, int] = {}
        parent: dict[int, int] = {}
        for r, lp in logp.items():
            j = slide_of[r]
            new_logp[r + 1] = lp + E[i, j] + lgrow
            new_slide[r + 1] = j
            parent[r + 1] = r
        best_r = max(logp, key=lambda r: logp[r])
        start = slide_of[best_r]
        j = int(E[i, start:].argmax()) + start
        if j <= start and start + 1 < m:           # a change must advance
            j = start + 1
        new_logp[0] = logp[best_r] + lchange + E[i, j]
        new_slide[0] = j
        parent[0] = best_r
        if len(new_logp) > max_hypotheses:
            keep = sorted(new_logp, key=lambda r: new_logp[r], reverse=True)[:max_hypotheses]
            new_logp = {r: new_logp[r] for r in keep}
            new_slide = {r: new_slide[r] for r in keep}
        logp, slide_of = new_logp, new_slide
        parents.append({r: (parent[r], new_slide[r]) for r in new_logp})
    out = [0] * n
    r = max(logp, key=lambda k: logp[k])
    for i in range(n - 1, -1, -1):
        r, out[i] = parents[i][r]
    return out


# --------------------------------------------------------------------------- #
# Wild binary segmentation
# --------------------------------------------------------------------------- #

def _contrast(cum: np.ndarray, a: int, b: int, t: int) -> float:
    """Length-normalised gain of splitting [a, b) at t: the two halves' best slides
    against one best slide for the whole, scaled like a CUSUM statistic."""
    nl, nr = t - a, b - t
    if nl <= 0 or nr <= 0:
        return NEG
    gain = float((cum[t] - cum[a]).max() + (cum[b] - cum[t]).max() - (cum[b] - cum[a]).max())
    return gain * np.sqrt(nl * nr / (nl + nr)) / (nl + nr)


def align_wbs2(sim, *, cfg: Optional[DPConfig] = None, n_intervals: int = 40,
               seed: int = 0) -> list[int]:
    """WBS2 (Fryzlewicz 2020): random intervals re-drawn inside every segment.

    Each segment draws up to `n_intervals` random sub-intervals, takes the best split
    over them (`_contrast`) and recurses into both sides, until segments are shorter than
    4 units or 4m detections exist. The m - 1 strongest detections at least n / (3m)
    apart are the boundaries (the number of slides is known, so the paper's SDLL model
    selection is not needed); runs labelled by `label_runs_dp`.

    Segments are visited breadth-first: a depth-first recursion spends the 4m-detection
    budget on the left part of the transcript before it reaches the right part, so long
    transcripts would get few or no boundaries in their second half.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    E = prepare_emissions(S, cfg)
    Z = standardize_rows(S, cfg)
    cum = np.vstack([np.zeros(m), np.cumsum(Z, axis=0)])
    rng = np.random.RandomState(seed)
    found: list[tuple[float, int]] = []
    queue = deque([(0, n, 0)])
    while queue and len(found) <= 4 * m:
        a, b, depth = queue.popleft()
        if b - a < 4 or depth > 24:
            continue
        cands = [(a, b)]
        for _ in range(min(n_intervals, max(4, (b - a) // 2))):
            s, e = sorted(rng.randint(a, b + 1, size=2))
            if e - s >= 4:
                cands.append((int(s), int(e)))
        best = (NEG, -1)
        for s, e in cands:
            for t in range(s + 1, e):
                v = _contrast(cum, s, e, t)
                if v > best[0]:
                    best = (v, t)
        if best[1] < 0:
            continue
        found.append(best)
        queue.append((a, best[1], depth + 1))
        queue.append((best[1], b, depth + 1))
    found.sort(reverse=True)
    min_gap = max(1, n // (m * 3))
    bounds: list[int] = []
    for _, t in found:
        if len(bounds) >= m - 1:
            break
        if 0 < t < n and all(abs(t - b) >= min_gap for b in bounds):
            bounds.append(t)
    return label_runs_dp(E, sorted(bounds))


# --------------------------------------------------------------------------- #
# TextTiling
# --------------------------------------------------------------------------- #

def _tiling_depth(S: np.ndarray, cfg: DPConfig, window: int) -> np.ndarray:
    """Depth score per gap g = window .. n - window (index 0 is gap `window`)."""
    return depth_scores(block_cohesion(standardize_rows(S, cfg), window))


def align_texttiling(sim, *, cfg: Optional[DPConfig] = None, window: int = 3) -> list[int]:
    """TextTiling (Hearst 1997): cut at the deepest cohesion dips, then label the runs.

    Cohesion at a gap = cosine between the mean similarity rows of the `window` units
    before and after it; the m - 1 deepest dips (depth score), deepest first, at least
    max(2, n / (2m)) units apart, are the boundaries; runs labelled greedily in order.
    Transcripts shorter than 2 * window + 1 units fall back to dtw.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if n < 2 * window + 1 or m == 1:
        return align_dtw(S, cfg=cfg)
    E = prepare_emissions(S, cfg)
    depth = _tiling_depth(S, cfg, window)
    k = min(m - 1, len(depth))
    cand = [int(c) + window for c in np.argsort(-depth, kind="stable")[:k * 3]]
    spacing = max(2, n // (m * 2))
    bounds: list[int] = []
    for c in cand:
        if len(bounds) >= k:
            break
        if 0 < c < n and all(abs(c - b) >= spacing for b in bounds):
            bounds.append(c)
    return label_runs_greedy(E, sorted(bounds))


def align_texttiling_dp(sim, *, cfg: Optional[DPConfig] = None, window: int = 3) -> list[int]:
    """TextTiling's depth score, with exactly m - 1 boundaries chosen by DP.

    The boundaries maximise the summed depth, at least max(1, n / (3m)) units apart
    (`best_k_boundaries`); runs labelled by `label_runs_dp`. With m - 1 boundaries the
    runs are slides 1..m in order: every slide gets exactly one run.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if n < 2 * window + 1 or m == 1:
        return align_dtw(S, cfg=cfg)
    E = prepare_emissions(S, cfg)
    bounds = best_k_boundaries(_tiling_depth(S, cfg, window), m - 1, window, n,
                               max(1, n // (m * 3)))
    return label_runs_dp(E, bounds)


__all__ = ["label_runs_greedy", "label_runs_dp", "best_k_boundaries", "align_pelt",
           "align_bocpd", "align_wbs2", "align_texttiling",
           "align_texttiling_dp"]
