"""Path methods that may go back to an earlier slide (at a price), and the soft decoders.

dp (Viterbi), rank_dp, cohesion_dp, jump_dtw, forward_backward, softdtw; plus the
posteriors `score_fusion` combines (`softdtw_posterior`, `cohesion_posterior`).

A lecturer mostly moves forward but sometimes flicks back to an earlier slide. These
methods allow that: dp, rank_dp and cohesion_dp anywhere at backward_penalty per slide,
jump_dtw up to a fixed number of slides. forward_backward and softdtw label each unit by
its posterior marginal rather than by one path, so consecutive labels can also go back.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from methods.common import (NEG, NO_SLIDE, DPConfig, check_matrix, cohesion_depth,
                            prepare_emissions, transition_scores)


# --------------------------------------------------------------------------- #
# Viterbi over slide states
# --------------------------------------------------------------------------- #

def viterbi(E: np.ndarray, T: np.ndarray) -> list[int]:
    """The best state path: sum of E[i, s_i] + T[s_{i-1}, s_i]. Ties: the lower slide."""
    n, k = E.shape
    D = E[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    for i in range(1, n):
        scores = D[:, None] + T
        back[i] = scores.argmax(axis=0)
        D = E[i] + scores.max(axis=0)
    path = [int(D.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return path


def align_dp(sim, *, cfg: Optional[DPConfig] = None,
             no_slide_score: Optional[float] = None) -> list[int]:
    """Viterbi (1967) over slide states with the MaViLS-style transition prior.

    Moving a slides forward costs forward_penalty * a, b slides back backward_penalty * b,
    staying earns stay_bonus (Anderer et al. 2024 penalise slide transitions the same
    way). `no_slide_score`: when set, an extra state "no slide" (label NO_SLIDE) with this
    emission and free transitions; off by default.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    E = prepare_emissions(S, cfg)
    T = transition_scores(m, cfg)
    if no_slide_score is not None:
        E = np.hstack([E, np.full((n, 1), float(no_slide_score))])
        Tf = np.zeros((m + 1, m + 1))
        Tf[:m, :m] = T
        T = Tf
    return [NO_SLIDE if s == m else s for s in viterbi(E, T)]


def rank_transform(sim: np.ndarray) -> np.ndarray:
    """Each similarity replaced by its rank within its row, scaled to [0, 1] (C99).

    Tied values share their average rank. Ranks from argsort would give tied values
    distinct, index-ordered ranks: a row of equal scores (a unit sharing no word with any
    slide, common with lexical scores) would become a ramp favouring the LAST slide, and
    the dp below would follow it.
    """
    S = np.asarray(sim, dtype=np.float64)
    if S.size == 0:
        return S
    m = S.shape[1]
    less = (S[:, None, :] < S[:, :, None]).sum(axis=2)
    equal = (S[:, None, :] == S[:, :, None]).sum(axis=2)
    return (less + 0.5 * (equal - 1)) / max(m - 1, 1)


def align_rank_dp(sim, *, cfg: Optional[DPConfig] = None) -> list[int]:
    """dp on within-row ranks instead of raw similarities (Choi 2000's rank idea)."""
    S = check_matrix(sim)
    return align_dp(rank_transform(S), cfg=cfg)


def _cohesion_modulation(sim: np.ndarray, cfg: DPConfig, window: int,
                         strength: float) -> np.ndarray:
    if not 0.0 <= strength <= 2.0:
        raise ValueError(f"strength must be in [0, 2] (above 2 a move into a deep valley "
                         f"would be rewarded instead of charged), got {strength}")
    return 1.0 + strength * (0.5 - cohesion_depth(sim, cfg, window))


def align_cohesion_dp(sim, *, cfg: Optional[DPConfig] = None, window: int = 3,
                      strength: float = 1.0) -> list[int]:
    """dp whose slide-change costs follow the transcript's own topic shifts.

    TextTiling's depth score (Hearst 1997) in [0, 1] says how strongly the topic changes
    just before unit i; every move into unit i (forward or back, not a stay) is charged
    its dp cost times 1 + strength * (0.5 - depth[i]): half price at the deepest dip, 1.5x
    mid-topic (strength 1). `window`: units per block on each side of a gap.

    The cohesion signal is computed from the z-scored rows without the positional prior
    (the prior would swamp it), and depth climbs to the nearest peaks as in Hearst.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    E = prepare_emissions(S, cfg)
    mod = _cohesion_modulation(S, cfg, window, strength)
    base = transition_scores(m, cfg)
    stay = np.eye(m, dtype=bool)
    f = E[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        cand = f[:, None] + np.where(stay, base, base * mod[i])
        back[i] = cand.argmax(axis=0)
        f = cand[back[i], np.arange(m)] + E[i]
    out = [0] * n
    j = int(f.argmax())
    for i in range(n - 1, -1, -1):
        out[i] = j
        j = int(back[i, j])
    return out


def align_jump_dtw(sim, *, cfg: Optional[DPConfig] = None, max_jump: int = 3,
                   jump_cost: Optional[float] = None) -> list[int]:
    """dtw plus backward jumps of up to `max_jump` slides (Fremerey et al. 2010).

    A jump back of d slides costs d * jump_cost (default backward_penalty, what dp charges
    for the same move). max_jump = 0 is dtw. Ties: stay, +1, +2, then the jumps.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if max_jump < 0:
        raise ValueError(f"max_jump must be >= 0, got {max_jump}")
    E = prepare_emissions(S, cfg)
    K = int(min(max_jump, m - 1))
    cost = cfg.backward_penalty if jump_cost is None else float(jump_cost)
    D = np.full((n, m), NEG)
    back = np.zeros((n, m), dtype=np.int32)
    D[0] = E[0] - cfg.forward_penalty * np.arange(m)
    steps = [0, 1, 2][:1 + min(2, m - 1)] + [-d for d in range(1, K + 1)]
    for i in range(1, n):
        prev = D[i - 1]
        cand = np.full((len(steps), m), NEG)
        for si, s in enumerate(steps):
            if s == 0:
                cand[si] = prev + cfg.stay_bonus
            elif s > 0:
                cand[si, s:] = prev[:m - s] - cfg.forward_penalty * s
            else:
                cand[si, :m + s] = prev[-s:] - cost * (-s)
        arg = cand.argmax(axis=0)
        D[i] = E[i] + cand[arg, np.arange(m)]
        back[i] = np.asarray(steps)[arg]
    j = int(D[n - 1].argmax())
    path = [j]
    for i in range(n - 1, 0, -1):
        j -= int(back[i, j])
        path.append(j)
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# Posterior decoders
# --------------------------------------------------------------------------- #

def _hmm_posterior(E: np.ndarray, trans) -> np.ndarray:
    """Forward-backward with per-row softmax emissions; trans(i) -> row-stochastic T."""
    n, m = E.shape
    B = np.exp(E - E.max(axis=1, keepdims=True))
    B /= B.sum(axis=1, keepdims=True) + 1e-12
    alpha = np.zeros((n, m))
    alpha[0] = B[0] / (B[0].sum() + 1e-12)
    for i in range(1, n):
        a = (alpha[i - 1] @ trans(i)) * B[i]
        alpha[i] = a / (a.sum() + 1e-12)
    beta = np.zeros((n, m))
    beta[-1] = 1.0 / m
    for i in range(n - 2, -1, -1):
        b = trans(i + 1) @ (B[i + 1] * beta[i + 1])
        beta[i] = b / (b.sum() + 1e-12)
    post = alpha * beta
    return post / (post.sum(axis=1, keepdims=True) + 1e-12)


def _stochastic(T: np.ndarray) -> np.ndarray:
    T = np.exp(T - T.max(axis=1, keepdims=True))
    return T / (T.sum(axis=1, keepdims=True) + 1e-12)


def forward_backward(sim, *, cfg: Optional[DPConfig] = None) -> np.ndarray:
    """P(slide | all units) per unit, under the HMM whose Viterbi path is dp.

    Transitions: dp's scores, exponentiated and normalised per row; emissions: the
    prepared row, softmax-normalised; uniform start.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return np.zeros((0, m))
    T = _stochastic(transition_scores(m, cfg))
    return _hmm_posterior(prepare_emissions(S, cfg), lambda i: T)


def align_forward_backward(sim, *, cfg: Optional[DPConfig] = None) -> list[int]:
    """Posterior decoding (Baum et al. 1970; Rabiner 1989): per unit, the most probable
    slide given ALL units. This is also MEA decoding of the same posterior."""
    post = forward_backward(sim, cfg=cfg)
    return [int(j) for j in post.argmax(axis=1)]


def cohesion_posterior(sim, *, cfg: Optional[DPConfig] = None, window: int = 3,
                       strength: float = 1.0) -> np.ndarray:
    """forward_backward with cohesion_dp's cohesion-modulated transitions."""
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return np.zeros((0, m))
    mod = _cohesion_modulation(S, cfg, window, strength)
    base = transition_scores(m, cfg)
    stay = np.eye(m, dtype=bool)
    return _hmm_posterior(prepare_emissions(S, cfg),
                          lambda i: _stochastic(np.where(stay, base, base * mod[i])))


def _lse3(a, b, c, gamma):
    mx = np.maximum(np.maximum(a, b), c)
    s = np.exp((a - mx) / gamma) + np.exp((b - mx) / gamma) + np.exp((c - mx) / gamma)
    return mx + gamma * np.log(np.maximum(s, 1e-300))


def _softdtw_tables(E: np.ndarray, cfg: DPConfig, gamma: float):
    """Forward F[i, j] and backward B[i, j]: gamma-smoothed best prefix ending at (i, j)
    and best suffix after it, over dtw's moves (stay / +1 / +2, free start and end)."""
    n, m = E.shape
    F = np.full((n, m), NEG)
    F[0] = E[0] - cfg.forward_penalty * np.arange(m)
    for i in range(1, n):
        a1 = np.full(m, NEG)
        a1[1:] = F[i - 1, :-1] - cfg.forward_penalty
        a2 = np.full(m, NEG)
        a2[2:] = F[i - 1, :-2] - 2 * cfg.forward_penalty
        F[i] = E[i] + _lse3(F[i - 1] + cfg.stay_bonus, a1, a2, gamma)
    B = np.zeros((n, m))
    for i in range(n - 2, -1, -1):
        nxt = E[i + 1] + B[i + 1]
        a1 = np.full(m, NEG)
        a1[:-1] = nxt[1:] - cfg.forward_penalty
        a2 = np.full(m, NEG)
        a2[:-2] = nxt[2:] - 2 * cfg.forward_penalty
        B[i] = _lse3(nxt + cfg.stay_bonus, a1, a2, gamma)
    return F, B


def softdtw_posterior(sim, *, cfg: Optional[DPConfig] = None, gamma: float = 0.2) -> np.ndarray:
    """P(unit i on slide j) over all dtw paths weighted by exp(score / gamma)."""
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return np.zeros((0, m))
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    F, B = _softdtw_tables(prepare_emissions(S, cfg), cfg, gamma)
    L = (F + B) / gamma
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / (P.sum(axis=1, keepdims=True) + 1e-300)


def align_softdtw(sim, *, cfg: Optional[DPConfig] = None, gamma: float = 0.2,
                  mode: str = "expected") -> list[int]:
    """Soft-DTW (Cuturi and Blondel 2017) used as a decoder.

    dtw's lattice with the max replaced by a gamma-smoothed max (gamma -> 0 is dtw).
    mode "expected": each unit takes the argmax of its marginal over all paths (labels
    may step back); mode "path": one path, backtraced through the smoothed scores
    (monotone).
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if mode not in ("expected", "path"):
        raise ValueError(f"unknown mode {mode!r}; expected 'expected' or 'path'")
    if m == 1:
        return [0] * n
    if mode == "expected":
        return [int(j) for j in softdtw_posterior(S, cfg=cfg, gamma=gamma).argmax(axis=1)]
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    F, _ = _softdtw_tables(prepare_emissions(S, cfg), cfg, gamma)
    j = int(F[n - 1].argmax())
    path = [j]
    for i in range(n - 1, 0, -1):
        cands = [(F[i - 1, j] + cfg.stay_bonus, j)]
        if j >= 1:
            cands.append((F[i - 1, j - 1] - cfg.forward_penalty, j - 1))
        if j >= 2:
            cands.append((F[i - 1, j - 2] - 2 * cfg.forward_penalty, j - 2))
        j = max(cands)[1]
        path.append(j)
    path.reverse()
    return path


__all__ = ["viterbi", "align_dp", "rank_transform", "align_rank_dp", "align_cohesion_dp",
           "align_jump_dtw", "forward_backward", "align_forward_backward",
           "cohesion_posterior", "softdtw_posterior", "align_softdtw"]
