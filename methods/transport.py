"""Transport methods: a soft coupling between units and slides, then a per-unit argmax.

opw, ot_unbalanced. Instead of a path, they solve for how much of each unit's mass goes
to each slide, under mass constraints (each slide should receive its share of the
transcript) and, for opw, a preference for the diagonal. The label is each row's argmax,
so consecutive labels may step back.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from methods.common import (NO_SLIDE, DPConfig, check_matrix, fill_drops,
                            prepare_emissions)


def sinkhorn_log(M: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float, *,
                 reg_m: Optional[float] = None, n_iter: int = 2000,
                 tol: float = 1e-9) -> tuple[np.ndarray, dict]:
    """Entropic OT in the log domain: the coupling T minimising <T, M> - eps * H(T).

    Balanced (marginals a and b exactly) or, with `reg_m`, both marginals relaxed by a
    KL penalty of strength reg_m (Chizat et al. 2018). The log domain does not underflow
    when eps is small relative to the cost range. Stops when the marginal violation
    (balanced) or its change (unbalanced) is below `tol`, checked every 25 iterations.
    """
    if eps <= 0:
        raise ValueError(f"the entropic regulariser must be > 0, got {eps}")
    n, m = M.shape
    f, g = np.zeros(n), np.zeros(m)
    log_a, log_b = np.log(a), np.log(b)
    kappa = 1.0 if reg_m is None else reg_m / (reg_m + eps)

    def lse(X):
        mx = X.max(axis=1, keepdims=True)
        return (mx + np.log(np.exp(X - mx).sum(axis=1, keepdims=True))).ravel()

    it, converged, prev = 0, False, None
    for it in range(1, n_iter + 1):
        f = kappa * eps * (log_a - lse((g[None, :] - M) / eps))
        g = kappa * eps * (log_b - lse(((f[:, None] - M) / eps).T))
        if it % 25 == 0 or it == n_iter:
            T = np.exp((f[:, None] + g[None, :] - M) / eps)
            rv = float(np.abs(T.sum(axis=1) - a).sum())
            cv = float(np.abs(T.sum(axis=0) - b).sum())
            if reg_m is None and max(rv, cv) < tol:
                converged = True
                break
            if reg_m is not None and prev is not None and abs(rv - prev[0]) < tol \
                    and abs(cv - prev[1]) < tol:
                converged = True
                break
            prev = (rv, cv)
    T = np.exp((f[:, None] + g[None, :] - M) / eps)
    return T, {"iters": it, "converged": converged,
               "row_viol": float(np.abs(T.sum(axis=1) - a).sum()),
               "col_viol": float(np.abs(T.sum(axis=0) - b).sum())}


def transport_cost(sim: np.ndarray, cfg: DPConfig) -> np.ndarray:
    """The prepared emissions as a cost, min-max scaled to [0, 1] per lecture, so one
    regulariser value means the same thing on every deck."""
    C = -prepare_emissions(sim, cfg)
    lo, hi = float(C.min()), float(C.max())
    return (C - lo) / (hi - lo) if hi > lo else np.zeros_like(C)


def opw_priors(n: int, m: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Su and Hua's two order priors: the inverse difference moment P (largest on the
    diagonal) and the log of a Gaussian reference measure S around the diagonal."""
    i = (np.arange(n)[:, None] + 1.0) / n
    j = (np.arange(m)[None, :] + 1.0) / m
    diff = i - j
    P = 1.0 / ((diff ** 2 + 1.0) ** 2)
    d = np.abs(diff) / np.sqrt(1.0 / n ** 2 + 1.0 / m ** 2)
    log_S = -(d ** 2) / (2.0 * sigma ** 2) - np.log(sigma * np.sqrt(2.0 * np.pi))
    return P, log_S


def align_opw(sim, *, cfg: Optional[DPConfig] = None, lambda1: float = 1.0,
              lambda2: float = 0.1, sigma: float = 1.0, n_iter: int = 2000,
              diagnostics: Optional[dict] = None) -> list[int]:
    """Order-preserving Wasserstein (Su and Hua 2017/2019): balanced OT near the diagonal.

    Minimises <T, C> - lambda1 * <T, P> + lambda2 * KL(T || S) with uniform marginals
    (every unit has equal mass, every slide receives an equal share): `lambda1` weights
    the diagonal reward P, `lambda2` the entropic smoothing towards the Gaussian band S,
    `sigma` its width (in the paper's normalised distance). The paper's lambda1 = 50 is
    for its own cost scale; here the cost is in [0, 1].
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    P, log_S = opw_priors(n, m, sigma)
    M = transport_cost(S, cfg) - lambda1 * P - lambda2 * log_S
    T, info = sinkhorn_log(M, np.full(n, 1.0 / n), np.full(m, 1.0 / m), lambda2,
                           n_iter=n_iter)
    if diagnostics is not None:
        diagnostics.update(info)
    return [int(j) for j in T.argmax(axis=1)]


def align_ot_unbalanced(sim, *, cfg: Optional[DPConfig] = None, eps: float = 0.05,
                        reg_m: float = 0.5, drop_mass: float = 0.0,
                        drop_fill: str = "carry", n_iter: int = 2000,
                        diagnostics: Optional[dict] = None) -> list[int]:
    """Unbalanced entropic OT (Chizat et al. 2018): the marginals are only encouraged.

    `eps`: entropic smoothing; `reg_m`: how strongly the uniform marginals are enforced
    (large = balanced OT, small = each unit free to pick its slide). `drop_mass` > 0: a
    unit sending less than that fraction of its mass is dropped, then labelled by
    `drop_fill` ("carry", "argmax", "none" = NO_SLIDE).
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    if m == 1:
        return [0] * n
    a = np.full(n, 1.0 / n)
    T, info = sinkhorn_log(transport_cost(S, cfg), a, np.full(m, 1.0 / m), eps,
                           reg_m=reg_m, n_iter=n_iter)
    if diagnostics is not None:
        diagnostics.update(info)
    labels = [int(j) for j in T.argmax(axis=1)]
    if drop_mass > 0.0:
        mass = T.sum(axis=1)
        labels = [NO_SLIDE if mass[i] < drop_mass * a[i] else s for i, s in enumerate(labels)]
        labels = fill_drops(labels, prepare_emissions(S, cfg), drop_fill)
    return labels


__all__ = ["sinkhorn_log", "transport_cost", "opw_priors", "align_opw",
           "align_ot_unbalanced"]
