"""Consensus methods: combine the answers of other methods.

ensemble (a vote over five decoders' labels and llm_single_call's), score_fusion (the sum of three decoders'
posteriors), mbr (a vote over dtw's k best paths).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from methods.boundaries import align_pelt
from methods.common import (DPConfig, Member, check_matrix, check_member_names,
                            resolve_member)
from methods.monotone import align_dtw, align_needleman_wunsch, kbest_dtw, nw_posterior
from methods.soft_order import (align_cohesion_dp, align_dp, align_softdtw,
                                cohesion_posterior, softdtw_posterior)

#: The decoders `ensemble` can vote with, by name; each runs at its defaults unless
#: `member_inputs` gives it its own matrix, cfg and settings.
VOTERS = {"dp": align_dp, "dtw": align_dtw, "needleman_wunsch": align_needleman_wunsch,
          "cohesion_dp": align_cohesion_dp, "pelt": align_pelt}
ENSEMBLE_MEMBERS = ("dp", "dtw", "needleman_wunsch", "cohesion_dp", "pelt")
#: ensemble's fixed LLM voter (its `member_inputs` entry may carry its own LLM).
LLM_VOTER = "llm_single_call"

#: score_fusion's members: the method whose (tuned) settings a member takes, and the
#: posterior those settings are passed to, in the order of the sum.
FUSION_MEMBERS = {"softdtw": (align_softdtw, softdtw_posterior),
                  "cohesion_dp": (align_cohesion_dp, cohesion_posterior),
                  "needleman_wunsch": (align_needleman_wunsch, nw_posterior)}
#: Member settings with no counterpart in the posterior, dropped (docs/TUNING_DESIGN.md
#: 2.6): the soft dtw posterior is always the marginal over all paths.
FUSION_UNMAPPED = {"softdtw": ("mode",)}


def align_ensemble(sim, slide_texts: Sequence[str], unit_texts: Sequence[str], *,
                   llm, cfg: Optional[DPConfig] = None,
                   members: Sequence[str] = ENSEMBLE_MEMBERS, smooth: bool = True,
                   member_inputs: Optional[Mapping[str, Member]] = None,
                   diagnostics: Optional[dict] = None) -> list[int]:
    """A per-unit vote (Dietterich 2000) of the matrix decoders in `members` plus
    llm_single_call, then a median clean-up.

    Every voter has one equal vote: the five members by default (dp, dtw,
    needleman_wunsch, cohesion_dp, pelt, each at its own defaults) and llm_single_call
    (at its defaults; its one request is answered from the cache when it was asked
    before). TIE RULE: when several slides share the most votes - with six voters 3-3,
    2-2-2 and 2-2-1-1 are possible - the unit takes the FIRST member's label (dp by
    default) if that label is among the tied ones, otherwise the lowest tied slide.
    `smooth`: a width-3 median over the voted labels removes one-unit spikes.

    DECIDED: there are no member weights and no generic extra voter; the LLM voter is
    fixed and votes like any member.

    `member_inputs` (by member name, LLM_VOTER included): each member's own matrix, cfg
    and settings, and the LLM voter's own `LLM` (its model), so that every voter runs at
    its tuned settings on its own tuned measure (docs/TUNING_DESIGN.md 2.6). A member
    without an entry, or a field left at None, takes ensemble's own input.
    """
    from methods.llm import align_llm_single_call      # llm.py imports this module's peers

    S = check_matrix(sim)
    n = S.shape[0]
    unknown = [name for name in members if name not in VOTERS]
    if unknown or not members:
        raise ValueError(f"unknown or empty ensemble members {unknown or members}; "
                         f"choose from {sorted(VOTERS)}")
    inputs = check_member_names(member_inputs, tuple(members) + (LLM_VOTER,), "ensemble")
    voter = inputs.get(LLM_VOTER)
    S_llm, _, llm_kw = resolve_member(LLM_VOTER, voter, S, None, align_llm_single_call)
    llm_diag: dict = {}
    llm_votes = align_llm_single_call(S_llm, slide_texts, unit_texts,
                                      llm=llm if voter is None or voter.llm is None else voter.llm,
                                      diagnostics=llm_diag, **llm_kw)
    if diagnostics is not None:
        diagnostics.update(llm_diag)
    if n == 0:
        return []
    preds = []
    for name in members:
        S_m, cfg_m, kw = resolve_member(name, inputs.get(name), S, cfg, VOTERS[name])
        preds.append(VOTERS[name](S_m, cfg=cfg_m, **kw))
    preds.append(llm_votes)
    first = preds[0]
    out: list[int] = []
    for i in range(n):
        counts: dict[int, int] = {}
        for p in preds:
            counts[int(p[i])] = counts.get(int(p[i]), 0) + 1
        best = max(counts.values())
        winners = [k for k, v in counts.items() if v == best]
        out.append(first[i] if first[i] in winners else min(winners))
    if smooth and n >= 3:
        out = [out[0]] + [int(sorted(out[i - 1:i + 2])[1]) for i in range(1, n - 1)] + [out[-1]]
    return out


def score_fusion_posterior(sim, *, cfg: Optional[DPConfig] = None, nw_gamma: float = 0.3,
                           member_inputs: Optional[Mapping[str, Member]] = None) -> np.ndarray:
    """The sum of three per-unit posteriors: soft dtw (gamma 0.2), cohesion_dp's HMM
    (window 3, strength 1) and soft needleman_wunsch (gaps 0.4 / 0.1, gamma `nw_gamma`),
    at these defaults unless `member_inputs` gives a member its own matrix, cfg and
    settings (the settings of softdtw, cohesion_dp or needleman_wunsch; softdtw's `mode`
    has no counterpart in the posterior and is dropped)."""
    S = check_matrix(sim)
    inputs = check_member_names(member_inputs, tuple(FUSION_MEMBERS), "score_fusion")
    F = None
    for name, (decoder, posterior) in FUSION_MEMBERS.items():
        S_m, cfg_m, kw = resolve_member(name, inputs.get(name), S, cfg, decoder)
        for k in FUSION_UNMAPPED.get(name, ()):
            kw.pop(k, None)
        if posterior is nw_posterior:
            kw["gamma"] = nw_gamma          # the hard needleman_wunsch has no gamma
        P = posterior(S_m, cfg=cfg_m, **kw)
        F = P if F is None else F + P
    return F


def align_score_fusion(sim, *, cfg: Optional[DPConfig] = None, nw_gamma: float = 0.3,
                       member_inputs: Optional[Mapping[str, Member]] = None) -> list[int]:
    """The sum rule (Kittler et al. 1998) over three decoders' posteriors, then argmax.

    Without `member_inputs` the shared DP settings reach all three members and their
    own knobs stay at the values in `score_fusion_posterior`; with it, each member
    brings its own (tuned) matrix, cfg and settings. `nw_gamma`, the softness of the
    soft Needleman-Wunsch member, is score_fusion's own setting because the hard
    needleman_wunsch it takes its other settings from has none (docs/TUNING_DESIGN.md 2.6).
    """
    F = score_fusion_posterior(sim, cfg=cfg, nw_gamma=nw_gamma, member_inputs=member_inputs)
    return [int(j) for j in F.argmax(axis=1)]


def mbr_votes(paths: np.ndarray, m: int, *, scores: Optional[np.ndarray] = None,
              temperature: Optional[float] = None) -> np.ndarray:
    """V[i, j]: the weight of the candidate paths putting unit i on slide j (rows sum to
    1). Uniform weights, or exp(score / temperature) when a temperature is given."""
    P = np.asarray(paths, dtype=int)
    K, n = P.shape
    if temperature is None or scores is None:
        w = np.full(K, 1.0 / K)
    else:
        s = np.asarray(scores, dtype=float) / max(float(temperature), 1e-9)
        w = np.exp(s - s.max())
        w /= w.sum()
    V = np.zeros((n, m))
    for kk in range(K):
        V[np.arange(n), P[kk]] += w[kk]
    return V


def align_mbr(sim, *, cfg: Optional[DPConfig] = None, k: int = 100,
              temperature: Optional[float] = None,
              diagnostics: Optional[dict] = None) -> list[int]:
    """Minimum Bayes risk decoding (Kumar and Byrne 2004) over dtw's k best paths.

    Under per-unit 0/1 loss the Bayes decision is a per-unit vote over the candidates:
    each unit takes the slide most of the `k` best dtw paths give it (uniform weights,
    or exp(score / temperature)). Ties: the lower slide. Labels may step back.
    """
    cfg = cfg or DPConfig()
    S = check_matrix(sim)
    n, m = S.shape
    if n == 0:
        return []
    paths, scores = kbest_dtw(S, k, cfg=cfg)
    V = mbr_votes(paths, m, scores=scores, temperature=temperature)
    labels = [int(j) for j in V.argmax(axis=1)]
    if diagnostics is not None:
        srt = np.sort(V, axis=1)
        diagnostics.update({"n_paths": int(paths.shape[0]),
                            "mean_margin": float(np.mean(srt[:, -1] - (srt[:, -2] if m > 1 else 0))),
                            "monotone": bool(np.all(np.diff(labels) >= 0))})
    return labels


__all__ = ["VOTERS", "ENSEMBLE_MEMBERS", "LLM_VOTER", "FUSION_MEMBERS", "FUSION_UNMAPPED",
           "align_ensemble", "score_fusion_posterior",
           "align_score_fusion", "mbr_votes", "align_mbr"]
