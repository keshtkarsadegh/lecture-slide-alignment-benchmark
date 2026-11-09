"""What the matrix methods share: the input check, the shared DP settings, the emission
preparation, the transcript-cohesion signal, and the independent baseline (argmax).

Every method takes a similarity matrix `sim` of shape (n_units, n_slides) - one row per
transcript unit, one column per slide, larger = more similar - and returns one label per
unit: a 0-based column index, or NO_SLIDE (-1) where a method may decline to name a slide.
Column k is the prepared slide with `index` k + 1 (data/prepared/<corpus>.jsonl.gz).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

#: A unit the method assigns to no slide. The ground truth gives every kept unit a slide,
#: so this label is always scored as wrong; only methods with an explicit gap or drop
#: setting can emit it.
NO_SLIDE = -1

#: The "minus infinity" of the dynamic programs: finite, so sums stay finite and a
#: comparison against NEG / 2 tells an unreachable cell from a reachable one.
NEG = -1e18


def check_matrix(sim) -> np.ndarray:
    """The matrix as float64, or a ValueError that says what is wrong with it.

    NaN and inf are rejected rather than tolerated: argmax and the DPs silently prefer a
    NaN column or propagate it into every later row, so a broken score file would come
    out as a plausible-looking alignment. A unit row with no slide column is rejected
    for the same reason: no label can be produced for it.
    """
    a = np.asarray(sim, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"the similarity matrix must be 2-D (units x slides), got shape {a.shape}")
    if a.shape[0] > 0 and a.shape[1] == 0:
        raise ValueError(f"{a.shape[0]} units but no slides: there is nothing to align to")
    if a.size and not np.isfinite(a).all():
        raise ValueError("the similarity matrix contains NaN or inf")
    return a


@dataclass(frozen=True)
class DPConfig:
    """The settings shared by the path decoders.

    Which method reads which field is listed in docs/METHODS.md; a monotone decoder, for
    instance, never reads `backward_penalty`, because it cannot go back.
    """

    #: cost per slide moved forward (DTW-style decoders: per step of +1)
    forward_penalty: float = 0.05
    #: cost per slide moved backward (decoders that may go back)
    backward_penalty: float = 0.5
    #: reward for staying on the same slide from one unit to the next
    stay_bonus: float = 0.2
    #: z-score each row before use: raw similarities are often compressed into a narrow
    #: band, and the transition costs above only mean something on a known scale
    standardize: bool = True
    #: weight of the positional prior: a unit at relative position p in the transcript is
    #: expected near slide p * (n_slides - 1); 0 switches the prior off
    pos_weight: float = 8.0


def standardize_rows(sim: np.ndarray, cfg: DPConfig) -> np.ndarray:
    """The rows z-scored (when cfg.standardize), with no positional prior.

    Used where a method measures how the TRANSCRIPT changes (cohesion, change-point
    contrasts): the positional prior is a smooth function of the row index alone, and
    mixed into those statistics it is what they would measure instead of the content.
    """
    Z = sim.astype(np.float64)
    if cfg.standardize:
        Z = (Z - Z.mean(axis=1, keepdims=True)) / (Z.std(axis=1, keepdims=True) + 1e-9)
    return Z


def expected_slide_linear(n: int, m: int) -> np.ndarray:
    """Constant pace: unit i is expected at slide i * (m - 1) / (n - 1)."""
    return np.arange(n) * (m - 1) / max(n - 1, 1)


def prepare_emissions(sim: np.ndarray, cfg: DPConfig,
                      expected: Optional[np.ndarray] = None) -> np.ndarray:
    """The score every path decoder adds up: z-scored rows minus a positional prior.

    The prior charges `pos_weight * |slide - expected| / m`, where `expected` is the slide
    a constant-pace lecture would be on (or a pacing curve, for `adaptive_prior` and
    `fusion`). Dividing by m keeps the prior's total range the same for every deck size.
    """
    E = standardize_rows(sim, cfg)
    if cfg.pos_weight:
        n, m = E.shape
        exp = expected_slide_linear(n, m) if expected is None else np.asarray(expected, float)
        E = E - cfg.pos_weight * np.abs(np.arange(m)[None, :] - exp[:, None]) / m
    return E


def transition_scores(m: int, cfg: DPConfig) -> np.ndarray:
    """T[a, b]: the score of moving from slide a to slide b between consecutive units.

    stay_bonus on the diagonal, -forward_penalty per slide forward, -backward_penalty per
    slide back.
    """
    idx = np.arange(m)
    diff = idx[None, :] - idx[:, None]
    return np.where(diff == 0, cfg.stay_bonus,
                    np.where(diff > 0, -cfg.forward_penalty * diff,
                             cfg.backward_penalty * diff))


# --------------------------------------------------------------------------- #
# Transcript cohesion (TextTiling's signal), shared by the boundary-first methods and
# cohesion_dp
# --------------------------------------------------------------------------- #

def block_cohesion(Z: np.ndarray, window: int) -> np.ndarray:
    """c[g] = cosine between the mean rows of the `window` units before and after gap g.

    Gap g sits between units g - 1 and g, for g = window .. n - window; the array is
    indexed from 0, so c[k] belongs to gap window + k. Rows are compared as vectors of
    slide similarities: two stretches of transcript that resemble the same slides are
    cohesive, a dip marks a change of topic.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    n = Z.shape[0]
    out = []
    for g in range(window, n - window + 1):
        a = Z[g - window:g].mean(axis=0)
        b = Z[g:g + window].mean(axis=0)
        out.append(float(a @ b / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9))))
    return np.asarray(out)


def depth_scores(scores: np.ndarray) -> np.ndarray:
    """Hearst's depth score: how far a gap's cohesion lies below the peaks around it.

    From each gap the score climbs left while the values keep rising, and right likewise;
    depth = (left peak - score) + (right peak - score). Climbing to the NEAREST peak is
    what makes a shallow dip between two high stretches count; measuring against the
    global maximum instead would rank gaps by their raw score.
    """
    s = np.asarray(scores, dtype=np.float64)
    k = len(s)
    d = np.zeros(k)
    for i in range(k):
        lo = i
        while lo > 0 and s[lo - 1] >= s[lo]:
            lo -= 1
        hi = i
        while hi < k - 1 and s[hi + 1] >= s[hi]:
            hi += 1
        d[i] = (s[lo] - s[i]) + (s[hi] - s[i])
    return d


def cohesion_depth(sim: np.ndarray, cfg: DPConfig, window: int) -> np.ndarray:
    """depth[i] in [0, 1] for every unit i: how strongly the topic changes just before i.

    Zero where the window does not fit (the first and last `window` units) and for
    transcripts too short to have an interior gap.
    """
    n = sim.shape[0]
    out = np.zeros(n)
    if n < 2 * window + 1:
        return out
    d = depth_scores(block_cohesion(standardize_rows(sim, cfg), window))
    out[window:window + len(d)] = d
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / (hi - lo) if hi > lo else np.zeros(n)


# --------------------------------------------------------------------------- #
# Independent labelling
# --------------------------------------------------------------------------- #

def align_argmax(sim) -> list[int]:
    """Each unit gets its most similar slide, independently of every other unit.

    No order and no prior: the baseline every ordered method is read against. Ties go
    to the lower slide index.

    DECIDED: there is no `no_slide_threshold` setting: it could only turn labels into
    NO_SLIDE, which is always scored wrong.
    """
    S = check_matrix(sim)
    if S.shape[0] == 0:
        return []
    return [int(j) for j in S.argmax(axis=1)]


def fill_drops(labels: list[int], fallback: np.ndarray, mode: str) -> list[int]:
    """Give every NO_SLIDE label a slide again, by rule `mode`.

    "carry"  - the previous label (a leading run takes the first real label);
    "argmax" - the unit's own best slide in `fallback` (a prepared emission matrix);
    "none"   - leave NO_SLIDE in place.
    """
    if mode == "none":
        return list(labels)
    if mode == "argmax":
        return [int(fallback[i].argmax()) if s == NO_SLIDE else int(s)
                for i, s in enumerate(labels)]
    if mode != "carry":
        raise ValueError(f"unknown drop_fill {mode!r}; expected carry, argmax or none")
    out = list(labels)
    first = next((s for s in out if s != NO_SLIDE), None)
    prev = first
    for i, s in enumerate(out):
        if s == NO_SLIDE:
            out[i] = prev if prev is not None else int(fallback[i].argmax())
        else:
            prev = s
    return [int(s) for s in out]


# --------------------------------------------------------------------------- #
# A consensus method's members: their own inputs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Member:
    """One member's own inputs inside a consensus method (ensemble, score_fusion,
    llm_rerank_kbest): its matrix, its DP settings, its own settings and, for an LLM
    member, its `LLM` (and so its model).

    WHY: the tuning design (docs/TUNING_DESIGN.md 2.6) runs every member at its own
    tuned settings on the matrix of its own tuned measure. A field left at None takes
    the consensus method's input (its matrix, cfg or llm) and empty `settings` means the
    member's defaults, so a consensus method given no members' inputs runs as before.
    """
    sim: Any = None
    cfg: Optional[DPConfig] = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    llm: Any = None


def check_member_names(member_inputs: Optional[Mapping[str, Member]],
                       allowed, method: str) -> dict[str, Member]:
    """The inputs by member name; a name that is not a member is an error (a misspelt
    name would otherwise leave that member silently at its defaults)."""
    inputs = dict(member_inputs or {})
    unknown = sorted(set(inputs) - set(allowed))
    if unknown:
        raise ValueError(f"{method} has no member(s) {unknown}; its members are {list(allowed)}")
    for name, mem in inputs.items():
        if not isinstance(mem, Member):
            raise TypeError(f"{method}: the input of member {name} must be a Member")
    return inputs


def resolve_member(name: str, member: Optional[Member], S: np.ndarray,
                   cfg: Optional[DPConfig], fn: Callable,
                   ) -> tuple[np.ndarray, Optional[DPConfig], dict[str, Any]]:
    """(matrix, cfg, settings) for one member: its own where given, else the consensus
    method's matrix and cfg and the member's defaults.

    The matrix must have the consensus matrix's shape, because the members' answers are
    combined unit by unit and slide by slide. Every setting must be one of `fn`'s own
    keyword settings, and a cfg or an llm is only accepted by a member that reads one.
    """
    if member is None:
        return S, cfg, {}
    params = inspect.signature(fn).parameters
    Sm = S if member.sim is None else check_matrix(member.sim)
    if Sm.shape != S.shape:
        raise ValueError(f"member {name}: its matrix has shape {Sm.shape}, "
                         f"the consensus matrix {S.shape}")
    if member.cfg is not None and "cfg" not in params:
        raise ValueError(f"member {name} takes no cfg")
    if member.llm is not None and "llm" not in params:
        raise ValueError(f"member {name} takes no llm")
    own = {p for p, v in params.items() if v.kind is v.KEYWORD_ONLY
           and p not in ("cfg", "llm", "diagnostics", "member_inputs")}
    unknown = sorted(set(member.settings) - own)
    if unknown:
        raise ValueError(f"member {name} has no setting(s) {unknown}; "
                         f"its settings are {sorted(own)}")
    return Sm, (cfg if member.cfg is None else member.cfg), dict(member.settings)


__all__ = ["NO_SLIDE", "NEG", "DPConfig", "check_matrix", "standardize_rows",
           "expected_slide_linear", "prepare_emissions", "transition_scores",
           "block_cohesion", "depth_scores", "cohesion_depth", "align_argmax",
           "fill_drops", "Member", "check_member_names", "resolve_member"]
