"""The 31 methods by name, their group, their settings and how to run one.

    from methods import run_method, lecture_texts
    labels = run_method("dtw", sim)                       # 0-based slide per unit
    labels = run_method("llm_monotone", sim, slide_texts=s, unit_texts=u, llm=llm)

A method's settings and their defaults are read off its function's signature, so the
registry cannot list a default the function does not use. Every run passes the settings
through `run_method`, which rejects a name the method does not have: a misspelt setting
is an error, not a silently ignored value.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Optional, Sequence

from methods import boundaries as B
from methods import consensus as C
from methods import llm as L
from methods import monotone as M
from methods import soft_order as SO
from methods import transport as T
from methods.common import DPConfig, Member, align_argmax

#: The groups, in the order of docs/METHODS.md.
GROUPS = {
    "independent": "Independent labelling",
    "monotone": "Monotone path",
    "back": "Path with going back",
    "boundaries": "Boundary-first",
    "transport": "Transport",
    "consensus": "Consensus",
    "llm_text": "LLM reads the text",
    "llm_matrix": "LLM + similarity numbers",
}

#: Arguments that are inputs, not settings.
_INPUTS = {"sim", "slide_texts", "unit_texts", "llm", "cfg", "diagnostics", "member_inputs"}

#: The consensus methods whose members can be given their own inputs (`member_inputs`
#: of `run_method`), and the members' names: each is a method of this registry, whose
#: (tuned) settings its `Member` carries (docs/TUNING_DESIGN.md 2.6).
MEMBERS = {"ensemble": C.ENSEMBLE_MEMBERS + (C.LLM_VOTER,),
           "score_fusion": tuple(C.FUSION_MEMBERS),
           "llm_rerank_kbest": L.RERANK_METHODS}


@dataclass(frozen=True)
class Method:
    name: str
    group: str
    fn: Callable[..., list[int]]
    reference: str
    #: "fine", "fixed" or "decided": the result of the step-3 check and of the
    #: decisions on the open questions (docs/METHODS.md)
    check: str

    @property
    def uses_llm(self) -> bool:
        """Needs the texts and an LLM: the six LLM methods and ensemble (its LLM voter)."""
        return "llm" in inspect.signature(self.fn).parameters

    @property
    def uses_cfg(self) -> bool:
        return "cfg" in inspect.signature(self.fn).parameters

    @property
    def settings(self) -> dict[str, Any]:
        """The method's own settings and their defaults (DPConfig's are separate)."""
        out = {}
        for name, p in inspect.signature(self.fn).parameters.items():
            if p.kind is p.VAR_KEYWORD:
                out.update(_hier_settings())
            elif name not in _INPUTS and p.default is not p.empty:
                out[name] = p.default
        return out


def _hier_settings() -> dict[str, Any]:
    sized = {"packet_size", "leaf_max_packets", "packet_chars"}
    return {f.name: ("sized per lecture" if f.name in sized else f.default)
            for f in fields(L.HierConfig)}


_TABLE = [
    ("argmax", "independent", align_argmax, "decided", "no source: the nearest-slide baseline"),
    ("dtw", "monotone", M.align_dtw, "fine", "Sakoe and Chiba 1978"),
    ("multiscale_dtw", "monotone", M.align_multiscale_dtw, "fixed", "Salvador and Chan 2007 (FastDTW)"),
    ("needleman_wunsch", "monotone", M.align_needleman_wunsch, "decided", "Needleman and Wunsch 1970"),
    ("gotoh", "monotone", M.align_gotoh, "decided", "Gotoh 1982"),
    ("drop_dtw", "monotone", M.align_drop_dtw, "fine", "Dvornik et al. 2021 (Drop-DTW)"),
    ("anchored_dp", "monotone", M.align_anchored_dp, "fine", "Liu and Zhu 2023 (Bertalign)"),
    ("semimarkov", "monotone", M.align_semimarkov, "decided", "Sarawagi and Cohen 2004"),
    ("adaptive_prior", "monotone", M.align_adaptive_prior, "fine",
     "no source; PAVA from Ayer et al. 1955"),
    ("dp", "back", SO.align_dp, "fine", "Viterbi 1967; transitions as Anderer et al. 2024 (MaViLS)"),
    ("rank_dp", "back", SO.align_rank_dp, "fixed", "Choi 2000 (C99)"),
    ("cohesion_dp", "back", SO.align_cohesion_dp, "fixed", "Hearst 1997 (TextTiling) as a prior"),
    ("jump_dtw", "back", SO.align_jump_dtw, "fine", "Fremerey et al. 2010"),
    ("forward_backward", "back", SO.align_forward_backward, "fine",
     "Baum et al. 1970; Rabiner 1989"),
    ("softdtw", "back", SO.align_softdtw, "fine", "Cuturi and Blondel 2017"),
    ("pelt", "boundaries", B.align_pelt, "fine", "Killick et al. 2012"),
    ("bocpd", "boundaries", B.align_bocpd, "decided", "Adams and MacKay 2007"),
    ("wbs2", "boundaries", B.align_wbs2, "fixed", "Fryzlewicz 2020"),
    ("texttiling", "boundaries", B.align_texttiling, "fixed", "Hearst 1997"),
    ("texttiling_dp", "boundaries", B.align_texttiling_dp, "fixed", "Hearst 1997"),
    ("opw", "transport", T.align_opw, "fine",
     "Su and Hua 2017 / 2019"),
    ("ot_unbalanced", "transport", T.align_ot_unbalanced, "fine", "Chizat et al. 2018"),
    ("ensemble", "consensus", C.align_ensemble, "decided", "Dietterich 2000"),
    ("score_fusion", "consensus", C.align_score_fusion, "fixed", "Kittler et al. 1998"),
    ("mbr", "consensus", C.align_mbr, "fixed", "Kumar and Byrne 2004"),
    ("llm_single_call", "llm_text", L.align_llm_single_call, "fixed", "no source"),
    ("llm_monotone", "llm_text", L.align_llm_monotone, "fixed", "no source"),
    ("hierarchical", "llm_text", L.align_hierarchical, "fixed",
     "no source; rebuilt after Anderer et al. 2024 (MaViLS)"),
    ("llm_rerank_kbest", "llm_matrix", L.align_llm_rerank_kbest, "decided",
     "Kumar and Byrne 2004 (reranking idea)"),
    ("fusion", "llm_matrix", L.align_fusion, "fixed", "no source"),
    ("llm_monotone_constrained", "llm_matrix", L.align_llm_monotone_constrained, "decided",
     "Willard and Louf 2023 (guided generation)"),
]

METHODS: dict[str, Method] = {name: Method(name, group, fn, ref, check)
                              for name, group, fn, check, ref in _TABLE}


def run_method(name: str, sim, *, cfg: Optional[DPConfig] = None,
               slide_texts: Optional[Sequence[str]] = None,
               unit_texts: Optional[Sequence[str]] = None, llm=None,
               member_inputs: Optional[Mapping[str, Member]] = None,
               diagnostics: Optional[dict] = None, **settings) -> list[int]:
    """Run one method on one lecture; returns a 0-based slide (or -1) per unit.

    `member_inputs` (ensemble, score_fusion, llm_rerank_kbest only; MEMBERS lists the
    names): a `Member` per member, its own matrix, cfg, settings and (LLM voter) LLM.
    Left out, every member runs at its defaults on `sim` under `cfg`, as before.
    """
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}")
    m = METHODS[name]
    unknown = set(settings) - set(m.settings)
    if unknown:
        raise ValueError(f"{name} has no setting(s) {sorted(unknown)}; "
                         f"its settings are {sorted(m.settings)}")
    kw: dict[str, Any] = dict(settings)
    params = inspect.signature(m.fn).parameters
    if "cfg" in params:
        kw["cfg"] = cfg
    if "diagnostics" in params:
        kw["diagnostics"] = diagnostics
    if member_inputs is not None:
        if "member_inputs" not in params:
            raise ValueError(f"{name} has no members; member_inputs is for {sorted(MEMBERS)}")
        kw["member_inputs"] = member_inputs
    if m.uses_llm:
        if slide_texts is None or unit_texts is None or llm is None:
            raise ValueError(f"{name} needs slide_texts, unit_texts and llm")
        return m.fn(sim, slide_texts, unit_texts, llm=llm, **kw)
    return m.fn(sim, **kw)


# --------------------------------------------------------------------------- #
# The prepared data format (data/prepared/<corpus>.jsonl.gz, scripts/prepare.py)
# --------------------------------------------------------------------------- #

def lecture_texts(record: dict) -> tuple[list[str], list[str]]:
    """(slide texts in slide order, unit texts in transcript order) of one record."""
    slides = sorted(record["slides"], key=lambda s: s["index"])
    if [s["index"] for s in slides] != list(range(1, len(slides) + 1)):
        raise ValueError(f"{record.get('lecture_id')}: slide indices are not 1..n")
    units = sorted(record["units"], key=lambda u: u["index"])
    return [s["text"] for s in slides], [u["text"] for u in units]


def gold_labels(record: dict) -> list[int]:
    """The true slide of every unit as a 0-based column (true_slide - 1)."""
    return [int(u["true_slide"]) - 1 for u in sorted(record["units"], key=lambda u: u["index"])]


def to_slide_index(labels: Sequence[int]) -> list[int]:
    """0-based labels -> the prepared data's 1-based slide index (NO_SLIDE stays -1)."""
    return [int(x) + 1 if x >= 0 else -1 for x in labels]


__all__ = ["GROUPS", "MEMBERS", "Method", "METHODS", "run_method", "lecture_texts", "gold_labels",
           "to_slide_index"]
