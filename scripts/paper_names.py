"""The readable names the paper uses for methods, kinds, measures, attributes and levels.

One place, so that every table and figure of scripts/paper_assets.py calls a method or a
category level by the same name. The keys are the repository's ids (methods/registry.py,
config/categories.toml, config/similarity.toml); the names were checked against what
each method does (docs/METHODS.md), not only against its id:

  dp                 "Viterbi DP", not "Monotone DP": dp may move back (backward_penalty;
                     the group "Path with going back"), it is Viterbi over the slides.
  llm_rerank_kbest   "LLM re-rank (5 methods)", not "(k-best)": since step 3 the candidates
                     are the alignments of five different methods, not dtw's k best paths.
  fusion             "LLM-prior DP": dp whose positional prior is the hierarchical LLM
                     answer, so it is a DP decoder, not a fusion of scores.
"""

from __future__ import annotations

#: Method id -> name in the paper.
METHOD_NAMES = {
    "argmax": "Argmax",
    "dtw": "DTW",
    "multiscale_dtw": "Multiscale DTW",
    "needleman_wunsch": "Needleman–Wunsch",
    "gotoh": "Gotoh",
    "drop_dtw": "Drop-DTW",
    "anchored_dp": "Anchored DP",
    "semimarkov": "Semi-Markov",
    "adaptive_prior": "Adaptive prior",
    "dp": "Viterbi DP",
    "rank_dp": "Rank DP",
    "cohesion_dp": "Cohesion DP",
    "jump_dtw": "Jump DTW",
    "forward_backward": "Forward–backward",
    "softdtw": "Soft-DTW",
    "pelt": "PELT",
    "bocpd": "BOCPD",
    "wbs2": "WBS2",
    "texttiling": "TextTiling",
    "texttiling_dp": "TextTiling DP",
    "opw": "OPW",
    "ot_unbalanced": "Unbalanced OT",
    "ensemble": "Ensemble vote",
    "score_fusion": "Score fusion",
    "mbr": "MBR decoding",
    "llm_single_call": "LLM single call",
    "llm_monotone": "LLM monotone",
    "hierarchical": "LLM hierarchical",
    "llm_rerank_kbest": "LLM re-rank (5 methods)",
    "fusion": "LLM-prior DP",
    "llm_monotone_constrained": "LLM monotone (constrained)",
}

#: The similarity measures (config/similarity.toml) as the text, the method table and the
#: measure table name them (one name each, so a reader can match them across the three).
MEASURE_NAMES = {
    "bm25": "BM25",
    "embed": "Embed",
    "splade": "SPLADE",
    "cross_encoder": "Cross-encoder",
    "image": "Image",
    "hybrid": "Embed–BM25 hybrid",
    "hybrid_splade": "Embed–SPLADE hybrid",
    "hybrid_image": "Image hybrid",
}

#: The attributes (step-6 axes, corpus is not one in the paper's analyses).
AXIS_NAMES = {
    "text_source": "Slide-text source",
    "length": "Length",
    "images": "Picture share",   # not "Images": "image" is also a similarity measure
    "tables": "Tables",
    "noise": "Noise",
    "text_density": "Text density",
    "discipline": "Discipline",
}

#: (axis, level) -> name. Discipline covers every level config/categories.toml can give
#: (the OECD areas, Computer and information sciences as its own level, "other"), not only
#: the ones the 140 test lectures have, so a rerun on other data still finds a name.
LEVEL_NAMES = {
    ("text_source", "pdf_text_layer"): "PDF text layer",
    ("text_source", "corpus_corrected_ocr"): "Corrected OCR (corpus)",
    ("text_source", "tesseract_ocr"): "Tesseract OCR",
    **{(a, lv): lv for a in ("images", "noise", "text_density")
       for lv in ("low", "medium", "high")},
    **{("length", lv): lv for lv in ("short", "medium", "long")},
    ("tables", "none"): "none",
    ("tables", "some"): "some",
    ("discipline", "Computer and information sciences"): "Computer science",
    ("discipline", "Natural sciences (other)"): "Natural sciences",
    ("discipline", "Natural sciences"): "Natural sciences",
    ("discipline", "Engineering and technology"): "Engineering",
    ("discipline", "Medical and health sciences"): "Medical",
    ("discipline", "Agricultural sciences"): "Agricultural sciences",
    ("discipline", "Social sciences"): "Social sciences",
    ("discipline", "Humanities"): "Humanities",
    ("discipline", "other"): "Other",
}

#: Level names shortened where a column header is narrow (the appendix's 22-level table);
#: the table's caption says what the short name stands for.
SHORT_LEVEL_NAMES = {
    ("text_source", "corpus_corrected_ocr"): "Corrected OCR",
}

#: The method families (methods/registry.py GROUPS) as the paper's Section 5 headings
#: name them, so the pipeline figure, the method table and the text use the same words.
FAMILY_NAMES = {
    "independent": "Independent labelling",
    "monotone": "Monotone paths",
    "back": "Paths that may go back",
    "boundaries": "Boundaries first",
    "transport": "Transport",
    "consensus": "Consensus",
    "llm_text": "LLM reads the text",
    "llm_matrix": "LLM with similarity scores",
}

#: The corpora (prepared names) as the paper names them; the superscript 3 is written as
#: a character, which paper_assets.tex turns into $^3$ for LaTeX.
CORPUS_NAMES = {
    "mavils": "MaViLS",
    "m3av_a": "M³AV A",
    "m3av_b": "M³AV B",
    "mcd": "MCD",
    "slide_highlight": "Slide-Highlight",
}


def method_name(method: str) -> str:
    return METHOD_NAMES[method]


def level_name(axis: str, level: str) -> str:
    return LEVEL_NAMES[(axis, level)]


def short_level_name(axis: str, level: str) -> str:
    return SHORT_LEVEL_NAMES.get((axis, level), LEVEL_NAMES[(axis, level)])
