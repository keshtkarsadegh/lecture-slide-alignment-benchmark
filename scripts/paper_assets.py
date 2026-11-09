#!/usr/bin/env python3
"""Step 17: every table and figure of the paper, made from the results.

It reads what steps 6, 7, 11, 13, 14 and 16 wrote and draws the paper's tables (LaTeX,
booktabs) and figures (PDF, with a PNG next to each for viewing). It runs no method and
pays no LLM call. Every number is one of those results, or (the cost per category) the
same per-lecture costs as Q1, averaged over the category's lectures instead of all 140,
or (the measure and tuning tables) step 11's score of a tuning combination, the highest
per method (and measure), or a count over results in git: the steps of the true path
that go back, the wrong and the wrong core units per method (Q4 against errors that
ignore importance), the LLM answers that could not be used, and the label and discipline
checks. Every number the text quotes is a macro of numbers.tex.

INPUT
  data/analysis/q1_ranking.csv, q2_<axis>_long.csv, q4_wrong_important.csv  (steps 14, 16)
  data/final/methods/<method>.csv.gz     the per-lecture accuracy and cost (step 13)
  data/final/predictions/<method>.csv.gz the slide of every unit and its true slide (step
                                         13; the example, the back-jumps, Q4's expectation)
  data/importance/labels.csv.gz          the importance label of every unit (step 15)
  data/importance/second/, check_sample_key.csv, check_manual.csv, raw/
                                         the label checks: the second labelling, the
                                         blind check, the agents' batches (step 15)
  data/attributes/discipline.csv, discipline_check_50_v3.csv  the disciplines and the
                                         check of 50 of them (step 5)
  data/similarity/test/tables.sqlite3    the example lecture's image table, and every
                                         table's build seconds (step 12)
  data/tuning/runs/<method>.csv.gz       every tuning combination on every tuning
                                         lecture (step 11; the measure table)
  data/similarity/test/times.csv         each measure's deps and model load (step 12)
  data/categories/document_categories.csv, level_counts.csv, cut_points.json  (step 6)
  data/selection/selected_lectures.csv   the tuning and test sets (step 7)
  data/splits/m3av_split.csv             M3AV's talks in its two versions (step 1b)
  data/attributes/document_attributes.csv  slides per lecture (step 5)
  data/prepared/summary.csv, <corpus>_dropped.csv  the preparation counts and the dropped
                                         lectures (step 2; Table B1). Until they are
                                         there, Table B1 is a placeholder and its macros
                                         are not written; everything else is
  config/tuned_settings.toml             the tuned settings and tuning accuracies (step 11)
  methods/ (the registry: groups, members; llm.py: the prompts), scripts/paper_names.py

OUTPUT (results/, in git)
  tables/tab_corpora.tex      the corpora: source, slide text, lectures, test units
  tables/tab_categories.tex   the 7 attributes, their levels and cut rules
  tables/tab_methods.tex      the 31 methods by family (longtable)
  tables/tab_q1.tex           Q1: rank, accuracy and cost per lecture
  tables/tab_q3.tex           Q3: per category level the highest and lowest method
  tables/tab_q4.tex           Q4's numbers (appendix)
  tables/tab_q2_full.tex      Q2: every method in all 22 levels (appendix, landscape)
  tables/tab_measures.tex     accuracy per similarity measure on the tuning lectures, and
                              the cost of each measure's table
  tables/tab_q4_expected.tex  Q4 against errors that ignore importance (appendix)
  tables/tab_tuning.tex       best and chosen tuning accuracy against test (appendix)
  tables/tab_datasets.tex     every method's accuracy per dataset (appendix)
  tables/tab_bootstrap.tex    every method's bootstrap interval over the test lectures,
                              and that of its difference to rank 1 (appendix)
  tables/tab_severity.tex     how far each method's wrong units are from their true
                              slide (appendix)
  tables/tab_prepare.tex      per dataset the lectures dropped, kept and cut and the
                              units removed and kept by the preparation (appendix)
  tables/prompts.tex          the prompts of the final run (appendix)
  tables/numbers.tex          the numbers the text quotes, as LaTeX macros
  figures/fig_q1_scatter      accuracy against total seconds per lecture, with a zoom
  figures/fig_q2_accuracy     heatmap: accuracy per category column
  figures/fig_q2_cost         heatmap: seconds per lecture per category column
  figures/fig_q4_heatmap      heatmap: important units on the wrong slide
  figures/fig_pipeline        the task on one lecture, and the benchmark's steps
  figures/fig_example         a worked example: one test lecture and four methods
  (each figure as .pdf)

Every figure is drawn FIG_W = 4.69 in wide on a fixed canvas (the journal's text width,
119 mm, so it prints 1:1 there and no smaller in a wider layout) in one house style
(style()). Before a figure is saved its layout is checked: every text at least 7.5 pt,
inside the canvas and clear of every other text; the pipeline's texts also inside their
boxes. A failed check stops the run.

THE CHECKS (the run stops with exit code 1 and writes nothing when one fails):
  1. the methods of Q1, of every Q2 level, of Q4 and of the per-lecture tables are the
     31 of methods/, and every method and level has a display name;
  2. the per-lecture tables hold exactly the test set of the selection;
  3. per method, the cost per lecture over all documents (the cost heatmap's first
     column) equals Q1's total seconds per lecture;
  4. the test units of the corpora table add up to Q4's units of all documents;
  5. (the measure table, check_measures) every method that tunes the measure has runs
     for all eight, its chosen measure is within step 11's tie of its best, and each
     measure's build seconds equal Q1's Tables column of every method reading it alone;
  6. (check_units) every method's predictions carry the same true slides, its wrong units
     are step 13's, its wrong core units give Q4's all-lectures percentage, and no true
     path of an in-order corpus (MCD, Slide-Highlight) goes back;
  7. (check_tuning) every chosen combination is within 0.5 points of its method's best,
     its per-lecture runs give step 11's tuning accuracy, and the tuning runs hold every
     combination of the method's grid;
  8. (check_top_band) TOP_BAND is where Q1's accuracy first drops by more than
     TOP_BAND_STEP points from one rank to the next, so the band the text reads as one
     group follows from the numbers and is not a choice;
  9. (check_datasets) the per-dataset accuracies average, over the lectures, to Q1's;
 10. (check_split) every pooled M3AV lecture is in its version of the split;
 11. (check_bootstrap) the per-lecture accuracies the bootstrap resamples average to Q1's;
 12. (check_prepared) summary.csv and the dropped files agree, every drop reason has a
     column, the M3AV talks in both dropped files are in neither half of the split, and
     the kept lectures are Table 1's pool per dataset (so \\PrepPool is \\PoolLectures).

The bootstrap draws BOOT_DRAWS resamples of the test lectures with the project seed
(BOOT_SEED), so its table is the same on every run.

    python scripts/paper_assets.py [--prepared data/prepared]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analysis as A  # noqa: E402
import analysis_q4 as Q  # noqa: E402
import final_report as F  # noqa: E402
import heatmap as H  # noqa: E402
import paper_names as N  # noqa: E402
import tune as T  # noqa: E402
from methods import GROUPS, MEMBERS, METHODS  # noqa: E402

SELECTION = REPO_ROOT / "data" / "selection" / "selected_lectures.csv"
ATTRIBUTES = REPO_ROOT / "data" / "attributes" / "document_attributes.csv"
SPLIT = REPO_ROOT / "data" / "splits" / "m3av_split.csv"
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

#: The per-lecture seconds that make up a method's total (Q1's cost columns that are
#: times; which of them a method has depends on its kind, A.COSTS).
SECONDS = ("compute_s", "build_cost_s", "used_seconds")
#: The key of the "all documents" column in the per-category dictionaries.
ALL = ("all", "all")
#: Tolerance of check 3 (two sums of the same floats in a different order).
TOL = 1e-9
#: Kind -> colour and marker in every figure (Paul Tol's colour-blind safe "bright"
#: palette; the marker too, so the kind survives grey print).
KIND_STYLE = {"non-LLM": ("#4477AA", "o"), "LLM": ("#EE6677", "s"),
              "mixed": ("#AA3377", "^")}
#: The other colours of the house style.
TRUE_GREY = "#777777"        # the true slide, reference lines, secondary text
EMPHASIS = "#EEEEEE"         # the top band's shading
BORDER = "#555555"
FILL = {"data": "#E8EEF7", "methods": "#F2F2F2", "results": "#FBEEE3"}
COST_COLOUR = "#B07A00"      # every cost annotation (the dark yellow of the cost map)
Q4_COLOUR = "#C0392B"
WRONG = "#DDAA33"            # a core unit on the wrong slide: a yellow no method line uses
#: The fonts tried in order; the first one matplotlib has is used (DejaVu Sans ships
#: with matplotlib, so there is always one). No font file is added to the repository.
FONTS = ("Latin Modern Sans", "TeX Gyre Heros", "DejaVu Sans")
#: The width of every figure in inches: the journal's text width (119 mm). The paper
#: includes each figure at \linewidth, so a figure drawn at this width prints 1:1.
FIG_W = 4.69
#: The smallest text size anywhere in a figure, in points at print size.
MIN_PT = 7.5
#: The figures' text sizes in points: body text, tick and point labels (TEXT_PT); axis
#: labels and the titles of boxes and panels (TITLE_PT); the panel letters (LETTER_PT).
TEXT_PT = 7.5
TITLE_PT = 8.0
LETTER_PT = 8.5
#: The top band of Q1: the ranks the paper reads as one group (Section 7.1). It ends at
#: the first step of more than TOP_BAND_STEP points between neighbouring ranks
#: (check_top_band): inside the band no step is that large.
TOP_BAND = 14
TOP_BAND_STEP = 1.0

Key = tuple[str, str]  # (corpus, lecture_id)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

def fnum(v: Optional[float]) -> Optional[float]:
    return None if v in (None, "") else float(v)


def seconds(v: Optional[float], latex: bool = True) -> str:
    """A time for reading, one decimal in every column and figure: below 0.05 s (which
    would print as 0.0) "<0.1", so a small cost is not shown as none; exactly 0 (a
    method that reads no table) as 0.0; None as an en dash. `latex`: the "<" in math."""
    if v is None:
        return "--"
    if 0 < abs(v) < 0.05:
        return r"$<$0.1" if latex else "<0.1"
    return f"{v:.1f}"


def compute_seconds(v: Optional[float]) -> str:
    """Q1's compute column, the only place the method's own run time is shown: three
    decimals, "<0.001" below 0.0005 s (which would print as 0.000)."""
    if v is None:
        return "--"
    return r"$<$0.001" if 0 < v < 0.0005 else f"{v:.3f}"


def tex(s: str) -> str:
    """Plain text into LaTeX: the special characters escaped, the en dash as --, the
    superscript 3 of M³AV as $^3$."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("–", "--"),
                 ("³", "$^3$")):
        s = s.replace(a, b)
    return s


def head_cell(*lines: str) -> str:
    """A header cell of several lines, right-aligned over its numbers and sitting on the
    row's baseline (a nested tabular, so no package is needed)."""
    return r"\begin{tabular}[b]{@{}r@{}}" + r"\\".join(lines) + r"\end{tabular}"


def lines_cell(lines: Sequence[str]) -> str:
    """A left-aligned cell of several lines, top-aligned with its row (a nested tabular:
    it breaks where the text says, with no justified gaps and no package)."""
    if len(lines) == 1:
        return tex(lines[0])
    return r"\begin{tabular}[t]{@{}l@{}}" + r"\\".join(tex(x) for x in lines) + r"\end{tabular}"


def text_col(width: str, hang: bool = False) -> str:
    """A text column of the given width, ragged right (array's >{...}; the preamble
    loads array): its long cells wrap inside it instead of widening the table past the
    text. With hang, the lines after the first are indented by 1em, under the text of a
    first line that starts with \\quad (a level under its bold attribute)."""
    return (r">{\raggedright\arraybackslash" + (r"\hangindent=1em" if hang else "")
            + rf"}}p{{{width}}}")


#: The column of method names in the method table (Appendix A): as wide as every name
#: but the two longest (Needleman--Wunsch is 74.4 pt in the 8 bp table font), which wrap.
NAME_COL = text_col("75pt")
#: The column space of the appendix's method tables (E1, E2, F1): 1 em on each side of
#: a column, so that neighbouring columns are at least 2 em apart.
APPENDIX_SEP = "1em"


def over_left(cell: str) -> str:
    """A header that may reach left over the space of the column before it (a box of
    no width, right-aligned): the column is then only as wide as its numbers."""
    return r"\llap{" + cell + "}"


# --------------------------------------------------------------------------- #
# Costs per category (the cost heatmap and Q3)
# --------------------------------------------------------------------------- #

def applies(cost: str, kind: str) -> bool:
    return kind in dict(A.COSTS)[cost]


def lecture_seconds(row: Mapping[str, Any], kind: str, load_share: float) -> float:
    """One lecture's total seconds, as Q1 counts them for the method's kind: compute
    (non-LLM, mixed), table build (all) and LLM seconds (LLM, mixed), plus the lecture's
    share of the model loads."""
    return load_share + sum(float(row[c]) for c in SECONDS if applies(c, kind))


def lecture_costs(rows: Mapping[str, Sequence[Mapping[str, Any]]], kinds: Mapping[str, str],
                  load_s: Mapping[str, float]) -> dict[str, dict[Key, float]]:
    """Per method and lecture its total seconds (compute, tables and LLM, as Q1 counts
    them for the kind).

    Q1's build total adds each model load once (step 13's summary), while the
    per-lecture rows hold no load. So that the mean over all lectures equals Q1's total
    per lecture exactly, each method's load is spread evenly: load / n added to every
    one of its n lectures."""
    out: dict[str, dict[Key, float]] = {}
    for m, rs in rows.items():
        share = load_s.get(m, 0.0) / len(rs)
        out[m] = {(r["corpus"], r["lecture_id"]): lecture_seconds(r, kinds[m], share)
                  for r in rs}
    return out


def category_costs(rows: Mapping[str, Sequence[Mapping[str, Any]]], kinds: Mapping[str, str],
                   load_s: Mapping[str, float],
                   levels: Mapping[str, Mapping[Key, str]]) -> dict[str, dict[Key, float]]:
    """Per method and (axis, level), and ALL, the mean seconds per lecture over the
    lectures of the level (lecture_costs, so a level's mean carries the same share of
    the model loads per lecture as Q1's total)."""
    out: dict[str, dict[Key, float]] = {}
    for m, per in lecture_costs(rows, kinds, load_s).items():
        cell = {ALL: sum(per.values()) / len(per)}
        for axis, lv in levels.items():
            by_level: dict[str, list[float]] = defaultdict(list)
            for k, v in per.items():
                by_level[lv[k]].append(v)
            for level, vs in by_level.items():
                cell[(axis, level)] = sum(vs) / len(vs)
        out[m] = cell
    return out


def q1_total(row: Mapping[str, Any]) -> float:
    """Q1's total seconds per lecture: compute + tables + LLM seconds, those the kind has."""
    return sum(row[f"{c}_per_lecture"] or 0.0 for c in SECONDS)


def check_costs(q1: Sequence[Mapping[str, Any]],
                costs: Mapping[str, Mapping[Key, float]]) -> list[str]:
    """Check 3: the all-documents cost of every method is Q1's total per lecture."""
    return [f"cost: {r['method']} all documents {costs[r['method']][ALL]!r} != Q1 total "
            f"{q1_total(r)!r}" for r in q1
            if not math.isclose(costs[r["method"]][ALL], q1_total(r), rel_tol=TOL, abs_tol=TOL)]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_q1(path: Path) -> list[dict]:
    out = []
    for r in A.read_csv(path):
        row: dict[str, Any] = {"rank": int(r["rank"]), "method": r["method"], "kind": r["kind"],
                               "accuracy": float(r["accuracy"])}
        for c in list(SECONDS) + ["used_calls"]:
            row[f"{c}_per_lecture"] = fnum(r[f"{c}_per_lecture"])
        out.append(row)
    return out


def q2_cells(long: Sequence[Mapping[str, Any]]) -> dict[Key, dict[str, Any]]:
    """Q2's long rows (analysis.q2_rows, or its CSV) by level: (axis, level) ->
    {"n": lectures, "acc": {method: accuracy}, "rank": {method: rank}}."""
    out: dict[Key, dict[str, Any]] = {}
    for r in long:
        c = out.setdefault((r["axis"], r["level"]), {"n": int(r["n"]), "acc": {}, "rank": {}})
        c["acc"][r["method"]] = float(r["accuracy"])
        c["rank"][r["method"]] = int(r["rank"])
    return out


def read_q2(axes: Sequence[str]) -> dict[Key, dict[str, Any]]:
    return q2_cells([r for axis in axes for r in A.read_csv(A.OUT / f"q2_{axis}_long.csv")])


def load(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Everything the tables and figures read, from the files of the module docstring."""
    methods = sorted(METHODS)
    d: dict[str, Any] = {"methods": methods, "kinds": {m: F.kind(m) for m in methods}}
    d["q1"] = read_q1(A.OUT / "q1_ranking.csv")
    d["order"] = A.level_order(A.CATEGORIES / "level_counts.csv")
    d["q2"] = read_q2(A.AXES)
    d["rows"] = {p.name[:-len(".csv.gz")]: A.read_csv(p)
                 for p in sorted((A.FINAL / "methods").glob("*.csv.gz"))}
    cats = {(c["corpus"], c["lecture_id"]): c
            for c in A.read_csv(A.CATEGORIES / "document_categories.csv")}
    d["cats"] = cats
    first = next(iter(d["rows"].values()))
    keys = [(r["corpus"], r["lecture_id"]) for r in first]
    d["levels"] = {a: {k: cats[k][a] for k in keys} for a in A.AXES}
    d["load_s"] = A.model_load_s(methods)
    d["q4"] = A.read_csv(A.OUT / "q4_wrong_important.csv")
    d["selection"] = A.read_csv(SELECTION)
    d["attributes"] = {(r["corpus"], r["lecture_id"]): r for r in A.read_csv(ATTRIBUTES)}
    d["cut_points"] = json.loads((A.CATEGORIES / "cut_points.json").read_text(encoding="utf-8"))
    d["tuned"] = T.load_tuned()
    d["costs"] = category_costs(d["rows"], d["kinds"], d["load_s"], d["levels"])
    return d


def check(d: Mapping[str, Any]) -> list[str]:
    """Checks 1-4 of the module docstring; the problems (empty = all passed)."""
    out = []
    want = set(d["methods"])
    for name, have in (("Q1", {r["method"] for r in d["q1"]}),
                       ("Q4", {r["method"] for r in d["q4"]}),
                       ("per-lecture tables", set(d["rows"]))):
        if have != want:
            out.append(f"methods: {name} has {sorted(have ^ want)} different")
    for k, c in d["q2"].items():
        if set(c["acc"]) != want:
            out.append(f"methods: Q2 {k} has {sorted(set(c['acc']) ^ want)} different")
    out += [f"names: no display name for method {m}" for m in want - set(N.METHOD_NAMES)]
    out += [f"names: no display name for level {k}" for k in d["q2"] if k not in N.LEVEL_NAMES]
    test = {(r["corpus"], r["lecture_id"]) for r in d["selection"] if r["set"] == "test"}
    for m, rs in d["rows"].items():
        if {(r["corpus"], r["lecture_id"]) for r in rs} != test or len(rs) != len(test):
            out.append(f"lectures: {m}'s per-lecture table is not the {len(test)} test lectures")
    out += check_costs(d["q1"], d["costs"])
    units = sum(int(r["n_units"]) for r in next(iter(d["rows"].values())))
    q4_units = {int(r["all_units"]) for r in d["q4"]}
    if q4_units != {units}:
        out.append(f"units: the test lectures hold {units} units, Q4's all documents {q4_units}")
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def table_env(caption: str, label: str, colspec: str, body: Sequence[str],
              tabcolsep: Optional[str] = None, stretch: Optional[str] = None) -> str:
    """A whole table environment: caption above (the journal's rule), booktabs rules.

    The table is set as wide as the line (tabular* with the space between the columns
    stretched), so that in the main text and in the appendices alike the caption, which
    the class sets as wide as the table, and the table start at the left edge of the
    text. The font is the class's table font (8 bp), never scaled: every table is laid
    out (column space, wrapped text columns, two-line headers) to fit the 119 mm text
    width, or the rotated page's width in Appendix C. The colspec starts with @{}, which
    is where the stretch goes. stretch sets \\arraystretch (rows closer than the font's
    line spacing where a table must fit one page)."""
    assert colspec.startswith("@{}"), colspec
    lines = [r"\begin{table}[t]", r"\centering"]
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    if stretch:
        lines.append(rf"\renewcommand{{\arraystretch}}{{{stretch}}}")
    spec = r"@{\extracolsep{\fill}}" + colspec[3:]
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular*}}{{\linewidth}}{{{spec}}}", r"\toprule", *body,
              r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def row(cells: Sequence[Any]) -> str:
    return " & ".join(str(c) for c in cells) + r" \\"


#: The corpora: a short description of the source (from data/raw/<corpus>/SOURCE.json and
#: the corpora's papers) and the slide's unit of annotation, in lines short enough that
#: the table fits the text width with room between its number columns.
CORPUS_SOURCE = {
    "mavils": ("University lectures,", "PDF decks"),
    "m3av_a": ("Academic talks,", "slides as video", "segments (half A)"),
    "m3av_b": ("Academic talks,", "slides as video", "segments (half B)"),
    "mcd": ("Conference", "presentations,", "slide images"),
    "slide_highlight": ("Conference talks,", "slide images"),
}


#: What a lecture is, for the corpora table's caption: per corpus, and for M³AV per source
#: (the prefix of the lecture id; config/attributes.toml names the five sources). The paper
#: calls all of them lectures; the caption says how many are course lectures.
TALK_KIND = {"mavils": "course", "mcd": "conference", "slide_highlight": "conference",
             "CHI": "conference", "Ubi": "conference", "NIH": "invited", "IPP": "course",
             "MLS": "course"}


def talk_kind(corpus: str, lecture_id: str) -> str:
    """The lecture's kind; "other" for a source TALK_KIND does not know (the caption then
    says how many), so that a rerun on other data still builds the table."""
    return TALK_KIND.get(lecture_id.split("-")[0] if corpus.startswith("m3av") else corpus,
                         "other")


def slide_text_source(corpus: str, cats: Mapping[Key, Mapping[str, str]]) -> list[str]:
    """The corpus's slide-text source, from the lectures' own: the usual one, and any
    other with its number of decks (MaViLS: one scanned deck is Tesseract OCR)."""
    count: dict[str, int] = defaultdict(int)
    for (c, _), r in cats.items():
        if c == corpus:
            count[r["text_source"]] += 1
    main, *rest = sorted(count, key=lambda s: -count[s])
    return [N.level_name("text_source", main)] + [
        f"({count[s]} deck{'s' if count[s] > 1 else ''}: {N.level_name('text_source', s)})"
        for s in rest]


def tab_corpora(d: Mapping[str, Any]) -> str:
    sel = d["selection"]
    rows0 = next(iter(d["rows"].values()))
    units = {(r["corpus"], r["lecture_id"]): int(r["n_units"]) for r in rows0}
    pool: dict[str, int] = defaultdict(int)
    for c, _ in d["cats"]:
        pool[c] += 1
    body = [r" & & & \multicolumn{3}{c}{Lectures} & \multicolumn{3}{c}{Test lectures} \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
            row(["Corpus", "Source", "Slide text", "Pool", "Tuning", "Test", "Units",
                 head_cell("Mean", "slides"), head_cell("Mean", "units")]), r"\midrule"]
    tot = {"pool": 0, "tuning": 0, "test": 0, "units": 0, "slides": 0}
    for corpus in N.CORPUS_NAMES:
        tun = sum(1 for r in sel if r["corpus"] == corpus and r["set"] == "tuning")
        test = [(r["corpus"], r["lecture_id"]) for r in sel
                if r["corpus"] == corpus and r["set"] == "test"]
        u = sum(units[k] for k in test)
        s = sum(int(d["attributes"][k]["n_slides"]) for k in test)
        body.append(row([tex(N.CORPUS_NAMES[corpus]), lines_cell(CORPUS_SOURCE[corpus]),
                         tex(" ".join(slide_text_source(corpus, d["cats"]))),
                         f"{pool[corpus]:,}", tun,
                         len(test), f"{u:,}", f"{s / len(test):.1f}", f"{u / len(test):.1f}"]))
        for k, v in (("pool", pool[corpus]), ("tuning", tun), ("test", len(test)),
                     ("units", u), ("slides", s)):
            tot[k] += v
    body += [r"\midrule",
             row(["All", "", "", f"{tot['pool']:,}", tot["tuning"], tot["test"],
                  f"{tot['units']:,}", f"{tot['slides'] / tot['test']:.1f}",
                  f"{tot['units'] / tot['test']:.1f}"])]
    kinds: dict[str, int] = defaultdict(int)
    for r in sel:
        if r["set"] == "test":
            kinds[talk_kind(r["corpus"], r["lecture_id"])] += 1
    cap = ("The corpora. Source: what the lectures are. Slide text: where the text of the "
           "slides comes from. Lectures: in the pool after preparation (at most 40 slides "
           "each), in the tuning set and in the test set. Test lectures: their transcript "
           "units (each with its true slide), and the mean slides and mean units per "
           "test lecture. M$^3$AV is split into two halves with different lectures (A and B). "
           f"Of the {tot['test']} test lectures, {kinds['course']} are course lectures "
           "(MaViLS; in M$^3$AV a clinical-research course and Oxford Mathematics), "
           f"{kinds['invited']} invited science talks (the NIH Director's lectures in "
           f"M$^3$AV) and {kinds['conference']} conference paper talks (CHI and UbiComp in "
           "M$^3$AV, MCD, Slide-Highlight)"
           + (f"; {kinds['other']} are of another kind." if kinds["other"] else "."))
    # the source in its own lines; the slide text wrapped to at most three
    return table_env(cap, "tab:corpora", r"@{}ll" + text_col("59pt") + "rrrrrr@{}", body,
                     tabcolsep="3pt")


def _cut(v: float, integer: bool) -> str:
    return f"{v:.0f}" if integer else f"{v:.1f}"


#: Per attribute: how it is measured, in a few words (docs/ATTRIBUTES.md, docs/CATEGORIES.md).
MEASURED = {
    "text_source": "where the slide text comes from",
    "length": "number of slides (after the 40-slide cap); terciles",
    "images": "\\% of the lecture's total slide area covered by pictures (layout "
              "detector); terciles",
    "tables": "\\% of the slide area covered by tables (layout detector)",
    "noise": "\\% of slide-text words not in an English word list; terciles",
    "text_density": "mean slide-text words per slide; terciles",
    "discipline": "OECD FOS 2007 area, chosen by Qwen3-30B-A3B-Instruct-2507 from six "
                  "slides, venue, transcript start",
}
#: The rule of each non-tercile level.
LEVEL_RULE = {
    ("text_source", "pdf_text_layer"): "the PDF's text layer",
    ("text_source", "corpus_corrected_ocr"): "the corpus's hand-corrected OCR",
    ("text_source", "tesseract_ocr"): "our Tesseract OCR of the slide images",
    ("tables", "none"): "0\\,\\%",
    ("tables", "some"): "$>$ 0\\,\\%",
    ("discipline", "Computer and information sciences"): "field 1.2, computer and "
                                                         "information sciences",
    ("discipline", "Natural sciences (other)"): "area 1 without field 1.2",
    ("discipline", "Engineering and technology"): "area 2, engineering and technology",
    ("discipline", "Medical and health sciences"): "area 3, medical and health sciences",
    ("discipline", "Agricultural sciences"): "area 4, agricultural sciences",
    ("discipline", "Social sciences"): "area 5, social sciences",
    ("discipline", "Humanities"): "area 6, humanities",
    ("discipline", "other"): "areas with fewer than 20 lectures",
}


def level_rule(axis: str, level: str, cuts: Mapping[str, Any]) -> str:
    if (axis, level) in LEVEL_RULE:
        return LEVEL_RULE[(axis, level)]
    c = cuts["cut_points"][axis]
    integer = axis == "length"
    unit = {"length": " slides", "images": "\\,\\%", "noise": "\\,\\%",
            "text_density": " words"}[axis]
    q1, q2 = _cut(c["q1"], integer), _cut(c["q2"], integer)
    i = c["levels"].index(level)
    if integer:  # whole numbers: the middle level is q1 + 1 .. q2
        return [f"$\\le$ {q1}{unit}", f"{int(c['q1']) + 1}--{q2}{unit}",
                f"$\\ge$ {int(c['q2']) + 1}{unit}"][i]
    return [f"$\\le$ {q1}{unit}", f"$>$ {q1} and $\\le$ {q2}{unit}", f"$>$ {q2}{unit}"][i]


def tab_categories(d: Mapping[str, Any]) -> str:
    body = [row(["Level", "Rule", "Test lectures"]), r"\midrule"]
    full = text_col(r"\linewidth")
    for i, axis in enumerate(A.AXES):
        if i:
            body.append(r"\addlinespace")
        # the attribute's line spans the table and wraps at the line's width
        body.append(rf"\multicolumn{{3}}{{@{{}}{full}@{{}}}}"
                    rf"{{\textbf{{{tex(N.AXIS_NAMES[axis])}}}: {MEASURED[axis]}}} \\")
        for level in d["order"][axis]:
            body.append(row([r"\quad " + tex(N.level_name(axis, level)),
                             level_rule(axis, level, d["cut_points"]),
                             d["q2"][(axis, level)]["n"]]))
    cap = ("The seven attributes of a lecture and their levels. Each test lecture has one "
           "level of every attribute. Terciles: two cut points computed once over all "
           f"{len(d['cats']):,} pooled lectures (the values at the 1/3 and 2/3 positions); "
           "a value on a cut point goes to the lower level. Discipline: the broad OECD "
           "FOS 2007 area, with computer and information sciences (field 1.2) as a level "
           "of its own and the rest of area 1 as natural sciences. Test lectures: the "
           "number of the 140 test lectures at that level.")
    return table_env(cap, "tab:categories", r"@{}llr@{}", body)


def compact_settings(t: T.Tuned) -> str:
    """The tuned settings in a few words: the shared DP ones abbreviated (fwd, back:
    costs; stay: a reward; prior: a weight; methods/common.py DPConfig), then the
    method's own as name value (fusion's own pos_weight, the weight of its LLM prior, is
    abbreviated as prior too). Names and text values are
    written with spaces ("gap segment 0.4"), since the text never uses the code names."""
    short = {"forward_penalty": "fwd", "backward_penalty": "back", "stay_bonus": "stay",
             "pos_weight": "prior"}

    def val(v: Any) -> str:
        if v is None:
            return "none"
        if isinstance(v, bool):
            return "on" if v else "off"
        return f"{v:g}" if isinstance(v, float) else str(v).replace("_", " ")

    def name(k: str) -> str:
        return short.get(k, k).replace("_", " ")

    parts = [f"{name(k)} {val(v)}" for k, v in sorted(t.cfg.items(),
                                                       key=lambda kv: list(short).index(kv[0])
                                                       if kv[0] in short else 9)]
    parts += [f"{name(k)} {val(v)}" for k, v in sorted(t.settings.items())]
    return ", ".join(parts)


def method_similarity(m: str, tuned: Mapping[str, T.Tuned]) -> str:
    """The similarity tables the method reads: its tuned measure, or its members'; the
    three single-answer LLM methods read hybrid only for the fallback of an unusable
    answer."""
    measures = sorted(F.method_measures(m, tuned))
    if not measures:
        return "--"
    text = "; ".join(N.MEASURE_NAMES[x] for x in measures)
    if METHODS[m].uses_llm and F.kind(m) == "LLM":
        text += " (fallback)"
    return text


def method_members(m: str) -> str:
    if m == "fusion":
        return N.method_name("hierarchical") + " (its prior)"
    if m in MEMBERS:
        return ", ".join(N.method_name(x) for x in MEMBERS[m])
    return "--"


def tab_methods(d: Mapping[str, Any]) -> str:
    """The method table, a longtable over two rotated pages, as wide as the rotated line
    (the space between the columns stretched, as in table_env) so that the appendix
    heading, the caption and the table share the left edge. The p columns are ragged
    right in the column spec (array's >{...}; the preamble must load array), and the
    head repeated on the second page says it is the table continued. The font is the
    class's table font (8 bp), as in every other table. Combinations: the size of each
    method's tuning grid (tune.load_grid), the measures included where the grid tries
    them, so that the column adds up to the total the text and the pipeline figure give
    (check_tuning checks every grid against the combinations in its tuning runs)."""
    tuned = d["tuned"]
    grids = T.load_grid()

    def trow(cells: Sequence[str]) -> str:
        # \tabularnewline, not \\: safe after a ragged-right p cell
        return " & ".join(cells) + r" \tabularnewline"

    rag = r">{\raggedright\arraybackslash}"
    head = [trow(["Method", "Kind", "Similarity", "Members", "Tuned settings",
                  "Combinations"])]
    # a fixed \LTleft and \LTright make \extracolsep{\fill} stretch the table to the line
    lines = [r"\begingroup\tablebodyfont", r"\setlength{\tabcolsep}{4pt}",
             r"\setlength{\LTcapwidth}{\linewidth}", r"\setlength{\LTleft}{0pt}",
             r"\setlength{\LTright}{0pt}",
             rf"\begin{{longtable}}{{@{{\extracolsep{{\fill}}}}{NAME_COL}l{rag}p{{80pt}}"
             rf"{rag}p{{110pt}}{rag}p{{156pt}}r@{{}}}}",
             r"\caption{The 31 methods by family (the grouping of the repository's "
             r"registry). Kind: non-LLM, LLM (asks a language model), or mixed (a "
             r"combination of both). Similarity: the similarity tables the method reads "
             r"(fallback: only when the model's answer is unusable). Members: the methods "
             r"a combined method runs, each at its own tuned settings. Tuned settings: "
             r"the values chosen on the 60 tuning lectures; fwd and back are the shared "
             r"costs per slide moved forward and back, stay the reward for staying on a "
             r"slide from one unit to the next, prior the weight of the positional prior "
             r"(Section~\ref{sec:methods}); the others are the method's own. Combinations: "
             r"the setting combinations in the method's tuning grid, times the eight "
             r"measures where the method tunes its measure; they add up to the "
             rf"{_int(sum(g.n_combos() for g in grids.values()))} combinations of "
             r"Section~\ref{sec:tuning}.}",
             r"\label{tab:methods}\\", r"\toprule", *head, r"\midrule", r"\endfirsthead",
             r"\multicolumn{6}{@{}l}{\itshape Table~\thetable{} (continued)}"
             r"\tabularnewline[2pt]",
             r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    first = True
    for group in GROUPS:
        title = N.FAMILY_NAMES[group]       # the text's family names, not the registry's
        if not first:
            lines.append(r"\addlinespace")
        first = False
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{tex(title)}}}}} \tabularnewline")
        for m, meth in METHODS.items():
            if meth.group != group:
                continue
            s = compact_settings(tuned[m])
            if m in MEMBERS or m == "fusion":
                s = (s + "; " if s else "") + "members tuned"
            lines.append(trow([tex(N.method_name(m)), F.kind(m),
                               tex(method_similarity(m, tuned)),
                               tex(method_members(m)), tex(s) if s else "--",
                               _int(grids[m].n_combos())]))
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def tab_q1(d: Mapping[str, Any]) -> str:
    body = [r" & & & & \multicolumn{5}{c}{Per lecture} \\", r"\cmidrule(lr){5-9}",
            row(["Rank", "Method", "Kind", head_cell("Accuracy", "(\\%)"),
                 head_cell("Compute", "(s)"), head_cell("Tables", "(s)"),
                 head_cell("LLM", "calls"), head_cell("LLM", "(s)"), head_cell("Total", "(s)")]),
            r"\midrule"]
    for r in d["q1"]:
        calls = r["used_calls_per_lecture"]
        body.append(row([r["rank"], tex(N.method_name(r["method"])), r["kind"],
                         f"{r['accuracy']:.1f}", compute_seconds(r["compute_s_per_lecture"]),
                         seconds(r["build_cost_s_per_lecture"]),
                         "--" if calls is None else f"{calls:.1f}",
                         seconds(r["used_seconds_per_lecture"]), seconds(q1_total(r))]))
    n = len(next(iter(d["rows"].values())))
    cap = (f"Q1: the {len(d['q1'])} methods on the {n} test lectures, by mean accuracy (the "
           "share of a lecture's transcript units placed on their true slide, averaged over "
           "the lectures). Rank 1 is the highest accuracy; ranks are by the unrounded "
           "accuracy. Per lecture, each method is costed as if it ran alone (a combined "
           "method carries its members' costs): Compute, its own run time; Tables, building "
           "the similarity tables it reads, model loads spread over the lectures; LLM calls "
           "and LLM, the successful language-model calls and their own duration; Total, the "
           "sum of the three times. Tables 0.0: reads no similarity table. --: not part of "
           "that kind's cost (an LLM method is charged its calls, not its own run time).")
    return table_env(cap, "tab:q1", r"@{}rllrrrrrr@{}", body, tabcolsep="3pt")


def extreme(cell: Mapping[str, Any], lowest: bool) -> list[str]:
    """The methods with the highest (rank 1) or lowest (largest rank) accuracy in a
    level, in name order; more than one only on a tie."""
    ranks = cell["rank"]
    target = max(ranks.values()) if lowest else min(ranks.values())
    return sorted(m for m, rk in ranks.items() if rk == target)


def display_tie(cell: Mapping[str, Any], lowest: bool) -> bool:
    """Whether the next method after the extreme one(s) prints the same accuracy to one
    decimal, so that a reader of the full table sees a tie the ranks do not make."""
    ms = extreme(cell, lowest)
    others = [a for m, a in cell["acc"].items() if m not in ms]
    nxt = min(others) if lowest else max(others)
    return f"{nxt:.1f}" == f"{cell['acc'][ms[0]]:.1f}"


def tab_q3(d: Mapping[str, Any]) -> str:
    q1 = d["q1"]
    cells: dict[Key, dict[str, Any]] = {ALL: {"n": len(next(iter(d["rows"].values()))),
                                              "acc": {r["method"]: r["accuracy"] for r in q1},
                                              "rank": {r["method"]: r["rank"] for r in q1}}}
    cells.update(d["q2"])
    costs = d["costs"]

    def side(k: Key, lowest: bool) -> list[str]:
        ms = extreme(cells[k], lowest)
        mark = r"$^\dagger$" if display_tie(cells[k], lowest) else ""
        return [" / ".join(tex(N.method_name(m)) for m in ms),
                " / ".join(f"{cells[k]['acc'][m]:.1f}" for m in ms) + mark,
                " / ".join(seconds(costs[m][k]) for m in ms)]

    body = [r" & & \multicolumn{3}{c}{Highest accuracy} & \multicolumn{3}{c}{Lowest accuracy} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            row(["Level", over_left("Lectures"), "Method",
                 over_left(head_cell("Accuracy", "(\\%)")), head_cell("Cost", "(s)"), "Method",
                 over_left(head_cell("Accuracy", "(\\%)")), head_cell("Cost", "(s)")]),
            r"\midrule",
            row(["All lectures", cells[ALL]["n"], *side(ALL, False), *side(ALL, True)])]
    # the attributes as in the category table: a bold label and a little space, no rule
    for axis in A.AXES:
        body += [r"\addlinespace",
                 rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{tex(N.AXIS_NAMES[axis])}}}}} \\"]
        for level in d["order"][axis]:
            k = (axis, level)
            body.append(row([r"\quad " + tex(N.level_name(axis, level)), cells[k]["n"],
                             *side(k, False), *side(k, True)]))
    cap = ("Q3: at every level of every attribute, the method with the highest and the one "
           "with the lowest mean accuracy over the level's test lectures, each with its total "
           "seconds per lecture there (compute, similarity tables and LLM, as in "
           "Table~\\ref{tab:q1}, averaged over the level's lectures).")
    if any(len(extreme(c, low)) > 1 for c in cells.values() for low in (False, True)):
        cap += " A slash lists methods tied in accuracy."
    if any(display_tie(c, low) for c in cells.values() for low in (False, True)):
        cap += (r" $^\dagger$: the next method of the level (Table~\ref{tab:q2-full}) has the "
                "same accuracy to one decimal; the one shown is ahead before rounding.")
    # the level wraps (Corrected OCR (corpus)); every method name stays on one line, the
    # headers over the numbers reaching left over the space of the column before them
    return table_env(cap, "tab:q3", "@{}" + text_col("76pt", hang=True) + "rlrrlrr@{}", body,
                     tabcolsep="3pt")


def q4_columns() -> list[tuple[str, Optional[str], Optional[str]]]:
    return list(Q.COLUMNS)


def tab_q4(d: Mapping[str, Any]) -> str:
    cols = q4_columns()
    units = {c: int(d["q4"][0][f"{c}_units"]) for c, _, _ in cols}
    body = [" & & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{tex(N.AXIS_NAMES[a])}}}"
                                 for a, k in (("length", 3), ("images", 3), ("tables", 2)))
            + r" \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}",
            row(["Method", "All"] + [tex(N.level_name(a, lv)) for _, a, lv in cols[1:]]),
            row(["", *[f"({units[c]:,})" for c, _, _ in cols]]), r"\midrule"]
    for r in d["q4"]:
        body.append(row([tex(N.method_name(r["method"]))]
                        + [f"{float(r[f'{c}_pct']):.1f}" for c, _, _ in cols]))
    cap = ("Q4: the important transcript units (labelled core) that each method placed on "
           "the wrong slide, as a percentage of all units of the test lectures in the "
           "column, pooled over the units. All: all test lectures; then the levels of "
           "length, picture share and tables. Units per column in brackets. Methods from "
           "the fewest to the most over all lectures.")
    # set on a rotated page (Appendix F): with every name on one line and the columns
    # 2 em apart, the unit counts make it wider than the upright text
    return table_env(cap, "tab:q4", r"@{}lrrrrrrrrr@{}", body, tabcolsep=APPENDIX_SEP,
                     stretch="0.9")


#: The attributes' names over the 22-level table's narrow columns, where they differ
#: from N.AXIS_NAMES (so the group header is not wider than its columns).
Q2_FULL_GROUP = {"text_source": "Slide text"}
#: A rotated level name longer than this many characters (Engineering, one word) is set
#: in two lines, so that the header row is no taller than Engineering is long.
Q2_FULL_LABEL_CHARS = 11


def rotated_label(name: str) -> str:
    """A level name turned upright over its column; a long one in two lines, broken at
    the space that makes the longer line shortest, closer than the font's line spacing
    so that the column stays as narrow as its numbers."""
    text = tex(name)
    words = name.split(" ")
    if len(name) > Q2_FULL_LABEL_CHARS and len(words) > 1:
        cut = min(range(1, len(words)), key=lambda i: max(len(" ".join(words[:i])),
                                                          len(" ".join(words[i:]))))
        text = (r"\renewcommand{\arraystretch}{0.8}\begin{tabular}{@{}l@{}}"
                + tex(" ".join(words[:cut])) + r"\\" + tex(" ".join(words[cut:]))
                + r"\end{tabular}")
    return rf"\rotatebox{{90}}{{{text}}}"


def tab_q2_full(d: Mapping[str, Any]) -> str:
    """Q2 in full: every method in all 22 levels, in one tabular sized for a landscape
    page (the text block's 552.7 pt height becomes its width, the 372 pt width its
    height). The level names are rotated, so a column is only as wide as its numbers;
    the rows are a little closer than the font's line spacing, so that the table, its
    caption and the appendix heading fit the rotated page; methods in Q1 order."""
    axes = A.AXES
    levels = [(a, lv) for a in axes for lv in d["order"][a]]
    spans, j = [], 2
    for a in axes:
        k = len(d["order"][a])
        spans.append((a, j, j + k - 1))
        j += k
    body = [" & " + " & ".join(rf"\multicolumn{{{b - a + 1}}}{{c}}"
                               rf"{{{tex(Q2_FULL_GROUP.get(x, N.AXIS_NAMES[x]))}}}"
                               for x, a, b in spans) + r" \\",
            "".join(rf"\cmidrule(lr){{{a}-{b}}}" for _, a, b in spans),
            row(["Method"] + [rotated_label(N.short_level_name(a, lv)) for a, lv in levels]),
            row([""] + [f"({d['q2'][k]['n']})" for k in levels]), r"\midrule"]
    for r in d["q1"]:
        m = r["method"]
        body.append(row([tex(N.method_name(m))] + [f"{d['q2'][k]['acc'][m]:.1f}" for k in levels]))
    cap = ("Q2 in full: the mean accuracy (\\%) of every method over the test lectures of "
           f"each of the {len(levels)} levels of the seven attributes. Methods in the order of "
           "Table~\\ref{tab:q1}; test lectures per level in brackets. Slide text: "
           "Corrected OCR is the corpus's hand-corrected OCR.")
    return table_env(cap, "tab:q2-full", "@{}l" + "r" * len(levels) + "@{}", body,
                     tabcolsep="2pt", stretch="0.78")


# --------------------------------------------------------------------------- #
# Accuracy per similarity measure (the tuning runs)
# --------------------------------------------------------------------------- #

#: The methods shown per measure: the baseline, the MaViLS decoder, two monotone decoders
#: of the top band, and the lowest method of Q1 that reads one tuned table (whether the
#: measure is what holds it back).
MEASURE_METHODS = ("argmax", "dp", "dtw", "needleman_wunsch", "texttiling")
#: Their names over the table's narrow columns, broken into lines where they are long.
MEASURE_HEADS = {"dp": ("Viterbi", "DP"), "needleman_wunsch": ("Needleman–", "Wunsch"),
                 "texttiling": ("Text-", "Tiling")}


def measure_data(d: Mapping[str, Any]) -> dict[str, Any]:
    """What the measure table shows, none of it a new statistic:
      best    per method whose tuning grid tries more than one measure, per measure, the
              highest mean accuracy (%) of its combinations with that measure on the
              tuning lectures, scored as step 11 scores them (select_settings.combo_stats:
              each lecture once, a run that is not ok counts 0);
      chosen  per such method, the measure step 11 chose (config/tuned_settings.toml);
      build   per measure, the seconds to build its table per test lecture, with the
              tables it is built from and their model loads, as step 13 charges a method
              that reads only this measure (final_report: with_deps, cost_s, load_s).
    The final run used each method's chosen measure only, so the per-measure accuracies
    exist for the tuning lectures alone."""
    import select_settings as S
    lectures = set(T.set_lectures(T.TUNING_SET))
    grids = T.load_grid()
    best: dict[str, dict[str, float]] = {}
    for m in sorted(METHODS):
        if len(grids[m].measures) < 2:
            continue
        per: dict[str, float] = {}
        for st in S.combo_stats(T.read_runs(m), lectures).values():
            x = st.values[("measure", "measure")]
            per[x] = max(per.get(x, -1.0), 100 * st.mean_acc)
        best[m] = per
    deps, load_s = F.read_times()
    costs = F.read_table_costs(T.tables_path(T.TEST_SET))
    test = [(r["corpus"], r["lecture_id"]) for r in d["selection"] if r["set"] == "test"]
    build = {}
    for x in T.MEASURES:
        ms = F.with_deps({x}, deps)
        build[x] = (sum(costs[(y, c, lec)] for y in ms for c, lec in test)
                    + sum(load_s.get(y, 0.0) for y in ms)) / len(test)
    return {"best": best, "chosen": {m: d["tuned"][m].measure for m in best},
            "build": build, "n_tuning": len(lectures),
            "excluded": [r["method"] for r in d["q1"] if r["method"] not in best]}


def check_measures(md: Mapping[str, Any], d: Mapping[str, Any]) -> list[str]:
    """The measure table's checks: every multi-measure method has a score for all eight
    measures, its chosen measure is within step 11's tie of its best (0.5 points, plus
    rounding), and a measure's build seconds equal Q1's Tables column for every method
    that reads that measure alone."""
    out = []
    for m, per in md["best"].items():
        if set(per) != set(T.MEASURES):
            out.append(f"measures: {m} has tuning runs for {sorted(per)}, not all eight")
        elif per[md["chosen"][m]] < max(per.values()) - 0.5 - 1e-9:
            out.append(f"measures: {m} chose {md['chosen'][m]} ({per[md['chosen'][m]]:.3f}), "
                       f"more than 0.5 points below its best ({max(per.values()):.3f})")
    q1 = {r["method"]: r["build_cost_s_per_lecture"] for r in d["q1"]}
    for m in d["methods"]:
        ms = F.method_measures(m, d["tuned"])
        if len(ms) == 1 and q1.get(m) is not None:
            x = next(iter(ms))
            if not math.isclose(md["build"][x], q1[m], rel_tol=1e-6, abs_tol=1e-6):
                out.append(f"measures: {x} costs {md['build'][x]!r} s per lecture, "
                           f"Q1's Tables of {m} {q1[m]!r}")
    return out


def tab_measures(md: Mapping[str, Any]) -> str:
    """Accuracy per similarity measure: one row per measure, from the cheapest table to
    the most expensive; the highest number of each accuracy column in bold."""
    best, build = md["best"], md["build"]
    order = sorted(T.MEASURES, key=lambda x: build[x])
    chosen = defaultdict(int)
    for x in md["chosen"].values():
        chosen[x] += 1
    cols = [[best[m][x] for x in order] for m in MEASURE_METHODS]
    cols.append([max(best[m][x] for m in best) for x in order])

    def acc(v: float, col: Sequence[float]) -> str:
        return rf"\textbf{{{v:.1f}}}" if f"{v:.1f}" == f"{max(col):.1f}" else f"{v:.1f}"

    k = len(cols)
    body = [rf" & & & \multicolumn{{{k}}}{{c}}{{Highest mean accuracy on the "
            rf"{md['n_tuning']} tuning lectures (\%)}} \\",
            rf"\cmidrule(l){{4-{3 + k}}}",
            row(["Measure", head_cell("Table", "(s)"), head_cell("Chosen", "by")]
                + [head_cell(*(tex(w) for w in MEASURE_HEADS.get(m, (N.method_name(m),))))
                   for m in MEASURE_METHODS]
                + [head_cell("Any of", f"the {len(best)}")]), r"\midrule"]
    for i, x in enumerate(order):
        body.append(row([tex(N.MEASURE_NAMES[x]), seconds(build[x]), chosen[x]]
                        + [acc(c[i], c) for c in cols]))
    excluded = _join([tex(N.method_name(m)) for m in md.get("excluded", [])])
    cap = ("Accuracy per similarity measure, on the tuning lectures: the test lectures were "
           "run with each method's chosen measure only. Measures from the cheapest table to "
           "the most expensive. Table: seconds to build the measure's table per test lecture, "
           "with the tables it is built from and the model load, as in the Tables column of "
           f"Table~\\ref{{tab:q1}}. Chosen by: of the {len(best)} methods that read one "
           "tuned table, how many picked this measure in tuning"
           + (f". The other {len(md['excluded'])} ({excluded}) pick no measure: they read "
              "no table, only the fixed embed--BM25 hybrid as a fallback, or their members' "
              "tables" if md.get("excluded") else "")
           + ". Then, for five methods (Argmax, the baseline; Viterbi DP, the MaViLS "
           "decoder; DTW and Needleman--Wunsch, two monotone decoders of the top band; "
           "TextTiling, the lowest method of Q1 that reads one tuned table), the highest "
           "mean accuracy over their tuning settings with that measure (each lecture counts "
           f"once); Any of the {len(best)}: the highest over all {len(best)} methods. Bold: "
           "the highest in the column.")
    return table_env(cap, "tab:measures", "@{}l" + "r" * (2 + k) + "@{}", body,
                     tabcolsep="3pt")


# --------------------------------------------------------------------------- #
# The numbers the text quotes, and three appendix tables: the true paths, Q4 against
# importance-blind errors, the LLM fallbacks, tuning against test, the prompts
# --------------------------------------------------------------------------- #

#: The methods Q4's text reads as one group: those in which an LLM alone gives the
#: alignment and those that give each slide at most one piece of the transcript, in
#: slide order (the paper's Q1 paragraph "Below the baseline").
ORDER_FORCED = ("llm_single_call", "hierarchical", "llm_monotone", "llm_monotone_constrained",
                "wbs2", "texttiling", "texttiling_dp")
#: The corpora whose ground truth is a file pairing in slide order, so their true path
#: can never go back (a back-jump there would mean the pairing was misread).
IN_ORDER_CORPORA = ("mcd", "slide_highlight")
#: The LLM methods whose own answer can fall back (methods/llm.py): to the argmax of the
#: table (the single-call methods), to dtw's alignment (re-rank), or, per call, to the
#: middle packet or an even spread (hierarchical). Ensemble and fusion only inherit their
#: member's fallbacks.
FALLBACK_METHODS = ("llm_single_call", "llm_monotone", "llm_monotone_constrained",
                    "llm_rerank_kbest", "hierarchical")
#: The methods in which an LLM alone gives the alignment (the first four of
#: ORDER_FORCED): the abstract's and the introduction's group, not the family "LLM reads
#: the text", which also holds no top-band method but is named by how the LLM is used.
LLM_ALONE = ORDER_FORCED[:4]
#: The methods whose models Score fusion adds (soft Needleman-Wunsch stands for the hard
#: one, which is in Q1): docs/METHODS.md, score_fusion.
FUSION_MEMBERS = ("needleman_wunsch", "softdtw", "cohesion_dp")


def unit_data(methods: Sequence[str], predictions: Path = Q.PREDICTIONS,
              labels: Path = Q.LABELS) -> dict[str, Any]:
    """Step 13's slide of every unit (step 16's replay) and step 15's labels, reduced to
    what the text needs: per lecture the true slides in unit order (every method's file
    must carry the same ones), per method the wrong units, the wrong core units, the
    wrong units one slide off and the summed slide distance of the wrong units, and
    the core units of all lectures, and the units per label. Read once here, so the
    back-jumps and Q4's expectation come from the same rows that Q4 itself was computed
    from."""
    label = {(r["corpus"], r["lecture_id"], int(r["unit_index"])): int(r["label"])
             for r in A.read_csv(labels)}
    core = {u: x == 2 for u, x in label.items()}
    counts: dict[int, int] = defaultdict(int)
    for x in label.values():
        counts[x] += 1
    truth: Optional[dict[Key, list[int]]] = None
    wrong: dict[str, int] = {}
    wrong_core: dict[str, int] = {}
    near: dict[str, int] = {}
    distance: dict[str, int] = {}
    for m in methods:
        by: dict[Key, dict[int, int]] = defaultdict(dict)
        w = wc = nr = dist = 0
        for r in A.read_csv(predictions / f"{m}.csv.gz"):
            i, t, p = int(r["unit_index"]), int(r["true_slide"]), int(r["pred_slide"])
            by[(r["corpus"], r["lecture_id"])][i] = t
            if p != t:
                w += 1
                wc += core[(r["corpus"], r["lecture_id"], i)]
                nr += abs(p - t) == 1
                dist += abs(p - t)
        paths = {k: [u[i] for i in range(len(u))] for k, u in by.items()}
        if truth is None:
            truth = paths
        elif paths != truth:
            raise ValueError(f"{m}'s predictions carry other true slides than {methods[0]}'s")
        wrong[m], wrong_core[m], near[m], distance[m] = w, wc, nr, dist
    assert truth is not None
    units = sum(len(t) for t in truth.values())
    if len(core) != units:
        raise ValueError(f"{len(core)} importance labels for {units} test units")
    return {"truth": truth, "wrong": wrong, "wrong_core": wrong_core, "near": near,
            "distance": distance, "units": units, "core": sum(core.values()),
            "labels": dict(counts)}


def back_jumps(truth: Mapping[Key, Sequence[int]]) -> dict[str, Any]:
    """How often the true path returns to an earlier slide: the lectures in which it
    does at least once, and the steps between consecutive units that go back, out of
    all such steps and out of the steps that change slide; also per corpus."""
    out: dict[str, Any] = {"lectures": 0, "back_lectures": 0, "steps": 0, "back": 0,
                           "changes": 0, "corpus": defaultdict(lambda: [0, 0])}
    for (corpus, _), t in truth.items():
        back = sum(b < a for a, b in zip(t, t[1:]))
        out["lectures"] += 1
        out["back_lectures"] += back > 0
        out["steps"] += max(len(t) - 1, 0)
        out["back"] += back
        out["changes"] += sum(b != a for a, b in zip(t, t[1:]))
        out["corpus"][corpus][0] += 1
        out["corpus"][corpus][1] += back > 0
    out["corpus"] = dict(out["corpus"])
    return out


def q4_expected(d: Mapping[str, Any], ud: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Q4 against importance-blind errors, per method in Q4's order: the observed share
    of all units that are core and wrong (Q4's all-lectures column), the pooled error
    rate, the share expected if the errors fell on units whatever their label (the core
    share of all units times the error rate), and observed / expected. Below 1: the
    method's errors spare important content."""
    share = ud["core"] / ud["units"]
    out = []
    for r in d["q4"]:
        m = r["method"]
        err = 100 * ud["wrong"][m] / ud["units"]
        obs = float(r["all_pct"])
        out.append({"method": m, "observed": obs, "error": err, "expected": share * err,
                    "ratio": obs / (share * err) if err else float("nan")})
    return out


def check_units(d: Mapping[str, Any], ud: Mapping[str, Any]) -> list[str]:
    """The unit data agree with the results they sit next to: each method's wrong units
    are step 13's (n_units - n_correct, summed), its wrong core units give Q4's
    all-lectures percentage, and no in-order corpus has a true path that goes back."""
    out = []
    for m, rs in d["rows"].items():
        want = sum(int(r["n_units"]) - int(r["n_correct"]) for r in rs)
        if ud["wrong"][m] != want:
            out.append(f"units: {m} has {ud['wrong'][m]} wrong units in its predictions, "
                       f"step 13 {want}")
    for r in d["q4"]:
        pct = 100 * ud["wrong_core"][r["method"]] / ud["units"]
        if not math.isclose(pct, float(r["all_pct"]), abs_tol=1e-9):
            out.append(f"units: {r['method']}'s wrong core units give {pct!r} %, Q4 "
                       f"{r['all_pct']}")
    bj = back_jumps(ud["truth"])
    for c in IN_ORDER_CORPORA:
        if bj["corpus"].get(c, [0, 0])[1]:
            out.append(f"units: a true path of {c} goes back, but its ground truth is "
                       f"in slide order")
    return out


def llm_fallbacks(d: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per LLM method of FALLBACK_METHODS, from step 13's per-lecture rows: the lectures,
    the answers used, the unusable answers (`parse_failed`, each one a fallback) and the
    answers cut off at max_tokens (`truncated`), summed over the test lectures. For the
    single-call methods an unusable answer is the lecture's one answer, so the count is
    also the lectures that fell back."""
    out = {}
    for m in FALLBACK_METHODS:
        rs = d["rows"][m]
        out[m] = {"lectures": len(rs),
                  "answers": int(sum(float(r["used_calls"]) for r in rs)),
                  "fallbacks": int(sum(float(r["parse_failed"]) for r in rs)),
                  "truncated": int(sum(float(r["truncated"]) for r in rs)),
                  "fallback_lectures": sum(float(r["parse_failed"]) > 0 for r in rs)}
    return out


def reweight(acc: Mapping[Key, float], level: Mapping[Key, str],
             weights: Mapping[str, float]) -> float:
    """The mean of per-lecture accuracies with each level's lectures weighted to the
    level's share in `weights` (shares that add up to 1) instead of their own count:
    what the mean would be with another mix of levels, if the accuracy within a level
    stayed the same."""
    by: dict[str, list[float]] = defaultdict(list)
    for k, a in acc.items():
        by[level[k]].append(a)
    if set(by) != {lv for lv, w in weights.items() if w}:
        raise ValueError(f"levels {sorted(by)} against weights {sorted(weights)}")
    return sum(w * sum(by[lv]) / len(by[lv]) for lv, w in weights.items() if w)


def tuning_data(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per method in Q1's order: the highest mean accuracy of any of its combinations
    on the tuning lectures (scored as step 11 scores them, select_settings.combo_stats),
    the accuracy there of the combination step 11 chose (config/tuned_settings.toml),
    and the test accuracy of that combination (Q1). Also the number of combinations
    tried, and the chosen combination's tuning accuracy reweighted to the test set's mix
    of short, medium and long lectures (reweight): how much of the change from tuning
    to test the different length mix of the two sets accounts for."""
    import select_settings as S
    lectures = set(T.set_lectures(T.TUNING_SET))
    chosen = F.tuning_accuracies()
    combos = F.chosen_combos()
    length = {(r["corpus"], r["lecture_id"]): r["length"] for r in d["selection"]}
    test_length = d["levels"]["length"]           # the test lectures' lengths
    weights = {lv: sum(v == lv for v in test_length.values()) / len(test_length)
               for lv in d["order"]["length"]}
    out = []
    for r in d["q1"]:
        m = r["method"]
        runs = T.read_runs(m)
        stats = S.combo_stats(runs, lectures)
        best = max(st.mean_acc for st in stats.values())
        acc = {(x["corpus"], x["lecture_id"]): F.lecture_accuracy(x) for x in runs
               if x["combo"] == combos[m]}
        out.append({"method": m, "best": 100 * best, "chosen": 100 * chosen[m],
                    "test": r["accuracy"], "combos": len(stats),
                    "chosen_runs": 100 * sum(acc.values()) / len(acc) if acc else float("nan"),
                    "reweighted": 100 * reweight(acc, length, weights) if acc else float("nan")})
    return out


def check_tuning(td: Sequence[Mapping[str, Any]]) -> list[str]:
    """The chosen combination is at most the best and within step 11's tie of it (0.5
    points, plus rounding); where its runs were read (tuning_data), they give the
    tuning accuracy step 11 recorded and hold every combination of its grid (the count
    Table A1 prints)."""
    out = [f"tuning: {r['method']} chose {r['chosen']:.3f}, best {r['best']:.3f}"
           for r in td if not (r["best"] - 0.5 - 1e-6 <= r["chosen"] <= r["best"] + 1e-6)]
    grids = T.load_grid()
    out += [f"tuning: {r['method']}'s grid has {grids[r['method']].n_combos()} combinations, "
            f"its tuning runs {r['combos']}" for r in td
            if "combos" in r and grids[r["method"]].n_combos() != r["combos"]]
    out += [f"tuning: {r['method']}'s chosen runs give {r['chosen_runs']:.6f}, step 11 "
            f"{r['chosen']:.6f}" for r in td if "chosen_runs" in r
            and not math.isclose(r["chosen_runs"], r["chosen"], abs_tol=1e-6)]
    return out


def top_band_end(q1: Sequence[Mapping[str, Any]], step: float = TOP_BAND_STEP) -> int:
    """How many methods, from the most accurate, come before the first drop of more than
    `step` points between neighbours in Q1's order (all of them if there is none)."""
    accs = sorted((r["accuracy"] for r in q1), reverse=True)
    return next((i + 1 for i in range(len(accs) - 1) if accs[i] - accs[i + 1] > step),
                len(accs))


def check_top_band(q1: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check 8: TOP_BAND is the band of top_band_end."""
    n = top_band_end(q1)
    return [] if n == TOP_BAND else [f"top band: the first step of more than {TOP_BAND_STEP} "
                                     f"points comes after rank {n}, TOP_BAND is {TOP_BAND}"]


def dataset_accuracy(d: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per method its mean accuracy (%) over the test lectures of each dataset (corpus,
    M3AV's halves apart), and the lectures per dataset ("n")."""
    out: dict[str, dict[str, Any]] = {}
    for m, rs in d["rows"].items():
        by: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            by[r["corpus"]].append(A.lecture_accuracy(r))
        out[m] = {c: 100 * sum(v) / len(v) for c, v in by.items()}
    first = next(iter(d["rows"].values()))
    out["n"] = {c: sum(r["corpus"] == c for r in first) for c in N.CORPUS_NAMES}
    return out


def check_datasets(da: Mapping[str, Mapping[str, Any]], d: Mapping[str, Any]) -> list[str]:
    """Check 9: the lecture-weighted mean of the dataset means is Q1's accuracy."""
    n = da["n"]
    out = []
    for r in d["q1"]:
        mean = sum(n[c] * da[r["method"]][c] for c in n if n[c]) / sum(n.values())
        if not math.isclose(mean, r["accuracy"], abs_tol=1e-9):
            out.append(f"datasets: {r['method']} averages {mean!r}, Q1 {r['accuracy']!r}")
    return out


def tab_datasets(da: Mapping[str, Mapping[str, Any]], d: Mapping[str, Any]) -> str:
    corpora = [c for c in N.CORPUS_NAMES if da["n"][c]]
    n_all = sum(da["n"].values())
    body = [row(["Method"] + [tex(N.CORPUS_NAMES[c]) for c in corpora] + ["All"]),
            row([""] + [f"({da['n'][c]})" for c in corpora] + [f"({n_all})"]), r"\midrule"]
    for r in d["q1"]:
        m = r["method"]
        body.append(row([tex(N.method_name(m))] + [f"{da[m][c]:.1f}" for c in corpora]
                        + [f"{r['accuracy']:.1f}"]))
    cap = ("Accuracy per dataset: the mean accuracy (\\%) of every method over the test "
           "lectures of each of the five datasets (test lectures in brackets), and over all "
           "of them (Table~\\ref{tab:q1}). Methods in the order of Table~\\ref{tab:q1}. The "
           "datasets differ in more than their lectures: the slide text, the units and the "
           "ground truth are made differently in each (Section~\\ref{sec:data}).")
    return table_env(cap, "tab:datasets", "@{}l" + "r" * (len(corpora) + 1) + "@{}", body,
                     tabcolsep="4pt")


def tab_q4_expected(rows: Sequence[Mapping[str, Any]], core_pct: float) -> str:
    heads = ["Method", head_cell("Wrong", "(\\%)"), head_cell("Expected", "(\\%)"), "Ratio"]
    cells = [[tex(N.method_name(r["method"])), f"{r['observed']:.1f}", f"{r['expected']:.1f}",
              f"{r['ratio']:.2f}"] for r in rows]
    body = [row(heads), r"\midrule", *(row(c) for c in cells)]
    cap = ("Q4 against errors that ignore importance, over all test lectures and pooled over "
           "the units. Wrong: the important (core) units the method placed on the wrong "
           "slide, as a percentage of all units (the All column of Table~\\ref{tab:q4}). "
           "Expected: the same percentage if its errors fell on units whatever their label, "
           f"that is {core_pct:.1f}\\,\\% (the core share of all units) times its error rate "
           "(all units on the wrong slide, as a percentage of all units). Ratio: wrong divided "
           "by expected; below 1, the method's errors spare important content. Methods in the "
           "order of Table~\\ref{tab:q4}.")
    # one method per row, every name on one line, columns at least 2 em apart (APPENDIX_SEP)
    return table_env(cap, "tab:q4-expected", "@{}lrrr@{}", body, tabcolsep=APPENDIX_SEP)


def tab_tuning(td: Sequence[Mapping[str, Any]], n_tuning: int, n_test: int) -> str:
    heads = ["Method", head_cell("Best", "tuning"), head_cell("Chosen", "tuning"), "Test",
             head_cell("Test $-$", "chosen")]
    cells = [[tex(N.method_name(r["method"])), f"{r['best']:.1f}", f"{r['chosen']:.1f}",
              f"{r['test']:.1f}", f"{r['test'] - r['chosen']:+.1f}".replace("-", "$-$")]
             for r in td]
    body = [row(heads), r"\midrule", *(row(c) for c in cells)]
    cap = (f"Tuning against test: mean accuracy (\\%) on the {n_tuning} tuning lectures and "
           f"on the {n_test} test lectures. Best tuning: the highest of any tuning "
           "combination of the method. Chosen tuning: the combination chosen in tuning "
           "(within 0.5 points of the best, Section~\\ref{sec:tuning}), the one run on the "
           "test lectures. Test: that combination on the test lectures (Table~\\ref{tab:q1}). "
           "Test $-$ chosen: the change from tuning to test, in points. Methods in the order "
           "of Table~\\ref{tab:q1}.")
    # one method per row, as Table F2
    return table_env(cap, "tab:tuning", "@{}lrrrr@{}", body, tabcolsep=APPENDIX_SEP)


# --------------------------------------------------------------------------- #
# The bootstrap over lectures (Appendix E) and the distance of the errors (Appendix F)
# --------------------------------------------------------------------------- #

#: Resamples of the test lectures, each as many lectures as the test set, drawn with
#: replacement.
BOOT_DRAWS = 1000
#: The project seed (config/attributes.toml; the M3AV split uses the same), so that a
#: rerun draws the same resamples and writes the same table.
BOOT_SEED = 20261011
#: The percentiles of the bootstrap means that bound a 95 % interval.
BOOT_PCT = (2.5, 97.5)


def lecture_accuracies(d: Mapping[str, Any]) -> dict[str, list[float]]:
    """Per method its accuracy (%) on every test lecture, in one lecture order for all
    methods: position i is the same lecture for every method, which is what pairs the
    methods in the bootstrap."""
    keys = sorted((r["corpus"], r["lecture_id"]) for r in next(iter(d["rows"].values())))
    out = {}
    for m, rs in d["rows"].items():
        by = {(r["corpus"], r["lecture_id"]): A.lecture_accuracy(r) for r in rs}
        out[m] = [100 * by[k] for k in keys]
    return out


def bootstrap_means(acc: Mapping[str, Sequence[float]], draws: int = BOOT_DRAWS,
                    seed: int = BOOT_SEED) -> dict[str, list[float]]:
    """Per method its mean accuracy in each of `draws` resamples of the lectures. Every
    method is averaged over the same resamples, so the difference of two methods per
    resample is a paired difference: a lecture that is hard for both lowers both means
    alike and does not widen the interval of their difference, as it would if each
    method were resampled on its own."""
    n = len(next(iter(acc.values())))
    if any(len(v) != n for v in acc.values()):
        raise ValueError("the methods are not on the same lectures")
    rng = random.Random(seed)
    picks = [[rng.randrange(n) for _ in range(n)] for _ in range(draws)]
    return {m: [sum(v[i] for i in p) / n for p in picks] for m, v in acc.items()}


def percentile(values: Sequence[float], pct: float) -> float:
    """The pct-th percentile, linear between the sorted values (numpy's default), so
    that the interval does not depend on which draw happens to sit at a rank."""
    v = sorted(values)
    h = (len(v) - 1) * pct / 100
    lo = math.floor(h)
    return v[lo] if lo + 1 >= len(v) else v[lo] + (h - lo) * (v[lo + 1] - v[lo])


def interval(values: Sequence[float]) -> tuple[float, float]:
    return percentile(values, BOOT_PCT[0]), percentile(values, BOOT_PCT[1])


def bootstrap_data(d: Mapping[str, Any], draws: int = BOOT_DRAWS,
                   seed: int = BOOT_SEED) -> dict[str, Any]:
    """Per method in Q1's order: Q1's accuracy, the interval of its bootstrap means and
    the interval of rank 1's mean minus its own on the same resamples (None for rank 1);
    also rank 1 ("ref") and the means themselves, for the differences the text quotes."""
    acc = lecture_accuracies(d)
    means = bootstrap_means(acc, draws, seed)
    q1 = sorted(d["q1"], key=lambda r: r["rank"])
    ref = q1[0]["method"]
    rows = [{"method": r["method"], "accuracy": r["accuracy"], "ci": interval(means[r["method"]]),
             "diff": None if r["method"] == ref else
             interval([a - b for a, b in zip(means[ref], means[r["method"]])])} for r in q1]
    return {"ref": ref, "rows": rows, "means": means, "acc": acc, "draws": draws}


def check_bootstrap(bd: Mapping[str, Any], d: Mapping[str, Any]) -> list[str]:
    """Check 11: the per-lecture accuracies the bootstrap resamples average to Q1's
    accuracy, so the resamples are of the lectures Q1 ranks."""
    mean = {m: statistics.mean(v) for m, v in bd["acc"].items()}
    return [f"bootstrap: {r['method']}'s lectures average {mean[r['method']]!r}, Q1 "
            f"{r['accuracy']!r}" for r in d["q1"]
            if not math.isclose(mean[r["method"]], r["accuracy"], abs_tol=1e-9)]


def signed(v: float) -> str:
    """One decimal, a minus as in mathematics (the table's and the text's)."""
    return f"{v:.1f}".replace("-", "$-$")


def bootstrap_numbers(bd: Mapping[str, Any], top: Sequence[str]) -> dict[str, str]:
    """The text's numbers of the bootstrap (Q1): the interval of rank 1 minus Argmax
    (the gain of the ordering constraint) and minus Gotoh, and how many other top-band
    methods have a difference interval to rank 1 that holds zero, and which."""
    means, ref = bd["means"], bd["ref"]

    def diff(m: str) -> tuple[float, float]:
        return interval([a - b for a, b in zip(means[ref], means[m])])

    gain, pair = diff("argmax"), diff("gotoh")
    rows = {r["method"]: r for r in bd["rows"]}
    overlap = [m for m in top if m != ref and rows[m]["diff"][0] <= 0 <= rows[m]["diff"][1]]
    return {"DecoderGainLo": signed(gain[0]), "DecoderGainHi": signed(gain[1]),
            "TopPairLo": signed(pair[0]), "TopPairHi": signed(pair[1]),
            "TopBandOverlapZero": _int(len(overlap)),
            "TopBandOverlapNames": _join([tex(N.method_name(m)) for m in overlap]),
            "TopBandOthers": _int(len(top) - 1),
            "BootDraws": _int(bd["draws"])}


def tab_bootstrap(bd: Mapping[str, Any]) -> str:
    ref = N.method_name(bd["ref"])
    lo, hi = (f"{p:g}\\,\\%" for p in BOOT_PCT)
    body = [r" & & \multicolumn{2}{c}{Bootstrap mean} & \multicolumn{2}{c}{"
            + tex(ref) + r" $-$ method} \\",
            r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
            row(["Method", head_cell("Accuracy", "(\\%)"), lo, hi, lo, hi]), r"\midrule"]
    for r in bd["rows"]:
        diff = ["--", "--"] if r["diff"] is None else [signed(x) for x in r["diff"]]
        body.append(row([tex(N.method_name(r["method"])), f"{r['accuracy']:.1f}",
                         *(f"{x:.1f}" for x in r["ci"]), *diff]))
    n = len(next(iter(bd["acc"].values())))
    cap = (f"Bootstrap intervals over the {n} test lectures: {_int(bd['draws'])} resamples of "
           "the lectures, drawn with replacement, every method averaged over the same "
           "resamples. Accuracy: the mean accuracy of Table~\\ref{tab:q1}. Bootstrap mean: "
           f"the {lo} and {hi} percentiles of the method's mean accuracy (\\%) over the "
           f"resamples. {tex(ref)} $-$ method: the same percentiles of {tex(ref)}'s mean "
           "minus the method's on each resample, in points; an interval that holds 0 does "
           "not separate the two methods on lectures like these. Methods in the order of "
           "Table~\\ref{tab:q1}.")
    # one method per row, every name on one line, columns 2 em apart (as Table F2)
    return table_env(cap, "tab:bootstrap", "@{}lrrrrr@{}", body, tabcolsep=APPENDIX_SEP)


def severity(ud: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Per method, pooled over all test units: the wrong units as a share of all units,
    the share of the wrong units whose slide is next to the true one (one slide off),
    and the mean distance in slides of the wrong units from their true slide. A wrong
    unit one slide off still reaches the right place after one step through the deck; a
    wrong start or cut can move a stretch of units several slides away."""
    out = {}
    for m, w in ud["wrong"].items():
        out[m] = {"wrong": 100 * w / ud["units"],
                  "near": 100 * ud["near"][m] / w if w else float("nan"),
                  "distance": ud["distance"][m] / w if w else float("nan")}
    return out


def severity_numbers(sv: Mapping[str, Mapping[str, float]], top: Sequence[str]) -> dict[str, str]:
    """The text's numbers of the error distance (Q4, Conclusion): the range of the share
    one slide off in the top band and in the methods in which an LLM alone aligns."""
    return {"TopNearMin": f"{min(sv[m]['near'] for m in top):.0f}",
            "TopNearMax": f"{max(sv[m]['near'] for m in top):.0f}",
            "AloneNearMin": f"{min(sv[m]['near'] for m in LLM_ALONE):.0f}",
            "AloneNearMax": f"{max(sv[m]['near'] for m in LLM_ALONE):.0f}",
            "TopDistanceMax": f"{max(sv[m]['distance'] for m in top):.1f}",
            "AloneDistanceMin": f"{min(sv[m]['distance'] for m in LLM_ALONE):.1f}",
            "AloneDistanceMax": f"{max(sv[m]['distance'] for m in LLM_ALONE):.1f}"}


def tab_severity(sv: Mapping[str, Mapping[str, float]], d: Mapping[str, Any]) -> str:
    heads = ["Method", head_cell("Wrong", "units (\\%)"), head_cell("One slide", "off (\\%)"),
             head_cell("Mean", "distance")]
    body = [row(heads), r"\midrule"]
    for r in sorted(d["q1"], key=lambda r: r["rank"]):
        s = sv[r["method"]]
        body.append(row([tex(N.method_name(r["method"])), f"{s['wrong']:.1f}",
                         f"{s['near']:.1f}", f"{s['distance']:.2f}"]))
    cap = ("How far the errors land, over all test lectures and pooled over the units. "
           "Wrong units: the units the method placed on the wrong slide, as a percentage of all "
           "units. One slide off: the wrong units whose slide is next to their true slide, "
           "as a percentage of the wrong units. Mean distance: the mean number of slides "
           "between the slide of a wrong unit and its true slide. Methods in the order of "
           "Table~\\ref{tab:q1}.")
    return table_env(cap, "tab:severity", "@{}lrrr@{}", body, tabcolsep=APPENDIX_SEP)


def prompt_text(template: str) -> str:
    """A prompt template as the model receives it, its fields shown as {name}: the
    template formatted with each field set to its own name in braces, so the doubled
    braces of the code become the single braces the model sees."""
    import string
    fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    return template.format(**{f: "{" + f + "}" for f in fields})


def prompts() -> list[tuple[str, str, str]]:
    """The prompts of the final run, from methods/llm.py: (heading, role, text). The
    hierarchical ones are the compact variants (ask_for_reason is fixed off) with the
    pacing sentence the aligner appends; boundary refinement was switched off in tuning
    (config/tuned_settings.toml), so its prompts were never sent and are not shown."""
    from methods import llm as L
    if T.load_tuned()["hierarchical"].settings.get("use_boundary_refinement", True):
        raise ValueError("hierarchical's tuned setting refines the boundaries: add the "
                         "boundary prompts to prompts()")
    return [
        ("LLM single call, LLM monotone, LLM monotone (constrained) and LLM re-rank",
         "system", L.SYSTEM_PROMPT),
        ("LLM single call (also the LLM voter of Ensemble vote)", "user",
         L.SINGLE_CALL_TEMPLATE),
        ("LLM monotone", "user", L.MONOTONE_TEMPLATE),
        ("LLM monotone (constrained)", "user", L.MONOTONE_CONSTRAINED_TEMPLATE),
        ("LLM re-rank (5 methods)", "user", L.RERANK_TEMPLATE),
        ("LLM hierarchical (also inside LLM-prior DP), split call", "system",
         L.HIER_SPLIT_SYSTEM_PROMPT),
        ("LLM hierarchical, split call", "user",
         L.HIER_SPLIT_USER_TEMPLATE_FAST + L.HIER_PACE_SPLIT_TEMPLATE),
        ("LLM hierarchical, leaf call", "system", L.HIER_LEAF_SYSTEM_PROMPT),
        ("LLM hierarchical, leaf call", "user",
         L.HIER_LEAF_USER_TEMPLATE_FAST + L.HIER_PACE_LEAF_TEMPLATE)]


def prompts_tex() -> str:
    """The prompts as listings: a bold heading line each (not a run-in \\paragraph,
    which a listing right after it would swallow), the text verbatim in a small monospace
    font, long lines broken (listings' breaklines)."""
    lines = []
    for heading, role, text in prompts():
        body = prompt_text(text)
        if r"\end{lstlisting}" in body:
            raise ValueError(f"{heading}: the prompt would end the listing")
        lines += [rf"\par\medskip\noindent\textbf{{{tex(heading)}: {role} message.}}\par\nobreak",
                  r"\begin{lstlisting}", body, r"\end{lstlisting}", ""]
    return "\n".join(lines)


def _join(names: Sequence[str]) -> str:
    """Names as running text: "none", "A", "A and B", "A, B and C"."""
    if not names:
        return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def split_counts(d: Mapping[str, Any], path: Path = SPLIT) -> dict[str, dict[str, int]]:
    """Per M3AV version: the talks the split gives it (the usable talks, split before the
    image rule), the lectures of it in the pool (Table 1), and the difference, the talks
    the image rule dropped after the split (prepare.py applies it there). The split is
    read with prepare.py's own reader, so its checks of the file apply here too."""
    import prepare as PR
    version = PR.read_split(path)
    out = {}
    for corpus, v in PR.M3AV_VERSIONS.items():
        split = sum(1 for x in version.values() if x == v)
        kept = sum(1 for c, _ in d["cats"] if c == corpus)
        out[corpus] = {"split": split, "kept": kept, "dropped": split - kept}
    return out


def check_split(d: Mapping[str, Any], path: Path = SPLIT) -> list[str]:
    """Check 10: every pooled M3AV lecture is in its version of the split, so that split
    minus pool counts the talks dropped after the split."""
    import prepare as PR
    version = PR.read_split(path)
    return [f"split: {c} lecture {lec} is {version.get(lec, 'in no version')} in the split"
            for c, lec in d["cats"]
            if c in PR.M3AV_VERSIONS and version.get(lec) != PR.M3AV_VERSIONS[c]]


def _count(n: int) -> str:
    """A count for running text: "none" for 0, else the number."""
    return "none" if n == 0 else _int(n)


# --------------------------------------------------------------------------- #
# The preparation counts (Appendix B, Table B1)
# --------------------------------------------------------------------------- #

#: Where scripts/prepare.py (step 2) writes summary.csv and the <corpus>_dropped.csv
#: files; those two kinds of file are in git (.gitignore), the prepared lectures are not.
PREPARED = REPO_ROOT / "data" / "prepared"

#: Every reason scripts/prepare.py drops a lecture for, in the order its rules apply, with
#: the two-line header of its column in Table B1. A reason the table does not know stops
#: the run, so that no dropped lecture goes unshown.
DROP_HEADS = {
    "no_ground_truth": ("No", "labels"),
    "no_slides": ("No", "slides"),
    "fewer_than_5_slides": ("Under", "5 slides"),
    "no_labelled_units": ("No labelled", "units"),
    "slide_without_image": ("Image", "missing"),
}

#: The row of Table B1 for the M3AV talks in neither half: the talks the rules drop before
#: the split. prepare.py loads them in both halves, so that each half's dropped file is
#: complete, and lists them in both; the table counts them once, in this row.
M3AV_NEITHER = "m3av_neither"


class PreparedMissing(FileNotFoundError):
    """summary.csv is not there yet: Table B1 is written as a placeholder and its macros
    are left out, so that the other tables and figures can be written before step 2's files
    are in the repository."""


def read_prepared(root: Path = PREPARED) -> dict[str, Any]:
    """summary.csv's rows by corpus and the rows of every <corpus>_dropped.csv there is.
    Without summary.csv, PreparedMissing; a missing dropped file is a problem for
    check_prepared, since summary.csv says how many lectures it must list."""
    summary = root / "summary.csv"
    if not summary.exists():
        raise PreparedMissing(f"{summary} is missing (scripts/prepare.py, step 2, writes it "
                              "with the <corpus>_dropped.csv files)")
    rows = {r["corpus"]: r for r in A.read_csv(summary)}
    dropped = {c: A.read_csv(root / f"{c}_dropped.csv") for c in rows
               if (root / f"{c}_dropped.csv").exists()}
    return {"summary": rows, "dropped": dropped}


def _reasons(field: str) -> dict[str, int]:
    """summary.csv's drop_reasons ("a=3;b=1", empty for none) as {reason: lectures}."""
    out: dict[str, int] = {}
    for part in filter(None, field.split(";")):
        k, v = part.split("=")
        out[k] = int(v)
    return out


def m3av_neither(pr: Mapping[str, Any]) -> dict[str, str]:
    """{talk: reason} of the M3AV talks listed in the dropped files of both halves: the
    talks in neither half, which prepare.py loads (and drops) in both."""
    import prepare as PR
    a, b = ({r["lecture_id"]: r["reason"] for r in pr["dropped"].get(c, [])}
            for c in PR.M3AV_VERSIONS)
    return {t: a[t] for t in sorted(a.keys() & b.keys())}


def prepare_counts(pr: Mapping[str, Any]) -> dict[str, Any]:
    """Per row of Table B1 its counts, and their total: the lectures in the corpus, those
    dropped per reason, the kept ones (the pool), the kept ones whose transcript the cap
    cut, and the units of the kept lectures without a true slide, cut by the cap and kept
    (summary.csv's units_before_cap counts the units with a true slide and text over the
    whole lecture, so before minus after is what the cap cut). An M3AV talk in neither
    half is counted once, in the row M3AV_NEITHER, and not in the halves' rows."""
    import prepare as PR
    neither = m3av_neither(pr)
    rows: dict[str, dict[str, Any]] = {}
    for corpus in N.CORPUS_NAMES:
        s = pr["summary"][corpus]
        drop = defaultdict(int)
        for r in pr["dropped"].get(corpus, []):
            if not (corpus in PR.M3AV_VERSIONS and r["lecture_id"] in neither):
                drop[r["reason"]] += 1
        rows[corpus] = {
            "lectures": int(s["lectures_available"]) - (len(neither) if corpus in PR.M3AV_VERSIONS else 0),
            "dropped": dict(drop), "kept": int(s["lectures_kept"]),
            "transcripts_cut": int(s["transcripts_cut"]),
            "units_without_slide": int(s["units_without_slide"]),
            "units_cut": int(s["units_before_cap"]) - int(s["units_after_cap"]),
            "units_kept": int(s["units_after_cap"])}
        if corpus == "m3av_b" and neither:
            drop = defaultdict(int)
            for reason in neither.values():
                drop[reason] += 1
            rows[M3AV_NEITHER] = {"lectures": len(neither), "dropped": dict(drop), "kept": 0,
                                  "transcripts_cut": 0, "units_without_slide": 0,
                                  "units_cut": 0, "units_kept": 0}
    total: dict[str, Any] = {k: sum(r[k] for r in rows.values())
                             for k in next(iter(rows.values())) if k != "dropped"}
    total["dropped"] = {reason: sum(r["dropped"].get(reason, 0) for r in rows.values())
                        for reason in DROP_HEADS}
    reasons = [x for x in DROP_HEADS if total["dropped"][x]]
    return {"rows": rows, "total": total, "reasons": reasons}


def check_prepared(pr: Mapping[str, Any], d: Optional[Mapping[str, Any]] = None,
                   split: Path = SPLIT) -> list[str]:
    """Check 12: summary.csv has the five datasets and each dataset's dropped file; the
    file lists summary.csv's dropped lectures, per reason, and lectures in the corpus are
    kept plus dropped; every reason is one Table B1 knows; the talks in both M3AV dropped
    files are in neither half of the split, with one reason, and every dropped talk in
    neither half is in both files. With d, the kept lectures are the pool of Table 1 per
    dataset, so \\PrepPool is \\PoolLectures."""
    import prepare as PR
    out = []
    have = set(pr["summary"])
    if have != set(N.CORPUS_NAMES):
        return [f"prepare: summary.csv has {sorted(have)}, not {sorted(N.CORPUS_NAMES)}"]
    for corpus, s in pr["summary"].items():
        if corpus not in pr["dropped"]:
            out.append(f"prepare: {corpus}_dropped.csv is missing")
            continue
        rows = pr["dropped"][corpus]
        listed = defaultdict(int)
        for r in rows:
            listed[r["reason"]] += 1
        if dict(listed) != _reasons(s["drop_reasons"]) or len(rows) != int(s["lectures_dropped"]):
            out.append(f"prepare: {corpus}_dropped.csv lists {dict(listed)}, summary.csv "
                       f"{s['lectures_dropped']} lectures, {s['drop_reasons'] or 'none'}")
        if int(s["lectures_available"]) != int(s["lectures_kept"]) + int(s["lectures_dropped"]):
            out.append(f"prepare: {corpus} has {s['lectures_available']} lectures, not "
                       f"{s['lectures_kept']} kept + {s['lectures_dropped']} dropped")
        out += [f"prepare: {corpus} drops lectures for {x!r}, which Table B1 has no column for"
                for x in sorted(listed.keys() - DROP_HEADS.keys())]
    if out:
        return out
    version = PR.read_split(split)
    a, b = ({r["lecture_id"]: r["reason"] for r in pr["dropped"][c]} for c in PR.M3AV_VERSIONS)
    for t in sorted(a.keys() & b.keys()):
        if t in version:
            out.append(f"prepare: M3AV talk {t} is dropped in both halves but is in half "
                       f"{version[t]} of the split")
        if a[t] != b[t]:
            out.append(f"prepare: M3AV talk {t} is dropped for {a[t]} in one half, {b[t]} "
                       "in the other")
    out += [f"prepare: M3AV talk {t} is in neither half but dropped in one only"
            for t in sorted((a.keys() ^ b.keys()) - version.keys())]
    if d is not None:
        pool: dict[str, int] = defaultdict(int)
        for c, _ in d["cats"]:
            pool[c] += 1
        kept = {c: int(s["lectures_kept"]) for c, s in pr["summary"].items()}
        out += [f"prepare: {c} keeps {kept[c]} lectures, the pool has {pool[c]}"
                for c in N.CORPUS_NAMES if kept[c] != pool[c]]
        if sum(kept.values()) != len(d["cats"]):
            out.append(f"prepare: PrepPool {sum(kept.values())} is not PoolLectures "
                       f"{len(d['cats'])}")
    return out


def prepare_numbers(pc: Mapping[str, Any]) -> dict[str, str]:
    """Table B1's numbers the text quotes: the pool lectures whose transcript the cap cut
    (Section 9) and the pool (equal to \\PoolLectures, check 12)."""
    return {"PrepCut": _int(pc["total"]["transcripts_cut"]),
            "PrepPool": _int(pc["total"]["kept"])}


#: What a drop reason means, for Table B1's caption where its column is not plain.
DROP_NOTES = {
    "no_ground_truth": "no labels: a MaViLS deck without its file of slide labels",
    "no_slides": "no slides: an M$^3$AV talk without slide segments",
    "no_labelled_units": "no labelled units: no unit with a true slide and text",
    "slide_without_image": "image missing: a kept slide without an image",
}


def prepare_caption(reasons: Sequence[str] = ()) -> str:
    """Table B1's caption; `reasons`: the drop reasons it has a column for (none for the
    placeholder), the less plain of which it explains."""
    notes = [DROP_NOTES[x] for x in reasons if x in DROP_NOTES]
    return ("Preparation of the five datasets. Lectures: in the corpus as distributed. "
            "Dropped: the lectures a rule of Section~\\ref{sec:data} removes, by reason"
            + (f" ({'; '.join(notes)})" if notes else "") + ". Kept: the pool of "
            "Table~\\ref{tab:corpora}. Cut: kept lectures whose transcript ends at the "
            "first unit of a slide after the 40th. Units of the kept lectures: without a "
            "true slide (removed), after the cut (removed) and kept. The M$^3$AV talks that "
            "the rules drop are in neither half of the split; the preparation lists them "
            "among the dropped lectures of both halves, and the table counts them once, in "
            "their own row.")


def tab_prepare(pc: Mapping[str, Any]) -> str:
    """Table B1: per dataset, the lectures in the corpus, dropped (one column per reason
    that drops any), kept and cut, and the units removed and kept."""
    reasons = pc["reasons"]
    k = len(reasons)
    dropped = [rf"\multicolumn{{{k}}}{{c}}{{Dropped}}"] if k else []
    body = [" & ".join(["", "", *dropped, "", "", r"\multicolumn{3}{c}{Units}"]) + r" \\",
            (rf"\cmidrule(lr){{3-{2 + k}}}" if k else "")
            + rf"\cmidrule(lr){{{5 + k}-{7 + k}}}",
            row(["Dataset", "Lectures", *(head_cell(*DROP_HEADS[x]) for x in reasons),
                 "Kept", "Cut", head_cell("No true", "slide"), head_cell("After", "the cut"),
                 "Kept"]), r"\midrule"]

    def cells(r: Mapping[str, Any]) -> list[str]:
        return [f"{r['lectures']:,}", *(f"{r['dropped'].get(x, 0):,}" for x in reasons),
                f"{r['kept']:,}", f"{r['transcripts_cut']:,}", f"{r['units_without_slide']:,}",
                f"{r['units_cut']:,}", f"{r['units_kept']:,}"]

    for key, r in pc["rows"].items():
        name = (lines_cell(("M³AV,", "neither half")) if key == M3AV_NEITHER
                else tex(N.CORPUS_NAMES[key]))
        body.append(row([name, *cells(r)]))
    body += [r"\midrule", row(["All", *cells(pc["total"])])]
    return table_env(prepare_caption(reasons), "tab:prepare", "@{}l" + "r" * (6 + k) + "@{}", body,
                     tabcolsep="3pt")


def tab_prepare_missing() -> str:
    """Table B1 before step 2's files are in the repository: its caption and label, and
    one line that says what is missing, so that the table exists and the gap shows."""
    body = [r"The counts come from data/prepared/summary.csv and the dropped-lecture "
            r"files of step 2, which are not in the repository yet. \\"]
    return table_env(prepare_caption(), "tab:prepare", "@{}" + text_col(r"\linewidth") + "@{}",
                     body)


def text_numbers(d: Mapping[str, Any], td: Sequence[Mapping[str, Any]],
                 image_build: float) -> dict[str, str]:
    """More numbers the text quotes, each a plain difference, ratio or share of results
    shown in the tables. `image_build`: the image table's seconds per test lecture (the
    measure table's Table column)."""
    from methods.common import DPConfig
    q1 = {r["method"]: r for r in d["q1"]}
    order = [r["method"] for r in sorted(d["q1"], key=lambda r: -r["accuracy"])]
    best = q1[order[0]]["accuracy"]
    top = order[:TOP_BAND]
    total = {m: q1_total(q1[m]) for m in order}
    acc = {m: q1[m]["accuracy"] for m in order}
    q4 = {r["method"]: float(r["all_pct"]) for r in d["q4"]}
    steps = [acc[a] - acc[b] for a, b in zip(order, order[1:])]
    # the methods that read the image table alone and make at most one LLM call
    img = [m for m in order if q1[m]["build_cost_s_per_lecture"] is not None
           and math.isclose(q1[m]["build_cost_s_per_lecture"], image_build, rel_tol=1e-9)
           and (q1[m]["used_calls_per_lecture"] or 0) <= 1]
    img_share = {m: 100 * image_build / total[m] for m in order}
    pic = d["order"]["images"]
    drops = [d["q2"][("images", pic[0])]["acc"][m] - d["q2"][("images", pic[-1])]["acc"][m]
             for m in top]
    units = d["q4"][0]
    tune = {r["method"]: r for r in td}
    mix = [r["chosen"] - r["reweighted"] for r in td if "reweighted" in r]
    cfg = DPConfig()
    vals = {
        # the methods in which an LLM alone gives the alignment (abstract, introduction)
        "AloneCostMin": f"{min(total[m] for m in LLM_ALONE):.1f}",
        "AloneCostMax": f"{max(total[m] for m in LLM_ALONE):.1f}",
        "AloneGapMin": f"{best - max(acc[m] for m in LLM_ALONE):.1f}",
        "AloneGapMax": f"{best - min(acc[m] for m in LLM_ALONE):.1f}",
        "AloneTimesMin": f"{min(q4[m] for m in LLM_ALONE) / max(q4[m] for m in top):.1f}",
        "AloneTimesMax": f"{max(q4[m] for m in LLM_ALONE) / min(q4[m] for m in top):.1f}",
        # the top band (Q1, conclusion)
        "TopBandStep": f"{steps[TOP_BAND - 1]:.1f}",
        "TopBandInnerStep": f"{max(steps[:TOP_BAND - 1]):.1f}",
        "TopBandSpread": f"{acc[order[0]] - acc[order[TOP_BAND - 1]]:.1f}",
        "TopBandAccMin": f"{acc[order[TOP_BAND - 1]]:.1f}",
        "TopBandAccMax": f"{acc[order[0]]:.1f}",
        # the top band's own run time: the most a further decoder adds to a built table
        "TopComputeMax": compute_seconds(max(q1[m]["compute_s_per_lecture"] or 0.0
                                             for m in top)),
        # every tuning combination (Table A1's column, the pipeline figure)
        "TuneCombos": _int(sum(g.n_combos() for g in T.load_grid().values())),
        "DecoderGain": f"{best - acc['argmax']:.1f}",
        # Score fusion against the methods behind its three members
        "FusionOverMembersMin": f"{min(acc['score_fusion'] - acc[m] for m in FUSION_MEMBERS):.1f}",
        "FusionOverMembersMax": f"{max(acc['score_fusion'] - acc[m] for m in FUSION_MEMBERS):.1f}",
        # the share of the image table in the cost (abstract, introduction, Q1)
        "ImageReaders": _int(len(img)),
        "ImageCostMin": seconds(min(total[m] for m in img)),
        "ImageCostMax": seconds(max(total[m] for m in img)),
        "ImageExtraMax": f"{max(total[m] - image_build for m in img):.1f}",
        "ImageShareFloor": f"{math.floor(min(img_share[m] for m in img))}",
        "TopImageShareMin": f"{min(img_share[m] for m in top):.0f}",
        "TopImageShareRest": f"{math.floor(sorted(img_share[m] for m in top)[1])}",
        "TopImageShareMinName": tex(N.method_name(min(top, key=lambda m: img_share[m]))),
        # Q2: the top band from the lowest to the highest picture share
        "PictureDropMean": f"{statistics.mean(drops):.1f}",
        "PictureDropMin": f"{min(drops):.1f}", "PictureDropMax": f"{max(drops):.1f}",
        # Q4: the units of long lectures, as a share of all units
        "LongUnitsPct": f"{100 * int(units['length_long_units']) / int(units['all_units']):.0f}",
        # the defaults of the tie rule (DPConfig)
        "DefaultFwd": f"{cfg.forward_penalty:g}", "DefaultBack": f"{cfg.backward_penalty:g}",
        "DefaultStay": f"{cfg.stay_bonus:g}", "DefaultPrior": f"{cfg.pos_weight:g}",
    }
    # the pool and the test lectures (Section 4, Limitations)
    test_rows = [r for r in d["selection"] if r["set"] == "test"]
    kinds: dict[str, int] = defaultdict(int)
    for r in test_rows:
        kinds[talk_kind(r["corpus"], r["lecture_id"])] += 1
    disc = {lv: c["n"] for (ax, lv), c in d["q2"].items() if ax == "discipline"}
    big = max(disc, key=disc.get)                 # computer science, the bulk
    short_talks = ("mcd", "slide_highlight")      # short, text-dense talks of one kind
    tab = d["levels"]["tables"]                   # the test lectures' table level
    with_tab = [k for k, lv in tab.items() if lv != "none"]
    without_tab = [k for k, lv in tab.items() if lv == "none"]
    vals.update({
        "PoolLectures": _int(len(d["cats"])),
        "TestCourse": _int(kinds["course"]), "TestInvited": _int(kinds["invited"]),
        "TestConference": _int(kinds["conference"]),
        "TestBigDiscipline": _int(disc[big]),
        "TestOtherDisciplineMin": _int(min(n for lv, n in disc.items() if lv != big)),
        "TestOtherDisciplineMax": _int(max(n for lv, n in disc.items() if lv != big)),
        "TestWithTables": _int(len(with_tab)), "TestWithoutTables": _int(len(without_tab)),
        "ShortTalksWithTables": _int(sum(c in short_talks for c, _ in with_tab)),
        "ShortTalksWithoutTables": _int(sum(c in short_talks for c, _ in without_tab)),
    })
    # Q3: the rows (all lectures and every level) whose lowest method is a TextTiling one
    rows3 = [{"rank": {r["method"]: r["rank"] for r in d["q1"]}}, *d["q2"].values()]
    vals["LowestTilingRows"] = _int(sum(
        any(m in ("texttiling", "texttiling_dp") for m in extreme(c, True)) for c in rows3))
    vals["QThreeRows"] = _int(len(rows3))
    # the M3AV split and the image rule after it (Appendix B)
    sc = split_counts(d)
    a, b = sc["m3av_a"], sc["m3av_b"]
    vals.update({"MAVUsable": _int(a["split"] + b["split"]), "MAVSplitA": _int(a["split"]),
                 "MAVSplitB": _int(b["split"]), "MAVDroppedA": _int(a["dropped"]),
                 "MAVDroppedB": _int(b["dropped"]),
                 "MAVDropped": _int(a["dropped"] + b["dropped"]), "MAVKeptA": _int(a["kept"]),
                 "MAVKeptB": _int(b["kept"]), "MAVKept": _int(a["kept"] + b["kept"])})
    if mix:
        vals["TuneLengthMix"] = f"{statistics.median(mix):.1f}"
    for m, name in (("argmax", "Argmax"), ("hierarchical", "Hier")):
        if "combos" in tune.get(m, {}):
            vals[f"Tune{name}Combos"] = _int(tune[m]["combos"])
            vals[f"Tune{name}Drop"] = f"{tune[m]['chosen'] - tune[m]['test']:.1f}"
    return vals


#: The measures whose table is built from the slide images; the others read only text.
IMAGE_MEASURES = ("image", "hybrid_image")


def measure_numbers(md: Mapping[str, Any]) -> dict[str, str]:
    """The numbers the text quotes from the measure table (Choosing a method, conclusion):
    the image table against the best text-only table (the most accurate over all methods,
    the table's last column) and against BM25, in cost and in accuracy. The accuracy gaps
    are taken over the method columns in which the image table is the most accurate, so
    that a column it does not lead (TextTiling) cannot turn "points above" into a loss."""
    best, build = md["best"], md["build"]
    text = [x for x in T.MEASURES if x not in IMAGE_MEASURES]
    any_of = {x: max(best[m][x] for m in best) for x in T.MEASURES}
    top_text = max(text, key=lambda x: any_of[x])
    cols = [m for m in MEASURE_METHODS if best[m]["image"] == max(best[m].values())]
    over_text = [best[m]["image"] - best[m][top_text] for m in cols]
    over_bm = [best[m]["image"] - best[m]["bm25"] for m in cols]
    return {
        "ImageCostRatio": f"{build['image'] / build[top_text]:.1f}",
        "ImageTableCost": f"{build['image']:.1f}", "TextTableCost": f"{build[top_text]:.1f}",
        "ImageOverTextMin": f"{min(over_text):.1f}", "ImageOverTextMax": f"{max(over_text):.1f}",
        "ImageOverBMMin": f"{min(over_bm):.1f}", "ImageOverBMMax": f"{max(over_bm):.1f}",
    }


# --------------------------------------------------------------------------- #
# The numbers the text quotes from single cells, groups of cells and the label checks
# --------------------------------------------------------------------------- #

#: The tracked files of the label checks: the second labelling of 10 lectures, the blind
#: check's key (the agents' labels) and the checker's labels (scripts/importance_sample.py),
#: the agents' batches (scripts/importance_merge.py) and the discipline check of 50 labels
#: against the final labels (docs/ATTRIBUTES.md).
SECOND_LABELS = REPO_ROOT / "data" / "importance" / "second"
CHECK_KEY = REPO_ROOT / "data" / "importance" / "check_sample_key.csv"
CHECK_MANUAL = REPO_ROOT / "data" / "importance" / "check_manual.csv"
LABEL_BATCHES = REPO_ROOT / "data" / "importance" / "raw"
DISCIPLINE = REPO_ROOT / "data" / "attributes" / "discipline.csv"
DISCIPLINE_CHECK = REPO_ROOT / "data" / "attributes" / "discipline_check_50_v3.csv"
#: The Q2 axes, and the methods whose accuracy per Q2 level the text quotes (Q2).
Q2_AXES = ("length", "images", "tables")
Q2_QUOTED = ("score_fusion", "llm_single_call", "texttiling")
#: The methods whose cost per length level the text quotes (Q2's cost paragraph).
COST_QUOTED = ("hierarchical", "fusion")
#: The methods that may go back and follow the top band (Q1), with LLM-prior DP.
BACK_FOLLOWERS = ("jump_dtw", "cohesion_dp", "forward_backward", "dp", "fusion")
#: The two LLM methods that read the text and keep no order of their own (Q1).
LLM_TEXT = ("llm_single_call", "hierarchical")
#: The two LLM monotone variants (Q1).
LLM_MONOTONE = ("llm_monotone", "llm_monotone_constrained")


def camel(name: str) -> str:
    """A macro-name part from a method's display name or a level id: its letter runs,
    each with a capital first letter ("LLM-prior DP" -> "LLMPriorDP", "pdf_text_layer"
    -> "PdfTextLayer"). A count in brackets ("LLM re-rank (5 methods)") is left out, and
    digits are dropped, since a LaTeX macro name holds letters only."""
    name = re.sub(r"\s*\(\d[^)]*\)", "", name)
    return "".join(w[0].upper() + w[1:] for w in re.findall(r"[A-Za-z]+", name))


def method_macro(m: str) -> str:
    return camel(N.method_name(m))


def level_macro(k: Key) -> str:
    return camel(k[0]) + camel(k[1])


def put(vals: dict[str, str], name: str, value: str) -> None:
    """Add one macro; a name made twice (two methods or levels with the same letters)
    is an error, not an overwrite."""
    if name in vals:
        raise ValueError(f"macro {name} made twice")
    vals[name] = value


def method_numbers(d: Mapping[str, Any], md: Mapping[str, Any]) -> dict[str, str]:
    """The text's numbers of single methods, from a few generic families:
      Acc<method>          Q1's accuracy (Table 4), every method;
      Acc<method><level>   the accuracy in a Q2 level (Fig. 4), the methods of Q2_QUOTED;
      Cost<method><level>  the cost per lecture in a length level (Fig. 5), whole seconds,
                           the methods of COST_QUOTED;
      LLMCalls<method>, LLMSec<method>  Q1's LLM calls and LLM seconds per lecture, whole,
                           every method that makes LLM calls;
      Highest<level>, Lowest<level>  the highest and the lowest accuracy of the level
                           (Q3, Table 5), every level;
      TuneBest<method>     the highest tuning accuracy over the measures (Table 3), every
                           method of the measure table."""
    vals: dict[str, str] = {}
    q1 = {r["method"]: r for r in d["q1"]}
    for m in d["methods"]:
        put(vals, f"Acc{method_macro(m)}", f"{q1[m]['accuracy']:.1f}")
    for m in Q2_QUOTED:
        for axis in Q2_AXES:
            for lv in d["order"][axis]:
                put(vals, f"Acc{method_macro(m)}{level_macro((axis, lv))}",
                    f"{d['q2'][(axis, lv)]['acc'][m]:.1f}")
    for m in COST_QUOTED:
        for lv in d["order"]["length"]:
            put(vals, f"Cost{method_macro(m)}{level_macro(('length', lv))}",
                f"{d['costs'][m][('length', lv)]:.0f}")
    for m in d["methods"]:
        if q1[m]["used_calls_per_lecture"]:
            put(vals, f"LLMCalls{method_macro(m)}", f"{q1[m]['used_calls_per_lecture']:.0f}")
            put(vals, f"LLMSec{method_macro(m)}", f"{q1[m]['used_seconds_per_lecture']:.0f}")
    for k, c in d["q2"].items():
        put(vals, f"Highest{level_macro(k)}", f"{max(c['acc'].values()):.1f}")
        put(vals, f"Lowest{level_macro(k)}", f"{min(c['acc'].values()):.1f}")
    for m in MEASURE_METHODS:
        put(vals, f"TuneBest{method_macro(m)}", f"{max(md['best'][m].values()):.1f}")
    return vals


def quoted_numbers(d: Mapping[str, Any], md: Mapping[str, Any], ud: Mapping[str, Any],
                   top: Sequence[str]) -> dict[str, str]:
    """The text's numbers of groups of methods or cells, each a mean, a range, a
    difference or a rounding of results the tables and figures show; and the label shares
    of all test units. "About" in the text goes with a whole-number rounding here."""
    q1 = {r["method"]: r for r in d["q1"]}
    acc = {m: q1[m]["accuracy"] for m in q1}
    build = md["build"]
    q2 = d["q2"]
    lengths, tabs = d["order"]["length"], d["order"]["tables"]
    rest = [m for m in acc if m not in top]

    def drop(m: str, axis: str, lo: str, hi: str) -> float:
        return q2[(axis, lo)]["acc"][m] - q2[(axis, hi)]["acc"][m]

    len_drop = [drop(m, "length", lengths[0], lengths[-1]) for m in top]
    rest_drop = [drop(m, "length", lengths[0], lengths[-1]) for m in rest]
    text_gap = (statistics.mean(acc[m] for m in top)
                - statistics.mean(acc[m] for m in LLM_TEXT))
    tab_gain = [-drop(m, "tables", tabs[0], tabs[-1]) for m in top]
    # the methods that read the image table alone and make at most one LLM call (Q1)
    img = [m for m in acc if q1[m]["build_cost_s_per_lecture"] is not None
           and math.isclose(q1[m]["build_cost_s_per_lecture"], build["image"], rel_tol=1e-9)
           and (q1[m]["used_calls_per_lecture"] or 0) <= 1]
    hyb = [m for m in acc if q1[m]["build_cost_s_per_lecture"] is not None
           and math.isclose(q1[m]["build_cost_s_per_lecture"], build["hybrid_image"],
                            rel_tol=1e-9) and not q1[m]["used_calls_per_lecture"]]
    cells = [v for m, c in d["costs"].items() for k, v in c.items() if k[0] in Q2_AXES]
    labels = ud["labels"]
    vals = {
        # the test units and their importance labels (Sections 1, 4, 6)
        "TestUnits": _int(ud["units"]),
        "SupportingPct": f"{100 * labels.get(1, 0) / ud['units']:.1f}",
        "FillerPct": f"{100 * labels.get(0, 0) / ud['units']:.1f}",
        # the top band's size and its rounded floor ("about 70 %")
        "TopBandSize": _int(len(top)),
        "TopBandAccRound": f"{min(acc[m] for m in top):.0f}",
        # the image table's cost (Introduction) and BM25's (Section 5.3)
        "ImageTableCostRound": f"{build['image']:.0f}",
        "BMTableUnder": f"{math.ceil(10 * build['bm25'] + 1e-9) / 10:.1f}",
        # Q1
        "BackFollowersMin": f"{min(acc[m] for m in BACK_FOLLOWERS):.0f}",
        "BackFollowersMax": f"{max(acc[m] for m in BACK_FOLLOWERS):.0f}",
        "HybridImageCost": f"{statistics.mean(q1_total(q1[m]) for m in hyb):.0f}",
        "LLMTextGap": f"{text_gap:.0f}",
        "MonotoneAcc": f"{statistics.mean(acc[m] for m in LLM_MONOTONE):.0f}",
        # Q2: the top band from short to long lectures, the rest, and tables
        "LengthDropMean": f"{statistics.mean(len_drop):.1f}",
        "LengthGainMax": f"{-min(len_drop):.1f}",
        "LengthDropMax": f"{max(len_drop):.1f}",
        "RestLengthDropMin": f"{math.floor(min(rest_drop))}",
        "TablesGainMean": f"{statistics.mean(tab_gain):.0f}",
        # Q2's cost: the image-table methods per length level, and the highest cell
        **{f"ImageCost{level_macro(('length', lv))}":
           f"{statistics.median(d['costs'][m][('length', lv)] for m in img):.0f}"
           for lv in lengths},
        "CostCellMax": f"{max(cells):.1f}",
        # Q4 and Choosing a method
        "FusionMeanErrorPct": f"{100 - acc['score_fusion']:.1f}",
        "FusionOverGotoh": f"{acc['score_fusion'] - acc['gotoh']:.1f}",
        "PriorOverViterbi": f"{acc['fusion'] - acc['dp']:.1f}",
    }
    return vals


def check_numbers(second: Path = SECOND_LABELS, key: Path = CHECK_KEY,
                  manual: Path = CHECK_MANUAL, batches: Path = LABEL_BATCHES,
                  discipline: Path = DISCIPLINE,
                  discipline_check: Path = DISCIPLINE_CHECK,
                  labels: Path = Q.LABELS) -> dict[str, str]:
    """The checks of the labels (Section 6, Appendix H, Limitations), from their tracked
    files and with the scripts that made them: the second labelling against the merged
    labels (importance_merge.agreement), the blind check (importance_sample.compare: its
    confusion has the checker's label in rows), the share of core labels per agent batch,
    and the discipline check of 50 labels, whose lectures must carry the label checked."""
    import csv
    import importance_merge as IM
    import importance_sample as IS
    merged = IS.read_labels(labels)
    redo, problems = IM.read_labels(second)
    if problems:
        raise ValueError(f"{second}: {problems[0]}")
    rows, problems = IM.agreement(merged, redo)
    if problems:
        raise ValueError(f"{second}: {problems[0]}")
    overall = rows[-1]
    pairs = [(merged[u], redo[u]) for u in redo if merged[u] != redo[u]]
    n, exact, within, conf = IS.compare(key, manual)
    shares = []
    for f in IM.label_files(batches):
        lab = [int(r["label"]) for r in A.read_csv(f)]
        shares.append(100 * lab.count(2) / len(lab))
    with open(discipline, newline="", encoding="utf-8") as fh:
        final = {(r["corpus"], r["lecture_id"]): f"{r['field_code']} {r['field']}"
                 for r in csv.DictReader(fh)}
    with open(discipline_check, newline="", encoding="utf-8") as fh:
        checked = list(csv.DictReader(fh))
    for r in checked:
        if final.get((r["corpus"], r["lecture_id"])) != r["assigned_v3"]:
            raise ValueError(f"{discipline_check}: {r['lecture_id']} was checked with "
                             f"{r['assigned_v3']!r}, its label is "
                             f"{final.get((r['corpus'], r['lecture_id']))!r}")
    verdicts = [r["verdict"] for r in checked]
    if set(verdicts) - {"yes", "near", "no"}:
        raise ValueError(f"{discipline_check}: verdicts {sorted(set(verdicts))}")
    return {
        "SecondLectures": _int(len(rows) - 1), "SecondUnits": _int(overall["n_units"]),
        "SecondExactPct": f"{overall['exact_pct']:.1f}",
        "SecondWithinPct": f"{overall['within_one_pct']:.1f}",
        "SecondKappaLinear": f"{overall['kappa_linear']:.2f}",
        "SecondKappaQuadratic": f"{overall['kappa_quadratic']:.2f}",
        "SecondDisagree": _int(len(pairs)),
        "SecondCoreSupporting": _int(sum(set(p) == {1, 2} for p in pairs)),
        "BlindUnits": _int(n), "BlindExact": _int(exact),
        "BlindExactPct": f"{100 * exact / n:.0f}", "BlindWithin": _int(within),
        "BlindDisagree": _int(n - exact), "BlindCoreOverSupporting": _int(conf[2][1]),
        "BatchCoreMin": f"{min(shares):.0f}", "BatchCoreMax": f"{max(shares):.0f}",
        "DisciplineChecked": _int(len(verdicts)),
        "DisciplineRight": _int(verdicts.count("yes")),
        "DisciplineNear": _int(verdicts.count("near")),
        "DisciplineWrong": _int(verdicts.count("no")),
    }


def example_numbers(ex: Mapping[str, Any]) -> dict[str, str]:
    """Fig. 2's caption: the filter that chose the lecture, the lecture's size, and LLM
    single call's core units on the wrong slide."""
    (s0, s1), (u0, u1) = EXAMPLE_FILTERS[ex["filter"]]
    pred = ex["methods"]["llm_single_call"]["pred"]
    core = [i for i, lab in enumerate(ex["labels"]) if lab == 2]
    return {"ExampleSlidesFrom": _int(s0), "ExampleSlidesTo": _int(s1),
            "ExampleUnitsFrom": _int(u0), "ExampleUnitsTo": _int(u1),
            "ExampleSlides": _int(ex["n_slides"]), "ExampleUnits": _int(ex["n_units"]),
            "ExampleCore": _int(len(core)),
            "ExampleSingleCoreWrong": _int(sum(pred[i] != ex["truth"][i] for i in core))}


def numbers_tex(bj: Mapping[str, Any], q4e: Sequence[Mapping[str, Any]],
                fb: Mapping[str, Mapping[str, int]], td: Sequence[Mapping[str, Any]],
                ud: Mapping[str, Any], top: Sequence[str],
                more: Optional[Mapping[str, str]] = None) -> str:
    """The numbers the text quotes, as LaTeX macros (results/tables/numbers.tex), so none
    is typed by hand. `top`: the methods of Q1's top band; `more`: further macros
    (text_numbers)."""
    q4 = {r["method"]: r for r in q4e}
    rest = [r["ratio"] for r in q4e if r["method"] not in ORDER_FORCED]
    forced = [q4[m]["ratio"] for m in ORDER_FORCED]
    tops = [q4[m]["ratio"] for m in top]
    drop = [r["test"] - r["chosen"] for r in td]
    over5 = [r["method"] for r in td if round(r["test"] - r["chosen"], 1) <= -5]
    over5.sort(key=lambda m: next(r["test"] - r["chosen"] for r in td if r["method"] == m))
    single = fb["llm_single_call"]
    vals = {
        # back-jumps of the true path (Q1, Conclusion)
        "TestLectures": _int(bj["lectures"]),
        "BackLectures": _int(bj["back_lectures"]),
        "BackLecturesPct": f"{100 * bj['back_lectures'] / bj['lectures']:.1f}",
        "BackMavilsLectures": _int(bj["corpus"]["mavils"][1]),
        "MavilsLectures": _int(bj["corpus"]["mavils"][0]),
        "BackSteps": _int(bj["back"]),
        "BackStepsPct": f"{100 * bj['back'] / bj['steps']:.1f}",
        "BackChangesPct": f"{100 * bj['back'] / bj['changes']:.1f}",
        # Q4 against importance-blind errors
        "CorePct": f"{100 * ud['core'] / ud['units']:.1f}",
        "CoreShare": f"{ud['core'] / ud['units']:.2f}",
        "FusionErrorPct": f"{q4['score_fusion']['error']:.1f}",
        "FusionExpectedPct": f"{q4['score_fusion']['expected']:.1f}",
        "FusionObservedPct": f"{q4['score_fusion']['observed']:.1f}",
        "RatioTopMin": f"{min(tops):.2f}", "RatioTopMax": f"{max(tops):.2f}",
        "RatioRestMin": f"{min(rest):.2f}", "RatioRestMax": f"{max(rest):.2f}",
        "RatioForcedMin": f"{min(forced):.2f}", "RatioForcedMax": f"{max(forced):.2f}",
        "AloneRatioMin": f"{min(q4[m]['ratio'] for m in LLM_ALONE):.2f}",
        "AloneRatioMax": f"{max(q4[m]['ratio'] for m in LLM_ALONE):.2f}",
        "RatioArgmax": f"{q4['argmax']['ratio']:.2f}",
        # the top band in Q4 (Q1's reading of its accuracy, Conclusion): important units on
        # the wrong slide as a share of all units and of the important units, and the share
        # of its wrong units that are supporting or filler
        "TopCoreWrongMin": f"{min(q4[m]['observed'] for m in top):.1f}",
        "TopCoreWrongMax": f"{max(q4[m]['observed'] for m in top):.1f}",
        "TopCoreRightMin": f"{100 - max(q4[m]['observed'] for m in top) * ud['units'] / ud['core']:.0f}",
        "TopCoreRightMax": f"{100 - min(q4[m]['observed'] for m in top) * ud['units'] / ud['core']:.0f}",
        "TopNonCoreErrorMin": f"{min(100 * (1 - q4[m]['observed'] / q4[m]['error']) for m in top):.0f}",
        "ForcedObservedMin": f"{min(q4[m]['observed'] for m in ORDER_FORCED):.1f}",
        "ForcedObservedMax": f"{max(q4[m]['observed'] for m in ORDER_FORCED):.1f}",
        "RestMethods": _int(len(rest)),
        # the LLM fallbacks
        "FallbackSingle": _int(single["fallback_lectures"]),
        "FallbackSinglePct": f"{100 * single['fallback_lectures'] / single['lectures']:.1f}",
        "TruncatedSingle": _int(single["truncated"]),
        "FallbackOthers": (_count(0) + " of them" if not any(
            fb[m]["fallback_lectures"] for m in FALLBACK_METHODS[1:4]) else
            _join([_int(fb[m]["fallback_lectures"]) for m in FALLBACK_METHODS[1:4]])
            + " of them"),
        "FallbackHier": _count(fb["hierarchical"]["fallbacks"]),
        "HierAnswers": _int(fb["hierarchical"]["answers"]),
        # tuning against test
        "TuneLower": _int(sum(x < 0 for x in drop)),
        "TuneMethods": _int(len(drop)),
        "TuneMedianDrop": f"{-sorted(drop)[len(drop) // 2]:.1f}",
        "TuneThreeFour": _int(sum(-4 <= x <= -3 for x in drop)),
        "TuneOverFive": _int(len(over5)),
        "TuneOverFiveNames": _join([tex(N.method_name(m)) for m in over5]),
        "TuneHigher": _join([tex(N.method_name(r["method"])) for r in td
                             if r["test"] - r["chosen"] >= 0]),
    }
    for k, v in (more or {}).items():
        put(vals, k, v)
    head = ["% Written by scripts/paper_assets.py from the results; do not edit by hand."]
    return "\n".join(head + [rf"\newcommand{{\{k}}}{{{v}\xspace}}" for k, v in vals.items()]) + "\n"


# --------------------------------------------------------------------------- #
# Figures: the house style and the layout check
# --------------------------------------------------------------------------- #

def style() -> str:
    """Set the house style of every figure (matplotlib's rcParams) and return the font
    used: the first of FONTS that matplotlib has. One place, so the six figures share
    sizes, line widths and colours; 7.5 pt text by default and nothing under MIN_PT."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    font = next(f for f in FONTS if f in have)
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font], "font.size": TEXT_PT,
        "axes.labelsize": TITLE_PT, "xtick.labelsize": TEXT_PT, "ytick.labelsize": TEXT_PT,
        "legend.fontsize": TEXT_PT, "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.labelcolor": "#222222", "ytick.labelcolor": "#222222",
        "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.bbox": "standard", "savefig.pad_inches": 0,
        # the layout check measures texts on the raster canvas: at 300 dpi their widths
        # match the PDF's (at 100 dpi hinting makes them about 4 % narrower)
        "figure.dpi": 300})
    return font


def _plt():
    style()
    import matplotlib.pyplot as plt
    return plt


class LayoutError(RuntimeError):
    """A figure failed its layout check (a text too small, clipped, or on another)."""


def drawn_texts(fig) -> list:
    """The texts the renderer really draws, recorded during one draw of the canvas.
    fig.findobj would also return tick labels outside the view, which are never drawn."""
    from matplotlib.text import Text
    seen: list = []
    original = Text.draw

    def draw(self, renderer):
        if self.get_visible() and self.get_text().strip():
            seen.append(self)
        return original(self, renderer)

    Text.draw = draw
    try:
        fig.canvas.draw()
    finally:
        Text.draw = original
    return seen


def layout_problems(fig, boxes: Sequence[tuple[Any, tuple[float, float, float, float]]] = ()
                    ) -> list[str]:
    """The figure's layout problems (empty = none): a text under MIN_PT, a text not
    wholly on the canvas, two texts overlapping, or a text of `boxes` (text, (x0, y0,
    x1, y1) in inches) not inside its rectangle. Extents are the rendered ones, so the
    check holds for the font really used."""
    from matplotlib.text import Text
    texts = drawn_texts(fig)
    r = fig.canvas.get_renderer()

    def extent(t):           # the text alone (an annotation's own extent adds its arrow)
        return Text.get_window_extent(t, r)

    dpi = fig.dpi
    tol = 0.25 * dpi / 72                      # a quarter point
    W, H = fig.bbox.width, fig.bbox.height
    ext = [(t, extent(t)) for t in texts]
    out = []
    for t, b in ext:
        if t.get_fontsize() < MIN_PT - 1e-9:
            out.append(f"{t.get_text()!r} is {t.get_fontsize():g} pt")
        if b.x0 < -tol or b.y0 < -tol or b.x1 > W + tol or b.y1 > H + tol:
            out.append(f"{t.get_text()!r} leaves the canvas")
    for i, (t, b) in enumerate(ext):
        for u, c in ext[i + 1:]:
            if (min(b.x1, c.x1) - max(b.x0, c.x0) > tol
                    and min(b.y1, c.y1) - max(b.y0, c.y0) > tol):
                out.append(f"{t.get_text()!r} overlaps {u.get_text()!r}")
    for t, (x0, y0, x1, y1) in boxes:
        b = extent(t)
        if (b.x0 < x0 * dpi - tol or b.y0 < y0 * dpi - tol or b.x1 > x1 * dpi + tol
                or b.y1 > y1 * dpi + tol):
            out.append(f"{t.get_text()!r} is not inside its box")
    return out


def save(fig, stem: str, out: Path, boxes: Sequence = ()) -> list[Path]:
    """Check the layout, then write the figure as PDF on its fixed canvas (no
    tight bounding box: the width stays FIG_W, so LaTeX prints it 1:1)."""
    import matplotlib.pyplot as plt
    problems = layout_problems(fig, boxes)
    if problems:
        plt.close(fig)
        raise LayoutError(f"{stem}: " + "; ".join(problems[:12])
                          + (f" (and {len(problems) - 12} more)" if len(problems) > 12 else ""))
    path = out / f"{stem}.pdf"
    fig.savefig(path, **H.save_options(path, 200))
    plt.close(fig)
    return [path]


def text_width(fig, text: str, size: float, weight: str = "normal") -> float:
    """The width of `text` in inches, rendered in the current font."""
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def wrap(fig, text: str, width: float, size: float, weight: str = "normal") -> list[str]:
    """`text` broken at spaces into lines no wider than `width` inches (a single word
    wider than that stays one line, and the layout check reports it)."""
    lines: list[str] = []
    for word in text.split(" "):
        if lines and text_width(fig, lines[-1] + " " + word, size, weight) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def line_h(size: float, spacing: float = 1.2) -> float:
    """The height of one text line in inches."""
    return size * spacing / 72


# --------------------------------------------------------------------------- #
# Heatmaps (Q2 accuracy and cost, Q4)
# --------------------------------------------------------------------------- #

def heat_keys() -> list[Key]:
    return [ALL] + [(a, lv) for _, a, lv in Q.COLUMNS[1:]]


def column_groups() -> list[tuple[str, int, int]]:
    """The nine columns' groups and their headers: All, then Length, Images, Tables."""
    out = [("All", 0, 0)]
    for axis in Q.AXES:
        idx = [j for j, (_, a, _) in enumerate(Q.COLUMNS) if a == axis]
        out.append((N.AXIS_NAMES[axis], idx[0], idx[-1]))
    return out


def heat_levels() -> list[str]:
    """The level name over every column; none over All."""
    return [""] + [N.level_name(a, lv) for _, a, lv in Q.COLUMNS[1:]]


def fig_q2(d: Mapping[str, Any], out: Path) -> None:
    keys = heat_keys()
    q1 = d["q1"]
    n_test = len(next(iter(d["rows"].values())))
    counts = [f"{n:,}" for n in [n_test] + [d["q2"][k]["n"] for k in keys[1:]]]
    names = [N.method_name(r["method"]) for r in q1]
    ranks = [r["rank"] for r in q1]
    kinds = [r["kind"] for r in q1]
    common = {"count_label": "lectures", "kind_style": KIND_STYLE, "band_rows": TOP_BAND}
    acc = [[r["accuracy"]] + [d["q2"][k]["acc"][r["method"]] for k in keys[1:]] for r in q1]
    _plt()
    fig = H.draw_paper_heatmap(acc, names, ranks, kinds, column_groups(), heat_levels(), counts,
                               cmap="Blues", vmin=20, vmax=80, fmt=lambda v: f"{v:.1f}",
                               cbar_label="Accuracy (%)", **common)
    save(fig, "fig_q2_accuracy", out)
    cost = [[d["costs"][r["method"]][k] for k in keys] for r in q1]
    vmax = 70.0
    fig = H.draw_paper_heatmap(cost, names, ranks, kinds, column_groups(), heat_levels(), counts,
                               cmap="YlOrBr", vmin=0, vmax=vmax,
                               fmt=lambda v: seconds(v, latex=False),
                               cbar_label="Seconds per lecture",
                               extend="max" if max(max(c) for c in cost) > vmax else "neither",
                               **common)
    save(fig, "fig_q2_cost", out)


def fig_q4(d: Mapping[str, Any], out: Path) -> None:
    """Q4's heatmap: rows sorted by the first column (as step 16 wrote them), so not in
    Q1 order; the rank column carries Q1's rank and there is no top-band rule."""
    rows = d["q4"]
    cols = Q.COLUMNS
    rank = {r["method"]: r["rank"] for r in d["q1"]}
    counts = [f"{int(rows[0][f'{c}_units']):,}" for c, _, _ in cols]
    vals = [[float(r[f"{c}_pct"]) for c, _, _ in cols] for r in rows]
    _plt()
    fig = H.draw_paper_heatmap(vals, [N.method_name(r["method"]) for r in rows],
                               [rank[r["method"]] for r in rows],
                               [d["kinds"][r["method"]] for r in rows], column_groups(),
                               heat_levels(), counts, count_label="units", cmap="Reds", vmin=0,
                               vmax=Q.VMAX, fmt=lambda v: f"{v:.1f}",
                               cbar_label="Important units on the wrong slide (% of all units)",
                               kind_style=KIND_STYLE, band_rows=None)
    save(fig, "fig_q4_heatmap", out)


# --------------------------------------------------------------------------- #
# Q1 scatter
# --------------------------------------------------------------------------- #

def spread_labels(targets: Sequence[float], gap: float, lo: float, hi: float) -> list[float]:
    """1-D label placement: positions as close to `targets` as possible, at least `gap`
    apart, inside [lo, hi]. Overlapping labels are merged into a block centred on their
    targets' mean, until no two blocks overlap (the usual cluster merge). Returns the
    positions in the order of `targets`."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    blocks = [[i] for i in order]  # each block: label indices, ascending target

    def start(b: list[int]) -> float:
        mean = sum(targets[i] for i in b) / len(b)
        s = mean - gap * (len(b) - 1) / 2
        return min(max(s, lo), hi - gap * (len(b) - 1))

    merged = True
    while merged:
        merged = False
        for j in range(len(blocks) - 1):
            a, b = blocks[j], blocks[j + 1]
            if start(a) + gap * len(a) > start(b) + 1e-9:
                blocks[j:j + 2] = [a + b]
                merged = True
                break
    pos = [0.0] * len(targets)
    for b in blocks:
        s = start(b)
        for k, i in enumerate(b):
            pos[i] = s + k * gap
    return pos


def _crosses(a: tuple, b: tuple) -> bool:
    """Whether segments a and b ((x, y), (x, y)) cross at a point inside both."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = side(*b, a[0]), side(*b, a[1])
    d3, d4 = side(*a, b[0]), side(*a, b[1])
    return d1 * d2 < 0 and d3 * d4 < 0


def _cuts_box(seg: tuple, box: tuple) -> bool:
    """Whether the segment passes through the box (x0, y0, x1, y1): it crosses one of
    the box's sides, or one of its ends lies inside."""
    x0, y0, x1, y1 = box
    if any(x0 < x < x1 and y0 < y < y1 for x, y in seg):
        return True
    sides = (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)),
             ((x0, y1), (x0, y0)))
    return any(_crosses(seg, e) for e in sides)


def crossing_leaders(lines: Sequence[tuple]) -> list:
    """The points whose leader crosses another leader or another name's box; `lines`
    holds (point, label y, leader segment, name box) per name."""
    bad = []
    for i, (p, _, seg, _) in enumerate(lines):
        for j, (q, _, other, box) in enumerate(lines):
            if i != j and (_crosses(seg, other) or _cuts_box(seg, box)):
                bad += [p, q]
    return bad


#: Where a rank label may go around its point, tried in this order: (dx, dy) in points,
#: horizontal and vertical alignment. The first is the house offset (3 pt right, 2 up).
LABEL_SPOTS = ((3, 2, "left", "bottom"), (-3, 2, "right", "bottom"),
               (3, -2, "left", "top"), (-3, -2, "right", "top"), (0, 4, "center", "bottom"),
               (0, -4, "center", "top"), (5, 0, "left", "center"), (-5, 0, "right", "center"),
               (0, 7, "center", "bottom"), (0, -7, "center", "top"))


def place_labels(ax, pts: Sequence[tuple], renderer, taken: Sequence = ()) -> None:
    """Rank labels next to their points without overlap (a small label-repel loop): each
    label takes the first spot of LABEL_SPOTS where it overlaps no point, no label placed
    before it, not the legend and nothing in `taken` (window extents); if every spot is
    taken, the first one (the layout check then reports it)."""
    from matplotlib.transforms import Bbox
    r = 3.6 * ax.figure.dpi / 72
    marks = []
    for x, y, _, _ in pts:
        px, py = ax.transData.transform((x, y))
        marks.append(Bbox.from_extents(px - r, py - r, px + r, py + r))
    taken = list(taken) + ([ax.get_legend().get_window_extent(renderer)]
                           if ax.get_legend() else [])
    for x, y, rk, _ in sorted(pts, key=lambda p: (-p[1], p[0])):
        for i, (dx, dy, ha, va) in enumerate(LABEL_SPOTS):
            t = ax.annotate(str(rk), (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=TEXT_PT, ha=ha, va=va, color="#333333")
            bb = t.get_window_extent(renderer)
            if not any(bb.overlaps(o) for o in taken + marks) or i == len(LABEL_SPOTS) - 1:
                if any(bb.overlaps(o) for o in taken + marks):
                    t.remove()
                    t = ax.annotate(str(rk), (x, y), xytext=LABEL_SPOTS[0][:2],
                                    textcoords="offset points", fontsize=TEXT_PT, ha="left",
                                    va="bottom", color="#333333")
                    bb = t.get_window_extent(renderer)
                taken.append(bb)
                break
            t.remove()


#: The scatter's enlarged region: the accuracies of the dense group of methods near
#: 23 s (its x range is set from the labels' widths, so the names fit), and the x
#: range a point must lie in to belong to it.
ZOOM = {"y": (60.5, 73.5), "x": (20.5, 30.0)}
#: Names shortened in the zoom when the full names do not fit the panel (the one
#: shortening the layout review allows).
ZOOM_SHORT = {"llm_rerank_kbest": "LLM re-rank"}


def fig_q1_scatter(d: Mapping[str, Any], out: Path) -> None:
    """Accuracy against total seconds per lecture. (a) all methods, log seconds, rank
    numbers; (b), under it at the full width, the dense group near 23 s enlarged, every
    point named.

    In (b) the names of the dense column go in one column at its left, spread to at
    least 9 pt apart, and the names of the other points in one column at the right of
    the farthest of them; the x range is chosen so that both name columns fit the panel
    (the seconds scale is what is left over), and (a) marks the same range as a box (no
    connecting lines: they would cross (a)'s axis labels; the caption names the box)."""
    plt = _plt()
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, MultipleLocator, NullFormatter, NullLocator
    q1 = d["q1"]
    pts = [(q1_total(r), r["accuracy"], r["rank"], r["kind"], r["method"]) for r in q1]
    # (a) on top, (b) under it, both the full width: inches from the bottom edge
    W, Hh = FIG_W, 5.22
    left, right = 0.44, 0.08
    b_bottom, b_h, a_h, between = 0.36, 2.55, 1.80, 0.45
    a_bottom = b_bottom + b_h + between
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes((left / W, a_bottom / Hh, (W - left - right) / W, a_h / Hh))
    az = fig.add_axes((left / W, b_bottom / Hh, (W - left - right) / W, b_h / Hh))
    y0, y1 = ZOOM["y"]
    band = [p[1] for p in pts if p[2] <= TOP_BAND]
    band_lo, band_hi = min(band), max(band)
    argmax = next(p[1] for p in pts if p[4] == "argmax")
    for a in (ax, az):
        a.axhspan(band_lo, band_hi, color=EMPHASIS, zorder=0, lw=0)
        a.grid(True, zorder=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for kind in ("non-LLM", "mixed", "LLM"):      # triangles last, so none is hidden
            c, mk = KIND_STYLE[kind]
            xs = [p[0] for p in pts if p[3] == kind]
            ys = [p[1] for p in pts if p[3] == kind]
            a.scatter(xs, ys, s=36, color=c, marker=mk, edgecolors="white", linewidths=0.6,
                      zorder=3)

    # (a): all methods
    ax.set_xscale("log")
    ax.set_xlim(0.005, 200)
    ax.set_ylim(25, 75)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([30, 40, 50, 60, 70]))
    # centred: the zoom lines (zoom_lines) pass left and right of it
    ax.set_xlabel("Total cost per lecture (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    t_band = ax.text(0.006, (band_lo + band_hi) / 2, f"top band (ranks 1–{TOP_BAND})",
                     fontsize=TEXT_PT, color=TRUE_GREY, va="center", ha="left")
    ax.axhline(argmax, color="#999999", lw=0.6, ls=":", zorder=1)
    t_arg = ax.text(0.006, argmax + 0.6, "Argmax (baseline)", fontsize=TEXT_PT, color="#333333",
                    va="bottom", ha="left")
    for kind in ("non-LLM", "LLM", "mixed"):
        c, mk = KIND_STYLE[kind]
        ax.scatter([], [], s=36, color=c, marker=mk, edgecolors="white", linewidths=0.6,
                   label=kind)
    # the legend in the empty region under the top band, left of the costly methods
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False,
              handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3)

    # (b): the zoom's members, and its x range from the widths of their names. The
    # column's names sit at its left; every other point's name at its own right. The
    # scale (inches per second) is the largest at which every name fits the panel, and
    # at least the one that keeps the axis at or above 0 s.
    zin = [p for p in pts if y0 <= p[1] <= y1 and ZOOM["x"][0] <= p[0] <= ZOOM["x"][1]]
    dense_x = min(p[0] for p in zin)
    fig.canvas.draw()
    wz = az.get_window_extent().width / fig.dpi
    pad, lead, gap = 0.04, 0.5, 0.08
    pt = fig.dpi / 72.0

    def label(p, short=False):
        name = ZOOM_SHORT[p[4]] if short and p[4] in ZOOM_SHORT else N.method_name(p[4])
        return f"{p[2]} {name}"

    def scale(column, others):
        """The panel's x range (x_lo, x_hi) and scale k (inches per second) for this
        split of the names: the column's at its left, the others' at their right."""
        col_x = min(p[0] for p in column)
        lw = max(text_width(fig, label(p), TEXT_PT) for p in column)
        c_in = pad + lw + lead                  # the column's x, inches from the left

        def k_max(short):
            return min(((wz - pad - c_in - gap - text_width(fig, label(p, short), TEXT_PT))
                        / (p[0] - col_x) for p in others), default=1.0)

        k_min = c_in / col_x
        short = k_max(False) < k_min
        k = k_max(short)
        if k < k_min:
            raise LayoutError(f"fig_q1_scatter: the zoom's names do not fit ({k:.3f} < "
                              f"{k_min:.3f} in/s)")
        return col_x - c_in / k, col_x - c_in / k + wz / k, k, short

    def leaders(column, x_lo, k):
        """Each column name's leader as a segment in inches (from 2 pt right of the end
        of its name to its point), with the name's box, for the crossing check. The
        names are spread in the order of their points' heights."""
        ys = [(p[1] - y0) / (y1 - y0) * b_h for p in column]
        lo_in, hi_in = 5 / 72, b_h - 5 / 72
        tx = (min(p[0] for p in column) - x_lo) * k - lead
        h = TEXT_PT / 72
        res = []
        for p, ty, ly in zip(column, ys, spread_labels(ys, 9 / 72, lo_in, hi_in)):
            seg = ((tx + 2 / 72, ly), ((p[0] - x_lo) * k, ty))
            box = (tx - text_width(fig, label(p), TEXT_PT), ly - h / 2, tx, ly + h / 2)
            res.append((p, ly, seg, box))
        return res

    # The column holds the points of the dense group at its least cost; its names are
    # stacked at its left in the order of their points' heights, and each leader runs
    # straight from the end of its name to its point. Leaders that would cross one
    # another or another name move that point's name to the right of the point.
    moved: set[str] = set()
    while True:
        column = [p for p in zin if p[0] < dense_x + 0.5 and p[4] not in moved]
        others = [p for p in zin if p not in column]
        x_lo, x_hi, k, short = scale(column, others)
        lines = leaders(column, x_lo, k)
        bad = crossing_leaders(lines)
        if not bad:
            break
        moved.add(max(bad, key=lambda q: q[0])[4])
    az.set_xlim(x_lo, x_hi)
    az.set_ylim(y0, y1)
    az.yaxis.set_major_locator(FixedLocator(range(62, 74, 2)))
    az.xaxis.set_major_locator(MultipleLocator(5))
    az.set_xlabel("Total cost per lecture (s)")
    outside = [p for p in pts if not (x_lo <= p[0] <= x_hi and y0 <= p[1] <= y1)]
    if len(outside) + len(zin) != len(pts):
        raise LayoutError(f"fig_q1_scatter: the zoom's range {x_lo:.1f}-{x_hi:.1f} s holds "
                          "a point that is not in the dense group")
    ax.add_patch(Rectangle((x_lo, y0), x_hi - x_lo, y1 - y0, fill=False, lw=0.8,
                           ec=ZOOM_GREY, zorder=2))
    # (b)'s frame in the box's grey, all four sides, so that box and panel read as one
    for side in az.spines.values():
        side.set_visible(True)
        side.set_color(ZOOM_GREY)
        side.set_linewidth(0.8)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # the box's "b", just right of its top-right corner, as the panel letters; the rank
    # labels keep clear of it and of the zoom lines (drawn last, by zoom_lines)
    tag = ax.annotate("b", (x_hi, y1), xytext=(1.5, 1.5), textcoords="offset points",
                      fontsize=LETTER_PT, fontweight="bold", ha="left", va="center")
    place_labels(ax, [p[:4] for p in outside], r,
                 taken=[t_band.get_window_extent(r), t_arg.get_window_extent(r),
                        tag.get_window_extent(r)]
                 + line_boxes(zoom_segments(ax, az, (x_lo, y0, x_hi, y1))))

    to_px = az.transData.transform
    lo, hi = to_px((x_lo, y0))[1] + 5 * pt, to_px((x_lo, y1))[1] - 5 * pt
    # relpos (1, 0.5) and no patchA: the leader starts at the end of the name, not at
    # the middle of the name's box, from where a steep leader would run through the name
    # and leave through the name above
    leader = {"arrowstyle": "-", "lw": 0.5, "color": "#999999", "relpos": (1.0, 0.5),
              "patchA": None, "shrinkA": 2, "shrinkB": 3}
    box = az.get_window_extent()
    tx = box.x0 + ((min(q[0] for q in column) - x_lo) * k - lead) * fig.dpi
    for p, ly, _, _ in lines:
        px = to_px(p[:2])
        ty = box.y0 + ly * fig.dpi
        az.annotate(label(p), p[:2], xytext=((tx - px[0]) / pt, (ty - px[1]) / pt),
                    textcoords="offset points", fontsize=TEXT_PT, ha="right", va="center",
                    color="#222222", arrowprops=leader)
    # the other points: each name left-aligned at its point's right
    if others:
        ys = [to_px(p[:2])[1] for p in others]
        for p, ty, py in zip(others, ys, spread_labels(ys, 9 * pt, lo, hi)):
            az.annotate(label(p, short), p[:2], xytext=(gap * 72, (py - ty) / pt),
                        textcoords="offset points", fontsize=TEXT_PT, ha="left", va="center",
                        color="#222222")
    az.text(x_hi - 0.03 / k, band_hi - 0.1, "top band", fontsize=TEXT_PT, color=TRUE_GREY,
            ha="right", va="top")
    for a, letter in ((ax, "a"), (az, "b")):
        bb = a.get_position()
        fig.text(bb.x0 - 0.42 / W, bb.y1 + 0.06 / Hh, letter, fontsize=LETTER_PT,
                 fontweight="bold", ha="left", va="top")
    zoom_lines(fig, ax, az, (x_lo, y0, x_hi, y1))
    save(fig, "fig_q1_scatter", out)


#: The zoom box in (a), (b)'s frame and the lines between them.
ZOOM_GREY = "#888888"


def zoom_segments(ax, az, box: tuple[float, float, float, float]) -> list[tuple]:
    """The zoom lines in pixels: from the box's bottom corners (data units of ax) to
    the top corners of az."""
    x0, y0, x1, _ = box
    px = ax.transData.transform
    pb = az.get_window_extent()
    return [(tuple(px((x0, y0))), (pb.x0, pb.y1)), (tuple(px((x1, y0))), (pb.x1, pb.y1))]


def line_boxes(segs: Sequence[tuple], step: float = 3.0) -> list:
    """Small boxes along each segment (pixels), for a label placement that must keep
    clear of the lines."""
    from matplotlib.transforms import Bbox
    out = []
    for (ax_, ay_), (bx_, by_) in segs:
        n = max(1, int(((bx_ - ax_) ** 2 + (by_ - ay_) ** 2) ** 0.5 / step))
        for i in range(n + 1):
            x, y = ax_ + (bx_ - ax_) * i / n, ay_ + (by_ - ay_) * i / n
            out.append(Bbox.from_extents(x - 1, y - 1, x + 1, y + 1))
    return out


def zoom_lines(fig, ax, az, box: tuple[float, float, float, float]) -> None:
    """Show that (b) enlarges the box (x0, y0, x1, y1, data units) of (a): dashed lines
    in figure coordinates from the box's bottom corners to (b)'s top corners. A line
    that would pass through a text stops the run (LayoutError)."""
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    segs = zoom_segments(ax, az, box)
    texts = drawn_texts(fig)
    r = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for seg in segs:
        for t in texts:
            if t.get_text() == "b":
                continue
            e = Text.get_window_extent(t, r)
            if _cuts_box(seg, (e.x0, e.y0, e.x1, e.y1)):
                raise LayoutError(f"fig_q1_scatter: a zoom line crosses {t.get_text()!r}")
        (ax_, ay_), (bx_, by_) = fig.transFigure.inverted().transform(seg)
        fig.add_artist(Line2D([ax_, bx_], [ay_, by_], transform=fig.transFigure,
                              color=ZOOM_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1))


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

#: The pipeline's canvas height (inches), the rule between its two parts, and the rows
#: of the benchmark part (y from, to). Taller than wide: at the text width the task takes
#: two rows and the benchmark three, and at 7.5 pt and more the boxes' text needs the room.
PIPE_H = 5.14
PIPE_SEP = 2.28
PIPE_R3 = (0.03, 0.71)
PIPE_R2 = (0.84, 1.47)


class Canvas:
    """A figure with one axes over the whole canvas whose data units are inches, so
    the pipeline is laid out in print measures; every text placed with a box is
    remembered for the layout check (it must lie inside that box)."""

    def __init__(self, plt, w: float, h: float):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.w, self.h = w, h
        self.boxes: list = []

    def box(self, x0: float, y0: float, x1: float, y1: float, fill: str,
            round_: bool = True, ec: str = BORDER, lw: float = 0.8) -> tuple:
        from matplotlib.patches import FancyBboxPatch
        style = "round,pad=0,rounding_size=0.06" if round_ else "square,pad=0"
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style, fc=fill,
                                         ec=ec, lw=lw, zorder=1))
        return (x0, y0, x1, y1)

    def text(self, x: float, y: float, s: str, rect: Optional[tuple] = None, **kw):
        kw.setdefault("fontsize", TEXT_PT)
        kw.setdefault("color", "#222222")
        t = self.ax.text(x, y, s, zorder=4, **kw)
        if rect is not None:
            self.boxes.append((t, rect))
        return t

    def lines(self, x: float, top: float, lines: Sequence[str], rect: Optional[tuple],
              size: float = TEXT_PT, ha: str = "center", **kw) -> float:
        """Lines of text from `top` down, one Text per line; returns the y under them."""
        for s in lines:
            self.text(x, top, s, rect, fontsize=size, ha=ha, va="top", **kw)
            top -= line_h(size)
        return top

    def arrow(self, p: tuple, q: tuple, dashed: bool = False, color: str = BORDER) -> None:
        self.ax.annotate("", q, p, arrowprops={
            "arrowstyle": "-|>", "lw": 0.8, "color": color, "shrinkA": 0, "shrinkB": 0,
            "mutation_scale": 7, "ls": (0, (3, 2)) if dashed else "-"}, zorder=3)

    def path(self, pts: Sequence[tuple], dashed: bool = False, color: str = BORDER) -> None:
        """A routed line through `pts`, with the arrow head on its last segment."""
        xs, ys = zip(*pts[:-1])
        self.ax.plot(xs, ys, color=color, lw=0.8, ls=(0, (3, 2)) if dashed else "-",
                     zorder=3, solid_capstyle="butt")
        self.arrow(pts[-2], pts[-1], dashed, color)


def synthetic_table(n: int = 14, m: int = 9, seed: int = 0):
    """The pipeline's picture of a similarity table: noise 0.1-0.4 and a monotone
    diagonal band 0.7-1.0 with one revisit (a fixed seed, so the figure does not change
    between runs). A picture, not data."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.1, 0.4, size=(n, m))
    path = [min(m - 1, int(i * m / n)) for i in range(n)]
    path[9] = 3                                        # the lecturer goes back once
    for i, j in enumerate(path):
        t[i, j] = rng.uniform(0.7, 1.0)
    return t


def fig_pipeline(d: Mapping[str, Any], out: Path) -> None:
    """The task on one lecture (top) and the benchmark's steps (bottom). The counts come
    from the data and the registry; every text is checked to lie inside its box.

    At the text width the task takes two rows: slides and units, the similarity table
    and the method side by side; under them the cost bracket and its formula, and under
    those the output against the truth (the arrow from the method runs down the right
    margin, clear of the cost texts). The benchmark is a snake of three rows: corpora,
    preparation, attributes; the split under the attributes, its tuning set going left to
    the tuning; the final run under the tuning and the results beside it."""
    plt = _plt()
    from matplotlib.patches import Rectangle
    n_tun = sum(1 for r in d["selection"] if r["set"] == "tuning")
    n_test = sum(1 for r in d["selection"] if r["set"] == "test")
    n_levels = sum(len(d["order"][a]) for a in A.AXES)
    n_units = sum(int(r["n_units"]) for r in next(iter(d["rows"].values())))
    n_methods = len(d["methods"])
    n_combos = sum(g.n_combos() for g in T.load_grid().values())
    c = Canvas(plt, FIG_W, PIPE_H)
    fig, ax = c.fig, c.ax
    fig.canvas.draw()
    grey = TRUE_GREY
    pt, tpt = TEXT_PT, TITLE_PT
    lh = line_h(pt)
    sep = PIPE_SEP
    ax.plot([0.03, FIG_W - 0.03], [sep, sep], color="#CCCCCC", lw=0.5)
    c.text(0.08, (sep + PIPE_H) / 2, "One lecture", fontsize=tpt, color=grey, rotation=90,
           ha="center", va="center")
    c.text(0.08, sep / 2, "The benchmark", fontsize=tpt, color=grey, rotation=90,
           ha="center", va="center")
    top = PIPE_H - 0.06

    # 1-2: slides and transcript units
    sx = 0.26
    for i in range(4):
        x0, y0 = sx + 0.06 * i, top - 0.32 - 0.06 * i
        ax.add_patch(Rectangle((x0, y0), 0.46, 0.32, fc="white", ec=BORDER, lw=0.6, zorder=2))
        ax.plot([x0 + 0.06, x0 + 0.30], [y0 + 0.24] * 2, color="#BBBBBB", lw=1.1, zorder=2)
        ax.plot([x0 + 0.06, x0 + 0.23], [y0 + 0.17] * 2, color="#BBBBBB", lw=1.1, zorder=2)
    slides_right = sx + 0.06 * 3 + 0.46
    slides_bottom = top - 0.32 - 0.18
    slides_mid = (top + slides_bottom) / 2
    c.text(sx + 0.32, slides_bottom - 0.04, "Slides 1…m", fontsize=tpt, ha="center", va="top")
    bars_top = slides_bottom - 0.28
    lengths = (0.47, 0.34, 0.51, 0.30, 0.43, 0.26, 0.38)
    for i, ln in enumerate(lengths):
        y = bars_top - i * 0.055
        ax.add_patch(Rectangle((sx, y - 0.035), ln, 0.035, lw=0, zorder=2,
                               fc=KIND_STYLE["non-LLM"][0] if i in (2, 4) else "#BBBBBB"))
    bars_bottom = bars_top - 6 * 0.055 - 0.035
    bars_mid = (bars_top + bars_bottom) / 2
    y = c.lines(sx, bars_bottom - 0.04, ["Units u₁…uₙ"], None, size=tpt, ha="left")
    c.lines(sx, y, ["blue: important", "units"], None, ha="left", color=grey)

    # 4: the similarity table
    bx0, bx1 = 1.20, 2.60
    bmid = (bx0 + bx1) / 2
    b_rect = (bx0, 0, bx1, PIPE_H)
    y = c.lines(bmid, top, ["Similarity table"], b_rect, size=tpt, fontweight="bold")
    y = c.lines(bmid + 0.05, y - 0.01, ["slides →"], b_rect, color=grey)
    gx0, gx1, gy1 = 1.50, 2.30, y - 0.03
    gy0 = gy1 - 0.66
    gmid = (gy0 + gy1) / 2
    tbl = synthetic_table()
    ax.imshow(tbl, cmap="Greys", vmin=0, vmax=1, extent=(gx0, gx1, gy0, gy1), aspect="auto",
              interpolation="nearest", zorder=2)
    for j in range(1, tbl.shape[1]):
        ax.plot([gx0 + j * (gx1 - gx0) / tbl.shape[1]] * 2, [gy0, gy1], color="white",
                lw=0.4, zorder=2)
    for i in range(1, tbl.shape[0]):
        ax.plot([gx0, gx1], [gy0 + i * (gy1 - gy0) / tbl.shape[0]] * 2, color="white", lw=0.4,
                zorder=2)
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec=BORDER, lw=0.6,
                           zorder=2))
    c.text(gx0 - 0.07, gmid, "units →", color=grey, rotation=270, ha="center", va="center")
    ax.set_xlim(0, c.w)
    ax.set_ylim(0, c.h)
    y = c.lines(bmid, gy0 - 0.05, [f"{len(T.MEASURES)} measures: BM25,", "SPLADE, embeddings,",
                                   "cross-encoder, image,", "3 hybrids"], b_rect)
    b_bottom = c.lines(bmid, y - 0.02, ["cost: tables (s), GPU"], b_rect, color=COST_COLOUR)
    for (ax_, ay) in ((slides_right, slides_mid), (sx + max(lengths), bars_mid)):
        c.arrow((ax_ + 0.04, ay), (gx0 - 0.15, gmid))

    # 5: the method
    mx0, mx1 = 2.80, 4.40
    m_rect = (mx0, 0, mx1, PIPE_H)
    families = [N.FAMILY_NAMES[g] for g in GROUPS]
    m_top = top
    y = c.lines((mx0 + mx1) / 2, m_top - 0.05, ["Method"], m_rect, size=tpt, fontweight="bold")
    y = c.lines((mx0 + mx1) / 2, y, [f"{n_methods}, in {len(families)} families"], m_rect,
                color=grey)
    y = c.lines(mx0 + 0.08, y - 0.03, families, m_rect, ha="left")
    m_bottom = y - 0.05
    c.box(mx0, m_bottom, mx1, m_top, FILL["methods"])
    c.boxes = [(t, (mx0, m_bottom, mx1, m_top)) if r is m_rect else (t, r) for t, r in c.boxes]
    c.arrow((gx1 + 0.05, gmid), (mx0 - 0.02, gmid))
    m_cost = c.lines((mx0 + mx1) / 2, m_bottom - 0.04,
                     ["cost: compute (s), 1 CPU core;", "LLM calls and LLM (s)"],
                     (mx0, 0, mx1, PIPE_H), color=COST_COLOUR)

    # 7: the cost bracket under the table and the method, the formula under it
    yb = min(b_bottom, m_cost) - 0.04
    ax.plot([bx0 + 0.05, bx0 + 0.05, mx1 - 0.05, mx1 - 0.05], [yb + 0.05, yb, yb, yb + 0.05],
            color=COST_COLOUR, lw=0.6)
    f_bottom = c.lines((bx0 + mx1) / 2, yb - 0.03,
                       ["Cost per lecture = tables + compute + LLM,", "as if the method ran alone"],
                       (bx0, sep, mx1, yb), color=COST_COLOUR)

    # 6: the output against the truth, under the formula: the picture at the left, the
    # title and its reading at the right; the arrow from the method runs down the right
    # margin to the title
    ox0, ox1 = bx0, mx1
    o_rect = (ox0, sep, ox1, f_bottom)
    o_top = f_bottom - 0.05
    title = "One slide per unit"
    ix0, ix1, iy1 = ox0 + 0.30, ox0 + 1.50, o_top - 0.04
    iy0 = iy1 - 0.52
    inset = fig.add_axes((ix0 / c.w, iy0 / c.h, (ix1 - ix0) / c.w, (iy1 - iy0) / c.h))
    truth, pred = pipeline_example()
    units = list(range(1, len(truth) + 1))
    inset.step(units, truth, where="mid", color="#BBBBBB", lw=2.2)
    inset.step(units, pred, where="mid", color=KIND_STYLE["non-LLM"][0], lw=1.1)
    wrong = [u for u, a, b in zip(units, truth, pred) if a != b]
    inset.vlines(wrong, 0.15, 0.75, color=WRONG, lw=0.9)
    inset.set_xlim(0.5, len(truth) + 0.5)
    inset.set_ylim(0, max(truth) + 0.6)
    inset.set_xticks([])
    inset.set_yticks([])
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    c.text(ix0 - 0.05, (iy0 + iy1) / 2, "slide", color=grey, rotation=90, ha="right",
           va="center")
    c.lines((ix0 + ix1) / 2, iy0 - 0.02, ["unit"], o_rect, color=grey)
    tx0 = ix1 + 0.15
    y = c.lines(tx0, o_top, [title], o_rect, size=tpt, fontweight="bold", ha="left")
    y = c.lines(tx0, y - 0.02, wrap(fig, "Accuracy = share of units on their true slide",
                                    ox1 - tx0, pt), o_rect, ha="left")
    q4_lines = wrap(fig, "Q4: important units on the wrong slide (% of all units)", ox1 - tx0, pt)
    q4_y = y - 0.04 - lh * (len(q4_lines) - 0.5)
    c.lines(tx0, y - 0.04, q4_lines, o_rect, ha="left", color=Q4_COLOUR)
    q4_right = tx0 + max(text_width(fig, s, pt) for s in q4_lines)
    xa = mx1 + 0.10
    t_y = o_top - lh / 2
    c.path([(mx1 + 0.01, gmid), (xa, gmid), (xa, t_y),
            (tx0 + text_width(fig, title, tpt, "bold") + 0.04, t_y)])

    # the benchmark: row 1 (data), row 2 (split and tuning), row 3 (final run, results)
    gap_x, xr = 0.18, 4.50
    r1 = (PIPE_R2[1] + 0.13, sep - 0.05)
    r2, r3 = PIPE_R2, PIPE_R3

    def chain_box(x0, x1, yy, title, body, fill):
        rect = c.box(x0, yy[0], x1, yy[1], fill)
        inner = (x0, yy[0], x1, yy[1])
        y = c.lines((x0 + x1) / 2, yy[1] - 0.05, [title], inner, size=tpt, fontweight="bold")
        lines = [ln for part in body for ln in wrap(fig, part, x1 - x0 - 0.10, pt)]
        c.lines((x0 + x1) / 2, y - 0.01, lines, inner)
        return rect

    row1 = [(0.24, 1.56), (1.56 + gap_x, 3.06), (3.06 + gap_x, xr)]
    n_public = len({"m3av" if k.startswith("m3av") else k for k in N.CORPUS_NAMES})  # M³AV: one
    chain_box(*row1[0], r1, f"{n_public} public corpora",
              ["MaViLS, M³AV (A, B),", "MCD, Slide-Highlight"], FILL["data"])
    chain_box(*row1[1], r1, "Preparation",
              [f"{len(d['cats']):,} lectures;", "≤ 40 slides; units with", "a true slide"],
              FILL["data"])
    chain_box(*row1[2], r1, "Attributes", [f"{len(A.AXES)} attributes,", f"{n_levels} levels"],
              FILL["data"])
    for (a0, a1), (b0, _) in zip(row1, row1[1:]):
        c.arrow((a1 + 0.02, sum(r1) / 2), (b0 - 0.02, sum(r1) / 2))

    left = (0.24, 1.80)                          # the tuning and the final run
    sx0, sx1 = left[1] + gap_x + 0.02, xr        # the split (and the results)
    c.box(sx0, r2[0], sx1, r2[1], FILL["data"])
    # the title at the top left of the two sets, so the box is no taller than the
    # tuning's (the tuning set's arrow leaves under it)
    s_title = c.text(sx0 + 0.08, r2[1] - 0.06, "Split", (sx0, r2[0], sx1, r2[1]), fontsize=tpt,
                     fontweight="bold", ha="left", va="top")
    s_left = sx0 + 0.08 + text_width(fig, s_title.get_text(), tpt, "bold") + 0.10
    smid = (s_left + sx1 - 0.08) / 2
    comp = [(s_left, smid - 0.04), (smid + 0.04, sx1 - 0.08)]
    cy0, cy1 = r2[0] + 0.06, r2[1] - 0.06
    for (a, b), body in zip(comp, (["tuning", f"{n_tun} lectures"],
                                   ["test", f"{n_test} lectures", f"{n_units:,} units"])):
        c.box(a, cy0, b, cy1, "white", round_=False, ec="#999999", lw=0.6)
        c.lines((a + b) / 2, cy1 - 0.03, body, (a, cy0, b, cy1))
    ax_attr = sum(row1[2]) / 2
    c.arrow((ax_attr, r1[0] - 0.02), (ax_attr, r2[1] + 0.02))

    chain_box(*left, r2, "Tuning",
              [f"{n_methods} methods × {len(T.MEASURES)} measures;",
               f"{n_combos:,} combinations;", f"on the {n_tun} tuning lectures"], FILL["methods"])
    chain_box(*left, r3, "Final run", [f"{n_methods} methods, once,",
                                       f"on the {n_test} test lectures"], FILL["methods"])
    qx0, qx1 = sx0, xr
    c.box(qx0, r3[0], qx1, r3[1], FILL["results"])
    q_rect = (qx0, r3[0], qx1, r3[1])
    y = c.lines((qx0 + qx1) / 2, r3[1] - 0.05, ["Results"], q_rect, size=tpt, fontweight="bold")
    results = ["Q1 ranking and cost", "Q2 per level: length, pictures, tables",
               "Q3 highest / lowest per level",
               "Q4 important units on the wrong slide"]
    y = c.lines(qx0 + 0.12, y - 0.01, results, q_rect, ha="left")
    q4_line = y + lh / 2
    lmid = sum(left) / 2
    c.arrow((lmid, r2[0] - 0.02), (lmid, r3[1] + 0.02))                 # tuning -> final run
    c.arrow((left[1] + 0.02, sum(r3) / 2), (qx0 - 0.02, sum(r3) / 2))   # final run -> results
    # the split's two sets: tuning (dashed) left to Tuning, test down and left to the run
    t_y = cy0 + 0.12
    c.arrow((comp[0][0] - 0.01, t_y), (left[1] + 0.02, t_y), dashed=True)
    s_mid = (comp[1][0] + comp[1][1]) / 2
    yg = (r2[0] + r3[1]) / 2
    xin = left[1] - 0.25
    c.path([(s_mid, cy0), (s_mid, yg), (xin, yg), (xin, r3[1] + 0.02)])
    # the importance labels feed Q4: from the Q4 note down the right margin
    xm = FIG_W - 0.07
    c.path([(q4_right + 0.03, q4_y), (xm, q4_y), (xm, q4_line), (qx1 + 0.01, q4_line)],
           dashed=True, color=Q4_COLOUR)
    save(fig, "fig_pipeline", out, c.boxes)


def pipeline_example() -> tuple[list[int], list[int]]:
    """The pipeline's mini output: the true slide of 40 units over 9 slides, and a
    method's answer that leaves the truth for units 22-27. A picture, not data."""
    starts = (1, 5, 9, 14, 19, 22, 28, 33, 37)
    truth = [sum(1 for s in starts if s <= u) for u in range(1, 41)]
    pred = [8 if 22 <= u <= 27 else t for u, t in zip(range(1, 41), truth)]
    return truth, pred


# --------------------------------------------------------------------------- #
# The worked example: one test lecture, four methods
# --------------------------------------------------------------------------- #

#: The example's methods: the highest of Q1 (non-LLM), the decoder that may go back
#: (the MaViLS scheme), an LLM reading the text, and the lowest of Q1 (one piece per
#: slide, in order).
EXAMPLE_METHODS = ("score_fusion", "dp", "llm_single_call", "texttiling_dp")
#: The lecture filters, tried in order: (slides from, to), (units from, to), inclusive.
#: The second, wider one is used only when no test lecture passes the first.
EXAMPLE_FILTERS = (((15, 25), (100, 200)), ((12, 30), (80, 300)))
#: The similarity table shown: the measure most methods chose in tuning.
EXAMPLE_MEASURE = "image"
#: The importance labels (docs/IMPORTANCE.md) and their colours in the strip.
LABEL_NAMES = {2: "core", 1: "supporting", 0: "filler"}
LABEL_COLOURS = {2: "#555555", 1: "#BBBBBB", 0: "#FFFFFF"}


def select_example(acc: Mapping[str, Mapping[Key, float]], mean: Mapping[str, float],
                   size: Mapping[Key, tuple[int, int]],
                   methods: Sequence[str] = EXAMPLE_METHODS,
                   filters: Sequence = EXAMPLE_FILTERS) -> tuple[Key, int, float]:
    """The most typical test lecture for the example: among the lectures whose (slides,
    units) in `size` pass the first filter that any lecture passes, the one with the
    least sum over `methods` of |accuracy on the lecture - the method's mean accuracy|
    (both in %); a tie goes to the smaller (corpus, lecture_id), so the choice is
    deterministic. Returns (lecture, index of the filter used, the sum)."""
    for i, ((s0, s1), (u0, u1)) in enumerate(filters):
        cands = sorted(k for k, (s, u) in size.items() if s0 <= s <= s1 and u0 <= u <= u1)
        if cands:
            score = {k: sum(abs(acc[m][k] - mean[m]) for m in methods) for k in cands}
            best = min(cands, key=lambda k: (score[k], k))
            return best, i, score[best]
    raise ValueError("no test lecture passes any of the example's filters")


def example_lecture(d: Mapping[str, Any]) -> tuple[Key, int, float]:
    """select_example on the final run: per-lecture accuracy from step 13's tables, the
    methods' means from Q1, slides from the attributes and units from the run."""
    acc = {m: {(r["corpus"], r["lecture_id"]): 100 * A.lecture_accuracy(r)
               for r in d["rows"][m]} for m in EXAMPLE_METHODS}
    mean = {r["method"]: r["accuracy"] for r in d["q1"]}
    size = {(r["corpus"], r["lecture_id"]):
            (int(d["attributes"][(r["corpus"], r["lecture_id"])]["n_slides"]), int(r["n_units"]))
            for r in d["rows"][EXAMPLE_METHODS[0]]}
    return select_example(acc, mean, size)


def example_data(d: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the example draws and reports, for the selected lecture: its image
    table (read with step 11's loader), true slides, importance labels, and per method
    the slide of every unit, the accuracy and the cost on this lecture.

    The true slides are read from step 13's predictions (each row carries the unit's true
    slide), not from the prepared corpora: those are not in git, and every figure must
    build from a fresh clone. The table loader only needs the lecture's size from them."""
    import numpy as np
    key, filt, score = example_lecture(d)
    corpus, lec_id = key
    preds = {m: {int(r["unit_index"]): (int(r["true_slide"]), int(r["pred_slide"]))
                 for r in A.read_csv(Q.PREDICTIONS / f"{m}.csv.gz")
                 if (r["corpus"], r["lecture_id"]) == key} for m in EXAMPLE_METHODS}
    first = preds[EXAMPLE_METHODS[0]]
    if sorted(first) != list(range(len(first))):
        raise ValueError(f"{key}: the predictions do not number the units 0..n-1")
    truth = [first[i][0] for i in range(len(first))]
    n_slides = int(d["attributes"][key]["n_slides"])
    lec = T.Lecture(0, corpus, lec_id, np.asarray([t - 1 for t in truth]),
                    [""] * n_slides, [""] * len(truth))
    table = T.load_matrices([lec], (EXAMPLE_MEASURE,), T.tables_path(T.TEST_SET))[
        (EXAMPLE_MEASURE, lec.name)]
    labels = {int(r["unit_index"]): int(r["label"]) for r in A.read_csv(Q.LABELS)
              if (r["corpus"], r["lecture_id"]) == key}
    if sorted(labels) != list(range(len(truth))):
        raise ValueError(f"{key}: the importance labels do not cover the {len(truth)} units")
    costs = lecture_costs({m: d["rows"][m] for m in EXAMPLE_METHODS}, d["kinds"], d["load_s"])
    rank = {r["method"]: r["rank"] for r in d["q1"]}
    mean = {r["method"]: r["accuracy"] for r in d["q1"]}
    methods = {}
    for m in EXAMPLE_METHODS:
        pred = preds[m]
        if (sorted(pred) != list(range(len(truth)))
                or [pred[i][0] for i in range(len(truth))] != truth):
            raise ValueError(f"{key}: {m}'s true slides are not the lecture's")
        row = next(r for r in d["rows"][m] if (r["corpus"], r["lecture_id"]) == key)
        methods[m] = {"pred": [pred[i][1] for i in range(len(truth))],
                      "accuracy": 100 * A.lecture_accuracy(row), "cost": costs[m][key],
                      "rank": rank[m], "mean": mean[m], "kind": d["kinds"][m]}
        right = sum(p == t for p, t in zip(methods[m]["pred"], truth))
        if right != int(row["n_correct"]):
            raise ValueError(f"{key}: {m}'s predictions give {right} right, step 13 "
                             f"{row['n_correct']}")
    if table.shape != (len(truth), n_slides):
        raise ValueError(f"{key}: the table is {table.shape}, the lecture "
                         f"{len(truth)} x {n_slides}")
    return {"key": key, "filter": filt, "score": score, "table": np.asarray(table),
            "truth": truth, "labels": [labels[i] for i in range(len(truth))],
            "methods": methods, "levels": {a: d["cats"][key][a] for a in A.AXES},
            "n_slides": n_slides, "n_units": len(truth)}


def fig_example(ex: Mapping[str, Any], out: Path) -> None:
    """(a) The lecture's image table, standardised per unit, with the true slide of
    every unit; (b-e) the slide each of four methods gives every unit against the true
    slide, over a strip of the units' importance labels, yellow where a core unit is on
    the wrong slide. All five panels put the units on x and the slides on y (slide 1 at
    the bottom), so the true-slide line has the same shape in each. At the text width
    (a) spans the width on top and (b-e) are a 2 x 2 grid under it."""
    plt = _plt()
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    W, Hh = FIG_W, 4.40
    left, right = 0.42, 0.08
    fig = plt.figure(figsize=(W, Hh))

    def axes_in(x0, y0, x1, y1, **kw):
        return fig.add_axes((x0 / W, y0 / Hh, (x1 - x0) / W, (y1 - y0) / Hh), **kw)

    t = ex["table"]
    z = (t - t.mean(axis=1, keepdims=True)) / np.where(t.std(axis=1, keepdims=True) > 0,
                                                       t.std(axis=1, keepdims=True), 1.0)
    n, m = z.shape
    a_top = Hh - 0.08
    a_bottom = a_top - 1.25
    a = axes_in(left, a_bottom, W - right, a_top)
    a.imshow(z.T, cmap="Greys", aspect="auto", interpolation="nearest", origin="lower",
             extent=(0.5, n + 0.5, 0.5, m + 0.5))
    a.step(np.arange(1, n + 1), ex["truth"], where="mid", color="white", lw=1.5,
           path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])
    a.xaxis.set_major_locator(MultipleLocator(25))
    a.yaxis.set_major_locator(MultipleLocator(5))
    a.set_xlabel("Unit")
    a.set_ylabel("Slide")
    fig.text(0.04 / W, (Hh - 0.04) / Hh, "a", fontsize=LETTER_PT, fontweight="bold",
             va="top")

    # b-e: 2 x 2 small multiples under (a), the legend over them
    mid = (left + W - right) / 2
    px = ((left, mid - 0.09), (mid + 0.09, W - right))
    legend = ("grey: true slide; colour: method; strip: core / supporting / filler units, "
              "yellow = core unit on the wrong slide")
    top = a_bottom - 0.36                               # under (a)'s ticks and "Unit"
    lines = wrap(fig, legend, px[1][1] - px[0][0], TEXT_PT)
    for s in lines:
        fig.text(px[0][0] / W, top / Hh, s, fontsize=TEXT_PT, color="#444444", va="top")
        top -= line_h(TEXT_PT)
    head_h = line_h(TITLE_PT) + line_h(TEXT_PT) + 0.04
    strip_h = 0.08
    bottom = 0.40                                       # room for ticks and "Unit"
    plot_h = (top - 0.05 - bottom - 2 * (head_h + strip_h) - 0.12) / 2
    units = np.arange(1, n + 1)
    core = [i for i, lb in enumerate(ex["labels"]) if lb == 2]
    strip = np.array([[[int(LABEL_COLOURS[lb][k:k + 2], 16) / 255 for k in (1, 3, 5)]
                       for lb in ex["labels"]]])
    first = None
    for idx, (meth, v) in enumerate(ex["methods"].items()):
        r, col = divmod(idx, 2)
        x0, x1 = px[col]
        y_head = top - 0.05 - r * (head_h + plot_h + strip_h + 0.12)
        y_plot1 = y_head - head_h
        y_plot0 = y_plot1 - plot_h
        p = axes_in(x0, y_plot0, x1, y_plot1, sharex=first, sharey=first)
        s = axes_in(x0, y_plot0 - strip_h, x1, y_plot0, sharex=p)
        first = first or p
        colour = KIND_STYLE[v["kind"]][0]
        p.step(units, ex["truth"], where="mid", color="#BBBBBB", lw=2.5)
        pred = np.array([np.nan if q < 1 else q for q in v["pred"]], dtype=float)
        p.step(units, pred, where="mid", color=colour, lw=1.2)
        p.set_xlim(0.5, n + 0.5)
        p.set_ylim(0.5, m + 0.5)
        p.yaxis.set_major_locator(MultipleLocator(5))
        p.xaxis.set_major_locator(MultipleLocator(25))
        p.tick_params(labelbottom=False, labelleft=(col == 0))
        s.imshow(strip, aspect="auto", interpolation="nearest", extent=(0.5, n + 0.5, 0, 1))
        wrong = [i + 1 for i in core if v["pred"][i] != ex["truth"][i]]
        s.vlines(wrong, 0, 1, color=WRONG, lw=0.8)
        s.set_yticks([])
        s.set_ylim(0, 1)
        s.tick_params(labelbottom=(r == 1), length=2 if r == 1 else 0)
        for side in ("top", "right"):
            p.spines[side].set_visible(False)
        s.spines["top"].set_visible(False)
        fig.text(x0 / W, y_head / Hh, f"{'bcde'[idx]}  {N.method_name(meth)}",
                 fontsize=TITLE_PT, fontweight="bold", va="top")
        fig.text(x0 / W, (y_head - line_h(TITLE_PT)) / Hh,
                 f"accuracy {v['accuracy']:.1f} %, cost {seconds(v['cost'], latex=False)} s",
                 fontsize=TEXT_PT, va="top")
    fig.text((px[0][0] + px[1][1]) / 2 / W, 0.04 / Hh, "Unit", fontsize=TITLE_PT,
             ha="center", va="bottom")
    fig.text(px[0][0] / W - 0.30 / W, (bottom + top) / 2 / Hh, "Slide", fontsize=TITLE_PT,
             rotation=90, ha="center", va="center")
    save(fig, "fig_example", out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def tables(d: Mapping[str, Any]) -> dict[str, str]:
    out = {"tab_corpora": tab_corpora(d), "tab_categories": tab_categories(d),
           "tab_methods": tab_methods(d), "tab_q1": tab_q1(d), "tab_q3": tab_q3(d),
           "tab_q4": tab_q4(d)}
    out["tab_q2_full"] = tab_q2_full(d)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", type=Path, default=PREPARED,
                    help="the folder of step 2's summary.csv and <corpus>_dropped.csv "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    d = load()
    problems = check(d)
    if problems:
        print("\n".join(["PROBLEMS:"] + [f"   {p}" for p in problems]), file=sys.stderr)
        print("nothing written: the checks failed", file=sys.stderr)
        return 1
    md = measure_data(d)
    problems = check_measures(md, d)
    ud = unit_data(d["methods"])
    problems += check_units(d, ud)
    td = tuning_data(d)
    problems += check_tuning(td)
    problems += check_top_band(d["q1"])
    da = dataset_accuracy(d)
    problems += check_datasets(da, d)
    problems += check_split(d)
    bd = bootstrap_data(d)
    problems += check_bootstrap(bd, d)
    try:
        pr: Optional[dict[str, Any]] = read_prepared(args.prepared)
    except PreparedMissing as e:
        pr = None
        print(f"note: {e}; Table B1 is written as a placeholder and \\PrepCut and "
              "\\PrepPool are not defined", file=sys.stderr)
    if pr is not None:
        problems += check_prepared(pr, d)
    if problems:
        print("\n".join(["PROBLEMS:"] + [f"   {p}" for p in problems]), file=sys.stderr)
        print("nothing written: the checks of the measure table, the unit data, the "
              "tuning table or the preparation counts failed", file=sys.stderr)
        return 1
    q4e = q4_expected(d, ud)
    top = [r["method"] for r in d["q1"] if r["rank"] <= TOP_BAND]
    n_test = len(next(iter(d["rows"].values())))
    ex = example_data(d)
    sv = severity(ud)
    more: dict[str, str] = {}
    for part in (text_numbers(d, td, md["build"]["image"]), measure_numbers(md),
                 method_numbers(d, md), quoted_numbers(d, md, ud, top), check_numbers(),
                 example_numbers(ex), bootstrap_numbers(bd, top),
                 severity_numbers(sv, top)):
        for k, v in part.items():
            put(more, k, v)
    pc = prepare_counts(pr) if pr is not None else None
    if pc is not None:
        for k, v in prepare_numbers(pc).items():
            put(more, k, v)
    extra = {"tab_measures": tab_measures(md),
             "tab_q4_expected": tab_q4_expected(q4e, 100 * ud["core"] / ud["units"]),
             "tab_tuning": tab_tuning(td, md["n_tuning"], n_test),
             "tab_datasets": tab_datasets(da, d),
             "tab_bootstrap": tab_bootstrap(bd),
             "tab_severity": tab_severity(sv, d),
             "tab_prepare": tab_prepare(pc) if pc is not None else tab_prepare_missing(),
             "prompts": prompts_tex(),
             "numbers": numbers_tex(back_jumps(ud["truth"]), q4e, llm_fallbacks(d), td, ud,
                                    top, more)}
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, text in {**tables(d), **extra}.items():
        (TABLES / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"font: {style()}")
    try:
        fig_pipeline(d, FIGURES)
        fig_q1_scatter(d, FIGURES)
        fig_q2(d, FIGURES)
        fig_q4(d, FIGURES)
        fig_example(ex, FIGURES)
    except LayoutError as e:
        print(f"layout check failed: {e}", file=sys.stderr)
        return 1
    print(f"checks passed; tables in {TABLES}, figures in {FIGURES} "
          f"({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
