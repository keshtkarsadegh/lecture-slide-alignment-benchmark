#!/usr/bin/env python3
"""Step 16: Q4, the important units each method places on the wrong slide.

It reads the final run's per-unit predictions (data/final/predictions/<method>.csv.gz,
written by scripts/final_predictions.py), the importance labels of step 15
(data/importance/labels.csv.gz; label 2 = core = "important") and the lecture categories
of step 6 (data/categories/document_categories.csv), and writes data/analysis/q4_*. It
runs no method and pays no LLM call. Claude assigned the importance labels; our scripts
did the evaluation. The labelling is not part of any method's cost.

THE NUMBER. One per method and column:

    100 x (units with label 2 and pred_slide != true_slide)
        / (all units of the documents in the column)

pooled over the units of the column's documents, not a mean of per-lecture values: the
share of all the units in the category that are important and on the wrong slide. The
denominator is every unit, not only the important ones, so the number is the part of
the category's transcript where an important statement is shown with the wrong slide.

THE COLUMNS, nine, in this order: all documents; length short, medium, long; images
low, medium, high; tables none, some. No other axis. Rows: the 31 methods, ascending by
the "all documents" value (the fewest important units on the wrong slide first; a tie
goes by method name).

THE CHECKS (the run stops with exit code 1 and writes no table when one fails; the
results are in q4_checks.txt either way):
  1. every unit has exactly one label and, for each of the 31 methods, exactly one
     prediction, with the same true slide in both, and its document one category row;
  2. per method, the wrong units recounted from the predictions equal
     sum(n_units - n_correct) over step 13's per-lecture table
     (data/final/methods/<method>.csv.gz);
  3. the unit totals of the length levels, of the image levels and of the table levels
     each sum to the units of all documents.

OUTPUT (data/analysis/):
  q4_wrong_important.csv   method, then per column <column>_wrong_important,
                           <column>_units and <column>_pct
  q4_wrong_important.md    the percentages, readable (1 decimal)
  q4_heatmap.pdf, .png     the same as a heatmap
  q4_checks.txt            the checks and their results

    python scripts/analysis_q4.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analysis as A  # noqa: E402
import heatmap as H  # noqa: E402
from methods import METHODS  # noqa: E402

FINAL = A.FINAL
PREDICTIONS = FINAL / "predictions"
LABELS = REPO_ROOT / "data" / "importance" / "labels.csv.gz"
OUT = A.OUT

#: The label of an important (core) unit (docs/IMPORTANCE.md).
IMPORTANT = 2

#: The nine columns: (name, axis, level); axis None = all documents.
COLUMNS = (("all", None, None),
           ("length_short", "length", "short"),
           ("length_medium", "length", "medium"),
           ("length_long", "length", "long"),
           ("images_low", "images", "low"),
           ("images_medium", "images", "medium"),
           ("images_high", "images", "high"),
           ("tables_none", "tables", "none"),
           ("tables_some", "tables", "some"))
AXES = ("length", "images", "tables")

#: The heatmap's colour range, fixed so that two runs are drawn on the same scale.
VMAX = 45.0
#: Cell values above this get white text (readable on the dark reds).
WHITE_ABOVE = 25.0

Unit = tuple[str, str, int]  # (corpus, lecture_id, unit_index)


# --------------------------------------------------------------------------- #
# The units and the number
# --------------------------------------------------------------------------- #

def unit_key(row: Mapping[str, Any]) -> Unit:
    return (row["corpus"], row["lecture_id"], int(row["unit_index"]))


def label_table(labels: Sequence[Mapping[str, Any]]) -> dict[Unit, tuple[int, int]]:
    """Per unit its (true_slide, label); a unit labelled twice raises."""
    out: dict[Unit, tuple[int, int]] = {}
    for r in labels:
        k = unit_key(r)
        if k in out:
            raise ValueError(f"unit {k} has more than one label")
        out[k] = (int(r["true_slide"]), int(r["label"]))
    return out


def document_levels(cats: Sequence[Mapping[str, str]],
                    docs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """Per document of `docs` its level on each of AXES; a document with no category row,
    or with more than one, raises."""
    rows: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for r in cats:
        rows.setdefault((r["corpus"], r["lecture_id"]), []).append(r)
    out = {}
    for d in docs:
        if len(rows.get(d, [])) != 1:
            raise ValueError(f"document {d} has {len(rows.get(d, []))} category rows, not 1")
        out[d] = {a: rows[d][0][a] for a in AXES}
    return out


def wrong_units(preds: Sequence[Mapping[str, Any]],
                labels: Mapping[Unit, tuple[int, int]]) -> dict[Unit, bool]:
    """Per unit whether the method put it on a wrong slide (pred_slide -1, no slide, is
    wrong). Every labelled unit must have exactly one prediction with the label's true
    slide, and every prediction a label: anything else raises, since a unit counted
    without its label (or left out) would change the number silently."""
    out: dict[Unit, bool] = {}
    for r in preds:
        k = unit_key(r)
        if k not in labels:
            raise ValueError(f"unit {k} has a prediction but no label")
        if k in out:
            raise ValueError(f"unit {k} has more than one prediction")
        if int(r["true_slide"]) != labels[k][0]:
            raise ValueError(f"unit {k}: true slide {r['true_slide']} in the prediction, "
                             f"{labels[k][0]} in the labels")
        out[k] = int(r["pred_slide"]) != int(r["true_slide"])
    missing = len(labels) - len(out)
    if missing:
        raise ValueError(f"{missing} labelled units have no prediction")
    return out


def in_column(levels: Mapping[str, str], axis: Optional[str], level: Optional[str]) -> bool:
    return axis is None or levels[axis] == level


def column_units(labels: Mapping[Unit, tuple[int, int]],
                 levels: Mapping[tuple[str, str], Mapping[str, str]]) -> dict[str, int]:
    """Per column the units of its documents (the denominator, the same for every
    method)."""
    n = {c: 0 for c, _, _ in COLUMNS}
    for (corpus, lec, _) in labels:
        for c, axis, level in COLUMNS:
            if in_column(levels[(corpus, lec)], axis, level):
                n[c] += 1
    return n


def pct(wrong: int, units: int) -> Optional[float]:
    """100 x wrong / units; None for a column without units (no number, not 0)."""
    return 100.0 * wrong / units if units else None


def method_row(method: str, wrong: Mapping[Unit, bool],
               labels: Mapping[Unit, tuple[int, int]],
               levels: Mapping[tuple[str, str], Mapping[str, str]],
               units: Mapping[str, int]) -> dict[str, Any]:
    """One method's row: per column the important units on the wrong slide, the
    column's units and the percentage, pooled over the column's units."""
    counts = {c: 0 for c, _, _ in COLUMNS}
    for k, is_wrong in wrong.items():
        if is_wrong and labels[k][1] == IMPORTANT:
            for c, axis, level in COLUMNS:
                if in_column(levels[k[:2]], axis, level):
                    counts[c] += 1
    row: dict[str, Any] = {"method": method}
    for c, _, _ in COLUMNS:
        row[f"{c}_wrong_important"] = counts[c]
        row[f"{c}_units"] = units[c]
        row[f"{c}_pct"] = pct(counts[c], units[c])
    return row


def sort_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Ascending by the all-documents percentage, a tie by method name."""
    return sorted(rows, key=lambda r: (r["all_pct"] is None, r["all_pct"] or 0.0, r["method"]))


def csv_columns() -> list[str]:
    return ["method"] + [f"{c}_{s}" for c, _, _ in COLUMNS
                         for s in ("wrong_important", "units", "pct")]


# --------------------------------------------------------------------------- #
# Reading and the checks
# --------------------------------------------------------------------------- #

def step13_wrong(rows: Sequence[Mapping[str, str]]) -> int:
    return sum(int(r["n_units"]) - int(r["n_correct"]) for r in rows)


def analyse(preds: Mapping[str, Sequence[Mapping[str, Any]]],
            labels_rows: Sequence[Mapping[str, Any]],
            cats: Sequence[Mapping[str, str]],
            step13: Mapping[str, Sequence[Mapping[str, str]]],
            expected: Sequence[str]) -> tuple[list[Mapping], dict[str, int], list[str], list[str]]:
    """The table, the column units, the check lines and the problems (empty = all
    checks passed)."""
    lines: list[str] = []
    problems: list[str] = []
    rows: list[Mapping] = []
    units: dict[str, int] = {}

    # 1. one label and one prediction per unit, true slides agree, one category row
    try:
        if sorted(preds) != sorted(expected):
            raise ValueError(f"prediction files {sorted(preds)} are not the methods "
                             f"{sorted(expected)}")
        labels = label_table(labels_rows)
        docs = sorted({k[:2] for k in labels})
        levels = document_levels(cats, docs)
        wrong = {m: wrong_units(preds[m], labels) for m in expected}
        lines.append(f"1. {len(labels)} units in {len(docs)} documents: each has exactly one "
                     f"label and one prediction for each of the {len(expected)} methods, the "
                     "true slides agree, and each document has one category row: ok")
    except ValueError as e:
        problems.append(f"1. {e}")
        lines.append(f"1. FAILED: {e}")
        return rows, units, lines, problems

    # 2. the wrong units recount to step 13
    bad = [f"{m}: {sum(wrong[m].values())} wrong in the predictions, "
           f"{step13_wrong(step13.get(m, []))} in step 13" for m in expected
           if sum(wrong[m].values()) != step13_wrong(step13.get(m, []))]
    problems += [f"2. {b}" for b in bad]
    lines.append("2. per method, the wrong units recounted from the predictions equal "
                 "sum(n_units - n_correct) of data/final/methods/<method>.csv.gz: "
                 + ("ok for all " + str(len(expected)) if not bad else f"FAILED for {len(bad)}"))

    # 3. the levels of each axis add up to all documents
    units = column_units(labels, levels)
    for axis in AXES:
        cols = [c for c, a, _ in COLUMNS if a == axis]
        total = sum(units[c] for c in cols)
        ok = total == units["all"]
        if not ok:
            problems.append(f"3. {axis}: the levels hold {total} units, all documents "
                            f"{units['all']}")
        lines.append(f"3. {axis}: " + " + ".join(f"{units[c]:,}" for c in cols)
                     + f" = {total:,} units; all documents {units['all']:,}: "
                     + ("ok" if ok else "FAILED"))

    rows = sort_rows([method_row(m, wrong[m], labels, levels, units) for m in expected])
    return rows, units, lines, problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def column_title(c: str) -> str:
    return "all documents" if c == "all" else c.replace("_", " ")


def markdown(rows: Sequence[Mapping[str, Any]], units: Mapping[str, int]) -> str:
    lines = ["# Q4: important units placed on the wrong slide", "",
             "Each number is the important units the method placed on the wrong slide, as "
             "a percentage of all units in that category. Claude assigned the importance "
             "labels; our scripts did the evaluation.", ""]
    head = ["method"] + [f"{column_title(c)} ({units[c]:,})" for c, _, _ in COLUMNS]
    body = [[r["method"]] + ["-" if r[f"{c}_pct"] is None else f"{r[f'{c}_pct']:.1f}"
                             for c, _, _ in COLUMNS] for r in rows]
    lines += A._md_table(head, body)
    return "\n".join(lines) + "\n"


def heatmap(rows: Sequence[Mapping[str, Any]], units: Mapping[str, int], out: Path) -> None:
    """q4_heatmap.pdf and .png: the approved layout (column groups all | length | images
    | tables with gaps, Reds on a fixed 0-45 range), drawn by the project's one heatmap
    function (scripts/heatmap.py), which the paper's heatmaps share."""
    groups = [("", 0, 0)]
    for axis in AXES:
        idx = [j for j, (_, a, _) in enumerate(COLUMNS) if a == axis]
        groups.append((axis, idx[0], idx[-1]))
    ticks = ["all\ndocuments"] + [lv for _, _, lv in COLUMNS[1:]]
    H.draw_heatmap([[r[f"{c}_pct"] for c, _, _ in COLUMNS] for r in rows],
                   [r["method"] for r in rows],
                   [f"{t}\n({units[c]:,})" for t, (c, _, _) in zip(ticks, COLUMNS)], groups,
                   [out / "q4_heatmap.pdf", out / "q4_heatmap.png"], cmap="Reds", vmin=0,
                   vmax=VMAX, fmt=lambda v: f"{v:.1f}", white_above=WHITE_ABOVE,
                   cbar_label="% of all units in the category",
                   title="Important units placed on the wrong slide\n(% of all units in the "
                         "category; units per category in brackets)")


def write(rows: Sequence[Mapping[str, Any]], units: Mapping[str, int], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    A.write_csv(out / "q4_wrong_important.csv", rows, csv_columns())
    (out / "q4_wrong_important.md").write_text(markdown(rows, units), encoding="utf-8")
    heatmap(rows, units, out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    expected = sorted(METHODS)
    preds = {p.name[:-len(".csv.gz")]: A.read_csv(p)
             for p in sorted(PREDICTIONS.glob("*.csv.gz"))}
    step13 = {p.name[:-len(".csv.gz")]: A.read_csv(p)
              for p in sorted((FINAL / "methods").glob("*.csv.gz"))}
    rows, units, lines, problems = analyse(
        preds, A.read_csv(LABELS), A.read_csv(A.CATEGORIES / "document_categories.csv"),
        step13, expected)

    text = "\n".join(["Step 16: Q4's checks (scripts/analysis_q4.py)", ""] + lines
                     + ["", f"PROBLEMS: {len(problems)}"] + [f"   {p}" for p in problems]) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "q4_checks.txt").write_text(text, encoding="utf-8")
    print(text)
    if problems:
        print("no table written: the checks failed", file=sys.stderr)
        return 1
    write(rows, units, OUT)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
