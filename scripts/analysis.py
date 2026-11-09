#!/usr/bin/env python3
"""Step 14: the analysis of the final run (step 13), Q1 to Q3.

It reads the final run's per-lecture tables (data/final/methods/<method>.csv.gz), its
summary (data/final/summary.csv) and the lecture categories of step 6
(data/categories/document_categories.csv, level_counts.csv), and writes data/analysis/.
It runs no method and pays no LLM call. Everything is descriptive: mean accuracy, rank
and cost, nothing else.

  Q1  ranking with cost: the 31 methods by mean test accuracy (the mean over the 140
      lectures of each lecture's accuracy, in %), with a rank (1 = highest; methods with
      the same accuracy share the rank, the next rank is skipped: 1, 2, 2, 4) and the
      cost of the method's kind, as a total over the lectures and per lecture:
        non-LLM  compute time and table build time
        LLM      LLM calls, LLM seconds and tokens (the answers used: successful calls
                 and the successful attempt's duration only), and build time
        mixed    all of these
      The costs are step 13's: each method costed as if it ran alone (a combined
      method carries its members', a shared table counts once per method), and the
      build total adds the model load times once, as step 13's summary does.
  Q2  every method within each category level: for every axis of step 6 except corpus
      (text_source, length, images, tables, noise, text_density, discipline) and every
      level, each method's mean accuracy over the test lectures of the level and its
      rank among the 31 within the level, with n lectures per level.
  Q3  each method's best and worst category level, over all levels of all axes: best =
      the smallest rank, worst = the largest; a tie in rank goes to the higher accuracy
      (best) or the lower (worst); a tie in both lists every level.
      This is not the paper's Q3. The paper's Q3 is the other view of the same Q2 rows:
      at every level of every attribute, the one method with the highest and the one
      with the lowest accuracy; scripts/paper_assets.py (tab_q3) reads it from
      q2_<axis>_long.csv. q3_best_worst.csv is kept as a per-method view.

Ties are decided on the accuracy rounded to TIE_DECIMALS decimals: two means of the
same per-lecture values summed in a different order can differ in the last bit, and that
must not split a tie.

THE CHECKS (the run stops with exit code 1 and writes no table when one fails; the
results are in checks.txt either way):
  1. the 31 methods of methods/ are the per-lecture tables and the summary's rows;
  2. every method has the same test lectures, one row each, status ok;
  3. every test lecture has exactly one row in the categories, with one step-6 level
     on each axis;
  4. Q1's mean accuracy equals the summary's test_accuracy, and Q1's costs equal its
     cost totals and per-lecture costs;
  5. for every method and axis, the lecture-weighted average of the level means equals
     the overall mean;
  6. every step-6 level has test lectures and all 31 methods in it.

OUTPUT (data/analysis/):
  q1_ranking.csv             one row per method, by rank
  q2_<axis>_long.csv         one row per (level, method): axis, level, n, method, kind,
                             accuracy, rank
  q2_<axis>_wide.csv         one row per method (by overall rank), an accuracy and a
                             rank column per level
  q3_best_worst.csv          one row per method (by overall rank)
  q1_ranking.md, q2_levels.md, q3_best_worst.md   the same, readable
  checks.txt                 the checks and their results

    python scripts/analysis.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import categorise as C  # noqa: E402
import final_report as F  # noqa: E402
import tune as T  # noqa: E402
from methods import METHODS  # noqa: E402

FINAL = REPO_ROOT / "data" / "final"
CATEGORIES = REPO_ROOT / "data" / "categories"
OUT = REPO_ROOT / "data" / "analysis"

#: The axes of Q2 and Q3: step 6's, without corpus.
AXES = tuple(a for a in C.AXES if a != "corpus")
#: Decimals of the accuracy (in %) on which ties are decided (module docstring).
TIE_DECIMALS = 9
#: Tolerance of the checks that compare two computations of the same number.
TOL = 1e-9

Key = tuple[str, str]  # (corpus, lecture_id)

#: The costs, in step 13's names: (column of the per-lecture tables, kinds it applies to).
COSTS = (("compute_s", ("non-LLM", "mixed")),
         ("build_cost_s", ("non-LLM", "LLM", "mixed")),
         ("used_calls", ("LLM", "mixed")),
         ("used_seconds", ("LLM", "mixed")),
         ("used_prompt_tokens", ("LLM", "mixed")),
         ("used_completion_tokens", ("LLM", "mixed")))


# --------------------------------------------------------------------------- #
# Ranks and the three tables
# --------------------------------------------------------------------------- #

def _tie(v: float) -> float:
    return round(v, TIE_DECIMALS)


def ranks(values: Mapping[str, float]) -> dict[str, int]:
    """1 = highest value; equal values (to TIE_DECIMALS) share the rank, and the next
    rank is skipped (1, 2, 2, 4), so a rank says how many methods are above."""
    tied = {m: _tie(v) for m, v in values.items()}
    return {m: 1 + sum(1 for o in tied.values() if o > v) for m, v in tied.items()}


def lecture_accuracy(row: Mapping[str, Any]) -> float:
    """Step 13's per-lecture accuracy (final_report.lecture_accuracy: a run not ok
    counts 0), from the per-lecture table's strings."""
    return F.lecture_accuracy({"status": row["status"], "n_correct": int(row["n_correct"]),
                               "n_units": int(row["n_units"])})


def mean_pct(acc: Mapping[Key, float], keys: Sequence[Key]) -> float:
    return 100 * sum(acc[k] for k in keys) / len(keys)


def q1_rows(rows: Mapping[str, Sequence[Mapping[str, Any]]], kinds: Mapping[str, str],
            load_s: Mapping[str, float]) -> list[dict]:
    """Q1: per method its mean accuracy (%), rank, kind and the costs of its kind, as
    `<cost>_total` and `<cost>_per_lecture` (None where the kind has no such cost;
    `used_tokens` = prompt + completion). The build total adds the method's model load
    times (`load_s`) once, as step 13's summary does."""
    out = []
    for m, rs in rows.items():
        n = len(rs)
        r = {"method": m, "kind": kinds[m], "n_lectures": n,
             "accuracy": 100 * sum(lecture_accuracy(x) for x in rs) / n}
        for c, applies in COSTS:
            if kinds[m] in applies:
                tot = sum(float(x[c]) for x in rs) + (load_s.get(m, 0.0)
                                                      if c == "build_cost_s" else 0.0)
                r[f"{c}_total"], r[f"{c}_per_lecture"] = tot, tot / n
            else:
                r[f"{c}_total"] = r[f"{c}_per_lecture"] = None
        if r["used_prompt_tokens_total"] is None:
            r["used_tokens_total"] = r["used_tokens_per_lecture"] = None
        else:
            r["used_tokens_total"] = r["used_prompt_tokens_total"] + r["used_completion_tokens_total"]
            r["used_tokens_per_lecture"] = r["used_tokens_total"] / n
        out.append(r)
    rk = ranks({r["method"]: r["accuracy"] for r in out})
    for r in out:
        r["rank"] = rk[r["method"]]
    return sorted(out, key=lambda r: (r["rank"], r["method"]))


def q1_columns() -> list[str]:
    cols = ["rank", "method", "kind", "accuracy", "n_lectures"]
    for c in [c for c, _ in COSTS] + ["used_tokens"]:
        cols += [f"{c}_total", f"{c}_per_lecture"]
    return cols


def q2_rows(axis: str, accs: Mapping[str, Mapping[Key, float]], kinds: Mapping[str, str],
            levels: Mapping[Key, str], order: Sequence[str]) -> list[dict]:
    """Q2 for one axis, long form: per level (in step-6 order) and method its mean
    accuracy over the level's lectures and its rank among the methods within the level.
    A level without lectures has no rows (check 6 reports it)."""
    out = []
    for level in order:
        keys = sorted(k for k, lv in levels.items() if lv == level)
        if not keys:
            continue
        means = {m: mean_pct(a, keys) for m, a in accs.items()}
        rk = ranks(means)
        for m in sorted(means, key=lambda m: (rk[m], m)):
            out.append({"axis": axis, "level": level, "n": len(keys), "method": m,
                        "kind": kinds[m], "accuracy": means[m], "rank": rk[m]})
    return out


def q2_wide(long: Sequence[Mapping], q1: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """Q2 for one axis, wide form: one row per method in Q1's order, with the overall
    rank and accuracy and `<level> accuracy`, `<level> rank` per level."""
    levels = list(dict.fromkeys(r["level"] for r in long))
    cell = {(r["method"], r["level"]): r for r in long}
    cols = ["method", "kind", "overall_rank", "overall_accuracy"]
    for lv in levels:
        cols += [f"{lv} accuracy", f"{lv} rank"]
    out = []
    for q in q1:
        row = {"method": q["method"], "kind": q["kind"], "overall_rank": q["rank"],
               "overall_accuracy": q["accuracy"]}
        for lv in levels:
            c = cell.get((q["method"], lv))
            row[f"{lv} accuracy"] = c["accuracy"] if c else None
            row[f"{lv} rank"] = c["rank"] if c else None
        out.append(row)
    return out, cols


def best_worst(entries: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping], list[Mapping]]:
    """Q3's rule over one method's (axis, level, rank, accuracy) entries: best = the
    smallest rank, then the higher accuracy; worst = the largest rank, then the lower
    accuracy; what is still tied is kept, every entry of it."""
    lo = min(e["rank"] for e in entries)
    best = [e for e in entries if e["rank"] == lo]
    top = max(_tie(e["accuracy"]) for e in best)
    hi = max(e["rank"] for e in entries)
    worst = [e for e in entries if e["rank"] == hi]
    bottom = min(_tie(e["accuracy"]) for e in worst)
    return ([e for e in best if _tie(e["accuracy"]) == top],
            [e for e in worst if _tie(e["accuracy"]) == bottom])


def q3_rows(long_all: Sequence[Mapping], q1: Sequence[Mapping]) -> list[dict]:
    """Q3: per method (in Q1's order) its best and worst level over all axes, with the
    rank and accuracy there and the overall rank and accuracy for reference."""
    out = []
    for q in q1:
        entries = [r for r in long_all if r["method"] == q["method"]]
        best, worst = best_worst(entries)
        out.append({"method": q["method"], "kind": q["kind"], "overall_rank": q["rank"],
                    "overall_accuracy": q["accuracy"],
                    "best_level": "; ".join(f"{e['axis']}: {e['level']}" for e in best),
                    "best_rank": best[0]["rank"], "best_accuracy": best[0]["accuracy"],
                    "worst_level": "; ".join(f"{e['axis']}: {e['level']}" for e in worst),
                    "worst_rank": worst[0]["rank"], "worst_accuracy": worst[0]["accuracy"]})
    return out


Q3_COLUMNS = ["method", "kind", "overall_rank", "overall_accuracy", "best_level", "best_rank",
              "best_accuracy", "worst_level", "worst_rank", "worst_accuracy"]


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #

def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def check_methods(rows: Mapping[str, Sequence], summary: Mapping[str, Mapping],
                  expected: Sequence[str]) -> list[str]:
    """Check 1: the expected methods are the per-lecture tables and the summary rows."""
    out = []
    for name, have in (("per-lecture tables", set(rows)), ("summary rows", set(summary))):
        if have != set(expected):
            out.append(f"methods: {name} missing {sorted(set(expected) - have)}, "
                       f"extra {sorted(have - set(expected))}")
    return out


def check_lectures(rows: Mapping[str, Sequence[Mapping]]) -> tuple[list[Key], list[str]]:
    """Check 2: every method has the same lectures, one row each, status ok. Returns the
    lectures (the first method's, in its order) and the problems."""
    out: list[str] = []
    first = next(iter(rows.values()), [])
    keys = [(r["corpus"], r["lecture_id"]) for r in first]
    for m, rs in rows.items():
        ks = [(r["corpus"], r["lecture_id"]) for r in rs]
        if len(set(ks)) != len(ks):
            out.append(f"lectures: {m} has repeated lectures")
        if set(ks) != set(keys):
            out.append(f"lectures: {m} differs from the others "
                       f"({len(set(ks) ^ set(keys))} lectures)")
        bad = [r for r in rs if r["status"] != "ok"]
        if bad:
            out.append(f"lectures: {m} has {len(bad)} runs not ok")
    return keys, out


def check_categories(keys: Sequence[Key], cats: Sequence[Mapping[str, str]],
                     order: Mapping[str, Sequence[str]]) -> tuple[dict[str, dict[Key, str]],
                                                                 list[str]]:
    """Check 3: each test lecture has exactly one categories row, and on each axis one
    of the step-6 levels. Returns axis -> lecture -> level, and the problems."""
    out: list[str] = []
    by_key: dict[Key, list[Mapping]] = {}
    for c in cats:
        by_key.setdefault((c["corpus"], c["lecture_id"]), []).append(c)
    levels: dict[str, dict[Key, str]] = {a: {} for a in AXES}
    for k in keys:
        rs = by_key.get(k, [])
        if len(rs) != 1:
            out.append(f"categories: {k[0]}/{k[1]} has {len(rs)} rows")
            continue
        for a in AXES:
            lv = rs[0].get(a, "")
            if lv not in order.get(a, ()):
                out.append(f"categories: {k[0]}/{k[1]} {a} = {lv!r} is not a step-6 level")
            else:
                levels[a][k] = lv
    return levels, out


def check_q1(q1: Sequence[Mapping], summary: Mapping[str, Mapping]) -> list[str]:
    """Check 4: Q1 against step 13's summary, the accuracy and every cost of the kind
    (total and per lecture; tokens: prompt and completion)."""
    out = []
    for r in q1:
        s = summary.get(r["method"])
        if s is None:
            continue
        if not _close(r["accuracy"], float(s["test_accuracy"])):
            out.append(f"Q1 {r['method']}: accuracy {r['accuracy']!r} != summary "
                       f"{s['test_accuracy']}")
        for c, _ in COSTS:
            for suffix in ("_total", "_per_lecture"):
                col = c + suffix
                if r[col] is None or col not in s:
                    continue
                if not _close(r[col], float(s[col])):
                    out.append(f"Q1 {r['method']}: {col} {r[col]!r} != summary {s[col]}")
    return out


def check_weighted(long: Sequence[Mapping], q1: Sequence[Mapping]) -> list[str]:
    """Check 5: per method, the n-weighted average of its level means is its overall
    mean (every lecture is in exactly one level)."""
    out = []
    overall = {r["method"]: r["accuracy"] for r in q1}
    for m, acc in overall.items():
        rs = [r for r in long if r["method"] == m]
        n = sum(r["n"] for r in rs)
        if not n or not _close(sum(r["n"] * r["accuracy"] for r in rs) / n, acc):
            out.append(f"Q2 {rs[0]['axis'] if rs else '?'} {m}: the weighted level means "
                       f"do not give the overall {acc!r}")
    return out


def check_levels(axis: str, long: Sequence[Mapping], order: Sequence[str],
                 methods: Sequence[str]) -> list[str]:
    """Check 6: every step-6 level of the axis has test lectures and every method."""
    out = []
    for lv in order:
        have = {r["method"] for r in long if r["level"] == lv}
        if not have:
            out.append(f"Q2 {axis}: level {lv!r} has no test lecture")
        elif have != set(methods):
            out.append(f"Q2 {axis}: level {lv!r} misses {sorted(set(methods) - have)}")
    return out


# --------------------------------------------------------------------------- #
# The analysis, from rows in memory
# --------------------------------------------------------------------------- #

def analyse(rows: Mapping[str, Sequence[Mapping[str, Any]]], kinds: Mapping[str, str],
            load_s: Mapping[str, float], summary: Mapping[str, Mapping],
            cats: Sequence[Mapping[str, str]], order: Mapping[str, Sequence[str]],
            expected: Sequence[str]) -> tuple[dict, list[str], list[str]]:
    """The three tables, the check lines and the problems. `rows`: per method its
    per-lecture rows (strings, as read); `order`: per axis its step-6 levels."""
    problems = check_methods(rows, summary, expected)
    if problems:
        return {}, ["1. methods: not the expected ones"], problems
    keys, p = check_lectures(rows)
    if p:
        return {}, ["2. lectures: not the same for every method"], p
    levels, p = check_categories(keys, cats, order)
    problems += p
    lines = [f"1. methods: {len(rows)} per-lecture tables, {len(summary)} summary rows, "
             f"{len(expected)} methods expected",
             f"2. lectures: {len(keys)} test lectures, the same for every method",
             f"3. categories: {len(keys)} test lectures, one level on each of {len(AXES)} axes"]
    if not rows or not keys:
        return {}, lines, problems + ["no rows"]

    accs = {m: {(r["corpus"], r["lecture_id"]): lecture_accuracy(r) for r in rs}
            for m, rs in rows.items()}
    q1 = q1_rows(rows, kinds, load_s)
    p = check_q1(q1, summary)
    problems += p
    lines.append(f"4. Q1 against summary.csv (accuracy, costs): {len(p)} differences")
    q2: dict[str, list[dict]] = {}
    n_weighted = n_levels = 0
    for a in AXES:
        q2[a] = q2_rows(a, accs, kinds, levels[a], order[a])
        p = check_weighted(q2[a], q1)
        n_weighted += len(p)
        problems += p
        p = check_levels(a, q2[a], order[a], list(rows))
        n_levels += len(p)
        problems += p
    lines.append(f"5. weighted level means = overall mean: {len(q1)} methods x {len(AXES)} "
                 f"axes, {n_weighted} differences")
    lines.append(f"6. levels with test lectures and all {len(rows)} methods: "
                 f"{sum(len(order[a]) for a in AXES)} levels, {n_levels} problems")
    q3 = q3_rows([r for a in AXES for r in q2[a]], q1)
    return {"q1": q1, "q2": q2, "q3": q3}, lines, problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    return repr(v) if isinstance(v, float) else v


def write_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _csv_value(r[c]) for c in columns})


def num(v: Optional[float]) -> str:
    """A cost for reading: '-' where the kind has none, whole numbers above 100, two
    decimals above 1, three significant digits below."""
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.2f}" if abs(v) >= 1 else f"{v:.3g}"


def _md_table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    return (["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
            + ["| " + " | ".join(str(c) for c in row) + " |" for row in body])


def q1_markdown(q1: Sequence[Mapping]) -> str:
    n = q1[0]["n_lectures"] if q1 else 0
    lines = ["# Q1: the methods by mean test accuracy, with cost", "",
             f"Mean accuracy over the {n} test lectures, in %. Rank 1 = highest; equal "
             "accuracies share a rank. Each cost: total over the lectures / per lecture; "
             "'-' where the method's kind has no such cost. compute s: the method's own "
             "time; build s: the similarity tables it reads (with the model loads); LLM "
             "calls, LLM s and tokens (prompt + completion): the answers it used "
             "(successful calls, successful attempt only). A combined method carries its "
             "members' cost.", ""]
    body = []
    for r in q1:
        def pair(c):
            return ("-" if r[f"{c}_total"] is None
                    else f"{num(r[f'{c}_total'])} / {num(r[f'{c}_per_lecture'])}")
        body.append([r["rank"], r["method"], r["kind"], f"{r['accuracy']:.2f}",
                     pair("compute_s"), pair("build_cost_s"), pair("used_calls"),
                     pair("used_seconds"), pair("used_tokens")])
    lines += _md_table(["rank", "method", "kind", "accuracy %", "compute s", "build s",
                        "LLM calls", "LLM s", "tokens"], body)
    return "\n".join(lines) + "\n"


def q2_markdown(q2: Mapping[str, Sequence[Mapping]], q1: Sequence[Mapping]) -> str:
    lines = ["# Q2: every method within each category level", "",
             "Per axis and level: each method's mean accuracy (%) over the test lectures "
             "of the level, with its rank among the methods within the level in brackets. "
             "Methods in the order of Q1 (overall rank and accuracy in the first columns). "
             "The last row is the mean over the methods, for reading only.", ""]
    for axis, long in q2.items():
        levels = list(dict.fromkeys(r["level"] for r in long))
        n = {r["level"]: r["n"] for r in long}
        cell = {(r["method"], r["level"]): r for r in long}
        lines += [f"## {axis}", "",
                  "n lectures: " + ", ".join(f"{lv} {n[lv]}" for lv in levels), ""]
        body = [[q["method"], q["rank"], f"{q['accuracy']:.2f}"]
                + [f"{cell[(q['method'], lv)]['accuracy']:.2f} "
                   f"({cell[(q['method'], lv)]['rank']})" for lv in levels] for q in q1]
        body.append(["mean over the methods", "", f"{sum(q['accuracy'] for q in q1) / len(q1):.2f}"]
                    + [f"{sum(r['accuracy'] for r in long if r['level'] == lv) / len(q1):.2f}"
                       for lv in levels])
        lines += _md_table(["method", "overall rank", "overall %"] + levels, body)
        lines.append("")
    return "\n".join(lines)


def q3_markdown(q3: Sequence[Mapping]) -> str:
    lines = ["# Q3: each method's best and worst category level", "",
             "Over all levels of all axes (corpus is not an axis here). Best level: the "
             "smallest rank among the methods within the level; worst: the largest. A tie "
             "in rank goes to the higher accuracy (best) or the lower (worst); a level list "
             "means a tie in both. Accuracy in %.", ""]
    body = [[r["method"], r["kind"], r["overall_rank"], f"{r['overall_accuracy']:.2f}",
             r["best_level"], r["best_rank"], f"{r['best_accuracy']:.2f}",
             r["worst_level"], r["worst_rank"], f"{r['worst_accuracy']:.2f}"] for r in q3]
    lines += _md_table(["method", "kind", "overall rank", "overall %", "best level", "rank",
                        "%", "worst level", "rank", "%"], body)
    return "\n".join(lines) + "\n"


def write(tables: Mapping, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "q1_ranking.csv", tables["q1"], q1_columns())
    for axis, long in tables["q2"].items():
        write_csv(out / f"q2_{axis}_long.csv", long,
                  ["axis", "level", "n", "method", "kind", "accuracy", "rank"])
        wide, cols = q2_wide(long, tables["q1"])
        write_csv(out / f"q2_{axis}_wide.csv", wide, cols)
    write_csv(out / "q3_best_worst.csv", tables["q3"], Q3_COLUMNS)
    (out / "q1_ranking.md").write_text(q1_markdown(tables["q1"]), encoding="utf-8")
    (out / "q2_levels.md").write_text(q2_markdown(tables["q2"], tables["q1"]), encoding="utf-8")
    (out / "q3_best_worst.md").write_text(q3_markdown(tables["q3"]), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reading and main
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def level_order(path: Path) -> dict[str, list[str]]:
    """Per axis its step-6 levels, in level_counts.csv's order."""
    out: dict[str, list[str]] = {}
    for r in read_csv(path):
        out.setdefault(r["axis"], []).append(r["level"])
    return out


def model_load_s(methods: Sequence[str]) -> dict[str, float]:
    """Per method the model load times of the measures it reads (with their deps),
    which step 13 adds once to the build total."""
    tuned = T.load_tuned()
    deps, load = F.read_times()
    return {m: sum(load.get(x, 0.0) for x in F.with_deps(F.method_measures(m, tuned), deps))
            for m in methods}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    rows = {p.name[:-len(".csv.gz")]: read_csv(p)
            for p in sorted((FINAL / "methods").glob("*.csv.gz"))}
    summary = {r["method"]: r for r in read_csv(FINAL / "summary.csv")}
    expected = sorted(METHODS)
    kinds = {m: F.kind(m) for m in expected}
    tables, lines, problems = analyse(
        rows, kinds, model_load_s(expected), summary,
        read_csv(CATEGORIES / "document_categories.csv"),
        level_order(CATEGORIES / "level_counts.csv"), expected)

    text = "\n".join(["Step 14: the analysis's checks (scripts/analysis.py)", ""] + lines
                     + ["", f"PROBLEMS: {len(problems)}"] + [f"   {p}" for p in problems]) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.txt").write_text(text, encoding="utf-8")
    print(text)
    if problems:
        print("no table written: the checks failed", file=sys.stderr)
        return 1
    write(tables, OUT)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
