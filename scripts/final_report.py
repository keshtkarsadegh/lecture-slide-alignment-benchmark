#!/usr/bin/env python3
"""Step 13: the final run's tables and checks, from data/final/runs (scripts/tune.py).

The final run is every method's one tuned combination (config/tuned_settings.toml) on
the 140 test lectures, made by scripts/tune.py with `--set test --settings ...` on the
code path of tuning (docs/FINAL_RUN.md). This script only reads its merged rows; it
runs no method and pays no LLM call.

OUTPUT (data/final/):

  methods/<method>.csv.gz  one row per test lecture, in selection order:
      lecture_id, corpus, accuracy (the share of the lecture's units on their true
      slide; the metric of tuning), n_correct, n_units, status, compute_s (the method's
      own time, as in tuning), build_cost_s (below), and for an LLM or mixed method:
      used_calls, used_seconds, used_prompt_tokens, used_completion_tokens (the cost of
      the answers the method used, cached ones included), paid_calls, paid_seconds,
      paid_prompt_tokens, paid_completion_tokens (the calls paid in this run),
      cache_hits, failed_attempts and failed_calls (information, never cost),
      truncated (answers cut at max_tokens), parse_failed
  summary.csv              one row per method: kind, measures, mean accuracy per lecture
      (in %; a run not ok counts 0, as in tuning), its tuning accuracy and the
      difference, and every cost as a total and per lecture
  summary_table.txt        the summary as a readable table, best first
  answers_flags.csv.gz     one row per LLM answer of the methods that ask their own
      questions: its finish reason and the copying flags (below)
  checks.txt               the checks and their results

BUILD COST. A method's build cost on a lecture is the step-12 time (data/similarity/
test/tables.sqlite3, `cost_s` per table) of every measure its output depends on, with
the measures those are built from (data/similarity/test/times.csv `deps`), each counted
once: hybrid_image counts hybrid_image, embed, bm25 and image. The measures of a method
are its tuned measure, and for a combined method also its members' (fusion: its own and
hierarchical's), so a combined method carries its members' build cost. hierarchical
reads no matrix and has none. llm_single_call, llm_monotone and
llm_monotone_constrained count hybrid: they read it for the fallback of an unusable
answer. The totals add each measure's model load time (`load_s`, once per build) to the
per-lecture sum, as times.csv's cost_with_deps_s does.

COMBINED METHODS. Their compute_s already holds the members' compute (the members run
inside the method), and their used_* the members' answers (ensemble: llm_single_call's;
fusion: hierarchical's; served from the cache of this run's LLM stage). The calls paid
in this run (paid_*) are counted where they were paid, so the sum of paid_calls over all
methods is the run's total with nothing counted twice.

THE CHECKS.
  1. every method has exactly the 140 test lectures, one row each, status ok, no NaN
     accuracy, and ran the tuned combination (the key of the run chosen in tuning,
     data/tuning/selection_report.csv);
  2. no failed call (failed attempts reported, never cost);
  3. per LLM and mixed method: the parse failures and the truncated answers;
  4. copying: every answer of llm_single_call, llm_monotone, llm_monotone_constrained,
     hierarchical and llm_rerank_kbest is read back from the LLM cache (offline: the
     methods are replayed with a cache-only client, which must reproduce each recorded
     accuracy) and flagged when it reproduces an old numeric example (starts 0, 12, 30;
     gaps 12, 18; a first block (0, 11, 0)) or echoes a letter-shaped answer (s1, g0,
     "from": a, <int>, a "..." inside a list); ensemble and fusion are replayed too,
     their answers are llm_single_call's and hierarchical's;
  5. test accuracy against tuning accuracy: the methods more than 5 points apart.

    python scripts/final_report.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import re
import sqlite3
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tune as T  # noqa: E402
from methods import MEMBERS, METHODS  # noqa: E402
from methods.llm import extract_json  # noqa: E402
from methods.llm_client import load_settings  # noqa: E402

FINAL = T.OUT_DIRS[T.TEST_SET]
RUNS = FINAL / "runs"
TIMES = REPO_ROOT / "data" / "similarity" / T.TEST_SET / "times.csv"
SELECTION_REPORT = T.OUT / "selection_report.csv"

#: An LLM answer combined with a matrix method's alignment (tune.py's "mixed").
MIXED = ("ensemble", "fusion", "llm_rerank_kbest")
#: The methods that ask their own questions: their answers are the copying check's.
OWN_CALLS = ("llm_single_call", "llm_monotone", "llm_monotone_constrained", "hierarchical",
             "llm_rerank_kbest")
#: Test minus tuning accuracy, in points, beyond which a method is listed.
DIFF_POINTS = 5.0

BASE_COLUMNS = ("lecture_id", "corpus", "accuracy", "n_correct", "n_units", "status",
                "compute_s", "build_cost_s")
#: (output column, runner column): the paid group renamed so it cannot be read as used.
LLM_COLUMNS = (("used_calls", "used_calls"), ("used_seconds", "used_seconds"),
               ("used_prompt_tokens", "used_prompt_tokens"),
               ("used_completion_tokens", "used_completion_tokens"),
               ("paid_calls", "calls"), ("paid_seconds", "call_seconds"),
               ("paid_prompt_tokens", "prompt_tokens"),
               ("paid_completion_tokens", "completion_tokens"),
               ("cache_hits", "cache_hits"), ("failed_attempts", "failed_attempts"),
               ("failed_calls", "failed_calls"), ("truncated", "truncated"),
               ("parse_failed", "parse_failed"))


def kind(method: str) -> str:
    if method in MIXED:
        return "mixed"
    return "LLM" if METHODS[method].uses_llm else "non-LLM"


# --------------------------------------------------------------------------- #
# Build cost
# --------------------------------------------------------------------------- #

def method_measures(method: str, tuned: Mapping[str, T.Tuned]) -> set[str]:
    """The measures whose tables the method's output depends on: its tuned measure and
    its members' (fusion: hierarchical's); "none" and "members" read no table."""
    out = {tuned[method].measure} - {T.NO_MEASURE, T.MEMBERS_MEASURE}
    for m in MEMBERS.get(method, ()) + (("hierarchical",) if method == "fusion" else ()):
        out |= method_measures(m, tuned)
    return out


def with_deps(measures: Iterable[str], deps: Mapping[str, Sequence[str]]) -> set[str]:
    """The measures and, recursively, those they are built from; each once."""
    out: set[str] = set()
    todo = list(measures)
    while todo:
        m = todo.pop()
        if m not in out:
            out.add(m)
            todo += list(deps.get(m, ()))
    return out


def read_times(path: Path = TIMES) -> tuple[dict[str, list[str]], dict[str, float]]:
    """times.csv: each measure's direct deps and its model load time."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return ({r["measure"]: r["deps"].split() for r in rows},
            {r["measure"]: float(r["load_s"]) for r in rows})


def read_table_costs(tables: Path) -> dict[tuple[str, str, str], float]:
    """(measure, corpus, lecture_id) -> the table's cost_s (step 12), read-only."""
    conn = sqlite3.connect(f"file:{tables}?mode=ro&immutable=1", uri=True)
    try:
        return {(m, c, l): float(s) for m, c, l, s in
                conn.execute("SELECT measure, corpus, lecture_id, cost_s FROM tables")}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# The per-lecture tables and the summary
# --------------------------------------------------------------------------- #

def lecture_rows(rows: Sequence[Mapping[str, Any]], measures: set[str],
                 costs: Mapping[tuple[str, str, str], float],
                 order: Mapping[tuple[str, str], int]) -> list[dict]:
    """A method's per-lecture rows in selection order: the output columns (the LLM ones
    for every method; non-LLM methods are written without them) and the error."""
    out = []
    for r in sorted(rows, key=lambda r: order.get((r["corpus"], r["lecture_id"]), 1 << 30)):
        row = {c: r[c] for c in BASE_COLUMNS if c != "build_cost_s"}
        row["build_cost_s"] = sum(costs[(m, r["corpus"], r["lecture_id"])] for m in measures)
        row.update({c: r[src] or 0 for c, src in LLM_COLUMNS})
        row["error"] = r.get("error", "")
        out.append(row)
    return out


def output_columns(method: str) -> tuple[str, ...]:
    """The per-lecture file's columns: the LLM ones for an LLM or mixed method only."""
    return BASE_COLUMNS + (tuple(c for c, _ in LLM_COLUMNS) if kind(method) != "non-LLM"
                           else ())


def lecture_accuracy(r: Mapping[str, Any]) -> float:
    """As the selector scores tuning: a run not ok counts 0."""
    return r["n_correct"] / r["n_units"] if r["status"] == "ok" and r["n_units"] else 0.0


def summary_row(method: str, stage: str, measures: set[str], rows: Sequence[Mapping],
                load_s: float, tuning_accuracy: Optional[float]) -> dict:
    """One method's summary: mean accuracy per lecture in %, and each cost in total
    and per lecture (build cost: the lectures' sum plus the model loads)."""
    n = len(rows)

    def tot(c):
        return float(sum(r[c] or 0 for r in rows))

    acc = 100 * sum(lecture_accuracy(r) for r in rows) / n if n else float("nan")
    tun = 100 * tuning_accuracy if tuning_accuracy is not None else float("nan")
    out = {"method": method, "kind": kind(method), "stage": stage,
           "measures": " ".join(sorted(measures)), "n_lectures": n,
           "n_ok": sum(r["status"] == "ok" for r in rows),
           "test_accuracy": acc, "tuning_accuracy": tun, "diff_points": acc - tun,
           "compute_s_total": tot("compute_s"),
           "compute_s_per_lecture": tot("compute_s") / n if n else 0.0,
           "build_cost_s_total": tot("build_cost_s") + load_s,
           "build_cost_s_per_lecture": (tot("build_cost_s") + load_s) / n if n else 0.0}
    for c, _ in LLM_COLUMNS:
        out[f"{c}_total"] = tot(c) if kind(method) != "non-LLM" else 0.0
    for c in ("used_calls", "used_seconds", "used_prompt_tokens", "used_completion_tokens"):
        out[f"{c}_per_lecture"] = out[f"{c}_total"] / n if n else 0.0
    return out


def _fmt(v: Any) -> Any:
    return repr(v) if isinstance(v, float) else v


def write_csv_gz(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    """A gzipped CSV with a fixed time stamp (the same rows give the same bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".gz.part")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r[c]) for c in columns})
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r[c]) for c in columns})


def summary_table(summary: Sequence[Mapping]) -> str:
    """The summary, best test accuracy first, as fixed-width text."""
    head = (f"{'method':26s} {'kind':8s} {'test %':>7s} {'tuning %':>8s} {'diff':>6s} "
            f"{'compute s':>10s} {'build s':>9s} {'LLM calls':>9s} {'LLM s':>8s} "
            f"{'tokens (prompt+compl.)':>22s} {'paid calls':>10s}")
    lines = [head, "-" * len(head)]
    for s in sorted(summary, key=lambda s: -s["test_accuracy"]):
        tokens = s["used_prompt_tokens_total"] + s["used_completion_tokens_total"]
        lines.append(f"{s['method']:26s} {s['kind']:8s} {s['test_accuracy']:7.2f} "
                     f"{s['tuning_accuracy']:8.2f} {s['diff_points']:+6.2f} "
                     f"{s['compute_s_total']:10.2f} {s['build_cost_s_total']:9.1f} "
                     f"{s['used_calls_total']:9.0f} {s['used_seconds_total']:8.1f} "
                     f"{tokens:22.0f} {s['paid_calls_total']:10.0f}")
    lines.append("")
    lines.append("LLM calls, LLM s, tokens: the answers each method used (cached ones "
                 "included: a combined method carries its members'); paid calls: paid in "
                 "this run, each once. compute s: the methods' own time; build s: the "
                 "step-12 tables of the measures used, with their deps and model loads.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_complete(method: str, rows: Sequence[Mapping], lectures: Sequence[tuple[str, str]],
                   want_combo: str, chosen_combo: Optional[str] = None) -> list[str]:
    """Problems of one method's rows: a test lecture missing or repeated, a lecture not
    in the test set, a run not ok, a NaN accuracy, a combination other than the tuned
    one (or other than the one chosen in tuning, when its key is known)."""
    out = []
    keys = [(r["corpus"], r["lecture_id"]) for r in rows]
    missing = sorted(set(lectures) - set(keys))
    extra = sorted(set(keys) - set(lectures))
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if missing:
        out.append(f"{method}: {len(missing)} test lectures missing, e.g. {missing[:3]}")
    if extra:
        out.append(f"{method}: {len(extra)} rows outside the test set, e.g. {extra[:3]}")
    if dup:
        out.append(f"{method}: repeated lectures {dup[:3]}")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        out.append(f"{method}: {len(bad)} runs not ok, e.g. {bad[0]['status']}: "
                   f"{bad[0]['error']}")
    nan = [r for r in rows if not isinstance(r["accuracy"], float)
           or math.isnan(r["accuracy"])]
    if nan:
        out.append(f"{method}: {len(nan)} rows without an accuracy")
    combos = {r["combo"] for r in rows}
    if combos - {want_combo}:
        out.append(f"{method}: ran {sorted(combos - {want_combo})}, not the tuned {want_combo}")
    if chosen_combo is not None and want_combo != chosen_combo:
        out.append(f"{method}: the tuned combination {want_combo} is not the one chosen in "
                   f"tuning {chosen_combo}")
    return out


def _ints(xs) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for x in xs if isinstance(xs, list) else []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def old_example(method: str, content: str) -> bool:
    """The answer begins with the numbers of the first prompts' answer shapes."""
    obj = extract_json(content)
    if not isinstance(obj, dict):
        return False
    if method == "llm_monotone":
        return _ints(obj.get("starts"))[:3] == [0, 12, 30]
    if method == "llm_monotone_constrained":
        return _ints(obj.get("gaps"))[:2] == [12, 18]
    if method == "llm_single_call":
        blocks = [b for b in obj.get("blocks") or [] if isinstance(b, dict)]
        return bool(blocks) and _ints([blocks[0].get(k) for k in ("from", "to", "slide")]) \
            == [0, 11, 0]
    return False


#: A letter-shaped answer echoed back: the placeholders of the answer shapes in the
#: prompts (a, b, j, s1, g0, <int>, "prev" | "next") or their "..." inside a list.
ECHO = re.compile(r'"(?:from|to|slide|choice)"\s*:\s*[a-z]\b'      # "from": a
                  r'|(?<![\w"])[sg]\d+(?![\w"])'                    # s1, g0
                  r'|<int>|"prev"\s*\|'                              # hierarchical
                  r'|\[[^\]]*\.\.\.')                                # [0, 3, ...]


def echoed(content: str) -> bool:
    return ECHO.search(content or "") is not None


class RecordingLLM(T.TimedLLM):
    """A `TimedLLM` that keeps every answer record it returns (the copying check)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.answers: list[dict] = []

    def ask(self, *args, **kw) -> dict:
        rec = super().ask(*args, **kw)
        self.answers.append(rec)
        return rec


def replay(methods: Sequence[str], lectures: Sequence[T.Lecture], mats,
           grids: Mapping[str, T.MethodGrid], tuned: Mapping[str, T.Tuned],
           recorded: Mapping[tuple[str, str, str], Mapping], llms: Mapping[str, RecordingLLM],
           log: Callable = print) -> tuple[list[dict], list[str]]:
    """Run each method's tuned combination again from the cache only (`llms` have no
    client: an answer not cached makes the run crash) and return the answers with
    their flags, and the problems: a run that is not ok, that paid a call, or whose
    accuracy differs from the recorded one."""
    answers, problems = [], []
    for method in methods:
        combo = grids[method].combos()[0]
        for lec in lectures:
            for llm in llms.values():
                llm.answers.clear()
            row = T.run_one(combo, lec, mats, llms=llms, tuned=tuned)
            rec = recorded.get((method, lec.corpus, lec.lecture_id))
            if row["status"] != "ok" or row["calls"]:
                problems.append(f"replay {method} {lec.name}: {row['status']} "
                                f"{row['error']} (paid calls {row['calls']})")
            elif rec is None or (row["n_correct"], row["used_calls"]) \
                    != (rec["n_correct"], rec["used_calls"]):
                problems.append(f"replay {method} {lec.name}: {row['n_correct']} correct, "
                                f"{row['used_calls']} answers used; recorded "
                                f"{None if rec is None else (rec['n_correct'], rec['used_calls'])}")
            if method not in OWN_CALLS:
                continue
            for i, a in enumerate(a for llm in llms.values() for a in llm.answers):
                content = str(a.get("content") or "")
                answers.append({"method": method, "corpus": lec.corpus,
                                "lecture_id": lec.lecture_id, "call": i,
                                "finish_reason": a.get("finish_reason"),
                                "old_example": old_example(method, content),
                                "echo": echoed(content),
                                "start": content[:80].replace("\n", " ")})
        log(f"replay {method}: done")
    return answers, problems


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def tuning_accuracies(path: Path = T.TUNED_PATH) -> dict[str, float]:
    with open(path, "rb") as fh:
        return {m: t.get("tuning_accuracy") for m, t in tomllib.load(fh).items()}


def chosen_combos(path: Path = SELECTION_REPORT) -> dict[str, str]:
    """The combination chosen in tuning per method (the selection report's `combo`)."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["method"]: r["combo"] for r in csv.DictReader(fh) if r.get("combo")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", type=Path, default=T.TUNED_PATH)
    args = ap.parse_args(argv)
    settings = args.settings if args.settings.is_absolute() else REPO_ROOT / args.settings
    tuned = T.load_tuned(settings)
    grids = T.tuned_grids(tuned, T.load_grid())
    tuning_acc = tuning_accuracies(settings)
    chosen = chosen_combos()
    deps, load_s = read_times()
    tables = T.tables_path(T.TEST_SET)
    costs = read_table_costs(tables)
    lectures = T.set_lectures(T.TEST_SET, tuned_only=True)
    order = {k: i for i, k in enumerate(lectures)}

    problems: list[str] = []
    summary: list[dict] = []
    all_rows: dict[tuple[str, str, str], dict] = {}
    for method, g in grids.items():
        rows = T.read_runs(method, RUNS)
        for r in rows:
            all_rows[(method, r["corpus"], r["lecture_id"])] = r
        problems += check_complete(method, rows, lectures, g.combos()[0].key,
                                   chosen.get(method))
        measures = with_deps(method_measures(method, tuned), deps)
        out_rows = lecture_rows(rows, measures, costs, order)
        write_csv_gz(FINAL / "methods" / f"{method}.csv.gz", out_rows, output_columns(method))
        summary.append(summary_row(method, g.stage, method_measures(method, tuned), out_rows,
                                   sum(load_s.get(m, 0.0) for m in measures),
                                   tuning_acc.get(method)))
    write_csv(FINAL / "summary.csv", summary, list(summary[0]))
    (FINAL / "summary_table.txt").write_text(summary_table(summary), encoding="utf-8")

    # 4. the copying check: the answers, read back from the cache
    lecs = T.load_lectures(T.TEST_SET, tuned_only=True)
    mats = T.load_matrices(lecs, tables=tables)
    s = load_settings()
    cache = T.MemoCache(s.cache_path)
    llms = {m: RecordingLLM(model=m, client=None, cache=cache, temperature=s.temperature,
                            disable_thinking=s.disable_thinking)
            for m in grids["llm_single_call"].models}
    replayed = [m for m in grids if kind(m) != "non-LLM"]
    answers, replay_problems = replay(replayed, lecs, mats, grids, tuned, all_rows, llms)
    problems += replay_problems
    write_csv_gz(FINAL / "answers_flags.csv.gz", answers,
                 ("method", "corpus", "lecture_id", "call", "finish_reason", "old_example",
                  "echo", "start"))

    lines = ["Step 13: the final run's checks (scripts/final_report.py)", ""]
    lines.append(f"1. completeness: {len(grids)} methods x {len(lectures)} test lectures; "
                 f"{sum(s['n_lectures'] for s in summary)} rows, "
                 f"{sum(s['n_ok'] for s in summary)} ok")
    failed = {s["method"]: (s["failed_calls_total"], s["failed_attempts_total"])
              for s in summary if s["kind"] != "non-LLM"}
    lines.append(f"2. failed calls: {sum(f for f, _ in failed.values()):.0f}; failed "
                 f"attempts (information, never cost): "
                 + ", ".join(f"{m} {a:.0f}" for m, (_, a) in failed.items()))
    if any(f for f, _ in failed.values()):
        problems.append("failed calls: " + ", ".join(f"{m} {f:.0f}"
                                                     for m, (f, _) in failed.items() if f))
    lines.append("3. parse failures / truncated answers per LLM and mixed method:")
    for s in summary:
        if s["kind"] != "non-LLM":
            lines.append(f"   {s['method']:26s} {s['kind']:6s} parse_failed "
                         f"{s['parse_failed_total']:4.0f}  truncated {s['truncated_total']:4.0f}"
                         f"  (answers used {s['used_calls_total']:.0f})")
    lines.append("4. copying (answers of " + ", ".join(OWN_CALLS) + "):")
    for m in OWN_CALLS:
        a = [x for x in answers if x["method"] == m]
        old = [f"{x['corpus']}/{x['lecture_id']}" for x in a if x["old_example"]]
        echo = [f"{x['corpus']}/{x['lecture_id']}" for x in a if x["echo"]]
        lines.append(f"   {m:26s} answers {len(a):5d}  old example {len(old)}  "
                     f"letter echo {len(echo)}" + (f"  {sorted(set(old + echo))[:10]}"
                                                   if old or echo else ""))
        if old or echo:
            problems.append(f"copying: {m}: old example {len(old)}, letter echo {len(echo)}")
    lines.append(f"   replay from the cache: {len(replayed)} methods x {len(lecs)} lectures, "
                 f"{len(replay_problems)} problems")
    lines.append(f"5. test vs tuning accuracy, more than {DIFF_POINTS:g} points apart:")
    far = [s for s in summary if abs(s["diff_points"]) > DIFF_POINTS]
    for s in sorted(far, key=lambda s: s["diff_points"]):
        lines.append(f"   {s['method']:26s} test {s['test_accuracy']:6.2f}  tuning "
                     f"{s['tuning_accuracy']:6.2f}  diff {s['diff_points']:+6.2f}")
    if not far:
        lines.append("   none")
    paid = {k: sum(s[f"{k}_total"] for s in summary)
            for k in ("paid_calls", "paid_seconds", "paid_prompt_tokens",
                      "paid_completion_tokens")}
    lines.append("")
    lines.append(f"LLM calls paid in this run: {paid['paid_calls']:.0f} calls, "
                 f"{paid['paid_seconds']:.1f} s (successful attempts), "
                 f"{paid['paid_prompt_tokens']:.0f} prompt + "
                 f"{paid['paid_completion_tokens']:.0f} completion tokens")
    lines.append("")
    lines.append(f"PROBLEMS: {len(problems)}")
    lines += [f"   {p}" for p in problems]
    text = "\n".join(lines) + "\n"
    (FINAL / "checks.txt").write_text(text, encoding="utf-8")
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
