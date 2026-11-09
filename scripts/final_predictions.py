#!/usr/bin/env python3
"""Step 16 (a): the final run's predicted slide for every unit, by an offline replay.

Step 13 kept one row per (method, lecture): its accuracy and cost, not the slide each
unit was given. The importance-weighted scoring of step 16 (scripts/analysis_q4.py)
needs the slide per unit, so this script runs the final run again, offline, and keeps
it. It is the final run's own code path, not a copy of it: every method's one tuned
combination (config/tuned_settings.toml, cut down by `tune.tuned_grids`) on the 140
test lectures and their step-12 tables, through `tune.run_one`, which now hands back
the predicted labels it scores (`keep_pred`). Nothing is chosen or tuned here.

CACHE ONLY. The LLM and mixed methods get their answers from data/llm_cache.sqlite3
alone: the LLMs are made with `tune.make_llms(offline=True)`, which builds no client
and never reads the API key, so an answer not in the cache raises (LLMUnavailable) and
the run stops at the first one. A run that is not ok, or that paid a call, also stops
it. The cost of a method stays step 13's; nothing here is timed for cost.

OUTPUT (data/final/predictions/):

  <method>.csv.gz   corpus, lecture_id, unit_index (0..n-1, as in data/prepared and
                    data/importance/labels.csv.gz), true_slide, pred_slide; slides are
                    1-based as in data/prepared; pred_slide -1 = no slide (always
                    wrong); selection order, then unit order; gzip without a time stamp
  checks.txt        the check below and its result

THE CHECK. From the files as written, each (method, lecture)'s units on their true slide
and its unit count are recounted and compared with step 13's per-lecture table
(data/final/methods/<method>.csv.gz): they must be exactly equal for all 31 x 140, and
the replay must have paid no call. Any difference is listed and the exit code is 1.

    python scripts/final_predictions.py [--settings config/tuned_settings.toml]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tune as T  # noqa: E402

FINAL = T.OUT_DIRS[T.TEST_SET]
OUT = FINAL / "predictions"
COLUMNS = ("corpus", "lecture_id", "unit_index", "true_slide", "pred_slide")

Key = tuple[str, str]  # (corpus, lecture_id)


class ReplayStopped(SystemExit):
    """The replay cannot be the final run's: a run not ok (an answer not cached is
    one) or a paid call. It stops everything; nothing is improvised around it."""


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #

def offline_llms(grids: Mapping[str, T.MethodGrid], tuned: Mapping[str, T.Tuned]
                 ) -> dict[str, T.TimedLLM]:
    """Cache-only LLMs for every model the tuned methods name. `make_llms(offline=True)`
    builds no client, so no API key is read and no call can be made."""
    models = sorted({m for g in grids.values() for m in g.models}
                    | {t.model for t in tuned.values() if t.model})
    llms = T.make_llms(models, offline=True)
    if any(llm.client is not None for llm in llms.values()):
        raise ReplayStopped("an offline LLM has a client: refusing to run")
    return llms


def predict(method: str, grid: T.MethodGrid, lec: T.Lecture, mats,
            llms: Mapping[str, T.LLM], tuned: Mapping[str, T.Tuned]) -> dict:
    """The method's tuned combination on one lecture, as `tune.run` runs it, with the
    predicted labels kept. Stops the replay on a run not ok or a paid call."""
    combos = grid.combos()
    if len(combos) != 1:
        raise ReplayStopped(f"{method}: {len(combos)} combinations; the tuned settings "
                            f"must leave exactly one")
    row = T.run_one(combos[0], lec, mats, llms=llms, tuned=tuned, keep_pred=True)
    if row["status"] != "ok" or row["pred"] is None:
        raise ReplayStopped(f"replay {method} {lec.name}: {row['status']}: {row['error']}")
    if row["calls"]:
        raise ReplayStopped(f"replay {method} {lec.name}: paid {row['calls']} calls")
    if len(row["pred"]) != lec.n_units:
        raise ReplayStopped(f"replay {method} {lec.name}: {len(row['pred'])} labels for "
                            f"{lec.n_units} units")
    return row


def unit_rows(lec: T.Lecture, pred: Sequence[int]) -> list[tuple]:
    """One row per unit, slides 1-based; a unit given no slide (-1) is written -1."""
    return [(lec.corpus, lec.lecture_id, i, int(g) + 1, int(p) + 1 if p >= 0 else -1)
            for i, (g, p) in enumerate(zip(lec.gold, pred))]


def write_predictions(path: Path, rows: Sequence[tuple]) -> None:
    """gzip with mtime 0: the same predictions give the same bytes; written next to the
    old file and moved over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".gz.part")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                                mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(tmp, path)


def read_predictions(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# The check against step 13
# --------------------------------------------------------------------------- #

def recount(rows: Sequence[Mapping[str, str]]) -> dict[Key, tuple[int, int]]:
    """(units on their true slide, units) per lecture, from the written rows."""
    out: dict[Key, list[int]] = {}
    for r in rows:
        c = out.setdefault((r["corpus"], r["lecture_id"]), [0, 0])
        c[0] += int(r["pred_slide"]) == int(r["true_slide"])
        c[1] += 1
    return {k: (a, b) for k, (a, b) in out.items()}


def compare(method: str, counts: Mapping[Key, tuple[int, int]],
            step13: Sequence[Mapping[str, str]]) -> list[str]:
    """Exact equality of (n_correct, n_units) per lecture with step 13's table, and the
    same lectures on both sides."""
    out = []
    rec = {(r["corpus"], r["lecture_id"]): (int(r["n_correct"]), int(r["n_units"]))
           for r in step13}
    for k in sorted(set(rec) | set(counts)):
        if counts.get(k) != rec.get(k):
            out.append(f"{method} {k[0]}/{k[1]}: predictions {counts.get(k)} "
                       f"(n_correct, n_units), step 13 {rec.get(k)}")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None, log: Callable = print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", type=Path, default=T.TUNED_PATH)
    ap.add_argument("--methods", nargs="*", help="default: all 31")
    args = ap.parse_args(argv)
    settings = args.settings if args.settings.is_absolute() else REPO_ROOT / args.settings
    tuned = T.load_tuned(settings)
    grids = T.tuned_grids(tuned, T.load_grid())
    methods = args.methods or list(grids)
    unknown = sorted(set(methods) - set(grids))
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")
    lecs = T.load_lectures(T.TEST_SET, tuned_only=True)
    mats = T.load_matrices(lecs, tables=T.tables_path(T.TEST_SET))
    llms = offline_llms(grids, tuned)
    log(f"{len(methods)} methods x {len(lecs)} test lectures, "
        f"{sum(L.n_units for L in lecs)} units; LLM answers from the cache only")

    problems: list[str] = []
    lines = ["Step 16 (a): the final run's per-unit predictions (scripts/final_predictions.py)",
             ""]
    for method in methods:
        t0 = time.perf_counter()
        rows, hits, paid_m = [], 0, 0
        for lec in lecs:
            r = predict(method, grids[method], lec, mats, llms, tuned)
            hits += r["cache_hits"]
            paid_m += r["calls"]
            rows += unit_rows(lec, r["pred"])
        path = OUT / f"{method}.csv.gz"
        write_predictions(path, rows)
        with gzip.open(FINAL / "methods" / f"{method}.csv.gz", "rt", encoding="utf-8",
                       newline="") as fh:
            rec = list(csv.DictReader(fh))
        diff = compare(method, recount(read_predictions(path)), rec)
        problems += diff
        lines.append(f"   {method:26s} {len(rec):3d} lectures  {len(rows):6d} units  "
                     f"answers from the cache {hits:5d}  paid {paid_m}  differences {len(diff)}")
        log(f"{method}: {len(rows)} units, {hits} cached answers, {len(diff)} differences "
            f"({time.perf_counter() - t0:.1f} s)")
    paid = sum(l.stats()["calls"] for l in llms.values())
    lines[2:2] = [f"Recounted from the written files against data/final/methods/<method>.csv.gz "
                  f"(n_correct and n_units per lecture, exact): {len(methods)} methods x "
                  f"{len(lecs)} lectures, {sum(L.n_units for L in lecs)} units per method.",
                  f"LLM calls paid by the replay: {paid} (cache only, no client).", ""]
    if paid:
        problems.append(f"the replay paid {paid} calls")
    lines += ["", f"PROBLEMS: {len(problems)}"] + [f"   {p}" for p in problems]
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.txt").write_text(text, encoding="utf-8")
    log(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
