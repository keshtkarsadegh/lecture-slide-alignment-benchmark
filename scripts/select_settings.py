#!/usr/bin/env python3
"""Step 11: choose one setting per method from the tuning runs (docs/TUNING_DESIGN.md 4).

THE RULE.
  1. A combination's score is its mean accuracy over the 60 tuning lectures, each
     lecture counting once (not weighted by units: the few long MaViLS lectures would
     outweigh the many short ones). A run that crashed, or whose LLM call never
     answered, gave no slide to any unit and counts as accuracy 0; it is reported, not
     dropped, so a combination cannot win by crashing on its hard lectures.
  2. Every combination within 0.5 percentage points of the best counts as tied (0.5
     points is about one wrong unit in each of 12 of the 60 lectures).
  3. Among the tied: the CHEAPER first. For a method that makes or uses LLM calls: fewer
     successful calls over the tuning set, then the smaller sum of their durations, both
     from "the cost of the answers used" (a cached answer brings its stored cost, so the
     order in which the runs were made does not matter). For the others only a setting
     that changes the amount of work counts (mbr k, wbs2 n_intervals, multiscale_dtw
     radius, jump_dtw max_jump): the smaller value is cheaper; measured times are not
     compared, they vary from node to node.
     Then the one CLOSEST TO THE DEFAULTS: the fewest tuned settings away from their
     defaults (the measure counts, its default is hybrid; the model counts, its default
     is config/llm.toml's). Then the HIGHER mean accuracy. Last, so that the choice is
     always the same, the earlier combination of the grid.
  Only a method whose whole grid has a row for every tuning lecture is chosen; the
  others are reported as incomplete.

hierarchical is also chosen once per model (`per_model`), by the same rule over that
model's combinations: fusion's curve for a model uses those settings.

OUTPUT.
  config/tuned_settings.toml        per method: the chosen combination (measure, model,
                                    cfg, settings); read by scripts/tune.py for the
                                    consensus stage
  data/tuning/selection_report.csv  per method: the choice, the tuning accuracy, the
                                    accuracy at the defaults, the gain, the number of
                                    tied combinations, the mean cost per lecture of the
                                    choice, and EDGE flags

EDGE: a tuned numeric setting whose chosen value is the smallest or the largest value
tried (the best may lie beyond the grid). Categorical settings (measure, model, drop_fill,
mode, smooth, use_boundary_refinement, mbr's temperature none / 1.0) are never flagged.
A two-valued numeric setting is always at one end, so it is always flagged; the flag
says which end. The grids are not widened here.

    python scripts/select_settings.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from methods import METHODS  # noqa: E402
from tune import (GRID_PATH, NONE, OUT, SELECTION, TUNED_PATH, WORK_SETTINGS,  # noqa: E402
                  MethodGrid, default_model, load_grid, read_runs, set_lectures)

#: Combinations within this many accuracy points (as a share) of the best are tied.
TIE = 0.005
REPORT = OUT / "selection_report.csv"


@dataclass
class ComboStats:
    key: str
    index: int
    values: dict[tuple[str, str], Any]
    n_lectures: int
    mean_acc: float
    not_ok: int
    used_calls: int
    used_seconds: float
    used_prompt_tokens: int
    used_completion_tokens: int
    compute_s: float


def combo_values(key: str) -> dict[tuple[str, str], Any]:
    c = json.loads(key)
    out: dict[tuple[str, str], Any] = {("measure", "measure"): c["measure"]}
    if c.get("model") is not None:
        out[("model", "model")] = c["model"]
    out.update({("cfg", k): v for k, v in c["cfg"].items()})
    out.update({("settings", k): v for k, v in c["settings"].items()})
    return out


def combo_stats(rows: Sequence[Mapping[str, Any]], lectures: set[tuple[str, str]]
                ) -> dict[str, ComboStats]:
    """Per combination: mean accuracy over its lectures (a run that is not ok counts
    0), runs not ok, and the summed cost. A row of a lecture outside the tuning set
    stops the selection."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if (r["corpus"], r["lecture_id"]) not in lectures:
            raise SystemExit(f"{r['method']}: a run on {r['corpus']}/{r['lecture_id']}, "
                             f"which is not a tuning lecture")
        by.setdefault(r["combo"], []).append(r)
    out = {}
    for key, rs in by.items():
        accs = [(r["n_correct"] / r["n_units"]) if r["status"] == "ok" and r["n_units"] else 0.0
                for r in rs]
        out[key] = ComboStats(
            key, int(rs[0]["combo_index"]), combo_values(key),
            len({(r["corpus"], r["lecture_id"]) for r in rs}), sum(accs) / len(accs),
            sum(r["status"] != "ok" for r in rs),
            int(sum(r["used_calls"] for r in rs)), float(sum(r["used_seconds"] for r in rs)),
            int(sum(r["used_prompt_tokens"] for r in rs)),
            int(sum(r["used_completion_tokens"] for r in rs)),
            float(sum(r["compute_s"] for r in rs)))
    return out


def tuned_axes(grid: MethodGrid) -> list[tuple[str, str, tuple]]:
    """The axes with more than one value: the settings this method actually tunes."""
    return [a for a in grid.axes() if len(a[2]) > 1]


def cost_key(s: ComboStats, grid: MethodGrid) -> tuple:
    """Smaller = cheaper (docs/TUNING_DESIGN.md 4.4.1)."""
    llm = (s.used_calls, s.used_seconds) if METHODS[grid.name].uses_llm else (0, 0.0)
    work = WORK_SETTINGS.get(grid.name)
    return llm + (s.values.get(("settings", work), 0) if work else 0,)


def distance(s: ComboStats, grid: MethodGrid, defaults: Mapping[tuple[str, str], Any]) -> int:
    """How many tuned settings are away from their defaults."""
    return sum(s.values[(g, k)] != defaults[(g, k)] for g, k, _ in tuned_axes(grid))


def choose(stats: Sequence[ComboStats], grid: MethodGrid, dm: str
           ) -> tuple[ComboStats, list[ComboStats]]:
    """The chosen combination and the tied ones (the chosen among them)."""
    best = max(s.mean_acc for s in stats)
    tied = [s for s in stats if s.mean_acc >= best - TIE - 1e-12]
    d = grid.defaults(dm)
    tied.sort(key=lambda s: (cost_key(s, grid), distance(s, grid, d), -s.mean_acc, s.index))
    return tied[0], tied


def _numeric(values: Sequence[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def edge_flags(s: ComboStats, grid: MethodGrid) -> list[str]:
    out = []
    for g, k, values in tuned_axes(grid):
        if g not in ("cfg", "settings") or not _numeric(values):
            continue
        v = s.values[(g, k)]
        if v == min(values):
            out.append(f"{k}={v} (smallest)")
        elif v == max(values):
            out.append(f"{k}={v} (largest)")
    return out


def describe(s: ComboStats, grid: MethodGrid) -> str:
    """The chosen values of the tuned cfg and own settings, "name=value; ..."."""
    return "; ".join(f"{k}={s.values[(g, k)]}" for g, k, _ in tuned_axes(grid)
                     if g in ("cfg", "settings")) or "-"


def complete(grid: MethodGrid, stats: Mapping[str, ComboStats], n_lectures: int) -> bool:
    return all(c.key in stats and stats[c.key].n_lectures == n_lectures
               for c in grid.combos())


@dataclass
class Choice:
    method: str
    chosen: ComboStats
    tied: list[ComboStats]
    default: Optional[ComboStats]
    per_model: dict[str, ComboStats]


def select(grids: Mapping[str, MethodGrid], rows_of: Callable[[str], Sequence[dict]],
           lectures: set[tuple[str, str]], dm: str
           ) -> tuple[dict[str, Choice], list[dict]]:
    """The choice per complete method, and the report rows of every method."""
    choices: dict[str, Choice] = {}
    report: list[dict] = []
    for name, grid in grids.items():
        rows = rows_of(name)
        stats = combo_stats(rows, lectures)
        base = {"method": name, "stage": grid.stage, "n_combos": grid.n_combos()}
        if not complete(grid, stats, len(lectures)):
            n_done = sum(st.n_lectures for st in stats.values())
            report.append({**base, "status": f"incomplete ({n_done} of "
                                              f"{grid.n_combos() * len(lectures)} runs)"})
            continue
        grid_stats = [stats[c.key] for c in grid.combos()]
        chosen, tied = choose(grid_stats, grid, dm)
        d = grid.defaults(dm)
        default = next((st for st in grid_stats
                        if all(st.values[k] == v for k, v in d.items())), None)
        per_model = {}
        if name == "hierarchical":
            for mdl in grid.models:
                per_model[mdl] = choose([st for st in grid_stats
                                         if st.values[("model", "model")] == mdl], grid, dm)[0]
        choices[name] = Choice(name, chosen, tied, default, per_model)
        n = len(lectures)
        report.append({
            **base, "status": "chosen",
            "measure": chosen.values[("measure", "measure")],
            "model": chosen.values.get(("model", "model"), ""),
            "settings": describe(chosen, grid),
            "tuning_acc": round(100 * chosen.mean_acc, 3),
            "default_acc": "" if default is None else round(100 * default.mean_acc, 3),
            "gain": "" if default is None else round(100 * (chosen.mean_acc - default.mean_acc), 3),
            "n_tied": len(tied),
            "mean_compute_s": chosen.compute_s / n,
            "mean_used_calls": chosen.used_calls / n,
            "mean_used_seconds": chosen.used_seconds / n,
            "mean_used_prompt_tokens": chosen.used_prompt_tokens / n,
            "mean_used_completion_tokens": chosen.used_completion_tokens / n,
            "not_ok_runs": chosen.not_ok,
            "not_ok_runs_grid": sum(st.not_ok for st in grid_stats),
            "edge": "EDGE: " + "; ".join(edge_flags(chosen, grid)) if edge_flags(chosen, grid) else "",
            "combo": chosen.key})
    return choices, report


REPORT_COLUMNS = ("method", "stage", "status", "measure", "model", "settings", "tuning_acc",
                  "default_acc", "gain", "n_combos", "n_tied", "mean_compute_s",
                  "mean_used_calls", "mean_used_seconds", "mean_used_prompt_tokens",
                  "mean_used_completion_tokens", "not_ok_runs", "not_ok_runs_grid", "edge",
                  "combo")


def _toml_key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def _toml_value(v: Any) -> str:
    if v is None:
        return json.dumps(NONE)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def tuned_toml(choices: Mapping[str, Choice]) -> str:
    """config/tuned_settings.toml: the chosen combination of each method."""
    lines = ["# The tuned settings of step 11, written by scripts/select_settings.py from",
             "# data/tuning/runs (the rule: docs/TUNING_DESIGN.md 4). Do not edit by hand.",
             "# Per method the chosen combination of config/tuning_grid.toml: measure, model",
             "# (LLM methods), [cfg] (DPConfig fields; the others at their defaults) and",
             "# [settings] (the method's own; the others at their defaults). \"none\" = None.",
             "# hierarchical also has its best settings per model ([per_model]), which",
             "# fusion's curve uses.", ""]
    for name, ch in choices.items():
        v = ch.chosen.values
        lines.append(f"[{name}]")
        lines.append(f"measure = {_toml_value(v[('measure', 'measure')])}")
        if ("model", "model") in v:
            lines.append(f"model = {_toml_value(v[('model', 'model')])}")
        lines.append(f"tuning_accuracy = {ch.chosen.mean_acc!r}")
        for group in ("cfg", "settings"):
            items = [(k, x) for (g, k), x in v.items() if g == group]
            if items:
                lines.append(f"[{name}.{group}]")
                lines += [f"{_toml_key(k)} = {_toml_value(x)}" for k, x in items]
        for mdl, st in ch.per_model.items():
            lines.append(f"[{name}.per_model.{_toml_key(mdl)}]")
            lines += [f"{_toml_key(k)} = {_toml_value(x)}"
                      for (g, k), x in st.values.items() if g == "settings"]
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="choose one setting per method (step 11)")
    ap.add_argument("--runs", type=Path, default=OUT / "runs")
    ap.add_argument("--grid", type=Path, default=GRID_PATH)
    ap.add_argument("--tuned", type=Path, default=TUNED_PATH)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    grids = load_grid(args.grid)
    lectures = set(set_lectures("tuning", SELECTION))
    choices, report = select(grids, lambda m: read_runs(m, args.runs), lectures,
                             default_model())
    args.tuned.write_text(tuned_toml(choices), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(report)
    for r in report:
        if r["status"] != "chosen":
            print(f"{r['method']:26s} {r['status']}")
        else:
            print(f"{r['method']:26s} {r['measure']:13s} {r['model'] or '':28s} "
                  f"acc {r['tuning_acc']:6.2f} (default {r['default_acc']}, gain {r['gain']}) "
                  f"tied {r['n_tied']}  {r['settings']}  {r['edge']}")
    print(f"wrote {args.tuned} ({len(choices)} methods) and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
