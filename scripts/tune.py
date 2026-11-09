#!/usr/bin/env python3
"""Step 11: the tuning run. Every combination of a method's grid on each tuning lecture.

The grids are config/tuning_grid.toml, exactly as docs/TUNING_DESIGN.md (sections 1-2;
the counts of its section 5.1 are checked by tests/test_tuning.py). For every (method,
combination, lecture) one row is recorded:

  accuracy    the share of the lecture's units whose predicted slide equals the true
              slide (`gold_labels`; -1 is always wrong). A run that raised is kept as a
              row with status `crash` (or `llm_failed`: a call that never answered) and
              no correct unit; it is never left out, so a crash cannot hide.
  compute_s   the time of `run_method` (time.perf_counter) on matrices already in
              memory, less the time spent inside `LLM.ask`: building a matrix is the
              measure's cost (step 10), and an LLM call's cost is counted apart, below.
              For a mixed method (ensemble, fusion, llm_rerank_kbest) this is the
              compute time of its matrix part.
  LLM cost    from the step-9 counters of methods/llm_client.py, taken as the
              difference of `LLM.stats()` around the run, in two groups that are never
              mixed: "paid in this run" (calls, call_seconds, prompt_tokens,
              completion_tokens) and "cost of the answers used" (used_calls,
              used_seconds, used_prompt_tokens, used_completion_tokens; a cached answer
              brings its stored duration and tokens). The failures (failed_attempts,
              failed_calls, errors) are stored apart and are never cost.

ONLY THE TUNING SET. Lectures come from data/selection/selected_lectures.csv rows with
set == tuning; asking for another set, or naming a lecture outside it, stops the run.
The prepared corpus files also hold the test lectures: their lines are skipped by the
lecture id at the start of the line and never parsed. Matrices come only from the tuning
set's tables (data/similarity/tuning/tables.sqlite3, opened read-only).

THE FINAL RUN (step 13) is the one exception, and it goes through this same code so that
a method behaves on the test lectures exactly as it did in tuning: `--set test --settings
config/tuned_settings.toml` turns every method's grid into its ONE tuned combination
(`tuned_grids`; the same combination key as the run chosen in tuning) and runs it on the
140 test lectures, with the test set's tables (data/similarity/test/tables.sqlite3), into
data/final/ instead of data/tuning/ (the same shards, merge and CSV per method). The
test set is refused without `--settings`, so no grid is ever run on it and nothing is
chosen from it. The stages and their order are those of tuning; the consensus methods'
members run at the same tuned settings. scripts/final_report.py turns data/final/runs
into the per-lecture tables, the summary and the checks.

STORAGE (the project space is short of inodes; never one file per run). A writer puts
its rows into one SQLite file of its own, data/tuning/shards/<name>.sqlite3 (one writer
per file: an array task, or the LLM process). `merge` folds the shards into one gzipped
CSV per method, data/tuning/runs/<method>.csv.gz, and removes them.

RESUMABLE AND IDEMPOTENT. A run is identified by (method, combination, lecture), the
combination by its canonical JSON (measure, model, cfg, settings). Before running, a
writer reads the runs already done (status ok) from the merged CSV and from its own
shard and skips them; a run that crashed or whose call failed is tried again. Every
method is deterministic (wbs2's seed is fixed), and the combinations and lectures are
always visited in the same order, so a rerun gives the same rows except for the times.

THE ORDER (docs/TUNING_DESIGN.md 4.5):
  (a) stage "matrix": the 22 non-LLM base methods and mbr; array jobs (`plan`, then
      `run --plan`), in parallel with (b), since they touch no LLM cache;
  (b) stage "llm": llm_single_call, llm_monotone, llm_monotone_constrained,
      hierarchical (refinement on before off), in ONE process (`run --stage llm`):
      the LLM cache is written by one process only and one shared client keeps the
      whole run under the request rate of config/llm.toml;
  (c) stage "consensus", after `merge` and scripts/select_settings.py have chosen
      the members' settings (config/tuned_settings.toml): score_fusion, ensemble,
      llm_rerank_kbest and fusion, also in one process (`run --stage consensus`). Their
      members run at their tuned settings on their own tuned measures; fusion uses
      hierarchical's tuned settings. All LLM methods use one model (decided
      2026-10-12: qwen3-30b-a3b-instruct-2507), a fixed value in the grid.

Commands (from the repository root; docs/TUNING.md has the whole sequence):
    python scripts/tune.py trial                       # 2 lectures, a few combinations
    python scripts/tune.py plan --name matrix --tasks 20
    python scripts/tune.py run --plan matrix --task 0  # one array task
    python scripts/tune.py run --stage llm             # the LLM process
    python scripts/tune.py merge
    python scripts/tune.py run --stage consensus
    python scripts/tune.py status
    # the final run: the same commands with  --set test --settings config/tuned_settings.toml
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import itertools
import json
import math
import os
import re
import socket
import sqlite3
import sys
import time
import tomllib
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods import MEMBERS, METHODS, DPConfig, Member, run_method  # noqa: E402
from methods.llm_client import (LLM, LLMCallFailed, OpenAICompatClient,  # noqa: E402
                                ResponseCache, api_key, load_settings)

GRID_PATH = REPO_ROOT / "config" / "tuning_grid.toml"
TUNED_PATH = REPO_ROOT / "config" / "tuned_settings.toml"
SELECTION = REPO_ROOT / "data" / "selection" / "selected_lectures.csv"
PREPARED = REPO_ROOT / "data" / "prepared"
TABLES = REPO_ROOT / "data" / "similarity" / "tuning" / "tables.sqlite3"
OUT = REPO_ROOT / "data" / "tuning"

#: The only lecture set this step may read.
TUNING_SET = "tuning"
#: The set of the final run (step 13): read only at the tuned settings.
TEST_SET = "test"
SETS = (TUNING_SET, TEST_SET)
#: Where each set's runs go: the tuning run's and the final run's.
OUT_DIRS = {TUNING_SET: OUT, TEST_SET: REPO_ROOT / "data" / "final"}
#: The 8 measures (docs/TUNING_DESIGN.md 1.1) and the default one.
MEASURES = ("bm25", "embed", "splade", "cross_encoder", "image", "hybrid", "hybrid_splade",
            "hybrid_image")
DEFAULT_MEASURE = "hybrid"
#: Grid words for a method without a measure of its own: "none" reads no matrix,
#: "members" lets each member read its own tuned one. Both get the default table, for
#: its shape only (and, for ensemble's LLM voter, as llm_single_call's fixed fallback).
NO_MEASURE, MEMBERS_MEASURE = "none", "members"
#: TOML has no null; this string in a settings list stands for None.
NONE = "none"
STAGES = ("matrix", "llm", "consensus")
#: The non-LLM settings that change the amount of work (docs/TUNING_DESIGN.md 4.4):
#: the smaller value is the cheaper one.
WORK_SETTINGS = {"mbr": "k", "wbs2": "n_intervals", "multiscale_dtw": "radius",
                 "jump_dtw": "max_jump"}
CFG_FIELDS = tuple(f.name for f in fields(DPConfig))


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

def from_toml(value: Any) -> Any:
    return None if value == NONE else value


def combo_key(measure: str, model: Optional[str], cfg: Mapping[str, Any],
              settings: Mapping[str, Any]) -> str:
    """A combination's identity: canonical JSON, the same for the same values."""
    return json.dumps({"measure": measure, "model": model, "cfg": dict(cfg),
                       "settings": dict(settings)}, sort_keys=True)


@dataclass(frozen=True)
class Combo:
    method: str
    index: int
    measure: str
    model: Optional[str]
    cfg: tuple[tuple[str, Any], ...]
    settings: tuple[tuple[str, Any], ...]

    @property
    def key(self) -> str:
        return combo_key(self.measure, self.model, dict(self.cfg), dict(self.settings))


@dataclass(frozen=True)
class MethodGrid:
    """One method's values to try: model, measure, DPConfig fields, own settings."""
    name: str
    stage: str
    measures: tuple[str, ...]
    models: tuple[str, ...] = ()
    cfg: Mapping[str, tuple] = field(default_factory=dict)
    settings: Mapping[str, tuple] = field(default_factory=dict)

    def axes(self) -> list[tuple[str, str, tuple]]:
        """(group, name, values) in the order of the product: model, measure, cfg,
        settings."""
        out: list[tuple[str, str, tuple]] = []
        if self.models:
            out.append(("model", "model", tuple(self.models)))
        out.append(("measure", "measure", tuple(self.measures)))
        out += [("cfg", k, tuple(v)) for k, v in self.cfg.items()]
        out += [("settings", k, tuple(v)) for k, v in self.settings.items()]
        return out

    def n_combos(self) -> int:
        return math.prod(len(v) for _, _, v in self.axes())

    def combos(self) -> list[Combo]:
        axes = self.axes()
        out = []
        for i, values in enumerate(itertools.product(*(v for _, _, v in axes))):
            pick = {(g, k): v for (g, k, _), v in zip(axes, values)}
            out.append(Combo(
                self.name, i, pick[("measure", "measure")], pick.get(("model", "model")),
                tuple((k, v) for (g, k), v in pick.items() if g == "cfg"),
                tuple((k, v) for (g, k), v in pick.items() if g == "settings")))
        return out

    def defaults(self, default_model: str) -> dict[tuple[str, str], Any]:
        """The default of every axis: DPConfig's and the method's own defaults, the
        hybrid measure, the configured model."""
        own = METHODS[self.name].settings
        base = DPConfig()
        out: dict[tuple[str, str], Any] = {}
        for g, k, values in self.axes():
            if g == "model":
                out[(g, k)] = default_model
            elif g == "measure":
                out[(g, k)] = DEFAULT_MEASURE if len(values) > 1 else values[0]
            elif g == "cfg":
                out[(g, k)] = getattr(base, k)
            else:
                out[(g, k)] = own[k]
        return out


def load_grid(path: Path = GRID_PATH) -> dict[str, MethodGrid]:
    """The grids, checked: every method of the registry, only settings it has, and the
    default of every setting among the values tried (so "the accuracy at the defaults"
    is always one of the runs)."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    missing = sorted(set(METHODS) - set(raw))
    unknown = sorted(set(raw) - set(METHODS))
    if missing or unknown:
        raise ValueError(f"{path}: methods missing {missing}, unknown {unknown}")
    grids: dict[str, MethodGrid] = {}
    for name in raw:
        spec = raw[name]
        m = METHODS[name]
        extra = sorted(set(spec) - {"stage", "measure", "model", "cfg", "settings"})
        if extra:
            raise ValueError(f"{name}: unknown grid keys {extra}")
        stage = spec.get("stage")
        if stage not in STAGES:
            raise ValueError(f"{name}: stage must be one of {STAGES}, got {stage!r}")
        measures = tuple(spec.get("measure", ()))
        if not measures or any(x not in MEASURES + (NO_MEASURE, MEMBERS_MEASURE)
                               for x in measures):
            raise ValueError(f"{name}: bad measures {measures}")
        models = tuple(spec.get("model", ()))
        if models and not m.uses_llm:
            raise ValueError(f"{name} makes no LLM call; it has no model")
        cfg = {k: tuple(v) for k, v in spec.get("cfg", {}).items()}
        if cfg and not m.uses_cfg:
            raise ValueError(f"{name} reads no DPConfig")
        bad = sorted(set(cfg) - set(CFG_FIELDS))
        if bad:
            raise ValueError(f"{name}: {bad} are not DPConfig fields")
        settings = {k: tuple(from_toml(x) for x in v)
                    for k, v in spec.get("settings", {}).items()}
        bad = sorted(set(settings) - set(m.settings))
        if bad:
            raise ValueError(f"{name} has no setting(s) {bad}")
        g = MethodGrid(name, stage, measures, models, cfg, settings)
        for (grp, k), d in g.defaults(models[0] if models else "").items():
            values = dict(((a, b), v) for a, b, v in g.axes())[(grp, k)]
            if not values or len(set(map(repr, values))) != len(values):
                raise ValueError(f"{name}: {k} has an empty or repeated value list")
            if grp in ("cfg", "settings") and d not in values:
                raise ValueError(f"{name}: the default {k} = {d!r} is not among {values}")
        grids[name] = g
    return grids


def default_model() -> str:
    return load_settings().model


# --------------------------------------------------------------------------- #
# The tuning lectures (and only those)
# --------------------------------------------------------------------------- #

@dataclass
class Lecture:
    pos: int
    corpus: str
    lecture_id: str
    gold: np.ndarray
    slide_texts: list[str]
    unit_texts: list[str]

    @property
    def name(self) -> str:
        return f"{self.corpus}/{self.lecture_id}"

    @property
    def n_units(self) -> int:
        return len(self.gold)

    @property
    def n_slides(self) -> int:
        return len(self.slide_texts)


def tables_path(set_name: str) -> Path:
    """A set's similarity tables (steps 10 and 12)."""
    return REPO_ROOT / "data" / "similarity" / set_name / "tables.sqlite3"


def set_lectures(set_name: str = TUNING_SET, selection: Path = SELECTION, *,
                 tuned_only: bool = False) -> list[tuple[str, str]]:
    """(corpus, lecture_id) of the set in the selection file's order. The tuning set
    always; the test set only for a run at the tuned settings (`tuned_only`, step 13),
    so that no grid is ever run on it; any other set is refused."""
    if not (set_name == TUNING_SET or (set_name == TEST_SET and tuned_only)):
        raise SystemExit(f"step 11 tunes on the {TUNING_SET!r} set only, and the "
                         f"{TEST_SET!r} set is run only at the tuned settings "
                         f"(--settings); refusing {set_name!r}")
    with open(selection, newline="", encoding="utf-8") as fh:
        out = [(r["corpus"], r["lecture_id"]) for r in csv.DictReader(fh)
               if r["set"] == set_name]
    if not out:
        raise SystemExit(f"{selection}: no lecture in set {set_name!r}")
    return out


_LECTURE_ID = re.compile(r'"lecture_id":\s*"((?:[^"\\]|\\.)*)"')


def load_lectures(set_name: str = TUNING_SET, *, only: Optional[Sequence[str]] = None,
                  selection: Path = SELECTION, prepared: Path = PREPARED,
                  tuned_only: bool = False) -> list[Lecture]:
    """The set's lectures (or the named ones, "corpus/lecture_id", which must be in the
    set), in selection order, with gold labels and texts. The set is the tuning set,
    or the test set for a run at the tuned settings (`set_lectures`).

    A prepared corpus file holds every lecture of the corpus, of both sets. A line is
    parsed only when the lecture id at its start is a wanted lecture of the set, so no
    other lecture's content is ever read into the run (in tuning: no test lecture's).
    """
    wanted = set_lectures(set_name, selection, tuned_only=tuned_only)
    pos = {w: i for i, w in enumerate(wanted)}
    if only is not None:
        names = {f"{c}/{l}": (c, l) for c, l in wanted}
        outside = sorted(set(only) - set(names))
        if outside:
            raise SystemExit(f"not {set_name} lectures (refused): {outside[:5]}")
        wanted = [names[n] for n in only]
    keep = set(wanted)
    out: dict[tuple[str, str], Lecture] = {}
    for corpus in sorted({c for c, _ in keep}):
        with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                hit = _LECTURE_ID.search(line[:1000])
                if hit is None or (corpus, json.loads(f'"{hit.group(1)}"')) not in keep:
                    continue
                r = json.loads(line)
                key = (r["corpus"], r["lecture_id"])
                if key not in keep:            # the id at the start was not this record's
                    raise SystemExit(f"{corpus}: record {key} read for a different id")
                slides = sorted(r["slides"], key=lambda s: s["index"])
                units = sorted(r["units"], key=lambda u: u["index"])
                out[key] = Lecture(pos[key], *key,
                                   np.asarray([int(u["true_slide"]) - 1 for u in units]),
                                   [s["text"] for s in slides], [u["text"] for u in units])
    missing = sorted(keep - set(out))
    if missing:
        raise SystemExit(f"not in {prepared}: {missing[:5]} ({len(missing)} lectures)")
    return sorted(out.values(), key=lambda L: L.pos)


def load_matrices(lectures: Sequence[Lecture], measures: Sequence[str] = MEASURES,
                  tables: Path = TABLES) -> dict[tuple[str, str], np.ndarray]:
    """(measure, lecture name) -> the (units x slides) table of step 10 (step 12 for
    the test set), read-only.

    `immutable=1`: the file is only read, from many nodes at once, so SQLite takes no
    lock on it."""
    import zlib
    if not tables.exists():
        raise SystemExit(f"{tables} is missing: build the set's tables first (step 10 "
                         f"or 12)")
    conn = sqlite3.connect(f"file:{tables}?mode=ro&immutable=1", uri=True)
    out: dict[tuple[str, str], np.ndarray] = {}
    try:
        for lec in lectures:
            for measure in measures:
                row = conn.execute("SELECT n, m, data FROM tables WHERE measure=? AND "
                                   "corpus=? AND lecture_id=?",
                                   (measure, lec.corpus, lec.lecture_id)).fetchone()
                if row is None:
                    raise SystemExit(f"{tables}: no {measure} table for {lec.name}")
                n, m, data = row
                if (n, m) != (lec.n_units, lec.n_slides):
                    raise SystemExit(f"{tables}: {measure} {lec.name} is {n} x {m}, the "
                                     f"lecture {lec.n_units} x {lec.n_slides}")
                out[(measure, lec.name)] = np.frombuffer(
                    zlib.decompress(data), dtype=np.float64).reshape(n, m).copy()
    finally:
        conn.close()
    return out


def accuracy(pred: Sequence[int], gold: Sequence[int]) -> tuple[int, int]:
    """(units whose predicted slide is the true slide, units). -1 (no slide) is always
    wrong; the true slides are 0-based columns, never -1."""
    p = np.asarray(pred, dtype=np.int64)
    g = np.asarray(gold, dtype=np.int64)
    if p.shape != g.shape:
        raise ValueError(f"{p.size} predicted labels for {g.size} units")
    return int(((p == g) & (p >= 0)).sum()), int(g.size)


# --------------------------------------------------------------------------- #
# The LLM side
# --------------------------------------------------------------------------- #

class TimedLLM(LLM):
    """An `LLM` that also adds up the wall time spent inside `ask` (cache lookups,
    calls, throttle), so that the runner can take it out of the compute time: an LLM
    call's cost is its own counters, never the wall clock around it."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ask_seconds = 0.0

    def ask(self, *args, **kw) -> dict:
        t0 = time.perf_counter()
        try:
            return super().ask(*args, **kw)
        finally:
            self.ask_seconds += time.perf_counter() - t0


class MemoCache(ResponseCache):
    """The LLM cache with an in-process memo in front. fusion re-walks hierarchical's
    tree for every one of its 648 combinations; the answers are the same each time,
    so the second walk need not go back to SQLite. Nothing about the stored records or
    the cost counting changes."""

    def __init__(self, path):
        super().__init__(path)
        self._memo: dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        if key in self._memo:
            return self._memo[key]
        rec = super().get(key)
        if rec is not None:
            self._memo[key] = rec
        return rec

    def put(self, key: str, method: str, record: dict) -> None:
        super().put(key, method, record)
        self._memo[key] = record


def make_llms(models: Iterable[str], *, offline: bool = False) -> dict[str, TimedLLM]:
    """One `TimedLLM` per model, all on ONE client (one throttle, so the whole process
    stays under the configured request rate) and one cache connection."""
    s = load_settings()
    client = None if offline else OpenAICompatClient(s, api_key())
    cache = MemoCache(s.cache_path)
    return {m: TimedLLM(model=m, client=client, cache=cache, temperature=s.temperature,
                        disable_thinking=s.disable_thinking) for m in models}


# --------------------------------------------------------------------------- #
# The tuned settings (written by scripts/select_settings.py)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Tuned:
    measure: str
    model: Optional[str]
    cfg: Mapping[str, Any]
    settings: Mapping[str, Any]
    #: hierarchical only: its best settings under each model (fusion's curve)
    per_model: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def load_tuned(path: Path = TUNED_PATH) -> dict[str, Tuned]:
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return {name: Tuned(t["measure"], t.get("model"), dict(t.get("cfg", {})),
                        {k: from_toml(v) for k, v in t.get("settings", {}).items()},
                        {mdl: {k: from_toml(v) for k, v in s.items()}
                         for mdl, s in t.get("per_model", {}).items()})
            for name, t in raw.items()}


def tuned_grids(tuned: Mapping[str, Tuned], grids: Mapping[str, MethodGrid]
                ) -> dict[str, MethodGrid]:
    """Every method's grid cut down to its one tuned combination (step 13's final run).

    The stage is the tuning grid's, so the final run has the same stages, order and
    processes as tuning. The tuned settings must name a value for exactly the axes of
    the grid, each among the values tried: then the combination has the same key as the
    run chosen in tuning, and `run_one` runs it exactly as it did then."""
    missing = sorted(set(grids) - set(tuned))
    if missing:
        raise SystemExit(f"no tuned setting for {missing}: run scripts/select_settings.py")
    out: dict[str, MethodGrid] = {}
    for name, g in grids.items():
        t = tuned[name]
        if set(t.cfg) != set(g.cfg) or set(t.settings) != set(g.settings):
            raise SystemExit(f"{name}: the tuned settings {sorted(t.cfg) + sorted(t.settings)} "
                             f"are not the grid's axes {sorted(g.cfg) + sorted(g.settings)}")
        model = (t.model,) if g.models else ()
        values = [("measure", t.measure, g.measures)] + \
            ([("model", t.model, g.models)] if g.models else []) + \
            [(k, t.cfg[k], g.cfg[k]) for k in g.cfg] + \
            [(k, t.settings[k], g.settings[k]) for k in g.settings]
        for k, v, tried in values:
            if v not in tried:
                raise SystemExit(f"{name}: the tuned {k} = {v!r} is not among {tried}")
        out[name] = MethodGrid(name, g.stage, (t.measure,), model,
                               {k: (t.cfg[k],) for k in g.cfg},
                               {k: (t.settings[k],) for k in g.settings})
    return out


def member_inputs(method: str, lec: Lecture, mats, tuned: Mapping[str, Tuned],
                  llms: Mapping[str, LLM]) -> dict[str, Member]:
    """Each member at its tuned settings on the matrix of its own tuned measure
    (docs/TUNING_DESIGN.md 2.6); the LLM voter with the LLM of its tuned model."""
    out: dict[str, Member] = {}
    for name in MEMBERS[method]:
        if name not in tuned:
            raise SystemExit(f"{method} needs {name}'s tuned settings: tune {name}, then "
                             f"run scripts/select_settings.py")
        t = tuned[name]
        S = mats[(t.measure, lec.name)]
        if METHODS[name].uses_llm:
            out[name] = Member(sim=S, settings=dict(t.settings), llm=llms[t.model])
        else:
            out[name] = Member(sim=S, cfg=DPConfig(**t.cfg), settings=dict(t.settings))
    return out


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #

#: The paid / used / failure counters of `LLM.stats()` kept per run.
LLM_FIELDS = ("calls", "call_seconds", "prompt_tokens", "completion_tokens",
              "used_calls", "used_seconds", "used_prompt_tokens", "used_completion_tokens",
              "used_untimed", "cache_hits", "truncated", "failed_attempts", "failed_calls")
COLUMNS = (("method", "TEXT"), ("combo", "TEXT"), ("combo_index", "INTEGER"),
           ("corpus", "TEXT"), ("lecture_id", "TEXT"), ("n_units", "INTEGER"),
           ("n_correct", "INTEGER"), ("accuracy", "REAL"), ("status", "TEXT"),
           ("error", "TEXT"), ("run_s", "REAL"), ("compute_s", "REAL")) \
    + tuple((f, "REAL" if f.endswith("seconds") else "INTEGER") for f in LLM_FIELDS) \
    + (("errors", "TEXT"), ("parse_failed", "INTEGER"), ("host", "TEXT"),
       ("finished", "TEXT"))
COLUMN_NAMES = tuple(c for c, _ in COLUMNS)
KEY = ("method", "combo", "corpus", "lecture_id")


def _llm_delta(llms: Mapping[str, LLM], before: Mapping[str, dict]) -> dict:
    out: dict[str, Any] = {f: 0 for f in LLM_FIELDS}
    errors: dict[str, int] = {}
    for m, llm in llms.items():
        after = llm.stats()
        for f in LLM_FIELDS:
            out[f] += after[f] - before[m][f]
        for kind, c in after["errors"].items():
            d = c - before[m]["errors"].get(kind, 0)
            if d:
                errors[kind] = errors.get(kind, 0) + d
    out["errors"] = json.dumps(errors, sort_keys=True)
    return out


def run_one(combo: Combo, lec: Lecture, mats, *, llms: Optional[Mapping[str, LLM]] = None,
            tuned: Optional[Mapping[str, Tuned]] = None,
            members_at_defaults: bool = False, keep_pred: bool = False) -> dict:
    """Run one combination on one lecture and return its row (never raises for a
    method's own error: the row says `crash` or `llm_failed`).

    `keep_pred` adds the predicted slide per unit (0-based, -1 = none; None when the
    run was not ok) as `pred`, a key the shards never store: the per-unit predictions
    of step 16 (scripts/final_predictions.py) come from this very call, so they are the
    alignment that was scored, not a second implementation of it."""
    name = combo.method
    m = METHODS[name]
    llms = llms or {}
    tuned = tuned or {}
    measure = (DEFAULT_MEASURE if combo.measure in (NO_MEASURE, MEMBERS_MEASURE)
               else combo.measure)
    S = mats[(measure, lec.name)]
    settings = dict(combo.settings)
    kw: dict[str, Any] = {}
    if m.uses_cfg:
        kw["cfg"] = DPConfig(**dict(combo.cfg))
    if name in MEMBERS and not members_at_defaults:
        kw["member_inputs"] = member_inputs(name, lec, mats, tuned, llms)
    if name == "fusion" and not members_at_defaults:
        if "hierarchical" not in tuned or combo.model not in tuned["hierarchical"].per_model:
            raise SystemExit("fusion needs hierarchical's tuned settings for each model: "
                             "tune hierarchical, then run scripts/select_settings.py")
        settings.update(tuned["hierarchical"].per_model[combo.model])
    if m.uses_llm:
        if combo.model is not None:
            model = combo.model
        elif members_at_defaults or "llm_single_call" not in tuned:
            model = default_model()
        else:
            model = tuned["llm_single_call"].model       # ensemble: its LLM voter's model
        kw.update(slide_texts=lec.slide_texts, unit_texts=lec.unit_texts, llm=llms[model])
    before = {k: l.stats() for k, l in llms.items()}
    ask0 = sum(getattr(l, "ask_seconds", 0.0) for l in llms.values())
    diag: dict = {}
    status, error, labels = "ok", "", None
    t0 = time.perf_counter()
    try:
        labels = run_method(name, S, diagnostics=diag, **kw, **settings)
    except LLMCallFailed as e:
        status, error = "llm_failed", f"LLMCallFailed: {e}"[:500]
    except Exception as e:                    # a crash is recorded, never hidden
        status, error = "crash", f"{type(e).__name__}: {e}"[:500]
    run_s = time.perf_counter() - t0
    ask_s = sum(getattr(l, "ask_seconds", 0.0) for l in llms.values()) - ask0
    n_correct = accuracy(labels, lec.gold)[0] if labels is not None else 0
    pf = diag.get("parse_failed", 0)
    row = {"method": name, "combo": combo.key, "combo_index": combo.index,
           "corpus": lec.corpus, "lecture_id": lec.lecture_id, "n_units": lec.n_units,
           "n_correct": n_correct, "accuracy": n_correct / lec.n_units if lec.n_units else 0.0,
           "status": status, "error": error, "run_s": run_s, "compute_s": run_s - ask_s,
           **_llm_delta(llms, before), "parse_failed": int(pf or 0),
           "host": socket.gethostname(),
           "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if keep_pred:
        row["pred"] = None if labels is None else [int(x) for x in labels]
    return row


# --------------------------------------------------------------------------- #
# Storage: shards (one writer each) and the merged CSV per method
# --------------------------------------------------------------------------- #

class Shard:
    """One writer's rows, data/tuning/shards/<name>.sqlite3. Plain rollback journal,
    not WAL: the file lives on Lustre and WAL's shared memory is unsafe there."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        cols = ", ".join(f"{c} {t}" for c, t in COLUMNS)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS runs ({cols}, "
                          f"PRIMARY KEY ({', '.join(KEY)}))")

    def put_many(self, rows: Sequence[dict]) -> None:
        if not rows:
            return
        q = (f"INSERT OR REPLACE INTO runs ({', '.join(COLUMN_NAMES)}) VALUES "
             f"({', '.join('?' * len(COLUMN_NAMES))})")
        self.conn.execute("BEGIN")
        self.conn.executemany(q, [tuple(r[c] for c in COLUMN_NAMES) for r in rows])
        self.conn.execute("COMMIT")

    def done(self, method: str) -> set[tuple[str, str, str]]:
        return {tuple(r) for r in self.conn.execute(
            "SELECT combo, corpus, lecture_id FROM runs WHERE method=? AND status='ok'",
            (method,))}

    def close(self) -> None:
        self.conn.close()


def read_shard(path: Path, method: Optional[str] = None) -> list[dict]:
    """A shard's rows (of one method, when given), read-only."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    try:
        q = f"SELECT {', '.join(COLUMN_NAMES)} FROM runs"
        cur = conn.execute(q + " WHERE method=?", (method,)) if method else conn.execute(q)
        return [dict(zip(COLUMN_NAMES, r)) for r in cur]
    finally:
        conn.close()


def shard_methods(path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    try:
        return {r[0] for r in conn.execute("SELECT DISTINCT method FROM runs")}
    finally:
        conn.close()


_INT = {c for c, t in COLUMNS if t == "INTEGER"}
_REAL = {c for c, t in COLUMNS if t == "REAL"}


def _typed(row: Mapping[str, str]) -> dict:
    out: dict[str, Any] = {}
    for c in COLUMN_NAMES:
        v = row.get(c, "")
        out[c] = int(v) if c in _INT and v != "" else float(v) if c in _REAL and v != "" else v
    return out


def runs_path(method: str, runs_dir: Path) -> Path:
    return runs_dir / f"{method}.csv.gz"


def read_runs(method: str, runs_dir: Path = OUT / "runs") -> list[dict]:
    """The merged rows of one method (empty when there are none yet)."""
    p = runs_path(method, runs_dir)
    if not p.exists():
        return []
    with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
        return [_typed(r) for r in csv.DictReader(fh)]


def write_runs(method: str, rows: Sequence[dict], runs_dir: Path) -> Path:
    """One gzipped CSV per method, sorted (combination, lecture), with a fixed gzip
    time stamp; written next to the old file and moved over it."""
    p = runs_path(method, runs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".gz.part")
    rows = sorted(rows, key=lambda r: (int(r["combo_index"]), r["corpus"], r["lecture_id"]))
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMN_NAMES, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: (repr(r[c]) if isinstance(r[c], float) else r[c])
                        for c in COLUMN_NAMES})
    os.replace(tmp, p)
    return p


def merge(out: Path = OUT, *, pattern: str = "*", keep_shards: bool = False,
          log: Callable = print) -> dict:
    """Fold the shards named `pattern` (a glob on the shard name; all by default) into
    data/tuning/runs/<method>.csv.gz; a shard's row replaces the merged row of the same
    run (a retried run is newer). Run it only when none of those shards' writers is
    running: the folded shards are removed. Returns the rows per method after the merge."""
    shards = sorted((out / "shards").glob(f"{pattern}.sqlite3"))
    methods = sorted(set().union(*(shard_methods(sp) for sp in shards))) if shards else []
    counts = {}
    for method in methods:                  # one method at a time: 1.6 M rows in all
        rows = {tuple(r[k] for k in KEY): r for r in read_runs(method, out / "runs")}
        for sp in shards:
            for r in read_shard(sp, method):
                k = tuple(r[c] for c in KEY)
                if r["status"] == "ok" or rows.get(k, {}).get("status") != "ok":
                    rows[k] = r
        write_runs(method, list(rows.values()), out / "runs")
        counts[method] = len(rows)
        log(f"merge: {method}: {len(rows)} runs")
    if not keep_shards:
        for sp in shards:
            for extra in (sp, sp.with_name(sp.name + "-journal")):
                if extra.exists():
                    extra.unlink()
    return counts


_MERGED_DONE: dict[str, tuple[int, set]] = {}


def merged_done(method: str, runs_dir: Path) -> set[tuple[str, str, str]]:
    """The runs of the merged CSV with status ok, read once per file version (an array
    task asks for the same method once per lecture)."""
    p = runs_path(method, runs_dir)
    if not p.exists():
        return set()
    stamp = p.stat().st_mtime_ns
    hit = _MERGED_DONE.get(str(p))
    if hit is None or hit[0] != stamp:
        with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
            keys = {(r["combo"], r["corpus"], r["lecture_id"]) for r in csv.DictReader(fh)
                    if r["status"] == "ok"}
        hit = _MERGED_DONE[str(p)] = (stamp, keys)
    return hit[1]


def done_keys(method: str, shard: Shard, runs_dir: Path) -> set[tuple[str, str, str]]:
    return merged_done(method, runs_dir) | shard.done(method)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

def run(methods: Sequence[str], lectures: Sequence[Lecture], mats, shard: Shard, *,
        grids: Mapping[str, MethodGrid], runs_dir: Path = OUT / "runs",
        llms: Optional[Mapping[str, LLM]] = None, tuned: Optional[Mapping[str, Tuned]] = None,
        combos: Optional[Mapping[str, Sequence[Combo]]] = None,
        members_at_defaults: bool = False, flush_every: int = 500,
        log: Callable = print) -> dict[str, int]:
    """Every combination of each method on each lecture, combination by combination
    (so hierarchical's refinement "on" runs on every lecture before "off"), skipping
    what is done. Rows are written in batches, so a killed job loses one batch at most.
    Returns the number of runs made per method."""
    made: dict[str, int] = {}
    for name in methods:
        todo_combos = list(combos[name]) if combos and name in combos else grids[name].combos()
        done = done_keys(name, shard, runs_dir)
        todo = [(c, lec) for c in todo_combos for lec in lectures
                if (c.key, lec.corpus, lec.lecture_id) not in done]
        log(f"{name}: {len(todo)} runs to do, "
            f"{len(todo_combos) * len(lectures) - len(todo)} done")
        buf: list[dict] = []
        bad = 0
        t_start = time.perf_counter()
        for i, (c, lec) in enumerate(todo, 1):
            row = run_one(c, lec, mats, llms=llms, tuned=tuned,
                          members_at_defaults=members_at_defaults)
            if row["status"] != "ok":
                bad += 1
                log(f"  {row['status']}: {name} {lec.name} {c.key}: {row['error']}")
            buf.append(row)
            if len(buf) >= flush_every:
                shard.put_many(buf)
                buf = []
                log(f"  {name}: {i}/{len(todo)} ({time.perf_counter() - t_start:.0f} s)")
        shard.put_many(buf)
        made[name] = len(todo)
        log(f"{name}: {len(todo)} runs made ({bad} not ok) in "
            f"{time.perf_counter() - t_start:.1f} s")
    return made


def stage_methods(grids: Mapping[str, MethodGrid], stage: str) -> list[str]:
    """The methods of a stage in the grid file's order (for "llm": the order of 4.5)."""
    return [n for n, g in grids.items() if g.stage == stage]


# --------------------------------------------------------------------------- #
# Array plan for the matrix stage
# --------------------------------------------------------------------------- #

def item_seconds(method: str, lec_n: int, lec_m: int, grids, model: Optional[dict]) -> float:
    """The estimated time of all of a method's combinations on one lecture: the
    trial's fit t = a * (n * m) ** b per run, or n * m when there is no fit."""
    c = float(lec_n * lec_m)
    per_run = (model[method]["a"] * c ** model[method]["b"]) if model and method in model else c
    return grids[method].n_combos() * per_run


def make_plan(methods: Sequence[str], lectures: Sequence[Lecture], n_tasks: int, grids,
              model: Optional[dict] = None) -> list[list[tuple[str, str]]]:
    """(method, lecture) items spread over `n_tasks` tasks, the longest first, each to
    the task with the least estimated work (deterministic: ties go to the lower task)."""
    items = [(item_seconds(m, L.n_units, L.n_slides, grids, model), m, L.name)
             for m in methods for L in lectures]
    items.sort(key=lambda x: (-x[0], x[1], x[2]))
    load = [0.0] * n_tasks
    tasks: list[list[tuple[str, str]]] = [[] for _ in range(n_tasks)]
    for sec, m, name in items:
        k = min(range(n_tasks), key=lambda t: (load[t], t))
        tasks[k].append((m, name))
        load[k] += sec
    return tasks


def load_fit(out: Path = OUT) -> Optional[dict]:
    p = out / "trial" / "fit.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The trial: 2 lectures, a few combinations, every method, a handful of calls
# --------------------------------------------------------------------------- #

def trial_lectures(lectures: Sequence[Lecture]) -> list[Lecture]:
    """The median-sized lecture (by units x slides) and the largest: the two ends the
    full run's time is extrapolated between."""
    by = sorted(lectures, key=lambda L: (L.n_units * L.n_slides, L.name))
    return [by[(len(by) - 1) // 2], by[-1]]


def trial_combos(grid: MethodGrid, dm: str, *, model: Optional[str] = None) -> list[Combo]:
    """The default combination and the all-first and all-last values of the tuned
    settings (the measure kept at its default), so the trial meets the cheap and the
    expensive ends of the work settings."""
    d = grid.defaults(dm)
    pick = []
    for which in ("default", "first", "last"):
        want = {}
        for g, k, values in grid.axes():
            if g in ("measure",) or (g == "model"):
                want[(g, k)] = model if (g == "model" and model) else d[(g, k)]
            else:
                want[(g, k)] = d[(g, k)] if which == "default" else (
                    values[0] if which == "first" else values[-1])
        pick.append(want)
    out, seen = [], set()
    for c in grid.combos():
        vals = {("model", "model"): c.model, ("measure", "measure"): c.measure,
                **{("cfg", k): v for k, v in c.cfg}, **{("settings", k): v for k, v in c.settings}}
        for want in pick:
            if all(vals.get(k) == v for k, v in want.items()) and c.key not in seen:
                out.append(c)
                seen.add(c.key)
    return out


def trial(*, out: Path = OUT, llm: bool = True, log: Callable = print) -> dict:
    """Run the trial into data/tuning/trial/ (never merged into the results: its
    consensus runs use default members) and write the time fit and the estimate."""
    grids = load_grid()
    dm = default_model()
    all_lecs = load_lectures()
    lecs = trial_lectures(all_lecs)
    small, large = lecs
    log(f"trial lectures: {small.name} ({small.n_units} x {small.n_slides}), "
        f"{large.name} ({large.n_units} x {large.n_slides})")
    mats = load_matrices(lecs)
    tdir = out / "trial"
    shard = Shard(tdir / "trial.sqlite3")
    runs_dir = tdir / "runs"                     # never written: nothing is merged here
    matrix = stage_methods(grids, "matrix") + ["score_fusion"]
    combos = {m: trial_combos(grids[m], dm) for m in matrix}
    run(matrix, lecs, mats, shard, grids=grids, runs_dir=runs_dir, combos=combos,
        members_at_defaults=True, log=log)
    if llm:
        models = list(grids["llm_single_call"].models)
        llms = make_llms(models)
        single = ["llm_single_call", "llm_monotone", "llm_monotone_constrained"]
        run(single, lecs, mats, shard, grids=grids, runs_dir=runs_dir, llms=llms, log=log)
        hier = [c for c in grids["hierarchical"].combos()
                if c.model == dm and dict(c.settings)["leaf_max_pages"] == 3]
        run(["hierarchical"], [small], mats, shard, grids=grids, runs_dir=runs_dir,
            llms=llms, combos={"hierarchical": hier}, log=log)
        rr = [c for c in grids["llm_rerank_kbest"].combos() if c.model == dm]
        run(["llm_rerank_kbest"], [small], mats, shard, grids=grids, runs_dir=runs_dir,
            llms=llms, combos={"llm_rerank_kbest": rr}, members_at_defaults=True, log=log)
        run(["ensemble"], lecs, mats, shard, grids=grids, runs_dir=runs_dir, llms=llms,
            members_at_defaults=True, log=log)
        run(["fusion"], [small], mats, shard, grids=grids, runs_dir=runs_dir, llms=llms,
            combos={"fusion": trial_combos(grids["fusion"], dm, model=dm)},
            members_at_defaults=True, log=log)
    shard.close()
    return estimate(grids, all_lecs, read_shard(tdir / "trial.sqlite3"), out=out, log=log)


def estimate(grids, lectures: Sequence[Lecture], rows: Sequence[dict], *, out: Path = OUT,
             tasks: int = 20, log: Callable = print) -> dict:
    """From the trial's rows: per non-LLM method a fit t = a * (n * m) ** b of the
    mean time per run (two lectures, two points; b kept in [0, 3]), the full run's
    CPU time, and the wall time on `tasks` parallel one-CPU jobs; for the LLM methods,
    the calls' mean durations. Written to data/tuning/trial/{fit.json,estimate.csv}."""
    by: dict[str, dict[tuple[str, str], list[float]]] = {}
    size: dict[tuple[str, str], int] = {}
    for r in rows:
        k = (r["corpus"], r["lecture_id"])
        by.setdefault(r["method"], {}).setdefault(k, []).append(float(r["compute_s"]))
    sizes = {(L.corpus, L.lecture_id): L.n_units * L.n_slides for L in lectures}
    fit: dict[str, dict] = {}
    for method, per in by.items():
        pts = sorted((sizes[k], float(np.mean(v))) for k, v in per.items())
        if len(pts) >= 2 and pts[-1][0] > pts[0][0] and pts[0][1] > 0:
            b = math.log(pts[-1][1] / pts[0][1]) / math.log(pts[-1][0] / pts[0][0])
            b = min(3.0, max(0.0, b))
        else:
            b = 1.0                                   # one lecture only: linear in n * m
        a = pts[0][1] / pts[0][0] ** b
        fit[method] = {"a": a, "b": b}
    if "fusion" in fit and "dp" in fit:               # fusion ran on one lecture: dp's shape
        c0, t0 = sorted((sizes[k], float(np.mean(v))) for k, v in by["fusion"].items())[0]
        fit["fusion"] = {"b": fit["dp"]["b"], "a": t0 / c0 ** fit["dp"]["b"]}
    lines = []
    total_matrix = 0.0
    biggest = 0.0
    for method, g in grids.items():
        if method not in fit:
            continue
        per_lec = [item_seconds(method, L.n_units, L.n_slides, grids, fit) for L in lectures]
        cpu = float(sum(per_lec))
        if g.stage == "matrix":
            total_matrix += cpu
            biggest = max(biggest, max(per_lec))
        lines.append({"method": method, "stage": g.stage, "combos": g.n_combos(),
                      "a": fit[method]["a"], "b": fit[method]["b"],
                      "cpu_hours": cpu / 3600, "largest_item_hours": max(per_lec) / 3600})
    wall = max(total_matrix / tasks, biggest)
    llm_rows = [r for r in rows if r["calls"] or r["used_calls"]]
    dur: dict[str, list[float]] = {}
    for r in llm_rows:
        if r["used_calls"] and r["method"] not in ("ensemble", "fusion"):
            dur.setdefault(r["method"], []).append(r["used_seconds"] / r["used_calls"])
    tdir = out / "trial"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "fit.json", "w", encoding="utf-8") as fh:
        json.dump(fit, fh, indent=1, sort_keys=True)
    with open(tdir / "estimate.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines[0]) if lines else ["method"])
        w.writeheader()
        w.writerows(sorted(lines, key=lambda x: -x["cpu_hours"]))
    summary = {"matrix_cpu_hours": total_matrix / 3600, "tasks": tasks,
               "matrix_wall_hours": wall / 3600, "largest_item_hours": biggest / 3600,
               "mean_call_seconds": {m: float(np.mean(v)) for m, v in dur.items()},
               "paid_calls": int(sum(r["calls"] for r in rows)),
               "paid_seconds": float(sum(r["call_seconds"] for r in rows)),
               "failed_calls": int(sum(r["failed_calls"] for r in rows)),
               "failed_attempts": int(sum(r["failed_attempts"] for r in rows))}
    with open(tdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    log(json.dumps(summary, indent=1, sort_keys=True))
    return summary


# --------------------------------------------------------------------------- #
# Status and totals
# --------------------------------------------------------------------------- #

def status(out: Path = OUT, log: Callable = print, *,
           grids: Optional[Mapping[str, MethodGrid]] = None, set_name: str = TUNING_SET,
           tuned_only: bool = False) -> list[dict]:
    """Per method: runs expected (combinations x the set's lectures), merged ok / not
    ok, and the rows still in shards; and the LLM totals (paid, used, failures apart)."""
    grids = grids if grids is not None else load_grid()
    n_lec = len(set_lectures(set_name, tuned_only=tuned_only))
    shards = sorted((out / "shards").glob("*.sqlite3"))
    table = []
    tot = {f: 0.0 for f in LLM_FIELDS}
    for name, g in grids.items():
        merged = read_runs(name, out / "runs")
        in_shards = [r for sp in shards for r in read_shard(sp, name)]
        for r in merged + in_shards:
            for f in LLM_FIELDS:
                tot[f] += r[f] or 0
        table.append({"method": name, "stage": g.stage,
                      "expected": g.n_combos() * n_lec,
                      "merged_ok": sum(r["status"] == "ok" for r in merged),
                      "merged_not_ok": sum(r["status"] != "ok" for r in merged),
                      "in_shards": len(in_shards)})
    for t in table:
        log("{method:26s} {stage:9s} expected {expected:7d}  ok {merged_ok:7d}  "
            "not ok {merged_not_ok:5d}  in shards {in_shards:7d}".format(**t))
    log("LLM totals: " + json.dumps({k: round(v, 3) for k, v in tot.items()}))
    return table


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _plan_path(out: Path, name: str) -> Path:
    return out / "plans" / f"{name}.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    # --set and --settings: the defaults are the tuning run; `--set test --settings
    # config/tuned_settings.toml` is the final run (step 13), on the same code path
    which = argparse.ArgumentParser(add_help=False)
    which.add_argument("--set", default=TUNING_SET, choices=SETS,
                       help="the lecture set (default: tuning); test needs --settings")
    which.add_argument("--settings", type=Path,
                       help="tuned settings (config/tuned_settings.toml): run each "
                            "method's one tuned combination instead of its grid")
    t = sub.add_parser("trial", help="2 lectures x a few combinations, every method")
    t.add_argument("--no-llm", action="store_true")
    p = sub.add_parser("plan", parents=[which],
                       help="spread the matrix stage over array tasks")
    p.add_argument("--name", default="matrix")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--methods", nargs="*", help="default: every 'matrix' method")
    r = sub.add_parser("run", parents=[which],
                       help="run a plan's task, a stage, or named methods")
    r.add_argument("--plan")
    r.add_argument("--task", type=int)
    r.add_argument("--stage", choices=STAGES)
    r.add_argument("--methods", nargs="*")
    r.add_argument("--lectures", nargs="*", help="corpus/lecture_id (of the set only)")
    r.add_argument("--shard", help="shard name (default: the plan task or the stage)")
    mg = sub.add_parser("merge", parents=[which],
                        help="fold the shards into <out>/runs/<method>.csv.gz")
    mg.add_argument("--keep-shards", action="store_true")
    mg.add_argument("--shards", default="*", help="glob of the shard names to fold "
                    "(e.g. 'matrix_*' while the LLM process still writes its own)")
    sub.add_parser("status", parents=[which], help="runs done per method and the LLM totals")
    args = ap.parse_args(argv)
    os.chdir(REPO_ROOT)

    if args.cmd == "trial":
        trial(llm=not args.no_llm)
        return 0
    if args.set == TEST_SET and args.settings is None:
        ap.error("--set test runs the tuned settings only: give --settings "
                 "config/tuned_settings.toml")
    out = OUT_DIRS[args.set]
    if args.cmd == "merge":
        merge(out, pattern=args.shards, keep_shards=args.keep_shards)
        return 0
    tuned_only = args.settings is not None
    grids = load_grid()
    tuned = load_tuned(args.settings) if tuned_only else load_tuned()
    if tuned_only:
        grids = tuned_grids(tuned, grids)
    if args.cmd == "status":
        status(out, grids=grids, set_name=args.set, tuned_only=tuned_only)
        return 0
    tables = tables_path(args.set)
    if args.cmd == "plan":
        methods = args.methods or stage_methods(grids, "matrix")
        lectures = load_lectures(args.set, tuned_only=tuned_only)
        fit = load_fit()                    # the tuning trial's time fit, for every set
        tasks = make_plan(methods, lectures, args.tasks, grids, fit)
        sizes = {L.name: (L.n_units, L.n_slides) for L in lectures}
        est = [sum(item_seconds(m, *sizes[n], grids, fit) for m, n in tk) for tk in tasks]
        path = _plan_path(out, args.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"name": args.name, "set": args.set, "fit": fit is not None,
                       "estimated_hours": [e / 3600 for e in est],
                       "tasks": [[list(x) for x in tk] for tk in tasks]}, fh, indent=1)
        print(f"{path}: {len(tasks)} tasks, {sum(map(len, tasks))} items; estimated "
              f"hours per task max {max(est) / 3600:.2f}, sum {sum(est) / 3600:.2f}")
        return 0

    # run
    if args.plan is not None:
        if args.task is None:
            ap.error("--plan needs --task")
        with open(_plan_path(out, args.plan), encoding="utf-8") as fh:
            items = [tuple(x) for x in json.load(fh)["tasks"][args.task]]
        shard_name = args.shard or f"{args.plan}_{args.task:03d}"
        names = sorted({n for _, n in items})
        lectures = load_lectures(args.set, only=names, tuned_only=tuned_only)
        mats = load_matrices(lectures, tables=tables)
        by_name = {L.name: L for L in lectures}
        shard = Shard(out / "shards" / f"{shard_name}.sqlite3")
        for method, name in items:
            if grids[method].stage != "matrix":
                raise SystemExit(f"{method} is not a matrix-stage method")
            run([method], [by_name[name]], mats, shard, grids=grids, runs_dir=out / "runs")
        shard.close()
        return 0
    if args.stage is None and not args.methods:
        ap.error("run needs --plan/--task, --stage or --methods")
    methods = args.methods or stage_methods(grids, args.stage)
    unknown = sorted(set(methods) - set(grids))
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")
    lectures = load_lectures(args.set, only=args.lectures, tuned_only=tuned_only)
    mats = load_matrices(lectures, tables=tables)
    needs_llm = any(METHODS[m].uses_llm for m in methods)
    llms = make_llms(list(grids["llm_single_call"].models)) if needs_llm else {}
    shard = Shard(out / "shards" / f"{args.shard or args.stage or 'methods'}.sqlite3")
    run(methods, lectures, mats, shard, grids=grids, runs_dir=out / "runs", llms=llms,
        tuned=tuned)
    shard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
