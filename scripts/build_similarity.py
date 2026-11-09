#!/usr/bin/env python3
"""Step 10: build the similarity tables of the 8 measures for one lecture set, timed.

A measure's table for a lecture is the (n_units, n_slides) matrix `build_matrix` returns
(similarity/, docs/SIMILARITY.md), with the settings of config/similarity.toml as they
are. The lecture sets are data/selection/selected_lectures.csv's `tuning` (60 lectures)
and `test` (140); a run builds one set. Building a table is the measure's cost, reported
apart from the methods (docs/TUNING_DESIGN.md 1.1 and 6), so every table is stored with
the time it took.

STORAGE. One SQLite file per set, data/similarity/<set>/tables.sqlite3 (in git; the
project space is short of inodes, so never one file per lecture):

  tables    one row per (measure, lecture): the matrix as zlib-compressed float64 bytes
            (read back bit-identical), its shape, its key (below), the run that built it
            and its times
  runs      one row per (run, measure): host, hardware (CPU model and cores, GPU name),
            device, model load time, lectures built and reused, package versions
  lectures  the set's lectures in selection order, with their unit and slide counts
  meta      the set's settings: what decides the numbers of every measure, fixed at the
            first build

IDEMPOTENT. A table is stored under `matrix_key` (the scorer, every setting that decides
its numbers, the unit texts and slide texts in order, the image digests): a rerun reuses
every table whose key still matches and builds only what is missing, so a job that died
can simply be submitted again. All tables of a set must come from one set of settings:
a build whose number-deciding settings differ from the set's stops (delete the set's
folder to start again). The model scorers' matrices and the embedding vectors also go to
the shared cache of similarity/cache.py, as `build_matrix` does anyway.

WHAT IS TIMED (time.perf_counter):
  load_s     a local model's load (weights from $PROJECT_ROOT/models_lsab/ to the device),
             once per run, apart from the compute; 0 when the run built nothing
  wall_s     per lecture, the whole `build_matrix` call
  cost_s     per lecture, wall_s less what the cost rule ignores: for embed, the
             rate-limit and backoff sleeps (wait_s) and the failed attempts (failed_s);
             so embed's cost is its successful requests' durations plus the local work.
             For every other measure cost_s = wall_s.
  requests, request_s, failed_attempts, errors   embed only: the successful requests and
             their durations (the cost), the failed attempts by kind (never cost)
A blend's own time reads the tables it is built from out of the cache; the `times`
command also reports each measure with the measures it is built from added
(`cost_with_deps_s`).

THE ORDER OF THE BUILD. The model scorers' matrices and the embedding vectors are read
from the shared cache, a SQLite file in WAL mode, which must not be written from two
nodes at once; the jobs run one after another (cluster/10_*.sbatch):
  1. CPU:  bm25, embed (the endpoint), hybrid
  2. GPU:  splade, image, cross_encoder (reranks hybrid, from the cached vectors)
  3. CPU:  hybrid_splade, hybrid_image (from the cached matrices), then the checks
A run gets only the models of the measures it is asked for; anything else it needs must
already be cached (the context is offline), else it stops with SimilarityUnavailable.

DEVICE. The local models run where `--model-device` says (cuda on the GPU job). The
device is a setting of splade, cross_encoder and image and part of their keys, and of
the keys of the blends built from them, so every run of a set passes the same value
(the sbatch scripts do); it is stored in `meta`.

SANITY (`sanity`): for every measure and lecture of the set, the table is there, has the
shape (units x slides) of the prepared lecture, has no NaN or inf, and its key matches
the one recomputed from the prepared data, which checks the lecture and the order of its
units and slides. As a check that the tables are not broken or misaligned - NOT a result
- it also gives, per measure, the median rank of the true slide in each unit's row (ties
share their mean rank) and the share of units whose true slide ranks in the top 3, next
to the share a random row would give. Written to data/similarity/<set>/sanity.csv.

Commands:
    python scripts/build_similarity.py fetch
    python scripts/build_similarity.py build --set tuning --measures bm25 embed hybrid --model-device cuda
    python scripts/build_similarity.py sanity --set tuning
    python scripts/build_similarity.py times --set tuning

`fetch` downloads the pinned weights (config/similarity_weights.toml; a compute node needs
the proxy). `build` of a model measure needs torch, transformers and colpali-engine: it
runs inside the image of cluster/similarity.def. embed needs the API key (LLM_API_KEY or
the untracked config/llm_credentials.toml; read by the client only).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import socket
import sqlite3
import sys
import time
import tomllib
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.registry import gold_labels  # noqa: E402
from project_root import project_path  # noqa: E402
from similarity import models as M  # noqa: E402
from similarity.cache import SimilarityCache  # noqa: E402
from similarity.registry import (CONFIG_PATH, SCORERS, Context, build_matrix,  # noqa: E402
                                 key_settings, lecture_from_record, load_settings,
                                 matrix_key, reads_images)

SELECTION = REPO_ROOT / "data" / "selection" / "selected_lectures.csv"
PREPARED = REPO_ROOT / "data" / "prepared"
OUT_ROOT = REPO_ROOT / "data" / "similarity"
WEIGHTS_CONFIG = REPO_ROOT / "config" / "similarity_weights.toml"
PIP_FREEZE = Path("/opt/pip-freeze.txt")     # inside the image of cluster/similarity.def

SETS = ("tuning", "test")
#: All 8, in an order in which each measure's inputs come first.
MEASURES = ("bm25", "embed", "hybrid", "splade", "image", "cross_encoder",
            "hybrid_splade", "hybrid_image")
#: The measures that run a local model; their device is `--model-device`.
MODEL_MEASURES = ("splade", "cross_encoder", "image")


# --------------------------------------------------------------------------- #
# The weights
# --------------------------------------------------------------------------- #

def load_weights_config(path: Path = WEIGHTS_CONFIG) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def weights_dir(cfg: dict, measure: str) -> Path:
    return project_path(cfg["dir"]) / cfg[measure]["subdir"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cfg: dict) -> None:
    """Download each pinned file once; a file already there must match its sha256."""
    for measure in MODEL_MEASURES:
        w = cfg[measure]
        out = weights_dir(cfg, measure)
        out.mkdir(parents=True, exist_ok=True)
        for name, want in w["files"].items():
            dest = out / name
            if dest.exists():
                got = sha256_file(dest)
                if got != want:
                    raise SystemExit(f"{dest}: sha256 {got} != pinned {want}; "
                                     f"delete it and rerun")
                print(f"ok      {measure}/{name}")
                continue
            url = f"https://huggingface.co/{w['repo']}/resolve/{w['revision']}/{name}"
            part = dest.with_name(dest.name + ".part")
            with urllib.request.urlopen(url, timeout=600) as resp, open(part, "wb") as fh:
                while chunk := resp.read(1 << 22):
                    fh.write(chunk)
            got = sha256_file(part)
            if got != want:
                part.unlink()
                raise SystemExit(f"{url}: sha256 {got} != pinned {want}")
            part.rename(dest)
            print(f"fetched {measure}/{name}  {dest.stat().st_size} bytes")


def verify_weights(cfg: dict, measure: str, settings: Mapping[str, Mapping[str, Any]]) -> Path:
    """The pinned files are there, and they are the model config/similarity.toml names."""
    w = cfg[measure]
    if w["repo"] != settings[measure]["model"]:
        raise SystemExit(f"[{measure}] model is {settings[measure]['model']!r} in "
                         f"config/similarity.toml but {w['repo']!r} in {WEIGHTS_CONFIG.name}")
    out = weights_dir(cfg, measure)
    for name, want in w["files"].items():
        p = out / name
        if not p.exists() or sha256_file(p) != want:
            raise SystemExit(f"{p}: missing or not the pinned file; run `fetch` first")
    return out


def local_model(measure: str, settings: Mapping[str, Mapping[str, Any]], path: Path) -> Any:
    """The model object `Context.model` would build, loading from `path`: the settings
    keep the Hugging Face name, so the keys do not depend on where the files lie."""
    s = settings[measure]
    if measure == "splade":
        return M.SpladeEncoder(model=str(path), max_length=s["max_length"], batch=s["batch"],
                               min_weight=s["min_weight"], device=s["device"])
    if measure == "cross_encoder":
        return M.CrossEncoderScorer(model=str(path), max_length=s["max_length"],
                                    batch=s["batch"], device=s["device"])
    if measure == "image":
        return M.ColQwenEncoder(model=str(path), image_batch=s["image_batch"],
                                query_batch=s["query_batch"], query_block=s["query_block"],
                                dtype=s["dtype"], device=s["device"])
    raise KeyError(measure)


def embed_client(settings: Mapping[str, Mapping[str, Any]]) -> M.EmbeddingClient:
    # The same endpoint and key as the LLMs (LLM_API_BASE, LLM_API_KEY).
    from methods.llm_client import api_base, api_key  # noqa: PLC0415

    s = settings["embed"]
    return M.EmbeddingClient(base_url=api_base(), model=s["model"], key=api_key(),
                             batch=s["batch"], batch_chars=s["batch_chars"],
                             timeout_s=s["timeout_s"], max_retries=s["max_retries"],
                             requests_per_minute=s["requests_per_minute"])


# --------------------------------------------------------------------------- #
# The lectures of a set
# --------------------------------------------------------------------------- #

def set_lectures(set_name: str, selection: Path = SELECTION) -> list[tuple[str, str]]:
    """(corpus, lecture_id) of the set, in the selection file's order."""
    if set_name not in SETS:
        raise SystemExit(f"unknown set {set_name!r}; the sets are {SETS}")
    with open(selection, newline="", encoding="utf-8") as fh:
        out = [(r["corpus"], r["lecture_id"]) for r in csv.DictReader(fh)
               if r["set"] == set_name]
    if not out:
        raise SystemExit(f"{selection}: no lecture in set {set_name!r}")
    return out


def load_records(wanted: Sequence[tuple[str, str]], prepared: Path = PREPARED
                 ) -> dict[tuple[str, str], dict]:
    """The prepared records of the wanted lectures (data/prepared/<corpus>.jsonl.gz)."""
    out: dict[tuple[str, str], dict] = {}
    for corpus in sorted({c for c, _ in wanted}):
        ids = {lid for c, lid in wanted if c == corpus}
        with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["lecture_id"] in ids:
                    out[(corpus, r["lecture_id"])] = r
    missing = [w for w in wanted if w not in out]
    if missing:
        raise SystemExit(f"not in {prepared}: {missing[:5]} ({len(missing)} lectures)")
    return out


class Images:
    """The corpora's slide-image zips, opened once each."""

    def __init__(self, prepared: Path = PREPARED):
        self.prepared = prepared
        self._zips: dict[str, zipfile.ZipFile] = {}

    def __call__(self, corpus: str) -> zipfile.ZipFile:
        if corpus not in self._zips:
            self._zips[corpus] = zipfile.ZipFile(self.prepared / f"{corpus}_images.zip")
        return self._zips[corpus]

    def close(self) -> None:
        for z in self._zips.values():
            z.close()
        self._zips.clear()


# --------------------------------------------------------------------------- #
# The store: one SQLite file per set
# --------------------------------------------------------------------------- #

def set_settings(model_device: Optional[str], overrides: Optional[dict] = None,
                 config: Path = CONFIG_PATH) -> tuple[dict, dict]:
    """(settings, the overrides applied): config/similarity.toml, the model device and
    any further overrides (tests: the cache path)."""
    ov: dict[str, dict] = {sec: dict(v) for sec, v in (overrides or {}).items()}
    if model_device:
        for m in MODEL_MEASURES:
            ov.setdefault(m, {})["device"] = model_device
    return load_settings(config, overrides=ov), ov


def number_settings(settings: Mapping[str, Mapping[str, Any]]) -> dict:
    """What decides the numbers of every measure (the keys' settings part)."""
    return {m: key_settings(m, settings) for m in MEASURES}


class Store:
    """data/similarity/<set>/tables.sqlite3. Plain rollback journal, not WAL: only one
    job writes it at a time, and WAL is unsafe when the file is opened from several
    nodes."""

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS lectures (pos INTEGER PRIMARY KEY, corpus TEXT NOT NULL, "
        "lecture_id TEXT NOT NULL, n_units INTEGER NOT NULL, n_slides INTEGER NOT NULL)",
        "CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "measure TEXT NOT NULL, started TEXT, finished TEXT, host TEXT, hardware TEXT, "
        "device TEXT, load_s REAL NOT NULL DEFAULT 0, built INTEGER NOT NULL DEFAULT 0, "
        "reused INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0, "
        "versions TEXT)",
        "CREATE TABLE IF NOT EXISTS tables (measure TEXT NOT NULL, corpus TEXT NOT NULL, "
        "lecture_id TEXT NOT NULL, key TEXT NOT NULL, n INTEGER NOT NULL, m INTEGER NOT NULL, "
        "data BLOB NOT NULL, run_id INTEGER NOT NULL, wall_s REAL NOT NULL, "
        "cost_s REAL NOT NULL, wait_s REAL NOT NULL DEFAULT 0, "
        "failed_s REAL NOT NULL DEFAULT 0, requests INTEGER NOT NULL DEFAULT 0, "
        "request_s REAL NOT NULL DEFAULT 0, failed_attempts INTEGER NOT NULL DEFAULT 0, "
        "texts_sent INTEGER NOT NULL DEFAULT 0, errors TEXT, "
        "PRIMARY KEY (measure, corpus, lecture_id))",
    )

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        for q in self.SCHEMA:
            self.conn.execute(q)

    @classmethod
    def for_set(cls, set_name: str, out_root: Path = OUT_ROOT) -> "Store":
        return cls(out_root / set_name / "tables.sqlite3")

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- meta

    def get_meta(self, name: str) -> Optional[Any]:
        row = self.conn.execute("SELECT value FROM meta WHERE name=?", (name,)).fetchone()
        return None if row is None else json.loads(row[0])

    def check_or_set_meta(self, name: str, value: Any) -> None:
        """Store `value` the first time; later it must be the same."""
        have = self.get_meta(name)
        if have is None:
            self.conn.execute("INSERT INTO meta (name, value) VALUES (?, ?)",
                              (name, json.dumps(value, sort_keys=True)))
        elif json.loads(json.dumps(value, sort_keys=True)) != have:
            raise SystemExit(f"{self.path}: the set was built with another {name}; "
                             f"delete {self.path.parent} to rebuild it from scratch")

    def set_lectures(self, rows: Sequence[tuple[str, str, int, int]]) -> None:
        have = self.conn.execute("SELECT corpus, lecture_id, n_units, n_slides FROM lectures "
                                 "ORDER BY pos").fetchall()
        if have and [tuple(r) for r in have] != [tuple(r) for r in rows]:
            raise SystemExit(f"{self.path}: the set's lectures changed; delete "
                             f"{self.path.parent} to rebuild it from scratch")
        if not have:
            self.conn.execute("BEGIN")
            self.conn.executemany("INSERT INTO lectures (pos, corpus, lecture_id, n_units, "
                                  "n_slides) VALUES (?, ?, ?, ?, ?)",
                                  [(i, *r) for i, r in enumerate(rows)])
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------- runs

    def start_run(self, measure: str, host: str, hardware: str, device: str,
                  versions: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (measure, started, host, hardware, device, versions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (measure, _now(), host, hardware, device, json.dumps(versions, sort_keys=True)))
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, load_s: float, built: int, reused: int,
                   failed: int) -> None:
        self.conn.execute("UPDATE runs SET finished=?, load_s=?, built=?, reused=?, failed=? "
                          "WHERE run_id=?", (_now(), load_s, built, reused, failed, run_id))

    # ------------------------------------------------------------- tables

    def key_of(self, measure: str, corpus: str, lecture_id: str) -> Optional[str]:
        row = self.conn.execute("SELECT key FROM tables WHERE measure=? AND corpus=? AND "
                                "lecture_id=?", (measure, corpus, lecture_id)).fetchone()
        return None if row is None else row[0]

    def put(self, measure: str, corpus: str, lecture_id: str, key: str, S: np.ndarray,
            run_id: int, timing: Mapping[str, Any]) -> None:
        S = np.ascontiguousarray(S, dtype=np.float64)
        self.conn.execute(
            "INSERT OR REPLACE INTO tables (measure, corpus, lecture_id, key, n, m, data, "
            "run_id, wall_s, cost_s, wait_s, failed_s, requests, request_s, "
            "failed_attempts, texts_sent, errors) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (measure, corpus, lecture_id, key, int(S.shape[0]), int(S.shape[1]),
             zlib.compress(S.tobytes(), 6), run_id, float(timing["wall_s"]),
             float(timing["cost_s"]), float(timing.get("wait_s", 0.0)),
             float(timing.get("failed_s", 0.0)), int(timing.get("requests", 0)),
             float(timing.get("request_s", 0.0)), int(timing.get("failed_attempts", 0)),
             int(timing.get("texts_sent", 0)),
             json.dumps(timing.get("errors") or {}, sort_keys=True)))

    def get(self, measure: str, corpus: str, lecture_id: str) -> Optional[np.ndarray]:
        row = self.conn.execute("SELECT n, m, data FROM tables WHERE measure=? AND corpus=? "
                                "AND lecture_id=?", (measure, corpus, lecture_id)).fetchone()
        if row is None:
            return None
        n, m, data = row
        return np.frombuffer(zlib.decompress(data), dtype=np.float64).reshape(n, m).copy()

    def rows(self, measure: Optional[str] = None) -> list[dict]:
        q = ("SELECT t.measure, t.corpus, t.lecture_id, t.key, t.n, t.m, t.run_id, t.wall_s, "
             "t.cost_s, t.wait_s, t.failed_s, t.requests, t.request_s, t.failed_attempts, "
             "t.texts_sent, t.errors FROM tables t")
        args: tuple = ()
        if measure is not None:
            q += " WHERE t.measure=?"
            args = (measure,)
        cols = ["measure", "corpus", "lecture_id", "key", "n", "m", "run_id", "wall_s",
                "cost_s", "wait_s", "failed_s", "requests", "request_s", "failed_attempts",
                "texts_sent", "errors"]
        return [dict(zip(cols, r)) for r in self.conn.execute(q, args)]

    def runs(self) -> list[dict]:
        cols = ["run_id", "measure", "started", "finished", "host", "hardware", "device",
                "load_s", "built", "reused", "failed"]
        return [dict(zip(cols, r)) for r in self.conn.execute(
            "SELECT %s FROM runs ORDER BY run_id" % ", ".join(cols))]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Hardware and versions
# --------------------------------------------------------------------------- #

def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
            else:
                name = platform.processor() or "unknown CPU"
    except OSError:
        name = platform.processor() or "unknown CPU"
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 0
    return f"{name} ({cores} cores)"


def hardware(measure: str, settings: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """(hardware, device) of a measure's run."""
    if measure in MODEL_MEASURES:
        device = str(settings[measure]["device"])
        if device.startswith("cuda"):
            try:
                import torch  # noqa: PLC0415  (inside the image)

                gpu = torch.cuda.get_device_name(0)
            except (ImportError, RuntimeError, AssertionError) as e:
                gpu = f"unknown ({type(e).__name__})"
            return f"GPU {gpu}; CPU {cpu_name()}", device
        return f"CPU {cpu_name()}", device
    if measure == "embed":
        model = settings["embed"]["model"]
        return f"OpenAI-compatible endpoint ({model}); client CPU {cpu_name()}", "endpoint"
    return f"CPU {cpu_name()}", "cpu"


def versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    if PIP_FREEZE.exists():
        out["pip_freeze"] = PIP_FREEZE.read_text(encoding="utf-8").split()
    return out


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #

def _embed_diff(before: Optional[dict], after: Optional[dict]) -> dict:
    if before is None or after is None:
        return {}
    errors = {k: after["errors"].get(k, 0) - before["errors"].get(k, 0)
              for k in after["errors"]}
    return {"requests": after["requests"] - before["requests"],
            "request_s": after["request_seconds"] - before["request_seconds"],
            "texts_sent": after["texts_sent"] - before["texts_sent"],
            "wait_s": after["wait_seconds"] - before["wait_seconds"],
            "failed_attempts": after["failed_attempts"] - before["failed_attempts"],
            "failed_s": after["failed_seconds"] - before["failed_seconds"],
            "errors": {k: v for k, v in errors.items() if v}}


def build(set_name: str, measures: Sequence[str], *, model_device: Optional[str] = None,
          overrides: Optional[dict] = None, models: Optional[dict] = None,
          selection: Path = SELECTION, prepared: Path = PREPARED, out_root: Path = OUT_ROOT,
          weights_cfg: Optional[dict] = None, clock: Callable[[], float] = time.perf_counter,
          log: Callable[[str], None] = print) -> dict:
    """Build (or reuse) `measures`' tables for every lecture of the set.

    `models` ({measure: object}) replaces the real models and endpoint client (tests);
    otherwise the measures asked for get theirs: the local models from the pinned
    weights, embed the endpoint client. Returns {measure: {"built", "reused", "failed"}}.
    """
    unknown = [m for m in measures if m not in SCORERS]
    if unknown:
        raise SystemExit(f"unknown measure(s) {unknown}; the measures are {MEASURES}")
    settings, _ = set_settings(model_device, overrides)
    wanted = set_lectures(set_name, selection)
    records = load_records(wanted, prepared)

    store = Store.for_set(set_name, out_root)
    store.check_or_set_meta("settings", number_settings(settings))
    store.check_or_set_meta("model_device", {m: settings[m]["device"] for m in MODEL_MEASURES})
    store.set_lectures([(c, lid, len(records[(c, lid)]["units"]),
                         len(records[(c, lid)]["slides"])) for c, lid in wanted])

    cache = SimilarityCache(settings["cache"]["path"])
    ctx = Context(settings, cache=cache, models=dict(models or {}), offline=True)
    images = Images(prepared)
    summary: dict[str, dict] = {}
    try:
        for measure in measures:
            summary[measure] = _build_measure(measure, ctx, store, wanted, records, images,
                                              weights_cfg, clock, log)
    finally:
        images.close()
        cache.close()
        store.close()
    return summary


def _build_measure(measure: str, ctx: Context, store: Store, wanted, records, images,
                   weights_cfg, clock, log) -> dict:
    settings = ctx.settings
    need_images = reads_images(measure, settings)
    hw, device = hardware(measure, settings)
    run_id = store.start_run(measure, socket.gethostname(), hw, device, versions())
    built = reused = failed = 0
    load_s = 0.0
    loaded = False
    for corpus, lid in wanted:
        rec = records[(corpus, lid)]
        lec = lecture_from_record(rec, images_zip=images(corpus) if need_images else None)
        key = matrix_key(measure, lec, settings)
        if store.key_of(measure, corpus, lid) == key:
            reused += 1
            continue
        if not loaded:
            # The model (or endpoint client) is set up only when something is to be built,
            # and its load is timed apart from the compute.
            load_s = _provide_model(measure, ctx, weights_cfg, clock)
            loaded = True
        client = ctx.models.get("embed") if measure == "embed" else None
        before = client.stats() if hasattr(client, "stats") else None
        t0 = clock()
        try:
            S = build_matrix(measure, lec, ctx)
        except Exception as e:  # noqa: BLE001  (record it, go on, fail the run at the end)
            failed += 1
            log(f"FAILED {measure} {corpus}/{lid}: {type(e).__name__}: {e}")
            continue
        wall = clock() - t0
        timing: dict[str, Any] = {"wall_s": wall}
        timing.update(_embed_diff(before, client.stats() if before is not None else None))
        timing["cost_s"] = wall - timing.get("wait_s", 0.0) - timing.get("failed_s", 0.0)
        store.put(measure, corpus, lid, key, S, run_id, timing)
        built += 1
        log(f"{measure} {corpus}/{lid} {S.shape[0]}x{S.shape[1]} cost {timing['cost_s']:.3f}s")
    store.finish_run(run_id, load_s=load_s, built=built, reused=reused, failed=failed)
    log(f"== {measure}: built {built}, reused {reused}, failed {failed}, load {load_s:.1f}s, "
        f"on {hw}")
    return {"built": built, "reused": reused, "failed": failed, "load_s": load_s}


def _provide_model(measure: str, ctx: Context, weights_cfg: Optional[dict],
                   clock: Callable[[], float]) -> float:
    """Put the measure's own model into the context if a test did not; return the
    seconds its load took (0 for measures without a model)."""
    if measure == "embed":
        if "embed" not in ctx.models:
            ctx.models["embed"] = embed_client(ctx.settings)
        return 0.0
    if measure not in MODEL_MEASURES:
        return 0.0
    if measure not in ctx.models:
        cfg = weights_cfg if weights_cfg is not None else load_weights_config()
        path = verify_weights(cfg, measure, ctx.settings)
        ctx.models[measure] = local_model(measure, ctx.settings, path)
    obj = ctx.models[measure]
    t0 = clock()
    if hasattr(obj, "_load"):
        obj._load()
    torch = getattr(obj, "_t", None)        # the real models keep torch here once loaded
    if torch is not None and str(ctx.settings[measure]["device"]).startswith("cuda"):
        torch.cuda.synchronize()
    return clock() - t0


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #

def true_slide_ranks(S: np.ndarray, gold: Sequence[int]) -> np.ndarray:
    """The rank (1 = best) of each unit's true slide in its row; tied slides share
    their mean rank, so a row with no spread puts the true slide in the middle."""
    out = []
    for row, g in zip(S, gold):
        if g < 0 or g >= len(row):
            continue
        greater = int(np.sum(row > row[g]))
        ties = int(np.sum(row == row[g])) - 1
        out.append(1.0 + greater + ties / 2.0)
    return np.asarray(out, dtype=np.float64)


def sanity(set_name: str, *, model_device: Optional[str] = None,
           overrides: Optional[dict] = None, selection: Path = SELECTION,
           prepared: Path = PREPARED, out_root: Path = OUT_ROOT,
           measures: Sequence[str] = MEASURES) -> list[dict]:
    """Check every table of the set; write data/similarity/<set>/sanity.csv.

    The model device is the one the set was built with (stored in `meta`) unless given."""
    store = Store.for_set(set_name, out_root)
    if model_device is None:
        built_on = store.get_meta("model_device") or {}
        overrides = {**(overrides or {}),
                     **{m: {**(overrides or {}).get(m, {}), "device": d}
                        for m, d in built_on.items()}}
    settings, _ = set_settings(model_device, overrides)
    wanted = set_lectures(set_name, selection)
    records = load_records(wanted, prepared)
    have = store.get_meta("settings")
    if have is not None and json.loads(json.dumps(number_settings(settings),
                                                  sort_keys=True)) != have:
        raise SystemExit("the settings given are not the ones the set was built with "
                         "(--model-device?)")
    images = Images(prepared)
    out = []
    try:
        for measure in measures:
            need_images = reads_images(measure, settings)
            n_tables = shape_ok = finite_ok = key_ok = units = 0
            ranks, chance = [], []
            for corpus, lid in wanted:
                rec = records[(corpus, lid)]
                S = store.get(measure, corpus, lid)
                if S is None:
                    continue
                n_tables += 1
                lec = lecture_from_record(rec, images_zip=images(corpus) if need_images
                                          else None)
                shape = (len(lec.unit_texts), len(lec.slide_texts))
                ok_shape = S.shape == shape
                shape_ok += ok_shape
                finite_ok += bool(np.isfinite(S).all())
                key_ok += store.key_of(measure, corpus, lid) == matrix_key(measure, lec,
                                                                          settings)
                if ok_shape:
                    gold = gold_labels(rec)
                    r = true_slide_ranks(S, gold)
                    ranks.append(r)
                    units += len(r)
                    m = S.shape[1]
                    chance.extend([min(3, m) / m] * len(r))
            allr = np.concatenate(ranks) if ranks else np.zeros(0)
            row = {"measure": measure, "lectures": len(wanted), "tables": n_tables,
                   "shape_ok": shape_ok, "finite_ok": finite_ok, "order_ok": key_ok,
                   "units": units,
                   "median_true_rank": float(np.median(allr)) if allr.size else float("nan"),
                   "top3_share": float(np.mean(allr <= 3)) if allr.size else float("nan"),
                   "chance_top3_share": float(np.mean(chance)) if chance else float("nan")}
            row["ok"] = (n_tables == len(wanted) and shape_ok == finite_ok == key_ok
                         == n_tables)
            out.append(row)
    finally:
        images.close()
        store.close()
    path = out_root / set_name / "sanity.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return out


def times(set_name: str, *, out_root: Path = OUT_ROOT) -> list[dict]:
    """Per measure: lectures, load and compute time, hardware, endpoint counts; and the
    cost with the measures it is built from. Written to data/similarity/<set>/times.csv."""
    store = Store.for_set(set_name, out_root)
    try:
        settings_meta = store.get_meta("settings") or {}
        runs = {r["run_id"]: r for r in store.runs()}
        per: dict[str, dict] = {}
        for measure in MEASURES:
            rows = store.rows(measure)
            used_runs = sorted({r["run_id"] for r in rows})
            loads = [runs[i]["load_s"] for i in used_runs if runs[i]["built"] > 0]
            per[measure] = {
                "measure": measure, "lectures": len(rows),
                "load_s": float(sum(loads)), "loads": len(loads),
                "compute_s": float(sum(r["cost_s"] for r in rows)),
                "wall_s": float(sum(r["wall_s"] for r in rows)),
                "requests": int(sum(r["requests"] for r in rows)),
                "request_s": float(sum(r["request_s"] for r in rows)),
                "wait_s_ignored": float(sum(r["wait_s"] for r in rows)),
                "failed_attempts_ignored": int(sum(r["failed_attempts"] for r in rows)),
                "failed_s_ignored": float(sum(r["failed_s"] for r in rows)),
                "hardware": "; ".join(sorted({runs[i]["hardware"] for i in used_runs})),
                "deps": " ".join(d["scorer"] for d in
                                 settings_meta.get(measure, {}).get("from", [])),
            }

        def total(m: str) -> float:
            p = per[m]
            return p["load_s"] + p["compute_s"] + sum(total(d) for d in p["deps"].split())

        for m in MEASURES:
            per[m]["cost_with_deps_s"] = total(m)
    finally:
        store.close()
    out = [per[m] for m in MEASURES]
    path = out_root / set_name / "times.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    return out


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the pinned weights of the model measures")
    for name in ("build", "sanity"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True, choices=SETS)
        p.add_argument("--model-device", default=None,
                       help="device of splade, cross_encoder and image (cpu or cuda); "
                            "default: config/similarity.toml")
        if name == "build":
            p.add_argument("--measures", nargs="+", required=True,
                           help=f"some of {' '.join(MEASURES)}, or all")
    p = sub.add_parser("times")
    p.add_argument("--set", required=True, choices=SETS)
    args = ap.parse_args(argv)

    if args.command == "fetch":
        fetch(load_weights_config())
        return 0
    if args.command == "build":
        ms = list(MEASURES) if args.measures == ["all"] else args.measures
        res = build(args.set, ms, model_device=args.model_device)
        return 1 if any(r["failed"] for r in res.values()) else 0
    if args.command == "sanity":
        rows = sanity(args.set, model_device=args.model_device)
        for r in rows:
            print(r)
        return 0 if all(r["ok"] for r in rows) else 1
    for r in times(args.set):
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
