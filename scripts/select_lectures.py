#!/usr/bin/env python3
"""Step 7: the lecture selection, two disjoint sets of lectures (tuning and test).

It reads the categories (data/categories/document_categories.csv, step 6) and writes
data/selection/. The rules, in plain words (docs/SELECTION.md has them in full; the
numbers are in config/selection.toml):

  sizes         tuning 60 lectures, test 140
  small corpora mavils, mcd and slide_highlight are taken in whole; tuning gets
                round(n/3) of each (half up), test the rest
  fill corpora  the rest of each set comes from m3av_a and m3av_b, as equally as whole
                numbers allow (a remainder goes to the first listed)
  minima        every level of every axis except corpus has at least 5 lectures in
                tuning and 10 in test; corpus is fixed by the counts above

The pool is put in one seeded random order (sorted by corpus and lecture first, so the
input's row order does not matter). Each set is then filled in two phases, the small
corpora first so their lectures also count towards the minima, then the fill corpora;
in each phase:

  1. greedy: add the lecture that covers the most unmet level minima of the set (one per
     axis whose level is still short), the earliest in the seeded order on a tie, while
     some lecture covers one and its corpus still has room;
  2. random: fill each corpus's count with its lectures in the seeded order.

The tuning set is filled first; the test set comes from the lectures left. The run stops
before selecting if a level has fewer lectures in the pool than the minima add up to,
and after a set if a minimum is still unmet: a minimum is never relaxed.

Output, in data/selection/:
    selected_lectures.csv   corpus, lecture_id, set, one column per axis
    selection_counts.csv    axis, level, n in tuning, n in test, n in the pool
    selection.json          the input's path, row count and sha256, the seed, the
                            settings

Usage:
    python scripts/select_lectures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tomllib
from pathlib import Path
from typing import Sequence

import categorise as ct

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "selection.toml"
IN_PATH = REPO_ROOT / "data" / "categories" / "document_categories.csv"
OUT_DIR = REPO_ROOT / "data" / "selection"

SETS = ("tuning", "test")
POOL = "pool"
UNCOVERED = ("corpus",)   # fixed by the per-corpus counts, so no minimum of its own


# --------------------------------------------------------------------------- the counts

def round_half_up(n: int, num: int, den: int) -> int:
    """n * num / den rounded half up, in whole numbers (no float rounding to choose)."""
    return (2 * n * num + den) // (2 * den)


def corpus_counts(pool_sizes: dict[str, int], cfg: dict) -> dict[str, dict[str, int]]:
    """Set -> corpus -> number of lectures.

    The small corpora are split whole between the sets; the fill corpora take what is
    left of each set's size, as equally as whole numbers allow.
    """
    small = list(cfg["small_corpora"]["corpora"])
    fill = list(cfg["fill_corpora"]["corpora"])
    if set(small) & set(fill):
        raise ValueError(f"a corpus is both small and fill: {sorted(set(small) & set(fill))}")
    if not fill:
        raise ValueError("no fill corpora")
    num, den = cfg["small_corpora"]["tuning_share"]
    out: dict[str, dict[str, int]] = {s: {} for s in SETS}
    for c in small:
        n = pool_sizes.get(c, 0)
        out["tuning"][c] = round_half_up(n, num, den)
        out["test"][c] = n - out["tuning"][c]
    for s in SETS:
        rest = int(cfg["sets"][s]["size"]) - sum(out[s][c] for c in small)
        if rest < 0:
            raise ValueError(f"set {s}: the small corpora alone give "
                             f"{sum(out[s][c] for c in small)} lectures, more than its "
                             f"size {cfg['sets'][s]['size']}")
        k = len(fill)
        for i, c in enumerate(fill):
            out[s][c] = rest // k + (1 if i < rest % k else 0)
    for c in fill:
        want = sum(out[s][c] for s in SETS)
        if want > pool_sizes.get(c, 0):
            raise ValueError(f"{c}: the sets need {want} lectures, the pool has "
                             f"{pool_sizes.get(c, 0)}")
    return out


def check_pool(pool: Sequence[dict], cfg: dict, levels: dict[str, list[str]]) -> None:
    """Stop if a level has fewer lectures in the pool than the sets' minima add up to."""
    need = sum(int(cfg["sets"][s]["min_per_level"]) for s in SETS)
    short = []
    for axis in cover_axes(levels):
        for level in levels[axis]:
            n = sum(r[axis] == level for r in pool)
            if n < need:
                short.append(f"{axis}={level} ({n})")
    if short:
        raise ValueError(f"minimum not attainable: the sets need at least {need} lectures "
                         f"per level, the pool has fewer for " + ", ".join(short))


def cover_axes(levels: dict[str, list[str]]) -> list[str]:
    """The axes whose every level must reach the minimum."""
    return [a for a in levels if a not in UNCOVERED]


# --------------------------------------------------------------------------- the fill

def seeded_order(pool: Sequence[dict], seed: int,
                 corpora: Sequence[str] = ct.CORPORA) -> list[dict]:
    """The pool in one seeded random order, shuffled from a fixed sort so the input's
    row order cannot change the result."""
    rank = {c: i for i, c in enumerate(corpora)}
    ordered = sorted(pool, key=lambda r: (rank.get(r["corpus"], len(rank)), r["corpus"],
                                          r["lecture_id"]))
    random.Random(seed).shuffle(ordered)
    return ordered


def _fill_phase(order: Sequence[dict], taken: set, chosen: list[dict],
                room: dict[str, int], need: dict[tuple[str, str], int],
                axes: Sequence[str]) -> None:
    """Fill the corpora in `room` up to their counts: greedy for the minima, then in the
    seeded order. `chosen`, `taken`, `room` and `need` are updated in place."""

    def add(r):
        chosen.append(r)
        taken.add((r["corpus"], r["lecture_id"]))
        room[r["corpus"]] -= 1
        for a in axes:
            need[(a, r[a])] = need.get((a, r[a]), 0) - 1

    def free(r):
        return room.get(r["corpus"], 0) > 0 and (r["corpus"], r["lecture_id"]) not in taken

    while True:
        best, best_score = None, 0
        for r in order:
            if not free(r):
                continue
            score = sum(need.get((a, r[a]), 0) > 0 for a in axes)
            if score > best_score:            # strict: a tie keeps the earlier lecture
                best, best_score = r, score
        if best is None:
            break
        add(best)
    for r in order:
        if free(r):
            add(r)


def select(pool: Sequence[dict], cfg: dict,
           levels: dict[str, list[str]]) -> list[dict]:
    """The selected lectures, each with its set, tuning first; every row of the pool
    must carry a level of `levels` on every axis."""
    keys = [(r["corpus"], r["lecture_id"]) for r in pool]
    if len(set(keys)) != len(keys):
        raise ValueError("a lecture appears more than once")
    for r in pool:
        for axis, lv in levels.items():
            if r[axis] not in lv:
                raise ValueError(f"{r['corpus']}/{r['lecture_id']}: {axis} {r[axis]!r} "
                                 f"not in {lv}")
    if set(cfg["sets"]) != set(SETS):
        raise ValueError(f"[sets] must define exactly {SETS}")
    pool_sizes: dict[str, int] = {}
    for r in pool:
        pool_sizes[r["corpus"]] = pool_sizes.get(r["corpus"], 0) + 1
    counts = corpus_counts(pool_sizes, cfg)
    check_pool(pool, cfg, levels)

    axes = cover_axes(levels)
    order = seeded_order(pool, int(cfg["seed"]))
    taken: set = set()
    out = []
    for s in SETS:
        minimum = int(cfg["sets"][s]["min_per_level"])
        need = {(a, lv): minimum for a in axes for lv in levels[a]}
        chosen: list[dict] = []
        for phase in ("small_corpora", "fill_corpora"):
            room = {c: counts[s][c] for c in cfg[phase]["corpora"]}
            _fill_phase(order, taken, chosen, room, need, axes)
            if any(room.values()):
                raise ValueError(f"set {s}: not enough lectures left for {room}")
        unmet = [f"{a}={lv} ({minimum - n} short)" for (a, lv), n in need.items() if n > 0]
        if unmet:
            raise ValueError(f"set {s}: minimum {minimum} per level not met from the "
                             f"lectures left: " + ", ".join(unmet))
        out += [{"corpus": r["corpus"], "lecture_id": r["lecture_id"], "set": s,
                 **{a: r[a] for a in levels if a != "corpus"}} for r in chosen]
    rank = {c: i for i, c in enumerate(ct.CORPORA)}
    out.sort(key=lambda r: (SETS.index(r["set"]), rank.get(r["corpus"], len(rank)),
                            r["corpus"], r["lecture_id"]))
    return out


def selection_counts(selected: Sequence[dict], pool: Sequence[dict],
                     levels: dict[str, list[str]]) -> list[dict]:
    """One row per axis and level: n in each set and in the pool (zero rows included)."""
    rows = []
    for axis, lv in levels.items():
        for level in lv:
            rows.append({"axis": axis, "level": level,
                         **{s: sum(r["set"] == s and r[axis] == level for r in selected)
                            for s in SETS},
                         POOL: sum(r[axis] == level for r in pool)})
    return rows


# --------------------------------------------------------------------------- the files

def read_categories(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pool_levels(pool: Sequence[dict], repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Every axis's levels, in step 6's order (categorise.axis_levels)."""
    with open(repo_root / "config" / "categories.toml", "rb") as fh:
        cat_cfg = tomllib.load(fh)
    fos = ct.read_fos(repo_root / cat_cfg["discipline"]["fos_list"])
    return ct.axis_levels(cat_cfg, fos, pool)


def build(cfg: dict, in_path: Path = IN_PATH,
          repo_root: Path = REPO_ROOT) -> tuple[list[dict], dict, list[dict]]:
    pool = read_categories(in_path)
    levels = pool_levels(pool, repo_root)
    selected = select(pool, cfg, levels)
    try:
        shown = str(in_path.resolve().relative_to(repo_root))
    except ValueError:
        shown = str(in_path)
    record = {"input": {"path": shown, "rows": len(pool), "sha256": ct.sha256_file(in_path)},
              "seed": int(cfg["seed"]), "settings": cfg}
    return selected, record, selection_counts(selected, pool, levels)


def columns() -> list[str]:
    """The columns of selected_lectures.csv: the lecture, its set, one per axis."""
    return ["corpus", "lecture_id", "set", *ct.AXES[1:]]


def write(selected: Sequence[dict], record: dict, counts: Sequence[dict],
          out_dir: Path = OUT_DIR) -> Path:
    """No timestamp in any file: the same input and settings give byte-identical output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selected_lectures.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns())
        w.writeheader()
        w.writerows(selected)
    with open(out_dir / "selection_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["axis", "level", *SETS, POOL])
        w.writeheader()
        w.writerows(counts)
    with open(out_dir / "selection.json", "w") as fh:
        json.dump(record, fh, indent=1)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    selected, record, counts = build(cfg)
    path = write(selected, record, counts)
    print(f"{len(selected)} lectures -> {path}")
    for s in SETS:
        sub = [r for r in selected if r["set"] == s]
        print(f"  {s}: {len(sub)}: " + ", ".join(
            f"{c} {sum(r['corpus'] == c for r in sub)}" for c in ct.CORPORA))
    print("axis, level, " + ", ".join(SETS) + f", {POOL}")
    for r in counts:
        print(f"  {r['axis']}, {r['level']}, " + ", ".join(str(r[s]) for s in SETS)
              + f", {r[POOL]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
