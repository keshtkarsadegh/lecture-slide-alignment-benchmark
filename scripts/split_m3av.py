#!/usr/bin/env python3
"""Step 2b: split the kept M3AV lectures into two versions, A and B, with no lecture in both.

The two versions are independent datasets, m3av_a and m3av_b, with different lectures;
both are prepared like any corpus, with M3AV's own slide text. So that they are
comparable, the halves are drawn with the same mix of topic and length: 50/50 at random
with a fixed seed (SEED), within each stratum.

  topic   the M3AV source of the talk: the prefix of its id (CHI, IPP, MLS, NIH, Ubi).
          M3AV ships no other topic or field metadata; the prefix is how the corpus
          itself groups its talks.
  length  the tercile of the deck's slide count (slides before the 40-slide cap); the
          two cut points come from all kept lectures: short <= t1 < medium <= t2 < long.

Within a stratum the lecture ids are sorted, shuffled with the seeded generator, and the
first half goes to A. When a stratum has an odd count the extra lecture goes to the
version that is smaller so far (A on a tie), so the totals differ by at most one.

The kept lectures are exactly those scripts/prepare.py keeps: the same loader and rules
(prepare.load_m3av on the raw archive, prepare.decide).

Output (a definition: not in .gitignore; committed once confirmed):
    data/splits/m3av_split.csv    lecture_id, version (A|B), stratum
    data/splits/m3av_split.json   the seed, the cut points and the balance of the halves

Usage:
    python scripts/split_m3av.py [--raw-root data/raw] [--out data/splits/m3av_split.csv]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import prepare

SEED = 20261011  # the date the split was made; fixed, never changed
LENGTHS = ("short", "medium", "long")


def tercile_cuts(values: list[int]) -> tuple[int, int]:
    """(t1, t2): the values at the 1/3 and 2/3 positions of the sorted list."""
    v = sorted(values)
    n = len(v)
    return v[math.ceil(n / 3) - 1], v[math.ceil(2 * n / 3) - 1]


def length_class(n: int, cuts: tuple[int, int]) -> str:
    return LENGTHS[0] if n <= cuts[0] else LENGTHS[1] if n <= cuts[1] else LENGTHS[2]


def split_lectures(lectures: list[dict], seed: int = SEED) -> tuple[list[dict], tuple[int, int]]:
    """lectures: dicts with lecture_id, topic, n_slides_deck. Returns (rows of lecture_id,
    version, stratum sorted by lecture id, the length cut points)."""
    ids = [lec["lecture_id"] for lec in lectures]
    if len(set(ids)) != len(ids):
        raise ValueError("a lecture id appears twice")
    cuts = tercile_cuts([lec["n_slides_deck"] for lec in lectures])
    strata: dict[str, list[str]] = collections.defaultdict(list)
    for lec in lectures:
        strata[f"{lec['topic']}|{length_class(lec['n_slides_deck'], cuts)}"].append(lec["lecture_id"])
    rng = random.Random(seed)
    size = {"A": 0, "B": 0}
    rows = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng.shuffle(members)
        n_a = len(members) // 2
        if len(members) % 2:
            n_a += size["A"] <= size["B"]
        for i, lid in enumerate(members):
            version = "A" if i < n_a else "B"
            size[version] += 1
            rows.append({"lecture_id": lid, "version": version, "stratum": stratum})
    rows.sort(key=lambda r: r["lecture_id"])
    return rows, cuts


def balance(rows: list[dict], lectures: list[dict]) -> dict:
    """Counts per stratum per version, and per version the mean deck slides, slides kept
    and units kept."""
    by_id = {lec["lecture_id"]: lec for lec in lectures}
    per_stratum: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"A": 0, "B": 0})
    per_version: dict[str, list[dict]] = {"A": [], "B": []}
    for r in rows:
        per_stratum[r["stratum"]][r["version"]] += 1
        per_version[r["version"]].append(by_id[r["lecture_id"]])

    def mean(xs: list[dict], key: str) -> float:
        return round(sum(x[key] for x in xs) / len(xs), 2) if xs else 0.0

    return {
        "per_stratum": dict(sorted(per_stratum.items())),
        "per_version": {
            v: {
                "lectures": len(xs),
                "mean_slides_deck": mean(xs, "n_slides_deck"),
                "mean_slides_kept": mean(xs, "n_slides"),
                "mean_units_kept": mean(xs, "n_units"),
                "units_kept": sum(x["n_units"] for x in xs),
            }
            for v, xs in per_version.items()
        },
    }


def kept_lectures(src: prepare.Sources) -> tuple[list[dict], int]:
    """The M3AV lectures prepare.py keeps, with what the split needs; and the number of
    talks it drops."""
    out, dropped = [], 0
    for item in prepare.load_m3av(src, None):
        result = item if isinstance(item, prepare.Dropped) else prepare.decide("m3av", item)
        if isinstance(result, prepare.Dropped):
            dropped += 1
            continue
        rec, _ = result
        out.append(
            {
                "lecture_id": rec["lecture_id"],
                "topic": rec["meta"]["venue_prefix"],
                "n_slides_deck": rec["n_slides_before_cap"],
                "n_slides": rec["n_slides_after_cap"],
                "n_units": rec["n_units_after_cap"],
            }
        )
    return out, dropped


def write_split(path: Path, rows: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lecture_id", "version", "stratum"])
        w.writeheader()
        w.writerows(rows)
    os.replace(part, path)
    jpath = path.with_suffix(".json")
    part = jpath.with_name(jpath.name + ".part")
    part.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(part, jpath)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-root", type=Path, default=prepare.DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=prepare.DEFAULT_SPLIT)
    args = ap.parse_args(argv)
    prepare.log("== loading the kept M3AV lectures")
    lectures, dropped = kept_lectures(prepare.Sources(args.raw_root))
    rows, cuts = split_lectures(lectures, SEED)
    bal = balance(rows, lectures)
    meta = {
        "seed": SEED,
        "lectures": len(rows),
        "talks_dropped_by_the_rules": dropped,
        "topic": "M3AV source: the prefix of the talk id",
        "length": "tercile of the deck's slide count (before the cap)",
        "length_cuts": {"short": f"<= {cuts[0]}", "medium": f"{cuts[0] + 1}..{cuts[1]}", "long": f"> {cuts[1]}"},
        "balance": bal,
        "script": "scripts/split_m3av.py",
    }
    write_split(args.out, rows, meta)
    prepare.log(f"  {len(rows)} lectures ({dropped} talks dropped by the rules); length cuts {cuts}")
    prepare.log(f"  {'stratum':16s} {'A':>4s} {'B':>4s}")
    for s, c in bal["per_stratum"].items():
        prepare.log(f"  {s:16s} {c['A']:4d} {c['B']:4d}")
    for v, b in bal["per_version"].items():
        prepare.log(f"  {v}: {json.dumps(b)}")
    prepare.log(f"== wrote {args.out} and {args.out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
