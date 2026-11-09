#!/usr/bin/env python3
"""Step 15: the blind 100-unit check sample of the importance labels.

The agents' labels (data/importance/labels.csv.gz, scripts/importance_merge.py) are
checked on a random sample of units. The checker labels the sample without seeing the
agents' labels; the two are compared afterwards.

THE DRAW (seed 20261012): a lecture at random (every test lecture equally likely), then
a unit at random within it, until `n` distinct units are drawn. Drawing the lecture
first keeps the short lectures in the sample: drawing units uniformly would leave it
almost all long lectures (a few lectures hold most of the units).

OUTPUT (data/importance/):
  local/check_sample.md per item: its number, the corpus and lecture, the unit's true
                        slide (its text, first 400 characters), the two units before
                        and after it (cut short) and the unit itself; NO label
  check_sample_key.csv  item, corpus, lecture_id, unit_index, label (the agents')

The sheet quotes the corpora's transcripts and slide text verbatim, and two of the
corpora (MCD, Slide-Highlight) carry no licence that allows passing their text on, so
it goes into local/, which .gitignore keeps out of git; the key holds ids only and is
in git. The same seed draws the same sheet again from the prepared lectures.

The texts come from the prepared records, read as the packs read them
(scripts/importance_pack.py), so the checker sees what the agents saw.

THE COMPARISON (--check KEY MANUAL): the checker's labels (check_manual.csv: item,
corpus, lecture_id, unit_index, label_check) against the key, unit by unit: n, exact
agreement, agreement within one level, and the 3x3 confusion (rows: the checker's
label, columns: the agents').

    python scripts/importance_sample.py [--n 100] [--seed 20261012] [--out DIR]
    python scripts/importance_sample.py --check data/importance/check_sample_key.csv \
        data/importance/check_manual.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importance_pack as P  # noqa: E402

OUT = REPO_ROOT / "data" / "importance"
#: Where the sheet goes under `out`: kept out of git (module docstring, OUTPUT).
LOCAL = "local"
LABELS = OUT / "labels.csv.gz"
N = 100
SEED = 20261012
#: The true slide's text shown (characters).
SLIDE_CHARS = 400
#: Units shown before and after the sampled one, and their length (characters).
CONTEXT = 2
CONTEXT_CHARS = 160

Unit = tuple[str, str, int]  # (corpus, lecture_id, unit_index)


def read_labels(path: Path = LABELS) -> dict[Unit, int]:
    """The merged labels, in the file's order (selection order, then unit order)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {(r["corpus"], r["lecture_id"], int(r["unit_index"])): int(r["label"])
                for r in csv.DictReader(fh)}


def draw(units: Sequence[Unit], n: int = N, seed: int = SEED) -> list[Unit]:
    """`n` distinct units: a lecture uniformly at random, then one of its units
    uniformly at random; a unit drawn again is drawn anew (module docstring)."""
    by_lecture: dict[tuple[str, str], list[int]] = {}
    for c, l, i in units:
        by_lecture.setdefault((c, l), []).append(i)
    if n > len(units):
        raise SystemExit(f"cannot draw {n} distinct units from {len(units)}")
    keys = list(by_lecture)
    rng = random.Random(seed)
    out: list[Unit] = []
    seen: set[Unit] = set()
    while len(out) < n:
        c, l = rng.choice(keys)
        u = (c, l, rng.choice(by_lecture[(c, l)]))
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def cut(text: str, limit: int) -> str:
    s = P.one_line(text)
    return s if len(s) <= limit else s[:limit].rstrip() + " [...]"


def item_md(no: int, unit: Unit, lec: P.Lecture) -> str:
    """One item of check_sample.md: no label anywhere."""
    _, _, i = unit
    _, k, text = lec.units[i]
    slide = P.slide_line(dict(lec.slides)[k], SLIDE_CHARS)
    lines = [f"## {no}. {lec.corpus} / {lec.lecture_id}", "",
             f"**True slide {k}:** {slide}", ""]
    for j in range(max(0, i - CONTEXT), min(lec.n_units, i + CONTEXT + 1)):
        _, kj, tj = lec.units[j]
        if j == i:
            lines.append(f"- **u{j} [slide {kj}] {P.one_line(tj)}**")
        else:
            lines.append(f"- u{j} [slide {kj}] {cut(tj, CONTEXT_CHARS)}")
    return "\n".join(lines) + "\n"


def write_sample(sample: Sequence[Unit], labels: Mapping[Unit, int],
                 lectures: Sequence[P.Lecture], out: Path = OUT, *, seed: int = SEED
                 ) -> None:
    """`out`/local/check_sample.md (no labels) and `out`/check_sample_key.csv (the
    labels)."""
    by_key = {L.key: L for L in lectures}
    missing = sorted({(c, l) for c, l, _ in sample} - set(by_key))
    if missing:
        raise SystemExit(f"not a test lecture: {missing[:5]}")
    (out / LOCAL).mkdir(parents=True, exist_ok=True)
    head = ["# Importance labels: the blind check sample", "",
            f"{len(sample)} units drawn at random (a lecture first, then a unit in it; "
            f"seed {seed}). Label each unit in bold 0, 1 or 2 against its "
            f"true slide, following docs/IMPORTANCE.md. The agents' labels are in "
            f"check_sample_key.csv: do not open it before labelling.", ""]
    items = [item_md(no, u, by_key[u[:2]]) for no, u in enumerate(sample, start=1)]
    (out / LOCAL / "check_sample.md").write_text("\n".join(head) + "\n" + "\n".join(items),
                                                 encoding="utf-8")
    with open(out / "check_sample_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "corpus", "lecture_id", "unit_index", "label"])
        for no, u in enumerate(sample, start=1):
            w.writerow([no, *u, labels[u]])


def compare(key: Path, manual: Path) -> tuple[int, int, int, list[list[int]]]:
    """(n, exact, within one, confusion[check][agents]) over the units of `key`.

    Units are matched on (corpus, lecture_id, unit_index), not on the item number, so
    a manual file in another order or with renumbered items still compares right; a
    unit in one file and not the other stops the comparison rather than being skipped,
    since a silently smaller n would overstate what was checked."""
    def read(path: Path, col: str) -> dict[Unit, int]:
        with open(path, newline="", encoding="utf-8") as fh:
            return {(r["corpus"], r["lecture_id"], int(r["unit_index"])): int(r[col])
                    for r in csv.DictReader(fh)}
    agents, check = read(key, "label"), read(manual, "label_check")
    if set(agents) != set(check):
        raise SystemExit(f"{key} and {manual} hold different units")
    conf = [[0] * 3 for _ in range(3)]
    for u, a in agents.items():
        conf[check[u]][a] += 1
    exact = sum(conf[i][i] for i in range(3))
    within = sum(conf[i][j] for i in range(3) for j in range(3) if abs(i - j) <= 1)
    return len(agents), exact, within, conf


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=LABELS, help="the merged labels")
    ap.add_argument("--n", type=int, default=N, help="units in the sample")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", nargs=2, type=Path, metavar=("KEY", "MANUAL"),
                    help="compare the key with the checker's labels; draws nothing")
    a = ap.parse_args(argv)
    if a.check:
        n, exact, within, conf = compare(*a.check)
        print(f"n {n}  exact {100 * exact / n:.0f}%  within one {100 * within / n:.0f}%")
        print("confusion (rows: check 0/1/2, columns: agents 0/1/2)")
        for i, row in enumerate(conf):
            print(f"  {i}  {row}")
        return 0
    labels = read_labels(a.labels)
    sample = draw(list(labels), a.n, a.seed)
    write_sample(sample, labels, P.load_test_lectures(), a.out, seed=a.seed)
    print(f"{len(sample)} units from {len({u[:2] for u in sample})} lectures -> "
          f"{a.out / LOCAL / 'check_sample.md'} and {a.out / 'check_sample_key.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
