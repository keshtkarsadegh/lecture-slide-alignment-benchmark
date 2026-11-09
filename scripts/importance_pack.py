#!/usr/bin/env python3
"""Step 15: the packs for the importance labelling of the 140 test lectures.

Every transcript unit of a test lecture gets an importance label 0/1/2 (filler /
supporting / core) relative to its TRUE slide, assigned by Claude agents following
docs/IMPORTANCE.md. This script writes what those agents read: one plain-text pack per
test lecture, holding only the lecture's slide texts and its units with their true
slide. Nothing of any method's output is in a pack, so the labels are blind to the
methods.

The lectures are data/selection/selected_lectures.csv's `test` set; their slides and
units are the prepared records of step 2 (data/prepared/<corpus>.jsonl.gz): slide
`index` 1..n with its `text`, unit `index` 0..n-1 with its `text` and `true_slide`
(1-based). A pack names a unit `u<index>` and a slide `slide <index>`, so the labels
join back to the prepared data without any renumbering.

PACK (`<lecture_id>.txt`):

    lecture: <lecture_id>
    corpus: <corpus>
    n_slides: <n>
    n_units: <n>

    SLIDES
    slide <k>: <its text on one line, lines joined by " / ", cut at 700 characters
               with " [...]" when cut; "(no text)" when empty>

    UNITS
    u<index> [slide <k>] <unit text on one line>

A lecture with more than 250 units is split into parts of at most 250 units, as equal
as possible (`<lecture_id>.part<i>.txt`): each part has the header (n_units is the
lecture's, a `part:` line says which units the file holds), only the slides its units
are on, and its units. A labeller holds one part at a time; equal parts keep the
context of each about the same.

BATCHES. One labelling agent takes one batch of about 900 units. A lecture is never
split across batches. A lecture with more units than that is a batch of its own; the
others are spread over ceil(their units / 900) batches, largest first, each to the
batch with the fewest units so far, so the batches come out about equal.

OUTPUT (data/importance/packs/, not in git; the old pack files there are removed first):
    <lecture_id>.txt, <lecture_id>.part<i>.txt   the packs
    MANIFEST.csv   corpus, lecture_id, n_units, n_slides, batch (selection order)
    BATCHES.md     batch -> its files and units

    python scripts/importance_pack.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tune as T  # noqa: E402

SELECTION = T.SELECTION
PREPARED = T.PREPARED
OUT = REPO_ROOT / "data" / "importance" / "packs"

#: Slide text longer than this is cut (a pack must stay readable in one go).
SLIDE_CHARS = 700
#: A lecture with more units than this is split into parts.
PART_UNITS = 250
#: The units one labelling agent takes.
BATCH_UNITS = 900


@dataclass
class Lecture:
    corpus: str
    lecture_id: str
    slides: list[tuple[int, str]]          # (index 1..n, text)
    units: list[tuple[int, int, str]]      # (index 0..n-1, true slide 1..n, text)

    @property
    def key(self) -> tuple[str, str]:
        return (self.corpus, self.lecture_id)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_slides(self) -> int:
        return len(self.slides)


# --------------------------------------------------------------------------- #
# The test lectures
# --------------------------------------------------------------------------- #

def lecture_from_record(r: Mapping) -> Lecture:
    """A prepared record as a Lecture. It stops if the slides are not 1..n, the units
    not 0..n-1, or a true slide is outside the deck: the labels are keyed by these
    numbers, so they must be exactly the prepared ones."""
    slides = sorted((int(s["index"]), s["text"] or "") for s in r["slides"])
    units = sorted((int(u["index"]), int(u["true_slide"]), u["text"] or "") for u in r["units"])
    name = f"{r['corpus']}/{r['lecture_id']}"
    if [i for i, _ in slides] != list(range(1, len(slides) + 1)):
        raise SystemExit(f"{name}: slide indices are not 1..{len(slides)}")
    if [i for i, _, _ in units] != list(range(len(units))):
        raise SystemExit(f"{name}: unit indices are not 0..{len(units) - 1}")
    bad = [i for i, k, _ in units if not 1 <= k <= len(slides)]
    if bad:
        raise SystemExit(f"{name}: units {bad[:5]} have no true slide in 1..{len(slides)}")
    return Lecture(r["corpus"], r["lecture_id"], slides, units)


def load_test_lectures(selection: Path = SELECTION, prepared: Path = PREPARED
                       ) -> list[Lecture]:
    """The test set's lectures in selection order (tune.set_lectures: the same set
    file and reading as the final run)."""
    wanted = T.set_lectures(T.TEST_SET, selection, tuned_only=True)
    keep = set(wanted)
    out: dict[tuple[str, str], Lecture] = {}
    for corpus in sorted({c for c, _ in keep}):
        with gzip.open(prepared / f"{corpus}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if (r["corpus"], r["lecture_id"]) in keep:
                    out[(r["corpus"], r["lecture_id"])] = lecture_from_record(r)
    missing = [w for w in wanted if w not in out]
    if missing:
        raise SystemExit(f"not in {prepared}: {missing[:5]} ({len(missing)} lectures)")
    return [out[w] for w in wanted]


# --------------------------------------------------------------------------- #
# The packs
# --------------------------------------------------------------------------- #

def one_line(text: str) -> str:
    """A unit's text on one line (all whitespace runs become one space)."""
    return " ".join(text.split())


def slide_line(text: str, limit: int = SLIDE_CHARS) -> str:
    """A slide's text on one line: its non-empty lines joined by " / " (the line
    breaks of a slide carry its structure: title, bullets), cut at `limit`
    characters with " [...]" when cut."""
    s = " / ".join(one_line(x) for x in text.splitlines() if x.strip())
    if not s:
        return "(no text)"
    if len(s) > limit:
        s = s[:limit].rstrip() + " [...]"
    return s


def split_parts(n: int, size: int = PART_UNITS) -> list[tuple[int, int]]:
    """[start, end) ranges of n units in ceil(n / size) parts as equal as possible."""
    k = max(1, math.ceil(n / size))
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < extra else 0)
        out.append((start, end))
        start = end
    return out


def pack_files(lec: Lecture, part_units: int = PART_UNITS) -> list[tuple[str, str]]:
    """(file name, text) of a lecture's pack: one file, or parts when it has more
    than `part_units` units."""
    parts = split_parts(lec.n_units, part_units)
    texts = dict(lec.slides)
    out = []
    for i, (a, b) in enumerate(parts, start=1):
        units = lec.units[a:b]
        head = [f"lecture: {lec.lecture_id}", f"corpus: {lec.corpus}",
                f"n_slides: {lec.n_slides}", f"n_units: {lec.n_units}"]
        if len(parts) > 1:
            head.append(f"part: {i} of {len(parts)}, units u{units[0][0]} to "
                        f"u{units[-1][0]} ({len(units)} units; only their slides are shown)")
            shown = sorted({k for _, k, _ in units})
            name = f"{lec.lecture_id}.part{i}.txt"
        else:
            shown = [k for k, _ in lec.slides]
            name = f"{lec.lecture_id}.txt"
        lines = head + ["", "SLIDES"]
        lines += [f"slide {k}: {slide_line(texts[k])}" for k in shown]
        lines += ["", "UNITS"]
        lines += [f"u{ix} [slide {k}] {one_line(t)}" for ix, k, t in units]
        out.append((name, "\n".join(lines) + "\n"))
    return out


# --------------------------------------------------------------------------- #
# The batches
# --------------------------------------------------------------------------- #

def make_batches(sizes: Mapping[tuple[str, str], int], target: int = BATCH_UNITS
                 ) -> list[list[tuple[str, str]]]:
    """The lectures in batches of about `target` units (module docstring): a lecture
    above `target` alone, the rest largest first into the emptiest of
    ceil(rest / target) batches (ties: the first batch). Each batch's lectures are
    sorted by (corpus, lecture_id); the lone lectures come first."""
    big = sorted((k for k, n in sizes.items() if n > target), key=lambda k: (-sizes[k], k))
    rest = sorted((k for k, n in sizes.items() if n <= target), key=lambda k: (-sizes[k], k))
    n_b = math.ceil(sum(sizes[k] for k in rest) / target) if rest else 0
    bins: list[list[tuple[str, str]]] = [[] for _ in range(n_b)]
    load = [0] * n_b
    for k in rest:
        j = min(range(n_b), key=lambda i: (load[i], i))
        bins[j].append(k)
        load[j] += sizes[k]
    return [[k] for k in big] + [sorted(b) for b in bins]


def batch_name(i: int) -> str:
    return f"b{i + 1:02d}"


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def export(lectures: Sequence[Lecture], out: Path = OUT, *, part_units: int = PART_UNITS,
           target: int = BATCH_UNITS) -> dict:
    """Write the packs, MANIFEST.csv and BATCHES.md into `out` (old pack files there
    are removed first, so no stale pack survives). Returns the batches:
    {batch: [(corpus, lecture_id, [files], n_units)]}."""
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.txt")) + [out / "MANIFEST.csv", out / "BATCHES.md"]:
        if old.exists():
            old.unlink()
    by_key = {L.key: L for L in lectures}
    files = {}
    for L in lectures:
        files[L.key] = []
        for name, text in pack_files(L, part_units):
            (out / name).write_text(text, encoding="utf-8")
            files[L.key].append(name)
    bins = make_batches({L.key: L.n_units for L in lectures}, target)
    batch_of = {k: batch_name(i) for i, b in enumerate(bins) for k in b}
    batches = {batch_name(i): [(c, l, files[(c, l)], by_key[(c, l)].n_units) for c, l in b]
               for i, b in enumerate(bins)}

    with open(out / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "lecture_id", "n_units", "n_slides", "batch"])
        for L in lectures:
            w.writerow([L.corpus, L.lecture_id, L.n_units, L.n_slides, batch_of[L.key]])

    n_units = sum(L.n_units for L in lectures)
    md = ["# Importance labelling: the batches", "",
          f"{len(lectures)} lectures, {n_units} units, {len(batches)} batches of about "
          f"{target} units. One agent labels one batch, following docs/IMPORTANCE.md, and "
          f"writes `<batch>.csv`.", "",
          "| batch | lectures | units |", "|---|---|---|"]
    md += [f"| {b} | {len(v)} | {sum(x[3] for x in v)} |" for b, v in batches.items()]
    for b, v in batches.items():
        md += ["", f"## {b}: {len(v)} lectures, {sum(x[3] for x in v)} units", "",
               "| corpus | lecture | files | units |", "|---|---|---|---|"]
        md += [f"| {c} | {l} | {', '.join(f)} | {n} |" for c, l, f, n in v]
    (out / "BATCHES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return batches


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="the packs folder")
    a = ap.parse_args(argv)
    lectures = load_test_lectures()
    batches = export(lectures, a.out)
    n_files = sum(len(f) for v in batches.values() for _, _, f, _ in v)
    print(f"{len(lectures)} lectures, {sum(L.n_units for L in lectures)} units, "
          f"{n_files} pack files, {len(batches)} batches -> {a.out}")
    for b, v in batches.items():
        print(f"  {b}: {len(v):3d} lectures, {sum(x[3] for x in v):5d} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
