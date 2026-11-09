#!/usr/bin/env python3
"""Step 15: merge the importance labels of the 140 test lectures, and their agreement.

The labelling agents (docs/IMPORTANCE.md) write one CSV per batch, header
`corpus,lecture_id,unit_index,label`, one row per unit; those files are kept in
data/importance/raw/. This script checks them against the prepared test lectures
(scripts/importance_pack.py reads them the same way the packs were made) and writes
the annotation:

  data/importance/labels.csv.gz    corpus, lecture_id, unit_index, true_slide, label;
                                   selection order, then unit order; gzip without a
                                   time stamp, so the same labels give the same bytes
  data/importance/labels_meta.json who labelled (Claude Opus 5.5),
                                   the guideline and its sha256, the labelling date,
                                   the raw files and their sha256, the counts

THE CHECKS (any failure: the problems are printed, nothing is written, exit code 1):
every test unit is labelled exactly once (a second row for a unit is an error even
with the same label), every label is 0, 1 or 2, every row names a unit of a test
lecture, and every file has exactly the header above.

It prints the label counts overall and per corpus, and each lecture's share of 2s.

AGREEMENT (`--agreement A B`). A and B are two labellings (a file or a folder of CSVs
each), e.g. the full labelling and the second labelling of 10 lectures. Over the
lectures that both label (each must label the same units of such a lecture), per
lecture and overall (all their units pooled): exact agreement (%), agreement within
one level (%), and Cohen's weighted kappa with linear and with quadratic weights
(empty when it is undefined: both labellings use one and the same label throughout).
Written to data/importance/agreement.csv (the last row, corpus ALL, is the overall).

    python scripts/importance_merge.py [--raw DIR] [--date YYYY-MM-DD]
    python scripts/importance_merge.py --agreement A B [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importance_pack as P  # noqa: E402

OUT = REPO_ROOT / "data" / "importance"
RAW = OUT / "raw"
GUIDELINE = REPO_ROOT / "docs" / "IMPORTANCE.md"
LABELLER = "Claude Opus 5.5"
HEADER = ["corpus", "lecture_id", "unit_index", "label"]
LABELS = (0, 1, 2)
#: Problems printed before the rest are only counted.
SHOW = 30

Unit = tuple[str, str, int]  # (corpus, lecture_id, unit_index)


# --------------------------------------------------------------------------- #
# Reading the agents' CSVs
# --------------------------------------------------------------------------- #

def label_files(path: Path) -> list[Path]:
    """A CSV file, or every *.csv in a folder (sorted)."""
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


def read_labels(path: Path) -> tuple[dict[Unit, int], list[str]]:
    """The labels of a file or folder, and the problems found: a wrong header, a unit
    index that is not an integer, a label not in 0/1/2, a unit given twice. A unit
    with a problem has no label in the result."""
    labels: dict[Unit, int] = {}
    seen: Counter = Counter()
    problems: list[str] = []
    files = label_files(path)
    if not files:
        problems.append(f"{path}: no label CSV")
    for f in files:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if not rows or [c.strip() for c in rows[0]] != HEADER:
            problems.append(f"{f.name}: the header is not {','.join(HEADER)}")
            continue
        for n, row in enumerate(rows[1:], start=2):
            if not any(c.strip() for c in row):
                continue
            if len(row) != len(HEADER):
                problems.append(f"{f.name}:{n}: {len(row)} fields, not {len(HEADER)}")
                continue
            corpus, lid, idx, lab = (c.strip() for c in row)
            try:
                unit = (corpus, lid, int(idx.removeprefix("u")))
            except ValueError:
                problems.append(f"{f.name}:{n}: unit_index {idx!r} is not an integer")
                continue
            seen[unit] += 1
            if seen[unit] > 1:
                problems.append(f"{f.name}:{n}: {unit} labelled more than once")
                labels.pop(unit, None)
                continue
            if lab not in {str(x) for x in LABELS}:
                problems.append(f"{f.name}:{n}: label {lab!r} is not 0, 1 or 2")
                continue
            labels[unit] = int(lab)
    return labels, problems


def check_complete(labels: dict[Unit, int], lectures: Sequence[P.Lecture]) -> list[str]:
    """Problems: test units without a label, labelled units that are not test units."""
    expected = {(L.corpus, L.lecture_id, i) for L in lectures for i, _, _ in L.units}
    missing = sorted(expected - set(labels))
    unknown = sorted(set(labels) - expected)
    out = [f"not labelled: {u}" for u in missing]
    out += [f"not a test unit: {u}" for u in unknown]
    return out


# --------------------------------------------------------------------------- #
# The annotation
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_rows(labels: dict[Unit, int], lectures: Sequence[P.Lecture]) -> list[dict]:
    return [{"corpus": L.corpus, "lecture_id": L.lecture_id, "unit_index": i,
             "true_slide": k, "label": labels[(L.corpus, L.lecture_id, i)]}
            for L in lectures for i, k, _ in L.units]


def write_labels(rows: Sequence[dict], path: Path) -> None:
    """gzip with mtime 0 and no file name: the same labels give the same bytes."""
    cols = ["corpus", "lecture_id", "unit_index", "true_slide", "label"]
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                                mtime=0) as gz:
        text = "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols)
                                             for r in rows]) + "\n"
        gz.write(text.encode("utf-8"))


def summary(rows: Sequence[dict]) -> list[str]:
    """Label counts overall and per corpus, and each lecture's share of 2s."""
    def counts(rs: Sequence[dict]) -> str:
        c = Counter(r["label"] for r in rs)
        return "  ".join(f"{x}: {c[x]:6d} ({100 * c[x] / len(rs):5.1f} %)" for x in LABELS)
    lines = [f"all {len(rows)} units   {counts(rows)}"]
    for corpus in sorted({r["corpus"] for r in rows}):
        rs = [r for r in rows if r["corpus"] == corpus]
        lines.append(f"{corpus:16s} {len(rs):6d}   {counts(rs)}")
    lines += ["", "share of 2s per lecture:"]
    lectures: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        lectures.setdefault((r["corpus"], r["lecture_id"]), []).append(r["label"])
    for (c, l), labs in lectures.items():
        lines.append(f"  {c:16s} {l:40s} {len(labs):5d} units  "
                     f"{100 * labs.count(2) / len(labs):5.1f} % core")
    return lines


def merge(raw: Path, lectures: Sequence[P.Lecture], out: Path = OUT, *,
          date: str, guideline: Path = GUIDELINE) -> tuple[list[str], list[str]]:
    """Check the raw labels and write labels.csv.gz and labels_meta.json into `out`.
    Returns (summary lines, problems); with problems nothing is written."""
    labels, problems = read_labels(raw)
    problems += check_complete(labels, lectures)
    if problems:
        return [], problems
    rows = label_rows(labels, lectures)
    out.mkdir(parents=True, exist_ok=True)
    write_labels(rows, out / "labels.csv.gz")

    def rel(p: Path) -> str:
        p = p.resolve()
        return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
    meta = {"labeller": LABELLER,
            "guideline": rel(guideline),
            "guideline_sha256": sha256_file(guideline) if guideline.exists() else None,
            "date": date,
            "levels": {"0": "filler", "1": "supporting", "2": "core"},
            "n_lectures": len(lectures), "n_units": len(rows),
            "label_counts": {str(x): sum(1 for r in rows if r["label"] == x) for x in LABELS},
            "raw_files": [{"file": rel(f), "sha256": sha256_file(f)}
                          for f in label_files(raw)],
            "labels_sha256": sha256_file(out / "labels.csv.gz")}
    (out / "labels_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return summary(rows), []


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

def weighted_kappa(a: Sequence[int], b: Sequence[int], power: int) -> Optional[float]:
    """Cohen's weighted kappa of two labellings of the same units over the levels 0..2,
    with weight (|i - j| / 2) ** power (1 linear, 2 quadratic): 1 - observed weighted
    disagreement / the one expected from the two labellings' own label shares. None
    when the expected disagreement is 0 (both use one and the same label only)."""
    n = len(a)
    k = len(LABELS)
    w = [[(abs(i - j) / (k - 1)) ** power for j in range(k)] for i in range(k)]
    pa = [a.count(x) / n for x in LABELS]
    pb = [b.count(x) / n for x in LABELS]
    obs = sum(w[x][y] for x, y in zip(a, b)) / n
    exp = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return None if exp == 0 else 1 - obs / exp


def agreement_row(corpus: str, lid: str, a: Sequence[int], b: Sequence[int]) -> dict:
    n = len(a)
    return {"corpus": corpus, "lecture_id": lid, "n_units": n,
            "exact_pct": 100 * sum(x == y for x, y in zip(a, b)) / n,
            "within_one_pct": 100 * sum(abs(x - y) <= 1 for x, y in zip(a, b)) / n,
            "kappa_linear": weighted_kappa(a, b, 1),
            "kappa_quadratic": weighted_kappa(a, b, 2)}


def agreement(la: dict[Unit, int], lb: dict[Unit, int]) -> tuple[list[dict], list[str]]:
    """Per lecture labelled in both (lecture order: sorted), then the overall row.
    A lecture in both whose units differ is a problem."""
    def by_lecture(lab: dict[Unit, int]) -> dict[tuple[str, str], dict[int, int]]:
        out: dict[tuple[str, str], dict[int, int]] = {}
        for (c, l, i), x in lab.items():
            out.setdefault((c, l), {})[i] = x
        return out
    A, B = by_lecture(la), by_lecture(lb)
    common = sorted(set(A) & set(B))
    problems = [f"{c}/{l}: the two labellings label different units"
                for c, l in common if set(A[(c, l)]) != set(B[(c, l)])]
    if not common:
        problems.append("no lecture is labelled in both")
    if problems:
        return [], problems
    rows, all_a, all_b = [], [], []
    for c, l in common:
        idx = sorted(A[(c, l)])
        a = [A[(c, l)][i] for i in idx]
        b = [B[(c, l)][i] for i in idx]
        rows.append(agreement_row(c, l, a, b))
        all_a += a
        all_b += b
    rows.append(agreement_row("ALL", "", all_a, all_b))
    return rows, []


AGREEMENT_COLS = ["corpus", "lecture_id", "n_units", "exact_pct", "within_one_pct",
                  "kappa_linear", "kappa_quadratic"]


def write_agreement(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AGREEMENT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r[c] is None else r[c]) for c in AGREEMENT_COLS})


def fmt(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{x:5.2f}"


# --------------------------------------------------------------------------- #

def report_problems(problems: Sequence[str]) -> None:
    print(f"PROBLEMS: {len(problems)}", file=sys.stderr)
    for p in problems[:SHOW]:
        print(f"   {p}", file=sys.stderr)
    if len(problems) > SHOW:
        print(f"   ... and {len(problems) - SHOW} more", file=sys.stderr)
    print("nothing written", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW, help="the agents' label CSVs")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="the labelling date (default: today)")
    ap.add_argument("--agreement", nargs=2, type=Path, metavar=("A", "B"),
                    help="two labellings (file or folder each): their agreement")
    ap.add_argument("--out", type=Path, default=OUT / "agreement.csv",
                    help="the agreement table (with --agreement)")
    a = ap.parse_args(argv)

    if a.agreement:
        (la, pa), (lb, pb) = read_labels(a.agreement[0]), read_labels(a.agreement[1])
        rows, problems = agreement(la, lb)
        problems = [f"A: {p}" for p in pa] + [f"B: {p}" for p in pb] + problems
        if problems:
            report_problems(problems)
            return 1
        write_agreement(rows, a.out)
        print(f"{'corpus':16s} {'lecture':40s} {'units':>5s}  exact  within1  "
              f"k_lin  k_quad")
        for r in rows:
            print(f"{r['corpus']:16s} {r['lecture_id']:40s} {r['n_units']:5d}  "
                  f"{r['exact_pct']:5.1f}  {r['within_one_pct']:6.1f}   "
                  f"{fmt(r['kappa_linear'])}  {fmt(r['kappa_quadratic'])}")
        print(f"written to {a.out}")
        return 0

    lines, problems = merge(a.raw, P.load_test_lectures(), date=a.date)
    if problems:
        report_problems(problems)
        return 1
    print("\n".join(lines))
    print(f"written to {OUT / 'labels.csv.gz'} and {OUT / 'labels_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
