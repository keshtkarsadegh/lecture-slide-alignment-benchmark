#!/usr/bin/env python3
"""Step 6: the categories, one level per lecture on each axis.

It reads the attributes table (data/attributes/document_attributes.csv, step 5) and
writes data/categories/. The axes, in plain words (docs/CATEGORIES.md has the rules):

  corpus        the corpus, as is (5 levels)
  text_source   the lecture's own text source, as is: pdf_text_layer,
                corpus_corrected_ocr, tesseract_ocr (so the one scanned MaViLS deck is
                tesseract_ocr)
  length        n_slides       short / medium / long     (terciles)
  images        image_pct      low / medium / high       (terciles)
  tables        table_pct      none (== 0) / some (> 0)
  noise         noise          low / medium / high       (terciles)
  text_density  text_density   low / medium / high       (terciles)
  discipline    the broad OECD FOS 2007 area of the lecture's field (the first digit of
                the field code), except that the fields listed in own_level (default
                1.2, Computer and information sciences) are levels of their own and the
                rest of their area becomes "<area> (other)"; a level with fewer than
                min_count lectures becomes "other"

The tercile cut points are computed once, over all lectures pooled: q1 and q2 are the
values at the 1/3 and 2/3 positions of the sorted values (the lower empirical quantile,
the rule scripts/split_m3av.py uses for its length strata), so a cut point is always a
value some lecture has and the result does not depend on an interpolation choice. A
lecture is the first level if its value <= q1, the second if q1 < value <= q2, the third
otherwise. Ties at a cut point all go to the lower level, so the groups can be unequal
(n_slides is a whole number and many lectures have 40).

The run stops if a lecture appears twice, a text source is not one of the three, or a
discipline is not an OECD FOS 2007 field (the column is empty while step 5's discipline
step is blocked).

Output, in data/categories/:
    document_categories.csv   corpus, lecture_id, one column per axis
    cut_points.json           the cut points, the discipline areas and merges, the
                              input's sha256 and the settings
    level_counts.csv          axis, level, n, and n per corpus

Usage:
    python scripts/categorise.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "categories.toml"
IN_PATH = REPO_ROOT / "data" / "attributes" / "document_attributes.csv"
OUT_DIR = REPO_ROOT / "data" / "categories"
CORPORA = ("mavils", "m3av_a", "m3av_b", "mcd", "slide_highlight")

AXES = ("corpus", "text_source", "length", "images", "tables", "noise", "text_density",
        "discipline")
TERCILE_AXES = ("length", "images", "noise", "text_density")
TABLE_LEVELS = ("none", "some")
OTHER = "other"


# --------------------------------------------------------------------------- the rules

def tercile_cuts(values: Sequence[float]) -> tuple[float, float]:
    """(q1, q2): the values at the 1/3 and 2/3 positions of the sorted values."""
    v = sorted(values)
    if not v:
        raise ValueError("no values")
    n = len(v)
    return v[math.ceil(n / 3) - 1], v[math.ceil(2 * n / 3) - 1]


def tercile_level(value: float, cuts: tuple[float, float], levels: Sequence[str]) -> str:
    """levels[0] if value <= q1, levels[1] if q1 < value <= q2, levels[2] otherwise."""
    return levels[0] if value <= cuts[0] else levels[1] if value <= cuts[1] else levels[2]


def table_level(table_pct: float) -> str:
    """none if no table box covers any of the lecture's slides, some otherwise."""
    return TABLE_LEVELS[0] if table_pct == 0 else TABLE_LEVELS[1]


def read_fos(path: Path) -> dict[str, str]:
    """Field name -> field code (config/oecd_fos2007.csv)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["field"]: r["code"] for r in csv.DictReader(fh)}


def _own_fields(fos: dict[str, str], own_codes: Sequence[str]) -> dict[str, str]:
    """Field code -> field name for the codes that are levels of their own."""
    by_code = {code: name for name, code in fos.items()}
    unknown = [c for c in own_codes if c not in by_code]
    if unknown:
        raise ValueError(f"own_level codes not in the FOS list: {unknown}")
    return {c: by_code[c] for c in own_codes}


def field_level(field: str, fos: dict[str, str], areas: dict[str, str],
                own_codes: Sequence[str] = ()) -> str:
    """The discipline level of a FOS field, before the min_count merge.

    Normally the area named by the first digit of the field's code. A field listed in
    own_codes is its own level (named by the field), and the other fields of its area
    become "<area> (other)": one field can dominate an area (742 of the 919 Natural
    sciences lectures are Computer and information sciences), and the area alone would
    then mostly measure that field.
    """
    if field not in fos:
        raise ValueError(f"discipline {field!r} is not an OECD FOS 2007 field")
    own = _own_fields(fos, own_codes)
    code = fos[field]
    if code in own:
        return field
    digit = code.split(".")[0]
    if any(c.split(".")[0] == digit for c in own):
        return f"{areas[digit]} (other)"
    return areas[digit]


def discipline_levels(fos: dict[str, str], areas: dict[str, str],
                      own_codes: Sequence[str] = ()) -> list[str]:
    """Every level field_level can give, in code order (an area's own fields first,
    then the rest of the area)."""
    own = _own_fields(fos, own_codes)
    out = []
    for digit, area in areas.items():
        mine = [name for c, name in own.items() if c.split(".")[0] == digit]
        out += mine + ([f"{area} (other)"] if mine else [area])
    return out


def merge_small(labels: Sequence[str], min_count: int) -> tuple[list[str], dict[str, int]]:
    """Each label, or "other" if fewer than min_count lectures carry it; and the merged
    labels with their counts.

    Small levels are merged rather than kept because a category with a handful of
    lectures cannot be scored reliably, and the merge is decided once, on all lectures.
    """
    counts: dict[str, int] = {}
    for a in labels:
        counts[a] = counts.get(a, 0) + 1
    merged = {a: n for a, n in counts.items() if n < min_count}
    return [OTHER if a in merged else a for a in labels], merged


# --------------------------------------------------------------------------- the table

def read_attributes(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def categorise(rows: Sequence[dict], cfg: dict,
               fos: dict[str, str]) -> tuple[list[dict], dict]:
    """The category rows (in the input's order) and the record of how they were made."""
    keys = [(r["corpus"], r["lecture_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("a lecture appears more than once")
    sources = cfg["text_source"]["levels"]
    for r in rows:
        if r["corpus"] not in CORPORA:
            raise ValueError(f"unknown corpus {r['corpus']!r}")
        if r["text_source"] not in sources:
            raise ValueError(f"{r['corpus']}/{r['lecture_id']}: text source "
                             f"{r['text_source']!r} not in {sources}")
    out = [{"corpus": r["corpus"], "lecture_id": r["lecture_id"],
            "text_source": r["text_source"]} for r in rows]

    terciles = cfg["terciles"]
    if sorted(terciles) != sorted(TERCILE_AXES):
        raise ValueError(f"[terciles] must define exactly {TERCILE_AXES}")
    cut_points = {}
    for axis in TERCILE_AXES:
        spec = terciles[axis]
        vals = [float(r[spec["column"]]) for r in rows]
        cuts = tercile_cuts(vals)
        cut_points[axis] = {"column": spec["column"], "q1": cuts[0], "q2": cuts[1],
                            "levels": list(spec["levels"])}
        for o, v in zip(out, vals):
            o[axis] = tercile_level(v, cuts, spec["levels"])
    for o, r in zip(out, rows):
        o["tables"] = table_level(float(r[cfg["tables"]["column"]]))

    dcfg = cfg["discipline"]
    own_codes = list(dcfg["own_level"])
    raw = [field_level(r["discipline"], fos, dcfg["areas"], own_codes) for r in rows]
    merged_labels, merged = merge_small(raw, int(dcfg["min_count"]))
    for o, a in zip(out, merged_labels):
        o["discipline"] = a
    area_counts = {name: raw.count(name)
                   for name in discipline_levels(fos, dcfg["areas"], own_codes)}
    out = [{k: o[k] for k in columns()} for o in out]

    record = {
        "tercile_rule": "q1, q2 = sorted values at positions ceil(n/3), ceil(2n/3) "
                        "(1-based), over all lectures pooled; level 1 if value <= q1, "
                        "level 2 if q1 < value <= q2, level 3 otherwise",
        "cut_points": cut_points,
        "tables": {"column": cfg["tables"]["column"],
                   "rule": "none if table_pct == 0, some if > 0"},
        "discipline": {"min_count": int(dcfg["min_count"]),
                       "own_level": own_codes,
                       "level_counts_before_merge": area_counts,
                       "merged_into_other": merged},
    }
    return out, record


def columns() -> list[str]:
    """The columns of document_categories.csv: the lecture, then one per axis."""
    return ["corpus", "lecture_id", *AXES[1:]]


def axis_levels(cfg: dict, fos: dict[str, str], cats: Sequence[dict]) -> dict[str, list[str]]:
    """Every axis's levels in a fixed order; discipline lists the levels that are left
    after the merge (in code order) and then "other" if any lecture is in it."""
    present = {c["discipline"] for c in cats}
    disc = [a for a in discipline_levels(fos, cfg["discipline"]["areas"],
                                         cfg["discipline"]["own_level"]) if a in present]
    if OTHER in present:
        disc.append(OTHER)
    levels = {"corpus": list(CORPORA), "text_source": list(cfg["text_source"]["levels"]),
              "tables": list(TABLE_LEVELS), "discipline": disc,
              **{axis: list(cfg["terciles"][axis]["levels"]) for axis in TERCILE_AXES}}
    return {axis: levels[axis] for axis in AXES}


def level_counts(cats: Sequence[dict], cfg: dict, fos: dict[str, str]) -> list[dict]:
    """One row per axis and level: n lectures, then n per corpus (zero rows included)."""
    rows = []
    for axis, levels in axis_levels(cfg, fos, cats).items():
        for level in levels:
            sub = [c for c in cats if c[axis] == level]
            rows.append({"axis": axis, "level": level, "n": len(sub),
                         **{c: sum(s["corpus"] == c for s in sub) for c in CORPORA}})
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build(cfg: dict, in_path: Path = IN_PATH,
          repo_root: Path = REPO_ROOT) -> tuple[list[dict], dict, list[dict]]:
    fos = read_fos(repo_root / cfg["discipline"]["fos_list"])
    rows = read_attributes(in_path)
    cats, rec = categorise(rows, cfg, fos)
    try:
        shown = str(in_path.resolve().relative_to(repo_root))
    except ValueError:
        shown = str(in_path)
    record = {"input": {"path": shown, "rows": len(rows), "sha256": sha256_file(in_path)},
              **rec, "settings": cfg}
    return cats, record, level_counts(cats, cfg, fos)


def write(cats: Sequence[dict], record: dict, counts: Sequence[dict], cfg: dict,
          out_dir: Path = OUT_DIR) -> Path:
    """No timestamp in any file: the same input and settings give byte-identical output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "document_categories.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns())
        w.writeheader()
        w.writerows(cats)
    with open(out_dir / "cut_points.json", "w") as fh:
        json.dump(record, fh, indent=1)
    with open(out_dir / "level_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["axis", "level", "n", *CORPORA])
        w.writeheader()
        w.writerows(counts)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    cats, record, counts = build(cfg)
    path = write(cats, record, counts, cfg)
    print(f"{len(cats)} lectures -> {path}")
    for axis, c in record["cut_points"].items():
        print(f"  {axis} ({c['column']}): q1 {c['q1']:g}, q2 {c['q2']:g}")
    print(f"  discipline merged into {OTHER!r}: {record['discipline']['merged_into_other']}")
    print("axis, level, n, " + ", ".join(CORPORA))
    for r in counts:
        print(f"  {r['axis']}, {r['level']}, {r['n']}, "
              + ", ".join(str(r[c]) for c in CORPORA))
    return 0


if __name__ == "__main__":
    sys.exit(main())
