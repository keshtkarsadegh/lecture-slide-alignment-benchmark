# Step 6: the categories

`scripts/categorise.py` gives every lecture of the attributes table
(`data/attributes/document_attributes.csv`, step 5, docs/ATTRIBUTES.md) one level on
each axis. The settings are in `config/categories.toml`.

## The axes

| axis | levels | rule |
|---|---|---|
| `corpus` | mavils, m3av_a, m3av_b, mcd, slide_highlight | as is |
| `text_source` | pdf_text_layer, corpus_corrected_ocr, tesseract_ocr | the lecture's own text source, as is (so the one scanned MaViLS deck is tesseract_ocr) |
| `length` | short / medium / long | `n_slides`, terciles |
| `images` | low / medium / high | `image_pct`, terciles |
| `tables` | none / some | `table_pct` == 0: none; > 0: some |
| `noise` | low / medium / high | `noise`, terciles |
| `text_density` | low / medium / high | `text_density`, terciles |
| `discipline` | Computer and information sciences, Natural sciences (other), and the other broad OECD FOS 2007 areas; `other` | the area of the lecture's field: 1 Natural sciences, 2 Engineering and technology, 3 Medical and health sciences, 4 Agricultural sciences, 5 Social sciences, 6 Humanities (the first digit of the field code in `config/oecd_fos2007.csv`), except that the fields listed in `own_level` (default `["1.2"]`, Computer and information sciences) are levels of their own and the rest of their area becomes "<area> (other)". A level with fewer than `min_count` lectures (20) is merged into `other` |

## The rules

- **Terciles.** The cut points q1 and q2 are computed once, over all lectures pooled
  (not per corpus): the values at the 1/3 and 2/3 positions of the sorted values
  (1-based positions ceil(n/3) and ceil(2n/3)), the lower empirical quantile, the same
  rule as the M3AV split's length strata. A cut point is therefore always a value some
  lecture has, and no interpolation choice enters. A lecture is low (short) if its
  value <= q1, medium if q1 < value <= q2, high (long) otherwise.
- **Ties.** Every lecture at a cut point goes to the lower level, so the groups can be
  unequal: n_slides is a whole number and many lectures sit on the same value.
  `level_counts.csv` gives the sizes.
- **Discipline split.** Computer and information sciences (1.2) holds 742 of the 919
  Natural sciences lectures, so the area alone would mostly measure that one field: 1.2
  is its own level and the other area-1 fields are "Natural sciences (other)". The list
  of such field codes is the setting `own_level`.
- **Discipline merge.** Decided once, on all lectures, after the split: a level with
  fewer than `min_count` lectures becomes `other`. Levels no lecture has do not appear.
- **Checks.** The run stops if a lecture appears twice, a corpus or text source is not
  one of the listed ones, or a discipline is not an OECD FOS 2007 field (empty or
  `unassigned` included).
- **Deterministic.** No randomness and no timestamp: the same input and settings give
  byte-identical files.

## Output

In `data/categories/`:

| file | what |
|---|---|
| `document_categories.csv` | one row per lecture: corpus, lecture_id, text_source, length, images, tables, noise, text_density, discipline |
| `cut_points.json` | q1 and q2 per tercile axis, the `own_level` codes, the discipline level counts before the merge and the levels merged into `other`, the input's path, row count and sha256, the settings |
| `level_counts.csv` | axis, level, n lectures, and n per corpus |

By hand (light; the login node is fine):

    python scripts/categorise.py

Tests: `tests/test_categorise.py` (tercile edges and ties, tables none/some, the field
areas, the own-level split and the merge into `other`, the checks, the counts, determinism).
