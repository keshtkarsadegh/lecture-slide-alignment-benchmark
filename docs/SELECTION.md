# Step 7: the lecture selection

`scripts/select_lectures.py` chooses two disjoint sets of lectures from the categories
(`data/categories/document_categories.csv`, step 6, docs/CATEGORIES.md): a tuning set
and a test set. The settings are in `config/selection.toml`.

## The rules

- **Sizes.** Tuning 60 lectures, test 140.
- **Small corpora, whole.** mavils (20), mcd (23) and slide_highlight (14) are taken in
  whole. Tuning gets n/3 rounded half up, test the rest: mavils 7/13, mcd 8/15,
  slide_highlight 5/9.
- **Fill corpora.** The rest of each set comes from m3av_a and m3av_b, as equally as
  whole numbers allow (a remainder goes to m3av_a): tuning 20/20, test 52/51.
- **Minima.** Every level of every axis has at least 5 lectures in the tuning set and
  10 in the test set. The corpus axis is fixed by the counts above and has no minimum
  (slide_highlight has 9 test lectures, the whole corpus less its tuning share).
- **Method.** The pool is put in one seeded random order (seed 20261012; the pool is
  sorted by corpus and lecture first, so the input's row order does not matter). The
  tuning set is filled first, then the test set from the lectures left. Each set is
  filled in two phases, the small corpora first, so their lectures also count towards
  the minima, then the fill corpora. In each phase:
  1. greedy: add the lecture that covers the most unmet level minima of the set (one per
     axis whose level is still short), the earliest in the seeded order on a tie, while
     some lecture covers one and its corpus still has room;
  2. random: fill each corpus's count with its lectures in the seeded order.
- **Checks.** The run stops before selecting if a level has fewer lectures in the pool
  than the two minima add up to, and after a set if a minimum is still unmet. A minimum
  is never relaxed. It also stops if a lecture appears twice or a level is not one of
  step 6's.
- **Deterministic.** The only randomness is the seeded order and no file has a
  timestamp: the same input and settings give byte-identical files.

## Output

In `data/selection/`:

| file | what |
|---|---|
| `selected_lectures.csv` | one row per selected lecture: corpus, lecture_id, set (tuning or test), text_source, length, images, tables, noise, text_density, discipline |
| `selection_counts.csv` | axis, level, n in tuning, n in test, n in the pool |
| `selection.json` | the input's path, row count and sha256, the seed, the settings |

By hand (light; the login node is fine):

    python scripts/select_lectures.py

Tests: `tests/test_select_lectures.py` (disjoint sets, exact sizes, minima, the small
corpora whole with the exact split, the m3av balance, determinism, the stops).
