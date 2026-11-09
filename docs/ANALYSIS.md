# Step 14: the analysis

`scripts/analysis.py` reads the final run of step 13 (`data/final/methods/<method>.csv.gz`
and `data/final/summary.csv`, docs/FINAL_RUN.md) and the lecture categories of step 6
(`data/categories/`, docs/CATEGORIES.md), and answers three questions about the 31
methods on the 140 test lectures. It runs no method and pays no LLM call. It only
describes: mean accuracy, rank and cost.

Accuracy is a lecture's share of units on their true slide; a method's accuracy on a set
of lectures is the mean over those lectures, in %. Rank 1 is the highest accuracy;
methods with the same accuracy share a rank and the next rank is skipped (1, 2, 2, 4).

## The tables (`data/analysis/`)

| file | what |
|---|---|
| `q1_ranking.csv`, `.md` | **Q1, ranking with cost.** The 31 methods by mean test accuracy, with rank, kind and cost. Cost by kind: non-LLM: compute time and table build time; LLM: LLM calls, LLM seconds and tokens (only successful calls, only the successful attempt), and build time; mixed: all of these. Each as a total over the 140 lectures and per lecture. The costs are step 13's: each method as if it ran alone (a combined method carries its members' cost; a shared table counts once per method; the build total includes the model load times). '-' (empty in the CSV): the kind has no such cost. |
| `q2_<axis>_long.csv`, `q2_<axis>_wide.csv`, `q2_levels.md` | **Q2, every method within each category level.** One table per axis (text_source, length, images, tables, noise, text_density, discipline; corpus is not used). For every level: n test lectures, and each method's mean accuracy over those lectures and its rank among the 31 within the level. Long: one row per (level, method). Wide: one row per method, an accuracy and a rank column per level. The markdown adds a last row, the mean over the methods per level. |
| `q3_best_worst.csv`, `.md` | **Each method's best and worst category level** (the files keep the name Q3, but this is not the paper's Q3). Over all levels of all axes: best = the level where the method's rank is smallest, worst = where it is largest. A tie in rank goes to the higher accuracy (best) or the lower accuracy (worst); a tie in both lists every tied level. The rank and the accuracy there, and the method's overall rank and accuracy. The paper's Q3 is the other view of the Q2 rows: at every level of every attribute, the method with the highest and the one with the lowest accuracy (`results/tables/tab_q3.tex`, step 17). |
| `checks.txt` | the checks below and their results |

## The checks

The script stops with exit code 1, and writes no table, when one fails:

1. the 31 methods are the per-lecture tables and the summary's rows;
2. every method has the same test lectures, one row each, status ok;
3. every test lecture has exactly one level on each axis, a step-6 level;
4. Q1's accuracy equals the summary's `test_accuracy`, and Q1's costs its cost totals and
   per-lecture costs;
5. for every method and axis, the lecture-weighted average of the level means equals the
   overall mean;
6. every step-6 level has test lectures and all 31 methods.

Ties are decided on the accuracy rounded to 9 decimals, so two equal means summed in a
different order are not split by their last bit.

## Rerun

From the repository root (light; the login node is fine):

    python scripts/analysis.py

Tests: `tests/test_analysis.py` (made-up methods and lectures: ranks and shared ranks, the
costs by kind, the means and ranks within levels, Q3's tie rules, and every check failing
on a broken input).

# Step 16: Q4, important units placed on the wrong slide

Q4 asks how much of what matters each method gets wrong. The importance labels are step
15's (`data/importance/labels.csv.gz`; label 2 = core = important; docs/IMPORTANCE.md).
Claude assigned the importance labels; our scripts did the evaluation. The labelling is
not part of any method's cost.

The number, one per method and column: the important units the method placed on the
wrong slide, divided by all units of the documents in the column, times 100. It is
pooled over the units, not a mean of per-lecture values. A unit with no slide counts as
wrong.

Nine columns, in this order: all documents; length short, medium, long; images low,
medium, high; tables none, some (the step-6 levels). Rows: the 31 methods, the lowest
"all documents" value first.

The slide each unit was given comes from `scripts/final_predictions.py`, an offline
replay of the final run (LLM answers from `data/llm_cache.sqlite3` only, no call paid)
into `data/final/predictions/<method>.csv.gz`, which reproduces step 13's correct and
unit counts exactly.

## Output (`data/analysis/`)

| file | what |
|---|---|
| `q4_wrong_important.csv` | one row per method: per column the important units on the wrong slide, the column's units and the percentage |
| `q4_wrong_important.md` | the nine percentages (1 decimal), each column headed with its unit count |
| `q4_heatmap.pdf`, `.png` | the same as a heatmap: column groups all, length, images, tables; Reds on a fixed 0-45 % scale |
| `q4_checks.txt` | the checks below |

## The checks

The script stops with exit code 1, and writes no table, when one fails:

1. every unit has exactly one label and one prediction per method, and the true slides
   agree;
2. per method, the wrong units recounted from the predictions equal the sum of
   `n_units - n_correct` in `data/final/methods/<method>.csv.gz`;
3. the unit totals of the length levels, of the image levels and of the table levels
   each sum to all units (18,315).

## Rerun

On the cluster, from the repository root (the replay runs every method again, so it is a
job; it pays no call):

    sbatch cluster/16_q4.sbatch

or the two steps by hand: `python scripts/final_predictions.py`, then
`python scripts/analysis_q4.py` (light; the login node is fine for this one).

Tests: `tests/test_analysis_q4.py` (made-up lectures, labels and methods: the
percentage, the sort order, a column without units, a missing label; and the replay
answering from the cache only).

# Step 17: the paper's tables and figures

`scripts/paper_assets.py` makes every table and figure of the paper from the results: Q1,
Q2 and Q4 (`data/analysis/`), the per-lecture tables of the final run
(`data/final/methods/`), the example lecture's units (`data/prepared/`, steps 1-2, not in
git), the categories and their cut points (`data/categories/`), the
selection (`data/selection/`), the slides per lecture (`data/attributes/`), the tuned
settings (`config/tuned_settings.toml`), the prompts (`methods/llm.py`) and the registry
(`methods/`). It runs no method and pays no LLM call.

The numbers not in an earlier table are counts and shares of results that are in git,
written as LaTeX macros into `tables/numbers.tex` so the text quotes none by hand:

- **back-jumps** of the true path, from the true slides of `data/final/predictions/`: the
  test lectures in which it returns to an earlier slide at least once, and the steps
  between consecutive units that go back, as a share of all steps and of the slide
  changes;
- **Q4 against importance-blind errors** (`tables/tab_q4_expected.tex`): per method the
  pooled error rate from the predictions, times the core share of all units (step 15's
  labels), against Q4's all-lectures percentage, and their ratio;
- **LLM fallbacks**: per LLM method the unusable answers (`parse_failed`) and the answers
  cut off (`truncated`) of step 13's per-lecture tables;
- **tuning against test** (`tables/tab_tuning.tex`): per method the highest tuning
  accuracy of any combination (step 11's runs, scored as step 11 scores them), the tuned
  combination's, and its test accuracy; also the number of combinations tried, and the
  chosen combination's tuning accuracy reweighted to the test set's shares of short,
  medium and long lectures (how much of the drop the length mix explains);
- **text numbers** (`text_numbers`): plain differences, ratios and shares of the Q1, Q2
  and Q4 results the text quotes (the LLM-alone methods' cost and accuracy gap, the top
  band's end, the image table's share of the cost, the picture-share drop, the share of
  units in long lectures) and the DP defaults of the tie rule.
- **accuracy per dataset** (`tables/tab_datasets.tex`): every method's mean accuracy over
  the test lectures of each dataset, from step 13's per-lecture tables.
- **bootstrap intervals** (`tables/tab_bootstrap.tex`): 1,000 resamples of the test
  lectures with replacement (seed 20261011, the project seed), every method averaged over
  the same resamples; per method the 2.5 and 97.5 percentiles of its mean and of rank 1's
  mean minus its own (a paired difference), from step 13's per-lecture tables;
- **paired tests** (`tables/tab_significance.tex`): per method a two-sided paired t-test
  of rank 1's per-lecture accuracy minus its own over the test lectures, with Holm's
  correction over the 30 comparisons at the 5 % level (the same per-lecture tables);
- **error distance** (`tables/tab_severity.tex`): per method, pooled over the test units
  of `data/final/predictions/`, the wrong units as a share of all units, the share of
  the wrong units one slide from their true slide, and their mean distance in slides;
- **preparation counts** (`tables/tab_prepare.tex`, Appendix B): per dataset, from step
  2's `data/prepared/summary.csv` and `<corpus>_dropped.csv` (README, Step 2, says what
  they count), the lectures in the corpus, dropped per reason, kept and with the
  transcript cut, and the units without a true slide, after the cut and kept. The M3AV
  talks in neither half, which both halves' dropped files list, are one row of their own,
  counted once. Check 12 stops the run if the files disagree or the kept lectures are not
  Table 1's pool; the macros `\PrepCut` and `\PrepPool` come from this table. Without
  `summary.csv` (`--prepared` names another folder) the table is a placeholder line and
  the two macros are not written; every other table and figure is still written.

The other number not in an earlier table is the **cost per category** (the cost heatmap and
Q3's seconds): per lecture the seconds Q1 counts for the method's kind (compute for non-LLM
and mixed, table build for all, LLM seconds for LLM and mixed), averaged over the
category's lectures. Q1's build total adds each model load once and the per-lecture rows
hold none, so each method's load is spread evenly, load / 140 on every lecture; the
all-documents column is then Q1's total seconds per lecture exactly (check 3).

**Names.** Every table and figure names the methods and the category levels through
`scripts/paper_names.py`. Three names say what the method does rather than echo its id:
`dp` is "Viterbi DP" (it can move back, so not "monotone"), `llm_rerank_kbest` is "LLM
re-rank (5 methods)" (its candidates are five methods' alignments, not dtw's k best paths,
since step 3), and `fusion` is "LLM-prior DP" (dp with the hierarchical LLM answer as its
positional prior).

**One heatmap.** The three heatmaps here and Q4's own (`scripts/analysis_q4.py`) are drawn
by `scripts/heatmap.py`, so the approved layout is defined once.

## Output (`results/`, in git)

| file | what |
|---|---|
| `tables/tab_corpora.tex` | per corpus: source, slide-text source, lectures in the pool, tuning and test, test units, mean slides and units per test lecture; totals |
| `tables/tab_categories.tex` | the 7 attributes: how each is measured, its levels with their cut rule, test lectures per level |
| `tables/tab_methods.tex` | the 31 methods by family (longtable): kind, similarity tables read, members, tuned settings |
| `tables/tab_q1.tex` | Q1: rank, method, kind, accuracy, and per lecture compute s, tables s, LLM calls, LLM s, total s |
| `tables/tab_q3.tex` | Q3: for all documents and each of the 22 levels, the highest and the lowest method with its accuracy and seconds per lecture |
| `tables/tab_q4.tex` | Q4's nine columns as numbers (appendix) |
| `tables/tab_q2_full.tex` | Q2: every method's accuracy in all 22 levels (appendix, landscape) |
| `tables/tab_measures.tex` | accuracy per similarity measure on the tuning lectures, and each table's cost |
| `tables/tab_q4_expected.tex` | Q4 against importance-blind errors: wrong core units, the expected share, the ratio (appendix) |
| `tables/tab_tuning.tex` | per method the best and the chosen tuning accuracy and the test accuracy (appendix) |
| `tables/tab_datasets.tex` | every method's accuracy per dataset (appendix) |
| `tables/tab_bootstrap.tex` | per method the bootstrap interval of its mean accuracy and of its difference to rank 1 (appendix) |
| `tables/tab_significance.tex` | per method the paired t-test against rank 1 over the test lectures, Holm-corrected (appendix) |
| `tables/tab_severity.tex` | per method the wrong units, the share one slide off and their mean distance (appendix) |
| `tables/tab_prepare.tex` | per dataset the lectures dropped, kept and cut and the units removed and kept by step 2 (appendix) |
| `tables/prompts.tex` | the prompts of the final run, verbatim from `methods/llm.py` (appendix) |
| `tables/numbers.tex` | the numbers the text quotes, as LaTeX macros |
| `figures/fig_q1_scatter.pdf` | accuracy against total seconds per lecture (log), by kind, each point its Q1 rank; a zoom panel for the dense column near 23 s |
| `figures/fig_q2_accuracy.pdf` | heatmap, the 31 methods in Q1's order x the nine columns (all documents; length; images; tables): accuracy, Blues 20-80 |
| `figures/fig_q2_cost.pdf` | the same layout: seconds per lecture, Oranges 0-70 (a cell above 70 is clipped to the darkest colour and keeps its true number) |
| `figures/fig_q4_heatmap.pdf` | Q4's heatmap with the paper's names, its own order |
| `figures/fig_pipeline.pdf` | the benchmark's pipeline |

The figures are 6.5 in
wide, with every text at least 7 pt and TrueType fonts in the PDFs.

## The checks

The script stops with exit code 1, and writes nothing, when one fails:

1. Q1, every Q2 level, Q4 and the per-lecture tables hold the 31 methods of `methods/`,
   and every method and level has a display name;
2. the per-lecture tables hold exactly the test set of the selection;
3. per method, the all-documents cost per lecture equals Q1's total seconds per lecture;
4. the test units of the corpora table add up to Q4's units of all documents;
5. (the measure table) every method that tunes the measure has runs for all eight, its
   chosen measure is within step 11's tie of its best, and each measure's build seconds
   equal Q1's Tables column of every method reading it alone;
6. (the unit data) every method's predictions carry the same true slides, its wrong
   units are step 13's, its wrong core units give Q4's all-lectures percentage, there is
   one importance label per test unit, and no true path of MCD or Slide-Highlight goes
   back;
7. (the tuning table) each chosen combination is within 0.5 points of its method's best,
   and its per-lecture runs give step 11's tuning accuracy;
8. (the top band) TOP_BAND is where Q1's accuracy first drops by more than one point
   from one rank to the next;
9. (the dataset table) the per-dataset accuracies average, over the lectures, to Q1's.

## Rerun

From the repository root, after steps 14 and 16 (light; the login node is fine):

    python scripts/paper_assets.py

Tests: `tests/test_paper_assets.py` (the names cover the 31 methods and every level the
categories can give; on made-up lectures: the all-documents cost equals Q1's total, the
cost per level by hand, the check catching a cost without the loads, the tables are sane
LaTeX with the display names and the number formats; the scatter's label spreading; the
back-jump counts, Q4's expectation, the fallback counts, the tuning table, the
reweighting by length, the top band's end, the Q3 rounding mark, the dataset table and
the prompts on made-up data).
