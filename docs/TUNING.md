# Step 11: the tuning run

The design (which settings, which values, the rule, the cost) is `docs/TUNING_DESIGN.md`.
This page says how the run is made, where its results are, and how to read the report.

## The pieces

| file | what it is |
|---|---|
| `config/tuning_grid.toml` | every method's values to try, as in the design; `tests/test_tuning.py` checks the counts against the design's section 5.1 table (25,773 combinations; one LLM model, qwen3-30b-a3b-instruct-2507) |
| `scripts/tune.py` | the runner: every combination on each of the 60 tuning lectures, one row per run |
| `scripts/select_settings.py` | the selector: one combination per method, by the design's rule |
| `config/tuned_settings.toml` | the selector's choice per method (written; read back by the runner for the consensus stage) |
| `cluster/11a`-`11d` | the trial, the array of the non-LLM methods, the LLM process, merge + select |

Only the tuning set is read: the runner refuses any other set and any lecture outside it,
skips the test lectures' lines of the prepared corpus files without parsing them, and
reads matrices only from `data/similarity/tuning/tables.sqlite3`. The one exception is
step 13's final run, `--set test --settings config/tuned_settings.toml`: the tuned
combination of each method only, on the test set, into `data/final/`
(`docs/FINAL_RUN.md`).

## How to run

From the repository root, in this order (every step can be resubmitted: a rerun skips
the runs already done and retries the ones that crashed or whose LLM call failed).

    sbatch cluster/11a_tune_trial.sbatch                   # trial: 2 lectures, timed
    python scripts/tune.py plan --name matrix --tasks 20   # spread (a) over 20 tasks
    sbatch --array=0-19%20 --export=ALL,PLAN=matrix cluster/11b_tune_matrix.sbatch   # (a)
    sbatch --export=ALL,STAGE=llm cluster/11c_tune_llm.sbatch                        # (b)
    # when (a) and (b) have ended:
    sbatch cluster/11d_tune_select.sbatch                  # merge, select, status
    sbatch --export=ALL,STAGE=consensus cluster/11c_tune_llm.sbatch                  # (c)
    # when (c) has ended:
    sbatch cluster/11d_tune_select.sbatch                  # merge, final selection

- (a) `matrix`: the 22 non-LLM base methods and mbr, as array tasks; the plan balances
  (method, lecture) items by the trial's time fit (`data/tuning/trial/fit.json`).
- (b) `llm`: llm_single_call, llm_monotone, llm_monotone_constrained, hierarchical
  (refinement on before off), in one process: the LLM cache has a single writer and
  one shared client keeps the request rate of `config/llm.toml` (20 per minute).
  The answer shapes in the prompts of the first three show letters, not numbers
  (`{"starts": [0, s1, s2, ...]}`), because the model copied the numbers of the first
  version; those three and ensemble were rerun with them.
- (c) `consensus`: score_fusion, ensemble, llm_rerank_kbest and fusion, after their
  members are chosen; each member runs at its tuned settings on its tuned measure, and
  fusion follows hierarchical's tuned settings.

`python scripts/tune.py status` prints the runs done per method and the LLM totals
(paid in the run, cost of the answers used, and the failures apart).

## Where the results are

| path | content |
|---|---|
| `data/tuning/runs/<method>.csv.gz` | one row per (method, combination, lecture); the merged results |
| `data/tuning/shards/*.sqlite3` | the rows of writers not merged yet (one file per writer; `merge` removes them) |
| `data/tuning/plans/<name>.json` | the array plan: which (method, lecture) items each task runs |
| `data/tuning/trial/` | the trial's rows, the time fit and the estimate of the full run |
| `data/tuning/selection_report.csv` | the selection report (below) |

A row: `combo` (the combination as JSON: measure, model, cfg, settings), the lecture,
`n_units`, `n_correct`, `accuracy` (true slide predicted; -1 is always wrong), `status`
(`ok`, `crash`, or `llm_failed`: a call that never answered; `error` holds the message),
`compute_s` (the time of `run_method` on matrices in memory, less the time inside
`LLM.ask`), the LLM cost paid in this run (`calls`, `call_seconds`, `prompt_tokens`,
`completion_tokens`), the cost of the answers used (`used_calls`, `used_seconds`,
`used_prompt_tokens`, `used_completion_tokens`; a cached answer brings its stored cost),
and the failures, never cost (`failed_attempts`, `failed_calls`, `errors`).

## How to read the selection report

One row per method. `status` is `chosen`, or `incomplete` when not every combination has
a row for all 60 lectures (then the method is not in `config/tuned_settings.toml`).

| column | meaning |
|---|---|
| `measure`, `model`, `settings` | the chosen combination: measure, model (LLM methods; always qwen3-30b-a3b-instruct-2507) and the tuned cfg and own settings |
| `tuning_acc` | its mean accuracy over the 60 tuning lectures, in % (each lecture counts once; a run not ok counts 0) |
| `default_acc`, `gain` | the same at the defaults (hybrid, DPConfig's and the method's defaults, the configured model), and the difference in points. A consensus method's default keeps its members tuned |
| `n_combos`, `n_tied` | the grid's size, and how many combinations were within 0.5 points of the best |
| `mean_compute_s`, `mean_used_calls`, `mean_used_seconds`, `mean_used_*_tokens` | the chosen setting's mean cost per lecture |
| `not_ok_runs`, `not_ok_runs_grid` | crashed or failed runs of the choice, and of the whole grid |
| `edge` | `EDGE:` and each tuned numeric setting whose chosen value is the smallest or largest tried (two-valued settings are always at an end). Categorical settings are not flagged. The grids are not widened |
| `combo` | the chosen combination as JSON |

Among tied combinations the cheaper wins (LLM: fewer used calls, then less summed
duration; non-LLM: the smaller of mbr `k`, wbs2 `n_intervals`, multiscale_dtw `radius`,
jump_dtw `max_jump`), then the one with fewer settings away from the defaults, then the
higher accuracy, then the earlier combination in the grid.
