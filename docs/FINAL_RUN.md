# Step 13: the final run

Every method with its one tuned setting (`config/tuned_settings.toml`, step 11) on the
140 test lectures; accuracy and cost per lecture.

## How it is made

The final run goes through the tuning runner, `scripts/tune.py`, with two options:
`--set test` (the test lectures, their step-12 tables `data/similarity/test/tables.sqlite3`,
and the output folder `data/final/`) and `--settings config/tuned_settings.toml` (each
method's grid cut down to its one tuned combination). Everything else is the code of
tuning, so a method behaves on the test lectures as it did there: the same stages in the
same order, the same accuracy and cost counting, the consensus methods' members at their
tuned settings on their tuned measures, fusion with hierarchical's tuned settings, one
LLM model (qwen3-30b-a3b-instruct-2507) at temperature 0. The runner refuses the test
set without `--settings`, so no grid is ever run on it. Without the two options the
runner tunes exactly as before.

From the repository root:

    python scripts/tune.py plan --set test --settings config/tuned_settings.toml --name matrix --tasks 4
    sbatch --array=0-3 --export=ALL,PLAN=matrix,SET=test,SETTINGS=config/tuned_settings.toml cluster/11b_tune_matrix.sbatch
    sbatch --export=ALL,STAGE=llm,SET=test,SETTINGS=config/tuned_settings.toml cluster/11c_tune_llm.sbatch
    # when the llm stage has ended (its answers are the members' of ensemble and fusion):
    sbatch --export=ALL,STAGE=consensus,SET=test,SETTINGS=config/tuned_settings.toml cluster/11c_tune_llm.sbatch
    # when all of them have ended: merge, status, tables and checks
    sbatch cluster/13_final_report.sbatch

- the non-LLM methods (stage `matrix`, 23 methods): an array of 4 tasks;
- the LLM stage: llm_single_call, llm_monotone, llm_monotone_constrained, hierarchical,
  in one process (one writer of the LLM cache, one client under the request rate);
- the consensus stage: score_fusion, ensemble, llm_rerank_kbest, fusion;
- `cluster/13_final_report.sbatch`: `tune.py merge --set test`, then
  `scripts/final_report.py`, which reads the merged rows and the LLM cache (no call).

Every job can be resubmitted: a rerun skips the runs already done.

## The outputs (`data/final/`)

| path | content |
|---|---|
| `runs/<method>.csv.gz` | the runner's rows, as in tuning (`docs/TUNING.md`, "Where the results are") |
| `methods/<method>.csv.gz` | one row per test lecture (below) |
| `summary.csv` | one row per method (below) |
| `summary_table.txt` | the summary as a table, best test accuracy first |
| `answers_flags.csv.gz` | one row per LLM answer: finish reason and the copying flags |
| `checks.txt` | the checks and their results |
| `plans/`, `shards/` | the array plan; the writers' rows before the merge |

A per-lecture row: `lecture_id`, `corpus`, `accuracy` (the share of the lecture's units
on their true slide, the metric of tuning), `n_correct`, `n_units`, `status`,
`compute_s` (the method's own time on matrices in memory, less the time inside the LLM
client), `build_cost_s` (the step-12 time of the tables the method reads, with the
measures they are built from, each once). An LLM or mixed method also has the cost of
the answers it used, cached ones included (`used_calls`, `used_seconds`,
`used_prompt_tokens`, `used_completion_tokens`; only successful calls, and only the
successful attempt's duration), the calls paid in this run (`paid_calls`,
`paid_seconds`, `paid_prompt_tokens`, `paid_completion_tokens`), `cache_hits`,
`failed_attempts` and `failed_calls` (information, never cost), `truncated` (answers
cut at max_tokens) and `parse_failed`.

A summary row: `kind` (non-LLM, LLM, or mixed: ensemble, fusion, llm_rerank_kbest),
`stage`, `measures`, `n_lectures`, `n_ok`, `test_accuracy` (mean per lecture, in %; a run
not ok counts 0), `tuning_accuracy` (from `config/tuned_settings.toml`, in %),
`diff_points`, and each cost as `<cost>_total` and, for compute, build and the used
group, `<cost>_per_lecture`. The build cost total adds the model load times once, as
`data/similarity/test/times.csv` does.

Combined methods carry their members' cost: their compute time holds the members'
(they run inside the method), their build cost the members' measures, and their used
calls the members' answers (ensemble: llm_single_call's; fusion: hierarchical's), which
come from the cache of this run's LLM stage. The paid group counts each call once,
where it was paid, so the sum of `paid_calls_total` over the methods is the run's
total. hierarchical reads no table and has no build cost; llm_single_call, llm_monotone
and llm_monotone_constrained count hybrid's, which they read for the fallback of an
unusable answer.

## The checks (`checks.txt`)

1. every method has the 140 test lectures, one row each, status ok, no NaN accuracy,
   and ran the combination chosen in tuning (`data/tuning/selection_report.csv`);
2. no failed call (the failed attempts are listed);
3. parse failures and truncated answers per LLM and mixed method;
4. copying: the answers of the methods that ask their own questions are read back by
   replaying each LLM and mixed method from the cache only (the replay must reproduce
   every recorded accuracy and pay nothing); an answer that begins with an old example
   (starts 0, 12, 30; gaps 12, 18; a first block (0, 11, 0)) or echoes a letter-shaped
   answer shape (s1, g0, `"from": a`, `<int>`, `...` in a list) is flagged;
5. the methods whose test accuracy is more than 5 points from their tuning accuracy.

`checks.txt` ends with the LLM calls paid in the run and the list of problems (the
script exits 1 when there is one; check 5 is information, not a problem).
