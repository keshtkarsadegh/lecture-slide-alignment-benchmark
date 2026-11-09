#!/bin/bash
# Step 11 (a): the non-LLM base methods (stage "matrix": 22 methods and mbr), as an
# array over the tasks of a plan (scripts/tune.py plan --name matrix --tasks N). Each
# task writes its own shard, data/tuning/shards/<plan>_<task>.sqlite3; it reads the
# tuning tables read-only and touches no LLM cache, so it runs next to the LLM process.
# A rerun skips the runs already done. After all tasks: scripts/tune.py merge.
# Submit from the repository root (N tasks, at most 20 at once):
#   sbatch --array=0-19%20 --export=ALL,PLAN=matrix cluster/11b_tune_matrix.sbatch
# Step 13, the final run on the test set (the plan made with the same --set/--settings;
# the shards go to data/final/shards/):
#   sbatch --array=0-3 --export=ALL,PLAN=matrix,SET=test,SETTINGS=config/tuned_settings.toml \
#       cluster/11b_tune_matrix.sbatch
#SBATCH --job-name=lsab-11b-matrix
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/11b_tune_matrix_%A_%a.log

set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PLAN=${PLAN:-matrix}
SET=${SET:-tuning}
SETTINGS=${SETTINGS:-}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  set $SET  settings ${SETTINGS:-grid}  plan $PLAN  task $SLURM_ARRAY_TASK_ID  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/tune.py run --set "$SET" ${SETTINGS:+--settings "$SETTINGS"} \
    --plan "$PLAN" --task "$SLURM_ARRAY_TASK_ID"
echo "end $(date -Is)"
