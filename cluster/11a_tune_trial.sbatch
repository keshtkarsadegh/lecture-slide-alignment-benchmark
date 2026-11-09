#!/bin/bash
# Step 11, the trial: 2 tuning lectures (the median-sized and the largest) x a few
# combinations of every method, LLM methods included (about 40 calls, through the
# proxy), to check that everything runs and to time it. Writes data/tuning/trial/
# (never merged into the results) and the estimate of the full run's time.
# Submit from the repository root:
#   mkdir -p cluster/logs && sbatch cluster/11a_tune_trial.sbatch
#SBATCH --job-name=lsab-11a-trial
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/11a_tune_trial_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# one CPU per job: no hidden BLAS threads, so the times are one core's
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/tune.py trial
echo "end $(date -Is)"
