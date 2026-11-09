#!/bin/bash
# Step 16: Q4, important units placed on the wrong slide. First the final run's per-unit
# predictions by an offline replay (scripts/final_predictions.py: every method's tuned
# combination on the 140 test lectures, LLM answers from data/llm_cache.sqlite3 only;
# a cache miss stops it), checked against step 13's per-lecture counts; then the
# table and heatmap (scripts/analysis_q4.py). No API key, endpoint or proxy: this job
# cannot reach the LLM endpoint. Submit from the repository root:
#   sbatch cluster/16_q4.sbatch
#SBATCH --job-name=lsab-16-q4
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/16_q4_%j.log

set -euo pipefail
unset LLM_API_KEY LLM_API_BASE https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SETTINGS=${SETTINGS:-config/tuned_settings.toml}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  settings $SETTINGS  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
srun -n 1 "$ENV/bin/python" scripts/final_predictions.py --settings "$SETTINGS"
srun -n 1 "$ENV/bin/python" scripts/analysis_q4.py
echo "end $(date -Is)"
