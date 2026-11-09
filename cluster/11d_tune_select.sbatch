#!/bin/bash
# Step 11: fold the shards into data/tuning/runs/<method>.csv.gz (only when no writer
# is running), then choose one setting per complete method: config/tuned_settings.toml
# and data/tuning/selection_report.csv. Run after (a) and (b), and again after (c).
# Submit from the repository root:
#   sbatch cluster/11d_tune_select.sbatch
#SBATCH --job-name=lsab-11d-select
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/11d_tune_select_%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/tune.py merge
"$ENV/bin/python" scripts/select_settings.py
"$ENV/bin/python" scripts/tune.py status
echo "end $(date -Is)"
