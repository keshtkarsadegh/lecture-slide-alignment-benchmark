#!/bin/bash
# Step 2b: split the kept M3AV lectures into versions A and B (see README.md).
# Reads data/raw/m3av (step 1), writes data/splits/m3av_split.csv and .json.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/02_split_m3av.sbatch
#SBATCH --job-name=lsab-02-split
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/02_split_m3av_%j.log

set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/split_m3av.py
echo "end $(date -Is)"
