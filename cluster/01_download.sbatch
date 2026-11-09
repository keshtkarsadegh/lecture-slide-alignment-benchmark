#!/bin/bash
# Step 1: download all four corpora into data/raw/ (see README.md).
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/01_download.sbatch
#SBATCH --job-name=lsab-01-download
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/01_download_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

# Compute nodes reach the internet only through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/download.py --corpus all
echo "end $(date -Is)"
