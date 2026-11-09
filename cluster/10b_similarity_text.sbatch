#!/bin/bash
# Step 10, job 1 of 3: the text measures that need no local model - bm25, embed (the LLM
# endpoint, through the proxy, at config/similarity.toml's request rate) and hybrid - for
# one lecture set (SET, default tuning), into data/similarity/<set>/tables.sqlite3.
# scripts/build_similarity.py explains the order of the three jobs: the shared cache is
# written by one job at a time, so job 2 starts after this one, and job 3 after job 2.
# Rerunning reuses what is built. MODEL_DEVICE is the device the model measures run on
# (job 2); it is part of the set's settings, so all three jobs pass the same value.
# Submit from the repository root:
#   mkdir -p cluster/logs && sbatch --export=ALL,SET=tuning cluster/10b_similarity_text.sbatch
#SBATCH --job-name=lsab-10b-text
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/10b_similarity_text_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

# Compute nodes reach the endpoint only through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

SET=${SET:-tuning}
MODEL_DEVICE=${MODEL_DEVICE:-cuda}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  set $SET  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/build_similarity.py build --set "$SET" \
    --measures bm25 embed hybrid --model-device "$MODEL_DEVICE"
echo "end $(date -Is)"
