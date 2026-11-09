#!/bin/bash
# Step 5: the discipline of every lecture (scripts/assign_discipline.py, the LLM), then
# the attributes table (scripts/attributes.py): data/attributes/document_attributes.csv.
# Needs data/prepared/ (step 2) and data/attributes/layout_boxes.jsonl.gz (05).
# Without an API key (LLM_API_KEY or config/llm_credentials.toml) the discipline step
# reports "blocked" and the table is written with an empty discipline column.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/06_attributes.sbatch
#SBATCH --job-name=lsab-06-attributes
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/06_attributes_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The LLM endpoint is reached through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY" http_proxy="$CLUSTER_PROXY"

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
status=0
"$ENV/bin/python" scripts/assign_discipline.py || status=$?
if [ "$status" -eq 3 ]; then
    echo "discipline blocked: no API key (the table joins data/attributes/discipline.csv"
    echo "if an earlier run wrote it, else its discipline column is empty)"
elif [ "$status" -ne 0 ]; then
    exit "$status"
fi
"$ENV/bin/python" scripts/attributes.py
echo "end $(date -Is)"
