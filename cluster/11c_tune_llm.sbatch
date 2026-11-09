#!/bin/bash
# Step 11 (b) and (c): the one LLM process. STAGE=llm runs the LLM base methods
# (llm_single_call, llm_monotone, llm_monotone_constrained, hierarchical with refinement
# on before off); STAGE=consensus, after merge and scripts/select_settings.py, runs
# score_fusion, ensemble, llm_rerank_kbest and fusion with their members at the tuned
# settings. One process: the LLM cache has one writer, and one client keeps the whole
# run under config/llm.toml's request rate. Writes data/tuning/shards/<stage>.sqlite3;
# a rerun skips what is done (failed calls are asked again).
# Submit from the repository root:
#   sbatch --export=ALL,STAGE=llm cluster/11c_tune_llm.sbatch
# Step 13, the final run on the test set (into data/final/shards/; consensus after llm,
# so that the members' answers are in the cache):
#   sbatch --export=ALL,STAGE=llm,SET=test,SETTINGS=config/tuned_settings.toml cluster/11c_tune_llm.sbatch
#SBATCH --job-name=lsab-11c-llm
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster/logs/11c_tune_llm_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGE=${STAGE:-llm}
SET=${SET:-tuning}
SETTINGS=${SETTINGS:-}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  set $SET  settings ${SETTINGS:-grid}  stage $STAGE  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/tune.py run --set "$SET" ${SETTINGS:+--settings "$SETTINGS"} \
    --stage "$STAGE"
echo "end $(date -Is)"
