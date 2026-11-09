#!/bin/bash
# Step 10, job 3 of 3: the blends of the model measures - hybrid_splade and hybrid_image,
# from the matrices jobs 1 and 2 cached - for one lecture set (SET, default tuning); then
# the checks of all 8 measures' tables (data/similarity/<set>/sanity.csv) and their build
# times (data/similarity/<set>/times.csv). Rerunning reuses what is built.
# Submit from the repository root:
#   sbatch --export=ALL,SET=tuning --dependency=afterok:<job 2> cluster/10d_similarity_blends.sbatch
#SBATCH --job-name=lsab-10d-blends
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/10d_similarity_blends_%j.log

set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

SET=${SET:-tuning}
MODEL_DEVICE=${MODEL_DEVICE:-cuda}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  set $SET  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/build_similarity.py build --set "$SET" \
    --measures hybrid_splade hybrid_image --model-device "$MODEL_DEVICE"
"$ENV/bin/python" scripts/build_similarity.py times --set "$SET"
"$ENV/bin/python" scripts/build_similarity.py sanity --set "$SET"
echo "end $(date -Is)"
