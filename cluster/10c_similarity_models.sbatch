#!/bin/bash
# Step 10, job 2 of 3: the local-model measures - splade, image (ColQwen2) and
# cross_encoder (which reranks hybrid, read from the vectors job 1 cached) - on one GPU,
# for one lecture set (SET, default tuning). Needs the image and the weights of
# cluster/10a_similarity_image.sbatch, and job 1 finished (cluster/10b_similarity_text.sbatch).
# Each model's load is timed apart from its compute. Rerunning reuses what is built.
# Submit from the repository root:
#   sbatch --export=ALL,SET=tuning --dependency=afterok:<job 1> cluster/10c_similarity_models.sbatch
#SBATCH --job-name=lsab-10c-models
#SBATCH --partition=grete
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/10c_similarity_models_%j.log

set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

SET=${SET:-tuning}
MODEL_DEVICE=${MODEL_DEVICE:-cuda}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
BASE=$PROJECT_ROOT
ENV=$BASE/envs/lsab
SIF=$BASE/models_lsab/images/similarity.sif
cd "$SLURM_SUBMIT_DIR"
REPO=$(pwd -P)                   # the real Lustre path, bound into the image
MODELS=$(readlink -f "$BASE")/models_lsab
module load apptainer/1.4.3

echo "host $(hostname)  set $SET  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

apptainer exec --nv --containall --pwd "$REPO" -B "$REPO" -B "$MODELS" -B "$BASE" \
    --env PROJECT_ROOT="$PROJECT_ROOT" "$SIF" python scripts/build_similarity.py build --set "$SET" \
    --measures splade image cross_encoder --model-device "$MODEL_DEVICE"
echo "end $(date -Is)"
