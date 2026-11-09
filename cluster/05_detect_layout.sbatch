#!/bin/bash
# Step 5: detect pictures and tables on every kept slide image (scripts/detect_layout.py),
# then draw the 100-slide validation sheet for the check by eye.
# Needs the image from cluster/00_layout_image.sbatch and data/prepared/ (step 2).
# Downloads the model weights once into $PROJECT_ROOT/models_lsab/ (fetch), then writes
# data/attributes/layout_boxes.jsonl.gz + .json and data/attributes/validation_100.*.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/05_detect_layout.sbatch
#SBATCH --job-name=lsab-05-layout
#SBATCH --partition=grete
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/05_detect_layout_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
BASE=$PROJECT_ROOT
ENV=$BASE/envs/lsab
SIF=$ENV/images/layout.sif
cd "$SLURM_SUBMIT_DIR"
REPO=$(pwd -P)                   # the real Lustre path, bound into the image
MODELS=$(readlink -f "$BASE")/models_lsab
mkdir -p "$MODELS"
module load apptainer/1.4.3

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# The weights: downloaded once at the pinned revision, every file checked by sha256.
https_proxy="$CLUSTER_PROXY" http_proxy="$CLUSTER_PROXY" \
    "$ENV/bin/python" scripts/detect_layout.py fetch

apptainer exec --nv --containall --pwd "$REPO" -B "$REPO" -B "$MODELS" -B "$BASE" \
    --env PROJECT_ROOT="$PROJECT_ROOT" "$SIF" python scripts/detect_layout.py detect

"$ENV/bin/python" scripts/detect_layout.py sheet
echo "end $(date -Is)"
