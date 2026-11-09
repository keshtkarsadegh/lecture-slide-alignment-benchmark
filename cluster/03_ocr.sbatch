#!/bin/bash
# Step 2b: OCR the slide images that need it with Tesseract (see README.md), one array
# task per OCR corpus: 0 mavils, 1 mcd, 2 slide_highlight (M3AV is not OCRed).
# Needs the image from cluster/00_tesseract_image.sbatch. Writes data/ocr/<name>.jsonl.gz and data/ocr/<name>.json.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/03_ocr.sbatch
#SBATCH --job-name=lsab-03-ocr
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster/logs/03_ocr_%A_%a.log

set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
NAMES=(mavils mcd slide_highlight)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
cd "$SLURM_SUBMIT_DIR"
module load apptainer/1.4.3

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)  corpus $NAME"
"$ENV/bin/python" scripts/ocr.py --corpus "$NAME" --workers "$SLURM_CPUS_PER_TASK"
echo "end $(date -Is)"
