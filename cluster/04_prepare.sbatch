#!/bin/bash
# Step 2: prepare every lecture of the five prepared corpora (mavils, m3av_a, m3av_b, mcd,
# slide_highlight) with the 40-slide cap (see README.md).
# Reads data/raw/ (step 1), data/splits/m3av_split.csv and data/ocr/ (step 2b);
# writes data/prepared/.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/04_prepare.sbatch
# Some corpora only:  sbatch cluster/04_prepare.sbatch mavils mcd     (one run per name)
#SBATCH --job-name=lsab-04-prepare
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/04_prepare_%j.log

set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
for corpus in "${@:-all}"; do
    "$ENV/bin/python" scripts/prepare.py --corpus "$corpus"
done
echo "end $(date -Is)"
