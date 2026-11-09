#!/bin/bash
# Step 2b setup: build the Tesseract image (cluster/tesseract.def) once, into the
# environment folder outside the repository. scripts/ocr.py runs Tesseract from it.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/00_tesseract_image.sbatch
#SBATCH --job-name=lsab-00-tesseract
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/00_tesseract_image_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

# Compute nodes reach the internet only through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
OUT=$PROJECT_ROOT/envs/lsab-tesseract
cd "$SLURM_SUBMIT_DIR"
module load apptainer/1.4.3

# Build in the node-local job directory: no layer cache or temporary tree on Lustre.
export APPTAINER_TMPDIR="$TMPDIR/apptainer-tmp" APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$OUT"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
apptainer build --fakeroot "$TMPDIR/tesseract.sif" cluster/tesseract.def
mv "$TMPDIR/tesseract.sif" "$OUT/tesseract.sif.part"
mv "$OUT/tesseract.sif.part" "$OUT/tesseract.sif"
sha256sum "$OUT/tesseract.sif"
apptainer exec "$OUT/tesseract.sif" tesseract --version
apptainer exec "$OUT/tesseract.sif" cat /opt/eng.traineddata.sha256
echo "end $(date -Is)"
