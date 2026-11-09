#!/bin/bash
# Step 5 setup: build the layout-detection image (cluster/layout.def) once, into the
# project's environment folder outside the repository. scripts/detect_layout.py detect
# runs inside it (cluster/05_detect_layout.sbatch).
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/00_layout_image.sbatch
#SBATCH --job-name=lsab-00-layout
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/00_layout_image_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

# Compute nodes reach the internet only through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
OUT=$PROJECT_ROOT/envs/lsab/images
cd "$SLURM_SUBMIT_DIR"
module load apptainer/1.4.3

# Build in the node-local job directory: no layer cache or temporary tree on Lustre.
export APPTAINER_TMPDIR="$TMPDIR/apptainer-tmp" APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$OUT"

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
apptainer build --fakeroot "$TMPDIR/layout.sif" cluster/layout.def
mv "$TMPDIR/layout.sif" "$OUT/layout.sif.part"
mv "$OUT/layout.sif.part" "$OUT/layout.sif"
sha256sum "$OUT/layout.sif"
apptainer exec "$OUT/layout.sif" cat /opt/pip-freeze.txt
echo "end $(date -Is)"
