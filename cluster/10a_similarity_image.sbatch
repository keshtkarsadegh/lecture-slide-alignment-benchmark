#!/bin/bash
# Step 10 setup: build the model scorers' image (cluster/similarity.def) once, into
# $PROJECT_ROOT/models_lsab/images/ outside the repository, and fetch the pinned weights
# (config/similarity_weights.toml) into $PROJECT_ROOT/models_lsab/. Both are idempotent.
# Submit from the repository root:  mkdir -p cluster/logs && sbatch cluster/10a_similarity_image.sbatch
#SBATCH --job-name=lsab-10a-image
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/10a_similarity_image_%j.log

set -euo pipefail

# Site values (proxy, LLM endpoint, API key): the untracked cluster/env.local (ACTIVATE.md).
source "$SLURM_SUBMIT_DIR/cluster/env.local"

# Compute nodes reach the internet only through the cluster's proxy.
export https_proxy="$CLUSTER_PROXY"
export http_proxy="$CLUSTER_PROXY"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
BASE=$PROJECT_ROOT
ENV=$BASE/envs/lsab
OUT=$BASE/models_lsab/images
cd "$SLURM_SUBMIT_DIR"
module load apptainer/1.4.3

echo "host $(hostname)  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"

# The weights: downloaded once at the pinned revisions, every file checked by sha256.
"$ENV/bin/python" scripts/build_similarity.py fetch

if [ -f "$OUT/similarity.sif" ]; then
    echo "image exists: $OUT/similarity.sif"
else
    # Build in the node-local job directory: no layer cache or temporary tree on Lustre.
    export APPTAINER_TMPDIR="$TMPDIR/apptainer-tmp" APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$OUT"
    apptainer build --fakeroot "$TMPDIR/similarity.sif" cluster/similarity.def
    mv "$TMPDIR/similarity.sif" "$OUT/similarity.sif.part"
    mv "$OUT/similarity.sif.part" "$OUT/similarity.sif"
fi
sha256sum "$OUT/similarity.sif"
apptainer exec "$OUT/similarity.sif" cat /opt/pip-freeze.txt
echo "end $(date -Is)"
