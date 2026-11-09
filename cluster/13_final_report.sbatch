#!/bin/bash
# Step 13: fold the final run's shards into data/final/runs/<method>.csv.gz (only when no
# writer of the final run is running), then write its tables and checks
# (scripts/final_report.py: data/final/methods/, summary.csv, summary_table.txt,
# answers_flags.csv.gz, checks.txt). Reads the LLM cache only; pays no call. Run after
# the matrix array and the llm and consensus stages of the final run (docs/FINAL_RUN.md).
# Submit from the repository root:
#   sbatch cluster/13_final_report.sbatch
#SBATCH --job-name=lsab-13-report
#SBATCH --partition=medium96s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/13_final_report_%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SETTINGS=${SETTINGS:-config/tuned_settings.toml}
# PROJECT_ROOT: the directory next to the clone with envs/ and models_lsab/ (ACTIVATE.md).
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$SLURM_SUBMIT_DIR")}
ENV=$PROJECT_ROOT/envs/lsab
cd "$SLURM_SUBMIT_DIR"

echo "host $(hostname)  settings $SETTINGS  start $(date -Is)  commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ENV/bin/python" scripts/tune.py merge --set test --settings "$SETTINGS"
"$ENV/bin/python" scripts/tune.py status --set test --settings "$SETTINGS"
"$ENV/bin/python" scripts/final_report.py --settings "$SETTINGS"
echo "end $(date -Is)"
