#!/usr/bin/env bash
# Submit the 4-epoch S2 smoke (full corpus with REMBRANDT) to Picasso.
#
# Usage:
#   bash launcher_fm_train_rembrandt_smoke.sh             # submit
#   bash launcher_fm_train_rembrandt_smoke.sh --dry-run   # print sbatch command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_4epoch_s2_full_corpus.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/rembrandt_smoke_%j.out \
    --error=${LOGS_DIR}/rembrandt_smoke_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_fm_train_rembrandt_smoke.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/rembrandt_smoke_${JOB_ID}.out"
