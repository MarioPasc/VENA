#!/usr/bin/env bash
# Submit the 1-GPU variant of the 4-epoch S2 smoke (full corpus + REMBRANDT).
# Uses gres=gpu:1, exhaustive_val disabled — queue-friendly fallback when the
# 2-GPU worker pends too long.
#
# Usage:
#   bash launcher_fm_train_rembrandt_smoke_1gpu.sh             # submit
#   bash launcher_fm_train_rembrandt_smoke_1gpu.sh --dry-run   # preview

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_4epoch_s2_full_corpus_1gpu.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/rembrandt_smoke_1gpu_%j.out \
    --error=${LOGS_DIR}/rembrandt_smoke_1gpu_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_fm_train_rembrandt_smoke_1gpu.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (1-GPU variant)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/rembrandt_smoke_1gpu_${JOB_ID}.out"
