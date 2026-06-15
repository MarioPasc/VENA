#!/usr/bin/env bash
# Submit the full T1C-RFlow training to Picasso A100 DGX (single GPU, 7-day budget).
#
# T1C-RFlow has no VGG perceptual loss — no cache pre-warm step.
#
# Citation: Eidex et al. 2025, arXiv:2509.24194.
#
# Usage:
#   bash launcher_t1c_rflow_picasso_full.sh             # submit
#   bash launcher_t1c_rflow_picasso_full.sh --dry-run   # print sbatch command, no submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/competitors/t1c_rflow/configs/picasso_full.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/t1c_rflow_picasso_full_%j.out \
    --error=${LOGS_DIR}/t1c_rflow_picasso_full_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_t1c_rflow_picasso_full.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (T1C-RFlow picasso full training)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/t1c_rflow_picasso_full_${JOB_ID}.{out,err}"
echo "Sentinel: 't1c-rflow-train completed' in the training log."
