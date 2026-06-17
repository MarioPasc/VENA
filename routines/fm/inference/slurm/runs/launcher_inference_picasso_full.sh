#!/usr/bin/env bash
# Submit the full unified validation-inference to Picasso A100 DGX.
#
# Sweeps every method × every test patient × full NFE list. Single A100,
# sequential method execution (adapter teardown frees VRAM before the next
# method's setup). Wallclock dominated by C6-LDDPM at NFE=1000.
#
# Usage:
#   bash launcher_inference_picasso_full.sh             # submit
#   bash launcher_inference_picasso_full.sh --dry-run   # print sbatch command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/inference/configs/picasso_full.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/inference_picasso_full_%j.out \
    --error=${LOGS_DIR}/inference_picasso_full_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_inference_picasso_full.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (VENA unified inference, picasso_full)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/inference_picasso_full_${JOB_ID}.{out,err}"
echo "Sentinel: 'inference routine complete' in the routine log."
