#!/usr/bin/env bash
# Submit S3 LPL Batch-2 Arm C (λ_max=0.12 region, matched-strength region ablation) to Picasso.
#
# Design doc: .claude/notes/changes/lpl/2026-07-02_batch_2_lambda_calibration.md
# Prior batch analysis: .claude/notes/changes/lpl/2026-06-28_batch_1_default_recipe.md
#
# Usage:
#   bash launcher_picasso_s3lpl_b2_c_lambda012_region.sh             # submit
#   bash launcher_picasso_s3lpl_b2_c_lambda012_region.sh --dry-run   # print sbatch cmd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_s3lpl_b2_c_lambda012_region_fft.yaml"
JOB_NAME="vena-s3lpl-b2-c-lambda012-region"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

SBATCH_CMD="sbatch --parsable -J ${JOB_NAME} \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_fm_train_picasso.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    echo "CONFIG_PATH = ${CONFIG_PATH}"
    echo "JOB_NAME    = ${JOB_NAME}"
    echo "Logs dir    = ${LOGS_DIR}/${JOB_NAME}_<JOBID>.{out,err}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted ${JOB_NAME} job ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.{out,err}"
echo "Monitor:  squeue -j ${JOB_ID}"
