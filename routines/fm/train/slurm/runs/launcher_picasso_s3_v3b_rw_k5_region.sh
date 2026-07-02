#!/usr/bin/env bash
# Submit S3 LPL (K=5 REGION-aware α) warm-started from S1 v3b_rw to Picasso.
#
# Ablation S3-D of the 2026-06-28 S3-on-top-of-v3 tournament.
# Design doc: .claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md
#
# Usage:
#   bash launcher_picasso_s3_v3b_rw_k5_region.sh             # submit
#   bash launcher_picasso_s3_v3b_rw_k5_region.sh --dry-run   # print sbatch cmd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_s3_v3b_rw_k5_region_fft.yaml"
JOB_NAME="vena-s3-v3b-rw-k5-region"
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
