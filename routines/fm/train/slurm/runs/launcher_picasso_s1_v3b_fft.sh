#!/usr/bin/env bash
# Submit S1 v3 Variant B (channel-concat at trunk + ControlNet 3-ch mask) to Picasso.
#
# Background: .claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md
# Spec:        .claude/notes/changes/2026-06-22_s1_v3_model_implementation.md
#
# Usage:
#   bash launcher_picasso_s1_v3b_fft.sh             # submit
#   bash launcher_picasso_s1_v3b_fft.sh --dry-run   # print sbatch command
#   sbatch --test-only ...                          # offline parse-only check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_s1_v3b_concat_plus_cn3ch_fft.yaml"
JOB_NAME="vena-s1-v3b-fft"
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
