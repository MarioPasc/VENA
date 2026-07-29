#!/usr/bin/env bash
# Submit §18 ablation Arm B (L2, mean-seeking) to Picasso A100.
# Config: routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml
#
# Usage:
#   bash launcher_picasso_s1_v4_l2.sh             # submit
#   bash launcher_picasso_s1_v4_l2.sh --dry-run   # print sbatch command only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml"
JOB_NAME="vena-s1-v4-l2-fft"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

SBATCH_CMD="sbatch --parsable -J ${JOB_NAME} \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_fm_train_picasso_v4_ablation.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    echo "CONFIG_PATH = ${CONFIG_PATH}"
    echo "JOB_NAME    = ${JOB_NAME}"
    echo "Logs dir    = ${LOGS_DIR}/${JOB_NAME}_<JOBID>.{out,err}"
    exit 0
fi

# _clean_job_id: tail -n 1 discards any multi-line Lua warnings that Picasso's
# sbatch wrapper prints on stdout before the numeric ID. ANSI pattern uses
# [a-zA-Z] terminator (not just m) to cover all escape sequences. All
# non-numeric chars are then stripped so the result is a bare integer.
# CRITICAL: if the assert fires after the sbatch command ran, the job may
# already be live — squeue immediately before exiting.
_clean_job_id() {
    tail -n 1 <<<"$1" \
        | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

RAW=$(eval "${SBATCH_CMD}")
JOB_ID=$(_clean_job_id "${RAW}")
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
    printf 'FATAL: could not parse job ID from sbatch output:\n%s\n' "${RAW}" >&2
    echo "(squeue below — assume job was submitted until proven otherwise)" >&2
    squeue -u "${USER}"
    exit 1
}
echo "Submitted ${JOB_NAME} → job ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.{out,err}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Depend:   --dependency=afterok:${JOB_ID}"
