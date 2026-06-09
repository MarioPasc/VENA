#!/usr/bin/env bash
# Submit the S2 + LoRA r=16 + CFG dropout (p=0.15) 1000-epoch run to Picasso.
#
# Same v0.4 region-weighted contrastive as the s2_lora baseline; the only
# differentiator is `training.conditioning_dropout_p > 0` in the YAML.
# The CFG arm is identified post-hoc by reading
# `decision.json[conditioning_dropout_p]` from the run dir.
#
# Usage:
#   bash launcher_picasso_s2_lora_cfg.sh             # submit
#   bash launcher_picasso_s2_lora_cfg.sh --dry-run   # print sbatch command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/picasso_s2_1000ep_lora_r16_cfg.yaml"
JOB_NAME="vena-s2-lora-r16-cfg"
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
