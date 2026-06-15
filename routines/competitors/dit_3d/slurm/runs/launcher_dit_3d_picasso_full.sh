#!/usr/bin/env bash
# Submit the full 3D-DiT training to Picasso A100 DGX (single GPU, 3-day budget).
#
# 3D-DiT (Peebles & Xie 2023 backbone + Eidex 2025 §4 RFlow recipe) has no
# VGG perceptual loss — no cache pre-warm step.
#
# Paper-budget exposure (skill §3.11): 164 epochs × 1,751 latents/epoch ≈
# the 286,000-sample Eidex 2025 baseline budget. At batch=4 → ~71.8k steps.
# DiT-B/4 (~130M params) on A100 under AMP: ~0.5-1s/step → 10-20h wall-clock.
# `--time=3-00:00:00` is a comfortable 3-5× overshoot for step-time variance.
#
# Citation: arXiv:2212.09748; arXiv:2509.24194.
#
# Usage:
#   bash launcher_dit_3d_picasso_full.sh             # submit
#   bash launcher_dit_3d_picasso_full.sh --dry-run   # print sbatch command, no submit

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/competitors/dit_3d/configs/picasso_full.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/dit_3d_picasso_full_%j.out \
    --error=${LOGS_DIR}/dit_3d_picasso_full_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH} \
    ${SCRIPT_DIR}/worker_dit_3d_picasso_full.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (3D-DiT picasso full training)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/dit_3d_picasso_full_${JOB_ID}.{out,err}"
echo "Sentinel: 'dit-3d-train completed' in the training log."
