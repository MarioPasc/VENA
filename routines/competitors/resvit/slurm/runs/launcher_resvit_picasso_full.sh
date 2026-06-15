#!/usr/bin/env bash
# Submit the full ResViT training to Picasso A100 DGX (single GPU, 5-day budget).
#
# Usage:
#   bash launcher_resvit_picasso_full.sh             # submit
#   bash launcher_resvit_picasso_full.sh --dry-run   # print sbatch command, no submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/competitors/resvit/configs/picasso_full.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export VIT_NPZ="${REPO_DIR}/src/external/resvit/upstream/checkpoints/R50+ViT-B_16.npz"

mkdir -p "${LOGS_DIR}" "${TORCH_HOME}/hub/checkpoints" "$(dirname "${VIT_NPZ}")"

# Pre-warm ViT .npz cache on login node (compute nodes have no internet).
# This is the analogue of pgan's VGG16 pre-warm: the launcher guarantees the
# weight file exists on disk before the sbatch ever fires.
if [[ ! -f "${VIT_NPZ}" ]]; then
    echo "[warm] R50+ViT-B_16.npz missing — downloading on login node …"
    curl -sSL -o "${VIT_NPZ}" https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
else
    echo "[warm] R50+ViT-B_16.npz already cached at ${VIT_NPZ}"
fi
ls -la "${VIT_NPZ}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/resvit_picasso_full_%j.out \
    --error=${LOGS_DIR}/resvit_picasso_full_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH},TORCH_HOME=${TORCH_HOME},VIT_NPZ=${VIT_NPZ} \
    ${SCRIPT_DIR}/worker_resvit_picasso_full.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (ResViT picasso full training)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/resvit_picasso_full_${JOB_ID}.{out,err}"
