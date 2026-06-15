#!/usr/bin/env bash
# Submit the full pGAN training to Picasso A100 DGX (single GPU, 2-day budget).
#
# Usage:
#   bash launcher_pgan_picasso_full.sh             # submit
#   bash launcher_pgan_picasso_full.sh --dry-run   # print sbatch command, no submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/competitors/pgan_cgan/configs/picasso_full_t2.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
mkdir -p "${LOGS_DIR}" "${TORCH_HOME}/hub/checkpoints"

# Pre-warm VGG (login node has internet; compute nodes do not).
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
fi
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}/bin/python}"
if [[ -x "${PYTHON}" ]]; then
    echo "[warm] ensuring VGG16 weights are cached under ${TORCH_HOME} …"
    TORCH_HOME="${TORCH_HOME}" "${PYTHON}" -c "from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.DEFAULT); print('VGG16 cached OK')" \
        || echo "[warn] VGG warm-up failed; worker may fail if cache missing"
else
    echo "[warn] python interpreter ${PYTHON} not found on the login node; assuming VGG already cached"
fi

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/pgan_picasso_full_t2_%j.out \
    --error=${LOGS_DIR}/pgan_picasso_full_t2_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH},TORCH_HOME=${TORCH_HOME} \
    ${SCRIPT_DIR}/worker_pgan_picasso_full_t2.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (pGAN picasso full training)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/pgan_picasso_full_t2_${JOB_ID}.{out,err}"
