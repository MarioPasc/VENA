#!/usr/bin/env bash
# Launch the pGAN single-modality (t1pre → t1c) 2-epoch smoke on loginexa.
# Paper-faithful: one source modality, one target modality, no neighbouring-
# slice context (input_nc=1).
#
# Usage:
#   bash launcher_pgan_loginexa_t1pre_2ep.sh             # submit
#   bash launcher_pgan_loginexa_t1pre_2ep.sh --dry-run   # print plan

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/pgan_cgan/configs/smoke_loginexa_t1pre_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/VENA/logs/competitors/pgan_cgan}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
SESSION="${SESSION:-vena-pgan-loginexa-t1pre-smoke}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}" "${TORCH_HOME}/hub/checkpoints"

if [[ -z "${GPU_ID:-}" ]]; then
    GPU_ID=$(ssh loginexa "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null" \
        | sort -t',' -k2,2 -nr | head -1 | awk -F',' '{print $1}' | tr -d ' ')
    GPU_ID="${GPU_ID:-0}"
fi
echo "[plan] loginexa GPU=${GPU_ID}"
echo "[plan] tmux session: ${SESSION}"
echo "[plan] config:       ${CONFIG_PATH}"
echo "[plan] log:          ${LOG_DIR}/${SESSION}.log"

if ${DRY_RUN}; then
    exit 0
fi

TORCH_HOME="${TORCH_HOME}" "${PYTHON}" -c "from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.DEFAULT); print('VGG16 cached OK')" >/dev/null 2>&1 || true

REMOTE_CMD="cd '${REPO_DIR_REMOTE}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export TORCH_HOME='${TORCH_HOME}' \
&& '${PYTHON}' -m routines.competitors.pgan_cgan.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh loginexa "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach: ssh loginexa tmux attach -t ${SESSION}"
echo "  logs:   ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:   ssh loginexa tmux kill-session -t ${SESSION}"
