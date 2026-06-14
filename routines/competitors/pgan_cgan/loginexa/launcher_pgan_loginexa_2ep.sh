#!/usr/bin/env bash
# Launch the pGAN 4-epoch smoke on the Picasso loginexa V100 interactive node.
#
# Invocation: run this from the Picasso login node (picasso3). It pre-warms the
# VGG16 cache (login node has internet), then SSH-hops into loginexa, starts a
# detached tmux session, and returns immediately.
#
# Loginexa is NOT a SLURM partition — it is a standalone SSH-accessible
# interactive node (10.248.7.200, 4 × Tesla V100-DGXS-32GB). The 30-min budget
# is convention, not a hard kill; do not run anything that exceeds it.
#
# Usage:
#   bash launcher_pgan_loginexa_4ep.sh             # submit
#   bash launcher_pgan_loginexa_4ep.sh --dry-run   # print plan, no ssh
#
# Override GPU via GPU_ID env var (default: auto-pick freest by memory).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/pgan_cgan/configs/smoke_loginexa_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/VENA/logs/competitors/pgan_cgan}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
SESSION="${SESSION:-vena-pgan-loginexa-smoke}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}" "${TORCH_HOME}/hub/checkpoints"

# Auto-pick freest GPU on loginexa (4 V100s; sibling jobs may be running).
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

# 1) Pre-warm VGG (picasso login node has internet; loginexa shared FS).
echo "[step 1/2] warm VGG16 cache (login node) …"
TORCH_HOME="${TORCH_HOME}" "${PYTHON}" -c "from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.DEFAULT); print('VGG16 cached OK')"

# 2) Launch detached tmux on loginexa.
echo "[step 2/2] launch detached tmux session on loginexa …"
REMOTE_CMD="cd '${REPO_DIR_REMOTE}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export TORCH_HOME='${TORCH_HOME}' \
&& '${PYTHON}' -m routines.competitors.pgan_cgan.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh loginexa "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh loginexa tmux attach -t ${SESSION}"
echo "  list:     ssh loginexa tmux ls"
echo "  logs:     ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh loginexa tmux kill-session -t ${SESSION}"
