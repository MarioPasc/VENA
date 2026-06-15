#!/usr/bin/env bash
# Launch the ResViT 2-epoch smoke (stage 1: 1+1, stage 2: 1+1) on the Picasso
# loginexa V100 interactive node.
#
# Invocation: run this from the Picasso login node (picasso3). It verifies the
# ViT .npz cache (login node has internet — re-download if missing), then
# SSH-hops into loginexa, starts a detached tmux session, and returns
# immediately.
#
# Loginexa is NOT a SLURM partition — it is a standalone SSH-accessible
# interactive node (10.248.7.200, 4 × Tesla V100-DGXS-32GB). 30-min budget is
# convention, not a hard kill.
#
# Conda env: vena-v100 (torch 2.7.1+cu126, sm_70 compatible).

set -euo pipefail

REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/resvit/configs/smoke_loginexa_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/competitors/resvit}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
SESSION="${SESSION:-vena-resvit-loginexa-smoke}"
VIT_NPZ="${REPO_DIR_REMOTE}/src/external/resvit/upstream/checkpoints/R50+ViT-B_16.npz"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}" "${TORCH_HOME}/hub/checkpoints" "$(dirname "${VIT_NPZ}")"

if [[ -z "${GPU_ID:-}" ]]; then
    GPU_ID=$(ssh loginexa "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null" \
        | sort -t',' -k2,2 -nr | head -1 | awk -F',' '{print $1}' | tr -d ' ')
    GPU_ID="${GPU_ID:-0}"
fi
echo "[plan] loginexa GPU=${GPU_ID}"
echo "[plan] tmux session: ${SESSION}"
echo "[plan] config:       ${CONFIG_PATH}"
echo "[plan] log:          ${LOG_DIR}/${SESSION}.log"
echo "[plan] ViT npz:      ${VIT_NPZ}"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Ensure ViT .npz cached on shared FS (login node has internet).
echo "[step 1/2] verify R50+ViT-B_16.npz …"
if [[ ! -f "${VIT_NPZ}" ]]; then
    echo "[warn] ViT npz missing — re-downloading on login node"
    curl -sSL -o "${VIT_NPZ}" https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
fi
ls -la "${VIT_NPZ}"

# 2) Launch detached tmux on loginexa.
echo "[step 2/2] launch detached tmux session on loginexa …"
REMOTE_CMD="cd '${REPO_DIR_REMOTE}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export TORCH_HOME='${TORCH_HOME}' \
&& '${PYTHON}' -m routines.competitors.resvit.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh loginexa "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh loginexa tmux attach -t ${SESSION}"
echo "  list:     ssh loginexa tmux ls"
echo "  logs:     ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh loginexa tmux kill-session -t ${SESSION}"
