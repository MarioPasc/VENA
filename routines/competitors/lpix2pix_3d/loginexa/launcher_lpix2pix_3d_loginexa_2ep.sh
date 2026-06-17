#!/usr/bin/env bash
# Launch the 3D-Latent-Pix2Pix 2-epoch smoke on the Picasso loginexa V100
# interactive node.
#
# Invocation: run this from the LOCAL workstation. It rsyncs the local repo
# into Picasso's scratch path FIRST, then SSH-hops into loginexa (via
# ~/.ssh/config ProxyJump=picasso), starts a detached tmux session, and
# returns immediately. Pix2Pix has no VGG perceptual loss, so no cache
# pre-warm step is required.
#
# Loginexa is NOT a SLURM partition — it is a standalone SSH-accessible
# interactive node (4 × Tesla V100-DGXS-32GB). The 30-min budget is convention,
# not a hard kill; do not run anything that exceeds it.
#
# Citation: arXiv:1611.07004; arXiv:2509.24194.
#
# Usage:
#   bash launcher_lpix2pix_3d_loginexa_2ep.sh             # rsync + submit
#   bash launcher_lpix2pix_3d_loginexa_2ep.sh --dry-run   # print plan, no rsync, no ssh
#
# Override GPU via GPU_ID env var (default: auto-pick freest by memory).
# Skip the rsync step with SKIP_RSYNC=1 (e.g. when running from picasso login).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_LOCAL="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/lpix2pix_3d/configs/smoke_loginexa_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/competitors/lpix2pix_3d}"
SESSION="${SESSION:-vena-lpix2pix-3d-loginexa-smoke}"
SSH_PICASSO="${SSH_PICASSO:-picasso}"
SSH_LOGINEXA="${SSH_LOGINEXA:-loginexa}"
SKIP_RSYNC="${SKIP_RSYNC:-0}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

EXCLUDES=(
    --exclude=".git/"
    --exclude="__pycache__/"
    --exclude=".pytest_cache/"
    --exclude=".mypy_cache/"
    --exclude=".ruff_cache/"
    --exclude="artifacts/"
    --exclude="experiments/"
    --exclude="docs/"
    --exclude="*.h5"
    --exclude="*.nii.gz"
    --exclude="*.pth"
    --exclude="*.ckpt"
    --exclude="src/external/t1c_rflow/upstream/checkpoints/"
)

echo "[plan] local repo:      ${REPO_DIR_LOCAL}"
echo "[plan] remote repo:     ${SSH_PICASSO}:${REPO_DIR_REMOTE}"
echo "[plan] loginexa python: ${PYTHON}"
echo "[plan] tmux session:    ${SESSION}"
echo "[plan] config:          ${CONFIG_PATH}"
echo "[plan] log:             ${LOG_DIR}/${SESSION}.log"
echo "[plan] skip rsync:      ${SKIP_RSYNC}"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Rsync local repo into Picasso scratch (so loginexa sees the latest code).
if [[ "${SKIP_RSYNC}" != "1" ]]; then
    echo "[step 1/3] rsync local → ${SSH_PICASSO}:${REPO_DIR_REMOTE} …"
    rsync -azP "${EXCLUDES[@]}" "${REPO_DIR_LOCAL}/" "${SSH_PICASSO}:${REPO_DIR_REMOTE}/"
fi

# 2) Auto-pick freest GPU on loginexa (4 V100s; sibling jobs may be running).
if [[ -z "${GPU_ID:-}" ]]; then
    GPU_ID=$(ssh "${SSH_LOGINEXA}" "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null" \
        | sort -t',' -k2,2 -nr | head -1 | awk -F',' '{print $1}' | tr -d ' ')
    GPU_ID="${GPU_ID:-0}"
fi
echo "[step 2/3] loginexa GPU=${GPU_ID}"

# 3) Launch detached tmux on loginexa (ProxyJump picasso is in ~/.ssh/config).
echo "[step 3/3] detached tmux session on loginexa …"
mkdir -p "${LOG_DIR}" 2>/dev/null || true
REMOTE_CMD="cd '${REPO_DIR_REMOTE}' \
&& mkdir -p '${LOG_DIR}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& '${PYTHON}' -m routines.competitors.lpix2pix_3d.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh "${SSH_LOGINEXA}" "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh ${SSH_LOGINEXA} tmux attach -t ${SESSION}"
echo "  list:     ssh ${SSH_LOGINEXA} tmux ls"
echo "  logs:     ssh ${SSH_LOGINEXA} tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh ${SSH_LOGINEXA} tmux kill-session -t ${SESSION}"
echo ""
echo "Watch sentinel: 'lpix2pix-3d-train completed' in the training log."
