#!/usr/bin/env bash
# Launcher: rsync VENA repo → server-3, warm VGG cache, run pGAN 4-epoch smoke
# inside a detached tmux session, return immediately.
#
# Usage:
#   bash launcher_pgan_server3_4ep.sh             # submit
#   bash launcher_pgan_server3_4ep.sh --dry-run   # print plan, no rsync, no ssh
#
# Configuration: pulled from .claude/server3.yaml (ssh alias, paths). Sticking
# with the same ssh host and remote repo path as the FM trainer; only the
# command line + tmux session name differ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SSH_ALIAS="${SSH_ALIAS:-icai-server}"
REMOTE_REPO="${REMOTE_REPO:-/home/mariopascual/projects/VENA}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/mariopascual/.conda/envs/vena/bin/python}"
REMOTE_CONFIG="${REMOTE_REPO}/routines/competitors/pgan_cgan/configs/smoke_server3_4ep.yaml"
SESSION="${SESSION:-vena-pgan-smoke}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/media/hddb/mario/smoke_logs/competitors/pgan_cgan}"
GPU_ID="${GPU_ID:-0}"
TORCH_HOME_REMOTE="${TORCH_HOME_REMOTE:-/home/mariopascual/.cache/torch}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# rsync excludes: keep heavy data out, but DO include src/external (vendor code
# is needed at runtime).
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
)

echo "[plan] sync ${REPO_DIR}/ → ${SSH_ALIAS}:${REMOTE_REPO}/"
echo "[plan] tmux session: ${SESSION} on ${SSH_ALIAS}, GPU=${GPU_ID}"
echo "[plan] config: ${REMOTE_CONFIG}"
echo "[plan] logs:   ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "[plan] torch home (cached VGG): ${TORCH_HOME_REMOTE}"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Rsync
echo "[step 1/3] rsync …"
rsync -azP --delete-after "${EXCLUDES[@]}" "${REPO_DIR}/" "${SSH_ALIAS}:${REMOTE_REPO}/"

# 2) Warm VGG cache (server-3 has internet)
echo "[step 2/3] warm VGG16 cache on server-3 …"
ssh "${SSH_ALIAS}" "mkdir -p '${TORCH_HOME_REMOTE}' '${REMOTE_LOG_DIR}' && TORCH_HOME='${TORCH_HOME_REMOTE}' '${REMOTE_PYTHON}' -c 'from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.DEFAULT); print(\"VGG16 cached OK\")'"

# 3) Launch in detached GNU screen (tmux is not installed on icai-server).
echo "[step 3/3] launch detached screen session ${SESSION} …"
REMOTE_CMD="cd '${REMOTE_REPO}' \
&& export PYTHONPATH='${REMOTE_REPO}/src:${REMOTE_REPO}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export TORCH_HOME='${TORCH_HOME_REMOTE}' \
&& '${REMOTE_PYTHON}' -m routines.competitors.pgan_cgan.cli '${REMOTE_CONFIG}' 2>&1 | tee -a '${REMOTE_LOG_DIR}/${SESSION}.log'"

ssh "${SSH_ALIAS}" "screen -dmS '${SESSION}' bash -c \"${REMOTE_CMD}; echo 'pGAN-train completed-screen-marker'; sleep 2\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh ${SSH_ALIAS} screen -r ${SESSION}"
echo "  list:     ssh ${SSH_ALIAS} screen -ls"
echo "  logs:     ssh ${SSH_ALIAS} tail -F ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh ${SSH_ALIAS} screen -X -S ${SESSION} quit"
