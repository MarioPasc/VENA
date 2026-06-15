#!/usr/bin/env bash
# Launcher: rsync VENA repo → server-3, run 3D-DiT 4-epoch smoke inside a
# detached GNU screen session, return immediately.
#
# 3D-DiT (Peebles & Xie 2023 backbone + Eidex 2025 §4 RFlow recipe) has no
# VGG perceptual loss — no cache pre-warm step.
#
# Citation: arXiv:2212.09748; arXiv:2509.24194.
#
# Usage:
#   bash launcher_dit_3d_server3_4ep.sh             # submit
#   bash launcher_dit_3d_server3_4ep.sh --dry-run   # print plan, no rsync, no ssh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SSH_ALIAS="${SSH_ALIAS:-icai-server}"
REMOTE_REPO="${REMOTE_REPO:-/home/mariopascual/projects/VENA}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/mariopascual/.conda/envs/vena/bin/python}"
REMOTE_CONFIG="${REMOTE_REPO}/routines/competitors/dit_3d/configs/smoke_server3_4ep.yaml"
SESSION="${SESSION:-vena-dit-3d-smoke}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/media/hddb/mario/smoke_logs/competitors/dit_3d}"
GPU_ID="${GPU_ID:-0}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# rsync excludes: keep heavy data out, but DO include src/external (vendored
# upstream is needed at runtime — runner imports DiT3DWrapper from it).
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
    # T1C-RFlow vendored 80MB VAE checkpoint (not needed; we use MAISI-V2).
    --exclude="src/external/t1c_rflow/upstream/checkpoints/"
)

echo "[plan] sync ${REPO_DIR}/ → ${SSH_ALIAS}:${REMOTE_REPO}/"
echo "[plan] screen session: ${SESSION} on ${SSH_ALIAS}, GPU=${GPU_ID}"
echo "[plan] config: ${REMOTE_CONFIG}"
echo "[plan] logs:   ${REMOTE_LOG_DIR}/${SESSION}.log"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Rsync
echo "[step 1/2] rsync …"
rsync -azP --delete-after "${EXCLUDES[@]}" "${REPO_DIR}/" "${SSH_ALIAS}:${REMOTE_REPO}/"

# 2) Launch in detached GNU screen.
echo "[step 2/2] launch detached screen session ${SESSION} …"
REMOTE_CMD="cd '${REMOTE_REPO}' \
&& mkdir -p '${REMOTE_LOG_DIR}' \
&& export PYTHONPATH='${REMOTE_REPO}/src:${REMOTE_REPO}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& '${REMOTE_PYTHON}' -m routines.competitors.dit_3d.cli '${REMOTE_CONFIG}' 2>&1 | tee -a '${REMOTE_LOG_DIR}/${SESSION}.log'"

ssh "${SSH_ALIAS}" "screen -dmS '${SESSION}' bash -c \"${REMOTE_CMD}; echo 'dit-3d-train completed-screen-marker'; sleep 2\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh ${SSH_ALIAS} screen -r ${SESSION}"
echo "  list:     ssh ${SSH_ALIAS} screen -ls"
echo "  logs:     ssh ${SSH_ALIAS} tail -F ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh ${SSH_ALIAS} screen -X -S ${SESSION} quit"
echo ""
echo "Watch sentinel: 'dit-3d-train completed' in the training log."
