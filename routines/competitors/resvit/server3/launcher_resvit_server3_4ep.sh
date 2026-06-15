#!/usr/bin/env bash
# Launcher: rsync VENA repo → server-3, ensure ViT .npz cached, run ResViT
# 4-epoch smoke (stage 1: 1+1, stage 2: 1+1) inside a detached GNU screen
# session, return immediately.
#
# Usage:
#   bash launcher_resvit_server3_4ep.sh             # submit
#   bash launcher_resvit_server3_4ep.sh --dry-run   # print plan, no rsync, no ssh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SSH_ALIAS="${SSH_ALIAS:-icai-server}"
REMOTE_REPO="${REMOTE_REPO:-/home/mariopascual/projects/VENA}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/mariopascual/.conda/envs/vena/bin/python}"
REMOTE_CONFIG="${REMOTE_REPO}/routines/competitors/resvit/configs/smoke_server3_4ep.yaml"
SESSION="${SESSION:-vena-resvit-smoke}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/media/hddb/mario/smoke_logs/competitors/resvit}"
GPU_ID="${GPU_ID:-0}"
TORCH_HOME_REMOTE="${TORCH_HOME_REMOTE:-/home/mariopascual/.cache/torch}"
VIT_NPZ_REMOTE="${REMOTE_REPO}/src/external/resvit/upstream/checkpoints/R50+ViT-B_16.npz"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# rsync includes src/external (vendored ResViT code is needed at runtime). The
# 88 MB R50+ViT-B_16.npz inside src/external/resvit/upstream/checkpoints/ is
# small enough to ship across; no LFS-style exclusion needed for ResViT.
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
echo "[plan] screen session: ${SESSION} on ${SSH_ALIAS}, GPU=${GPU_ID}"
echo "[plan] config: ${REMOTE_CONFIG}"
echo "[plan] logs:   ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "[plan] ViT npz: ${VIT_NPZ_REMOTE}"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Rsync repo (carries the 88 MB ViT .npz with it).
echo "[step 1/3] rsync …"
rsync -azP --delete-after "${EXCLUDES[@]}" "${REPO_DIR}/" "${SSH_ALIAS}:${REMOTE_REPO}/"

# 2) Verify the ViT npz arrived (server-3 has internet — fall back to re-download).
echo "[step 2/3] verify R50+ViT-B_16.npz cache on server-3 …"
ssh "${SSH_ALIAS}" "if [[ ! -f '${VIT_NPZ_REMOTE}' ]]; then \
    echo '[warn] ViT npz missing on server-3 — re-downloading'; \
    mkdir -p \"\$(dirname '${VIT_NPZ_REMOTE}')\"; \
    curl -sSL -o '${VIT_NPZ_REMOTE}' https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz; \
fi && ls -la '${VIT_NPZ_REMOTE}'"

# 3) Launch in detached GNU screen (tmux not installed on icai-server).
echo "[step 3/3] launch detached screen session ${SESSION} …"
ssh "${SSH_ALIAS}" "mkdir -p '${REMOTE_LOG_DIR}'"
REMOTE_CMD="cd '${REMOTE_REPO}' \
&& export PYTHONPATH='${REMOTE_REPO}/src:${REMOTE_REPO}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export TORCH_HOME='${TORCH_HOME_REMOTE}' \
&& '${REMOTE_PYTHON}' -m routines.competitors.resvit.cli '${REMOTE_CONFIG}' 2>&1 | tee -a '${REMOTE_LOG_DIR}/${SESSION}.log'"

ssh "${SSH_ALIAS}" "screen -dmS '${SESSION}' bash -c \"${REMOTE_CMD}; echo 'resvit-train completed-screen-marker'; sleep 2\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh ${SSH_ALIAS} screen -r ${SESSION}"
echo "  list:     ssh ${SSH_ALIAS} screen -ls"
echo "  logs:     ssh ${SSH_ALIAS} tail -F ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh ${SSH_ALIAS} screen -X -S ${SESSION} quit"
