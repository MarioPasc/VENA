#!/usr/bin/env bash
# Launcher: rsync VENA repo → server-3, run SynDiff 4-epoch smoke (t1pre→t1c)
# inside a detached screen session, return immediately.
#
# Usage:
#   bash launcher_syndiff_server3_t1pre_4ep.sh             # submit
#   bash launcher_syndiff_server3_t1pre_4ep.sh --dry-run   # print plan, no rsync, no ssh
#
# Notes
# -----
# - Uses the dedicated `vena-syndiff` conda env (StyleGAN2 fused-op build deps
#   live there; main `vena` env is untouched). User must create the env once
#   per machine — see .claude/notes/validation/syndiff.md for the recipe.
# - No VGG pre-warm — SynDiff has no perceptual loss.
# - First import compiles `utils/op/upfirdn2d_kernel.cu` + `fused_bias_act_kernel.cu`
#   via ninja; the launcher exports TORCH_EXTENSIONS_DIR so the build cache
#   survives across runs.
# - screen is used because tmux is not installed on icai-server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SSH_ALIAS="${SSH_ALIAS:-icai-server}"
REMOTE_REPO="${REMOTE_REPO:-/home/mariopascual/projects/VENA}"
CONDA_SH="${CONDA_SH:-/opt/anaconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vena-syndiff}"
REMOTE_CONFIG="${REMOTE_REPO}/routines/competitors/syndiff/configs/smoke_server3_t1pre_4ep.yaml"
SESSION="${SESSION:-vena-syndiff-smoke}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/media/hddb/mario/smoke_logs/competitors/syndiff}"
GPU_ID="${GPU_ID:-0}"
TORCH_EXTENSIONS_DIR_REMOTE="${TORCH_EXTENSIONS_DIR_REMOTE:-/media/hddb/mario/.cache/torch_extensions/vena-syndiff}"

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
)

echo "[plan] sync ${REPO_DIR}/ → ${SSH_ALIAS}:${REMOTE_REPO}/"
echo "[plan] screen session: ${SESSION} on ${SSH_ALIAS}, GPU=${GPU_ID}"
echo "[plan] conda env: ${CONDA_ENV} (activated via ${CONDA_SH})"
echo "[plan] config: ${REMOTE_CONFIG}"
echo "[plan] logs:   ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "[plan] torch extensions cache: ${TORCH_EXTENSIONS_DIR_REMOTE}"

if ${DRY_RUN}; then
    exit 0
fi

# 1) Rsync
echo "[step 1/3] rsync …"
rsync -azP --delete-after "${EXCLUDES[@]}" "${REPO_DIR}/" "${SSH_ALIAS}:${REMOTE_REPO}/"

# 2) Verify the dedicated env exists.
echo "[step 2/3] verify ${CONDA_ENV} env on server-3 …"
ssh "${SSH_ALIAS}" "mkdir -p '${TORCH_EXTENSIONS_DIR_REMOTE}' '${REMOTE_LOG_DIR}' && test -f '${CONDA_SH}' && test -d '/home/mariopascual/.conda/envs/${CONDA_ENV}' || { echo 'ERROR: conda env ${CONDA_ENV} not found.' >&2; exit 1; }"

# 3) Launch in detached screen session.
# CRITICAL: conda activate must run inside the remote shell so that ninja /
# nvcc / cpp_extension subprocesses pick them up via PATH. Setting CUDA_HOME
# to CONDA_PREFIX makes torch's cpp_extension.load() find the cuda-toolkit
# we installed in the env (rather than the stale /usr/local/cuda-11.6 nvcc
# on the system PATH).
echo "[step 3/3] launch detached screen session ${SESSION} …"
REMOTE_CMD="source '${CONDA_SH}' && conda activate '${CONDA_ENV}' \
&& cd '${REMOTE_REPO}' \
&& export PYTHONPATH='${REMOTE_REPO}/src:${REMOTE_REPO}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export CUDA_HOME=\$CONDA_PREFIX \
&& export TORCH_CUDA_ARCH_LIST=8.9 \
&& export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR_REMOTE}' \
&& python -m routines.competitors.syndiff.cli '${REMOTE_CONFIG}' 2>&1 | tee -a '${REMOTE_LOG_DIR}/${SESSION}.log'"

ssh "${SSH_ALIAS}" "screen -dmS '${SESSION}' bash -c \"${REMOTE_CMD}; echo 'syndiff-train completed-screen-marker'; sleep 2\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh ${SSH_ALIAS} screen -r ${SESSION}"
echo "  list:     ssh ${SSH_ALIAS} screen -ls"
echo "  logs:     ssh ${SSH_ALIAS} tail -F ${REMOTE_LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh ${SSH_ALIAS} screen -X -S ${SESSION} quit"
