#!/usr/bin/env bash
# Launch the SynDiff t1pre → t1c 2-epoch smoke on loginexa (V100 sm_70).
#
# Usage:
#   bash launcher_syndiff_loginexa_t1pre_2ep.sh             # submit
#   bash launcher_syndiff_loginexa_t1pre_2ep.sh --dry-run   # print plan
#
# Notes
# -----
# - Uses the dedicated `vena-v100-syndiff` conda env (V100 cu121 wheels +
#   StyleGAN2 fused-op build deps). The main `vena-v100` env is untouched.
#   Create the env once via the recipe in .claude/notes/validation/syndiff.md.
# - No VGG pre-warm — SynDiff has no perceptual loss.
# - First import compiles upfirdn2d / fused_bias_act under TORCH_EXTENSIONS_DIR;
#   persisting the cache is critical because V100s share the box and rebuilds
#   are slow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
CONDA_SH="${CONDA_SH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100-syndiff}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/syndiff/configs/smoke_loginexa_t1pre_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/competitors/syndiff}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/.cache/torch_extensions/vena-v100-syndiff}"
SESSION="${SESSION:-vena-syndiff-loginexa-t1pre-smoke}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}" "${TORCH_EXTENSIONS_DIR}"

if [[ -z "${GPU_ID:-}" ]]; then
    GPU_ID=$(ssh loginexa "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null" \
        | sort -t',' -k2,2 -nr | head -1 | awk -F',' '{print $1}' | tr -d ' ')
    GPU_ID="${GPU_ID:-0}"
fi
echo "[plan] loginexa GPU=${GPU_ID}"
echo "[plan] tmux session: ${SESSION}"
echo "[plan] conda env:    ${CONDA_ENV_PATH}"
echo "[plan] config:       ${CONFIG_PATH}"
echo "[plan] log:          ${LOG_DIR}/${SESSION}.log"
echo "[plan] ext cache:    ${TORCH_EXTENSIONS_DIR}"

if ${DRY_RUN}; then
    exit 0
fi

# Sanity-check that the env exists.
ssh loginexa "test -d '${CONDA_ENV_PATH}'" || {
    echo "ERROR: ${CONDA_ENV_PATH} not found on loginexa — create the vena-v100-syndiff env first (see .claude/notes/validation/syndiff.md)." >&2
    exit 1
}

# conda activate in the remote shell so ninja / nvcc / cpp_extension
# subprocesses find them via PATH; set CUDA_HOME=$CONDA_PREFIX so the JIT
# extension build links against the in-env cuda-toolkit.
REMOTE_CMD="source '${CONDA_SH}' && conda activate '${CONDA_ENV_PATH}' \
&& cd '${REPO_DIR_REMOTE}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& export CUDA_HOME=\$CONDA_PREFIX \
&& export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
&& export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
&& export TORCH_CUDA_ARCH_LIST=7.0 \
&& export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}' \
&& python -m routines.competitors.syndiff.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh loginexa "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach: ssh loginexa tmux attach -t ${SESSION}"
echo "  logs:   ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:   ssh loginexa tmux kill-session -t ${SESSION}"
