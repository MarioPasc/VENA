#!/usr/bin/env bash
# Launch the T1C-RFlow 2-epoch smoke on the Picasso loginexa V100 interactive node.
#
# Invocation: run this from the Picasso login node. It SSH-hops into loginexa,
# starts a detached tmux session, and returns immediately. T1C-RFlow has no
# VGG perceptual loss, so no cache pre-warm step is required.
#
# Loginexa is NOT a SLURM partition — it is a standalone SSH-accessible
# interactive node (4 × Tesla V100-DGXS-32GB). The 30-min budget is convention,
# not a hard kill; do not run anything that exceeds it.
#
# Citation: Eidex et al. 2025, arXiv:2509.24194.
#
# Usage:
#   bash launcher_t1c_rflow_loginexa_2ep.sh             # submit
#   bash launcher_t1c_rflow_loginexa_2ep.sh --dry-run   # print plan, no ssh
#
# Override GPU via GPU_ID env var (default: auto-pick freest by memory).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100/bin/python}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/competitors/t1c_rflow/configs/smoke_loginexa_2ep.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/VENA/logs/competitors/t1c_rflow}"
SESSION="${SESSION:-vena-t1c-rflow-loginexa-smoke}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}"

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

# Launch detached tmux on loginexa.
echo "[launch] detached tmux session on loginexa …"
REMOTE_CMD="cd '${REPO_DIR_REMOTE}' \
&& export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\$PYTHONPATH \
&& export PYTHONUNBUFFERED=1 \
&& export CUDA_VISIBLE_DEVICES=${GPU_ID} \
&& '${PYTHON}' -m routines.competitors.t1c_rflow.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'"

ssh loginexa "tmux new-session -d -s '${SESSION}' \"${REMOTE_CMD}\""

echo ""
echo "Submitted ✓"
echo "  attach:   ssh loginexa tmux attach -t ${SESSION}"
echo "  list:     ssh loginexa tmux ls"
echo "  logs:     ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:     ssh loginexa tmux kill-session -t ${SESSION}"
echo ""
echo "Watch sentinel: 't1c-rflow-train completed' in the training log."
