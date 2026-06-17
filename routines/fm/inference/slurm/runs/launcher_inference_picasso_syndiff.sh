#!/usr/bin/env bash
# Submit the SynDiff (C3) inference companion to Picasso A100 DGX.
#
# Runs in the `vena-syndiff` env (Python 3.10 + torch 2.5.1+cu121 + the
# JIT-compiled fused / upfirdn2d CUDA extensions). The main `picasso_full`
# inference job (launcher_inference_picasso_full.sh) excludes C3, so the
# two SLURM jobs together produce the full benchmark.
#
# Usage:
#   bash launcher_inference_picasso_syndiff.sh             # submit
#   bash launcher_inference_picasso_syndiff.sh --dry-run   # print sbatch command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena-syndiff}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/fm/inference/configs/picasso_full_syndiff.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
# Persistent JIT-compile cache for fused.so / upfirdn2d.so. Same path
# convention as the SynDiff training launcher so the cache is reused.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/.cache/torch_extensions/vena-syndiff}"
mkdir -p "${LOGS_DIR}" "${TORCH_EXTENSIONS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Sanity-check the env is built before queueing — fail-fast saves SLURM
# slot from immediately erroring on missing python.
PYTHON="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${PYTHON} not found." >&2
    echo "Build the vena-syndiff env first — see .claude/notes/validation/syndiff.md." >&2
    exit 1
fi

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/inference_picasso_syndiff_%j.out \
    --error=${LOGS_DIR}/inference_picasso_syndiff_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH},TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR} \
    ${SCRIPT_DIR}/worker_inference_picasso_syndiff.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (VENA inference SynDiff companion, vena-syndiff env)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/inference_picasso_syndiff_${JOB_ID}.{out,err}"
echo "Sentinel: 'inference routine complete' in the routine log."
