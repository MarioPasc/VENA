#!/usr/bin/env bash
# Submit the full SynDiff t1pre → t1c training to Picasso A100 DGX.
#
# Usage:
#   bash launcher_syndiff_picasso_full_t1pre.sh             # submit
#   bash launcher_syndiff_picasso_full_t1pre.sh --dry-run   # print sbatch command, no submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena-syndiff}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
export CONFIG_PATH="${REPO_DIR}/routines/competitors/syndiff/configs/picasso_full_t1pre.yaml"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/.cache/torch_extensions/vena-syndiff}"
mkdir -p "${LOGS_DIR}" "${TORCH_EXTENSIONS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Sanity-check the env is built before queueing — failing here saves a 7-day
# SLURM slot from immediately erroring on missing python.
PYTHON="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${PYTHON} not found." >&2
    echo "Build the vena-syndiff env first — see .claude/notes/validation/syndiff.md." >&2
    exit 1
fi

SBATCH_CMD="sbatch --parsable \
    --output=${LOGS_DIR}/syndiff_picasso_full_t1pre_%j.out \
    --error=${LOGS_DIR}/syndiff_picasso_full_t1pre_%j.err \
    --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH},TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR} \
    ${SCRIPT_DIR}/worker_syndiff_picasso_full_t1pre.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

JOB_ID=$(eval "${SBATCH_CMD}")
echo "Submitted job ${JOB_ID} (SynDiff t1pre → t1c full training)"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/syndiff_picasso_full_t1pre_${JOB_ID}.{out,err}"
