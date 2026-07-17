#!/usr/bin/env bash
# Submit the paired_fidelity smoke run to Picasso (cpu_partition — no GPU needed).
#
# Usage (Picasso login node):
#   bash routines/validation/paired_fidelity/slurm/launcher_paired_fidelity_smoke.sh
#   bash routines/validation/paired_fidelity/slurm/launcher_paired_fidelity_smoke.sh --dry-run
#   REPO_DIR=/path/to/other/clone bash ...
#
# IMPORTANT — rsync the worktree first:
#   rsync -av --exclude='.git/' \
#       <WORKTREE>/ \
#       picasso:<REPO_DIR>/
#
# Smoke config: configs/smoke_picasso.yaml
#   3 methods × 3 cohorts, ~381 scan pairs, ~33 min on 8 cores.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-pf-agent}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena}"
LOGS_DIR="${LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs}"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
CONFIG_PATH=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#$//'
            exit 0
            ;;
        *) CONFIG_PATH="${arg}" ;;
    esac
done
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/routines/validation/paired_fidelity/configs/smoke_picasso.yaml}"

echo "[launcher] REPO_DIR        = ${REPO_DIR}"
echo "[launcher] CONDA_ENV_PATH  = ${CONDA_ENV_PATH}"
echo "[launcher] CONFIG_PATH     = ${CONFIG_PATH}"
echo "[launcher] LOGS_DIR        = ${LOGS_DIR}"

for path in "${REPO_DIR}" "${CONDA_ENV_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[FATAL] required path missing: ${path}" >&2
        echo "[hint]  rsync the worktree or check REPO_DIR / CONDA_ENV_PATH." >&2
        exit 1
    fi
done

JOB_NAME="vena-pf-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out"
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}",CONDA_ENV_PATH="${CONDA_ENV_PATH}"
    "${SCRIPT_DIR}/worker_paired_fidelity_smoke.sh"
)

if ${DRY_RUN}; then
    echo
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}")
echo
echo "Submitted job ${JOB_ID} (name: ${JOB_NAME})"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
