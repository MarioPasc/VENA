#!/usr/bin/env bash
#SBATCH -J vena-spatial-residual
#SBATCH --partition=cpu_partition
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Worker for vena spatial_residual on Picasso CPU nodes (no GPU required).
#
# Env vars (from launcher):
#   REPO_DIR        absolute path to the VENA worktree on Picasso scratch
#   CONFIG_PATH     absolute path to the YAML config
#   PYTHON          optional override for the Python binary (default: conda env)
#
# IMPORTANT: do NOT use set -u.  The vena conda env contains gxx_linux-64;
# its activation dereferences SYS_SYSROOT which is unbound and kills the job.
# Use set -eo pipefail only.
set -eo pipefail

START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}

# Use the full path to Python to avoid conda activation entirely.
CONDA_ENV_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena
PYTHON=${PYTHON:-${CONDA_ENV_DIR}/bin/python}

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "=========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date -u +%FT%TZ)"
echo "REPO_DIR:    ${REPO_DIR}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "PYTHON:      ${PYTHON}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT
# ============================================================================
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# Limit per-process thread count; scipy/sklearn use OpenMP.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[env] python: ${PYTHON}"
"${PYTHON}" --version

# Fail fast if Python is too old.
"${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' || {
    echo "[FATAL] Python < 3.11 at ${PYTHON}" >&2; exit 1
}

echo "[env] scipy: $("${PYTHON}" -c 'import scipy; print(scipy.__version__)')"
echo "[env] pandas: $("${PYTHON}" -c 'import pandas; print(pandas.__version__)')"
echo ""

# ============================================================================
# RUN
# ============================================================================
"${PYTHON}" -m routines.validation.spatial_residual.cli "${CONFIG_PATH}"

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
