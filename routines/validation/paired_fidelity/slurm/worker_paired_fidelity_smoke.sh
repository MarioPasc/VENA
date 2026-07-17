#!/usr/bin/env bash
#SBATCH -J vena-pf-smoke
#SBATCH --partition=cpu_partition
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Worker for the paired_fidelity smoke run on Picasso (CPU-only).
#
# ~381 scan pairs (3 methods × 3 cohorts × selection NFE).
# At 41 CPU-s/scan with 8 cores → ~33 min.  Budget: 2h.
#
# Env vars (from launcher):
#   REPO_DIR          absolute path to the VENA worktree on Picasso
#   CONFIG_PATH       absolute path to the routine YAML
#   CONDA_ENV_PATH    absolute path to the vena conda env directory
#
# IMPORTANT: do NOT add 'set -u' here.  The gxx_linux-64 package in the
# vena conda env leaves SYS_SYSROOT unbound during activation, which kills
# the job under 'set -u'.  Use 'set -eo pipefail' only.

set -eo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena}

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date -u +%FT%TZ)"
echo "REPO_DIR:     ${REPO_DIR}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "CONDA_ENV:    ${CONDA_ENV_PATH}"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem:          ${SLURM_MEM_PER_NODE:-?} MB"
echo "=========================================="

# ============================================================================
# CONDA ENVIRONMENT
# activate via the env's activate script; avoids module system dependency
# ============================================================================
PYTHON="${CONDA_ENV_PATH}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[FATAL] python not found at ${PYTHON}" >&2
    echo "[hint]  check CONDA_ENV_PATH" >&2
    exit 1
fi

echo "[env] python: ${PYTHON}"
"${PYTHON}" --version

"${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' || {
    echo "[FATAL] VENA requires Python >= 3.11." >&2
    exit 1
}

# ============================================================================
# ENVIRONMENT
# ============================================================================
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[env] PYTHONPATH=${PYTHONPATH}"
echo ""

# Import isolation check (detects split-brain: routines from worktree but
# vena from the main checkout).
"${PYTHON}" -c "
import pathlib, vena, routines
wt = pathlib.Path('${REPO_DIR}').resolve()
for m in (vena, routines):
    p = pathlib.Path(m.__file__).resolve()
    assert p.is_relative_to(wt), f'LEAK: {m.__name__} -> {p}'
print('import isolation OK')
" || {
    echo "[FATAL] import isolation check failed — vena or routines loaded from main checkout." >&2
    echo "[hint]  check PYTHONPATH and REPO_DIR." >&2
    exit 1
}

# ============================================================================
# RUN
# ============================================================================
echo "[run] Starting paired_fidelity smoke …"
"${PYTHON}" -m routines.validation.paired_fidelity.cli "${CONFIG_PATH}"

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
