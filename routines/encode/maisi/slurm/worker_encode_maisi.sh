#!/usr/bin/env bash
#SBATCH -J vena-encode-maisi
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# Worker for the VENA MAISI encode routine on Picasso.
#
# Env vars (from launcher):
#   REPO_DIR        absolute path to the VENA clone on Picasso
#   CONFIG_PATH     absolute path to the routine YAML
#   CONDA_ENV_NAME  conda env name (default: vena)

set -euo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-vena}

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "=========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date -u +%FT%TZ)"
echo "REPO_DIR:    ${REPO_DIR}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "CONDA_ENV:   ${CONDA_ENV_NAME}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# CONDA ENVIRONMENT
# ============================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[env] python: $(which python)"
python --version

# VENA targets Python ≥ 3.11 (pyproject.toml). Fail fast otherwise.
python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' || {
    echo "[FATAL] conda env '${CONDA_ENV_NAME}' has $(python --version 2>&1)." >&2
    echo "[FATAL] VENA requires Python >= 3.11." >&2
    exit 1
}

python -c "import torch; print(f'[env] torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

# ============================================================================
# RUN
# ============================================================================
python -m routines.encode.maisi.cli "${CONFIG_PATH}"

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
