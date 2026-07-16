#!/usr/bin/env bash
#SBATCH -J vena-downstream-seg
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

# Worker for the downstream-seg routine on Picasso.
#
# Resource rationale:
#   --mem=32G    SlidingWindowInferer roi=(240,240,160) peak ~12G host RAM;
#                32G gives 2× headroom for batch loading + h5py caches.
#   --gres=gpu:1 --constraint=a100
#                BraTS SegResNet ~1.8GB VRAM; A100-40GB is more than enough.
#                Do NOT use --constraint=dgx — that also matches B200 nodes
#                which are incompatible with the cu124 conda env.
#   --time=4h    Smoke (5 scans × 1 method): ~10 min.
#                Full sweep (all methods × all cohorts): up to ~4h on A100.
#
# Env vars (set by launcher):
#   REPO_DIR        absolute path to VENA clone on Picasso
#   CONFIG_PATH     absolute path to the routine YAML
#   CONDA_ENV_NAME  conda env name (default: vena)

set -euo pipefail
START_TIME=$(date +%s)

REPO_DIR=${REPO_DIR:?missing REPO_DIR}
CONFIG_PATH=${CONFIG_PATH:?missing CONFIG_PATH}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-vena}

# ============================================================================
# JOB HEADER
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
    # Picasso may expose the env directly at a fixed path.
    CONDA_PREFIX="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}"
    if [[ -d "${CONDA_PREFIX}" ]]; then
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
    else
        echo "[FATAL] conda not found and ${CONDA_PREFIX} missing" >&2
        exit 1
    fi
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[env] python: $(which python)"
python --version

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
python -m routines.validation.downstream_seg.cli "${CONFIG_PATH}"

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date -u +%FT%TZ)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
