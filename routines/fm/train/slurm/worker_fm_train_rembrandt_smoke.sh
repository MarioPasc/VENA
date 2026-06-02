#!/usr/bin/env bash
#SBATCH -J vena-rembrandt-smoke
#SBATCH --time=0-02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/rembrandt_smoke_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/rembrandt_smoke_%j.err

set -euo pipefail

START_TIME=$(date +%s)

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Working dir:  $(pwd)"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT
# ============================================================================
module_loaded=0
# Picasso's conda module is named `miniconda/3`; fall back to common variants
# used on other clusters so the same worker is portable.
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
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

# GPU info — expect 2× A100 (training pins to cuda:0, exhaustive_val to cuda:1).
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

# ============================================================================
# COMMAND
# ============================================================================
python -m routines.fm.train.cli "${CONFIG_PATH}"

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
