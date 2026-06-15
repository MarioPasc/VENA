#!/usr/bin/env bash
#SBATCH -J vena-resvit-picasso-full
# 1.5-day wallclock: at batch=4, paper-budget caps yield ~4 h of training
# (stage 1: 62 500 steps × 120 ms ≈ 2.1 h; stage 2: 31 250 steps × 200 ms ≈ 1.7 h).
# 1.5 days reserves ~9× slack for step-time variance and unexpected delays.
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/resvit_picasso_full_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/resvit_picasso_full_%j.err

set -euo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Working dir:  $(pwd)"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

module_loaded=0
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
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"

PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[fatal] python interpreter not found: ${PYTHON}"
    exit 1
fi

# Hard-fail before any GPU work if the ViT .npz isn't present on the compute
# node's view of the shared FS. The compute nodes have no internet, so the
# launcher must have placed it under VIT_NPZ already.
VIT_NPZ_DEFAULT="${REPO_DIR}/src/external/resvit/upstream/checkpoints/R50+ViT-B_16.npz"
VIT_NPZ="${VIT_NPZ:-${VIT_NPZ_DEFAULT}}"
if [[ ! -f "${VIT_NPZ}" ]]; then
    echo "[fatal] ViT init checkpoint not found at ${VIT_NPZ}."
    echo "[fatal] Pre-warm on the login node (which has internet) before sbatch:"
    echo "  curl -sSL -o '${VIT_NPZ}' https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
    exit 1
fi
echo "[ok] ViT .npz: $(ls -la "${VIT_NPZ}")"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

"${PYTHON}" -m routines.competitors.resvit.cli "${CONFIG_PATH}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
