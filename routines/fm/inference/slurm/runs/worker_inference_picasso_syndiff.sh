#!/usr/bin/env bash
#SBATCH -J vena-inference-picasso-syndiff
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --constraint=a100
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/inference_picasso_syndiff_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/inference_picasso_syndiff_%j.err

# VENA SynDiff inference companion to picasso_full inference. Runs C3 only
# in the vena-syndiff env (Python 3.10 + torch 2.5.1+cu121). The CUDA env
# vars below match the SynDiff training worker so the cached fused.so /
# upfirdn2d.so JIT artefacts are reused.

set -eo pipefail

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

# SynDiff JIT-compiled CUDA extensions (StyleGAN2 fused_bias_act + upfirdn2d).
# These env vars match the SynDiff TRAINING worker so the same cached .so
# files are reused — recompilation against a fresh inference toolchain
# would re-trigger ninja and burn 1-2 minutes per first-import.
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions/vena-syndiff}"

PYTHON="${PYTHON:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/${CONDA_ENV_NAME}/bin/python}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

"${PYTHON}" -m routines.fm.inference.cli "${CONFIG_PATH}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
