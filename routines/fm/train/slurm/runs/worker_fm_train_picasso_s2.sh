#!/usr/bin/env bash
# Picasso worker for the S2 T-13 oracle matrix (J0-J4).
#
# Forked from worker_fm_train_picasso.sh with ONE deliberate change:
#
#   --constraint=dgx   ->   --constraint=a100
#
# WHY (measured 2026-07-24, S5): `dgx` is a feature satisfied by BOTH the A100
# nodes (exa[01-04], untyped Gres=gpu:8, feature `a100`) AND the B200 nodes
# (blk[01-02], typed gpu:B200:8). A bare `--constraint=dgx` therefore silently
# schedules onto a B200 — observed on blk01. For a five-arm matrix whose whole
# purpose is a one-variable comparison, letting SLURM split the arms across two
# GPU generations would inject a hardware confound into the S3 verdict, so the
# node type is pinned. (`--gres=gpu:A100:2` matches NO node at all: the A100
# nodes advertise untyped gres. `dgx&a100` is an invalid feature spec.)
#
# Resources: 2x A100 40 GB (cuda:0 training, cuda:1 async exhaustive val),
# 16 CPUs, 256 GB RAM, 7 days walltime.
#
# Parameterised entirely by environment variables exported by the launcher:
#   CONDA_ENV_NAME   conda env name (default: vena)
#   REPO_DIR         absolute path to the VENA repo on fscratch
#   CONFIG_PATH      absolute path to the run YAML
#
# Auto-resubmits itself on SIGTERM whenever the YAML carries a
# ``run.resume_from`` value other than ``baseline`` / null. Every S2 arm
# warm-starts from an absolute ckpt path, so the engine auto-promotes the
# resubmit to CONTINUE once a recipe-matching sibling dir carries last.ckpt.

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpu_partition
#SBATCH --constraint=a100
#SBATCH --gres=gpu:2
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/%x_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/%x_%j.err

set -euo pipefail
START_TIME=$(date +%s)

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}  (${SLURM_JOB_NAME:-fm-train})"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Working dir:  $(pwd)"
echo "Config:       ${CONFIG_PATH}"
echo "Repo:         ${REPO_DIR}"
echo "Conda env:    ${CONDA_ENV_NAME}"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT
# ============================================================================
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# GPU info — expect 2x A100 40 GB. Printed so the log itself records which
# hardware the arm actually landed on (the constraint is pinned, but silence
# is not proof).
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

# ============================================================================
# COMMAND
# ============================================================================
set +e
python -m routines.fm.train.cli "${CONFIG_PATH}"
RC=$?
set -e

# ============================================================================
# AUTO-RESUBMIT ON SIGTERM (143) / SIGKILL (137) / TIMEOUT (124)
# ============================================================================
RESUBMIT_RC_SET="124 137 143"
if echo " ${RESUBMIT_RC_SET} " | grep -q " ${RC} "; then
    RESUME_VAL=$(grep -E "^[[:space:]]*resume_from:[[:space:]]" "${CONFIG_PATH}" \
                 | head -n 1 | sed -E "s/^[[:space:]]*resume_from:[[:space:]]*//; s/[[:space:]]+$//" \
                 | tr -d "\"'")
    case "${RESUME_VAL}" in
        ""|"baseline"|"null"|"~")
            echo "[auto-resubmit] python exited rc=${RC} but CONFIG resume_from='${RESUME_VAL}'; not resubmitting"
            ;;
        *)
            echo "[auto-resubmit] python exited rc=${RC}; resubmitting self (resume_from='${RESUME_VAL}')"
            sbatch \
                -J "${SLURM_JOB_NAME:-fm-train}" \
                --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}" \
                "$0" || echo "[auto-resubmit] sbatch failed; manual resubmission required"
            ;;
    esac
fi

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)  (python rc=${RC})"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
exit ${RC}
