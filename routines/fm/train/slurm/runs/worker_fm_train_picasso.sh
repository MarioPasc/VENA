#!/usr/bin/env bash
# Shared Picasso worker for production VENA FM training runs.
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
# ``run.resume_from`` value other than ``baseline`` / null. For
# ``resume_from: latest`` (CONTINUE) the run dir is reused natively.
# For ``resume_from: <run_id>`` or ``resume_from: <abs path>``
# (WARM_START) the engine auto-promotes to CONTINUE on resubmit when
# the recipe-matching sibling dir already exists with ``last.ckpt`` —
# so the second launch continues in place rather than re-warm-starting
# from the external source.

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpu_partition
#SBATCH --constraint=dgx
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

# GPU info — expect 2x A100 40 GB.
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
# AUTO-RESUBMIT ON SIGTERM (124) OR TIME LIMIT (142)
# ============================================================================
# Picasso may kill the job at the time cap. We resubmit on any
# ``resume_from`` value other than ``baseline`` / null:
#   * ``latest`` / ``best``   → CONTINUE (engine reuses dir).
#   * ``<run_id>`` / abs path → WARM_START on first launch, auto-promoted to
#                                CONTINUE on the resubmit by the engine (a
#                                recipe-matching sibling dir now carries
#                                ``last.ckpt``).
# RC values: 124 = python timeout, 137 = SIGKILL, 143 = SIGTERM.
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
