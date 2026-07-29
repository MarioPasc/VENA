#!/usr/bin/env bash
# Picasso worker for §18 v4 three-arm ablation (L1 / L2 / Huber).
#
# Resources: 2x A100 40 GB (cuda:0 training, cuda:1 async exhaustive val),
# 16 CPUs, 256 GB RAM, 6 days walltime.
#
# --mem=256G basis: every prior FM training job on Picasso (v3a, v3b, v3b-rw,
#   s3-k5-standard) completed with ReqMem=256G (sacct verified 2026-07-29).
#   The smoke-derived RSS (15.9 GB) does not extrapolate to production: full
#   9-cohort streaming over multi-GB H5 files with num_workers=8 has a higher
#   working set, and --mem=64G has already OOM-killed another pipeline on this
#   cluster (inference benchmark, 2026-07-14). Match what completed.
#
# --time=6-00:00:00: measured v3a cost = 60.1 h / 400 k steps; 800 k steps
#   → ~120.2 h = 5.01 days. 6-day limit gives 20 % headroom and queues faster
#   than the 7-day partition cap. Arms use resume_from: baseline so NO
#   auto-resubmit fires — all three arms must finish within one submission.
#
# Parameterised entirely by environment variables exported by the launcher:
#   CONDA_ENV_NAME   conda env name (default: vena)
#   REPO_DIR         absolute path to the VENA repo on fscratch
#   CONFIG_PATH      absolute path to the run YAML

#SBATCH --time=6-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpu_partition
# a100, NOT dgx: BOTH the A100 nodes (exa[01-04]) and the B200 nodes
# (blk[01-02]) advertise the `dgx` feature, so a bare `--constraint=dgx`
# silently schedules onto a B200 — observed on blk01 (2026-07-24). VENA FM
# training fits comfortably in 40 GB, so there is no reason to take a B200
# slot, and a matrix of runs split across two GPU generations is not
# comparable. (`--gres=gpu:A100:2` matches NO node: the A100 nodes advertise
# an UNTYPED `gpu:8`; only B200 is typed. `dgx&a100` is an invalid spec.)
#SBATCH --constraint=a100
#SBATCH --gres=gpu:2
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/%x_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/%x_%j.err

# set -u is intentionally absent as a precaution: CONDA_ENV_NAME is
# overridable (default: vena), and a future invocation pointing this worker
# at a dedicated compiler-toolchain env (e.g. vena-comp, which carries
# gxx_linux-64) would die at conda-activate under -u because activate.d/
# activate-gcc_linux-64.sh dereferences SYS_SYSROOT while it is unbound.
# The main vena env activates safely under -u (tested 2026-07-29, Picasso).
# -eo pipefail removes the footgun for free; revert only after confirming
# CONDA_ENV_NAME will never point at a compiler-toolchain env.
# (See memory note: feedback_conda_activate_set_u.md.)
set -eo pipefail
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
# v4 §18 arms all carry resume_from: baseline, so this block will always hit
# the "not resubmitting" branch. Kept for consistency with the shared worker.
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
