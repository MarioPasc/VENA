#!/usr/bin/env bash
# ============================================================================
# SLURM worker — mask_derive_aug (CPU-only, 1 task per aug-bank cohort)
# ============================================================================
# Submitted via launcher_mask_derive_aug.sh — do NOT invoke directly.
#
# Task → cohort mapping (0-indexed):
#   0: UCSF-PDGM
#   1: BraTS-GLI   (largest; drives --time budget)
#   2: UPENN-GBM
#   3: IvyGAP
#   4: LUMIERE
#   5: REMBRANDT
#
# Resource sizing:
#   SDT on (192,224,192) float32 ≈ 10-15 s/scan.
#   BraTS-GLI (largest): ~2800 aug rows × 15 s ≈ 11.7 h.
#   --time=1-00:00:00 is conservative; tighten after D4 measurement.
#   Peak memory: pre-allocated masks_out for BraTS-GLI ≈ 12 GB;
#   --mem=48G covers all cohorts with headroom.
#   CPU: 8 cores (OMP/MKL parallelism for scipy distance_transform_edt;
#   processing loop is sequential but scipy releases GIL internally).
#
# Usage (do NOT submit manually — use the launcher):
#   Submitted by launcher with --array=0-5 and
#   --export=ALL,REPO_DIR=...,CONDA_ENV_PATH=...,LOGS_DIR=...
# ============================================================================
#SBATCH -J vena-mask-derive-aug
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu

set -euo pipefail

START_TIME=$(date +%s)

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "============================================================"
echo "SLURM job id   : ${SLURM_JOB_ID}"
echo "Array task id  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : $(hostname)"
echo "Start          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo dir       : ${REPO_DIR}"
echo "============================================================"

# ============================================================================
# VALIDATE REQUIRED ENV VARS
# ============================================================================
: "${REPO_DIR:?REPO_DIR must be set by the launcher}"
: "${LOGS_DIR:?LOGS_DIR must be set by the launcher}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena}"
PYTHON="${CONDA_ENV_PATH}/bin/python"

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

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# scipy's distance_transform_edt releases the GIL and uses OpenMP internally;
# honour the SLURM allocation rather than oversubscribing.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# RESOLVE COHORT CONFIG FROM ARRAY TASK ID
# ============================================================================
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Associative array: task index → per-cohort YAML (relative to REPO_DIR)
declare -A COHORT_CONFIGS
COHORT_CONFIGS[0]="routines/segmentation/mask_derive_aug/configs/cohorts/ucsf_pdgm.yaml"
COHORT_CONFIGS[1]="routines/segmentation/mask_derive_aug/configs/cohorts/brats_gli.yaml"
COHORT_CONFIGS[2]="routines/segmentation/mask_derive_aug/configs/cohorts/upenn_gbm.yaml"
COHORT_CONFIGS[3]="routines/segmentation/mask_derive_aug/configs/cohorts/ivy_gap.yaml"
COHORT_CONFIGS[4]="routines/segmentation/mask_derive_aug/configs/cohorts/lumiere.yaml"
COHORT_CONFIGS[5]="routines/segmentation/mask_derive_aug/configs/cohorts/rembrandt.yaml"

declare -A COHORT_NAMES
COHORT_NAMES[0]="UCSF-PDGM"
COHORT_NAMES[1]="BraTS-GLI"
COHORT_NAMES[2]="UPENN-GBM"
COHORT_NAMES[3]="IvyGAP"
COHORT_NAMES[4]="LUMIERE"
COHORT_NAMES[5]="REMBRANDT"

[[ -v COHORT_CONFIGS["${TASK_ID}"] ]] || {
    echo "[FATAL] Unknown task id ${TASK_ID}; valid range is 0-5." >&2
    exit 1
}

COHORT_CONFIG="${REPO_DIR}/${COHORT_CONFIGS[${TASK_ID}]}"
COHORT_NAME="${COHORT_NAMES[${TASK_ID}]}"

[[ -f "${COHORT_CONFIG}" ]] || {
    echo "[FATAL] Config not found: ${COHORT_CONFIG}" >&2
    exit 1
}

echo "Cohort (task ${TASK_ID}): ${COHORT_NAME}"
echo "Config : ${COHORT_CONFIG}"
echo ""

# ============================================================================
# COMMAND
# ============================================================================
echo "Launching: ${PYTHON} -m routines.segmentation.mask_derive_aug.cli ${COHORT_CONFIG}"
"${PYTHON}" -m routines.segmentation.mask_derive_aug.cli "${COHORT_CONFIG}"
EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
WALL_SECS=$(( END_TIME - START_TIME ))
echo "============================================================"
echo "Cohort      : ${COHORT_NAME}"
echo "Exit code   : ${EXIT_CODE}"
echo "Wall time   : ${WALL_SECS} s ($(( WALL_SECS / 3600 ))h $(( (WALL_SECS % 3600) / 60 ))m)"
echo "End         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
exit "${EXIT_CODE}"
