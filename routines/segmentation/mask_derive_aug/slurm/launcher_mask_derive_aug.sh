#!/usr/bin/env bash
# ============================================================================
# Launcher — mask_derive_aug: cache masks/tumor_latent_soft into all 6
# offline-augmented latent H5s on Picasso (CPU-only array, 1 task/cohort).
# ============================================================================
# Usage:
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA
#   bash routines/segmentation/mask_derive_aug/slurm/launcher_mask_derive_aug.sh
#   bash routines/segmentation/mask_derive_aug/slurm/launcher_mask_derive_aug.sh --dry-run
#
# DO NOT submit this script manually — the launcher is the entry point.
# Run sbatch --test-only first if in doubt:
#   sbatch --test-only routines/segmentation/mask_derive_aug/slurm/worker_mask_derive_aug.sh
#
# Resource notes (pending D4 measurement — update --time once measured):
#   SDT on (192,224,192) @ ~10-15 s/scan; BraTS-GLI (largest cohort):
#   ~2800 aug rows × 15 s ≈ 11.7 h → 1-00:00:00 is conservative.
#   Pre-allocation peaks at ~12 GB for BraTS-GLI; --mem=48G covers all cohorts.
#   All 6 tasks run concurrently (--array=0-5 with no throttle).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
# VENA-validation, NOT the shared repos/VENA: the shared repo carries a stale
# HEAD (this routine does not exist there at all), and `vena` is pip-installed
# EDITABLE pointing at repos/VENA/src — so the worker's PYTHONPATH export is
# what keeps `vena` and `routines` on the same tree. See the memory
# `reference_picasso_split_brain_imports`.
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation"
CONDA_ENV_PATH="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs_seg"
# ---------------------------------------------------------------------------

mkdir -p "${LOGS_DIR}"

# Strip ANSI colour codes from sbatch --parsable output (Picasso's Lua wrapper
# injects colour codes that corrupt --dependency if used downstream).
_clean_job_id() {
    sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g' <<<"$1"
}

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --array=0-5 \
    --output=${LOGS_DIR}/mask_derive_aug_%A_%a.out \
    --error=${LOGS_DIR}/mask_derive_aug_%A_%a.err \
    --export=ALL,\
REPO_DIR=${REPO_DIR},\
CONDA_ENV_PATH=${CONDA_ENV_PATH},\
LOGS_DIR=${LOGS_DIR} \
    ${SCRIPT_DIR}/worker_mask_derive_aug.sh"

echo "Submitting mask_derive_aug array (6 tasks: cohorts 0-5)"
echo "Command: ${SBATCH_CMD}"

if ${DRY_RUN}; then
    echo "[DRY-RUN] Not submitting."
    exit 0
fi

RAW_ID=$(eval "${SBATCH_CMD}")
JOB_ID=$(_clean_job_id "${RAW_ID}")

[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "FATAL: unparsable job id from sbatch: '${RAW_ID}'" >&2
    exit 1
}

echo ""
echo "Submitted array job ${JOB_ID} (6 tasks: 0-5)"
echo "Monitor tasks : /usr/bin/squeue -j ${JOB_ID}"
echo "Logs dir      : ${LOGS_DIR}"
echo ""
echo "Per-cohort tasks:"
echo "  task 0 (UCSF-PDGM)  : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_0.out"
echo "  task 1 (BraTS-GLI)  : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_1.out"
echo "  task 2 (UPENN-GBM)  : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_2.out"
echo "  task 3 (IvyGAP)     : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_3.out"
echo "  task 4 (LUMIERE)    : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_4.out"
echo "  task 5 (REMBRANDT)  : ${LOGS_DIR}/mask_derive_aug_${JOB_ID}_5.out"
echo ""
echo "Cancel all: scancel ${JOB_ID}"
