#!/usr/bin/env bash
# ============================================================================
# Launcher — segmentation training, SegResNet FLOOR arm, 6-task array (folds 0-4 + all_train)
# ============================================================================
# Usage:
#   bash launcher_picasso_seg_segresnet.sh             # submit
#   bash launcher_picasso_seg_segresnet.sh --dry-run   # print sbatch command, no submit
#
# Picasso note: sbatch emits ANSI colour codes even with --parsable.
# Strip them before using the job ID in --dependency.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation"
# Arm production config — worker reads this and renders per-task YAMLs from it.
# Identical to the UKB arm in 59/62 config fields: same K=5 folds and fold_seed=1337
# (so the fold assignment is byte-identical and the arms are comparable per fold),
# same AdamW/cosine/lr/batch/patch/AMP, same DML+CE loss and deep-supervision
# weights, same epochs/patience, same seed. Only model.name, model.checkpoint
# (null = random init) and run.tag differ. This is the scratch FLOOR arm.
ARM_CONFIG="${REPO_DIR}/routines/segmentation/train/configs/runs/picasso_seg_segresnet.yaml"
CONDA_ENV_PATH="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs_seg"
# ---------------------------------------------------------------------------

mkdir -p "${LOGS_DIR}"

# Strip ANSI colour codes + non-digit characters from sbatch --parsable output.
# Picasso's sbatch wrapper injects colour codes that silently corrupt --dependency.
_clean_job_id() {
    sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g' <<<"$1"
}

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_CMD="sbatch --parsable \
    --array=0-5 \
    --output=${LOGS_DIR}/seg_segresnet_%A_%a.out \
    --error=${LOGS_DIR}/seg_segresnet_%A_%a.err \
    --export=ALL,\
REPO_DIR=${REPO_DIR},\
ARM_CONFIG=${ARM_CONFIG},\
CONDA_ENV_PATH=${CONDA_ENV_PATH},\
LOGS_DIR=${LOGS_DIR} \
    ${SCRIPT_DIR}/worker_picasso_seg.sh"

echo "Submitting segmentation training array (SegResNet floor arm, 6 tasks: folds 0-4 + all_train)"
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

echo "Submitted array job ${JOB_ID} (6 tasks: 0-5)"
echo "Monitor tasks : /usr/bin/squeue -j ${JOB_ID}"
echo "Logs dir      : ${LOGS_DIR}"
echo "Log pattern   : ${LOGS_DIR}/seg_segresnet_${JOB_ID}_<task>.out"
echo ""
echo "Individual task logs:"
for TASK in 0 1 2 3 4 5; do
    FOLD=$( [ "${TASK}" -eq 5 ] && echo "all_train" || echo "${TASK}" )
    echo "  task ${TASK} (fold ${FOLD}): ${LOGS_DIR}/seg_segresnet_${JOB_ID}_${TASK}.out"
done
