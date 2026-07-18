#!/usr/bin/env bash
# Launch the spatial_residual full sweep as a SLURM array + merge pipeline.
#
# Design (per orchestrator spec):
#   One array task per (method, cohort) at each method's selection_nfe.
#   --array=0-<N-1>%<MAX_CONCURRENT> — bounded concurrency.
#   --cpus-per-task=4 per shard; 16 for the merge node.
#   No --gres → stays clear of the gres/gpu group cap.
#   Merge job depends on afterok:<array_job_id> — runs only when all tasks succeed.
#
# Usage (Picasso login node):
#   bash routines/validation/spatial_residual/slurm/launcher_spatial_residual_sweep.sh
#   bash ... --dry-run        # print sbatch commands without submitting
#   bash ... --allow-partial  # merge even if some shard tasks failed
#
# Required env (override with env vars):
#   REPO_DIR        — absolute path to the VENA worktree clone on Picasso
#   CONDA_ENV_PATH  — absolute path to the vena conda env
#   DATA_ROOT       — inference tree root (contains shard sub-directories)
#   SWEEP_ROOT      — output root for shards + merged artifact
#
# DO NOT SUBMIT THE SWEEP YOURSELF — the orchestrator owns sweep submission.
# This script is ready; wait for the go-ahead.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena}"
DATA_ROOT="${DATA_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference}"
SWEEP_ROOT="${SWEEP_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/spatial_residual_sweep}"
LOGS_DIR="${LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/routines/validation/spatial_residual/configs/picasso_sweep.yaml}"
MAX_CONCURRENT="${MAX_CONCURRENT:-60}"

DRY_RUN=false
ALLOW_PARTIAL_FLAG=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run)       DRY_RUN=true ;;
        --allow-partial) ALLOW_PARTIAL_FLAG="--allow-partial" ;;
        -h|--help)
            grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#$//'
            exit 0
            ;;
    esac
done

echo "[launcher] REPO_DIR       = ${REPO_DIR}"
echo "[launcher] CONDA_ENV_PATH = ${CONDA_ENV_PATH}"
echo "[launcher] DATA_ROOT      = ${DATA_ROOT}"
echo "[launcher] SWEEP_ROOT     = ${SWEEP_ROOT}"
echo "[launcher] CONFIG_PATH    = ${CONFIG_PATH}"
echo "[launcher] MAX_CONCURRENT = ${MAX_CONCURRENT}"

for path in "${REPO_DIR}" "${CONDA_ENV_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[FATAL] required path missing: ${path}" >&2
        exit 1
    fi
done

PYTHON="${CONDA_ENV_PATH}/bin/python"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

SHARD_DIR="${SWEEP_ROOT}/shards"
OUTPUT_ROOT="${SWEEP_ROOT}/analyses"
MANIFEST="${SWEEP_ROOT}/manifest.csv"

mkdir -p "${SHARD_DIR}" "${OUTPUT_ROOT}" "${LOGS_DIR}"

# ---- Step 1: Generate manifest ----
echo ""
echo "[launcher] Generating manifest from ${DATA_ROOT} …"
MANIFEST_OUT=$(
    "${PYTHON}" -m routines.validation.spatial_residual.cli_manifest \
        --data-root "${DATA_ROOT}" \
        --output    "${MANIFEST}" \
    2>&1
)
echo "${MANIFEST_OUT}"

# Parse N_TASKS from the structured stdout line "manifest_tasks=N".
N_TASKS=$(echo "${MANIFEST_OUT}" | grep '^manifest_tasks=' | cut -d= -f2)
if [[ -z "${N_TASKS}" || "${N_TASKS}" -eq 0 ]]; then
    echo "[FATAL] Manifest is empty — check DATA_ROOT and discover_shards output." >&2
    exit 1
fi
LAST_TASK=$((N_TASKS - 1))
echo "[launcher] Manifest: ${N_TASKS} tasks (IDs 0–${LAST_TASK})"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

# ---- Step 2: Submit the shard array job ----
ARRAY_CMD=(
    sbatch
    --parsable
    --job-name="vena-sr-sweep-${RUN_TAG}"
    --array="0-${LAST_TASK}%${MAX_CONCURRENT}"
    --output="${LOGS_DIR}/sr_sweep_${RUN_TAG}_%A_%a.out"
    --error="${LOGS_DIR}/sr_sweep_${RUN_TAG}_%A_%a.err"
    --export="ALL,REPO_DIR=${REPO_DIR},CONDA_ENV_PATH=${CONDA_ENV_PATH},MANIFEST=${MANIFEST},SHARD_DIR=${SHARD_DIR},CONFIG_PATH=${CONFIG_PATH}"
    "${SCRIPT_DIR}/worker_spatial_residual_sweep.sh"
)

# ---- Step 3: Submit the merge job (depends on array) ----
MERGE_CMD=(
    sbatch
    --parsable
    --job-name="vena-sr-merge-${RUN_TAG}"
    --output="${LOGS_DIR}/sr_merge_${RUN_TAG}_%j.out"
    --error="${LOGS_DIR}/sr_merge_${RUN_TAG}_%j.err"
    --export="ALL,REPO_DIR=${REPO_DIR},CONDA_ENV_PATH=${CONDA_ENV_PATH},MANIFEST=${MANIFEST},SHARD_DIR=${SHARD_DIR},SWEEP_ROOT=${SWEEP_ROOT},CONFIG_PATH=${CONFIG_PATH},OUTPUT_ROOT=${OUTPUT_ROOT}"
    "${SCRIPT_DIR}/worker_spatial_residual_merge.sh"
)

if ${DRY_RUN}; then
    echo ""
    echo "[DRY-RUN] Array command:"
    echo "  ${ARRAY_CMD[*]}"
    echo ""
    echo "[DRY-RUN] Merge command (dependency placeholder <ARRAY_JOB_ID>):"
    echo "  ${MERGE_CMD[*]} --dependency=afterok:<ARRAY_JOB_ID>"
    exit 0
fi

# Picasso wraps sbatch and colours its output, so even --parsable emits ANSI
# escape codes around the job ID.  Interpolated raw into --dependency, that
# yields "afterok:<ESC>[31m<ESC>[0m1604488", which sbatch ACCEPTS and silently
# records as Dependency=(null) — the merge then runs immediately against a
# partial shard set.  Strip the escape codes and assert the ID is bare digits
# before trusting it.
_clean_job_id() {
    # shellcheck disable=SC2001
    sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g' <<<"$1"
}

ARRAY_JOB_ID_RAW=$("${ARRAY_CMD[@]}")
ARRAY_JOB_ID=$(_clean_job_id "${ARRAY_JOB_ID_RAW}")
if [[ ! "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] could not parse array job ID from sbatch output: ${ARRAY_JOB_ID_RAW@Q}" >&2
    exit 1
fi
echo ""
echo "[launcher] Array job:  ${ARRAY_JOB_ID} (tasks 0–${LAST_TASK}, concurrency ${MAX_CONCURRENT})"

# Wire the merge dependency after confirming the array job was accepted.
MERGE_CMD+=(--dependency="afterok:${ARRAY_JOB_ID}")
MERGE_JOB_ID_RAW=$("${MERGE_CMD[@]}")
MERGE_JOB_ID=$(_clean_job_id "${MERGE_JOB_ID_RAW}")
echo "[launcher] Merge job:  ${MERGE_JOB_ID} (depends on afterok:${ARRAY_JOB_ID})"

# A dependency sbatch failed to record is worse than no merge job at all: it
# runs unguarded.  Verify SLURM actually stored it.
DEP_RECORDED=$(scontrol show job "${MERGE_JOB_ID}" 2>/dev/null | grep -o 'Dependency=[^ ]*' | head -1)
echo "[launcher] SLURM recorded: ${DEP_RECORDED:-<unreadable>}"
if [[ "${DEP_RECORDED}" == "Dependency=(null)" ]]; then
    echo "[FATAL] merge job ${MERGE_JOB_ID} has NO dependency — it will run against" >&2
    echo "        an incomplete shard set. Cancelling it; fix the ID parsing." >&2
    scancel "${MERGE_JOB_ID}" 2>/dev/null || true
    exit 1
fi
echo ""
echo "Monitor:  squeue -j ${ARRAY_JOB_ID}"
echo "Merge:    squeue -j ${MERGE_JOB_ID}"
echo "Logs:     ${LOGS_DIR}/"
echo "Output:   ${OUTPUT_ROOT}/"
echo "Cancel:   scancel ${ARRAY_JOB_ID} ${MERGE_JOB_ID}"
echo ""
echo "Resubmit failed shards:"
echo "  sbatch --array=<failed_ids>%${MAX_CONCURRENT} \\"
echo "    --export=\"ALL,REPO_DIR=${REPO_DIR},...\" \\"
echo "    ${SCRIPT_DIR}/worker_spatial_residual_sweep.sh"
echo ""
echo "Manual merge after resubmit:"
echo "  bash ${SCRIPT_DIR}/worker_spatial_residual_merge.sh"
