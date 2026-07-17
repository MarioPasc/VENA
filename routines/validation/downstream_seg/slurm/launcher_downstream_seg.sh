#!/usr/bin/env bash
# Submit the downstream-seg validation routine to Picasso.
#
# Usage (login node):
#   bash routines/validation/downstream_seg/slurm/launcher_downstream_seg.sh
#   bash routines/validation/downstream_seg/slurm/launcher_downstream_seg.sh path/to/config.yaml
#   bash routines/validation/downstream_seg/slurm/launcher_downstream_seg.sh --dry-run
#
# Default config: configs/smoke.yaml (IvyGAP × C0-Identity, <10 min).
# Full sweep:     configs/default.yaml — all methods × all Ring-A/B cohorts.
#
# Shard by method: pass one config per method and submit multiple jobs:
#   for m in VENA-S1-v3b-rw C0-Identity; do
#     yaml=/tmp/ds_${m}.yaml
#     sed "s/methods: \[\]/methods: [${m}]/" configs/default.yaml > "${yaml}"
#     bash launcher_downstream_seg.sh "${yaml}"
#   done
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
LOGS_DIR="${LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs}"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
CONFIG_PATH=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#$//'
            exit 0
            ;;
        *) CONFIG_PATH="${arg}" ;;
    esac
done
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/routines/validation/downstream_seg/configs/smoke.yaml}"

echo "[launcher] REPO_DIR    = ${REPO_DIR}"
echo "[launcher] CONFIG_PATH = ${CONFIG_PATH}"
echo "[launcher] CONDA_ENV   = ${CONDA_ENV_NAME}"
echo "[launcher] LOGS_DIR    = ${LOGS_DIR}"

for path in "${REPO_DIR}" "${CONFIG_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[FATAL] required path missing: ${path}" >&2
        exit 1
    fi
done

# Pure-bash top-level scalar extraction from a flat YAML file.
_yaml_get() {
    local file="$1" key="$2"
    grep -E "^[[:space:]]*${key}[[:space:]]*:" "${file}" 2>/dev/null \
        | head -1 \
        | sed -E "s/^[[:space:]]*${key}[[:space:]]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]+$//; s/^[\"']//; s/[\"']$//"
}

_validate_yaml_paths() {
    local cfg="$1" fatal=0
    # bundle_path must exist; inference_root and output_root are created on
    # demand, so we warn rather than hard-fail if they are missing.
    for key in bundle_path; do
        val="$(_yaml_get "${cfg}" "${key}")"
        if [[ -z "${val}" || "${val}" == "null" ]]; then
            echo "[FATAL] '${key}' is not set in ${cfg}" >&2
            fatal=1
            continue
        fi
        if [[ ! -e "${val}" ]]; then
            echo "[FATAL] '${key}' points at a missing path: ${val}" >&2
            fatal=1
        fi
    done
    for key in inference_root; do
        val="$(_yaml_get "${cfg}" "${key}")"
        if [[ -n "${val}" && "${val}" != "null" && ! -e "${val}" ]]; then
            echo "[WARN] '${key}' does not exist yet: ${val}" >&2
        fi
    done
    if [[ ${fatal} -ne 0 ]]; then
        echo "[hint] fix the path in ${cfg} or rsync the missing artefact in." >&2
        return 1
    fi
    echo "[launcher] YAML paths validated."
    return 0
}
_validate_yaml_paths "${CONFIG_PATH}" || exit 1

JOB_NAME="vena-downstream-seg-$(date -u +%Y%m%dT%H%M%SZ)"

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out"
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}",CONDA_ENV_NAME="${CONDA_ENV_NAME}"
    "${SCRIPT_DIR}/worker_downstream_seg.sh"
)

if ${DRY_RUN}; then
    echo
    echo "[DRY-RUN] ${SBATCH_CMD[*]}"
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}")
echo
echo "Submitted job ${JOB_ID} (name: ${JOB_NAME})"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
