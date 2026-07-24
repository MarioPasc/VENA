#!/usr/bin/env bash
# Submit the S2 T-13 oracle matrix (J0-J4) to Picasso.
#
#   J0  trunk=freeze  tc=1   ControlNet-only floor (no trunk drift, no loss help)
#   J1  trunk=joint   tc=1   equal-weight ceiling; J0 -> J1 = the freeze -> joint gain
#   J2  trunk=joint   tc=5   region-weight sweep
#   J3  trunk=joint   tc=10  region-weight sweep
#   J4  trunk=joint   tc=20  region-weight sweep (WATCH FP-SAFETY at the top weight)
#
# The five YAMLs are a verified one-variable matrix: 104 leaf config keys, an
# identical key-set, and exactly three keys differing across arms
# (run.tag, model.trunk.trainable, loss.cfm.region_weights.tc).
#
# REPO_DIR is VENA-validation, NOT the shared repos/VENA: a long-running job
# (the v3a-a6 ablation) is executing out of repos/VENA, and its async
# exhaustive-val subprocess re-imports the tree at every cadence epoch — so
# rsyncing over it mid-run would change that job's code and its recorded
# git_sha. VENA-validation is a real git repo, so `git rev-parse` resolves and
# provenance is recorded correctly.
#
# Usage:
#   bash launcher_picasso_s2_t13_matrix.sh --test-only   # sbatch --test-only, submits nothing
#   bash launcher_picasso_s2_t13_matrix.sh --dry-run     # print the sbatch commands
#   bash launcher_picasso_s2_t13_matrix.sh               # submit all five
#   bash launcher_picasso_s2_t13_matrix.sh --only J2      # submit one arm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation}"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
WORKER="${SCRIPT_DIR}/worker_fm_train_picasso_s2.sh"

# arm : job-name : config basename
ARMS=(
  "J0:vena-s2-t13-j0-freeze-tc1:picasso_s2_t13_j0_v3a_cn2ch_freeze_fft.yaml"
  "J1:vena-s2-t13-j1-joint-tc1:picasso_s2_t13_j1_v3a_cn2ch_joint_tc1_fft.yaml"
  "J2:vena-s2-t13-j2-joint-tc5:picasso_s2_t13_j2_v3a_cn2ch_joint_tc5_fft.yaml"
  "J3:vena-s2-t13-j3-joint-tc10:picasso_s2_t13_j3_v3a_cn2ch_joint_tc10_fft.yaml"
  "J4:vena-s2-t13-j4-joint-tc20:picasso_s2_t13_j4_v3a_cn2ch_joint_tc20_fft.yaml"
)

MODE="submit"
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   MODE="dry" ;;
        --test-only) MODE="test" ;;
        --only)      ONLY="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# Picasso's sbatch wrapper emits ANSI colour codes even with --parsable; an
# unstripped id interpolated into a later flag is ACCEPTED by sbatch and then
# silently ignored. Strip, then assert numeric.
_clean_job_id() { sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g' <<<"$1"; }

[[ "${MODE}" == "dry" ]] || mkdir -p "${LOGS_DIR}"

SUBMITTED=()
for entry in "${ARMS[@]}"; do
    IFS=":" read -r ARM JOB_NAME CFG <<<"${entry}"
    [[ -n "${ONLY}" && "${ONLY}" != "${ARM}" ]] && continue

    CONFIG_PATH="${REPO_DIR}/routines/fm/train/configs/runs/${CFG}"
    EXPORTS="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${CONFIG_PATH}"

    if [[ "${MODE}" == "dry" ]]; then
        echo "[DRY-RUN] ${ARM}  ${JOB_NAME}"
        echo "          sbatch --parsable -J ${JOB_NAME} --export=${EXPORTS} ${WORKER}"
        continue
    fi

    # An unsatisfiable resource request fails here instantly and loudly;
    # submitted live it is indistinguishable from ordinary queue pressure.
    if ! OUT=$(sbatch --test-only -J "${JOB_NAME}" --export="${EXPORTS}" "${WORKER}" 2>&1); then
        echo "FATAL: sbatch --test-only rejected ${ARM}:" >&2
        echo "${OUT}" >&2
        exit 1
    fi
    echo "[test-only OK] ${ARM}: ${OUT}"
    [[ "${MODE}" == "test" ]] && continue

    RAW=$(sbatch --parsable -J "${JOB_NAME}" --export="${EXPORTS}" "${WORKER}")
    JOB_ID=$(_clean_job_id "${RAW}")
    if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "FATAL: unparsable job id for ${ARM}: ${RAW@Q}" >&2
        exit 1
    fi
    echo "Submitted ${ARM} ${JOB_NAME} job ${JOB_ID}"
    echo "  logs: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.{out,err}"
    SUBMITTED+=("${JOB_ID}")
done

if [[ ${#SUBMITTED[@]} -gt 0 ]]; then
    echo ""
    echo "JOB_IDS=$(IFS=,; echo "${SUBMITTED[*]}")"
    echo "Monitor: sacct -j $(IFS=,; echo "${SUBMITTED[*]}") -X -o JobID,JobName%28,State,Elapsed,NodeList"
fi
