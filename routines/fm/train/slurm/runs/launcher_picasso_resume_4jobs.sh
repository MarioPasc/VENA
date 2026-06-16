#!/usr/bin/env bash
# Submit the four resume / warm-start jobs together:
#
#   (1) CONTINUE  2026-06-12_01-27-55_s1_fft_cfm_c9b97556         (same dir)
#   (2) CONTINUE  2026-06-12_06-32-25_s2_fft_contrastive_a27962e7 (same dir)
#   (3) WARM_START from s1_fft_cfm ema_best.ckpt  → s2 + FFT     (new dir,
#                                                                tag _s1warm)
#   (4) WARM_START from s1_fft_cfm ema_best.ckpt  → s2 + LoRA r16(new dir,
#                                                                tag _s1warm)
#
# All four use the shared worker (worker_fm_train_picasso.sh); each gets a
# distinct SLURM job name so the log files at
# /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/<name>_<jobid>.{out,err}
# stay separable.
#
# Usage:
#   bash launcher_picasso_resume_4jobs.sh             # submit all four
#   bash launcher_picasso_resume_4jobs.sh --dry-run   # print sbatch cmds
#   bash launcher_picasso_resume_4jobs.sh 1 3         # submit only jobs 1 and 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/worker_fm_train_picasso.sh"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-vena}"
export REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
CFG_DIR="${REPO_DIR}/routines/fm/train/configs/runs"

# job_name : config_filename
JOB_1_NAME="vena-cont-s1-fft-cfm"
JOB_1_CFG="${CFG_DIR}/picasso_continue_s1_fft_cfm.yaml"

JOB_2_NAME="vena-cont-s2-fft-contr"
JOB_2_CFG="${CFG_DIR}/picasso_continue_s2_fft_contrastive.yaml"

JOB_3_NAME="vena-warm-s2-fft"
JOB_3_CFG="${CFG_DIR}/picasso_warm_s2_fft_from_s1.yaml"

JOB_4_NAME="vena-warm-s2-lora16"
JOB_4_CFG="${CFG_DIR}/picasso_warm_s2_lora_r16_from_s1.yaml"

DRY_RUN=false
SELECTED=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        1|2|3|4)   SELECTED+=("${arg}") ;;
        *)         echo "unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=(1 2 3 4)
fi

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

submit_one() {
    local idx="$1" name="$2" cfg="$3"
    if [[ ! -f "${cfg}" ]]; then
        echo "[skip] ${name}: config ${cfg} not found" >&2
        return 1
    fi
    local cmd
    cmd="sbatch --parsable -J ${name} \
        --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},CONFIG_PATH=${cfg} \
        ${WORKER}"
    if ${DRY_RUN}; then
        echo "[DRY-RUN ${idx}] ${cmd}"
        echo "          CONFIG = ${cfg}"
        return 0
    fi
    local job_id
    job_id=$(eval "${cmd}")
    echo "Submitted ${name} (job ${job_id})"
    echo "  config : ${cfg}"
    echo "  logs   : ${LOGS_DIR}/${name}_${job_id}.{out,err}"
}

for s in "${SELECTED[@]}"; do
    case "${s}" in
        1) submit_one 1 "${JOB_1_NAME}" "${JOB_1_CFG}" ;;
        2) submit_one 2 "${JOB_2_NAME}" "${JOB_2_CFG}" ;;
        3) submit_one 3 "${JOB_3_NAME}" "${JOB_3_CFG}" ;;
        4) submit_one 4 "${JOB_4_NAME}" "${JOB_4_CFG}" ;;
    esac
done

if ! ${DRY_RUN}; then
    echo ""
    echo "Monitor:  squeue -u \${USER}"
fi
