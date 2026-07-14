#!/usr/bin/env bash
# Submit the VENA test-set inference benchmark to Picasso as five concurrent
# A100 shards. PREDICTIONS ONLY — no metrics are computed here.
#
# Why five shards: one job for all 16 method-rows serialises to ~3 days because
# C6-LDDPM sweeps NFE=1000 (1035 sampler steps/patient vs 38 for a VENA row).
# Splitting by cost cuts the critical path to the longest single shard (~36 h)
# and stops a failure in one family from costing the whole sweep.
#
# Each shard carries its OWN run_id_tag under a SHARED output_root, so the jobs
# are write-disjoint (the engine writes
# <output_root>/<run_id_tag>/{predictions,logs,figures,decision.json}; a shared
# tag would race five jobs on one log and one decision.json). The downstream
# metrics routine merges by globbing <output_root>/*/predictions/.
#
# Shard E (SynDiff) runs in the vena-syndiff env (Python 3.10 + torch
# 2.5.1+cu121 + JIT-compiled fused/upfirdn2d CUDA extensions) and therefore
# uses its own worker script.
#
# GPU: every shard pins A100 via `--gres=gpu:1 --constraint=a100`. Picasso gained
# a B200 (Blackwell, sm_100) cluster and a bare `gpu:1` can land there; the `vena`
# env is cu124 and dies on B200 with "no kernel image is available".
# The pin MUST be the node feature, not a GRES type — sinfo shows the A100 nodes
# (exa01-04) expose an UNTYPED `gpu:8`, so `--gres=gpu:A100:1` matches no node and
# sbatch rejects it with "Requested node configuration is not available". Only the
# B200 nodes are typed (`gpu:B200:8`). `--constraint=dgx` is not enough — both
# A100 and B200 nodes carry the dgx feature.
#
# Usage:
#   bash launcher_inference_picasso_shards.sh             # submit all five
#   bash launcher_inference_picasso_shards.sh --dry-run   # print sbatch cmds
#   bash launcher_inference_picasso_shards.sh a_cheap d_lddpm   # submit a subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs"
CFG_DIR="${REPO_DIR}/routines/fm/inference/configs"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

# shard | config basename | conda env | walltime | worker
# Walltimes are ~2x the estimate so a slow queue/node never truncates a shard.
SHARDS=(
    "a_cheap|picasso_shard_a_cheap.yaml|vena|06:00:00|worker_inference_picasso_full.sh"
    "b_vena|picasso_shard_b_vena.yaml|vena|1-00:00:00|worker_inference_picasso_full.sh"
    "c_latent|picasso_shard_c_latent.yaml|vena|1-00:00:00|worker_inference_picasso_full.sh"
    "d_lddpm|picasso_shard_d_lddpm.yaml|vena|3-00:00:00|worker_inference_picasso_full.sh"
    "e_syndiff|picasso_full_syndiff.yaml|vena-syndiff|12:00:00|worker_inference_picasso_syndiff.sh"
)

WANTED=("$@")
want() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local s
    for s in "${WANTED[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

SUBMITTED=()
for row in "${SHARDS[@]}"; do
    IFS="|" read -r shard cfg env walltime worker <<<"${row}"
    want "${shard}" || continue

    config_path="${CFG_DIR}/${cfg}"
    job_name="vena-inf-${shard}"

    sbatch_cmd="sbatch --parsable \
        -J ${job_name} \
        --time=${walltime} \
        --gres=gpu:1 --constraint=a100 \
        --output=${LOGS_DIR}/inference_${shard}_%j.out \
        --error=${LOGS_DIR}/inference_${shard}_%j.err \
        --export=ALL,CONDA_ENV_NAME=${env},REPO_DIR=${REPO_DIR},CONFIG_PATH=${config_path} \
        ${SCRIPT_DIR}/${worker}"

    if ${DRY_RUN}; then
        echo "[DRY-RUN] ${shard}: ${sbatch_cmd}"
        continue
    fi

    job_id=$(eval "${sbatch_cmd}")
    SUBMITTED+=("${job_id} ${job_name}")
    echo "Submitted ${job_id}  ${job_name}  (${walltime}, env=${env})"
done

${DRY_RUN} && exit 0

echo ""
echo "Monitor:   squeue -u \$USER -o '%.10i %.18j %.8T %.10M %.6D %R'"
echo "Logs:      ${LOGS_DIR}/inference_<shard>_<jobid>.{out,err}"
echo "Output:    /mnt/home/users/tic_163_uma/mpascual/execs/vena/inference/<run_id_tag>/predictions/"
echo "Sentinel:  'inference routine complete' in each shard's routine log."
