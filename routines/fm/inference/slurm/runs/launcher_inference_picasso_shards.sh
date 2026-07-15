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

# HOST-RAM SIZING (2026-07-14, load-bearing). The first submission attempt ran
# with the worker's default --mem=64G and all three shards were OOM_KILLED at
# ~10 min having written NOTHING. Host RAM, not VRAM. Two additive consumers:
#
#  1. `_build_reference_cache` (engine.py:181) is built ONCE, UP FRONT, for every
#     patient of every cohort: 4 reference volumes + 2 masks ~= 161 MB/patient
#     x 393 test patients ~= 63 GB before a single prediction is made. This alone
#     exceeded 64G, which is why even the 1-NFE a_cheap shard died.
#  0. `per_cohort_selection_pred` (engine.py) caches one DECODED selection-NFE
#     tensor per (cohort, patient, method) for the best/worst comparison figures,
#     and is not freed until figure rendering at the very end. It therefore
#     scales with METHOD COUNT, so the a_cheap shard — the cheapest per-method but
#     with the MOST methods (6) — peaks highest: 6 x 393 x ~36 MB ~= 84 GB on top
#     of the ~63 GB reference cache = ~147 GB, which OOM-killed it at --mem=150G
#     (job 1574657, 41/48 files). Hence a_cheap alone is bumped to 300G. The real
#     fix is to free the tensor cache per method once its figure is rendered.
#  2. `records_by_nfe` accumulates a whole cohort x every NFE before flushing.
#     A PerPatientRecord is ~232 MB (6 float32 volumes + 2 int8 masks), and the
#     reference volumes are duplicated into EVERY NFE's record. Peak is the
#     largest cohort, BraTS-GLI at 114 patients:
#         a_cheap   1 NFE  x 114 x 232 MB =  26 GB
#         b_vena    5 NFE  x 114 x 232 MB = 132 GB
#         c_latent  6 NFE  x 114 x 232 MB = 158 GB
#         d_lddpm   4 NFE  x 114 x 232 MB = 105 GB
#         e_syndiff 1 NFE  x 114 x 232 MB =  26 GB
#
# --mem below is (63 GB cache + the row above) with ~30% headroom. The exa nodes
# carry 900 GB each, so this is comfortably affordable; do not trim it back to a
# round 64G. The real fix is to stream the H5 writes and make the reference cache
# per-cohort rather than global — until then, memory is the cost of the design.
#
# shard | config basename | conda env | walltime | mem | worker
# Walltimes are ~2x the estimate so a slow queue/node never truncates a shard.
SHARDS=(
    "a_cheap|picasso_shard_a_cheap.yaml|vena|06:00:00|300G|worker_inference_picasso_full.sh"
    "b_vena|picasso_shard_b_vena.yaml|vena|1-00:00:00|260G|worker_inference_picasso_full.sh"
    "c_latent|picasso_shard_c_latent.yaml|vena|1-00:00:00|300G|worker_inference_picasso_full.sh"
    "d_lddpm|picasso_shard_d_lddpm.yaml|vena|3-00:00:00|230G|worker_inference_picasso_full.sh"
    "e_syndiff|picasso_full_syndiff.yaml|vena|12:00:00|150G|worker_inference_picasso_full.sh"
)

# SHARD E runs in the plain `vena` env, NOT `vena-syndiff` (2026-07-14).
# The `vena-syndiff` env was deleted from fscratch/conda_envs after the 2026-06-15
# SynDiff training runs, and rebuilding it did not work: the recipe pins python
# 3.10 while pyproject requires >=3.11, and the resulting toolchain could not
# compile the StyleGAN2 fused CUDA kernels (missing cusparse.h; nvcc rejecting the
# conda gcc with "'timespec_get' has not been declared").
#
# Instead we take contingency C1 from src/external/syndiff/PATCHES.md:
# `_ensure_stylegan_ops()` (src/vena/competitors/syndiff/runner.py) tries the fused
# kernels first and, when they will not build, transparently substitutes the
# pure-PyTorch reference ops — same arithmetic, no nvcc/ninja, ~2-4x slower per
# layer, which is irrelevant at SynDiff's NFE=4. So SynDiff needs no special env
# and no compiler at all.

WANTED=("$@")
want() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local s
    for s in "${WANTED[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

SUBMITTED=()
for row in "${SHARDS[@]}"; do
    IFS="|" read -r shard cfg env walltime mem worker <<<"${row}"
    want "${shard}" || continue

    config_path="${CFG_DIR}/${cfg}"
    job_name="vena-inf-${shard}"

    sbatch_cmd="sbatch --parsable \
        -J ${job_name} \
        --time=${walltime} \
        --mem=${mem} \
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
    echo "Submitted ${job_id}  ${job_name}  (${walltime}, mem=${mem}, env=${env})"
done

${DRY_RUN} && exit 0

echo ""
echo "Monitor:   squeue -u \$USER -o '%.10i %.18j %.8T %.10M %.6D %R'"
echo "Logs:      ${LOGS_DIR}/inference_<shard>_<jobid>.{out,err}"
echo "Output:    /mnt/home/users/tic_163_uma/mpascual/execs/vena/inference/<run_id_tag>/predictions/"
echo "Sentinel:  'inference routine complete' in each shard's routine log."
