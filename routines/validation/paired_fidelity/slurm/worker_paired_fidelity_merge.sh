#!/usr/bin/env bash
# SLURM merge worker — concatenate all shards and run the full analysis pass.
# Submitted by launcher_paired_fidelity_sweep.sh with --dependency=afterok:<array_id>.
# Do not submit directly.
#
# Runs on a single node; no GPU needed.  The Holm-Bonferroni correction and
# patient collapse run exactly once here — never per shard.
#
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --constraint=cpu
set -eo pipefail

echo "[merge] Running on $(hostname) — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[merge] REPO_DIR      = ${REPO_DIR}"
echo "[merge] CONDA_ENV_PATH= ${CONDA_ENV_PATH}"
echo "[merge] MANIFEST      = ${MANIFEST}"
echo "[merge] SHARD_DIR     = ${SHARD_DIR}"
echo "[merge] SWEEP_ROOT    = ${SWEEP_ROOT}"
echo "[merge] CONFIG_PATH   = ${CONFIG_PATH}"
echo "[merge] OUTPUT_ROOT   = ${OUTPUT_ROOT:-${SWEEP_ROOT}/analyses}"

# ---- Environment ----
PYTHON="${CONDA_ENV_PATH}/bin/python"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ---- Import isolation self-check ----
VENA_SRC=$("${PYTHON}" -c "import vena; print(vena.__file__)" 2>/dev/null || echo "IMPORT_FAILED")
echo "[merge] vena source: ${VENA_SRC}"

# ---- Verify all shards landed ----
N_EXPECTED=$(tail -n +2 "${MANIFEST}" | wc -l)
N_FOUND=$(ls "${SHARD_DIR}"/shard_*.csv 2>/dev/null | wc -l)
echo "[merge] Shards expected=${N_EXPECTED} found=${N_FOUND}"
if [[ "${N_FOUND}" -lt "${N_EXPECTED}" ]]; then
    echo "[merge] WARNING: ${N_EXPECTED} tasks but only ${N_FOUND} shards — proceeding with --allow-partial."
    ALLOW_PARTIAL="--allow-partial"
else
    ALLOW_PARTIAL=""
fi

# ---- Run merge ----
"${PYTHON}" -m routines.validation.paired_fidelity.cli_merge \
    --manifest    "${MANIFEST}" \
    --shard-dir   "${SHARD_DIR}" \
    --config      "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT:-${SWEEP_ROOT}/analyses}" \
    ${ALLOW_PARTIAL}

echo "[merge] Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
