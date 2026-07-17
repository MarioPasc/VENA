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
# Missing shards are a STOP-THE-LINE condition, never an auto-downgrade.
# This block used to set --allow-partial itself whenever a shard was absent,
# which defeated the very guard cli_merge implements: a merge over a partial
# grid produces a complete-LOOKING artifact (full tables, figures, a plausible
# n_patients) built from a fraction of the methods, and publishes it as LATEST.
# That happened on 2026-07-17 (job 1604489: 54 of 405 files, n_patients=393 —
# a number that reconciles against the docs and is nonetheless wrong).
# --allow-partial is a deliberate human decision; pass ALLOW_PARTIAL=1 in the
# environment to make it, and say why in the run log.
N_EXPECTED=$(tail -n +2 "${MANIFEST}" | wc -l)
N_FOUND=$(ls "${SHARD_DIR}"/shard_*.csv 2>/dev/null | wc -l)
echo "[merge] Shards expected=${N_EXPECTED} found=${N_FOUND}"
if [[ "${N_FOUND}" -lt "${N_EXPECTED}" ]]; then
    if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
        echo "[merge] ALLOW_PARTIAL=1 was set explicitly — merging ${N_FOUND}/${N_EXPECTED} shards."
        ALLOW_PARTIAL_FLAG="--allow-partial"
    else
        echo "[merge] FATAL: ${N_EXPECTED} tasks but only ${N_FOUND} shards on disk." >&2
        echo "[merge] Resubmit the missing array tasks, then rerun this worker." >&2
        echo "[merge] To merge a partial grid anyway, resubmit with ALLOW_PARTIAL=1." >&2
        exit 1
    fi
else
    ALLOW_PARTIAL_FLAG=""
fi

# ---- Run merge ----
"${PYTHON}" -m routines.validation.paired_fidelity.cli_merge \
    --manifest    "${MANIFEST}" \
    --shard-dir   "${SHARD_DIR}" \
    --config      "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT:-${SWEEP_ROOT}/analyses}" \
    ${ALLOW_PARTIAL_FLAG}

echo "[merge] Done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
