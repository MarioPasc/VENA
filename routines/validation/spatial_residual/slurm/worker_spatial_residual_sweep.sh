#!/usr/bin/env bash
# SLURM array worker — one task per (method, cohort) prediction file.
# Called by launcher_spatial_residual_sweep.sh.  Do not submit directly.
#
# Each task reads row $SLURM_ARRAY_TASK_ID from $MANIFEST, processes one
# prediction H5 file, and writes shard_<NNNN>.csv to $SHARD_DIR.
# No GPU, no Docker — CPU only on cpu_partition.
#
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-16:00:00
#SBATCH --constraint=cpu
set -eo pipefail

echo "[worker] Task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[worker] REPO_DIR      = ${REPO_DIR}"
echo "[worker] CONDA_ENV_PATH= ${CONDA_ENV_PATH}"
echo "[worker] MANIFEST      = ${MANIFEST}"
echo "[worker] SHARD_DIR     = ${SHARD_DIR}"
echo "[worker] CONFIG_PATH   = ${CONFIG_PATH}"

# ---- Environment ----
PYTHON="${CONDA_ENV_PATH}/bin/python"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
# Let numpy use all allocated CPUs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ---- Import isolation self-check ----
VENA_SRC=$("${PYTHON}" -c "import vena; print(vena.__file__)" 2>/dev/null || echo "IMPORT_FAILED")
echo "[worker] vena source: ${VENA_SRC}"
if [[ "${VENA_SRC}" != "${REPO_DIR}/src/vena/__init__.py" ]]; then
    echo "[WARN] vena may be loading from shared env — PYTHONPATH override active: ${PYTHONPATH}"
fi

# ---- Run shard ----
"${PYTHON}" -m routines.validation.spatial_residual.cli_shard \
    --manifest  "${MANIFEST}" \
    --task-id   "${SLURM_ARRAY_TASK_ID}" \
    --shard-dir "${SHARD_DIR}" \
    --config    "${CONFIG_PATH}"

echo "[worker] Task ${SLURM_ARRAY_TASK_ID} done — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
