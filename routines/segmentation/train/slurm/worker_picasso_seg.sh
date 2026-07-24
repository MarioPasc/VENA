#!/usr/bin/env bash
# ============================================================================
# SLURM worker — segmentation training (K-fold + all_train array)
# ============================================================================
# Submitted via launcher_picasso_seg_ukb.sh (or equivalent).
# One task per model: tasks 0-4 train fold-0…fold-4; task 5 = all_train.
#
# Resource sizing (measured throughput):
#   H5 read ≈ 0.40 s/sample, soft-target ≈ 0.5 s/sample, 8 workers → ~0.11 s
#   → ~5-6 min/epoch at 1331 train scans/fold → 300 ep ≈ 25-30 h
#   --time=2-00:00:00 covers max_epochs=300 with early_stop_patience=30.
#
# Usage (do NOT submit manually — use the launcher):
#   Submitted by launcher with:
#     --array=0-5
#     --export=ALL,REPO_DIR=...,ARM_CONFIG=...,LOGS_DIR=...
# ============================================================================
#SBATCH -J vena-seg-train
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu_partition
# A100 selector: exa01-04 advertise UNTYPED `Gres=gpu:8` and carry the `a100`
# feature; only the B200 nodes (blk01-02) publish a typed `gpu:B200:8`.
# `--gres=gpu:A100:1` therefore matches NOTHING (verified: sbatch --test-only
# -> "Requested node configuration is not available"), and a bare
# `--constraint=dgx` matches B200 too since both node types carry `dgx`.
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1

set -euo pipefail

START_TIME=$(date +%s)

# ============================================================================
# JOB HEADER (reproducibility)
# ============================================================================
echo "============================================================"
echo "SLURM job id   : ${SLURM_JOB_ID}"
echo "Array task id  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : $(hostname)"
echo "Start          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo dir       : ${REPO_DIR}"
echo "Arm config     : ${ARM_CONFIG}"
echo "============================================================"

# ============================================================================
# VALIDATE REQUIRED ENV VARS
# ============================================================================
: "${REPO_DIR:?REPO_DIR must be set by the launcher}"
: "${ARM_CONFIG:?ARM_CONFIG must be set by the launcher (arm production YAML)}"
: "${LOGS_DIR:?LOGS_DIR must be set by the launcher}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena}"
PYTHON="${CONDA_ENV_PATH}/bin/python"

# ============================================================================
# ENVIRONMENT
# ============================================================================
# Picasso's conda module is named `miniconda/3`; fall back to common variants
# used on other clusters so the same worker is portable.
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# RESOLVE FOLD FROM ARRAY TASK ID
# ============================================================================
# tasks 0-4 → fold 0-4; task 5 → all_train
TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [ "${TASK_ID}" -eq 5 ]; then
    FOLD_VALUE="all_train"
else
    FOLD_VALUE="${TASK_ID}"
fi
echo "Fold for this task: ${FOLD_VALUE}"

# Derive tag suffix from the arm config filename (e.g. picasso_seg_ukb → ukb)
ARM_BASENAME=$(basename "${ARM_CONFIG}" .yaml)   # e.g. picasso_seg_ukb
TAG_SUFFIX="${ARM_BASENAME#picasso_seg_}"         # e.g. ukb

# ============================================================================
# RENDER PER-TASK YAML
# ============================================================================
# The routine takes ONE positional YAML with no extra flags.
# We copy the arm config and overwrite run.fold (and run.tag) for this task.
SCRATCH_DIR="${LOGS_DIR}/task_yamls/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
TASK_YAML="${SCRATCH_DIR}/task_${TASK_ID}.yaml"

# Use Python (already in CONDA_ENV_PATH) for safe YAML editing
"${PYTHON}" - <<EOF
import yaml
from pathlib import Path

src = Path("${ARM_CONFIG}")
with src.open() as fh:
    cfg = yaml.safe_load(fh)

# Override run.fold — "all_train" must be a string, not an int
fold_raw = "${FOLD_VALUE}"
cfg["run"]["fold"] = fold_raw if fold_raw == "all_train" else int(fold_raw)
# Override run.tag to make run_ids per-task unique
cfg["run"]["tag"] = "${TAG_SUFFIX}_k5_fold${FOLD_VALUE}"

dst = Path("${TASK_YAML}")
with dst.open("w") as fh:
    yaml.dump(cfg, fh, default_flow_style=False)

print(f"[render] written task YAML → {dst}")
EOF

echo "Task YAML: ${TASK_YAML}"
echo "--- task YAML head ---"
head -30 "${TASK_YAML}"
echo "--- end task YAML head ---"

# ============================================================================
# Expect 1 × A100.
# ============================================================================
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ============================================================================
# COMMAND
# ============================================================================
echo "Launching: ${PYTHON} -m routines.segmentation.train.cli ${TASK_YAML}"
"${PYTHON}" -m routines.segmentation.train.cli "${TASK_YAML}"
EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
END_TIME=$(date +%s)
WALL_SECS=$(( END_TIME - START_TIME ))
echo "============================================================"
echo "Exit code   : ${EXIT_CODE}"
echo "Wall time   : ${WALL_SECS} s ($(( WALL_SECS / 3600 ))h $(( (WALL_SECS % 3600) / 60 ))m)"
echo "End         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
exit "${EXIT_CODE}"
