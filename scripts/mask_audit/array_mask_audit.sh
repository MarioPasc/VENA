#!/bin/bash
#SBATCH --job-name=vena_maskaudit
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/maskaudit_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/maskaudit_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --constraint=cpu
#SBATCH --array=0-8
#
# Pure-CPU array: one task per cohort. Recomputes every mask invariant from the
# actual GT labels and audits the cached masks/tumor_latent_soft group.
# No GPU -> bypasses the gres/gpu group cap.

set -euo pipefail

REPO=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation
PY=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/python
OUT=/mnt/home/users/tic_163_uma/mpascual/execs/vena/mask_audit/metrics
SCRIPTS="$REPO/scripts/mask_audit"

mkdir -p "$OUT"
cd "$REPO"
export OMP_NUM_THREADS=4
export PYTHONPATH="$REPO/src:$SCRIPTS"

echo "task=${SLURM_ARRAY_TASK_ID} host=$(hostname) sha=$(git rev-parse --short HEAD) start=$(date -u +%FT%TZ)"
"$PY" "$SCRIPTS/audit_cohort.py" --cohort-index "${SLURM_ARRAY_TASK_ID}" --out-dir "$OUT"
echo "task=${SLURM_ARRAY_TASK_ID} done=$(date -u +%FT%TZ)"
