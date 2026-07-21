#!/usr/bin/env bash
# Worker script for ρ_S normalisation audit on Picasso.
#
# Run directly (no launcher) with:
#   sbatch routines/preflights/rho_s_norm_audit/slurm/worker_rho_s_norm_audit.sh \
#          <config.yaml>
#
# CPU-only.  Predictions must be present under inference_root on fscratch.

#SBATCH --job-name=vena_rho_s_norm_audit_worker
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/rho_s_norm_audit_%j.log
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/rho_s_norm_audit_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=short

set -euo pipefail

CONFIG="${1:-routines/preflights/rho_s_norm_audit/configs/default.yaml}"
REPO="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation"
SINGULARITY_IMAGE="/mnt/home/users/tic_163_uma/mpascual/fscratch/containers/vena_latest.sif"

echo "=== ρ_S normalisation audit worker ==="
echo "Config   : ${CONFIG}"
echo "Node     : $(hostname)"
echo "Date     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

cd "${REPO}"

singularity exec \
    --bind /mnt \
    "${SINGULARITY_IMAGE}" \
    bash -c "
        source ~/.conda/envs/vena/bin/activate
        python -m routines.preflights.rho_s_norm_audit.cli '${CONFIG}'
    "

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
