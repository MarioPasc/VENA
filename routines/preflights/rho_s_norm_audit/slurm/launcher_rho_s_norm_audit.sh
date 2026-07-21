#!/usr/bin/env bash
# Launcher for ρ_S normalisation audit on Picasso.
#
# Usage:
#   sbatch routines/preflights/rho_s_norm_audit/slurm/launcher_rho_s_norm_audit.sh \
#          routines/preflights/rho_s_norm_audit/configs/default.yaml
#
# This is a CPU-only job; no GPU requested.
# Predictions must be restored to Picasso fscratch before submitting.

#SBATCH --job-name=vena_rho_s_norm_audit
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

echo "=== ρ_S normalisation audit launcher ==="
echo "Config : ${CONFIG}"
echo "Repo   : ${REPO}"
echo "Node   : $(hostname)"
echo "Date   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Submit the worker.
sbatch \
    --job-name=vena_rho_s_norm_audit_worker \
    --output="${REPO}/logs/rho_s_norm_audit_worker_%j.log" \
    --error="${REPO}/logs/rho_s_norm_audit_worker_%j.err" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=04:00:00 \
    --partition=short \
    --wrap="singularity exec --nv ${SINGULARITY_IMAGE} \
        bash -c 'cd ${REPO} && \
            source ~/.conda/envs/vena/bin/activate && \
            python -m routines.preflights.rho_s_norm_audit.cli ${CONFIG}'"

echo "Worker submitted."
