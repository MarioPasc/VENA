#!/usr/bin/env bash
# Launch vena-spatial-residual-smoke on Picasso.
#
# Usage (run from any directory on Picasso login node):
#   bash launcher_spatial_residual_smoke.sh
#
# Writes SLURM output to ~/execs/vena/logs/spatial_residual_smoke_<JOBID>.log.
# Per-scan CSV and decision.json land at:
#   ~/execs/vena/inference/analyses/spatial_residual/<UTC>/
#
# Worktree location on Picasso scratch.  Do NOT point at fscratch/repos/VENA —
# that is the orchestrator's main repo.  This is an agent smoke path.
set -eo pipefail

REPO_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA_agent_spatial_smoke
CONFIG_PATH=${REPO_DIR}/routines/validation/spatial_residual/configs/picasso_smoke.yaml
WORKER=${REPO_DIR}/routines/validation/spatial_residual/slurm/worker_spatial_residual.sh
LOG_DIR=/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs

mkdir -p "${LOG_DIR}"
chmod +x "${WORKER}"

echo "Submitting vena-spatial-residual-smoke..."
echo "  REPO_DIR:    ${REPO_DIR}"
echo "  CONFIG_PATH: ${CONFIG_PATH}"
echo "  WORKER:      ${WORKER}"
echo "  LOG_DIR:     ${LOG_DIR}"
echo ""

JOB_ID=$(sbatch \
    --output="${LOG_DIR}/spatial_residual_smoke_%j.log" \
    --export=REPO_DIR="${REPO_DIR}",CONFIG_PATH="${CONFIG_PATH}" \
    "${WORKER}" | awk '{print $NF}')

echo "Submitted job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      tail -f ${LOG_DIR}/spatial_residual_smoke_${JOB_ID}.log"
