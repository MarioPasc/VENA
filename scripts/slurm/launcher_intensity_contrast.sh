#!/bin/bash
# Launcher: submit the intensity-contrast measurement job to Picasso.
# Usage:  bash scripts/slurm/launcher_intensity_contrast.sh
# Run from the root of VENA-validation on the Picasso login node.
set -eo pipefail

REPO=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation
WORKER="${REPO}/scripts/slurm/worker_intensity_contrast.sh"

OUT=$(sbatch --parsable \
    --job-name=intensity_contrast \
    --gres=gpu:1 \
    --constraint=a100 \
    --cpus-per-task=4 \
    --mem=48G \
    --time=02:00:00 \
    --output="${HOME}/execs/vena/analyses/s3_intensity/logs/slurm_%j.out" \
    --error="${HOME}/execs/vena/analyses/s3_intensity/logs/slurm_%j.err" \
    "${WORKER}")

# ── Parse job ID: strip ANSI codes and Lua warnings ──────────────────────────
# sbatch --parsable may print multi-line Lua warnings before the ID line.
# Strip colour codes and any non-numeric characters from the final line.
mkdir -p "${HOME}/execs/vena/analyses/s3_intensity/logs"
JOB_ID=$(tail -n 1 <<<"${OUT}" \
    | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
          -e 's/[^0-9]//g')

if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job ID from sbatch output:"
    echo "${OUT}"
    echo "Checking squeue in case job was submitted anyway..."
    squeue -u "${USER}" --format="%.18i %.10j %.8T %.10M %R" 2>/dev/null | head -20
    exit 1
fi

echo "Submitted intensity_contrast job: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  sacct -j ${JOB_ID} -X --format=JobID,State,Elapsed,MaxRSS -P"
echo "  tail -f ~/execs/vena/analyses/s3_intensity/logs/slurm_${JOB_ID}.out"
