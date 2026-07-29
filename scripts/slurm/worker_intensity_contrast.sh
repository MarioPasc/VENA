#!/bin/bash
# Worker script for the intensity-contrast measurement job.
# Runs on a single A100 GPU on Picasso.
# Launched by launcher_intensity_contrast.sh — do not run directly.
set -eo pipefail

REPO=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation
PYTHON=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena/bin/python

# ── PYTHONPATH isolation: force vena + routines to resolve from VENA-validation ──
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Confirm isolation before doing anything expensive
echo "PYTHONPATH=${PYTHONPATH}"
"${PYTHON}" -c "
import vena, routines, pathlib
repo = pathlib.Path('${REPO}').resolve()
vf   = pathlib.Path(vena.__file__).resolve()
rf   = pathlib.Path(routines.__file__).resolve()
print(f'vena.__file__     = {vf}')
print(f'routines.__file__ = {rf}')
assert repo in vf.parents,  f'ISOLATION FAIL: vena from wrong repo: {vf}'
assert repo in rf.parents,  f'ISOLATION FAIL: routines from wrong repo: {rf}'
print('ISOLATION PROOF: OK')
"

LOG_DIR="${HOME}/execs/vena/analyses/s3_intensity/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/intensity_contrast_${SLURM_JOB_ID:-local}.log"

echo "Job ID   : ${SLURM_JOB_ID:-<local>}"
echo "Node     : $(hostname)"
echo "GPU      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Log      : ${LOG}"
echo "Started  : $(date -u +%FT%TZ)"

"${PYTHON}" "${REPO}/scripts/measure_intensity_contrast.py" 2>&1 | tee "${LOG}"

echo "Finished : $(date -u +%FT%TZ)"
