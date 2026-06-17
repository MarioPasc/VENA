#!/usr/bin/env bash
# Launch the SynDiff inference smoke on loginexa (V100 sm_70).
#
# Runs C3-SynDiff (t1pre / t2 / flair, 3 panel runs) for 1 patient per
# cohort in the `vena-v100-syndiff` env (Python 3.10 + torch 2.5.1+cu121
# + ninja + the JIT-compiled fused/upfirdn2d CUDA extensions). The main
# `smoke_loginexa.yaml` covers every other method in the unified
# `vena-v100` env. Both write to the same `run_id_tag` so the per-method
# predictions trees merge.
#
# Usage:
#   bash launcher_inference_loginexa_syndiff.sh             # submit
#   bash launcher_inference_loginexa_syndiff.sh --dry-run   # print plan
#
# Mirrors `routines/competitors/syndiff/loginexa/launcher_syndiff_loginexa_t1pre_2ep.sh`
# — the same tmux-helper pattern is used to avoid the layered-quote escape
# problem of inlining the whole pipeline through `tmux new-session -d -s X`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA}"
CONDA_SH="${CONDA_SH:-/mnt/home/soft/miniconda/programs/x86_64/miniconda3_py310_23.1.0/etc/profile.d/conda.sh}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena-v100-syndiff}"
CONFIG_PATH="${REPO_DIR_REMOTE}/routines/fm/inference/configs/smoke_loginexa_syndiff.yaml"
LOG_DIR="${LOG_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/inference}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/.cache/torch_extensions/vena-v100-syndiff}"
SESSION="${SESSION:-vena-inference-loginexa-syndiff-smoke}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${LOG_DIR}" "${TORCH_EXTENSIONS_DIR}"

if [[ -z "${GPU_ID:-}" ]]; then
    GPU_ID=$(ssh loginexa "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null" \
        | sort -t',' -k2,2 -nr | head -1 | awk -F',' '{print $1}' | tr -d ' ')
    GPU_ID="${GPU_ID:-0}"
fi
echo "[plan] loginexa GPU=${GPU_ID}"
echo "[plan] tmux session: ${SESSION}"
echo "[plan] conda env:    ${CONDA_ENV_PATH}"
echo "[plan] config:       ${CONFIG_PATH}"
echo "[plan] log:          ${LOG_DIR}/${SESSION}.log"
echo "[plan] ext cache:    ${TORCH_EXTENSIONS_DIR}"

if ${DRY_RUN}; then
    exit 0
fi

ssh loginexa "test -d '${CONDA_ENV_PATH}'" || {
    echo "ERROR: ${CONDA_ENV_PATH} not found on loginexa — create the vena-v100-syndiff env first (see .claude/notes/validation/syndiff.md)." >&2
    exit 1
}

# Tmux-helper pattern (same as SynDiff training launcher) — avoids the
# layered-quote escaping problem of inlining the whole pipeline through
# `tmux new-session -d -s X "..."`.
HELPER="/tmp/vena_inference_loginexa_syndiff_runner.sh"
ssh loginexa "cat > '${HELPER}' <<'EOF'
#!/usr/bin/env bash
# Drop -u: conda's gxx_linux-64 activate script references SYS_SYSROOT
# without guarding for unset (\${SYS_SYSROOT:-}); set -u would kill activation.
set -eo pipefail
source '${CONDA_SH}'
conda activate '${CONDA_ENV_PATH}'
cd '${REPO_DIR_REMOTE}'
export PYTHONPATH='${REPO_DIR_REMOTE}/src:${REPO_DIR_REMOTE}:'\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_HOME=\$CONDA_PREFIX
export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST=7.0
export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}'
timeout 25m python -m routines.fm.inference.cli '${CONFIG_PATH}' 2>&1 | tee -a '${LOG_DIR}/${SESSION}.log'
EOF
chmod +x '${HELPER}'"

ssh loginexa "tmux new-session -d -s '${SESSION}' '${HELPER}'"

echo ""
echo "Submitted ✓"
echo "  attach: ssh loginexa tmux attach -t ${SESSION}"
echo "  logs:   ssh loginexa tail -F ${LOG_DIR}/${SESSION}.log"
echo "  kill:   ssh loginexa tmux kill-session -t ${SESSION}"
