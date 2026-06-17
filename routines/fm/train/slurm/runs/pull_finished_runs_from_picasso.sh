#!/usr/bin/env bash
# Pull two finished VENA FM training run directories from Picasso to local,
# skipping the redundant launcher snapshots in `exhaustive_val/`.
#
# Usage:
#   bash pull_finished_runs_from_picasso.sh                     # sequential, real
#   bash pull_finished_runs_from_picasso.sh --dry-run           # see what would move
#   bash pull_finished_runs_from_picasso.sh --parallel          # rsync both runs concurrently
#
# Safety contract (cannot lose data, cannot corrupt local checkpoints):
#   * rsync `--update`        — overwrite a local file ONLY when the remote
#                               file is newer. The stale local `last.ckpt`
#                               and `ema_best.ckpt` from the partial mid-run
#                               copy get refreshed; the immutable periodic
#                               `ema_epoch_NNNN.ckpt` files (mtimes match
#                               remote) are skipped.
#   * Atomic rename          — rsync writes to `.<name>.XXXXXX` and renames
#                               on completion; an interrupted transfer never
#                               corrupts an existing local file.
#   * `--partial --partial-dir=.rsync-partial` — resume from partial bytes on
#                               re-run.
#   * No `--delete`           — local-only files are preserved.
#   * `--itemize-changes` + per-run log under `.transfer_logs/<run>_<UTC>.log`
#                               — every change is auditable after the fact.
#
# What is excluded and why:
#   * `exhaustive_val/epoch_*/ema_snapshot.pt`
#   * `exhaustive_val/epoch_*/trunk_ema_snapshot.pt`
#   These are temporary EMA shadow snapshots the launcher writes to feed the
#   spawned validation subprocess. They are *byte-identical* (up to
#   floating-point save/load roundtrip) to the EMA state inside Lightning's
#   `ema_best.ckpt` / `ema_epoch_NNNN.ckpt`. The 2026-06-15 pruner already
#   deletes all but the most-recent two per run, so the on-disk savings are
#   bounded (~1-2 GB), but skipping the transfer is still strictly cheaper
#   than re-transmitting them.
#
# What is KEPT (everything else):
#   * `checkpoints/` (last.ckpt, ema_best.ckpt, ema_epoch_NNNN.ckpt,
#     ema_final.ckpt) — the `.pt` exclude cannot match these (they are
#     `.ckpt` and live outside `exhaustive_val/`).
#   * `exhaustive_val/epoch_*/metrics.csv|timing.csv|latent_preds.h5|figure_*.png|job.yaml|subprocess.log`
#   * `exhaustive_val/gpu_usage.log`
#   * `logs/`, `metrics/`, `plots/`, `config*.yaml`, `decision.json`,
#     `env.txt`, `git_commit.txt`, `hostname.txt`.

set -euo pipefail

PICASSO_HOST="${PICASSO_HOST:-picasso}"
PICASSO_EXP_ROOT="${PICASSO_EXP_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments}"
LOCAL_DST="${LOCAL_DST:-/media/mpascual/Sandisk2TB/research/vena/results/fm/vena}"

RUNS=(
    "2026-06-12_01-27-55_s1_fft_cfm_c9b97556"
    "2026-06-12_06-32-25_s2_fft_contrastive_a27962e7"
)

DRY_RUN=false
PARALLEL=false
for arg in "$@"; do
    case "${arg}" in
        --dry-run)  DRY_RUN=true ;;
        --parallel) PARALLEL=true ;;
        *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOCAL_DST}"
LOG_DIR="${LOCAL_DST}/.transfer_logs"
mkdir -p "${LOG_DIR}"
STAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)

EXCLUDES=(
    --exclude="exhaustive_val/epoch_*/ema_snapshot.pt"
    --exclude="exhaustive_val/epoch_*/trunk_ema_snapshot.pt"
)

RSYNC_OPTS=(
    -ah --update
    --info=progress2,stats2
    --partial --partial-dir=.rsync-partial
    --itemize-changes
    "${EXCLUDES[@]}"
)
${DRY_RUN} && RSYNC_OPTS+=(--dry-run)

# ---------------------------------------------------------------------------
# Pre-flight: verify each remote run carries checkpoints/{last,ema_best}.ckpt.
# A missing critical checkpoint aborts BEFORE any rsync touches local files.
# ---------------------------------------------------------------------------
echo "[pre-flight] host=${PICASSO_HOST}  remote_root=${PICASSO_EXP_ROOT}"
for run in "${RUNS[@]}"; do
    src_root="${PICASSO_EXP_ROOT}/${run}"
    if ! ssh "${PICASSO_HOST}" "test -d '${src_root}'" 2>/dev/null; then
        echo "  [FAIL] ${run}: remote dir missing" >&2
        exit 1
    fi
    if ! ssh "${PICASSO_HOST}" \
        "test -f '${src_root}/checkpoints/last.ckpt' && test -f '${src_root}/checkpoints/ema_best.ckpt'" 2>/dev/null; then
        echo "  [FAIL] ${run}: missing checkpoints/{last,ema_best}.ckpt — refusing to transfer" >&2
        exit 1
    fi
    total_h=$(ssh "${PICASSO_HOST}" "du -sh '${src_root}' 2>/dev/null | awk '{print \$1}'")
    pt_h=$(ssh "${PICASSO_HOST}" "find '${src_root}/exhaustive_val' -name '*_snapshot.pt' -printf '%s\n' 2>/dev/null \
            | awk '{s+=\$1} END {if (s==0) {print \"0\"} else {printf \"%.2f GB\", s/1024/1024/1024}}'")
    n_ckpts=$(ssh "${PICASSO_HOST}" "ls '${src_root}/checkpoints' 2>/dev/null | wc -l")
    n_epoch=$(ssh "${PICASSO_HOST}" "ls '${src_root}/exhaustive_val' 2>/dev/null | grep -c '^epoch_' || true")
    echo "  [ok]   ${run}"
    echo "         total=${total_h}  excluded(.pt)=${pt_h}  checkpoints=${n_ckpts}  epoch_dirs=${n_epoch}"
done

# ---------------------------------------------------------------------------
# Transfer.
# ---------------------------------------------------------------------------
rsync_one() {
    local run="$1"
    local src="${PICASSO_HOST}:${PICASSO_EXP_ROOT}/${run}/"
    local dst="${LOCAL_DST}/${run}/"
    local log="${LOG_DIR}/${run}_${STAMP}.log"
    mkdir -p "${dst}"
    echo "[transfer] ${run}"
    echo "  src: ${src}"
    echo "  dst: ${dst}"
    echo "  log: ${log}"
    rsync "${RSYNC_OPTS[@]}" "${src}" "${dst}" 2>&1 | tee "${log}"
}

if ${PARALLEL}; then
    pids=()
    for run in "${RUNS[@]}"; do
        rsync_one "${run}" &
        pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || rc=$?
    done
    if [[ ${rc} -ne 0 ]]; then
        echo "[done] one or more parallel transfers exited non-zero (rc=${rc})." >&2
        exit "${rc}"
    fi
else
    for run in "${RUNS[@]}"; do
        rsync_one "${run}"
    done
fi

echo ""
echo "[done] transfer complete."
echo "        logs:   ${LOG_DIR}"
echo "        dest:   ${LOCAL_DST}"
echo "        Verify: diff <(ssh ${PICASSO_HOST} 'sha256sum ${PICASSO_EXP_ROOT}/<run>/checkpoints/{last,ema_best}.ckpt') \\"
echo "                     <(cd ${LOCAL_DST}/<run>/checkpoints && sha256sum {last,ema_best}.ckpt)"
