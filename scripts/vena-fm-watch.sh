#!/usr/bin/env bash
# vena-fm-watch — tail the train log + metrics CSVs of a Picasso FM-train run
# from the local workstation.
#
# Usage:
#   bash scripts/vena-fm-watch.sh <run_id>          # tail logs/train.log
#   bash scripts/vena-fm-watch.sh <run_id> metrics  # tail metrics/train_step.csv
#   bash scripts/vena-fm-watch.sh <run_id> epoch    # cat metrics/train_epoch.csv
#   bash scripts/vena-fm-watch.sh <run_id> val      # ls + tail exhaustive_val/
#   bash scripts/vena-fm-watch.sh <run_id> pull <local-dest>  # rsync the whole run dir
#
# Resolves <run_id> against:
#   $PICASSO_HOST  (default: picasso — must be in ~/.ssh/config)
#   $PICASSO_EXPERIMENTS_ROOT  (default:
#     /mnt/home/users/tic_163_uma/mpascual/fscratch/experiments/vena)
#
# A trailing 'latest' resolves to the most-recently-modified run directory.

set -euo pipefail

HOST="${PICASSO_HOST:-picasso}"
ROOT="${PICASSO_EXPERIMENTS_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/experiments/vena}"

if [ $# -lt 1 ]; then
    sed -n '2,/^set -euo/p' "$0" | sed '$d'
    exit 1
fi

RUN_ID="$1"
MODE="${2:-log}"

if [ "${RUN_ID}" = "latest" ]; then
    RUN_ID="$(ssh "${HOST}" "ls -t '${ROOT}' | head -1")"
    echo "[vena-fm-watch] latest = ${RUN_ID}"
fi

RUN_DIR="${ROOT}/${RUN_ID}"

case "${MODE}" in
    log)
        echo "[vena-fm-watch] tailing ${HOST}:${RUN_DIR}/logs/train.log"
        exec ssh "${HOST}" "tail -F '${RUN_DIR}/logs/train.log'"
        ;;
    metrics)
        echo "[vena-fm-watch] tailing ${HOST}:${RUN_DIR}/metrics/train_step.csv"
        exec ssh "${HOST}" "tail -F '${RUN_DIR}/metrics/train_step.csv'"
        ;;
    epoch)
        exec ssh "${HOST}" "cat '${RUN_DIR}/metrics/train_epoch.csv'"
        ;;
    val)
        echo "[vena-fm-watch] exhaustive_val tree:"
        ssh "${HOST}" "ls -la '${RUN_DIR}/exhaustive_val/' 2>/dev/null | head -30"
        echo
        LATEST_EPOCH="$(ssh "${HOST}" "ls '${RUN_DIR}/exhaustive_val/' | grep '^epoch_' | sort | tail -1")"
        if [ -n "${LATEST_EPOCH}" ]; then
            echo "[vena-fm-watch] latest val: ${LATEST_EPOCH}"
            ssh "${HOST}" "wc -l '${RUN_DIR}/exhaustive_val/${LATEST_EPOCH}/metrics.csv' 2>/dev/null"
            ssh "${HOST}" "head -3 '${RUN_DIR}/exhaustive_val/${LATEST_EPOCH}/metrics.csv' 2>/dev/null"
        fi
        ;;
    pull)
        DEST="${3:?usage: pull <local-dest>}"
        mkdir -p "${DEST}"
        echo "[vena-fm-watch] rsyncing ${HOST}:${RUN_DIR}/ → ${DEST}/"
        exec rsync -av --exclude='*.ckpt' --exclude='*.pt' --exclude='latent_preds.h5' \
            "${HOST}:${RUN_DIR}/" "${DEST}/"
        ;;
    *)
        echo "[vena-fm-watch] unknown mode '${MODE}' — use: log | metrics | epoch | val | pull"
        exit 1
        ;;
esac
