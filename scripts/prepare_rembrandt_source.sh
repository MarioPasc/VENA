#!/usr/bin/env bash
#
# Stage the REMBRANDT source tree into a flat directory for the HD-BET
# skull-strip + H5 converter routines.
#
# Background — the upstream archive uses two container layouts:
#   * batches 1_10, 11_20, 21_30, 31_40, 51_60: doubly-nested
#       Patients_X_Y/Patients_X_Y/<pid>_<date>/...
#   * batches 41_50, 61_65: flat
#       Patients_X_Y/<pid>_<date>/...
# This script symlinks every per-session directory into one flat staging dir
# (one symlink per session, name preserved) so the REMBRANDTDataset reader
# sees a uniform layout.
#
# It also renames sessions known to be incomplete:
#   * HF0920_1991.06.14 is missing the T1 pre-contrast NIfTI — moved aside
#     as `.SKIPPED_HF0920_no_t1pre` so the HD-BET runner never sees it
#     (per .claude/skills/add-dataset playbook Step 2 landmine).
#
# Usage:
#     bash scripts/prepare_rembrandt_source.sh \
#         /media/mpascual/MeningD2/GLIOMA/REMBRANDT/source \
#         /media/mpascual/MeningD2/GLIOMA/REMBRANDT/staged

set -euo pipefail

SOURCE_ROOT=${1:-/media/mpascual/MeningD2/GLIOMA/REMBRANDT/source}
STAGED_ROOT=${2:-/media/mpascual/MeningD2/GLIOMA/REMBRANDT/staged}

if [ ! -d "$SOURCE_ROOT" ]; then
  echo "ERROR: source_root does not exist: $SOURCE_ROOT" >&2
  exit 1
fi

mkdir -p "$STAGED_ROOT"

n_total=0
n_linked=0
n_skipped=0

# Walk the tree finding every per-session directory matching the REMBRANDT
# ID convention (900-00-* or HF*), regardless of how deeply it is nested.
while IFS= read -r -d '' session_dir; do
  n_total=$((n_total + 1))
  pid=$(basename "$session_dir")
  target="$STAGED_ROOT/$pid"
  if [ -L "$target" ] || [ -e "$target" ]; then
    n_skipped=$((n_skipped + 1))
    continue
  fi
  ln -s "$session_dir" "$target"
  n_linked=$((n_linked + 1))
done < <(/usr/bin/find "$SOURCE_ROOT" -mindepth 1 -type d \
            \( -name '900-00-*_[0-9][0-9][0-9][0-9].[0-9][0-9].[0-9][0-9]' \
               -o -name 'HF*_[0-9][0-9][0-9][0-9].[0-9][0-9].[0-9][0-9]' \) \
            -not -path '*__MACOSX*' -print0)

# Quarantine sessions known to be incomplete BEFORE HD-BET sees them.
INCOMPLETE=("HF0920_1991.06.14")
for pid in "${INCOMPLETE[@]}"; do
  target="$STAGED_ROOT/$pid"
  if [ -L "$target" ]; then
    quarantine="$STAGED_ROOT/.SKIPPED_${pid}_no_t1pre"
    rm -f "$quarantine"
    mv "$target" "$quarantine"
    echo "quarantined: $pid → .SKIPPED_${pid}_no_t1pre"
  fi
done

echo "staged $n_linked of $n_total sessions into $STAGED_ROOT (skipped existing: $n_skipped)"
