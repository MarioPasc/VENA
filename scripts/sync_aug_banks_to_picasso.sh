#!/bin/bash
# Drive server3 -> Picasso transfer of merged offline-aug H5 banks.
#
# Polls every 60 s for cohorts whose merged image+latent aug H5 are present
# on disk and not yet transferred (.synced_picasso sentinel absent), then
# rsyncs each pair to Picasso under
# /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<DEST>/h5/.
# Sentinel is written only after both files arrive successfully.
#
# Runs in nohup + disown on icai-server; logs to PICASSO_LOG.
set -u
COHORTS=(
  "UCSF_PDGM|ucsf_pdgm_image_aug.h5|ucsf_pdgm_latents_aug.h5|UCSF_PDGM"
  "BRATS_GLI/PRE_OPERATIVE|brats_gli_image_aug.h5|brats_gli_latents_aug.h5|BRATS_GLI"
  "upenn_gbm|upenn_gbm_image_aug.h5|upenn_gbm_latents_aug.h5|upenn_gbm"
  "ivy_gap|ivy_gap_image_aug.h5|ivy_gap_latents_aug.h5|ivy_gap"
  "lumiere|lumiere_image_aug.h5|lumiere_latents_aug.h5|lumiere"
  "rembrandt|rembrandt_image_aug.h5|rembrandt_latents_aug.h5|rembrandt"
)
SRC_ROOT=/media/hddb/mario/data/GLIOMAS
DST_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena
PICASSO_HOST=picasso3.scbi.uma.es
LOG=/media/hddb/mario/smoke_logs/offline_aug/picasso_transfer.log
SENT_DIR=/media/hddb/mario/smoke_logs/offline_aug/sentinels
mkdir -p "$SENT_DIR"
mkdir -p "$(dirname "$LOG")"

log() {
  echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"
}

transfer_one() {
  local subdir="$1" image_name="$2" latent_name="$3" dest_cohort="$4"
  local image_src="$SRC_ROOT/$subdir/h5/$image_name"
  local latent_src="$SRC_ROOT/$subdir/h5/$latent_name"
  local sentinel="$SENT_DIR/${dest_cohort}.synced"
  if [ -e "$sentinel" ]; then
    return 0
  fi
  if [ ! -f "$image_src" ] || [ ! -f "$latent_src" ]; then
    return 1  # not ready
  fi
  log "START $dest_cohort image=$image_src latent=$latent_src"
  # Make destination dir on picasso.
  if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "$PICASSO_HOST" \
        "mkdir -p $DST_ROOT/$dest_cohort/h5" >> "$LOG" 2>&1; then
    log "FAIL mkdir picasso for $dest_cohort"
    return 1
  fi
  # rsync image then latent; --partial --append-verify so safe to resume.
  if ! rsync -av --partial --append-verify --inplace --no-whole-file \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30" \
        "$image_src" "$PICASSO_HOST:$DST_ROOT/$dest_cohort/h5/" \
        >> "$LOG" 2>&1; then
    log "FAIL rsync image for $dest_cohort"
    return 1
  fi
  if ! rsync -av --partial --append-verify --inplace --no-whole-file \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30" \
        "$latent_src" "$PICASSO_HOST:$DST_ROOT/$dest_cohort/h5/" \
        >> "$LOG" 2>&1; then
    log "FAIL rsync latent for $dest_cohort"
    return 1
  fi
  date -u +%FT%TZ > "$sentinel"
  log "DONE  $dest_cohort"
  return 0
}

log "driver started; cohorts=${#COHORTS[@]}; PID=$$"
while :; do
  remaining=0
  for entry in "${COHORTS[@]}"; do
    IFS='|' read -r subdir image_name latent_name dest_cohort <<< "$entry"
    sentinel="$SENT_DIR/${dest_cohort}.synced"
    if [ -e "$sentinel" ]; then
      continue
    fi
    if transfer_one "$subdir" "$image_name" "$latent_name" "$dest_cohort"; then
      :  # synced
    else
      remaining=$((remaining + 1))
    fi
  done
  if [ "$remaining" -eq 0 ]; then
    log "all cohorts synced; exiting"
    break
  fi
  sleep 60
done
