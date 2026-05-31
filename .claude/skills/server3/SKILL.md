---
name: server3
description: Launch and monitor VENA training/eval trials on GPU server 3 (icai-server, 2x RTX 4090). Use whenever a smoke test or trial must run on a real GPU.
---

# Run a trial on GPU server 3

Server 3 (`icai-server`, host `icaigpuserver3`) has **2× RTX 4090 24 GB**. The
local workstation has the repo but **no `pytorch_lightning`**, so any run or
module-level test executes on the server. Use this skill for every GPU trial.

## Fixed facts

| Item | Value |
|---|---|
| SSH host | `icai-server` (alias in `~/.ssh/config`; `User mariopascual`, port 33430, `ControlMaster` on) |
| Repo on server | `/home/mariopascual/projects/VENA` |
| Python / env | `~/.conda/envs/vena/bin/python` (PL 2.6.5 + editable `vena`) |
| Image H5 / latent H5 cache root | `/media/hddb/mario/data/GLIOMAS/<COHORT>/h5/` |
| Corpus registries | `routines/fm/train/configs/corpus/corpus_server3.json` (UCSF-PDGM + BraTS-GLI) |
| MAISI checkpoints | `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/{diff_unet_3d_rflow-mr.pt,autoencoder_v2.pt}` |
| Latent-aug preflight | `/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json` |
| Experiment outputs | `/media/hddb/mario/experiments/<UTC>_s1_<hash>/` |
| Scratch logs | `/media/hddb/mario/smoke_logs/` (create with `mkdir -p` — does not persist) |

## Workflow

1. **Sync code with `rsync`, never `git push`/`pull`** (the FM tree is a working
   copy on the server; the user manages git separately). Batch many files in
   one call:
   ```bash
   rsync -avR <file1> <file2> <dir1/> <dir2/> ... icai-server:/home/mariopascual/projects/VENA/
   ```
   - `-R` preserves the *relative* path layout (run from repo root).
   - Trailing slash on a directory recurses into it.
   - **Deletions need an explicit `ssh ... rm -rf`** — rsync without `--delete`
     leaves orphan files on the server. If you removed a routine dir locally
     (e.g. cleanup of pycache-only dirs), mirror the delete on the server:
     ```bash
     ssh icai-server 'cd /home/mariopascual/projects/VENA && rm -rf routines/old_dir routines/fm/cli.py'
     ```
   - Verify when you doubt parity: `md5sum` locally vs
     `ssh icai-server "cd .../VENA && md5sum <files>"`.

2. **Run the non-GPU tests first** (catches import/contract breaks fast):
   ```bash
   ssh icai-server 'cd /home/mariopascual/projects/VENA && ~/.conda/envs/vena/bin/python -m pytest tests/ -q -m "not gpu and not slow"'
   ```
   Tests for FM-specific work also accept `tests/model/fm/` as a scope. A
   passing local suite is necessary but not sufficient — the server has the
   real Lightning + MAISI weights and surfaces what local mocking misses.

3. **Launch detached** so the trial survives the SSH session. Pick the GPU(s):
   - **Single-GPU trial** (most smokes): pin to one GPU.
     ```bash
     ssh icai-server 'cd /home/mariopascual/projects/VENA && mkdir -p /media/hddb/mario/smoke_logs && \
       LOG=/media/hddb/mario/smoke_logs/run_$(date +%Y%m%d_%H%M%S).log && \
       CUDA_VISIBLE_DEVICES=0 nohup ~/.conda/envs/vena/bin/python -m routines.fm.train.cli <config.yaml> > "$LOG" 2>&1 & disown; echo "LOG=$LOG"'
     ```
   - **Dual-GPU trial** (training + async exhaustive validation): launch with
     **both GPUs visible** — do NOT set `CUDA_VISIBLE_DEVICES`. Training uses
     `cuda:0`; the `ExhaustiveValLauncher` subprocess uses `cuda:1` (config
     `exhaustive_val.device`).
   - The launch call returns in <1 s. Its tool-notification means **the
     launcher exited cleanly**, NOT that training is done. Training is a
     separate detached process — see step 4 for the real completion signal.

4. **Monitor without blocking.** Two-layer pattern:

   a) **Process watch** — `run_in_background: true` Bash with a poll loop:
   ```bash
   ssh icai-server 'until ! pgrep -f "vena/bin/python -m routines.fm.train.cli" >/dev/null; do sleep 60; done; echo DONE_AT=$(date +%H:%M:%S)'
   ```
   You get one tool-notification when the loop exits. Pick the sleep cadence
   to fit the expected run length (60 s for ≤30 min runs, 5 min for hours).

   b) **`ScheduleWakeup` fallback** — long delay (1800–3600 s) so the loop
   survives if the watcher dies (e.g. someone killed its parent ssh, or
   `pgrep` self-matches and never reports done — see "pgrep traps" below).

   The true completion signals to verify are:
   - "FM-train completed" in `<run>/logs/train.log`
   - `<run>/checkpoints/ema_final.ckpt` and `last.ckpt` present
   - Subprocess no longer in `pgrep -af "routines.fm.train.cli"`

   After `Trainer.fit`, the process may stay alive briefly while `on_fit_end`
   joins the last exhaustive-val subprocess — that is normal, not a hang.

5. **Verify the run directory** (`/media/hddb/mario/experiments/<run>/`).
   The run is healthy only if ALL of these check out:
   - `logs/train.log` — ends with "FM-train completed"; no `Traceback`.
   - `metrics/train_step.csv` — fully populated columns (frozen header
     guarantees this); rows ≈ `n_optimizer_steps`.
   - `metrics/train_epoch.csv` — N rows for N epochs; loss trend decreasing.
   - `metrics/augmentations_per_epoch.csv` (if augmentations enabled) —
     non-zero counts.
   - `checkpoints/` — `ema_epoch_NNN.ckpt` × n_epochs + `ema_best.ckpt` +
     `last.ckpt` + (when trunk trainable) `trunk_ema_snapshot.pt`.
   - `decision.json` — `schema_version: "0.3.0"`, `trunk_checkpoint_sha256` /
     `vae_checkpoint_sha256` populated, `cohorts_used` non-empty.
   - `exhaustive_val/epoch_NNN/` per cadence epoch with `metrics.csv` having
     **`wc -l` > 1** (header + actual rows). See "subprocess silent-fail trap".
   - `exhaustive_val/gpu_usage.log` — co-residency: cuda:0 several GB +
     cuda:1 several GB at val launches.

## pgrep traps (the ones that bit me)

- **`pgrep -f` matches command lines, not process names.** A bash watchdog
  launched as `bash -c "... python -m routines.fm.train.cli ..."` matches the
  same pattern as the actual python process. Two consequences:
  1. The launcher's own bash matches briefly until it exits (~1 s); use
     `sleep 2` before the first poll if you launch and poll in the same call.
  2. **Stale watchdog loops from prior sessions match too.** If a prior
     watchdog is still polling for the same pattern, your new watcher will
     also match it and never report done. Either filter or kill:
     ```bash
     # Filter: exclude bash launchers/watchdogs (pattern in their argv).
     pgrep -f "vena/bin/python -m routines.fm.train.cli" | grep -qv "bash -c"

     # Or list and kill the stale watchdogs explicitly first:
     ssh icai-server 'pgrep -af "vena/bin/python -m routines.fm.train.cli"'
     ssh icai-server 'kill <pid1> <pid2> ...'
     ```
   The python-specific anchor `vena/bin/python -m routines.fm.train.cli` is
   precise enough to skip bash lines if you also filter out `bash -c`.

- **Always inspect `pgrep -af "..."`** (not just `-f`) before relying on
  exit-or-not. The argv tells you which processes are watchdogs vs real work.

## Subprocess silent-fail trap (real bug we hit)

`routines.fm.exhaustive_val` catches per-patient exceptions and logs a
WARNING per failure, then exits 0 even when **every** patient failed. The
training process's launcher reports `epoch N validation exit code 0` and
training proceeds — but `exhaustive_val/epoch_NNN/metrics.csv` contains only
the header. Symptoms:

- `wc -l metrics.csv` = 1 (header only).
- `subprocess.log` has many `WARNING exhaustive-val: patient '<id>' failed (<error>); skipping.` lines.
- No `figure_best.png` / `figure_worst.png` (the `_render_best_worst` helper
  warns "no SSIM scores; skipping qualitative figures.").

**Always check `wc -l` on `metrics.csv`** during verification — `exit 0` is
necessary, not sufficient.

### Replay a single epoch's exhaustive_val after a bug fix

If training succeeded but a regression broke the exhaustive subprocess, you
do **not** need to retrain. The launcher already snapshotted the EMA weights
and wrote a self-contained job YAML. Replay with the fix in place:

```bash
ssh icai-server "cd /home/mariopascual/projects/VENA && \
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vena/bin/python -m routines.fm.exhaustive_val.cli \
  /media/hddb/mario/experiments/<run>/exhaustive_val/epoch_<NNN>/job.yaml 2>&1 | tail -30"
```

The job YAML carries all paths and hyperparameters; the snapshot `.pt` next
to it is the EMA shadow at that epoch. Each replay overwrites the same
`epoch_NNN/` dir. Quick way to confirm the fix end-to-end without a 20-min
training.

## Verifying both GPUs are busy (dual-GPU runs)

GPU0 training is **bursty** (sub-second compute spikes), so instantaneous
`utilization.gpu` snapshots understate overlap. Use **memory co-residency**
as the robust signal:

```bash
ssh icai-server 'F=/media/hddb/mario/smoke_logs/gpu_ts.log; for i in $(seq 1 120); do \
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader >> "$F"; sleep 2; done'
```

Then count timestamps with both indices > ~1 GB. For a definitive snapshot,
`nvidia-smi --query-compute-apps=pid,used_memory` shows which PID is on
which GPU. The training process is the canonical 12 GB+ resident on cuda:0;
the per-epoch validation subprocess holds 2–4 GB on cuda:1 only during its
~3–5 min lifetime.

## Timing reference (4-epoch multi-cohort + per-epoch blocking val)

Measured 2026-05-30 on `server3_4epoch_aug.yaml`:

| Phase | Wall-clock |
|---|---|
| Setup (trunk load, ControlNet init, EMA build) | ~30 s |
| Training, one epoch (1255 scans, batch 2, bf16) | ~2.5 min |
| Exhaustive val, one epoch (20 patients × 5 NFE) | ~3–5 min |
| **Total 4-epoch run (block_until_complete)** | **~20 min** |
| Single-epoch exhaustive replay (fix-verification) | ~2.5 min |

`exhaustive_val.block_until_complete: true` waits for each epoch's val
before launching the next epoch's training — slower than the production
async/skip-if-busy mode but lets you catch problems epoch-by-epoch.

## Config conventions

Configs are split into two folders:
- `routines/fm/train/configs/smoke/` — fast sanity configs (`smoke.yaml`,
  `smoke_short`, `smoke_full`, `smoke_polish`, `smoke_exhaustive`,
  `smoke_trunk_unfrozen[_resume]`). 4 train subjects, ⇒ a few micro-batches per
  epoch; use for end-to-end / resume checks.
- `routines/fm/train/configs/runs/` — serious full-cohort runs (`server3.yaml`,
  `baseline_100ep.yaml`, `server3_trunk_{unfrozen,frozen}.yaml`,
  `server3_4epoch_exhaustive[_frozen].yaml`,
  `server3_4epoch_aug.yaml` — multi-cohort + augmentations + dual-GPU
  exhaustive val).

`max_steps`/`total_steps` count **optimiser** steps; `max_epochs` caps epochs
(stops at whichever is reached first). Keep smokes short but exercise every
path (checkpoint rotation, NFE sweep, exhaustive cadence, resume-in-place).

**Data path is registry-only.** Every training YAML carries
`data.corpus_registry`; the legacy `data.latents_h5` key is rejected at
config-validation time with a clear error. To run a single-cohort smoke,
write a single-cohort registry JSON next to the multi-cohort one.

**Augmentation YAMLs gate at startup.** When `data.augmentation_config_path`
is set, `data.preflight_decision_path` is mandatory and every requested
augmentation must appear in the preflight's `latent_safe_augmentations`
allowlist; otherwise `_assert_preflight_gates` raises `PreflightGateError`
before any side effect. Use
`/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json`
as the default preflight.

## Editing on the server vs locally

Edit locally, rsync up. Never edit on the server — divergence is silent and
the formatter that runs locally on Edit may rewrite imports that a
manually-edited server copy lacks. If you do need to inspect server state
mid-run, treat it as read-only: `ssh icai-server 'cat / tail / less <path>'`.

## Cleanup at session end

Kill any stale watchdog loops left from this session:
```bash
ssh icai-server 'pgrep -af "until ! pgrep -f \"vena/bin/python" | head -10'
ssh icai-server 'kill <stale-pids>'
```
They are harmless (just a `sleep` loop) but they will trap the next
session's pgrep-based watcher into never reporting done.

$ARGUMENTS
