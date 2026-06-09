---
name: server3
description: VENA project-local override of the global `server3` skill. Project config at `.claude/server3.yaml`. Adds VENA-specific verification (decision.json schema 0.3.0, exhaustive_val per-cadence-epoch CSV check, dual-GPU co-residency) and traps (exhaustive_val silent-fail when every patient fails). All workflow logic — rsync sync, detached launch, ScheduleWakeup-based /loop monitoring, server3→picasso transfer, pgrep filtering — is in the global skill. Use whenever a VENA smoke test, FM training trial, exhaustive-val run, or related job must execute on icai-server.
---

# Server 3 (icai-server) — VENA overlay

VENA uses the global `server3` workflow with project-specific verification
rules. **Read both files before acting**:

1. **Global workflow** (canonical): `~/.claude/skills/server3/SKILL.md` —
   rsync rules, detached launch pattern, `/loop` + `ScheduleWakeup` monitor
   pattern, server3→picasso transfer, pgrep / silent-fail / divergence
   footguns.
2. **Project config** (paths, conda env, launch template, completion
   signal): `.claude/server3.yaml` at the VENA repo root.

This file adds only what does not generalise.

## VENA-specific facts

| Item | Value |
|---|---|
| SSH host | `icai-server` (alias `~/.ssh/config`; `User mariopascual`, port 33430, `ControlMaster` on) |
| Repo on server | `/home/mariopascual/projects/VENA` |
| Python / env | `~/.conda/envs/vena/bin/python` (PL 2.6.5 + editable `vena`) |
| Image / latent H5 root | `/media/hddb/mario/data/GLIOMAS/<COHORT>/h5/` |
| Multi-cohort corpus | `routines/fm/train/configs/corpus/corpus_server3.json` (UCSF-PDGM + BraTS-GLI) |
| MAISI checkpoints | `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/{diff_unet_3d_rflow-mr.pt,autoencoder_v2.pt}` |
| Latent-aug preflight | `/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json` |
| Experiments root | `/media/hddb/mario/experiments/<UTC>_s1_<hash>/` |
| Scratch logs | `/media/hddb/mario/smoke_logs/` |

The complete machine-readable form is `.claude/server3.yaml`.

## VENA run-directory verification (extends the global check)

After the three-layer completion check passes (log pattern + artifacts +
process), a VENA FM training run is healthy only if **all** of these also
hold (`<run> = experiments/<UTC>_<stage>_<hash>`):

- `logs/train.log` — ends with `FM-train completed`; no `Traceback` since
  the last `LOG opened`.
- `metrics/train_step.csv` — fully-populated columns (frozen header
  guarantees this); rows ≈ `n_optimizer_steps`.
- `metrics/train_epoch.csv` — N rows for N epochs; loss trend decreasing.
- `metrics/augmentations_per_epoch.csv` (if augmentations enabled) — non-zero counts.
- `checkpoints/` — `ema_epoch_NNN.ckpt` × n_epochs + `ema_best.ckpt` +
  `last.ckpt` + (when trunk trainable) `trunk_ema_snapshot.pt`.
- `decision.json` — `schema_version: "0.3.0"`, `trunk_checkpoint_sha256` /
  `vae_checkpoint_sha256` populated, `cohorts_used` non-empty
  (`.claude/rules/preflight-pattern.md` §"`decision.json` for training routines").
- `exhaustive_val/epoch_NNN/` per cadence epoch with `metrics.csv` having
  **`wc -l` > 1** (header + actual rows). See trap below.
- `exhaustive_val/gpu_usage.log` — co-residency: cuda:0 several GB +
  cuda:1 several GB at val launches (dual-GPU runs only).

## VENA-specific traps

### `exhaustive_val` silent-fail (real bug we hit)

`routines.fm.exhaustive_val` catches per-patient exceptions and logs a
WARNING per failure, then exits 0 **even when every patient failed**. The
training launcher reports `epoch N validation exit code 0` and training
proceeds — but `exhaustive_val/epoch_NNN/metrics.csv` contains only the
header.

Symptoms:

- `wc -l metrics.csv` = 1 (header only).
- `subprocess.log` has many `WARNING exhaustive-val: patient '<id>' failed (<error>); skipping.` lines.
- No `figure_best.png` / `figure_worst.png` (the `_render_best_worst`
  helper warns "no SSIM scores; skipping qualitative figures.").

**Always include the `wc -l metrics.csv` check in the VENA verification
pass**, even on runs that exited cleanly.

#### Replay a single epoch's exhaustive_val after a fix

Training does not need to re-run. The launcher already snapshotted the EMA
weights and wrote a self-contained job YAML:

```bash
ssh icai-server "cd /home/mariopascual/projects/VENA && \
  CUDA_VISIBLE_DEVICES=1 ~/.conda/envs/vena/bin/python -m routines.fm.exhaustive_val.cli \
  /media/hddb/mario/experiments/<run>/exhaustive_val/epoch_<NNN>/job.yaml 2>&1 | tail -30"
```

Each replay overwrites `epoch_NNN/`. Takes ~2.5 min.

### `pgrep` anchor — must include the env prefix

The project config sets `process_anchor: 'vena/bin/python -m routines.fm.train.cli'`.
The `vena/bin/python` prefix skips bash watchdog matches without an explicit
`grep -v 'bash -c'` filter, per the global skill's pgrep-traps section. If
the anchor is shortened to `routines.fm.train.cli`, every stale watchdog
from a prior session will be a false positive.

### Verifying both GPUs busy (dual-GPU runs)

GPU0 training is **bursty** (sub-second compute spikes); instantaneous
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

### Single-GPU vs dual-GPU launches

- **Single-GPU trial** (most smokes): set `CUDA_VISIBLE_DEVICES=0`.
- **Dual-GPU trial** (training + async exhaustive validation): launch with
  **both GPUs visible** — do NOT set `CUDA_VISIBLE_DEVICES`. Training uses
  `cuda:0`; the `ExhaustiveValLauncher` subprocess uses `cuda:1` (config
  `exhaustive_val.device`).

The project config's `default_cuda_visible_devices: "0"` is correct for the
default (smoke) case; override at invocation time with the dual-GPU recipe
for full runs.

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

This grounds the wake-interval choice: for `block_until_complete` runs a
1200 s (20 min) interval gives ~one wake during the run; for 4-h+ async
runs raise to 1800 s.

## Config conventions

- `routines/fm/train/configs/smoke/` — fast sanity configs (4 train subjects,
  a few micro-batches per epoch). Use for end-to-end / resume checks.
- `routines/fm/train/configs/runs/` — full-cohort runs.

`max_steps` / `total_steps` count **optimiser** steps; `max_epochs` caps
epochs (stops at whichever is reached first).

**Data path is registry-only.** Every training YAML carries
`data.corpus_registry`; the legacy `data.latents_h5` key is rejected at
config-validation time. To run a single-cohort smoke, write a single-cohort
registry JSON next to the multi-cohort one.

**Augmentation YAMLs gate at startup.** When `data.augmentation_config_path`
is set, `data.preflight_decision_path` is mandatory and every requested
augmentation must appear in the preflight's `latent_safe_augmentations`
allowlist; otherwise `_assert_preflight_gates` raises `PreflightGateError`
before any side effect. The default preflight is
`/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json`.

$ARGUMENTS
