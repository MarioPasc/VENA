# Pre-launch Tasks — 7-day Picasso Training Run

**Date:** 2026-06-05 (refreshed 2026-06-19 with the post-audit-fix data state).

## 2026-06-19 update (post data-audit fix-up)

Picasso production data is now consistent end-to-end:

- **9 cohorts in `corpus_picasso.json`** (BraTS-PED re-included as
  `test_only`, 260 patients; image H5 was retransferred 2026-06-18 to fix
  the 3.95→5.1 GB truncation).
- **Three audit-§0 critical findings closed**: v4 brain-mask synth-ones,
  BraTS-Africa z-score skew, IvyGAP CC noise — see
  `.claude/notes/data/2026-06-19_data_audit_v2.md`.
- **Aug-image + aug-latent schema bumped to 0.2.0** (added `masks/brain`
  to the aug-image manifest; conditional `masks/brain_latent` validator on
  the aug-latent gated on root attr `produced_by_brain_to_latent`).
- **Splits unified**: every cv cohort image + latent H5 carries
  `splits/cv/fold_N/* + splits/test`; every test_only cohort carries
  `splits/test`. REMBRANDT image H5 needed an in-place `splits/cv` copy
  from the latent because the converter never wrote it natively.
- **Loader fixes**: `routines/fm/exhaustive_val/engine.py::_cohort_val_patients`
  now falls back to `splits/test` when `splits/cv/fold_<fold>/val` is
  absent, so test_only cohorts (now without the legacy `splits/cv/fold_0/val`
  alias) are still iterated. BraTS-PED is automatically picked up
  (`registry.test_cohorts()` enumerates 3 cohorts now).
- **Tests**: 771/771 pass (`pytest -m "not slow and not gpu"`); +15 new
  unit tests across encoder mask, bank-builder brain LabelMap, conditional
  validator, splits normalize, recompute_union_of_four, seed-replay.

Pre-launch acceptance gate (below) is unchanged in shape — the launchers
land cleanly because the FM training data path only reads
`splits/cv/fold_<fold>/{train,val} + splits/test` on cv cohorts and that
form is present everywhere.


**Status:** Three 1000-epoch Picasso configurations (S1, S2+LoRA r=16, S2+FFT) are
written and validated locally. SLURM scripts at `routines/fm/train/slurm/runs/`
land cleanly with `bash launcher_picasso_<stage>.sh --dry-run`. All Picasso-side
preflight artifacts have been (re-)synced. PEFT + transformers installed on
Picasso vena env. 401 fast tests pass locally (398 + 3 new patience tests).
Acceptance gate: the user runs `bash launcher_picasso_s1.sh`,
`bash launcher_picasso_s2_lora.sh`, `bash launcher_picasso_s2_fft.sh` on the
Picasso login node.

**Historical context (2026-06-01):** Lp-aware contrastive (S2) lands in code; smoke runs validated on icai-server (4 epochs each, ablation-clean cfm, Δ_WT/Δ_BG = 15.5× by epoch 3); see `scratch/2026-06-01_s2_smoke_results.md` for the numbers. The 4-epoch smoke is the floor for stability; nothing below blocks training mathematically.

**P1 (Diagnostic completeness) and P3 (Robustness niceties) IMPLEMENTED.** See the corresponding sections below for `[x]` markers. 334 fast tests pass locally and on server 3. Server-3 validation smoke (S2 + new logging) ran clean — see *Validation smoke* at the bottom of this file.

## Picasso audit (live, refreshed 2026-06-05)

Repo on Picasso: `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA`
(synced; `/mnt2/fscratch/users/tic_163_uma/mpascual/repos/VENA` is the same
inode set seen from compute nodes).
Data on Picasso: `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<cohort>/h5/`.

### Cohort H5 inventory

| Cohort | image H5 | latent H5 | image_aug_h5 | latent_aug_h5 | role | usable? |
|---|---|---|---|---|---|---|
| UCSF-PDGM (495) | ✅ | ✅ | ✅ | ✅ | cv | yes |
| BraTS-GLI pre-op (1133) | ✅ (under `BRATS_GLI/PRE_OPERATIVE/h5/`) | ✅ | ✅ | ✅ | cv | yes |
| UPENN-GBM (611) | ✅ | ✅ | ✅ | ✅ | cv | yes |
| IvyGAP (34) | ✅ | ✅ | ✅ | ✅ | cv | yes |
| LUMIERE (91 pat, 599 scans) | ✅ | ✅ | ✅ | ✅ | cv | yes |
| REMBRANDT (63) | ✅ | ✅ | ✅ | ✅ | cv | yes |
| BraTS-Africa-Glioma (95) | ✅ | ✅ | — | — | test_only | yes |
| BraTS-Africa-Other (51) | ✅ | ✅ | — | — | test_only | yes |
| BraTS-PED (260) | ✅ 5.1 GB intact (retransferred 2026-06-18 via server3→/tmp→picasso) | ✅ (1.9 GB intact; `masks/brain_latent` populated 2026-06-18) | — | — | test_only | **yes — back in `corpus_picasso.json` 2026-06-19** |

### Assets

| Asset | Status |
|---|---|
| MAISI VAE + FM trunk checkpoints | ✅ |
| Conda env `vena` (torch, PL 2.6+, monai, h5py, pydantic, einops, nibabel) | ✅ |
| `peft 0.19.1` + `transformers 4.57` installed in `vena` env | ✅ (installed 2026-06-05) |
| `vena` package editable install | ✅ resolves to `src/vena` under `repos/VENA` |
| `corpus_picasso.json` carries `image_aug_h5` + `latent_aug_h5` | ✅ (updated 2026-06-05) |
| Latent-aug equivariance decision (`fscratch/artifacts/latent_aug_equivariance/LATEST/decision.json`) | ✅ (synced 2026-06-05 from server3 LATEST) |
| Cohort-dedup decision (`fscratch/artifacts/preflights/cohort_dedup/LATEST/decision.json`) | ✅ (re-generated 2026-06-05 against `corpus_picasso.json` SHA) |
| `BraTS2021_MappingToTCIA.xlsx` at `fscratch/datasets/` | ✅ (synced 2026-06-05 for dedup re-run) |
| Per-run experiments root `execs/vena/experiments/` | created on first run |
| Per-run logs `execs/vena/logs/` | created by launcher |
| SLURM `routines/fm/train/slurm/runs/{worker,launcher_picasso_s1,launcher_picasso_s2_lora,launcher_picasso_s2_fft}.sh` | ✅ (added 2026-06-05) |
| Picasso training YAMLs (`picasso_s1_1000ep.yaml`, `picasso_s2_1000ep_lora_r16.yaml`, `picasso_s2_1000ep_fft.yaml`) | ✅ (added 2026-06-05) |

### Net blockers — 2026-06-05

All blockers from the 2026-06-01 audit are resolved. The three Picasso YAMLs +
the shared worker + three launchers are in place. `sbatch --test-only` on
the worker accepts the 7-day / 2 × A100 / `--constraint=dgx` ask (verified
2026-06-05: SLURM would start on `exa03`, queue wait depends on partition
load).

### Deferred — BraTS-PED H5 re-transfer

The BraTS-PED `image_h5` on Picasso is truncated (3.95 GB / 5.42 GB; the
`latent_h5` is intact). The cohort is `test_only` so it does not affect
training; it has been removed from `corpus_picasso.json` for the production
runs. When the user has time to re-transfer the file (server 3 → local →
Picasso, ~5 GB at ~15 MB/s = ~5 minutes on a faster window), the procedure
to re-add is:

1. `rsync icai-server:/media/hddb/mario/data/GLIOMAS/brats_ped/h5/BraTS_PED_image.h5
   /tmp/ && rsync /tmp/BraTS_PED_image.h5
   picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/brats_ped/h5/`
2. Re-add the BraTS-PED block to `corpus_picasso.json` (same shape as the
   other test_only cohorts).
3. Re-run `python -m routines.preflights.cohort_dedup.cli
   routines/preflights/cohort_dedup/configs/default_picasso.yaml` on Picasso
   to refresh the decision SHA against the new corpus JSON, then point
   `LATEST` at the new dir.

**Scope of this list.** Out of scope: model validation, reader study, paper writing. In scope: everything that (a) would block the 7-day SLURM job from running to completion, or (b) we need decided/wired *now* so the diagnostic + sampling data we want from the run is actually captured, instead of discovered missing on day 7.

Tasks are ordered by blocking-ness. Each names the file(s) and the acceptance check.

---

## 2026-06-05 — New production setup (superseding P0)

Three configurations under `routines/fm/train/configs/runs/`:

| File | Stage | Trunk regime | Loss | Notes |
|---|---|---|---|---|
| `picasso_s1_1000ep.yaml` | s1 | fft | cfm only | baseline |
| `picasso_s2_1000ep_lora_r16.yaml` | s2 | peft (LoRA r=16, QKVO) | cfm + Lp-aware contrastive | recommended production arm |
| `picasso_s2_1000ep_fft.yaml` | s2 | fft | cfm + Lp-aware contrastive | head-to-head FFT vs LoRA |

Shared training settings (all three): `max_epochs=1000`, `patience=100` (new
EarlyStopping wiring on `train/total_epoch`), `bs=4 × grad_accum=2` (effective
batch 8 on A100 40 GB), `exhaustive_val every_epochs=50` async on cuda:1,
`block_until_complete=false`, `n_patients=60` (10 / cohort × 6 cv cohorts),
`nfe_levels=[1,5,20]`, `resume_from=latest`, `precision=bf16-mixed`. Offline
aug bank with uniform variant weights `{v0..v4: 0.2}`. Online latent aug:
flip_lr + translate (equivariant_v1).

Three logging additions land in this push so the SLURM `.out` / `train.log`
files actually surface useful state across the 7-day run:

1. `TrainMetricsCSV.on_train_epoch_end` emits one INFO line per epoch with
   total / cfm / contrastive / gpu_peak / samples_per_sec / elapsed_s.
2. `ExhaustiveValLauncher` parses the child's `metrics.csv` after each async
   pass and emits one INFO line per NFE with mean ± std PSNR / SSIM /
   latent_mse / gen_sec.
3. `EarlyStopping(monitor="train/total_epoch", mode="min", patience=100,
   verbose=True)` emits Lightning's standard "best so far + epochs without
   improvement" line, plus the engine logs `EarlyStopping ENABLED: ...` at
   startup. If the loss plateaus, training releases the allocation early.

SLURM (`routines/fm/train/slurm/runs/`):

- Shared worker `worker_fm_train_picasso.sh`:
  `--time=7-00:00:00`, `--gres=gpu:2`, `--constraint=dgx`,
  `--partition=gpu_partition`, `--cpus-per-task=16`, `--mem=256G`,
  `--output/--error=execs/vena/logs/%x_%j.{out,err}`. SIGTERM-aware
  auto-resubmit (rc ∈ {124,137,143}) when the YAML has `resume_from: latest`.
- Per-stage launchers `launcher_picasso_s1.sh`, `launcher_picasso_s2_lora.sh`,
  `launcher_picasso_s2_fft.sh`. Each exports the appropriate `CONFIG_PATH` +
  `JOB_NAME` and submits the shared worker. `--dry-run` mode prints the
  resolved `sbatch` command without submitting.

Acceptance — the user runs three commands from the Picasso login node:

```bash
bash routines/fm/train/slurm/runs/launcher_picasso_s1.sh
bash routines/fm/train/slurm/runs/launcher_picasso_s2_lora.sh
bash routines/fm/train/slurm/runs/launcher_picasso_s2_fft.sh
```

Each writes a job id; each lands in its own
`execs/vena/experiments/<UTC>_<stage>_<sha>/` with self-contained
`logs/train.log` + `metrics/{train_step,train_epoch}.csv` + per-50-epoch
`exhaustive_val/epoch_NNN/{metrics,timing}.csv`.

---

## P0 — Hard blockers (must exist before sbatch)  **[historical — closed 2026-06-05]**

### P0.1  SLURM launcher + worker for `routines/fm/train`

**Why:** Picasso is Singularity + SLURM only. The trainer has no `slurm/` directory at all; the only existing pair is `routines/encode/maisi/slurm/{launcher,worker}_encode_maisi.sh`. Copy that pair into `routines/fm/train/slurm/`, adapting:

- `#SBATCH --time=4-00:00:00` (Picasso typical per-job limit; longer runs need a chain).
- `#SBATCH --gres=gpu:2` (training on `cuda:0` + exhaustive val on `cuda:1`).
- `#SBATCH --constraint=dgx`.
- `--cpus-per-task=16`, `--mem=128G`.
- The worker invokes `python -m routines.fm.train.cli "${CONFIG_PATH}"` — same surface as on icai-server.
- Auto-resume: if `${RUN_DIR}/checkpoints/last.ckpt` exists *and* the YAML's `run.resume_from: latest` is set, the worker should just re-`sbatch` itself (SLURM job chain) at the end so a 7-day run completes via 2× 4-day jobs.

**Files**

- `routines/fm/train/slurm/launcher_fm_train.sh` (new)
- `routines/fm/train/slurm/worker_fm_train.sh` (new)
- The exhaustive-val subprocess inherits the same allocation (it's spawned from the trainer), so no separate SLURM script is needed.

**Acceptance:** `sbatch --test-only routines/fm/train/slurm/worker_fm_train.sh` passes; `--dry-run` of the launcher prints the resolved config and the sbatch flags.

**Notes:** Use the `picasso-sbatch` skill — it knows the conventions and produces this kind of pair to spec.

---

### P0.2  Picasso training YAML (S1 baseline + S2 follow-on)

**Why:** Every YAML under `routines/fm/train/configs/runs/` has icai-server paths (`/media/hddb/mario/...`). Picasso needs its own YAML.

**Files**

- `routines/fm/train/configs/runs/picasso_s1_4d.yaml` (new) — 4-day S1 baseline.
- `routines/fm/train/configs/runs/picasso_s2_3d.yaml` (new) — 3-day S2 fine-tune from S1's `ema_best`.

Settings to change vs `smoke_s1_4ep_logging.yaml`:

| Key | Smoke value | Picasso value |
|---|---|---|
| `data.corpus_registry` | `corpus_server3.json` | `corpus_picasso.json` (already exists) |
| `data.preflight_decision_path` | `/media/hddb/...` | `/mnt/home/users/tic_163_uma/mpascual/fscratch/artifacts/latent_aug_equivariance/LATEST/decision.json` (see P0.4) |
| `model.trunk.checkpoint` | `/media/hddb/...` | `/mnt/home/.../fscratch/checkpoints/NV-Generate-MR/diff_unet_3d_rflow-mr.pt` |
| `model.vae_checkpoint` | `/media/hddb/...` | `/mnt/home/.../fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt` |
| `output.experiments_root` | `/media/hddb/mario/experiments` | `/mnt/home/.../fscratch/experiments/vena` |
| `training.max_epochs` | 4 | enough to fill 4 days at A100 throughput (preflight estimate, see P0.3) |
| `training.batch_size` | 2 | 4 (A100 40 GB headroom: 17.6 GB at S2 + headroom for cohort batches) |
| `training.grad_accum` | 1 | 1 (raise if OOM at batch=4) |
| `exhaustive_val.block_until_complete` | true | **false** (production async + skip-if-busy) |
| `exhaustive_val.every_epochs` | 1 | 10 (4-day cadence: ~30 epochs at A100; 3 exhaustive snapshots is plenty) |
| `exhaustive_val.n_patients` | 20 | 80 (full multi-cohort val coverage — partition is uniform across cohorts) |
| `exhaustive_val.nfe_levels` | `[1, 2, 5, 10, 20]` | `[1, 5, 20]` (drop NFE 2 and 10 — they add latent noise without resolution; saves ~40% of exhaustive-val wall-clock) |
| `output.retention_n_checkpoints` | 5 | 3 (3 × 3.86 GB ≈ 12 GB + `ema_best` + `last`, ≈ 20 GB total — comfortable under fscratch quota) |
| `run.resume_from` | null | `latest` (so SLURM chain auto-resumes) |

S2 YAML adds the `loss.contrastive` block from `smoke_s2_4ep.yaml` and sets `run.resume_from: <path to S1 ema_best.ckpt>` plus `run.stage: s2`.

**Acceptance:** `vena-fm-train --help` parses both YAMLs without error (the existing Pydantic `from_yaml` validates everything that matters).

---

### P0.3  Throughput sanity check on Picasso (one-shot smoke)

**Why:** All current throughput numbers are from RTX 4090. A100 40 GB may be faster or slower depending on bf16 fall-through and the controlnet IO pattern. Need one short smoke (1 epoch, `max_train_patients_per_cohort: 16`) on Picasso to measure `samples/s` and set `max_epochs` for the real run.

**Procedure:** Create `routines/fm/train/configs/runs/picasso_smoke_1ep.yaml` (a `picasso_s1_4d.yaml` with `max_epochs: 1`, `n_patients: 8` exhaustive). Submit. Read `metrics/train_epoch.csv` → `samples_per_sec_mean`. Compute the epoch wall-clock and back-solve `max_epochs` for a 3.5-day budget (leaving 12 h for the final exhaustive val + checkpoint retention).

**Acceptance:** Picasso `samples_per_sec` measured; `max_epochs` set in the production YAMLs accordingly. Expected ~30–50 samples/s on A100 (compared to 30 on RTX 4090 for S1; the bigger batch will push it).

---

### P0.4  Latent-aug preflight artifact on Picasso fscratch

**Why:** `_assert_preflight_gates` hard-fails at startup if the preflight `decision.json` is absent. Currently it lives at `/media/hddb/mario/artifacts/latent_aug_equivariance/LATEST/decision.json` on icai-server. On Picasso fscratch, `/mnt/home/.../fscratch/artifacts/` is empty (verified during this session).

**Two options:**

1. **Rsync the existing artifact** from icai-server to Picasso. Cheapest; preserves the SHA-256 lineage in the smoke run's `decision.json`. `rsync -av icai-server:/media/hddb/mario/artifacts/latent_aug_equivariance/ <picasso>:/mnt/home/.../fscratch/artifacts/latent_aug_equivariance/`. Fix the `LATEST` symlink afterwards (rsync follows the link; we want it preserved).

2. **Rerun** the preflight on Picasso. Cleanest provenance (the decision was computed on the actual hardware) but ~3 h of A100 time. Only worth doing if there's any reason to suspect the cohort decision (which there isn't — strict 35 dB / 0.95 SSIM is hardware-agnostic).

**Decision needed.** Recommendation: option 1.

**Acceptance:** `ls /mnt/home/.../fscratch/artifacts/latent_aug_equivariance/LATEST/decision.json` succeeds; `jq .latent_safe_augmentations <that file>` returns the expected list.

---

### P0.5  Verify the multi-cohort latent H5s are on Picasso fscratch

**Why:** `corpus_picasso.json` points at e.g. `/mnt/home/.../fscratch/datasets/vena/UCSF_PDGM/h5/UCSFPDGM_latents.h5`. If any of the per-cohort H5s is missing or is the image-domain file (not the latent file) the training fails on first batch. Some cohorts in the registry that exist on icai-server (BraTS-PED, IvyGAP, BraTS-Africa-*) may not have been mirrored yet.

**Procedure:** SSH to Picasso login node and `for p in $(jq -r '.cohorts[].latent_h5' corpus_picasso.json); do ls -la "$p" || echo MISSING; done`. For any MISSING, either (a) rsync from icai-server, or (b) drop the cohort from the registry for this run.

**Acceptance:** every `latent_h5` path in `corpus_picasso.json` resolves to a real H5 with `schema_version` attr present. Same check for `image_h5` (needed by exhaustive_val).

---

### P0.6  Singularity image with the current `vena` env

**Why:** Picasso is Singularity-only. The encode routine already has a working pattern (`worker_encode_maisi.sh` loads a conda env that lives in the singularity image). Confirm the same image (a) has `pytorch_lightning==2.6.5`, (b) has MONAI, (c) has the in-tree `vena` package installed in editable mode against `${REPO_DIR}/src`.

**Two paths:**

1. The encode worker shows `conda activate vena` working — if `vena` env in the image already has PL + MONAI, just pip install -e from `${REPO_DIR}` at job start (the encode worker pattern). Cheapest.
2. If PL is missing, build a new image (~30 min) with a `.def` file and push to fscratch.

**Acceptance:** `worker_fm_train.sh --dry-run` succeeds — meaning the env activation block from the encode worker, copy-pasted, returns a python whose `pytorch_lightning` import succeeds.

---

## P1 — Diagnostic completeness (cheap, schedule together)  **[done 2026-06-01]**

All four P1 items implemented + tested. The validation smoke on server 3 ran an S2 4-epoch run with all new code paths exercised cleanly (anneal schedule, per-cohort cfm, top-K figures, snapshot pruner). Numbers in `scratch/2026-06-01_s2_smoke_results.md`'s addendum.

### [x] P1.1  λ_contrast anneal 0.01 → 0.001 at step half (proposal §3)

**Why:** The proposal pins this; the smoke ran at fixed `λ = 0.01` because 4 epochs is too short for an anneal to matter. For a 3-day S2 leg it matters: the anneal is what prevents the contrastive from biasing the converged model.

**Implementation sketch.** Add a `schedule:` block to the `loss.contrastive` cfg:

```yaml
loss:
  contrastive:
    weight: 0.01
    schedule:
      step_half_factor: 0.1   # at total_steps/2, multiply weight by this
      kind: step              # alternatives could be 'linear', 'cosine' — keep step
```

`CompositeLoss` reads this via the builder and exposes `current_weight(global_step, total_steps)` instead of the static dict lookup at line 114. The simplest change is to make `CompositeLoss.weights` callable — a `Callable[[int, int], dict[str, float]]` — and have the LightningModule pass `(trainer.global_step, trainer.estimated_stepping_batches)` into `composite(inputs, step, total_steps)`.

**Files**

- `src/vena/model/fm/controlnet/losses/base.py` — add a `WeightSchedule` helper, update `CompositeLoss.forward` signature.
- `src/vena/model/fm/controlnet/losses/builder.py` — parse `schedule:` block, default = static.
- `src/vena/model/fm/lightning/module.py` — pass step into `self.composite(...)`.
- Unit test in `tests/model/fm/test_losses_contrastive.py` checking the weight at step 0 = 0.01 and at total/2 = 0.001.

**Acceptance:** `train/contrastive` in train_step.csv shows a visible step-down at the midway step.

---

### [x] P1.2  Per-cohort loss breakdown in `train_epoch.csv`

**Why:** Flagged as out-of-scope in the smoke plan, but for a multi-cohort run lasting a week, cohort-imbalanced loss drift is the most likely failure mode that won't trigger a NaN guard. Catching "BraTS-PED is 2× the loss of UCSF-PDGM and getting worse" requires the per-cohort breakdown.

**Implementation sketch.** The DataModule already tags each batch element with a `cohort` field (the temperature sampler needs it). The LightningModule needs to (a) split `cfm` and `total` across cohort tags per step and (b) accumulate per-cohort to `_EPOCH_AGG_KEYS`-equivalents.

**Files**

- `src/vena/model/fm/lightning/module.py` — in `training_step`, after the loss call, compute `per_cohort_cfm[c] = F.mse_loss(v_orig[mask_c], u_target[mask_c])` for each cohort. Log under `train/cfm_<cohort>`.
- `src/vena/model/fm/lightning/callbacks/train_csv.py` — extend `_EPOCH_AGG_KEYS` discovery to pick up `cfm_<cohort>` automatically (it currently uses a fixed tuple).

**Acceptance:** `train_epoch.csv` contains `cfm_UCSF-PDGM_mean`, `cfm_BraTS-GLI_mean`, etc.

**Caveat:** Adds B-many indexing ops per step. At batch=4 negligible. Worth measuring on the smoke.

---

### [x] P1.3  Wire `figure_top_k` through the trainer config

**Why:** I added `figure_top_k = 3` as the default on `ExhaustiveValJobConfig`, but `_ExhaustiveValCfg` in `routines/fm/train/engine.py` does not pass it through `_build_exhaustive_job_base`. So the default (3) takes effect but it cannot be tuned per-run. Trivial fix.

**Files**

- `routines/fm/train/engine.py` — add `figure_top_k: int = 3` to `_ExhaustiveValCfg` and pass it into the `job` dict in `_build_exhaustive_job_base`.

**Acceptance:** Setting `exhaustive_val.figure_top_k: 5` in a training YAML produces 10 panels per epoch dir.

---

### [x] P1.4  Bound checkpoint disk footprint

**Why:** Each checkpoint is ~3.86 GB (trunk + ControlNet + EMA shadows). At `retention_n_checkpoints=5` and 30 epochs that is 19 GB live but the older epochs are not auto-pruned beyond the retention window — `ema_best` is preserved separately. Per epoch there is also `exhaustive_val/epoch_NNN/{ema_snapshot.pt, trunk_ema_snapshot.pt}` (~1 GB combined). Across 30 epochs of exhaustive snapshots = 30 GB even at every_epochs=10 → 3 snapshots × 1 GB = 3 GB. OK.

**Action:** confirm the production `output.retention_n_checkpoints: 3` is honoured. Add a one-time `ExhaustiveSnapshotPruner` callback that, after exhaustive_val for epoch N completes, deletes `epoch_(N-2)/ema_snapshot.pt` and `trunk_ema_snapshot.pt` (the latent_preds.h5 + metrics.csv stay forever — they are the diagnostic record).

**Files**

- `src/vena/model/fm/lightning/callbacks/exhaustive_launcher.py` — extend with the prune-on-success branch.

**Acceptance:** the run finishes with at most `retention_n_checkpoints` ckpts plus `ema_best` plus `last`; old `ema_snapshot.pt` files removed.

---

## P2 — Sampling pipeline (so the model is *useful* on day 7)

### P2.1  `routines/fm/inference` — load checkpoint, sample T1c volumes

**Why:** After 7 days of training we have a checkpoint and zero infrastructure to use it on a held-out patient. `src/vena/model/fm/inference/` has the Euler sampler but no top-level CLI that (a) loads a checkpoint, (b) iterates a held-out patient list, (c) decodes to image space, (d) writes NIfTI + a comparison figure.

**This is essentially the exhaustive_val engine** but operating on a different data path (test split or external Málaga cohort, not the cv val split) and writing NIfTI volumes instead of latent_preds.h5.

**Files**

- `routines/fm/inference/cli.py` (new) — `python -m routines.fm.inference.cli <yaml>`.
- `routines/fm/inference/engine/inference_engine.py` (new) — copies the orchestration from `routines/fm/exhaustive_val/engine.py:_process_patient`, but at the end writes each `img_pred` as a NIfTI through `nibabel`.
- `routines/fm/inference/configs/{default,smoke}.yaml`.

**Acceptance:** Running on one UCSF-PDGM test patient produces a `.nii.gz` file viewable in ITK-SNAP that visually resembles the real T1c.

**Estimate:** 4-6 h to write + smoke. Do this *before* the long run starts so we can sanity-check the ema_best of the smoke S2 produces sensible NIfTI.

---

### P2.2  Inference SLURM pair

**Why:** Same as P0.1, for the inference routine. Smaller allocation (`--time=0-04:00:00`, `--gres=gpu:1`, 64 GB RAM).

**Files**

- `routines/fm/inference/slurm/{launcher,worker}_fm_inference.sh` (new).

**Acceptance:** `sbatch --test-only` passes.

---

## P3 — Robustness niceties (not blocking but high ROI)  **[done 2026-06-01]**

### [x] P3.1  Trunk-EMA restore on resume — VERIFIED SAFE

**Why:** Current code rule in `.claude/rules/model-coding-standards.md` says explicitly: *"This path is single-shot — not resume-safe (the trunk EMA is built in setup(), after Lightning's checkpoint load): do not rely on resume_from for unfrozen runs without first hardening trunk-EMA restore."* A 7-day Picasso run **will** be preempted at the 4-day mark (P0.1 splits it into a chain). When the second job resumes, the trunk-EMA shadow is rebuilt from the *original* trunk and overwrites half of the training so far.

**Implementation sketch.** Lightning loads the state_dict *after* `setup()`. The trunk EMA shadow is registered as `self.trunk_ema` (a `nn.Module`) so its parameters *are* in `state_dict`. The actual problem is the order: `setup()` calls `WarmupEMA(self.trunk, ...)` which clones the *current* trunk weights into the EMA shadow. Lightning then overrides the shadow with the checkpoint's shadow weights — that should already work? Worth re-verifying with a test:

- Add `tests/model/fm/test_resume.py::test_trunk_ema_restored_on_resume` — train 2 steps, save, reload, verify `module.trunk_ema.ema_model` state_dict equals the saved one (within fp16 noise).

If the test fails, the fix is to add a `__setstate__` / `on_load_checkpoint` hook that re-clones the live trunk and *then* applies the saved shadow.

**Files**

- `src/vena/model/fm/lightning/module.py` — possible `on_load_checkpoint` extension.
- `tests/model/fm/test_resume.py` — new resume test for the unfrozen-trunk path.

**Acceptance:** the new test passes; a manual 2-job SLURM chain shows the cfm at step (chain_start+1) is continuous with step chain_start (no regression in loss).

---

### [x] P3.2  `vena-fm-watch` — tail Picasso logs from local

**Why:** During a 7-day run, the dev workflow benefits from one command that does `sshfs picasso:/...experiments/<run>/ scratch/picasso-mount/ && tail -f scratch/picasso-mount/logs/train.log`. The encode routine has a similar pattern; mirror it.

**Files**

- `scripts/vena-fm-watch.sh` (new) — single bash script that takes a run id, sshfs-mounts the dir, and tails the log.

**Acceptance:** `bash scripts/vena-fm-watch.sh 2026-06-15_xx_s1_yyyy` prints rolling log lines.

---

### [x] P3.3  DECISIONS.md — pin the v0.3 contrastive choice + smoke results

**Why:** `CLAUDE.md` already states *"DECISIONS.md … will be created the first time a non-trivial architectural decision is made"*. The Lp-aware contrastive merger (v0.2 factorised → v0.3 unified) IS that decision. Write it once now, while the rationale is fresh.

**Files**

- `DECISIONS.md` (new) — one entry per the format in `CLAUDE.md`:
  - Date: 2026-06-01
  - Decision: S2 = CFM + λ_contrast·(λ_roi·ROI^{p_t} + λ_bg·BG^{p_b}); p_t=1, p_b=3, λ_contrast=0.01 (annealed at step half once P1.1 lands).
  - Rationale: see `.claude/notes/foundations/proposal_contrastive_loss.md` §5–§6 and `scratch/2026-06-01_s2_smoke_results.md` (Δ_WT/Δ_BG = 15.5×; cfm trajectory byte-equal to S1; no NaN).
  - Reversibility: setting `loss.contrastive.weight: 0` recovers S1 exactly (tested in `test_lambda_contrast_zero_recovers_s1_total`).

---

## What is NOT on this list (and why)

- **Final test-set evaluation, reader study, external Málaga cohort.** Out of scope per the brief.
- **FID-3D gate from S1 → S2.** Proposal calls for it; the gate value is unknown for the MR conditional baseline; in practice we'll launch S1 → S2 on a fixed step budget and audit the resulting numbers post-hoc. Adding the gate now would require building the MR conditional FID-3D reference baseline first (≥1 day of compute), which we can defer until after the first long run.
- **S3 capped-Lp velocity-reconstruction term.** Proposal flags S3 as a separate ablation row. Stub stays in place; S2 v0.3 is the headline.
- **Brain-mask channel in `m_bg`.** Resolved during planning (option *Pure proposal*); revisit only if the per-voxel BG cap-hit fraction stays at 0% across the whole long run *and* the BG term contribution dominates the contrastive.

---

## Recommended order

1. P0.5, P0.4 in parallel (data audit + preflight artifact rsync).
2. P0.1 + P0.6 in parallel (SLURM pair + env confirmation).
3. P0.2 (write the two Picasso YAMLs).
4. P0.3 (1-epoch throughput smoke on Picasso; set `max_epochs` in the YAMLs).
5. P1.1, P1.2, P1.3, P1.4 — diagnostic completeness; do these *before* the 4-day S1 because they affect what we get back from it.
6. P2.1, P2.2 — inference pipeline. Smoke against the local S2 checkpoint (run id `2026-06-01_18-03-38_s2_5f431b98`).
7. P3.1 — resume safety test. Either confirm it just works or patch it. Must pass before the second job in the chain matters.
8. P3.3 — DECISIONS.md entry. Pure documentation, do whenever.
9. P3.2 — vena-fm-watch. Quality-of-life; do whenever.
10. Launch.
