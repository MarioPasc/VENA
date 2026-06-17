# 3D-LDDPM competitor integration — VENA validation note

3D-LDDPM (Ho *et al.* 2020 DDPM scheduler + Eidex *et al.* 2025 §4 baseline
recipe) is the **latent DDPM** entry in the VENA competitor matrix
(equivalent to the C3 / "DDPM" row reported in Eidex 2025's baseline table).
The integration follows the same 7-step recipe as pGAN, ResViT, SynDiff,
T1C-RFlow, and DiT-3D.

## Citation

```bibtex
@inproceedings{ho2020denoising,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
  eprint    = {2006.11239},
  archivePrefix = {arXiv}
}

@article{eidex2025efficient3dlatentdiffusion,
  title   = {An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation},
  author  = {Eidex, Zach and Safari, Mojtaba and Ding, Jie and Qiu, Richard
             and Roper, Justin and Yu, David and Shu, Hui-Kuo and Tian, Zhen
             and Mao, Hui and Yang, Xiaofeng},
  journal = {arXiv preprint arXiv:2509.24194},
  year    = {2025}
}
```

## Why the design fits this slot

Per `validation_proposal.md` §4 and `literature.md` §11b, the "DDPM" baseline
slot is the **scheduler-isolated** delta against T1C-RFlow / VENA-S2:
T1C-RFlow's headline method is RFlow on the MAISI latent space; Eidex 2025
§4 cites latent DDPM (Ho *et al.* 2020) as the *baseline diffusion* row that
their RFlow method improves upon. Pinning the **only** axis vs T1C-RFlow to
the scheduler (DDPM vs RFlow) + loss (MSE-ε vs L1-velocity) — same MAISI U-Net
backbone, same MAISI-V2 VAE, same conditioning, same optimiser — means a
T1C-RFlow vs 3D-LDDPM gap isolates the rectified-flow improvement cleanly.

| Comparison | Backbone | Scheduler | Loss |
|---|---|---|---|
| VENA-S1 | MAISI U-Net (NV-Generate-MR) | RFlow | L1 velocity + ControlNet conditioning |
| T1C-RFlow | MAISI U-Net (paper-faithful) | RFlow | L1 velocity |
| **3D-LDDPM** | **MAISI U-Net (paper-faithful)** | **DDPM** | **MSE epsilon** |
| DiT-3D | DiT-B/4 transformer | RFlow | L1 velocity |

3D-LDDPM and T1C-RFlow share the U-Net backbone byte-for-byte
(`vena.competitors.lddpm_3d.runner._build_unet` and
`vena.competitors.t1c_rflow.runner._build_unet` use the same MAISI
`DiffusionModelUNetMaisi` kwargs).

## License flag

The upstream (Eidex et al. 2025 repo, SHA `fc8314f6`) has **no LICENSE file**.
Same flag as T1C-RFlow and DiT-3D — vendored under assumed academic-use
intent of the arXiv preprint. See `src/external/lddpm_3d/UPSTREAM.md`.

## Scope contract (the no-augmentation rule)

The wrapper reads `cohort.latent_h5` only (never `cohort.latent_aug_h5`); the
dataset is **deterministic** by contract, pinned by the per-cohort
`LDDPM3DLatentDataset.__getitem__` returning the same tensor for the same
index (the carry-over from `T1CRFlowLatentDataset`). VENA owns the
augmentation regime; the competitor does not.

## Code layout

```
src/external/lddpm_3d/
├── upstream/                      # vendored snapshot at SHA fc8314f6
│   ├── train_ddpm.py              # paper-baseline DDPM trainer (reference)
│   └── test_ddpm_t1_flair_final.py  # paper-baseline DDPM inference (reference)
├── UPSTREAM.md                    # repo URL, SHA, scope, paper-vs-code table
├── UPSTREAM_SHA.txt               # `fc8314f60d877f9ee55996f960f89b17b269200f`
└── PATCHES.md                     # none — vendored verbatim

src/vena/competitors/lddpm_3d/
├── __init__.py                    # re-exports public API
├── dataset.py                     # LDDPM3DLatentDataset + MultiCohortLDDPM3DLatentDataset
├── runner.py                      # train_lddpm_3d — DDPM scheduler + MSE-eps loss
└── inference.py                   # run_inference — K-step DDPM denoising + MAISI decode

routines/competitors/lddpm_3d/
├── __init__.py
├── cli.py                         # vena-competitor-lddpm-3d
├── infer_cli.py                   # vena-competitor-lddpm-3d-infer
├── engine.py                      # Pydantic config, decision.json (schema 1.0)
├── configs/
│   ├── smoke_server3_4ep.yaml
│   ├── smoke_loginexa_2ep.yaml
│   └── picasso_full.yaml
├── server3/launcher_lddpm_3d_server3_4ep.sh
├── loginexa/launcher_lddpm_3d_loginexa_2ep.sh
└── slurm/runs/
    ├── launcher_lddpm_3d_picasso_full.sh
    └── worker_lddpm_3d_picasso_full.sh
```

## Patches to upstream

**None.** Vendored verbatim; the wrapper does not import from the snapshot —
the two upstream files are kept as a reference for future readers. The
wrapper rebuilds the model from MONAI primitives directly (mirroring
T1C-RFlow's policy in `src/external/t1c_rflow/PATCHES.md`). See
`src/external/lddpm_3d/PATCHES.md`.

## Data contract

- Reads VENA's latent H5 (schema 2.0.0): `latents/{t1pre,flair,t1c}` shaped
  `(N, 4, 48, 56, 48)`.
- Many-to-one: target `z_T1c`, conditioning `z_T1pre + z_FLAIR` concatenated
  along channel axis at the runner step → `(B, 12, 48, 56, 48)` fed to the
  U-Net via `DiffusionInferer(mode="concat")`.
- All cohorts in the production corpus must share latent shape
  `(4, 48, 56, 48)` (VENA schema 2.0.0 trunk-÷8 constraint). The MAISI U-Net
  is fully-convolutional so this is not a hard architectural requirement
  (unlike DiT-3D's fixed positional embeddings), but VENA's encoder pins
  the grid.

### Cohort-schema fallbacks (carried over from T1C-RFlow / DiT-3D / pGAN)

1. **Longitudinal patient-id resolver** for BraTS-GLI / LUMIERE (scan-level
   `/ids`, patient-level splits). Prefix match recovers both scans per
   patient.
2. **Flat-splits fallback** for REMBRANDT (`splits/{train,val,test}` without
   k-fold).
3. **Missing-cohort skip-with-WARNING** at the multi-cohort layer.

## Training contract

Wrapper matches Eidex 2025 §4 / Ho *et al.* 2020 on every load-bearing axis:

| Axis | Value | Source |
|---|---|---|
| Backbone | **MAISI U-Net (paper-faithful)** — `spatial_dims=3, num_channels=[128, 128, 256], attention_levels=[False, False, False], num_res_blocks=2, num_head_channels=[32, 32, 32], resblock_updown=True, in_channels=12` | Eidex 2025 §3 verbatim; identical to T1C-RFlow wrapper for backbone-symmetric comparison |
| Parameter count | **49.63 M** (measured at build time) | `src/vena/competitors/lddpm_3d/runner.py::_build_unet`; matches T1C-RFlow's 49.6M |
| Scheduler | `monai.networks.schedulers.DDPMScheduler` direct, kwargs `num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195, schedule="scaled_linear_beta", clip_sample=False` | upstream `train_ddpm.py:113-119` verbatim |
| Inferer | `monai.inferers.DiffusionInferer(scheduler)` with `mode="concat"` | upstream `train_ddpm.py:120, 162-169` |
| Forward target | predict `ε` given `(x_noisy = sqrt(α̅_t)·z_T1c + sqrt(1-α̅_t)·ε, t)` | Ho *et al.* 2020 Eq. 11 / DDPMScheduler.add_noise |
| Loss | `F.mse_loss(eps_pred, eps)` — **MSE**, not L1 | Ho *et al.* 2020 Eq. 14; upstream `train_ddpm.py:170` |
| Optimiser | `AdamW(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)` | Eidex 2025 §4 — same as T1C-RFlow / DiT-3D wrappers |
| Conditioning | channel concat `[noisy_z_T1c, z_T1pre, z_FLAIR]` via `DiffusionInferer(mode="concat")` | upstream `train_ddpm.py:162-169`; symmetric with T1C-RFlow |
| Mixed precision | `torch.amp.GradScaler` + `autocast(fp16)` enabled by default | upstream `train_ddpm.py:122,161-173` |
| EMA / grad clipping / augmentation | none | paper-faithful (Ho *et al.* 2020 did not use EMA for the eps-prediction net; Eidex 2025 baseline mirrors that) |
| Checkpoints | `best_net_unet.pth` (min epoch-mean train loss), `latest_net_unet.pth`, `epoch_<N>_net_unet.pth` every `save_epoch_freq` | wrapper-added — exhaustive-val is asynchronous |
| Patience | epoch-mean train loss; default 0 for paper-budget production, 0 for smokes (deliberately fixed schedule per skill §3.11) | VENA paired-axis |

## VAE for inference decode

VENA's MAISI-V2 (`autoencoder_v2.pt`) via `vena.common.load_autoencoder`.
The vendored upstream's `autoencoder_epoch273.pt` (Git LFS, ~80 MB) is **not**
loaded — the wrapper consumes pre-encoded MAISI-V2 latents from VENA's H5
cache. Symmetric with T1C-RFlow / DiT-3D — the VAE-choice confound is
neutralised across the three Eidex-2025 competitors.

The path is required at `vena-competitor-lddpm-3d-infer --vae-checkpoint ...`
time and documented in `src/external/LINKS.md` per platform.

## Per-platform recipe and validation

### Server-3 (RTX 4090, no SLURM)

- Conda env `vena` (torch 2.5.1+cu121). No new dependencies vs T1C-RFlow.
- `server3/launcher_lddpm_3d_server3_4ep.sh`: rsync repo → ssh icai-server →
  `screen -dmS vena-lddpm-3d-smoke …` → exit. **rsync excludes** include
  `src/external/t1c_rflow/upstream/checkpoints/` (T1C-RFlow's 80 MB LFS
  VAE — not needed for any competitor; we use MAISI-V2).
- Expected smoke result: 6 cohorts loaded (UCSF-PDGM + BraTS-GLI + UPENN-GBM
  + IvyGAP + LUMIERE + REMBRANDT), 1 patient each. Loss should descend
  from ≈ 1.0 (random eps prediction at init) across 4 epochs.
- See section "Smoke results — server-3" below for the recorded outcome.

### Loginexa (Picasso V100-DGXS-32GB interactive node)

- Conda env `vena-v100` (torch 2.7.1+cu126). No new dependencies vs T1C-RFlow.
- Auto-picks freest V100 by `nvidia-smi memory.free`.
- See section "Smoke results — loginexa" below.

### Picasso (A100 DGX, full training)

- Conda env `vena` (torch 2.12+cu130). No new dependencies vs T1C-RFlow.
- Worker `slurm/runs/worker_lddpm_3d_picasso_full.sh` requests
  `--constraint=dgx --partition=gpu_partition --gres=gpu:1
  --time=3-00:00:00 --mem=64G --cpus-per-task=8`. `experiments_root` uses
  lowercase `…/execs/vena/…` (skill §4 note).
- Same `corpus_picasso.json` as VENA / T1C-RFlow / DiT-3D (6 cv cohorts).

## Inference contract

```
vena-competitor-lddpm-3d-infer \
    --run-dir   <run_dir> \
    --image-h5  <UCSFPDGM_image.h5> \
    --latent-h5 <UCSFPDGM_latents.h5> \
    --vae-checkpoint /abs/path/to/autoencoder_v2.pt \
    --epoch best --n-patients 10 --phase val \
    --nfe 200 --nfe 500 --nfe 1000
```

DDPM uses **more NFE** than RFlow because its time-discretisation is
non-adaptive (Ho *et al.* 2020 defaults to K=T=1000; reducing K
degrades quality monotonically). The CLI exposes `(200, 500, 1000)` as the
default NFE panel — 1000 matches the upstream `test_ddpm_t1_flair_final.py`
default; 200 is the practical floor before quality collapses.

Writes 10 NIfTI volumes + PNG midslices + `metrics.csv` (PSNR / SSIM /
gen+decode seconds per patient × NFE) + `summary.json`.

## Paired-comparison axes (vs `picasso_s1_1000ep_fft.yaml`)

| Axis | Match exactly | Match conceptually | Differ by design |
|---|---|---|---|
| seed | ✅ (1337) | | |
| fold | ✅ (0) | | |
| corpus_registry | ✅ (`corpus_picasso.json`) | | |
| batch_size | ✅ (4) | | |
| num_workers | ✅ (8) | | |
| save cadence | ✅ (every 25 epochs) | | |
| `max_epochs` | | | ⚠ **164** (Eidex 2025 paper-budget cap; skill §3.11) — VENA runs ~1000 |
| patience | | | ⚠ 0 (deliberately fixed schedule per paper budget) |
| backbone | | | ⚠ paper-faithful MAISI U-Net (same as T1C-RFlow; vs VENA's NV-Generate-MR trunk) |
| trunk fine-tuning | | | ⚠ none — randomly-initialised; VENA warm-starts NV-Generate-MR |
| ControlNet / mask conditioning | | | ⚠ absent — Eidex 2025 §4 LDDPM baseline has no mask conditioning |
| scheduler | | | ⚠ **DDPM** (vs VENA / T1C-RFlow's RFlow) |
| loss | | | ⚠ **MSE on ε** (vs VENA / T1C-RFlow's L1 on velocity) |
| target metric | | ⚠ best on epoch-mean train loss | |

## Paper-budget rationale (skill §3.11)

Same calculation as T1C-RFlow / DiT-3D — Eidex 2025 baselines were all
trained on identical budgets:

```
paper_train_samples  = 2,860 (BraTS-2024)
paper_epochs         = 100
paper_total_exposure = 286,000
our_latents/epoch    ≈ 1,751 (multi-cohort fold-0 train union, probed 2026-06-15)
max_epochs           = ceil(286,000 / 1,751) ≈ 164
```

`max_epochs: 164, patience: 0, batch: 4` is the same cap T1C-RFlow and
DiT-3D use — keeps the three Eidex-2025 baselines on identical budgets so
their head-to-head numbers are comparable.

## Things to watch / things that nearly bit me

1. **Vendoring the same upstream three times.** T1C-RFlow at
   `src/external/t1c_rflow/upstream/`, DiT-3D at
   `src/external/dit_3d/upstream/`, and now LDDPM at
   `src/external/lddpm_3d/upstream/` — all at SHA `fc8314f6`. Skill
   anti-pattern 7 forbids cross-imports; the duplication is intentional
   (each snapshot vendors only the files it actually needs as reference).
2. **No new dependencies.** Unlike DiT-3D's `timm`, LDDPM uses only MONAI
   primitives already available in VENA's three envs (`vena`, `vena-v100`,
   Picasso's `vena`). The integration adds zero env friction.
3. **DDPM `beta_start` discrepancy in upstream.** `train_ddpm.py:115` uses
   `0.0015`; `test_ddpm_t1_flair_final.py:125` uses `0.0005`. This looks
   like an upstream typo — a sampler that does not match the training
   schedule produces a different noise profile than the model was trained
   under. The wrapper uses `0.0015` (training-time value) at inference;
   `--beta-start 0.0005` is an ablation knob (see `UPSTREAM.md` table
   row 3 and `PATCHES.md` C1).
4. **`DiffusionInferer.mode="concat"` vs concat-by-hand.** Upstream uses
   `DiffusionInferer(mode="concat")`; T1C-RFlow wrapper does the
   `torch.cat([noisy, cond], dim=1)` by hand. The two are byte-equivalent
   on the forward pass (DiffusionInferer just concatenates and calls
   `diffusion_model(model_in, timesteps)` internally). I went with the
   upstream's API to keep the LDDPM wrapper close to `train_ddpm.py:162-169`
   verbatim — easier audit.
5. **Backbone-symmetric with T1C-RFlow.** The wrapper's `_build_unet` is a
   verbatim copy of `vena.competitors.t1c_rflow.runner._build_unet` (same
   MAISI kwargs). This is a deliberate cross-competitor parity choice:
   when the T1C-RFlow vs LDDPM table row appears in the paper, the only
   delta the reader can attribute the gap to is the scheduler/loss.
6. **NFE panel asymmetry.** DiT-3D and T1C-RFlow infer at NFE
   `(50, 100, 200)`. LDDPM infers at `(200, 500, 1000)`. This is a
   deliberate asymmetry — DDPM at NFE=50 is essentially random; the
   meaningful regime starts at K≥200 (Ho *et al.* 2020 Fig. 3). Keeping
   them on the same panel would either inflate DDPM's NFE budget for the
   other two or degrade DDPM unfairly.

## Smoke results — server-3 (2026-06-17 — green)

Run id: `2026-06-17T08-46-38_competitor_lddpm_3d_smoke_4ep_multicohort_fc68720`

- Device: RTX 4090, `vena` env (torch 2.5.1+cu121).
- All 6 cohorts loaded (UCSF-PDGM + BraTS-GLI + UPENN-GBM + IvyGAP + LUMIERE
  + REMBRANDT), 1 patient each → 6 total train patients.
- U-Net parameter count: **49.63 M** (matches T1C-RFlow wrapper byte-for-byte).
- Loss trajectory (epoch-mean ε-MSE):
  `0.9971 → 0.9909 → 0.9802 → 0.9807` (descending; sane DDPM init noise).
- Wall-clock: ~1 s/epoch at batch=1 on AMP — entire 4-epoch smoke in ≈ 22 s.
- Artifacts present: `best_net_unet.pth` + `latest_net_unet.pth` +
  `epoch_{0..3}_net_unet.pth` (6 checkpoints × ~190 MB each), step + epoch
  CSVs, `decision.json` (`completed=true`, `schema_version=1.0`,
  `upstream_sha=fc8314f60d87…`).

## Smoke results — loginexa (2026-06-17 — green)

Run id: `2026-06-17T09-03-07_competitor_lddpm_3d_smoke_2ep_multicohort_a1dd749`

- Device: V100-DGXS-32GB (GPU 3, auto-picked by `nvidia-smi memory.free`),
  `vena-v100` env (torch 2.7.1+cu126).
- Same 6 cohorts × 1 patient as server-3.
- Loss trajectory: `0.9975 → 0.9913` (2 epochs).
- Wall-clock: ~2 s/epoch on V100 (~2× server-3 RTX 4090 step time — expected).
- All artifacts present; `decision.json` `completed=true`, `platform=loginexa`.

## Picasso full training (2026-06-17 — submitted)

- SLURM job id: **1113548** on partition `gpu_partition`,
  `--constraint=dgx --gres=gpu:1 --time=3-00:00:00 --cpus-per-task=8
  --mem=64G`, env `vena` (torch 2.12+cu130).
- `sbatch --test-only` accepted (eligible start `2026-07-17T10:52:53` on
  `exa02` per current queue priority).
- Status: `PD` (priority wait).
- Config: `picasso_full.yaml` — same `corpus_picasso.json` as
  VENA / T1C-RFlow / DiT-3D; `max_epochs=164, patience=0, batch=4,
  fold=0, seed=1337, num_workers=8, save_epoch_freq=25` (paper-budget
  fairness contract — skill §3.11).
- Expected wallclock at full budget: ~6-12 h on A100 (paper-faithful U-Net
  ~49.6 M params; ~71.8 k optimiser steps); `--time=3-00:00:00` is a
  ≥6× overshoot for queue variance.
- Sentinel: `lddpm-3d-train completed` in the SLURM `.out` log.
