# 3D-Latent-Pix2Pix competitor integration — VENA validation note

3D-Latent-Pix2Pix (Isola *et al.* 2017 conditional-GAN recipe + Eidex *et al.*
2025 §4 baseline) is the **GAN-paradigm** entry in the VENA latent-baseline
competitor matrix (alongside T1C-RFlow flow-matching and 3D-DiT diffusion
transformer). The integration follows the same 7-step recipe as pGAN,
ResViT, SynDiff, T1C-RFlow, and 3D-DiT.

## Citation

```bibtex
@inproceedings{isola2017image,
  title     = {Image-to-Image Translation with Conditional Adversarial Networks},
  author    = {Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle = {CVPR},
  year      = {2017},
  eprint    = {1611.07004},
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

Per `validation_proposal.md` and `literature.md`, the GAN-baseline slot
isolates the **training paradigm** (BCE+L1 adversarial vs flow-matching)
against the rest of the latent-baseline matrix. Pinning the **only**
training-paradigm delta against T1C-RFlow / VENA to GAN-vs-FM (everything
else — backbone, conditioning route, data path, MAISI-V2 latents —
held identical) means a VENA-vs-Pix2Pix gap isolates the choice of
training paradigm cleanly. Backbone (paper-faithful MAISI 3-level U-Net)
matches the T1C-RFlow wrapper byte-for-byte; the only difference between
T1C-RFlow and this competitor is the training paradigm (adversarial GAN
+ L1 instead of rectified-flow + L1-velocity).

## License flag

The upstream (Eidex *et al.* 2025 repo) has **no LICENSE file** at HEAD
`fc8314f6`. Same flag as T1C-RFlow / DiT-3D — vendored under assumed
academic-use intent of the arXiv preprint. The Isola *et al.* 2017
Pix2Pix architecture is in the public domain (no copyright claim on the
PatchGAN definition); only the specific code lift from the Eidex repo is
governed by the upstream's all-rights-reserved default. See
`src/external/lpix2pix_3d/UPSTREAM.md`.

## Scope contract (the no-augmentation rule)

The wrapper reads `cohort.latent_h5` only (never `cohort.latent_aug_h5`); the
dataset is **deterministic** by contract, pinned by
`tests/competitors/lpix2pix_3d/test_dataset.py::test_dataset_is_deterministic`.
VENA owns the augmentation regime; the competitor does not.

## Code layout

```
src/external/lpix2pix_3d/
├── upstream/                          # vendored snapshot at SHA fc8314f6
│   ├── train_pix2pix_t1n_t2f.py       # GAN-refactor reference trainer
│   └── test_pix2pix_t1n_t2f.py        # reference inference (not invoked)
├── UPSTREAM.md                        # repo URL, SHA, scope, paper-vs-code table
├── UPSTREAM_SHA.txt                   # `fc8314f60d877f9ee55996f960f89b17b269200f`
└── PATCHES.md                         # no patches — vendored byte-identical

src/vena/competitors/lpix2pix_3d/
├── __init__.py                        # re-exports public API
├── dataset.py                         # Pix2PixLatentDataset, MultiCohort
├── runner.py                          # train_lpix2pix_3d — G+D + BCE+L1 loss
└── inference.py                       # run_inference — single G forward + MAISI decode

routines/competitors/lpix2pix_3d/
├── __init__.py
├── cli.py                             # vena-competitor-lpix2pix-3d
├── infer_cli.py                       # vena-competitor-lpix2pix-3d-infer
├── engine.py                          # Pydantic config, decision.json (schema 1.0)
├── configs/
│   ├── smoke_server3_4ep.yaml         # kept on disk; user opted to skip server-3
│   ├── smoke_loginexa_2ep.yaml
│   └── picasso_full.yaml
├── loginexa/launcher_lpix2pix_3d_loginexa_2ep.sh
└── slurm/runs/
    ├── launcher_lpix2pix_3d_picasso_full.sh
    └── worker_lpix2pix_3d_picasso_full.sh

tests/competitors/lpix2pix_3d/
├── test_dataset.py                    # 10 tests
├── test_inference.py                  # 6 tests
└── test_multicohort.py                # 7 tests  → 23 tests total, all PASSED
```

## Patches to upstream

None. The vendored files are byte-identical to upstream SHA `fc8314f6`
and are kept on disk for reference only — the wrapper at
`vena.competitors.lpix2pix_3d` re-implements the two short classes
(`_GeneratorUNetWrapper`, `_PatchDiscriminator3D`, ~50 LOC combined)
against MONAI primitives directly so the vendored script's
`argparse`/`tqdm`/`matplotlib`/`importlib` plumbing never enters VENA's
import graph. See `src/external/lpix2pix_3d/PATCHES.md`.

## Data contract

- Reads VENA's latent H5 (schema 2.0.0): `latents/{t1pre,flair,t1c}` shaped
  `(N, 4, 48, 56, 48)`.
- Many-to-one: target `z_T1c`, conditioning `z_T1pre + z_FLAIR`.
  - **Generator** input shape: `(B, 8, 48, 56, 48)` — `[z_T1pre, z_FLAIR]`
    concat along channel axis. **No noisy target injection** — Pix2Pix is
    non-diffusive; one forward pass maps conditioning to prediction.
  - **Discriminator** input shape: `(B, 12, 48, 56, 48)` — `[cond_8,
    target_or_fake_4]` concat along channel axis.
- The MAISI U-Net is shape-agnostic (unlike DiT-3D); the runner therefore
  does **not** enforce a single latent grid across cohorts. VENA's schema
  2.0.0 trunk-÷8 constraint guarantees uniform latent shapes in practice.

### Cohort-schema fallbacks (carried over from pGAN / T1C-RFlow / DiT-3D)

1. **Longitudinal patient-id resolver** for BraTS-GLI / LUMIERE (scan-level
   `/ids`, patient-level splits). Prefix match recovers both scans per
   patient. Pinned by `test_dataset_resolves_longitudinal_patient_ids`.
2. **Flat-splits fallback** for REMBRANDT (`splits/{train,val,test}` without
   k-fold). Pinned by `test_dataset_falls_back_to_flat_splits_schema`.
3. **Missing-cohort skip-with-WARNING** at the multi-cohort layer. Pinned by
   `test_multicohort_skips_missing_h5_with_warning`.

## Training contract

Wrapper matches Isola *et al.* 2017 §3.2 (loss form, λ_L1) and the
vendored `train_pix2pix_t1n_t2f.py` (optimiser kwargs, AMP, conditioning
route) on every load-bearing axis:

| Axis | Value | Source |
|---|---|---|
| Generator backbone | `DiffusionModelUNetMaisi` at paper-faithful 3-level config (`num_channels=[128, 128, 256]`, `num_res_blocks=2`, no self-attention) | Same as `vena.competitors.t1c_rflow.runner._build_unet` — symmetric with T1C-RFlow |
| Generator parameter count | **49.62 M** (measured on smoke run, 2 cond → in=8 / out=4) | runner first-build log |
| Generator wrapper | `_GeneratorUNetWrapper(unet)` — feeds `t = zeros((B,), dtype=long)` so the diffusion U-Net runs deterministically | Vendored `train_pix2pix_t1n_t2f.py:98-106` |
| Discriminator | `_PatchDiscriminator3D(in=12, ndf=64, num_layers=4)` — 4 strided Conv3d + InstanceNorm + LeakyReLU(0.2) layers + terminal 1-ch head | Vendored `train_pix2pix_t1n_t2f.py:108-132` |
| Discriminator parameter count | **11.09 M** (measured on smoke run) | runner first-build log |
| Loss | `BCEWithLogitsLoss` adversarial + `lambda_l1 * F.l1_loss(fake, real)` with `lambda_l1 = 100` | Isola 2017 §3.2 |
| Generator total loss | `g_total = g_adv + lambda_l1 * g_l1` | Isola Eq. 4 |
| Discriminator loss | `d_loss = 0.5 * (d_real + d_fake)` (each BCE against ones/zeros) | Isola §3.2 |
| Conditioning route | channel concat `[z_T1pre, z_FLAIR]` (G); `[cond, target_or_fake]` (D) | Vendored `train_pix2pix_t1n_t2f.py:226, 233-234` |
| Optimiser | `AdamW(lr_G=lr_D=1e-4, betas=(0.5, 0.999), weight_decay=1e-4)` for both G and D | Vendored `train_pix2pix_t1n_t2f.py:195-196` |
| Mixed precision | `torch.amp.GradScaler` + `autocast(fp16)` for both G and D updates | Vendored `train_pix2pix_t1n_t2f.py:197, 230, 248` |
| EMA / grad clipping / augmentation | none | paper-faithful |
| Checkpoints | `best_net_pix2pix.pth` (min epoch-mean `g_l1`), `latest_net_pix2pix.pth`, `epoch_<N>_net_pix2pix.pth` every `save_epoch_freq`. **Each contains both `G_state_dict` and `D_state_dict`** so a reviewer can audit D weights post-hoc. | wrapper-added — symmetric with T1C-RFlow / DiT-3D checkpoint shape |
| Model selection metric | epoch-mean `loss_g_l1` (NOT `g_total`) — adversarial term oscillates against D; L1 component is the stable reconstruction signal | Isola §6 reports L1 as the primary metric |
| Patience | epoch-mean `g_l1`; default 100 for production, 0 for smokes / paper-budget schedule | VENA paired-axis |

### Architecture recovery at inference time

The wrapper persists the architecture kwargs (`latent_channels`,
`cond_latents`, `disc_ndf`, `disc_num_layers`) as an `arch_meta` block
**inside every saved checkpoint** so `run_inference` can rebuild G (and
optionally D) without consulting the training YAML. Pinned by
`test_inference.py::test_rebuild_generator_from_meta_matches_fresh_build`
and the sibling discriminator test. Same pattern as 3D-DiT (skill §3.6.1).

## VAE for inference decode

VENA's MAISI-V2 (`autoencoder_v2.pt`) via `vena.common.load_autoencoder`.
The path is required at `vena-competitor-lpix2pix-3d-infer
--vae-checkpoint ...` time and documented in `src/external/LINKS.md` per
platform.

## Per-platform recipe and validation

### Server-3 (RTX 4090, no SLURM) — **skipped 2026-06-17**

The user opted to skip server-3 testing for this competitor and go
directly to loginexa (note added to chat 2026-06-17). The
`smoke_server3_4ep.yaml` config + the (unwritten) launcher are kept on
disk for reproducibility.

### Loginexa (Picasso V100-DGXS-32GB interactive node) — **smoke green 2026-06-17 11:00**

- Conda env `vena-v100` (torch 2.7.1+cu126); no extra deps beyond what
  the env already carries (no `timm`, no perceptual loss).
- Auto-picked freest V100 by `nvidia-smi memory.free` → GPU 3.
- Smoke result: 6 cohorts loaded (UCSF-PDGM + BraTS-GLI + UPENN-GBM +
  IvyGAP + LUMIERE + REMBRANDT), 1 patient each. G_L1 descended
  0.7807 → 0.7681 across 2 epochs at ~1.7-3 s/epoch (batch=1, AMP).
  All checkpoints (`best`, `latest`, `epoch_{0..1}`) + step/epoch CSVs +
  `decision.json` (`schema_version=1.0`, `completed=true`,
  `platform=loginexa`) written. Run id:
  `2026-06-17T09-00-10_competitor_lpix2pix_3d_smoke_2ep_multicohort_a1dd749`.
- Sentinel `lpix2pix-3d-train completed` emitted; tmux torn down.

### Picasso (A100 DGX, full training) — **submitted 2026-06-17, job 1113481**

- Conda env `vena` (torch 2.12+cu130); no extra deps.
- Worker `slurm/runs/worker_lpix2pix_3d_picasso_full.sh` requests
  `--constraint=dgx --partition=gpu_partition --gres=gpu:1
  --time=3-00:00:00 --mem=64G --cpus-per-task=8`. `experiments_root`
  uses lowercase `…/execs/vena/…` (skill §4 note).
- Same `corpus_picasso.json` as VENA / T1C-RFlow / DiT-3D (6 cv cohorts).
- Initial state at submission: pending (Priority queue).

## Inference contract

```
vena-competitor-lpix2pix-3d-infer \
    --run-dir   <run_dir> \
    --image-h5  <UCSFPDGM_image.h5> \
    --latent-h5 <UCSFPDGM_latents.h5> \
    --vae-checkpoint /abs/path/to/autoencoder_v2.pt \
    --epoch best --n-patients 10 --phase val
```

Writes 10 NIfTI volumes + PNG midslices + `metrics.csv` (PSNR / SSIM /
gen + decode seconds per patient — single G forward pass, no NFE panel)
+ `summary.json`.

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
| training paradigm | | | ⚠ **conditional GAN (BCE + λ·L1)** vs VENA flow-matching |
| backbone | | | ✅ same paper-faithful MAISI 3-level U-Net as T1C-RFlow |
| optimiser | | | ⚠ AdamW(lr=1e-4, β1=0.5) (Isola §3.2 / vendored code) vs VENA AdamW(lr=5e-5, β1=0.9) |
| ControlNet / mask conditioning | | | ⚠ absent — Pix2Pix has no mask conditioning |
| target metric | | ⚠ best on epoch-mean `g_l1` (L1 component, not BCE+L1 total) | |

## Paper-budget rationale (skill §3.11)

The Eidex 2025 paper trained ALL baselines (DDPM, Pix2pix, DiT-3D, T1C-RFlow)
on the same BraTS-2024 budget: 2,860 train samples × 100 epochs = 286,000
sample exposures. VENA's multi-cohort fold-0 train union (probed
2026-06-15 on Picasso) ≈ 1,751 latent volumes/epoch. The fairness contract:

```
max_epochs = ceil(286,000 / 1,751) ≈ 164
```

The picasso `max_epochs: 164, patience: 0, batch: 4` cap is the same one
T1C-RFlow and 3D-DiT use — keeps all three latent-baseline competitors
comparable head-to-head and defensible against the
"undertrained-competitor" charge.

## Things to watch / things that nearly bit me

1. **Vendoring the same upstream a third time.** The Eidex 2025 repo
   ships three baselines (RFlow U-Net, DiT-3D, Pix2Pix). T1C-RFlow lives
   at `src/external/t1c_rflow/upstream/` and DiT-3D at
   `src/external/dit_3d/upstream/`. Skill anti-pattern 7 forbids
   cross-importing from a sibling competitor's vendored snapshot, so the
   right move is a third independent copy at
   `src/external/lpix2pix_3d/upstream/`. Disk cost ~25 KB (2 .py files
   for this competitor); independence cost 0.
2. **Generator vs scheduler-driven training shape.** Pix2Pix is
   non-diffusive — the generator input is **`(B, 8, ...)`**
   (`[z_T1pre, z_FLAIR]` only), NOT `(B, 12, ...)` with a noisy target
   prepended like T1C-RFlow and DiT-3D. Got this right on the first
   shot because the vendored upstream is explicit
   (`train_pix2pix_t1n_t2f.py:182,226`).
3. **MAISI U-Net's `num_head_channels` constraint.** Even when
   `attention_levels=[False, False, False]`, MAISI's `SpatialAttention`
   modules are always constructed and need `num_head_channels[i]` to be
   a positive divisor of `num_channels[i]`. Reused the T1C-RFlow
   wrapper's pattern (`num_head_channels=[32, 32, 32]`) — 32 divides
   both 128 and 256 cleanly. The blocks remain inactive.
4. **Two AMP scalers, two optimisers.** Pix2Pix has separate G and D
   updates with separate gradient scalers — naively sharing one scaler
   makes the D step skip when G overflows and vice versa. Two
   `GradScaler` instances (`scalerG`, `scalerD`) is the canonical pattern
   in `train_pix2pix_t1n_t2f.py:197`.
5. **Model selection metric.** The total G loss is dominated by the BCE
   term (oscillates against D), so `best` selection on `g_total` would
   thrash. Picked `g_l1` (the stable L1 reconstruction component) per
   Isola §6 — same metric the original Pix2Pix paper reports.
6. **Saving both G and D in every checkpoint.** A reviewer auditing the
   GAN dynamics needs the D weights too — saving `G_state_dict` and
   `D_state_dict` in one blob per checkpoint costs ~10 MB extra but
   removes any ambiguity. The inference path only loads `G_state_dict`.
7. **`generate_run_id` length.** The run id is now `2026-06-17T09-00-10_competitor_lpix2pix_3d_smoke_2ep_multicohort_a1dd749`
   — 14 chars longer than DiT-3D's because the competitor name is
   12-char vs 6-char. No paths are truncated under VENA's filesystem;
   noting because future competitors with longer names may need a
   `tag` shortening pass.
