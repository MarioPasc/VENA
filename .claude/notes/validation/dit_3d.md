# 3D-DiT competitor integration — VENA validation note

3D-DiT (Peebles & Xie 2023 backbone + Eidex *et al.* 2025 §4 baseline recipe)
is the **transformer-backbone diffusion** entry in the VENA competitor matrix
(C4 row in `validation_proposal.md`). The integration follows the same 7-step
recipe as pGAN, ResViT, SynDiff, and T1C-RFlow.

## Citation

```bibtex
@inproceedings{peebles2023scalable,
  title     = {Scalable Diffusion Models with Transformers},
  author    = {Peebles, William and Xie, Saining},
  booktitle = {ICCV},
  year      = {2023},
  eprint    = {2212.09748},
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

Per `validation_proposal.md` §4 and `literature.md` §11b, the C4 slot is the
"transformer-backbone diffusion baseline that T1C-RFlow themselves cite as
their DiT-3D row." Pinning the **only** architectural delta against
T1C-RFlow / VENA to the backbone (DiT vs U-Net) means a VENA-vs-3D-DiT gap
isolates the backbone choice cleanly. The scheduler (RFlow, Eidex 2025
§3.2), loss (L1 velocity), and data path (MAISI-V2 latents) all match the
T1C-RFlow wrapper.

## License flag

The upstream (Eidex et al. 2025 repo) has **no LICENSE file** at HEAD
`fc8314f6`. Same flag as T1C-RFlow — vendored under assumed academic-use
intent of the arXiv preprint. The `DiT` class itself is adapted from Meta's
2D DiT (Peebles & Xie 2023, license header preserved inside `dit3d.py`),
which is governed by Meta's DiT repo licence (MIT-style). See
`src/external/dit_3d/UPSTREAM.md`.

## Scope contract (the no-augmentation rule)

The wrapper reads `cohort.latent_h5` only (never `cohort.latent_aug_h5`); the
dataset is **deterministic** by contract, pinned by
`tests/competitors/dit_3d/test_dataset.py::test_dataset_is_deterministic`.
VENA owns the augmentation regime; the competitor does not.

## Code layout

```
src/external/dit_3d/
├── upstream/                 # vendored snapshot at SHA fc8314f6
│   ├── dit3d.py              # DiT model definition (Meta-attributed)
│   ├── dit3d_wrapper.py      # 10-line shim: forward(x, t)
│   └── test_dit.py           # reference inference script (not invoked)
├── UPSTREAM.md               # repo URL, SHA, scope, paper-vs-code table
├── UPSTREAM_SHA.txt          # `fc8314f60d877f9ee55996f960f89b17b269200f`
└── PATCHES.md                # P1: remove stdout print in get_3d_sincos_pos_embed

src/vena/competitors/dit_3d/
├── __init__.py               # re-exports public API
├── dataset.py                # DiT3DLatentDataset, MultiCohortDiT3DLatentDataset
├── runner.py                 # train_dit_3d — RFlow + DiT-B/4 + L1 velocity
└── inference.py              # run_inference — Euler integration + MAISI decode

routines/competitors/dit_3d/
├── __init__.py
├── cli.py                    # vena-competitor-dit-3d
├── infer_cli.py              # vena-competitor-dit-3d-infer
├── engine.py                 # Pydantic config, decision.json (schema 1.0)
├── configs/
│   ├── smoke_server3_4ep.yaml
│   ├── smoke_loginexa_2ep.yaml
│   └── picasso_full.yaml
├── server3/launcher_dit_3d_server3_4ep.sh
├── loginexa/launcher_dit_3d_loginexa_2ep.sh
└── slurm/runs/
    ├── launcher_dit_3d_picasso_full.sh
    └── worker_dit_3d_picasso_full.sh

tests/competitors/dit_3d/
├── test_dataset.py           # 10 tests
├── test_inference.py         # 5 tests
└── test_multicohort.py       # 7 tests
```

## Patches to upstream

Only one — cosmetic. See `src/external/dit_3d/PATCHES.md`:
removed the stdout `print('grid_size:', grid_size)` in
`get_3d_sincos_pos_embed` (`dit3d.py:312`). Pure cosmetic; numerics
unchanged.

## Data contract

- Reads VENA's latent H5 (schema 2.0.0): `latents/{t1pre,flair,t1c}` shaped
  `(N, 4, 48, 56, 48)`.
- Many-to-one: target `z_T1c`, conditioning `z_T1pre + z_FLAIR`
  concatenated along channel axis at the runner step → `(B, 12, 48, 56, 48)`
  fed to the DiT.
- All cohorts in the production corpus must share latent shape `(4, 48, 56, 48)`
  (VENA schema 2.0.0 trunk-÷8 constraint). The runner peeks the first
  sample's shape at start and **rejects** any per-step shape mismatch — DiT
  positional embeddings are fixed-size buffers.

### Cohort-schema fallbacks (carried over from pGAN / T1C-RFlow)

1. **Longitudinal patient-id resolver** for BraTS-GLI / LUMIERE (scan-level
   `/ids`, patient-level splits). Prefix match recovers both scans per
   patient. Pinned by `test_dataset_resolves_longitudinal_patient_ids`.
2. **Flat-splits fallback** for REMBRANDT (`splits/{train,val,test}` without
   k-fold). Pinned by `test_dataset_falls_back_to_flat_splits_schema`.
3. **Missing-cohort skip-with-WARNING** at the multi-cohort layer. Pinned by
   `test_multicohort_skips_missing_h5_with_warning`.

## Training contract

Wrapper matches Eidex 2025 §4 / T1C-RFlow training on every non-backbone
axis:

| Axis | Value | Source |
|---|---|---|
| Backbone | DiT-B/4 in 3D (`depth=12`, `hidden=768`, `num_heads=12`, `patch=4`, `mlp_ratio=4.0`) | Peebles & Xie 2023 standard "base" config; patch_size=4 chosen because 48 / 56 / 48 are all cleanly divisible by 4 |
| Parameter count | **131.85 M** (measured on smoke run) | runner first-build log |
| Scheduler | `monai.networks.schedulers.RFlowScheduler` direct, kwargs `num_train_timesteps=1000, use_discrete_timesteps=True, sample_method="logit-normal", use_timestep_transform=True, base_img_size_numel=64*64*48, spatial_dim=3` | T1C-RFlow upstream `train_rflow.py:136-143` — `base_img_size_numel` intentionally pinned to the paper's value even though our latent grid is `(48, 56, 48)`, so the timestep prior is identical to T1C-RFlow's |
| Velocity target | `u_t = z_T1c − z_noise` (rectified-flow linear interpolant) | Eidex 2025 Eq. 3 |
| Loss | `F.l1_loss(v_pred, u_target)` — **L1**, not L2 | Eidex 2025 Eq. 4 |
| Optimiser | `AdamW(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)` | Eidex 2025 §4 |
| Conditioning | channel concat `[noisy_z_T1c, z_T1pre, z_FLAIR]`; `y=None` at every forward | DiT-3D paper-faithful (no class label) |
| Mixed precision | `torch.amp.GradScaler` + `autocast(fp16)` enabled by default | T1C-RFlow upstream uses it; Peebles & Xie silent |
| EMA / grad clipping / augmentation | none | paper-faithful |
| Checkpoints | `best_net_dit.pth` (min epoch-mean train loss), `latest_net_dit.pth`, `epoch_<N>_net_dit.pth` every `save_epoch_freq` | wrapper-added — exhaustive-val is asynchronous |
| Patience | epoch-mean train loss; default 100 for production, 0 for smokes | VENA paired-axis |

### Architecture recovery at inference time

The wrapper persists the DiT architecture kwargs (`input_size`,
`in_channels`, `out_channels`, `hidden_size`, `depth`, `num_heads`,
`patch_size`, `mlp_ratio`) as an `arch_meta` block **inside every saved
checkpoint** so `run_inference` can rebuild the model with no YAML
consultation. Pinned by
`test_inference.py::test_rebuild_dit_from_meta_matches_fresh_build`. This
deviates from T1C-RFlow's pattern (`--unet-arch-config` required at infer
time) and is the cleaner default — added to the skill as Step 3.6.1.

## VAE for inference decode

VENA's MAISI-V2 (`autoencoder_v2.pt`) via `vena.common.load_autoencoder`.
The vendored upstream ships no VAE in the `dit_3d/` snapshot (we only
vendored `dit3d.py`, `dit3d_wrapper.py`, `test_dit.py`). The path is
required at `vena-competitor-dit-3d-infer --vae-checkpoint ...` time and
documented in `src/external/LINKS.md` per platform.

## Per-platform recipe and validation

### Server-3 (RTX 4090, no SLURM) — **smoke green 2026-06-15 16:43**

- Conda env `vena` (torch 2.5.1+cu121). `timm>=0.9` installed via
  `pip install timm` (~80 MB).
- `server3/launcher_dit_3d_server3_4ep.sh`: rsync repo → ssh icai-server →
  `screen -dmS vena-dit-3d-smoke …` → exit. **rsync excludes** include
  `src/external/t1c_rflow/upstream/checkpoints/` (T1C-RFlow's 80 MB LFS
  VAE — not needed for any competitor; we use MAISI-V2).
- Smoke result: 6 cohorts loaded (UCSF-PDGM + BraTS-GLI + UPENN-GBM +
  IvyGAP + LUMIERE + REMBRANDT), 1 patient each. Loss descended
  1.1187 → 1.1168 across 4 epochs (~0.5 s/epoch on RTX 4090 + AMP at
  batch=1). All checkpoints (`best`, `latest`, `epoch_{0..3}`) + step/epoch
  CSVs + `decision.json` (`completed=true`) written. Run id:
  `2026-06-15T14-43-07_competitor_dit_3d_smoke_4ep_multicohort_fc68720`.

### Loginexa (Picasso V100-DGXS-32GB interactive node) — **smoke green 2026-06-15 16:58**

- Conda env `vena-v100` (torch 2.7.1+cu126). `timm>=0.9` installed
  (preserved torch 2.7.1+cu126).
- Auto-picked freest V100 by `nvidia-smi memory.free` → GPU 3.
- Smoke result: same 6 cohorts × 1 patient, 2 epochs in ~10 s
  (~0.8 s/epoch), loss 1.1191 → 1.1173. Sentinel emitted; tmux detached.

### Picasso (A100 DGX, full training) — **submitted 2026-06-15 17:01, job 1082295**

- Conda env `vena` (torch 2.12+cu130). `timm>=0.9` installed (preserved
  torch 2.12+cu130).
- Worker `slurm/runs/worker_dit_3d_picasso_full.sh` requests
  `--constraint=dgx --partition=gpu_partition --gres=gpu:1
  --time=3-00:00:00 --mem=64G --cpus-per-task=8`. `experiments_root` uses
  lowercase `…/execs/vena/…` (skill §4 note).
- Same `corpus_picasso.json` as VENA / T1C-RFlow (6 cv cohorts).

## Inference contract

```
vena-competitor-dit-3d-infer \
    --run-dir   <run_dir> \
    --image-h5  <UCSFPDGM_image.h5> \
    --latent-h5 <UCSFPDGM_latents.h5> \
    --vae-checkpoint /abs/path/to/autoencoder_v2.pt \
    --epoch best --n-patients 10 --phase val \
    --nfe 50 --nfe 100 --nfe 200
```

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
| backbone | | | ⚠ **DiT-B/4 transformer** (vs VENA's MAISI U-Net trunk) |
| trunk fine-tuning | | | ⚠ none — DiT is randomly initialised; VENA warm-starts NV-Generate-MR |
| ControlNet / mask conditioning | | | ⚠ absent — Eidex 2025 §4 DiT-3D baseline has no mask conditioning |
| target metric | | ⚠ best on epoch-mean train loss | |

## Paper-budget rationale (skill §3.11)

The Eidex 2025 paper trained ALL baselines (DDPM, Pix2pix, DiT-3D, T1C-RFlow)
on the same BraTS-2024 budget: 2,860 train samples × 100 epochs = 286,000
sample exposures. VENA's multi-cohort fold-0 train union (probed
2026-06-15 on Picasso) ≈ 1,751 latent volumes/epoch. The fairness contract:

```
max_epochs = ceil(286,000 / 1,751) ≈ 164
```

The picasso `max_epochs: 164, patience: 0, batch: 4` cap is the same one
T1C-RFlow's full config uses — keeps the two competitors comparable
head-to-head and defensible against the "undertrained competitor" charge.

## Things to watch / things that nearly bit me

1. **Vendoring the same upstream twice.** T1C-RFlow already lives at
   `src/external/t1c_rflow/upstream/` (same SHA `fc8314f6`). I copied the
   DiT-3D files from there into a fresh `src/external/dit_3d/upstream/`
   rather than cross-importing — skill anti-pattern 7 forbids touching a
   sibling competitor's vendored snapshot from this one. Disk cost: ~35 KB
   (3 .py files); independence cost: 0.
2. **timm dependency.** The `dit3d.py` `from timm.layers import to_2tuple`
   and `from timm.models.vision_transformer import PatchEmbed, Attention, Mlp`
   makes timm a hard dep. Added to `pyproject.toml`; installed on
   server-3, loginexa (`vena-v100`), Picasso (`vena`). All envs got
   `timm 1.0.27` without disturbing the pinned torch.
3. **DiT positional embeddings are fixed-size.** Unlike T1C-RFlow's U-Net
   (which is shape-agnostic), DiT's `pos_embed` is registered as a
   `nn.Parameter(requires_grad=False)` of shape `(1, num_patches,
   hidden_size)` built at `__init__`. Multi-cohort training **requires**
   every cohort's latent grid to match the input_size used to build the
   model — VENA's schema 2.0.0 trunk-÷8 constraint enforces this at the
   encoder. The runner peeks the first batch's shape and validates every
   subsequent batch (`DiT3DRunnerError` on mismatch).
4. **`hidden_size % 3 == 0` constraint.** The 3D sin-cos positional
   embedding (`dit3d.py:332`) splits `embed_dim` into thirds, one per
   spatial axis, with `assert embed_dim % 3 == 0`. DiT-B (768), DiT-L (1152),
   DiT-XL (1152) all satisfy this. The engine's Pydantic validator
   rejects mismatched values upfront.
5. **`arch_meta` in the checkpoint.** Storing the DiT kwargs alongside
   the state dict avoided the T1C-RFlow pattern of requiring
   `--unet-arch-config` at infer time. The infer CLI is simpler (one
   fewer flag) and there is no risk of YAML-vs-checkpoint drift. Added as
   a generalizable pattern to the skill (Step 3.6.1).
6. **`y=None` at every forward.** DiT's `LabelEmbedder` accepts an optional
   class label; we never use one (conditioning is fully in the channel
   concat). The runner's forward call passes `y=None` explicitly and
   tolerates wrappers that don't accept the `y` kwarg via a
   `co_varnames` check.
7. **Stdout `print` in `get_3d_sincos_pos_embed`.** The upstream's debug
   `print('grid_size:', grid_size)` was patched out (PATCHES.md P1); it
   would pollute the rich-formatted training log.
8. **Server-3 GPU 0 may be busy.** Override with `GPU_ID=N
   bash launcher_…sh` (env var is forwarded into the screen session).
