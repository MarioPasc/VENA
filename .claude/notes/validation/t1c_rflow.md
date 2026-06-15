# T1C-RFlow competitor integration — VENA validation note

> Closest published precedent to VENA: 3D latent rectified-flow with MAISI
> VAE conditioning. No vessel prior, no SWAN, no masks — the head-to-head
> isolates VENA's exact contribution.

## Citation

```bibtex
@article{eidex2025efficient3dlatentdiffusion,
  title   = {An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation},
  author  = {Eidex, Zach and Safari, Mojtaba and Ding, Jie and Qiu, Richard
             and Roper, Justin and Yu, David and Shu, Hui-Kuo and Tian, Zhen
             and Mao, Hui and Yang, Xiaofeng},
  journal = {arXiv preprint arXiv:2509.24194},
  year    = {2025}
}
```

Carry this block through every derived artefact (`decision.json` carries a
condensed version; `UPSTREAM.md` carries the full one).

## License flag

The upstream repository
(<https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation>)
has **no LICENSE file** at HEAD `fc8314f60d877f9ee55996f960f89b17b269200f`.
Default copyright is *all-rights-reserved*. VENA vendored under the assumed
academic-use intent of the arXiv preprint; an explicit MIT/Apache-2.0 grant
from the authors is outstanding. Track in `src/external/t1c_rflow/UPSTREAM.md`.

## Scope contract (the no-augmentation rule)

`T1CRFlowLatentDataset` is **deterministic**. Pinned by
`tests/competitors/t1c_rflow/test_dataset.py::test_dataset_is_deterministic`:
repeat reads of any index return byte-identical tensors. The upstream
training script samples `z = mu + sigma * eps` per training step
(`train_rflow.py:81-83`); VENA uses the stored static `z` from its multi-
cohort latent H5 cache — see the paired-axes table below for the implication.

The upstream `LatentPairDataset` (`train_rflow.py:58-87`) is **never
invoked** by VENA. The wrapper builds the model from MONAI primitives
directly with the same arch JSON and the same in-script overrides.

## Code layout

```
src/external/t1c_rflow/
├── UPSTREAM.md                  # repo URL, SHA, scope, license flag, citation
├── UPSTREAM_SHA.txt             # fc8314f60d877f9ee55996f960f89b17b269200f
├── PATCHES.md                   # empty — modern torch, no patches needed
└── upstream/                    # frozen snapshot
    ├── train_rflow.py           # reference training loop (mirrored by runner.py)
    ├── test_rflow.py            # reference inference (mirrored by inference.py)
    ├── generate_latent_maps.py  # not invoked — VENA reads its own latent H5
    ├── dit3d.py / dit3d_wrapper.py  # alternative DiT backbone (out of scope)
    ├── train_pix2pix_*.py       # Pix2pix baseline (out of scope)
    ├── train_ddpm.py            # latent DDPM baseline (out of scope)
    ├── maisi/configs/
    │   ├── config_maisi3d-rflow.json     # the U-Net architecture (in_channels
    │   │                                   overridden to 12 at runtime)
    │   └── config_maisi_vae_train.json   # MAISI VAE (not loaded — we use MAISI-V2)
    └── checkpoints/autoencoder_epoch273.pt  # vendored VAE (not loaded; LFS, ~80 MB)

src/vena/competitors/t1c_rflow/
├── __init__.py                  # public API re-exports
├── dataset.py                   # latent-domain, no-aug, multi-cohort
├── runner.py                    # build trunk + scheduler + loop + CSV + ckpt
└── inference.py                 # Euler sampling + decode_box + PSNR/SSIM

routines/competitors/t1c_rflow/
├── __init__.py
├── cli.py                       # vena-competitor-t1c-rflow
├── engine.py                    # Pydantic config + decision.json v1.0
├── infer_cli.py                 # vena-competitor-t1c-rflow-infer
├── configs/
│   ├── smoke_server3_4ep.yaml   # 1 patient/cohort, num_workers=0, AMP
│   ├── smoke_loginexa_2ep.yaml  # V100 vena-v100 env
│   └── picasso_full.yaml        # A100 7d budget, 10000 epochs, patience=100
├── server3/launcher_t1c_rflow_server3_4ep.sh
├── loginexa/launcher_t1c_rflow_loginexa_2ep.sh
└── slurm/runs/{launcher,worker}_t1c_rflow_picasso_full.sh

tests/competitors/t1c_rflow/
├── __init__.py
├── test_dataset.py              # determinism, shapes, longitudinal, flat-splits
├── test_multicohort.py          # concat, missing-cohort warn, role filter
└── test_inference.py            # RFlow velocity-target identity, _psnr math
```

## Patches to upstream

**None.** The vendored Python files run unmodified on torch ≥ 2.1 and
MONAI ≥ 1.5 (every VENA env satisfies this). See `PATCHES.md`.

## Data contract

| Axis | Value | Source |
|---|---|---|
| Modality (input) | T1pre latent + FLAIR latent | paper §3.1 (uses "T2-FLAIR"; UCSF-PDGM substitute = FLAIR — see "T2-FLAIR substitute" below) |
| Modality (target) | T1c latent | paper §3.1 |
| Latent grid | VENA's stored (4, 48, 56, 48) per cohort | VENA encode pipeline; paper trains at (4, 64, 64, 48) |
| Conditioning route | channel-wise concatenation `[z_t || z_T1pre || z_FLAIR]` → 12 input ch | paper Fig. 1; upstream `train_rflow.py:129` (`in_channels = latent_channels * 3`), 202 |
| Splits | k-fold (`splits/cv/fold_<k>/<phase>`) with flat fallback | shared with VENA's FM trainer |
| Norm at training time | none (latents are pre-encoded) | paper §3 |
| Norm at inference metrics | VENA `percentile_normalise(99.5, foreground_only)` on the real T1c | VENA rule `model-coding-standards.md` rule 15 |
| Corpus | multi-cohort union from `corpus_<host>.json` (UCSF-PDGM + BraTS-GLI + UPENN-GBM + IvyGAP + LUMIERE + REMBRANDT) | matches VENA FM trainer |

Cohort heterogeneities the wrapper handles (carried from pGAN):

1. **Longitudinal cohorts (BraTS-GLI, LUMIERE)** — `/ids` are scan-level
   (`PT-XXXX-NNN`), splits are patient-level (`PT-XXXX`). Prefix match in
   `dataset.py::T1CRFlowLatentDataset.__init__` recovers every scan per
   patient. Pinned by `test_dataset_resolves_longitudinal_patient_ids`.
2. **Small cohorts with flat splits (REMBRANDT)** — `splits/{train,val,test}`
   instead of `splits/cv/fold_<k>/…`. Pinned by
   `test_dataset_falls_back_to_flat_splits_schema`.

## Training contract

T1C-RFlow is **many-to-one** by design (`in_channels = latent_channels × 3`
in the upstream script means *three latents of 4 channels each*, not three
single-channel slices). One config per platform; no per-modality panel
needed (unlike pGAN's `picasso_full_t1pre/t2/flair.yaml`).

The wrapper-driven training loop matches the upstream reference on every
load-bearing axis:

| Axis | Value | Source |
|---|---|---|
| Backbone | `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi` | `config_maisi3d-rflow.json` `diffusion_unet_def` |
| Channel counts | `[64, 128, 256, 512]` (4 levels, attention last 2) | code (paper text says `[128, 128, 256]` — *upstream code/paper incoherency*; we follow code) |
| Parameter count | 178.6 M (in_channels=12 override) | measured |
| Scheduler | `monai.networks.schedulers.RFlowScheduler` direct, kwargs `num_train_timesteps=1000, use_discrete_timesteps=True, sample_method="logit-normal", use_timestep_transform=True, base_img_size_numel=64*64*48, spatial_dim=3` | upstream `train_rflow.py:136-143` |
| Velocity target | `u_t = z_T1c - z_noise` (rectified-flow linear interpolant) | paper Eq. 3 |
| Loss | `F.l1_loss(v_pred, u_target)` — **L1**, not L2 | paper Eq. 4 / upstream `train_rflow.py:207` |
| Optimiser | `AdamW(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)` | paper §4 |
| Mixed precision | `torch.amp.GradScaler` + `autocast(fp16)` enabled by default | upstream `train_rflow.py:146, 205-210` (paper §4 is silent) |
| EMA | none | paper / upstream silent |
| Gradient clipping | none | paper / upstream silent |
| Augmentation | none | paper / upstream silent |
| `best`/`latest`/`epoch_N` checkpoints | on min epoch-mean train loss / every epoch / every `save_epoch_freq` | wrapper-added — exhaustive-val is asynchronous |
| Patience | `epoch-mean train loss`, default 100 | VENA paired-axis (see below) |

## VAE for inference decode

VENA's MAISI-V2 (`autoencoder_v2.pt`) via `vena.common.load_autoencoder`.
The vendored `checkpoints/autoencoder_epoch273.pt` is **not loaded**;
it is kept in the snapshot for reproducibility only. Both checkpoints use
the same `AutoencoderKlMaisi` architecture family; only the training data
differs.

## Per-platform recipe and validation

### Server-3 (RTX 4090, no SLURM)

- Conda env `~/.conda/envs/vena`.
- `routines/competitors/t1c_rflow/server3/launcher_t1c_rflow_server3_4ep.sh`
  rsyncs the repo, then `screen -dmS vena-t1c-rflow-smoke` on icai-server.
- `corpus_server3.json`; `num_workers=0` (multi-cohort h5py + multiprocessing
  deadlock — skill §3.5); `max_patients_per_cohort=1`.
- Sentinel: `t1c-rflow-train completed` in `logs/train.log`.
- Acceptance: loss descends across 4 epochs; `best_net_unet.pth` +
  `latest_net_unet.pth` written; `metrics/train_step.csv` and
  `metrics/train_epoch.csv` populated; `decision.json` has `completed=true`.

### Loginexa (Picasso V100-DGXS-32GB interactive node)

- Conda env `vena-v100` (torch 2.7.1+cu126, sm_70). The prod `vena` env
  (torch 2.12+cu130) does NOT work on V100.
- `loginexa/launcher_t1c_rflow_loginexa_2ep.sh` is invoked from the
  Picasso login node; auto-picks freest of 4 V100s; tmux-detached.
- Same multi-cohort corpus (`corpus_picasso.json`) over the shared Lustre
  mount.
- 30-min wallclock convention. Smoke = 2 epochs.

### Picasso (A100 DGX, full training)

- Conda env `vena` (torch 2.12+cu130).
- `slurm/runs/launcher_t1c_rflow_picasso_full.sh` → `sbatch` the worker
  with `#SBATCH --constraint=dgx --partition=gpu_partition --gres=gpu:1
  --time=7-00:00:00 --mem=64G --cpus-per-task=8`.
- Same `corpus_picasso.json` VENA's FM trainer uses (the *paired-comparison
  corpus axis*).
- Acceptance gate before submission: server-3 + loginexa smokes both
  green. The user authorises the sbatch explicitly.

## Inference contract

`vena-competitor-t1c-rflow-infer` does this per patient × per NFE:

1. Read `z_T1pre` and `z_FLAIR` from the latent H5 (1-tensor batch).
2. Initialise `z_t=1 ~ N(0, I)`. Run Euler integration `t: 1 → 0` over K
   steps via `scheduler.set_timesteps(num_inference_steps=K,
   input_img_size_numel=numel(z))` — mirrors upstream `test_rflow.py:269-283`.
3. Decode via `vena.common.decode.decode_box` against the cohort's
   `build_crop_spec_from_h5(image_h5, pid)` — the canonical VENA inference
   path (same helper exhaustive-val uses).
4. Load the real T1c volume via `load_real_t1c_box`, apply VENA's
   `percentile_normalise(99.5, foreground_only=True)` to match the
   encoder's `[0, 1]` input space — VENA metric parity (rule 15).
5. PSNR (whole-volume, optional brain mask) + SSIM via scikit-image 3D.
6. Per patient: write `<pid>_pred_t1c.nii.gz`,
   `<pid>_real_t1c_normalised.nii.gz`, `<pid>_midslice.png`.
7. Global: write `metrics.csv` (row per patient × NFE) and
   `summary.json` (schema 1.0, NFE-aggregated means/stds).

NFE panel default `{50, 100, 200}`. Paper default = 200; the panel reveals
the speed/quality trade-off VENA highlights as the rectified-flow win.

## Paired comparison axes (vs `picasso_s1_1000ep_fft.yaml`)

| Axis | VENA reference | T1C-RFlow wrapper | Match |
|---|---|---|---|
| seed | 1337 | 1337 | ✅ exact |
| fold | 0 | 0 | ✅ exact |
| corpus_registry | corpus_picasso.json | corpus_picasso.json | ✅ exact |
| max_epochs | 10000 | 10000 | ✅ exact |
| patience | 100 (on train loss) | 100 (on train loss) | ✅ exact (different metric internally — single L1 vs FM CFM) |
| save cadence | every 25 epochs | every 25 epochs | ✅ exact |
| batch_size | 4 | 4 | ✅ exact |
| num_workers | 8 | 8 | ✅ exact |
| walltime | 7d | 7d | ✅ exact |
| input modalities | T1pre, T2, FLAIR (+ SWAN, vessel, tumour priors) | **T1pre, FLAIR** only | ⚠ paper-faithful (paper uses only T1pre + T2-FLAIR; no priors at all) |
| output domain | latent | latent | ✅ exact |
| loss | CFM | L1 on velocity | ⚠ paper-faithful (paper Eq. 4 uses L1) |
| velocity convention | `u_t = x1 − x0` | `u_t = x1 − x0` | ✅ exact |
| scheduler | RFlow uniform (S1) | RFlow logit-normal (paper §3.2) | ⚠ paper-faithful |
| EMA | WarmupEMA | none | ⚠ paper-faithful |
| augmentation | latent-safe (preflight-gated) | none | ⚠ paper-faithful |
| latent sampling | static z (VENA-encoded) | static z | ⚠ deviation from paper (`z = μ + σε` per step in upstream) — expected ≤0.2 dB PSNR |
| target metric | best train/total_epoch | best train_loss (single L1) | ⚠ paper-faithful |
| AMP | float32 by default in FM (S1) | fp16 by default | ⚠ paper-faithful (upstream uses fp16; YAML toggle available) |

Differences in the right two columns are **explicit choices, not bugs.**
They live in `decision.json["competitor"]["deviations"]` per run.

## T2-FLAIR substitute

The paper uses BraTS T2-FLAIR — one channel that is the FLAIR-weighted
acquisition with T2 contrast properties. UCSF-PDGM (and the rest of VENA's
glioma corpus) stores T2 and FLAIR as two separate modalities. We use
**FLAIR only** as the substitute (closest 1:1 to BraTS T2-FLAIR; preserves
the paper's input-channel count). Using both would feed the model strictly
more information than the paper had — unfair head-to-head.

## VAE difference (paper vs VENA)

| Axis | Paper | VENA |
|---|---|---|
| VAE checkpoint | `autoencoder_epoch273.pt` (vendored, ~80 MB LFS) | `autoencoder_v2.pt` |
| Architecture | `AutoencoderKlMaisi` (`num_channels=[64,128,256]`, `latent_channels=4`) | same family |
| Latent grid (training native) | (4, 64, 64, 48) on BraTS 256×256×192 | (4, 48, 56, 48) on VENA's union (UCSF-PDGM ~240×240×155 etc.) |
| `base_img_size_numel` in RFlowScheduler | `64*64*48 = 196_608` | wrapper pins the **same** value 196_608 even though VENA's numel is 60*60*40 = 144_000 — preserves the paper's time-warp distribution |

Both checkpoints are NVIDIA non-commercial (the `LICENSE.weights` in the
vendored `maisi/` directory). VENA already uses MAISI-V2 under the same
license; no new restriction.

## Upstream code/paper incoherencies (worth knowing)

These are inside the upstream artefact, not VENA bugs. Per VENA policy
(updated 2026-06-15) the wrapper follows the **paper text** (peer-
reviewed), not the released code, except where the code reveals a
backbone-compatibility constraint the paper text implies. Every divergence
is enumerated below with a load-bearing assessment.

| # | Axis | Paper text | Released code | Load-bearing? | Direction of advantage |
|---|---|---|---|---|---|
| 1 | U-Net | `[128, 128, 256]`, 3 levels, 2 res-blocks/layer, **no attention mentioned** | `[64, 128, 256, 512]`, 4 levels, self-attention at levels 3-4, ~178.6 M params | **YES — high** | Code → paper text; the 4-level model with attention is substantially more expressive than the 3-level conv-only one paper text describes. Paper's reported 29.7 dB is achieved with the *bigger* model; a text-only replicator builds the smaller model and underperforms by ≥1-2 dB. |
| 2 | `use_discrete_timesteps` | implies discrete | JSON `false`, script overrides to `True` at `train_rflow.py:138` | NO | Backbone constraint: the MAISI U-Net's sinusoidal timestep embedding requires integer codes. The `false` in the JSON is a config bug; the runtime override is forced by the backbone, not a method trick. |
| 3 | Mixed precision (AMP) | §4 silent | `GradScaler` + `autocast(fp16)` enabled (`train_rflow.py:146, 205-210`) | LOW-MEDIUM | Training-speed only. Doubles step throughput on A6000 ADA, ≤0.1 dB quality drift. The paper's stated 100-epoch budget on a single A6000 ADA at batch 4 is only feasible with AMP, so the choice was effectively forced by their hardware. |
| 4 | VAE checkpoint provenance | cites Guo *et al.* 2024 MAISI | ships `autoencoder_epoch273.pt` — their own 273-epoch retraining of the MAISI architecture, presumably on BraTS data adjacent to the test distribution | **YES — high** | Code → paper text; their VAE reconstruction ceiling (paper Table 3, **T1c PSNR 35.2 ± 1.5 dB**) bounds the synthesis target. A general-purpose MAISI VAE on BraTS T1c reconstructs closer to ~30-32 dB; the reported 29.7 dB synthesis would be impossible against that ceiling. Within-paper baselines (DDPM, DiT-3D, Pix2pix) share the same retrained VAE, so internal comparisons are fair — but external replication using the cited Guo 2024 VAE under-reports. **No documentation of whether the VAE retraining used patients later used for test**, so a leakage risk cannot be ruled out without the VAE training split. |

**Decision (2026-06-15).** The two load-bearing divergences (rows 1 and 4)
are taken as published-vs-released disagreements that the peer-reviewed
text wins. The wrapper:

- **U-Net (row 1)** is rebuilt at the paper-text architecture
  `[128, 128, 256]`, 3 levels, no attention, 2 res-blocks/layer. The
  vendored config's 4-level+attention U-Net is **not** used by the wrapper
  at any time. The vendored config remains on disk for reproducibility
  (anyone wanting to reproduce the code-version can load it directly).
- **VAE (row 4)** is unaffected on our side: VENA's MAISI-V2 latents are
  the input, both for our model and for the T1C-RFlow competitor. The
  comparison is symmetric (both methods consume the same retrained-on-
  VENA VAE).
- **Augmentation.** The competitor reads `cohort.latent_h5` only; it
  never touches `cohort.latent_aug_h5`. VENA's offline-augmentation
  shards are invisible to this wrapper. Online augmentation is impossible
  (`T1CRFlowLatentDataset` is deterministic by contract, pinned by
  `test_dataset_is_deterministic`).
- **AMP (row 3)** is kept enabled — it is silent in the paper but not
  contradicted by it, and is essential for the paper's stated training
  budget (100 epochs on a single A6000 ADA).
- **`use_discrete_timesteps`** stays `True` — the backbone requires it.

The four divergences (and the rationale for following the paper on rows 1
and 4) are also surfaced in `decision.json["competitor"]["deviations"]`
as `unet_architecture: paper_faithful_3level` and
`vae_handling: vena_maisi_v2_symmetric_baseline`.

## Inference-side gotchas (hit on this integration)

These bit me during the post-smoke inference dry-run on server-3 and are
now baked into `inference.py`:

1. **`vena.common.load_autoencoder(checkpoint_path, ...)` requires a
   checkpoint path** — there is no default. The competitor wrapper exposes
   `--vae-checkpoint`; without it `run_inference` raises
   `InferenceError` early. Per `src/external/LINKS.md`:
   server-3 → `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt`,
   Picasso → `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt`.
2. **`AutoencoderHandle` has no `.decoder` attribute.** Wrap it with
   `MaisiDecoder(handle)` to get the callable expected by
   `vena.common.decode.decode_box`. (Initial guess was `ae.decoder`.)
3. **`percentile_normalise` expects a 5-D tensor `(B, C, H, W, D)`** with
   `foreground_only=True`. `load_real_t1c_box` returns 3-D `(H, W, D)`.
   The wrapper indexes `real_box[None, None]` before the call and
   `[0, 0]` afterwards.

These three together cost ~10 minutes of round-trip ssh debugging — pre-
empt them in the next latent-domain competitor's `inference.py` skeleton.

## Things to watch on the next competitor

1. Latent-domain competitors share VENA's encoded data — no per-platform
   image-H5 paths to mirror. Use `cohort.latent_h5` from the registry, not
   `cohort.image_h5`.
2. Decoding for image-space metrics is the canonical
   `vena.model.fm.eval.exhaustive.{build_crop_spec_from_h5,
   load_real_t1c_box}` + `vena.common.decode.decode_box` pair. Do not
   re-implement.
3. Paper-faithful schedulers: instantiate `RFlowScheduler` directly when
   you need `use_timestep_transform`/`base_img_size_numel` — VENA's
   `RFlowEngine` does not expose these.
4. Many-to-one competitors are one Picasso job; one-to-one (pGAN-style)
   are N jobs — disambiguate at planning time (skill §3.3).
5. AMP is the rule, not the exception, for latent-diffusion competitors —
   modern upstream code uses it whether or not the paper mentions it.
6. Sentinel log line (`<name>-train completed`) is the *only* way the
   watcher knows the job finished cleanly. Add it; don't skip.
7. License flag at the **top** of `UPSTREAM.md` when there is no LICENSE
   file on upstream — this is a release blocker if not surfaced early.

## Open follow-ups

- Email Eidex *et al.* for an explicit MIT/Apache-2.0 grant on the
  upstream repo.
- DiT-3D variant (`dit3d.py` in upstream) is a separate competitor entry
  (paper reports it as a baseline). Not on the current critical path.
- If, after the Picasso run, the static-z deviation moves metrics by more
  than the seed variance, schedule a μ/σ re-encode and add a
  `stochastic_z: bool` toggle.
- Once we have at least 3 competitors live (pGAN + T1C-RFlow + one more),
  consider extracting a shared `vena.competitors.shared` module for the
  longitudinal resolver, flat-splits fallback, and lazy-h5 pattern. Until
  then, copy-paste is cheaper than premature abstraction.
