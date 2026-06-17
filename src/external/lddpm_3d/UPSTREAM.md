# 3D-LDDPM upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation |
| Vendored at SHA | `fc8314f60d877f9ee55996f960f89b17b269200f` |
| Vendored on | 2026-06-17 |
| Licence | **None at HEAD `fc8314f6`** — no LICENSE file at the repository root. Default copyright (all-rights-reserved) applies. Vendored under assumed academic-use intent of the arXiv preprint; awaiting an explicit MIT/Apache-2.0 grant from the authors. Cite Eidex *et al.* 2025 in every derived artefact. |
| Method paper (DDPM) | Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. arXiv:2006.11239. |
| Baseline reference | Eidex, Z. *et al.* 2025, "An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation," arXiv:2509.24194 — §4 uses 3D LDDPM as the diffusion-baseline row (`train_ddpm.py` + `test_ddpm_t1_flair_final.py`). |

## Why a separate vendoring from `src/external/t1c_rflow/` and `src/external/dit_3d/`

The same upstream repository (Eidex *et al.* 2025, SHA `fc8314f6`) ships three
distinct competitor methods that VENA benchmarks separately:

1. A rectified-flow U-Net trainer (`train_rflow.py`) — vendored at
   `src/external/t1c_rflow/upstream/` and consumed by the
   `vena.competitors.t1c_rflow` wrapper.
2. A DiT-3D transformer backbone (`dit3d.py`, `dit3d_wrapper.py`,
   `test_dit.py`) — vendored at `src/external/dit_3d/upstream/` and consumed
   by the `vena.competitors.dit_3d` wrapper.
3. A latent DDPM trainer (`train_ddpm.py`, `test_ddpm_t1_flair_final.py`) —
   **this** snapshot vendors those two files only.

Each competitor's vendored snapshot is independent so deleting one cannot
break the others (skill anti-pattern 7). The three snapshots are byte-
identical at SHA `fc8314f6` for their respective files; we keep them
duplicated rather than cross-importing.

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

## Scope of use in VENA

The VENA wrapper (`src/vena/competitors/lddpm_3d/`) **does not import** from
this vendored snapshot at runtime. The wrapper rebuilds the model from MONAI
primitives directly — `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi`
and `monai.networks.schedulers.DDPMScheduler` — matching the upstream's
in-script overrides (`train_ddpm.py` line 103:
`in_channels = latent_channels * 2`; line 113-119: scheduler kwargs
`num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195,
schedule="scaled_linear_beta", clip_sample=False`).

The training script's loss form is reproduced verbatim:

```python
# train_ddpm.py:170
loss = F.mse_loss(noise_pred, noise)
```

This is the standard DDPM epsilon-prediction loss (Ho *et al.* 2020 Eq. 14).
The diffusion forward is driven by `monai.inferers.DiffusionInferer(scheduler)`
with `mode="concat"` for conditioning (cf. `train_ddpm.py:162-169`).

## Differences between upstream paper text and upstream code (incoherencies)

These are not bugs in VENA; they are inconsistencies inside the upstream
artefact that future readers should know about. The wrapper follows the
**code** (canonical) for the scheduler and the **paper-faithful U-Net** for
the backbone (mirroring the T1C-RFlow wrapper policy — symmetric backbone,
delta is the scheduler/loss only).

| # | Axis | Eidex 2025 paper text | Vendored code | Wrapper default | Load-bearing? |
|---|---|---|---|---|---|
| 1 | U-Net channel counts | §3 reports `[128, 128, 256]` (3 levels, 2 res-blocks); §4 reuses same backbone for the LDDPM baseline | `config_maisi3d-rflow.json` declares `[64, 128, 256, 512]` (4 levels + attention) | **Paper-faithful**: `[128, 128, 256]`, 3 levels, 2 res-blocks, no self-attention — same as T1C-RFlow wrapper | HIGH — symmetric backbone with T1C-RFlow isolates the scheduler/loss as the only delta |
| 2 | DDPM beta schedule (train) | not explicit | `train_ddpm.py:113-119` sets `beta_start=0.0015, beta_end=0.0195, schedule="scaled_linear_beta", clip_sample=False` | follow code (canonical kwargs) | NO — these are the DDPM defaults from MONAI / Ho *et al.* 2020 |
| 3 | DDPM beta schedule (inference) | not explicit | `test_ddpm_t1_flair_final.py:121-127` sets `beta_start=0.0005, beta_end=0.0195` (different from training!) | **follow training values** (`beta_start=0.0015`) — the inference-time `0.0005` looks like an upstream typo (a smaller `beta_start` produces a different noise schedule than the model was trained under) | LOW — would shift the noise schedule at sampling; we prefer training-time consistency |
| 4 | Loss | not explicit; "denoising diffusion" implies eps-prediction | `train_ddpm.py:170` `F.mse_loss(noise_pred, noise)` | follow code | NO — standard Ho *et al.* 2020 DDPM loss |
| 5 | Mixed precision | not mentioned | `train_ddpm.py:122,161-173` uses `torch.amp.GradScaler` + `autocast(dtype=fp16)` | follow code (`use_amp: true` by default; YAML toggle for ablations) | LOW — matches T1C-RFlow wrapper's policy |
| 6 | Conditioning | not explicit in §4 | `train_ddpm.py:162-169` uses `DiffusionInferer(..., condition=cond, mode="concat")`; effectively `in_channels = latent_channels * (1 + cond_latents)` after stacking | follow code, same as T1C-RFlow wrapper | NO — channel-concat conditioning is the standard latent-DDPM recipe |
| 7 | VAE checkpoint | §3.1 cites the MAISI VAE (Guo *et al.* 2024) | `test_ddpm_t1_flair_final.py:99-105` ships `autoencoder_epoch273.pt` (Git LFS, ~80 MB, their retraining) | **VENA uses MAISI-V2** (`autoencoder_v2.pt`); VENA reads pre-encoded MAISI-V2 latents from the project's latent H5 cache so the upstream VAE checkpoint is never loaded | HIGH — same policy as T1C-RFlow wrapper |
| 8 | Sampler steps | not explicit | `test_ddpm_t1_flair_final.py:69` defaults to 1000 DDPM steps | inference CLI exposes `--nfe` (default 1000) | NO — DDPM K=1000 is the textbook default; we also expose K=200 for DDIM-like comparison |

## What is NOT modified

- Model architecture (built from MONAI primitives at the **paper-faithful**
  channel set, same as T1C-RFlow), scheduler kwargs (mirrored from
  `train_ddpm.py:113-119`), loss form, optimiser, AMP toggle — byte-
  identical to upstream where applicable.
- The vendored Python files are **read-only references**. No runtime patch.

## What is invoked from VENA

Nothing. See *Scope of use* above. The wrapper consumes MONAI primitives
directly.

## Reproducing the snapshot

```bash
cd src/external/lddpm_3d
git clone --depth 1 https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation.git tmp_clone
cd tmp_clone && git rev-parse HEAD > ../UPSTREAM_SHA.txt
cp train_ddpm.py test_ddpm_t1_flair_final.py ../upstream/
cd .. && rm -rf tmp_clone
find upstream -name __pycache__ -type d -exec rm -rf {} +
find upstream -name '*.pyc' -delete
```

## Files of interest under `upstream/`

| File | What it does |
|---|---|
| `train_ddpm.py` | Latent DDPM trainer (the reference implementation our wrapper mirrors): `DDPMScheduler` + `DiffusionInferer` + AdamW + AMP, MSE loss on noise prediction. |
| `test_ddpm_t1_flair_final.py` | Inference: K-step DDPM sampling loop (`scheduler.set_timesteps(K)` → for-loop step) + `autoencoder.decode`. VENA does not invoke this script — VENA's inference path is `vena.competitors.lddpm_3d.inference`. |
