# T1C-RFlow upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation |
| Vendored at SHA | `fc8314f60d877f9ee55996f960f89b17b269200f` |
| Vendored on | 2026-06-15 |
| Licence | **None at HEAD `fc8314f6`** — no LICENSE file at the repository root. Default copyright (all-rights-reserved) applies. Vendored under assumed academic-use intent of the arXiv preprint; awaiting an explicit MIT/Apache-2.0 grant from the authors. Cite Eidex *et al.* 2025 in every derived artefact. |
| Citation | Eidex, Z., Safari, M., Ding, J., Qiu, R., Roper, J., Yu, D., Shu, H.-K., Tian, Z., Mao, H., Yang, X. "An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation." *arXiv preprint* arXiv:2509.24194, 2025. |

## BibTeX

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

## Scope of use in VENA

The upstream files are vendored for **reference and reproducibility only**. The VENA
competitor wrapper (`src/vena/competitors/t1c_rflow/`) does not `import` from
`src/external/t1c_rflow/upstream/`. The wrapper rebuilds the model from MONAI
primitives directly — `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi`
and `monai.networks.schedulers.rectified_flow.RFlowScheduler` — matching the upstream's
architecture config (`maisi/configs/config_maisi3d-rflow.json`) and its in-script
overrides (`train_rflow.py` line 129: `in_channels = latent_channels * 3 = 12`; line 136:
scheduler kwargs `use_discrete_timesteps=True, sample_method="logit-normal",
use_timestep_transform=True, base_img_size_numel=64*64*48`).

The training script's loss form is reproduced verbatim:

```python
# train_rflow.py:207
loss = F.l1_loss(noise_pred, (tgt - noise))
```

This is the velocity target `u_t = z_T1c - z_noise` (paper Eq. 3) under L1 norm
(paper Eq. 4).

## Differences between upstream paper text and upstream code (incoherencies)

These are not bugs in VENA; they are inconsistencies inside the upstream artefact
that future readers should know about. The wrapper follows the **code** (canonical),
not the paper text.

1. **U-Net channel counts.** Paper §3 reports `[128, 128, 256]` with two residual
   blocks per layer (three levels). The vendored `config_maisi3d-rflow.json`
   `diffusion_unet_def.num_channels` is `[64, 128, 256, 512]` with attention at the
   last two levels (four levels). The wrapper uses the config value.
2. **`use_discrete_timesteps`.** Paper §3.2 implies discrete-time training (1000
   timesteps). The vendored config sets `use_discrete_timesteps: false`, but
   `train_rflow.py` line 138 overrides it to `True` at runtime. The wrapper sets
   `True` to match the script.
3. **Mixed precision.** Paper §4 does not mention AMP. `train_rflow.py` uses
   `torch.amp.GradScaler` + `autocast(dtype=torch.float16)` (lines 146, 205-210).
   The wrapper enables AMP by default (`use_amp: true`) for code-faithfulness, with
   a YAML toggle to disable for ablations.
4. **VAE checkpoint.** Paper §3.1 cites the MAISI VAE (Guo *et al.* 2024). The
   upstream ships `checkpoints/autoencoder_epoch273.pt` (Git LFS, ~80 MB). VENA
   uses MAISI-V2 (`autoencoder_v2.pt`); these are the same architecture family but
   different training data. VENA reads pre-encoded MAISI-V2 latents from the
   project's latent H5 cache, so the upstream VAE checkpoint is never loaded.

## What is NOT modified

- Model architecture, scheduler config, loss form, optimiser, training schedule —
  byte-identical to upstream.
- The vendored Python files are **read-only references**. No runtime patch.

## What is invoked from VENA

Nothing. See *Scope of use* above.

## Reproducing the snapshot

```bash
cd src/external/t1c_rflow
git clone --depth 1 https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation.git upstream
cd upstream && git rev-parse HEAD > ../UPSTREAM_SHA.txt
rm -rf .git
find . -name __pycache__ -type d -exec rm -rf {} +
find . -name '*.pyc' -delete
```

## Files of interest under `upstream/`

| File | What it does |
|---|---|
| `train_rflow.py` | RFlow training loop (the reference implementation our wrapper mirrors). |
| `test_rflow.py` | Inference: Euler integration `t: 1 → 0` over K steps, decodes via the frozen VAE. |
| `generate_latent_maps.py` | One-shot VAE encode pass producing per-modality `_z_mu.pt` / `_z_sigma.pt` files. VENA does not run this — VENA reads its own latent H5 cache. |
| `dit3d.py`, `dit3d_wrapper.py`, `test_dit.py` | Alternative DiT-3D backbone (paper baseline). Out of scope for this competitor entry. |
| `train_pix2pix_t1n_t2f.py`, `test_pix2pix_t1n_t2f.py` | Pix2pix baseline. Out of scope. |
| `train_ddpm.py`, `test_ddpm_t1_flair_final.py` | Latent DDPM baseline. Out of scope. |
| `maisi/configs/config_maisi3d-rflow.json` | The U-Net + scheduler architecture config. The wrapper reads this verbatim, then overrides `diffusion_unet_def.in_channels` to 12 (concat conditioning). |
| `maisi/configs/config_maisi_vae_train.json` | MAISI VAE config. Not loaded by the wrapper (we use VENA's MAISI-V2). |
| `checkpoints/autoencoder_epoch273.pt` | MAISI VAE weights at epoch 273 (Git LFS). Not loaded by the wrapper. |
| `requirements.txt` | `torch>=2.1`, `monai>=1.5.0`, `numpy>=1.23`, `nibabel>=5.2`, `tqdm>=4.66`, `matplotlib>=3.7`, `timm>=0.9`. All already satisfied by VENA's `vena` conda env. |
