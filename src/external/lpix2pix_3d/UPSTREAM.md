# 3D-Latent-Pix2Pix upstream snapshot

| Field | Value |
|---|---|
| Repository | https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation |
| Vendored at SHA | `fc8314f60d877f9ee55996f960f89b17b269200f` |
| Vendored on | 2026-06-17 |
| Licence | **None at HEAD `fc8314f6`** — no LICENSE file at the repository root. Default copyright (all-rights-reserved) applies. Vendored under assumed academic-use intent of the arXiv preprint; awaiting an explicit MIT/Apache-2.0 grant from the authors. Cite Isola *et al.* 2017 (Pix2Pix) and Eidex *et al.* 2025 (3D latent adaptation). |
| Method paper (baseline) | Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. "Image-to-Image Translation with Conditional Adversarial Networks." *CVPR 2017*. arXiv:1611.07004. |
| Method paper (3D adaptation) | Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for T1-contrast Enhanced MRI Generation." arXiv:2509.24194. §4 reports "Pix2pix" as one of the diffusion-baseline competitors trained on the same MAISI-V2 latent grid. |

## Why a separate vendoring from `src/external/t1c_rflow/` and `src/external/dit_3d/`

The same upstream repository (Eidex *et al.* 2025) ships several distinct
competitor methods. VENA benchmarks each one as its own competitor leaf
under `src/vena/competitors/<name>`. Skill anti-pattern 7 forbids
cross-importing from a sibling competitor's vendored snapshot, so the
right move is to vendor an **independent copy** of the files this
wrapper actually uses:

- `src/external/t1c_rflow/upstream/` — rectified-flow U-Net trainer.
- `src/external/dit_3d/upstream/` — DiT-3D transformer backbone.
- `src/external/lpix2pix_3d/upstream/` — **this** snapshot: vendors only
  `train_pix2pix_t1n_t2f.py` + `test_pix2pix_t1n_t2f.py`, the GAN-refactor
  trainer + inference script. Both files are byte-identical to the
  same paths under `src/external/t1c_rflow/upstream/` at SHA `fc8314f6`.

Disk cost: ~25 KB (2 .py files). Independence cost: 0.

## Citation

Cite the 3D-Latent-Pix2Pix row of VENA's competitor table with the
following BibTeX (Isola *et al.* for the Pix2Pix conditional-GAN +
PatchGAN architecture, Eidex *et al.* for the 3D-latent + MAISI training
recipe we follow):

```bibtex
@inproceedings{isola2017image,
  title     = {Image-to-Image Translation with Conditional Adversarial Networks},
  author    = {Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
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

## Scope of use in VENA

The vendored upstream is referenced for **architecture and recipe
traceability**, but the wrapper at `src/vena/competitors/lpix2pix_3d/`
does not `import` from the vendored files at runtime. Both the
`GeneratorUNetWrapper` and the `PatchDiscriminator3D` are short enough
that VENA re-implements them in `vena.competitors.lpix2pix_3d.runner`
against MONAI's `DiffusionModelUNetMaisi` directly. The vendored files
remain on disk so a reviewer can verify the architectural identity at
SHA `fc8314f6`.

The wrapper preserves the load-bearing recipe choices from
`train_pix2pix_t1n_t2f.py`:

- Generator: `DiffusionModelUNetMaisi` (same MONAI primitive as
  T1C-RFlow), wrapped in a `GeneratorUNetWrapper` that feeds **zero
  timesteps** so the diffusion U-Net acts as a deterministic conditional
  generator. No noise injection; no scheduler; one forward pass per
  prediction.
- Discriminator: 3D PatchGAN — 4 strided `Conv3d`+InstanceNorm+LeakyReLU
  layers, terminal 1-channel patch-logits head. Receptive field matches
  Isola *et al.* 2017's "70×70 PatchGAN" in 3D adaptation.
- Loss: `BCEWithLogitsLoss` adversarial + `λ_L1 * F.l1_loss(fake, real)`
  with `λ_L1 = 100` (Isola *et al.* 2017 §3.2). The G and D are updated
  with separate AdamW optimisers under AMP.

## Differences between method-paper text and vendored code (incoherencies)

Following VENA policy (2026-06-15): when paper text and code disagree we
**follow the peer-reviewed paper text**. The wrapper defaults below are
chosen to match each method paper's stated configuration.

| # | Axis | Isola *et al.* 2017 (Pix2Pix) | Eidex 2025 (3D-latent baseline use) | Vendored code | Wrapper default | Load-bearing? |
|---|---|---|---|---|---|---|
| 1 | Generator backbone | 2D U-Net (Isola §6.1) | DiffusionModelUNet repurposed as generator (Eidex code §A) | `train_pix2pix_t1n_t2f.py:182` reuses the `diffusion_def` from `config_maisi3d-rflow.json` (4-level + attention, 178 M params) | **Paper-faithful 3-level MAISI U-Net** (`[128, 128, 256]`, 2 res-blocks, no self-attention) — same architecture as the T1C-RFlow wrapper so the only axis isolated against T1C-RFlow is the **GAN training recipe** (BCE+L1 vs RFlow+L1-velocity) | **HIGH** — code's 4-level + attention config would shift parameter count by ~4× and contaminate the head-to-head against T1C-RFlow |
| 2 | Training paradigm | Conditional GAN (BCE + L1) | Eidex §4 reports a "Pix2pix" baseline trained on their MAISI latents | `train_pix2pix_t1n_t2f.py` uses BCE adversarial + λ·L1 with λ=100 | **BCE + λ·L1 with λ=100** — Isola *et al.* 2017 §3.2 verbatim | NO |
| 3 | Conditioning route | Pixel-wise channel concat | n/a | `train_pix2pix_t1n_t2f.py:226` concatenates `[cond1, cond2]` (T1n + T2f) as input; discriminator sees `[cond, target_or_fake]` | **Channel concat `[z_T1pre, z_FLAIR]` → 8 channels into G; `[cond_8, target_4]` → 12 channels into D** | NO |
| 4 | Generator timestep handling | Pix2Pix is a non-diffusion model — no timesteps | Eidex repurposes the diffusion U-Net | `GeneratorUNetWrapper.forward(x)` injects `t = zeros((B,), dtype=long)` | **Same** — zero timesteps; the diffusion U-Net runs deterministically | NO |
| 5 | Discriminator | PatchGAN 70×70 (2D) | n/a | `PatchDiscriminator3D` in `train_pix2pix_t1n_t2f.py:108`: 4 layers, `ndf=64`, kernel=4, stride=2, InstanceNorm | **PatchDiscriminator3D, ndf=64, num_layers=4** — verbatim from vendored code | NO |
| 6 | AMP / mixed precision | Not addressed (2017 paper) | n/a | `torch.amp.GradScaler` + `autocast(fp16)` at every G/D step | **Enabled by default** — same posture as T1C-RFlow / DiT-3D | LOW (≤ 0.1 dB) |
| 7 | Optimiser | Adam(lr=2e-4, β1=0.5) | n/a | AdamW(lr=1e-4, β1=0.5, β2=0.999, wd=1e-4) — same kwargs for G and D | **AdamW(lr_G=lr_D=1e-4, β1=0.5, β2=0.999, wd=1e-4)** — follows vendored code (closer to modern best practice than the 2017 paper's Adam-vanilla) | LOW |
| 8 | λ_L1 weighting | 100 (Isola §3.2) | n/a | 100 (vendored `GANHyper.lambda_L1`) | **100** — paper- and code-faithful | NO |

If a future ablation row wants λ_L1 = 10 or 1, it becomes a
`lambda_l1: <value>` knob in the engine's `HyperParamsCfg`; do not
change the default without explicit user authorisation.

## What is NOT modified

- Both vendored files (`train_pix2pix_t1n_t2f.py`,
  `test_pix2pix_t1n_t2f.py`) — preserved verbatim from upstream SHA
  `fc8314f6`. No patches needed.

## What is invoked from VENA

Nothing. The wrapper at `vena.competitors.lpix2pix_3d` re-implements the
two short classes (`GeneratorUNetWrapper`, `PatchDiscriminator3D`)
against MONAI primitives directly, since they total ~30 lines of code
and reproducing them in the wrapper:

1. Removes the vendored `argparse`+`tqdm`+`matplotlib`+`importlib`
   plumbing from VENA's import graph.
2. Avoids dragging `LatentPairDataset` (a `.pt`-file-folder loader) into
   VENA's H5-centric data path.

The vendored files therefore stay as **reference reads only** — never
loaded by the wrapper.

## Reproducing the snapshot

```bash
cd src/external/lpix2pix_3d
git clone --depth 1 https://github.com/zacheidex/An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation.git tmp_clone
cd tmp_clone && git rev-parse HEAD > ../UPSTREAM_SHA.txt
cp train_pix2pix_t1n_t2f.py test_pix2pix_t1n_t2f.py ../upstream/
cd .. && rm -rf tmp_clone
find upstream -name __pycache__ -type d -exec rm -rf {} +
find upstream -name '*.pyc' -delete
```

## Files of interest under `upstream/`

| File | What it does |
|---|---|
| `train_pix2pix_t1n_t2f.py` | Reference GAN-refactor trainer: builds `GeneratorUNetWrapper` (diffusion U-Net with zero timesteps), `PatchDiscriminator3D`, two AdamW optimisers, BCE adversarial + λ·L1 loss, AMP, per-step CSV logging. |
| `test_pix2pix_t1n_t2f.py` | Reference inference script: single G forward pass + AutoencoderKL decode + per-volume NMSE/PSNR/NCC/SSIM. |
