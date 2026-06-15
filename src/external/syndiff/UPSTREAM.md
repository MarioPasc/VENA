# SynDiff — vendored upstream snapshot

## Source

- Repository: <https://github.com/icon-lab/SynDiff>
- Commit SHA: `fff3d8449e8c7ba38339be2f9ffd4aa5572beb4b` (short `fff3d844`)
- Commit message: `Update README.md`
- Snapshot date (UTC): 2026-06-15
- Snapshot author commit date: 2025-05-27 12:57:03 -0700

## Paper

Özbey M., Dalmaz O., Dar S.U.H., Bedel H.A., Özturk Ş., Güngör A., Çukur T.
"Unsupervised Medical Image Translation with Adversarial Diffusion Models."
*IEEE Transactions on Medical Imaging*, 2023. arXiv:2207.08208v3.

## Licence (mixed — read carefully)

- **Top-level (`LICENSE`):** MIT, Copyright (c) 2022 ICON Lab.
- **Per-file headers diverge** — the codebase composes work under three regimes:
  - `utils/EMA.py`: **NVIDIA Source Code License — for Denoising Diffusion GAN**
    (non-commercial research only). Adapted from NVlabs/LSGM.
  - `backbones/ncsnpp_generator_adagn.py`, `backbones/generator_resnet.py`,
    `backbones/layerspp.py`, `backbones/layers.py`,
    `backbones/up_or_down_sampling.py`, `backbones/dense_layer.py`:
    derived from Google Research `score_sde_pytorch` under **Apache 2.0**.
  - `backbones/discriminator.py`: NVIDIA Source Code License (Denoising
    Diffusion GAN). Non-commercial research only.
  - `backbones/im2im.py`: BSD-2 fork of the pytorch-CycleGAN-and-pix2pix
    repository (Zhu, Park, Isola, Efros).
  - `utils/op/upfirdn2d.{cpp,py,_kernel.cu}`, `utils/op/fused_*`: bear an
    `LICENSE_MIT` next to them (StyleGAN2 fused ops, Karras et al.).

The MIT header on the top-level `LICENSE` does **not** override the per-file
NVIDIA / Apache headers. **Practical implication**: SynDiff in this form is
usable for *research* only. Any future commercial re-distribution of a VENA
release that bundles SynDiff weights or the NVIDIA-licensed backbone code
needs a backbone replacement (or explicit NVIDIA grant) — flagged here so a
future maintainer does not miss it.

## Scope used in VENA

We import (via runtime `sys.path` injection into `upstream/`):

- `backbones.ncsnpp_generator_adagn.NCSNpp` — diffusive generator U-Net.
- `backbones.generator_resnet.define_G(netG='resnet_6blocks', ...)` — non-diffusive
  ResNet generators (one per direction).
- `backbones.generator_resnet.define_D(...)` — PatchGAN discriminators for the
  cycle path (one per direction).
- `backbones.discriminator.Discriminator_large` — time-conditional discriminator
  used by the diffusive path.
- `utils.EMA.EMA` — optimiser-wrapper EMA shadow (patched, see `PATCHES.md`).

We do **not** invoke:

- `train.py` — replaced by `src/vena/competitors/syndiff/runner.py`, which
  reimplements the training loop without DDP, without `DistributedSampler`,
  without `set_detect_anomaly(True)`, and with VENA-style CSV logging,
  preflight gating, and `decision.json` accounting.
- `test.py` — replaced by `src/vena/competitors/syndiff/inference.py`, which
  reuses `sample_from_model` (re-implemented locally to avoid the TF import
  chain in `utils/utils.py`) and writes VENA-style NIfTI / PNG / metrics.
- `dataset.py` — the upstream contract is a pre-sliced `.mat` v7.3 cache. VENA
  reads cohort H5s directly via `SynDiffSliceDataset` (mirrors the
  `TensorDataset` contract: 2-tuple `(src_slice, tgt_slice)` in `[-1,1]`).
- `utils/utils.py` — has a hard `import tensorflow as tf` for
  `restore_checkpoint`. We bypass it entirely (load with `torch.load`).

## Code vs paper coherency (per skill sub-step 3.7)

| Axis | Paper text | Released code | Load-bearing? | Direction of advantage |
|---|---|---|---|---|
| Epochs | 50 | README example: 500; `--num_epoch` default 1200 | **Yes** — affects compute and quality | Code/README inflates training budget; paper schedule is shorter. We follow paper (50 epochs) per VENA policy 2026-06-15, with `--num_epoch 50` in `picasso_full_*.yaml`. |
| Cycle weight λ₁_φ = λ₁_θ | 0.5 (Eq. 22) | `--lambda_l1_loss` default 0.5; train.py uses single `lambda_l1_loss` for both cycle and L1 terms (`errG = lambda * errG_cycle + ... + lambda * errG_L1`) | No (same value) | Paper and code agree at 0.5; the code merges the φ/θ split into one knob. |
| Module weight λ₂_φ = λ₂_θ | 1.0 (Eq. 23) | Implicit (no separate flag — adv terms enter with coefficient 1) | No | OK. |
| Grad-penalty weight η | "validated via cross-validation" — value omitted from paper | `--r1_gamma` default 0.05; README example uses `--r1_gamma 1.0` | Maybe | The README example value `1.0` is what we use, matching the canonical training invocation. |
| Lazy regularisation | Not described | `--lazy_reg 10` in README; applies grad-penalty every 10 steps | No | Optimisation detail; follow code. |
| Discriminator | "6 blocks, 2 conv layers + 2× downsampling each" (paper §III) | `Discriminator_large` matches; `Discriminator_small` is a smaller variant unused in training | No | Code defines two, train uses `_large` — match. |
| ResNet generator | "3 encoding, 6 residual, 3 decoding blocks" (paper §III) | `define_G(netG='resnet_6blocks')` — matches | No | OK. |
| `args.num_channels` mutation | Not described | `train.py:245` sets `args.num_channels = 1` after NCSNpp construction to make ResNet `define_G` use 1-channel | No | Code-side hack with no paper analogue. We avoid it by passing `input_nc=1, output_nc=1` explicitly to `define_G` in our runner. |
| `set_detect_anomaly(True)` | Not described | Hard-coded in `train.py:582` (upstream issue #43) | No | Slows training, triggers NaN crashes. We do not replicate this in our runner. |

No load-bearing-YES rows require the user to override the policy. Recorded for
the audit trail.

## Notes for future patches

If the StyleGAN2 fused ops (`utils/op/upfirdn2d.cpp`, `fused_bias_act.cpp`) fail
to compile against a future CUDA toolkit, the path of least resistance is to
monkey-patch `utils/op/__init__.py` to use the pure-PyTorch reference
implementations bundled at `utils/op/upfirdn2d.py` (function
`upfirdn2d_native`) — slower but builds nowhere. Documented contingency in
`PATCHES.md`; not enabled by default.
