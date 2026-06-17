"""3D-Latent-Pix2Pix competitor wrapper (Isola 2017 Pix2Pix recipe over MAISI-V2 latents).

Generator: ``DiffusionModelUNetMaisi`` (same MONAI primitive as T1C-RFlow),
wrapped to feed **zero timesteps** so the diffusion U-Net runs as a
deterministic conditional generator. Discriminator: 3D PatchGAN (4 strided
conv layers, ndf=64). Loss: BCE adversarial + λ·L1 with λ=100 (Isola
*et al.* 2017 §3.2). No noise injection, no scheduler — one forward pass
maps ``[z_T1pre, z_FLAIR]`` to ``z_T1c``.

This wraps the "Pix2pix" baseline used in Eidex *et al.* 2025 §4 (Table 2),
adapted to VENA's MAISI-V2 latent grid ``(4, 48, 56, 48)``. Compared to
T1C-RFlow and 3D-DiT (which use the same backbone / data but a flow-matching
training paradigm), this competitor isolates the **training paradigm** axis
(GAN vs flow-matching) while keeping backbone, conditioning, and data
contract identical.

Public API:

- ``Pix2PixLatentDataset`` — per-cohort deterministic latent reader.
- ``MultiCohortPix2PixLatentDataset`` — concat over a VENA corpus registry.
- ``train_lpix2pix_3d`` — programmatic training entrypoint.
- ``run_inference`` — single forward pass + VAE decode + per-patient PSNR/SSIM.

Citation
--------
Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. "Image-to-Image Translation
with Conditional Adversarial Networks." *CVPR 2017*. arXiv:1611.07004.

Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." arXiv:2509.24194 — §4 "Pix2pix" baseline.
"""

from __future__ import annotations

from .dataset import (
    DatasetError,
    MultiCohortPix2PixLatentDataset,
    Pix2PixLatentDataset,
)
from .inference import InferenceError, run_inference
from .runner import Pix2PixRunnerError, train_lpix2pix_3d

__all__ = [
    "DatasetError",
    "InferenceError",
    "MultiCohortPix2PixLatentDataset",
    "Pix2PixLatentDataset",
    "Pix2PixRunnerError",
    "run_inference",
    "train_lpix2pix_3d",
]
