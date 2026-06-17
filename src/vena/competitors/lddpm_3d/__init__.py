"""3D-LDDPM competitor wrapper (Eidex *et al.* 2025, arXiv:2509.24194).

3D latent denoising diffusion probabilistic model (Ho *et al.* 2020) for
T1c synthesis from T1pre + FLAIR latents, mirroring the reference
implementation at ``src/external/lddpm_3d/upstream/train_ddpm.py``
(SHA ``fc8314f6``). Uses the **same U-Net backbone** as the T1C-RFlow
wrapper (paper-faithful: ``[128, 128, 256]``, 3 levels, 2 res-blocks, no
attention) — isolating the **scheduler and loss** (DDPM-eps L2 vs
RFlow-velocity L1) as the only competitor-internal delta vs T1C-RFlow.

Public API:

- ``LDDPM3DLatentDataset`` — per-cohort deterministic latent reader.
- ``MultiCohortLDDPM3DLatentDataset`` — concat over a VENA corpus registry.
- ``train_lddpm_3d`` — programmatic training entrypoint.
- ``run_inference`` — DDPM K-step denoising + VAE decode + per-patient PSNR/SSIM.

Citation
--------
Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models."
*NeurIPS 2020*. arXiv:2006.11239.

Eidex, Z., Safari, M., Ding, J., Qiu, R., Roper, J., Yu, D., Shu, H.-K.,
Tian, Z., Mao, H., Yang, X. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." *arXiv preprint* arXiv:2509.24194,
2025.
"""

from __future__ import annotations

from .dataset import (
    DatasetError,
    LDDPM3DLatentDataset,
    MultiCohortLDDPM3DLatentDataset,
)
from .inference import InferenceError, run_inference
from .runner import LDDPM3DRunnerError, train_lddpm_3d

__all__ = [
    "DatasetError",
    "InferenceError",
    "LDDPM3DLatentDataset",
    "LDDPM3DRunnerError",
    "MultiCohortLDDPM3DLatentDataset",
    "run_inference",
    "train_lddpm_3d",
]
