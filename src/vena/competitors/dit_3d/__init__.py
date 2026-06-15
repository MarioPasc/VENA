"""3D-DiT competitor wrapper (DiT backbone over MAISI-V2 latents + R-Flow).

Backbone: DiT-3D (Peebles & Xie 2023 architecture, 3D adaptation per
Mo *et al.* NeurIPS 2023 and vendored at
``src/external/dit_3d/upstream/dit3d.py``). Scheduler: rectified flow
(``monai.networks.schedulers.rectified_flow.RFlowScheduler``) with the
paper-pinned kwargs — same as the T1C-RFlow wrapper, so the only axis
isolated against VENA is the **backbone** (DiT vs U-Net).

This is the transformer-backbone diffusion baseline that the FM/diffusion
literature uses as its standard reference, including Eidex *et al.* 2025 §4
("DiT-3D" in their Table 2).

Public API:

- ``DiT3DLatentDataset`` — per-cohort deterministic latent reader.
- ``MultiCohortDiT3DLatentDataset`` — concat over a VENA corpus registry.
- ``train_dit_3d`` — programmatic training entrypoint.
- ``run_inference`` — Euler integration + VAE decode + per-patient PSNR/SSIM.

Citation
--------
Peebles, W., & Xie, S. "Scalable Diffusion Models with Transformers."
*ICCV 2023*. arXiv:2212.09748.

Mo, S., Xie, E., Chu, R., Hong, L., Niessner, M., & Li, Z. "DiT-3D:
Exploring Plain Diffusion Transformers for 3D Shape Generation."
*NeurIPS 2023*. arXiv:2307.01831.

Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." arXiv:2509.24194 — §4 baseline.
"""

from __future__ import annotations

from .dataset import (
    DatasetError,
    DiT3DLatentDataset,
    MultiCohortDiT3DLatentDataset,
)
from .inference import InferenceError, run_inference
from .runner import DiT3DRunnerError, train_dit_3d

__all__ = [
    "DatasetError",
    "DiT3DLatentDataset",
    "DiT3DRunnerError",
    "InferenceError",
    "MultiCohortDiT3DLatentDataset",
    "run_inference",
    "train_dit_3d",
]
