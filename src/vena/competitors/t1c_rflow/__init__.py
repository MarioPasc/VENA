"""T1C-RFlow competitor wrapper (Eidex *et al.* 2025, arXiv:2509.24194).

3D latent rectified-flow synthesis of T1c from T1pre + FLAIR latents,
matching the reference implementation at
``src/external/t1c_rflow/upstream/`` (SHA ``fc8314f6``).

Public API:

- ``T1CRFlowLatentDataset`` — per-cohort deterministic latent reader.
- ``MultiCohortT1CRFlowLatentDataset`` — concat over a VENA corpus registry.
- ``train_t1c_rflow`` — programmatic training entrypoint.
- ``run_inference`` — Euler integration + VAE decode + per-patient PSNR/SSIM.

Citation
--------
Eidex, Z., Safari, M., Ding, J., Qiu, R., Roper, J., Yu, D., Shu, H.-K.,
Tian, Z., Mao, H., Yang, X. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." *arXiv preprint* arXiv:2509.24194,
2025.
"""

from __future__ import annotations

from .dataset import (
    DatasetError,
    MultiCohortT1CRFlowLatentDataset,
    T1CRFlowLatentDataset,
)
from .inference import InferenceError, run_inference
from .runner import T1CRFlowRunnerError, train_t1c_rflow

__all__ = [
    "DatasetError",
    "InferenceError",
    "MultiCohortT1CRFlowLatentDataset",
    "T1CRFlowLatentDataset",
    "T1CRFlowRunnerError",
    "run_inference",
    "train_t1c_rflow",
]
