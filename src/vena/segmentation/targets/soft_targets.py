"""Soft target generation: hard BraTS labels → soft [WT, NETC] via SDT → sigmoid.

Pipeline (per class):
1. ``harmonise_labels`` — integer label map → boolean WT and NETC masks.
2. ``signed_distance`` — per-class signed distance transform (SDT > 0 inside,
   < 0 outside, ≈ 0 at boundary, clipped to ±clip_vox).
3. ``sigmoid(SDT / sigma_vox)`` — soft probability in [0, 1]; 0.5 at the
   boundary, ~0.95 at 3σ inside, ~0.05 at 3σ outside.
4. Nesting enforcement: ``NETC_soft ≤ WT_soft`` elementwise (NETC ⊆ WT by
   anatomy; clamp after softening as belt-and-suspenders for edge cases).
5. Return float32 ``(2, H, W, D)`` with channel 0 = WT, channel 1 = NETC.

Design authority: segmenter-conditioning design §B.c step 1, §B.d, §B.f-4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vena.segmentation.exceptions import SegTargetError
from vena.segmentation.targets.harmonise import harmonise_labels
from vena.segmentation.targets.sdt import signed_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vena.segmentation.config import TargetConfig

__all__ = ["make_soft_targets", "soft_target"]

_LOG2 = float(np.log(2.0))  # sigmoid(0) = 0.5 by definition; kept for documentation


def _sigmoid(x: NDArray) -> NDArray:
    """Numerically stable sigmoid in float64, returned as float32.

    Uses the exp-of-negative formulation which avoids overflow for large
    positive inputs (exp(-large) → 0, result → 1).
    """
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def soft_target(
    mask: NDArray,
    *,
    sigma_vox: float,
    mode: str,
    image: NDArray | None = None,
    clip_vox: float,
) -> NDArray:
    """Convert a boolean mask to a soft probability map via SDT → sigmoid.

    Parameters
    ----------
    mask : NDArray
        Boolean array of arbitrary shape.
    sigma_vox : float
        Gaussian sigma controlling how sharply the soft boundary transitions.
        ``sigmoid(SDT / sigma_vox)`` is ≈ 0.5 at the boundary, ≈ 0.95 at
        3σ inside, ≈ 0.05 at 3σ outside.
    mode : str
        SDT operator: ``"euclidean_percomponent"`` or ``"geodesic"``.
    image : NDArray or None
        Intensity array same shape as *mask*.  Required for ``"geodesic"``
        mode; ignored otherwise.
    clip_vox : float
        Absolute SDT clipping radius in voxels, passed through to
        :func:`~vena.segmentation.targets.sdt.signed_distance`.

    Returns
    -------
    NDArray
        Float32 soft-probability array in ``[0, 1]``, same shape as *mask*.
        Value ≈ 0.5 on the boundary; increases monotonically inward.

    Raises
    ------
    SegTargetError
        If *sigma_vox* ≤ 0.
    """
    if sigma_vox <= 0.0:
        raise SegTargetError(f"sigma_vox must be positive, got {sigma_vox}")

    sdt = signed_distance(mask, mode=mode, image=image, clip_vox=clip_vox)
    # Epsilon-guard: sigma_vox > 0 is asserted above but guard floating arithmetic
    _eps = 1e-8
    return _sigmoid(sdt / (float(sigma_vox) + _eps))


def make_soft_targets(
    label: NDArray,
    cfg: TargetConfig,
    image: NDArray | None = None,
) -> NDArray:
    """Build a ``(2, H, W, D)`` soft-target tensor from a hard BraTS label map.

    Channel order: channel 0 = WT (whole-tumour), channel 1 = NETC (necrotic
    core).  Both channels are float32 in ``[0, 1]``.  Nesting is enforced
    elementwise: ``channel1 ≤ channel0``.

    Parameters
    ----------
    label : NDArray
        Integer label array of shape ``(H, W, D)``.  Accepted conventions:

        - BraTS-2021 (values in ``{0, 1, 2, 4}``): UCSF-PDGM, UPENN-GBM,
          IvyGAP, REMBRANDT.
        - BraTS-2023 (values in ``{0, 1, 2, 3}``): BraTS-GLI, BraTS-PED,
          BraTS-Africa, LUMIERE.

        Both map to the same boolean regions via code-agnostic rules.
    cfg : TargetConfig
        Pydantic model with fields ``soft``, ``sdt_sigma_vox``,
        ``netc_operator``, and ``clip_vox``.
    image : NDArray or None
        Intensity volume of shape ``(H, W, D)``.  Required when
        ``cfg.netc_operator == "geodesic"``.

    Returns
    -------
    NDArray
        Float32 array of shape ``(2, H, W, D)``: channel 0 = WT soft mask,
        channel 1 = NETC soft mask, with ``channel1 ≤ channel0`` guaranteed.

    Raises
    ------
    SegTargetError
        If ``cfg.soft is False``, if *label* is not 3-D, or if *image* is
        required by ``cfg.netc_operator`` but not provided.
    """
    if not cfg.soft:
        raise SegTargetError(
            "make_soft_targets requires cfg.soft=True; hard targets are not "
            "produced by this function."
        )
    if label.ndim != 3:
        raise SegTargetError(f"label must be 3-D (H, W, D), got shape {label.shape}")

    regions = harmonise_labels(label)
    wt_mask: NDArray = regions["wt"]
    netc_mask: NDArray = regions["netc"]

    # WT: typically a single connected region; per-component is still correct
    wt_soft = soft_target(
        wt_mask,
        sigma_vox=cfg.sdt_sigma_vox,
        mode="euclidean_percomponent",
        image=None,
        clip_vox=cfg.clip_vox,
    )

    # NETC: multifocal lesions → use cfg.netc_operator for correct per-lesion SDT
    netc_soft = soft_target(
        netc_mask,
        sigma_vox=cfg.sdt_sigma_vox,
        mode=cfg.netc_operator,
        image=image,
        clip_vox=cfg.clip_vox,
    )

    # Enforce anatomical nesting: NETC ⊆ WT (necrotic core is inside whole-tumour)
    netc_soft = np.minimum(netc_soft, wt_soft)

    return np.stack([wt_soft, netc_soft], axis=0).astype(np.float32)
