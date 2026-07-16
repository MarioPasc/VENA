"""Phase-2 region mask computation.

Reuses the binary dilation semantics from ``vena.model.fm.metrics.regions``
(``F.max_pool3d(x.float(), kernel_size=k, stride=1, padding=k//2) > 0.5``).
Does **not** reuse ``RegionResolver`` — that class is bound to the training
batch dict.

Returned keys
-------------
``"brain"``
    Full brain mask (copy of input, cast to bool).
``"wt"``
    Whole-tumour mask, undilated.
``"wt_dilated"``
    WT after morphological dilation (kernel_size=dilate_k, exact binary
    expansion — see §11 trap #8 in SHARED_CONTRACTS).
``"bg"``
    Safe background: ``brain AND NOT wt_dilated``.  Used by §4.3 spatial
    residual where the tumour *margin* must be excluded.
``"bg_undilated"``
    Exact anatomical non-tumour brain region: ``brain AND NOT wt``.
    Used by §4.2 paired fidelity for the healthy-tissue endpoint.
    Distinct from ``"bg"`` — conflating them is trap #8.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def region_masks(
    brain: np.ndarray,
    wt: np.ndarray,
    *,
    dilate_k: int = 5,
) -> dict[str, np.ndarray]:
    """Compute region masks for Phase-2 region-aware metrics.

    The dilation is ``max_pool3d(kernel_size=dilate_k, stride=1,
    padding=dilate_k//2) > 0.5``.  For ``dilate_k=5`` this is exactly a
    radius-2 dilation (a 5×5×5 structuring element centred on each foreground
    voxel).

    Parameters
    ----------
    brain :
        ``(H, W, D)`` bool brain mask.
    wt :
        ``(H, W, D)`` bool whole-tumour mask.
    dilate_k :
        Dilation kernel size (odd integer).  YAML-configurable so every
        run is auditable.  Default is 5 (radius 2) per contracts §11-8.

    Returns
    -------
    dict[str, ndarray]
        Keys: ``"brain"``, ``"wt"``, ``"wt_dilated"``, ``"bg"``,
        ``"bg_undilated"``.
    """
    # Binary dilation via max_pool3d — identical to the training path.
    k = int(dilate_k)
    pad = k // 2
    wt_t = torch.from_numpy(wt.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    dilated = F.max_pool3d(wt_t, kernel_size=k, stride=1, padding=pad)
    wt_dilated = (dilated.squeeze().numpy() > 0.5).astype(bool)

    brain_bool = brain.astype(bool)
    bg = brain_bool & ~wt_dilated  # exclude dilated tumour margin
    bg_undilated = brain_bool & ~wt.astype(bool)  # exact non-tumour region

    return {
        "brain": brain_bool,
        "wt": wt.astype(bool),
        "wt_dilated": wt_dilated,
        "bg": bg,
        "bg_undilated": bg_undilated,
    }
