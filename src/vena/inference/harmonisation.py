"""§4.1 intensity-harmonisation pipeline shared by every adapter.

The validation protocol §4.1 prescribes a per-volume percentile
normalisation over the brain foreground so the predicted T1c lives in
the same intensity range as the encoder's input. The exact contract
matches the encoder side (per ``.claude/rules/model-coding-standards.md``
rule 15): ``percentile_normalise(lower=0.0, upper=99.5,
foreground_only=True)``. Background voxels outside the brain mask are
forced to zero — the §5.3 schema requires the harmonised prediction's
range to be ``⊆ [0, 1]`` *inside* the brain mask with the exterior
forced to 0.

Note on the §4.1 vs rule-15 ``lower`` discrepancy
-------------------------------------------------
The validation proposal text reads ``lower=0.5`` while the encoder
training code (``model-coding-standards.md`` rule 15 + the encoder
preprocessing in ``vena.model.autoencoder.maisi.preprocessing``) uses
``lower=0.0``. We use ``lower=0.0`` so that the decoded prediction and
the reference live in the *same* intensity space the encoder was
calibrated against — that is the actual rule-15 contract the §4.1 prose
intends to cite.

The recipe string written into the predictions H5 attrs reflects what
this function actually does, so a future reader of the H5 can verify
exactly which percentile range was applied.
"""

from __future__ import annotations

import torch

from vena.common import percentile_normalise

# What lands in the H5 attrs (validation §5.3) so the exact harmonisation
# is recoverable from the file alone.
HARMONISATION_RECIPE: str = "percentile_normalise(lower=0.0, upper=99.5, foreground_only=True)"


def apply_harmonisation(
    volume: torch.Tensor,
    brain_mask: torch.Tensor | None = None,
    *,
    lower: float = 0.0,
    upper: float = 99.5,
    foreground_only: bool = True,
) -> torch.Tensor:
    """Apply the §4.1 percentile harmonisation, then zero the brain exterior.

    Parameters
    ----------
    volume
        Predicted T1c of shape ``(H, W, D)`` or ``(1, 1, H, W, D)``,
        any float dtype. The function accepts both because adapters
        sometimes have one or the other in hand and packing/unpacking
        the 5-D shape is the most error-prone step in the call site.
    brain_mask
        Optional binary mask of shape matching ``volume`` (after the
        squeeze to ``(H, W, D)``). When ``None`` the function infers the
        brain from ``volume > 0`` — the same fallback the exhaustive-val
        engine uses. When supplied, the mask is the cohort's
        ``masks/brain`` (HD-BET / CBICA), per validation §4.1 step 1.
    lower, upper, foreground_only
        Forwarded to :func:`vena.common.percentile_normalise`. Defaults
        match the encoder side (rule 15).

    Returns
    -------
    torch.Tensor
        Shape ``(H, W, D)`` float32, range ``[0, 1]`` inside the brain
        mask, exterior forced to 0. Contiguous.
    """
    vol = volume.detach()
    # Always work in 5-D for percentile_normalise; un-squeeze at the end.
    if vol.dim() == 3:
        vol5 = vol[None, None]
    elif vol.dim() == 5:
        if vol.shape[0] != 1 or vol.shape[1] != 1:
            raise ValueError(
                f"apply_harmonisation expects (1, 1, H, W, D) for 5-D input, got {tuple(vol.shape)}"
            )
        vol5 = vol
    else:
        raise ValueError(
            f"apply_harmonisation expects 3-D (H,W,D) or 5-D (1,1,H,W,D), "
            f"got {vol.dim()}-D {tuple(vol.shape)}"
        )

    normed = percentile_normalise(
        vol5.float(),
        lower=lower,
        upper=upper,
        foreground_only=foreground_only,
    )
    out = normed[0, 0]

    if brain_mask is not None:
        mask = brain_mask
        if mask.dim() == 5:
            mask = mask[0, 0]
        if mask.shape != out.shape:
            raise ValueError(
                f"apply_harmonisation: brain_mask shape {tuple(mask.shape)} "
                f"!= volume shape {tuple(out.shape)}"
            )
        out = out * mask.to(out.device, dtype=out.dtype)

    return out.contiguous()


__all__ = ["HARMONISATION_RECIPE", "apply_harmonisation"]
