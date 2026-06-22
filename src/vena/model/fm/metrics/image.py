"""Region-masked image-space metrics: 3D PSNR + 3D SSIM via MONAI.

We wrap :class:`monai.metrics.PSNRMetric` and :class:`monai.metrics.SSIMMetric`
so VENA's region-masked PSNR/SSIM has the same call signature as
:class:`LatentMetrics`. Each call computes one scalar per batch element.

Masking strategy: for PSNR we restrict the per-voxel ``(pred - target)^2`` to
the region (and adjust the mean-square denominator). MONAI's
:class:`PSNRMetric` is not mask-aware so we implement the masked PSNR
manually with the same fixed ``max_val``. For SSIM we set out-of-region voxels
to a *neutral* fill (the per-volume mean of the in-region voxels) so the
sliding 3D window contains representative content; this is an approximation
acceptable for training-time tracking — final metrics use a dedicated harness.

S1 v3 (2026-06-22) adds masked **MAE** and **MSE** (linear and squared
error means inside the region) plus a one-shot ``metrics_per_region`` driver
that returns ``{region: {psnr_db, ssim, mae, mse}}`` for every region in
``RegionMasks`` that is not ``None``. This is the load-bearing API for the
exhaustive_val CSV: per-(cohort, epoch, patient, nfe) row carries all four
metrics × seven regions in one call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_PER_REGION_DRIVER_REGIONS: tuple[str, ...] = (
    "brain",  # "whole" alias — the full brain foreground
    "wt",
    "bg",
    "brain_not_wt",
    "netc",
    "ed",
    "et",
)


class ImageMetrics:
    """Stateful: holds the MONAI metric instances (window state)."""

    def __init__(self, data_range: float = 1.0, ssim_window_size: int = 7) -> None:
        self.data_range = float(data_range)
        self.ssim_window_size = int(ssim_window_size)
        # MONAI metrics are imported lazily so test environments without
        # MONAI-on-import still load the module.
        from monai.metrics import SSIMMetric

        self._ssim = SSIMMetric(
            spatial_dims=3,
            data_range=self.data_range,
            win_size=self.ssim_window_size,
            reduction="none",
        )

    @staticmethod
    def _masked_psnr_3d(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        data_range: float,
    ) -> torch.Tensor:
        """Region-masked PSNR. Returns ``(B,)``.

        Implementation: masked-MSE over the supplied region, then
        ``10 * log10(data_range^2 / mse)``. Voxels with zero in-mask count
        return ``nan``.
        """
        diff_sq = (pred - target) ** 2
        m = mask.expand_as(diff_sq).to(diff_sq.dtype)
        num = (diff_sq * m).flatten(1).sum(dim=1)
        den = m.flatten(1).sum(dim=1)
        mse = torch.where(
            den > 0,
            num / den.clamp_min(1.0),
            torch.full_like(num, float("nan")),
        )
        psnr = 10.0 * torch.log10(
            (data_range * data_range) / mse.clamp_min(torch.finfo(mse.dtype).eps)
        )
        return psnr

    def psnr(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked PSNR-3D per batch element."""
        return self._masked_psnr_3d(pred, target, mask, self.data_range)

    def ssim(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Approximate masked SSIM-3D — fills out-of-region with mean intensity.

        Returns ``(B,)``; entries are NaN if the region is empty.
        """
        m = mask.expand_as(pred).to(pred.dtype)
        # Per-volume in-region mean. Voxels outside are set to this mean so the
        # SSIM sliding window is not biased by hard zeros at boundaries.
        denom = m.flatten(1).sum(dim=1).clamp_min(1.0)
        pred_mean = (pred * m).flatten(1).sum(dim=1) / denom
        target_mean = (target * m).flatten(1).sum(dim=1) / denom
        pred_filled = pred * m + pred_mean[:, None, None, None, None] * (1 - m)
        target_filled = target * m + target_mean[:, None, None, None, None] * (1 - m)

        # MONAI SSIMMetric expects (B, C, H, W, D) tensors and reduces over
        # spatial dims. Reset internal state each call to get a per-batch
        # scalar (we use the same data_range for every batch).
        self._ssim.reset()
        scores = self._ssim(y_pred=pred_filled, y=target_filled)
        # scores shape: (B, 1) — squeeze and replace empty-region with NaN.
        scores = scores.squeeze(-1)
        empty = m.flatten(1).sum(dim=1) == 0
        scores = torch.where(empty, torch.full_like(scores, float("nan")), scores)
        return scores

    @staticmethod
    def _masked_mae_3d(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean absolute error inside ``mask``. NaN for empty regions.

        Returns ``(B,)``. Matches the masked-PSNR conventions: voxels
        outside the mask are excluded from both the numerator and the
        denominator.
        """
        m = mask.expand_as(pred).to(pred.dtype)
        num = ((pred - target).abs() * m).flatten(1).sum(dim=1)
        den = m.flatten(1).sum(dim=1)
        return torch.where(
            den > 0,
            num / den.clamp_min(1.0),
            torch.full_like(num, float("nan")),
        )

    @staticmethod
    def _masked_mse_3d(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error inside ``mask``. NaN for empty regions."""
        m = mask.expand_as(pred).to(pred.dtype)
        num = (((pred - target) ** 2) * m).flatten(1).sum(dim=1)
        den = m.flatten(1).sum(dim=1)
        return torch.where(
            den > 0,
            num / den.clamp_min(1.0),
            torch.full_like(num, float("nan")),
        )

    def mae(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked MAE-3D per batch element. Returns ``(B,)`` (NaN if region empty)."""
        return self._masked_mae_3d(pred, target, mask)

    def mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked MSE-3D per batch element. Returns ``(B,)`` (NaN if region empty)."""
        return self._masked_mse_3d(pred, target, mask)

    def metrics_per_region(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        region_masks: RegionMasks,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Compute PSNR + SSIM + MAE + MSE for every populated region.

        Iterates over ``("brain", "wt", "bg", "brain_not_wt", "netc", "ed",
        "et")`` (skipping any field that is ``None``) and returns a dict
        ``{region: {psnr_db, ssim, mae, mse}}`` of per-batch tensors.

        The ``"brain"`` field is aliased to ``"whole"`` in the returned dict
        — it represents the full-brain foreground (matches the "whole-volume"
        metric naming convention used by the exhaustive_val output). This
        avoids double work for the most common metric and keeps the
        downstream CSV column scheme stable.

        Parameters
        ----------
        pred, target : Tensor
            ``(B, 1, H, W, D)`` decoded volumes in the same intensity space
            (typically VAE-decoded ``[0, 1]`` brain volumes).
        region_masks : RegionMasks
            Output of :class:`RegionResolver.resolve`. ``None`` fields are
            silently skipped — the caller is responsible for emitting NaN
            columns when a region is unavailable for a given batch.

        Returns
        -------
        dict[region_name → dict[metric_name → Tensor of shape (B,)]]
            Region names: ``{whole, wt, bg, brain_not_wt, netc, ed, et}``.
            Metric names: ``{psnr_db, ssim, mae, mse}``.
        """
        out: dict[str, dict[str, torch.Tensor]] = {}
        for region_name in _PER_REGION_DRIVER_REGIONS:
            mask = region_masks.get(region_name)
            if mask is None:
                continue
            key = "whole" if region_name == "brain" else region_name
            out[key] = {
                "psnr_db": self.psnr(pred, target, mask),
                "ssim": self.ssim(pred, target, mask),
                "mae": self.mae(pred, target, mask),
                "mse": self.mse(pred, target, mask),
            }
        return out
