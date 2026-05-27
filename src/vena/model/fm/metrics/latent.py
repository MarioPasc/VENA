"""Region-masked latent metrics: MSE, L1, cosine similarity.

Each metric reduces over channel + spatial dims inside the supplied region
mask, producing one scalar per batch element. Aggregation across patients is
handled at a higher level (:class:`MetricAccumulator` in the val callback).
"""

from __future__ import annotations

import torch


def _masked_reduce(
    diff: torch.Tensor, mask: torch.Tensor, op: str
) -> torch.Tensor:
    """Reduce ``diff`` inside the boolean mask. Returns ``(B,)``.

    Parameters
    ----------
    diff : Tensor
        ``(B, C, H, W, D)`` per-voxel quantity to reduce.
    mask : Tensor
        ``(B, 1, H, W, D)`` boolean mask, broadcast over channels.
    op : {"mean_abs", "mean_sq"}
        ``"mean_abs"`` → L1; ``"mean_sq"`` → MSE.
    """
    if op == "mean_abs":
        v = diff.abs()
    elif op == "mean_sq":
        v = diff * diff
    else:
        raise ValueError(f"unknown op {op!r}")
    m = mask.expand_as(v).to(v.dtype)
    num = (v * m).flatten(1).sum(dim=1)
    den = m.flatten(1).sum(dim=1).clamp_min(1.0)
    return num / den


class LatentMetrics:
    """Stateless: applies the masked metric on a (pred, target) pair."""

    @staticmethod
    def mse(
        z_pred: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return _masked_reduce(z_pred - z_target, mask, op="mean_sq")

    @staticmethod
    def l1(
        z_pred: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return _masked_reduce(z_pred - z_target, mask, op="mean_abs")

    @staticmethod
    def cosine(
        z_pred: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity of the masked-flattened vectors per batch element.

        Returns ``(B,)``.
        """
        m = mask.expand_as(z_pred).to(z_pred.dtype)
        a = (z_pred * m).flatten(1)
        b = (z_target * m).flatten(1)
        eps = torch.finfo(a.dtype).eps
        num = (a * b).sum(dim=1)
        den = a.norm(dim=1).clamp_min(eps) * b.norm(dim=1).clamp_min(eps)
        return num / den
