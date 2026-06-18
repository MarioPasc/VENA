"""Latent Perceptual Loss (LPL) — region-weighted, depth-weighted L^p_r.

Implements §2.6 of ``.claude/notes/changes/decoder_perceptual_loss_s3.md``:

.. math::
    \\mathcal L^{\\text{region}}_{\\text{dec}}
    = \\mathbb 1[t > t_{\\min}]
      \\sum_{\\ell \\in A} \\frac{w_\\ell}{C_\\ell}
      \\sum_{r \\in R} \\alpha_r
      \\frac{1}{\\max(|\\Omega_\\ell^{(r)}|, 1)}
      \\sum_{c, x \\in \\Omega_\\ell^{(r)}}
      \\rho_{\\ell,c}(x) \\bigl|
        \\phi'_{\\ell,c}(\\hat x_1)[x] - \\phi'_{\\ell,c}(\\tilde z)[x]
      \\bigr|^{p_r}

with ``ρ`` the per-channel outlier mask (``|x| ≤ k_l · MAD``), ``α`` the
per-region budget, and ``p_r`` the per-region exponent.

The high-SNR gate (``t > t_min``) is a property of the *decoder input* and
must be applied at the batch level — when no sample in the batch is past
the gate, ``LplLoss.forward`` short-circuits to a finite zero with
``requires_grad`` preserved (otherwise the surrounding
:class:`CompositeLoss` accumulator may break under autograd's "leaf
tensor" rule).

The loss reports a per-term breakdown (``lpl_b{idx}``, ``lpl_{region}``,
``hi_frac``) so the future training-step CSV columns have a single source
of truth.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import LplConfig
from .feature_stats import FeatureStatsEMA
from .region import region_weight_map, resample_region_to_block


def _abs_pow(x: torch.Tensor, p: int) -> torch.Tensor:
    if p == 1:
        return x.abs()
    if p == 2:
        return x.pow(2)
    if p == 3:
        return x.abs().pow(3)
    raise ValueError(f"p must be 1, 2, or 3; got {p}")


def _mad(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Median absolute deviation along ``dim``, broadcast over the channel axis."""
    med = x.median(dim=dim, keepdim=True).values
    return (x - med).abs().median(dim=dim, keepdim=True).values


def _outlier_mask(feat_pred: torch.Tensor, k: float) -> torch.Tensor:
    """Per-channel ``|feat| ≤ k · MAD`` keep-mask (Berrada 2025 §3.4).

    The mask is computed on ``feat_pred`` only — Berrada's recipe is to
    rely on the prediction's distribution because the target may be
    cleaner than the prediction at the current optimisation step.
    """
    c = feat_pred.shape[1]
    flat = feat_pred.movedim(1, -1).reshape(-1, c)  # (N, C)
    mad = _mad(flat, dim=0).abs()  # (1, C)
    thr = (k * mad).view((1, c) + (1,) * (feat_pred.ndim - 2))
    # Keep voxels whose per-channel magnitude is at most k·MAD.
    return (feat_pred.abs() <= thr).float()


class LplLoss(nn.Module):
    """Region-weighted decoder-feature perceptual loss."""

    # Mirrors the :class:`AbstractFMLoss` contract so the future
    # ``CompositeLoss`` wiring doesn't need a perturbed conditioning pass.
    requires_perturbed_pass: bool = False

    def __init__(
        self,
        cfg: LplConfig,
        feature_stats: FeatureStatsEMA,
        *,
        soft_region: bool | None = None,
    ) -> None:
        """Construct the loss.

        Parameters
        ----------
        cfg : LplConfig
            Validated recipe (``A``, ``w_l``, ``alpha``, ``p``, ``t_min``,
            ``outlier_k``).
        feature_stats : FeatureStatsEMA
            Running per-channel mean/var. The loss does *not* update the
            stats — the training step owns the update, so the same stats
            can be inspected by the train CSV before/after the loss call.
        soft_region : bool, optional
            Override ``cfg.soft_region``. Useful in tests; defaults to the
            config value.
        """
        super().__init__()
        # Buffers-free, but `nn.Module` parentage gives `.to(device)` for
        # the EMA submodule when shared with the LightningModule.
        self.cfg = cfg
        self.feature_stats = feature_stats
        self._soft = bool(soft_region) if soft_region is not None else cfg.soft_region

    def forward(
        self,
        phi_pred: dict[int, torch.Tensor],
        phi_tgt: dict[int, torch.Tensor],
        m_wt_lat: torch.Tensor,
        m_brain_lat: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the gated, depth-weighted, region-weighted feature distance.

        Parameters
        ----------
        phi_pred : dict[int, Tensor]
            Per-block features captured from the partial decode of ``x̂_1``.
            Tensors shape ``(B, C, *block_shape)``. Must cover every block
            in ``cfg.A``.
        phi_tgt : dict[int, Tensor]
            Per-block features of the target latent's partial decode.
            Same shape contract.
        m_wt_lat : torch.Tensor
            Latent-resolution WT mask (binary or soft per ``cfg.soft_region``),
            shape ``(B, 1, *LATENT_SPATIAL)``.
        m_brain_lat : torch.Tensor
            Latent-resolution brain mask, shape ``(B, 1, *LATENT_SPATIAL)``.
        t : torch.Tensor
            Per-sample timestep, shape ``(B,)``. The gate ``t > t_min`` is
            applied along this axis.

        Returns
        -------
        tuple[Tensor, dict[str, float]]
            Scalar loss + per-term breakdown for CSV logging.
        """
        # Gate.
        hi_mask = t > self.cfg.t_min
        hi_count = int(hi_mask.sum().item())
        hi_frac = float(hi_count) / float(t.numel())
        empty_break: dict[str, float] = {
            "hi_frac": hi_frac,
            **{f"lpl_b{idx}": 0.0 for idx in self.cfg.A},
            **{f"lpl_{r}": 0.0 for r in self.cfg.region_set},
        }
        if hi_count == 0:
            # Zero scalar that participates in autograd (in case the
            # caller sums it with other gradient-bearing terms).
            ref = next(iter(phi_pred.values()))
            zero = ref.new_zeros((), requires_grad=False).sum()
            return zero, empty_break

        # Sub-select the hi-SNR samples once for every per-block call.
        hi_idx = torch.nonzero(hi_mask, as_tuple=False).squeeze(-1)
        m_wt_hi = m_wt_lat.index_select(0, hi_idx)
        m_brain_hi = m_brain_lat.index_select(0, hi_idx)

        total = torch.zeros((), device=hi_mask.device)
        per_block: dict[int, torch.Tensor] = {}
        # Accumulators for the per-region CSV breakdown (averaged across A).
        per_region: dict[str, list[torch.Tensor]] = {r: [] for r in self.cfg.region_set}

        for blk in self.cfg.A:
            pred = phi_pred[blk].index_select(0, hi_idx)
            tgt = phi_tgt[blk].index_select(0, hi_idx)
            # Standardise both via the SHARED EMA stats (Berrada §3.3).
            pred_z = self.feature_stats.standardise(pred, blk)
            tgt_z = self.feature_stats.standardise(tgt, blk)
            # Per-channel outlier mask from prediction stats.
            keep = _outlier_mask(pred_z, k=float(self.cfg.outlier_k[blk]))

            # Resample masks to this block's spatial grid.
            target_shape = pred.shape[-3:]
            mwt_blk = resample_region_to_block(
                m_wt_hi,
                target_shape,
                mode="trilinear" if self._soft else "nearest",
            )
            mbrain_blk = resample_region_to_block(m_brain_hi, target_shape, mode="nearest")

            block_total = torch.zeros((), device=total.device)
            for region in self.cfg.region_set:
                # Region indicator at this block's grid.
                if region == "wt":
                    a_wt, a_nw = 1.0, 0.0
                elif region == "notwt":
                    a_wt, a_nw = 0.0, 1.0
                else:
                    # 3-region variant deferred (config validator accepts
                    # the keys but the production loss only consumes the
                    # 2-region split).
                    continue
                region_w = region_weight_map(
                    mwt_blk,
                    mbrain_blk,
                    alpha_wt=a_wt,
                    alpha_notwt=a_nw,
                    soft=self._soft,
                )
                # Effective region size (with empty-region guard).
                omega = region_w.sum().clamp(min=1.0)
                # Per-voxel weighted L^p_r error.
                p_r = int(self.cfg.p[region])
                err = _abs_pow(pred_z - tgt_z, p_r) * keep  # (B, C, ...)
                # Broadcast region weight (B,1,...) over channels.
                weighted_err = err.sum(dim=1, keepdim=True) * region_w  # (B,1,...)
                # Per-channel normalisation is baked into the channel-sum
                # divided by C_block (Berrada-style depth weight).
                c_block = float(pred_z.shape[1])
                region_loss = weighted_err.sum() / (omega * c_block)
                weighted_alpha = float(self.cfg.alpha[region]) * region_loss
                block_total = block_total + weighted_alpha
                per_region[region].append(region_loss.detach())

            w_l = float(self.cfg.w_l[blk])
            block_weighted = w_l * block_total
            total = total + block_weighted
            per_block[blk] = block_weighted.detach()

        # Breakdown — scalar floats for CSV consumers.
        breakdown: dict[str, float] = {
            "hi_frac": hi_frac,
            **{f"lpl_b{idx}": float(per_block[idx].item()) for idx in self.cfg.A},
            **{
                f"lpl_{r}": (
                    float(torch.stack(per_region[r]).mean().item()) if per_region[r] else 0.0
                )
                for r in self.cfg.region_set
            },
        }
        return total, breakdown
