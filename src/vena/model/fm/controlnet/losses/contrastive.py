r"""Lp-aware mask-perturbation contrastive regulariser (proposal §5.3, v0.3).

Given the velocity differential :math:`\Delta_\theta(x_t) = G_\theta(x_t,t,c_{\text{orig}})
- G_\theta(x_t,t,c_{\text{perturb}})` (the ControlNet trained on the original
conditioning vs the same tensor with the WT-mask channel zeroed), the loss has
two terms:

* **ROI**, pushing :math:`|\Delta|` *up* inside the tumour but capped on the
  aggregate mean to keep the term bounded:

  .. math::
      \mathcal{L}_{\text{roi}}^{(p_t)} =
        -\min\Big(\tfrac{1}{|m|}\!\sum_{x \in m} |\Delta_\theta(x)|^{p_t},
                  \;\delta^{p_t}\Big)

* **BG**, pushing :math:`|\Delta|` *down* outside the dilated tumour, with the
  cap applied **per-voxel** so single outliers saturate without dragging the
  regional mean to the cap:

  .. math::
      \mathcal{L}_{\text{bg}}^{(p_b)} =
        \tfrac{1}{|m^-|}\!\sum_{x \in m^-} \min\big(|\Delta_\theta(x)|^{p_b},
                                                   \;\delta^{p_b}\big)

The two terms are returned as a single weighted sum
:math:`\lambda_{\text{roi}}\mathcal{L}_{\text{roi}} + \lambda_{\text{bg}}\mathcal{L}_{\text{bg}}`;
the outer :math:`\lambda_{\text{contrast}}` is applied by
:class:`CompositeLoss` via the ``contrastive`` weight in the builder.

Notes
-----
* Operates entirely on latent-space velocity fields; no VAE decode.
* The cap :math:`\delta` (default 2.0) is the MAISI-v2 default — approximately
  2σ of KL-regularised MAISI latents.
* Empty-mask guard via ``clamp_min(1.0)`` on each region's voxel count so the
  term contributes 0 (not NaN) when the WT mask is empty for a batch element.
* The four diagnostics ``delta_abs_mean_in``, ``delta_abs_mean_out``,
  ``roi_cap_hit_frac``, ``bg_cap_hit_frac`` are exposed via :meth:`aux` so the
  composite forwards them as ``contrastive/<name>`` for CSV logging.

References
----------
Zhao et al. *MAISI-V2: Image-to-Image MR/CT Synthesis via Latent Flow Matching*,
AAAI 2026 (arXiv:2508.05772) §3.4.1 — the upstream version uses a single L1
contrastive with a window-ReLU mean cap and one combined weight. We generalise
to per-region :math:`L^p` and a per-voxel BG cap per proposal §5.3.
"""

from __future__ import annotations

import torch

from .base import AbstractFMLoss, LossInputs


class ContrastiveTumourLoss(AbstractFMLoss):
    """Lp-aware mask-perturbation contrastive (proposal §5.3)."""

    def __init__(
        self,
        lambda_roi: float = 0.3,
        lambda_bg: float = 1.0,
        delta: float = 2.0,
        p_t: float = 1.0,
        p_b: float = 3.0,
    ) -> None:
        super().__init__()
        if delta <= 0.0:
            raise ValueError(f"delta must be positive; got {delta}")
        if p_t <= 0.0 or p_b <= 0.0:
            raise ValueError(f"exponents must be positive; got p_t={p_t}, p_b={p_b}")
        self.lambda_roi = float(lambda_roi)
        self.lambda_bg = float(lambda_bg)
        self.delta = float(delta)
        self.p_t = float(p_t)
        self.p_b = float(p_b)

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        if inputs.v_orig is None or inputs.v_perturb is None:
            raise ValueError(
                "ContrastiveTumourLoss requires v_orig and v_perturb; the composite "
                "must request the perturbed pass (requires_perturbed_pass=True)."
            )
        if inputs.m_wt is None or inputs.m_bg is None:
            raise ValueError(
                "ContrastiveTumourLoss requires m_wt and m_bg in LossInputs; the "
                "LightningModule must populate them when stage == 'S2'."
            )

        v_orig = inputs.v_orig
        delta_v = (v_orig - inputs.v_perturb).abs()  # (B, C, h, w, d)
        m = inputs.m_wt.to(delta_v.dtype)
        m_bg = inputs.m_bg.to(delta_v.dtype)
        n_chan = float(v_orig.shape[1])

        # ROI: aggregate-cap mean, negated. Mean is over (voxels x channels).
        roi_pow = delta_v.pow(self.p_t)
        roi_num = (roi_pow * m).flatten(1).sum(dim=1)
        roi_den = (m.flatten(1).sum(dim=1) * n_chan).clamp_min(1.0)
        roi_mean = roi_num / roi_den
        cap_t = float(self.delta**self.p_t)
        cap_t_tensor = torch.full_like(roi_mean, cap_t)
        loss_roi = -torch.minimum(roi_mean, cap_t_tensor)

        # BG: per-voxel cap, then mean.
        bg_pow = delta_v.pow(self.p_b)
        cap_b = float(self.delta**self.p_b)
        bg_capped = torch.minimum(bg_pow, torch.full_like(bg_pow, cap_b))
        bg_num = (bg_capped * m_bg).flatten(1).sum(dim=1)
        bg_den = (m_bg.flatten(1).sum(dim=1) * n_chan).clamp_min(1.0)
        loss_bg = bg_num / bg_den

        total = (self.lambda_roi * loss_roi + self.lambda_bg * loss_bg).mean()

        # Diagnostics — surfaced via ``aux()`` for CSV logging. Detached so the
        # autograd graph is not duplicated.
        with torch.no_grad():
            abs_in_num = (delta_v * m).flatten(1).sum(dim=1)
            abs_out_num = (delta_v * m_bg).flatten(1).sum(dim=1)
            delta_in = (abs_in_num / roi_den).mean()
            delta_out = (abs_out_num / bg_den).mean()
            roi_cap_hit = (roi_mean >= cap_t).float().mean()
            bg_over_cap = (bg_pow >= cap_b).to(delta_v.dtype)
            bg_cap_num = (bg_over_cap * m_bg).flatten(1).sum(dim=1)
            bg_cap_hit = (bg_cap_num / bg_den).mean()
            self._aux = {
                "delta_abs_mean_in": delta_in.detach(),
                "delta_abs_mean_out": delta_out.detach(),
                "roi_cap_hit_frac": roi_cap_hit.detach(),
                "bg_cap_hit_frac": bg_cap_hit.detach(),
            }
        return total
