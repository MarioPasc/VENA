"""Unit tests for the Lp-aware ContrastiveTumourLoss (proposal §5.3)."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.losses import (
    CompositeLoss,
    ContrastiveTumourLoss,
    LossInputs,
    build_loss,
)

pytestmark = pytest.mark.unit


def _inputs(
    *,
    v_orig: torch.Tensor,
    v_perturb: torch.Tensor,
    m_wt: torch.Tensor,
    m_bg: torch.Tensor,
) -> LossInputs:
    """Build LossInputs with the fields the contrastive term reads.

    The other fields (x_clean / x_t / etc.) are filled with zeros — they only
    matter for CFMLoss, which the contrastive term does not call.
    """
    z = torch.zeros_like(v_orig)
    return LossInputs(
        x_clean=z,
        noise=z,
        x_t=z,
        timesteps=torch.zeros(z.shape[0], dtype=torch.long),
        u_target=z,
        v_orig=v_orig,
        v_perturb=v_perturb,
        m_wt=m_wt,
        m_bg=m_bg,
    )


def test_empty_mask_returns_zero_loss_without_nan() -> None:
    """No tumour voxels and no BG voxels ⇒ both region means hit the empty
    branch (denominator clamped to 1.0, numerator = 0) ⇒ loss = 0, not NaN."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v_orig = torch.zeros(B, C, h, w, d)
    v_perturb = torch.zeros(B, C, h, w, d)
    m_wt = torch.zeros(B, 1, h, w, d)
    m_bg = torch.zeros(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(lambda_roi=0.3, lambda_bg=1.0, delta=2.0, p_t=1.0, p_b=3.0)
    val = loss(_inputs(v_orig=v_orig, v_perturb=v_perturb, m_wt=m_wt, m_bg=m_bg))
    assert torch.isfinite(val), "empty-mask path must not produce NaN/Inf"
    assert torch.allclose(val, torch.zeros(()), atol=1e-7)


def test_bg_per_voxel_cap_activates() -> None:
    """A single BG voxel with |Δ|=3, δ=2, p_b=3 — capped contribution is
    min(27, 8) = 8 per channel. Two-voxel toy.
    """
    # One voxel inside WT (zero Δ, doesn't matter); one voxel in BG with |Δ|=3.
    v_orig = torch.zeros(1, 1, 2, 1, 1)
    v_perturb = torch.zeros(1, 1, 2, 1, 1)
    v_orig[0, 0, 1, 0, 0] = 3.0  # BG voxel: |Δ| = 3
    m_wt = torch.zeros(1, 1, 2, 1, 1)
    m_wt[0, 0, 0, 0, 0] = 1.0
    m_bg = torch.zeros(1, 1, 2, 1, 1)
    m_bg[0, 0, 1, 0, 0] = 1.0

    loss = ContrastiveTumourLoss(lambda_roi=0.0, lambda_bg=1.0, delta=2.0, p_t=1.0, p_b=3.0)
    val = loss(_inputs(v_orig=v_orig, v_perturb=v_perturb, m_wt=m_wt, m_bg=m_bg))
    # n_chan=1; bg_num = min(3^3, 2^3) = 8; bg_den = 1 * 1 = 1; loss_bg = 8.
    assert torch.allclose(val, torch.tensor(8.0), atol=1e-6)
    # Cap-hit diagnostic should report 100% of BG voxels capped.
    aux = loss.aux()
    assert torch.allclose(aux["bg_cap_hit_frac"], torch.tensor(1.0), atol=1e-6)


def test_roi_aggregate_cap_clips_negative_loss() -> None:
    """ROI term lives in [-δ^pt, 0]. With huge |Δ| and the aggregate cap, the
    ROI loss must equal -δ^pt and never go more negative.
    """
    v_orig = torch.zeros(1, 1, 1, 1, 1)
    v_perturb = torch.zeros(1, 1, 1, 1, 1)
    v_orig[0, 0, 0, 0, 0] = 10.0  # |Δ| = 10
    m_wt = torch.ones(1, 1, 1, 1, 1)
    m_bg = torch.zeros(1, 1, 1, 1, 1)

    loss = ContrastiveTumourLoss(lambda_roi=1.0, lambda_bg=0.0, delta=2.0, p_t=1.0, p_b=3.0)
    val = loss(_inputs(v_orig=v_orig, v_perturb=v_perturb, m_wt=m_wt, m_bg=m_bg))
    # ROI mean = 10; cap = δ^pt = 2; loss_roi = -min(10, 2) = -2.
    assert torch.allclose(val, torch.tensor(-2.0), atol=1e-6)
    aux = loss.aux()
    assert torch.allclose(aux["roi_cap_hit_frac"], torch.tensor(1.0), atol=1e-6)


def test_aux_keys_present_after_forward() -> None:
    """The four diagnostics must be returned by ``aux()`` after a successful
    forward, so the composite can fan them out to CSV columns.
    """
    B, C, h, w, d = 2, 4, 4, 4, 4
    v_orig = torch.randn(B, C, h, w, d)
    v_perturb = torch.randn(B, C, h, w, d)
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_bg = 1.0 - m_wt
    loss = ContrastiveTumourLoss()
    _ = loss(_inputs(v_orig=v_orig, v_perturb=v_perturb, m_wt=m_wt, m_bg=m_bg))
    aux = loss.aux()
    assert set(aux.keys()) == {
        "delta_abs_mean_in",
        "delta_abs_mean_out",
        "roi_cap_hit_frac",
        "bg_cap_hit_frac",
    }
    for v in aux.values():
        assert v.ndim == 0 and torch.isfinite(v)


def test_lambda_contrast_zero_recovers_s1_total() -> None:
    """Ablation-cleanliness: with the composite's contrastive weight at 0, the
    S2 total must equal the S1 total bit-by-bit (within fp32 noise).
    """
    B, C, h, w, d = 2, 4, 4, 4, 4
    torch.manual_seed(0)
    v_orig = torch.randn(B, C, h, w, d)
    v_perturb = torch.randn(B, C, h, w, d)  # ignored by CFM, used by contrastive
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_bg = 1.0 - m_wt
    x1 = torch.randn(B, C, h, w, d)
    x0 = torch.randn(B, C, h, w, d)
    u = x1 - x0
    inputs = LossInputs(
        x_clean=x1,
        noise=x0,
        x_t=0.5 * (x0 + x1),
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=u,
        v_orig=v_orig,
        v_perturb=v_perturb,
        m_wt=m_wt,
        m_bg=m_bg,
    )
    s1 = build_loss("S1", {"cfm": {"weight": 1.0}})
    s2 = build_loss("S2", {"cfm": {"weight": 1.0}, "contrastive": {"weight": 0.0}})
    s1_total, _ = s1(inputs)
    s2_total, per_term = s2(inputs)
    assert torch.allclose(s1_total, s2_total, atol=1e-6)
    # And the auxiliary diagnostics are still surfaced under namespaced keys.
    assert {"contrastive/delta_abs_mean_in", "contrastive/bg_cap_hit_frac"}.issubset(per_term)


def test_contrastive_loss_p_equal_two_is_signed() -> None:
    """With p_t = p_b = 2 the loss is well-defined; ROI pushes negative, BG
    pushes positive. Just confirms numerical sanity for an exponent != 1, 3.
    """
    B, C, h, w, d = 1, 4, 4, 4, 4
    v_orig = torch.randn(B, C, h, w, d) * 0.5
    v_perturb = torch.zeros_like(v_orig)
    m_wt = torch.ones(B, 1, h, w, d)
    m_bg = torch.zeros(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(lambda_roi=1.0, lambda_bg=0.0, delta=2.0, p_t=2.0, p_b=2.0)
    val = loss(_inputs(v_orig=v_orig, v_perturb=v_perturb, m_wt=m_wt, m_bg=m_bg))
    # All in WT, well below the δ^2=4 cap, so loss_roi = -mean(Δ^2) < 0.
    assert val.item() < 0.0
    assert torch.isfinite(val)


def test_invalid_exponents_rejected() -> None:
    with pytest.raises(ValueError, match="exponents must be positive"):
        ContrastiveTumourLoss(p_t=0.0)
    with pytest.raises(ValueError, match="exponents must be positive"):
        ContrastiveTumourLoss(p_b=-1.0)


def test_missing_v_perturb_raises() -> None:
    """S2 composite forgetting to request the perturbed pass — the loss must
    error clearly rather than silently produce garbage."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v = torch.zeros(B, C, h, w, d)
    m = torch.zeros(B, 1, h, w, d)
    loss = ContrastiveTumourLoss()
    bad = LossInputs(
        x_clean=v,
        noise=v,
        x_t=v,
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=v,
        v_orig=v,
        v_perturb=None,
        m_wt=m,
        m_bg=m,
    )
    with pytest.raises(ValueError, match="v_perturb"):
        loss(bad)


def test_composite_s2_fans_out_aux_keys() -> None:
    """End-to-end: a CompositeLoss built for S2 must expose the contrastive
    aux scalars under namespaced ``contrastive/<aux>`` keys so the
    LightningModule's existing ``train/*`` plumbing logs them automatically.
    """
    B, C, h, w, d = 1, 4, 4, 4, 4
    composite: CompositeLoss = build_loss("S2", {"contrastive": {"weight": 0.01}})
    v_orig = torch.randn(B, C, h, w, d)
    v_perturb = torch.randn(B, C, h, w, d)
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_bg = 1.0 - m_wt
    inputs = LossInputs(
        x_clean=v_orig,
        noise=v_orig,
        x_t=v_orig,
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=v_orig,
        v_orig=v_orig,
        v_perturb=v_perturb,
        m_wt=m_wt,
        m_bg=m_bg,
    )
    _, per_term = composite(inputs)
    expected_aux = {
        "contrastive/delta_abs_mean_in",
        "contrastive/delta_abs_mean_out",
        "contrastive/roi_cap_hit_frac",
        "contrastive/bg_cap_hit_frac",
    }
    assert expected_aux.issubset(per_term)
