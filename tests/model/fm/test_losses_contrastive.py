"""Unit tests for the v0.4 region-weighted ContrastiveTumourLoss.

The 2026-06-09 overhaul (CHANGE 2 of
``.claude/notes/changes/2026-06-09_training-regime-overhaul.md``) replaced
the v0.3 mask-sensitivity recipe with a list of region-weighted Lp residuals.
This file rewrites the v0.3 test cases under the new API; the legacy
``lambda_roi``/``delta``/``p_t``/``p_b`` cases were dropped because that
machinery no longer exists.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.losses import (
    CompositeLoss,
    ContrastiveTumourLoss,
    LossInputs,
    RegionTerm,
    build_loss,
)

pytestmark = pytest.mark.unit


def _inputs(
    *,
    v_orig: torch.Tensor,
    u_target: torch.Tensor,
    m_wt: torch.Tensor,
    m_brain: torch.Tensor | None = None,
) -> LossInputs:
    """Build LossInputs with the fields the v0.4 contrastive reads."""
    z = torch.zeros_like(v_orig)
    return LossInputs(
        x_clean=z,
        noise=z,
        x_t=z,
        timesteps=torch.zeros(z.shape[0], dtype=torch.long),
        u_target=u_target,
        v_orig=v_orig,
        v_perturb=None,
        m_wt=m_wt,
        m_bg=None,
        m_brain=m_brain,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_empty_terms_rejected() -> None:
    with pytest.raises(ValueError, match="at least one RegionTerm"):
        ContrastiveTumourLoss(terms=[])


def test_duplicate_term_names_rejected() -> None:
    terms = [
        RegionTerm(name="a", region="healthy", p=2.0),
        RegionTerm(name="a", region="wt", p=2.0),
    ]
    with pytest.raises(ValueError, match="unique"):
        ContrastiveTumourLoss(terms=terms)


def test_invalid_region_rejected() -> None:
    with pytest.raises(ValueError, match="region must be one of"):
        RegionTerm(name="bad", region="ozone", p=2.0)  # type: ignore[arg-type]


def test_invalid_p_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        RegionTerm(name="bad", region="healthy", p=0.0)


# ---------------------------------------------------------------------------
# Forward — math correctness
# ---------------------------------------------------------------------------


def test_single_healthy_term_lp_matches_manual() -> None:
    """The healthy-region Lp residual matches the analytical sum / (n × C)."""
    B, C, h, w, d = 1, 4, 2, 2, 2
    v_orig = torch.zeros(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    # Plant a constant residual of 2.0 in every voxel of every channel.
    v_orig[...] = 2.0
    m_wt = torch.zeros(B, 1, h, w, d)
    m_brain = torch.ones(B, 1, h, w, d)
    # Healthy = brain ∩ ¬wt = all voxels.
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )
    val = loss(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    # |2|^2 = 4 per voxel; mean over (voxels × C) of 4 is 4.
    assert val.item() == pytest.approx(4.0, abs=1e-5)


def test_two_term_sum_is_linear_in_weights() -> None:
    """Doubling a term's weight doubles its contribution to the total."""
    B, C, h, w, d = 2, 4, 4, 4, 4
    torch.manual_seed(0)
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_brain = torch.ones(B, 1, h, w, d)

    loss_x1 = ContrastiveTumourLoss(
        terms=[
            RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0),
            RegionTerm(name="wt", region="wt", p=2.0, weight=1.0),
        ]
    )
    loss_x2 = ContrastiveTumourLoss(
        terms=[
            RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0),
            RegionTerm(name="wt", region="wt", p=2.0, weight=2.0),
        ]
    )
    v1 = loss_x1(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    v2 = loss_x2(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))

    # Manually compute the wt term's per-sample mean.
    residual2 = (v_orig - u_target).abs().pow(2)
    num_wt = (residual2 * m_wt).flatten(1).sum(dim=1)
    den_wt = (m_wt.flatten(1).sum(dim=1) * C).clamp_min(1.0)
    wt_ps = (num_wt / den_wt).mean()
    assert v2.item() == pytest.approx(v1.item() + wt_ps.item(), abs=1e-5)


def test_zero_loss_when_region_empty() -> None:
    """An empty region contributes 0 — no NaN, no DBZ."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_brain = torch.zeros(B, 1, h, w, d)  # empty brain
    m_wt = torch.zeros(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )
    val = loss(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    assert torch.isfinite(val), "empty region must not produce NaN/Inf"
    assert val.item() == pytest.approx(0.0, abs=1e-8)


def test_loss_isolates_to_brain_minus_wt() -> None:
    """Vary the residual outside ``healthy`` — the loss must NOT change."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    u_target = torch.zeros(B, C, h, w, d)

    # Build masks: half-volume brain; half-of-brain is WT.
    m_brain = torch.zeros(B, 1, h, w, d)
    m_brain[..., :2, :, :] = 1.0
    m_wt = torch.zeros(B, 1, h, w, d)
    m_wt[..., :1, :, :] = 1.0
    # healthy = m_brain ∩ ¬m_wt = slice [1:2, :, :]
    healthy = m_brain * (1.0 - m_wt)
    assert healthy.sum().item() > 0

    # Residual: zero everywhere; we will toggle "outside healthy" cells.
    base = torch.zeros(B, C, h, w, d)
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )
    val0 = loss(_inputs(v_orig=base, u_target=u_target, m_wt=m_wt, m_brain=m_brain))

    # Toggle inside WT and inside background (both outside healthy).
    perturb = base.clone()
    perturb[..., 0, :, :] = 5.0  # WT region
    perturb[..., 3, :, :] = 7.0  # background
    val_outside = loss(_inputs(v_orig=perturb, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    assert val_outside.item() == pytest.approx(val0.item(), abs=1e-6)

    # Toggle inside healthy → loss must change.
    perturb_inside = base.clone()
    perturb_inside[..., 1, :, :] = 3.0
    val_inside = loss(_inputs(v_orig=perturb_inside, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    assert val_inside.item() != pytest.approx(val0.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# per_sample()/aux() contract
# ---------------------------------------------------------------------------


def test_per_sample_returns_b_shaped_tensor_after_forward() -> None:
    B, C, h, w, d = 3, 4, 4, 4, 4
    torch.manual_seed(0)
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_brain = torch.ones(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )

    assert loss.per_sample() is None  # before forward, no cache
    scalar = loss(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    cached = loss.per_sample()

    assert cached is not None
    assert cached.shape == (B,)
    assert not cached.requires_grad
    assert torch.allclose(scalar, cached.mean())


def test_per_sample_returns_none_before_forward_on_fresh_instance() -> None:
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )
    assert loss.per_sample() is None


def test_aux_keys_per_term() -> None:
    """Aux dict carries one entry per term + sentinel diagnostics."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_wt = torch.ones(B, 1, h, w, d)
    m_brain = torch.ones(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(
        terms=[
            RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0),
            RegionTerm(name="wt_l1", region="wt", p=1.0, weight=0.5),
        ]
    )
    _ = loss(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=m_brain))
    aux = loss.aux()
    # Per-term entries.
    assert "term_healthy_lp_mean" in aux
    assert "term_healthy_voxel_frac" in aux
    assert "term_wt_l1_lp_mean" in aux
    assert "term_wt_l1_voxel_frac" in aux
    # Sentinel diagnostics that always log.
    assert "residual_lp_mean_wt" in aux
    assert "residual_lp_mean_healthy" in aux
    assert "healthy_voxel_frac" in aux
    for v in aux.values():
        assert v.ndim == 0 and torch.isfinite(v)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_m_brain_raises_when_term_needs_it() -> None:
    """`healthy` requires m_brain — clear ValueError pointing at the encoder."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v = torch.zeros(B, C, h, w, d)
    m_wt = torch.zeros(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(
        terms=[RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    )
    with pytest.raises(ValueError, match="vena-encode-brain-to-latent"):
        loss(_inputs(v_orig=v, u_target=v, m_wt=m_wt, m_brain=None))


def test_terms_using_only_wt_do_not_need_m_brain() -> None:
    """A `wt`-only configuration works without m_brain."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_wt = torch.ones(B, 1, h, w, d)
    loss = ContrastiveTumourLoss(terms=[RegionTerm(name="wt", region="wt", p=2.0, weight=1.0)])
    val = loss(_inputs(v_orig=v_orig, u_target=u_target, m_wt=m_wt, m_brain=None))
    assert torch.isfinite(val)


# ---------------------------------------------------------------------------
# Builder + CompositeLoss fan-out
# ---------------------------------------------------------------------------


def test_legacy_v03_keys_rejected_with_migration_message() -> None:
    with pytest.raises(ValueError, match="v0.3 keys"):
        build_loss("S2", {"contrastive": {"weight": 0.1, "lambda_roi": 0.3}})


def test_composite_s2_fans_out_per_term_aux_keys() -> None:
    """End-to-end: CompositeLoss exposes contrastive aux under namespaced keys."""
    B, C, h, w, d = 1, 4, 4, 4, 4
    composite: CompositeLoss = build_loss(
        "S2",
        {
            "contrastive": {
                "weight": 0.1,
                "terms": [{"name": "healthy", "region": "healthy", "p": 2.0, "weight": 1.0}],
            }
        },
    )
    v_orig = torch.randn(B, C, h, w, d)
    u_target = torch.zeros(B, C, h, w, d)
    m_wt = torch.zeros(B, 1, h, w, d)
    m_brain = torch.ones(B, 1, h, w, d)
    inputs = LossInputs(
        x_clean=v_orig,
        noise=v_orig,
        x_t=v_orig,
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=u_target,
        v_orig=v_orig,
        v_perturb=None,
        m_wt=m_wt,
        m_brain=m_brain,
    )
    _, per_term = composite(inputs)
    assert "contrastive/term_healthy_lp_mean" in per_term
    assert "contrastive/residual_lp_mean_healthy" in per_term


def test_composite_s2_requires_perturbed_pass_is_false() -> None:
    """The v0.4 contrastive does NOT need v_perturb."""
    composite = build_loss(
        "S2",
        {
            "contrastive": {
                "weight": 0.1,
                "terms": [{"name": "healthy", "region": "healthy", "p": 2.0, "weight": 1.0}],
            }
        },
    )
    assert composite.requires_perturbed_pass is False


def test_lambda_contrast_zero_recovers_s1_total() -> None:
    """Ablation-cleanliness: contrastive weight=0 recovers the S1 total."""
    B, C, h, w, d = 2, 4, 4, 4, 4
    torch.manual_seed(0)
    v_orig = torch.randn(B, C, h, w, d)
    x1 = torch.randn(B, C, h, w, d)
    x0 = torch.randn(B, C, h, w, d)
    u = x1 - x0
    m_wt = torch.randint(0, 2, (B, 1, h, w, d)).float()
    m_brain = torch.ones(B, 1, h, w, d)
    inputs = LossInputs(
        x_clean=x1,
        noise=x0,
        x_t=0.5 * (x0 + x1),
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=u,
        v_orig=v_orig,
        v_perturb=None,
        m_wt=m_wt,
        m_brain=m_brain,
    )
    s1 = build_loss("S1", {"cfm": {"weight": 1.0}})
    s2 = build_loss(
        "S2",
        {
            "cfm": {"weight": 1.0},
            "contrastive": {
                "weight": 0.0,
                "terms": [{"name": "healthy", "region": "healthy", "p": 2.0, "weight": 1.0}],
            },
        },
    )
    s1_total, _ = s1(inputs)
    s2_total, _ = s2(inputs)
    assert torch.allclose(s1_total, s2_total, atol=1e-6)
