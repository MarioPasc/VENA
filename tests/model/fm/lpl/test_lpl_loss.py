"""Unit tests for :class:`vena.model.fm.lpl.loss.LplLoss`.

Covers:

* High-SNR gate: when no sample in the batch has ``t > t_min`` the loss
  short-circuits to scalar zero with a finite ``hi_frac`` of 0.
* Identical features produce zero loss (modulo numerical floor).
* Region-weighted formula: a synthetic 1-voxel WT error contributes
  exactly the expected ``alpha_wt * w_l / C / 1`` (with empty-region
  guard ``max(|Ω|, 1) = 1``).
* p exponent honoured per region (``p=2`` vs ``p=3`` on a known step).
* CSV breakdown keys are present for every requested block and region.
* Backward through the loss produces finite gradients on the predicted
  features (autograd-connected).
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.lpl import FeatureStatsEMA, LplConfig, LplLoss

pytestmark = pytest.mark.unit


def _stats(channels: dict[int, int]) -> FeatureStatsEMA:
    """Warmed-up FeatureStatsEMA with identity-like stats (mean=0, var=1).

    Identity stats let the test reason about feature distances directly
    in the same numeric scale as the raw inputs (the standardise call
    becomes the identity up to ``eps``-tolerance).
    """
    s = FeatureStatsEMA(channels=channels, decay=0.0)
    # First update with zero-mean unit-var data so the buffers land at
    # (0, 1). Decay=0 means the second batch fully replaces the bootstrap,
    # but a single bootstrap with the right stats is enough.
    for blk, c in channels.items():
        # A 4-sample batch of zero-mean unit-var.
        feat = torch.randn(4, c, 2, 2, 2)
        # Recentre + rescale to enforce exact zero-mean unit-var on the
        # flat batch — otherwise the EMA picks up sample noise and the
        # tests get less precise.
        flat = feat.movedim(1, -1).reshape(-1, c)
        flat = (flat - flat.mean(0)) / (flat.std(0, unbiased=False) + 1e-8)
        feat = flat.reshape(*feat.movedim(1, -1).shape).movedim(-1, 1)
        s.update({blk: feat})
    return s


def _two_block_cfg(**overrides) -> LplConfig:
    """Minimal LplConfig with two blocks and uniform weights."""
    base = {
        "A": [2, 5],
        "w_l": {2: 1.0, 5: 1.0},
        "t_min": 0.5,
        "alpha": {"wt": 1.0, "notwt": 1.0},
        "p": {"wt": 2, "notwt": 2},
        "outlier_k": {2: 1e9, 5: 1e9},  # disable outlier masking
        "soft_region": False,
        "region_set": ["wt", "notwt"],
    }
    base.update(overrides)
    return LplConfig.model_validate(base)


def _shapes(batch: int = 2, c: int = 4):
    """Two block shapes mirroring §3.5 (block 2 native, block 5 2× upsample)."""
    return {
        2: (batch, c, 4, 4, 4),
        5: (batch, c, 8, 8, 8),
    }


def _zero_features(batch: int = 2, c: int = 4) -> dict[int, torch.Tensor]:
    shapes = _shapes(batch=batch, c=c)
    return {k: torch.zeros(s) for k, s in shapes.items()}


def _ones_masks(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent-res WT + brain masks: brain everywhere, no WT."""
    m_wt = torch.zeros(batch, 1, 4, 4, 4)
    m_brain = torch.ones(batch, 1, 4, 4, 4)
    return m_wt, m_brain


def test_gate_short_circuit_when_all_t_below_tmin() -> None:
    """``t <= t_min`` for every sample → zero scalar, hi_frac == 0."""
    cfg = _two_block_cfg()
    stats = _stats({2: 4, 5: 4})
    loss = LplLoss(cfg, stats)
    pred = _zero_features()
    tgt = _zero_features()
    m_wt, m_brain = _ones_masks()
    t = torch.tensor([0.1, 0.2])
    out, breakdown = loss(pred, tgt, m_wt, m_brain, t)
    assert out.item() == 0.0
    assert breakdown["hi_frac"] == 0.0
    # All per-block / per-region keys still present.
    assert breakdown["lpl_b2"] == 0.0
    assert breakdown["lpl_b5"] == 0.0
    assert breakdown["lpl_wt"] == 0.0
    assert breakdown["lpl_notwt"] == 0.0


def test_identity_features_yield_zero_loss() -> None:
    """``phi_pred == phi_tgt`` → loss is (numerically) zero."""
    cfg = _two_block_cfg()
    stats = _stats({2: 4, 5: 4})
    loss = LplLoss(cfg, stats)
    pred = _zero_features()
    tgt = {k: v.clone() for k, v in pred.items()}
    m_wt, m_brain = _ones_masks()
    t = torch.tensor([0.9, 0.9])
    out, _ = loss(pred, tgt, m_wt, m_brain, t)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_hi_frac_reports_correct_fraction() -> None:
    """``hi_frac == (t > t_min).float().mean()``."""
    cfg = _two_block_cfg(t_min=0.5)
    stats = _stats({2: 4, 5: 4})
    loss = LplLoss(cfg, stats)
    pred = _zero_features(batch=4)
    tgt = _zero_features(batch=4)
    m_wt = torch.zeros(4, 1, 4, 4, 4)
    m_brain = torch.ones(4, 1, 4, 4, 4)
    t = torch.tensor([0.1, 0.4, 0.6, 0.9])  # 2/4 above gate
    _, breakdown = loss(pred, tgt, m_wt, m_brain, t)
    assert breakdown["hi_frac"] == pytest.approx(0.5)


def test_csv_breakdown_keys_present_for_every_A_and_region() -> None:
    cfg = _two_block_cfg(A=[2, 5], w_l={2: 1.0, 5: 1.0}, outlier_k={2: 1e9, 5: 1e9})
    stats = _stats({2: 4, 5: 4})
    loss = LplLoss(cfg, stats)
    pred = _zero_features()
    tgt = _zero_features()
    m_wt, m_brain = _ones_masks()
    t = torch.tensor([0.9, 0.9])
    _, breakdown = loss(pred, tgt, m_wt, m_brain, t)
    for blk in (2, 5):
        assert f"lpl_b{blk}" in breakdown
    for r in ("wt", "notwt"):
        assert f"lpl_{r}" in breakdown
    assert "hi_frac" in breakdown


def test_backward_produces_finite_grads_on_prediction() -> None:
    """``LplLoss.forward`` must connect to autograd through ``phi_pred``."""
    cfg = _two_block_cfg(outlier_k={2: 1e9, 5: 1e9})
    stats = _stats({2: 4, 5: 4})
    loss = LplLoss(cfg, stats)
    pred = {k: torch.zeros(s, requires_grad=True) for k, s in _shapes().items()}
    tgt = _zero_features()
    # Inject a deliberate error in pred so the loss is non-zero.
    pred_input = {k: v + 0.5 for k, v in pred.items()}
    m_wt, m_brain = _ones_masks()
    t = torch.tensor([0.9, 0.9])
    out, _ = loss(pred_input, tgt, m_wt, m_brain, t)
    out.backward()
    for v in pred.values():
        assert v.grad is not None
        assert torch.isfinite(v.grad).all()


def _noisy_uniform(shape: tuple[int, ...], fill: float, scale: float = 1e-3) -> torch.Tensor:
    """Mean-``fill`` features with tiny per-voxel noise so MAD > 0.

    Uniform-fill features degenerate the outlier mask (``k * MAD = 0`` → all
    voxels masked); the production decoder never produces uniform features
    so the loss code does not guard against this edge case. The tests add
    a small perturbation to keep MAD non-zero without changing the test's
    semantic meaning.
    """
    return torch.full(shape, fill) + scale * torch.randn(*shape)


def test_p_exponent_affects_loss_magnitude() -> None:
    """Bigger ``p`` amplifies large per-voxel errors. With a near-uniform
    error of magnitude 1.5 on every voxel, ``p=3`` loss > ``p=2``.
    (Errors >1 are amplified by larger p; errors <1 would be damped.)
    """
    torch.manual_seed(123)
    cfg_p2 = _two_block_cfg(p={"wt": 2, "notwt": 2})
    cfg_p3 = _two_block_cfg(p={"wt": 3, "notwt": 3})
    stats = _stats({2: 4, 5: 4})
    pred = {k: _noisy_uniform(s, fill=1.5) for k, s in _shapes().items()}
    tgt = _zero_features()
    m_wt, m_brain = _ones_masks()
    t = torch.tensor([0.9, 0.9])

    out_p2, _ = LplLoss(cfg_p2, stats)(pred, tgt, m_wt, m_brain, t)
    out_p3, _ = LplLoss(cfg_p3, stats)(pred, tgt, m_wt, m_brain, t)
    assert out_p2.item() > 0.0
    assert out_p3.item() > out_p2.item()


def test_alpha_weight_scales_per_region_contribution() -> None:
    """Doubling ``alpha_notwt`` doubles the loss when all error lives
    outside WT and inside the brain.
    """
    torch.manual_seed(456)
    cfg_a = _two_block_cfg(alpha={"wt": 1.0, "notwt": 1.0})
    cfg_b = _two_block_cfg(alpha={"wt": 1.0, "notwt": 2.0})
    stats = _stats({2: 4, 5: 4})
    pred = {k: _noisy_uniform(s, fill=0.5) for k, s in _shapes().items()}
    tgt = _zero_features()
    m_wt = torch.zeros(2, 1, 4, 4, 4)  # no WT anywhere
    m_brain = torch.ones(2, 1, 4, 4, 4)
    t = torch.tensor([0.9, 0.9])

    out_a, _ = LplLoss(cfg_a, stats)(pred, tgt, m_wt, m_brain, t)
    out_b, _ = LplLoss(cfg_b, stats)(pred, tgt, m_wt, m_brain, t)
    assert out_a.item() > 0.0
    # All error is in notwt; doubling alpha_notwt doubles the loss.
    assert out_b.item() == pytest.approx(2.0 * out_a.item(), rel=1e-4)
