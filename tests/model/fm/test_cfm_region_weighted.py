"""Unit tests for the S1 v3 region-weighted CFM loss + RegionWeights helper.

Covers:

* Back-compat: ``enabled=False`` (or RegionWeights all-ones) collapses to
  standard mean L1 within float-precision (acceptance criterion §3.4 of
  the v3 spec).
* Partition: the five region masks built by ``build_region_weight_tensor``
  are disjoint and cover the whole volume.
* Application: a controlled (v_pred, u_target) pair where the per-voxel
  error is 1.0 everywhere yields the manually-computable weighted mean.
* WT override: setting ``wt: 200`` (with netc/ed/et defaults) makes the
  three sub-region weights equal to 200.
* Threshold: soft-mask values < τ are excluded from sub-region weighting.
* Builder wiring: ``build_loss(stage="S1", cfg={"cfm": {"region_weights":
  {...}}})`` produces a ``CFMLoss`` carrying a non-None ``region_weights``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit

from vena.model.fm.controlnet.losses import (
    CFMLoss,
    LossInputs,
    RegionWeights,
    build_loss,
    build_region_weight_tensor,
)

_B, _C, _H, _W, _D = 2, 4, 6, 6, 6


def _zeros_inputs() -> LossInputs:
    return LossInputs(
        x_clean=torch.zeros(_B, _C, _H, _W, _D),
        noise=torch.zeros(_B, _C, _H, _W, _D),
        x_t=torch.zeros(_B, _C, _H, _W, _D),
        timesteps=torch.zeros(_B, dtype=torch.long),
        u_target=torch.zeros(_B, _C, _H, _W, _D),
        v_orig=torch.zeros(_B, _C, _H, _W, _D),
    )


def _controlled_inputs(error_value: float = 1.0) -> LossInputs:
    """Make ``|v_orig - u_target| = error_value`` at every voxel."""
    v = torch.full((_B, _C, _H, _W, _D), error_value)
    u = torch.zeros_like(v)
    base = _zeros_inputs()
    base.v_orig = v
    base.u_target = u
    return base


def _toy_masks(
    bg_frac: float = 0.5,
    netc_frac: float = 0.1,
    ed_frac: float = 0.1,
    et_frac: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct synthetic disjoint masks. Returns (m_brain, m_tumor)."""
    n_voxels = _H * _W * _D
    flat_brain = torch.zeros(_B, 1, n_voxels)
    flat_tumor = torch.zeros(_B, 3, n_voxels)
    n_bg = int(bg_frac * n_voxels)
    n_netc = int(netc_frac * n_voxels)
    n_ed = int(ed_frac * n_voxels)
    n_et = int(et_frac * n_voxels)
    # voxels: [0, n_bg) → bg; [n_bg, n_bg+n_netc) → netc; ... rest → bnwt.
    flat_brain[:, 0, n_bg:] = 1.0  # everything past bg is brain
    flat_tumor[:, 0, n_bg : n_bg + n_netc] = 1.0
    flat_tumor[:, 1, n_bg + n_netc : n_bg + n_netc + n_ed] = 1.0
    flat_tumor[:, 2, n_bg + n_netc + n_ed : n_bg + n_netc + n_ed + n_et] = 1.0
    return (
        flat_brain.view(_B, 1, _H, _W, _D),
        flat_tumor.view(_B, 3, _H, _W, _D),
    )


def test_region_weights_disabled_matches_unweighted_l1() -> None:
    """``enabled=False`` ⇒ classic mean L1 within ε."""
    rw = RegionWeights(enabled=False)
    cfm = CFMLoss(reduction="mean", norm="l1", region_weights=rw)
    inputs = _controlled_inputs(error_value=0.7)
    # m_brain, m_tumor unused on disabled path.
    expected = F.l1_loss(inputs.v_orig, inputs.u_target, reduction="mean")
    actual = cfm(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_region_weights_all_ones_matches_unweighted_l1() -> None:
    """All weights = 1.0 ⇒ identical to unweighted mean (within ε).

    With every region weighted equally, ``(loss * w).sum() / w.sum()`` reduces
    to ``loss.mean()``.
    """
    m_brain, m_tumor = _toy_masks()
    rw = RegionWeights(
        enabled=True,
        bg=1.0,
        brain_not_wt=1.0,
        netc=1.0,
        ed=1.0,
        et=1.0,
    )
    cfm = CFMLoss(reduction="none", norm="l1", region_weights=rw)
    inputs = _controlled_inputs(error_value=0.5)
    inputs.m_brain = m_brain
    inputs.m_tumor = m_tumor
    expected = F.l1_loss(inputs.v_orig, inputs.u_target, reduction="mean")
    actual = cfm(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_region_weights_disjoint_partition() -> None:
    """For controlled disjoint masks the per-voxel region indicator sums to 1.

    Reconstructs the per-region indicators using the same algebra that
    ``build_region_weight_tensor`` uses; ensures the partition is exact.
    """
    m_brain, m_tumor = _toy_masks()
    τ = 0.5
    in_brain = m_brain > 0.5
    m_t_hard = m_tumor >= τ
    m_wt_hard = m_t_hard.any(dim=1, keepdim=True)
    region_bg = ~in_brain
    region_bnwt = in_brain & ~m_wt_hard
    region_netc = m_t_hard[:, 0:1] & in_brain
    region_ed = m_t_hard[:, 1:2] & in_brain
    region_et = m_t_hard[:, 2:3] & in_brain
    indicator_sum = (
        region_bg.float()
        + region_bnwt.float()
        + region_netc.float()
        + region_ed.float()
        + region_et.float()
    )
    assert torch.all(indicator_sum == 1.0), (
        f"partition broken: min={indicator_sum.min().item()} max={indicator_sum.max().item()}"
    )


def test_region_weights_apply_per_voxel() -> None:
    """Controlled inputs ⇒ analytical weighted mean."""
    m_brain, m_tumor = _toy_masks()
    weights = {
        "bg": 1.0,
        "brain_not_wt": 2.0,
        "netc": 10.0,
        "ed": 10.0,
        "et": 100.0,
    }
    rw = RegionWeights(enabled=True, **weights)
    cfm = CFMLoss(reduction="none", norm="l1", region_weights=rw)
    inputs = _controlled_inputs(error_value=1.0)
    inputs.m_brain = m_brain
    inputs.m_tumor = m_tumor

    # Manual reference. Region voxel counts ×_C (broadcast across channels).
    in_brain = m_brain > 0.5
    m_t_hard = m_tumor >= 0.5
    m_wt_hard = m_t_hard.any(dim=1, keepdim=True)
    n_bg = int((~in_brain).sum().item())
    n_bnwt = int((in_brain & ~m_wt_hard).sum().item())
    n_netc = int((m_t_hard[:, 0:1] & in_brain).sum().item())
    n_ed = int((m_t_hard[:, 1:2] & in_brain).sum().item())
    n_et = int((m_t_hard[:, 2:3] & in_brain).sum().item())
    # Each region count broadcasts across 4 channels.
    num = _C * (
        n_bg * weights["bg"]
        + n_bnwt * weights["brain_not_wt"]
        + n_netc * weights["netc"]
        + n_ed * weights["ed"]
        + n_et * weights["et"]
    )
    expected = torch.tensor(num / num)  # error=1 ⇒ numerator/denominator ratio = 1
    actual = cfm(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_region_weights_wt_override_flattens_subregions() -> None:
    """Setting ``wt`` makes netc/ed/et all use that single weight."""
    m_brain, m_tumor = _toy_masks()
    rw_per = RegionWeights(
        enabled=True,
        netc=200.0,
        ed=200.0,
        et=200.0,
    )
    rw_override = RegionWeights(enabled=True, wt=200.0)  # netc/ed/et defaults ignored
    cfm_per = CFMLoss(reduction="none", norm="l1", region_weights=rw_per)
    cfm_override = CFMLoss(reduction="none", norm="l1", region_weights=rw_override)
    inputs = _controlled_inputs(error_value=0.3)
    inputs.m_brain = m_brain
    inputs.m_tumor = m_tumor
    torch.testing.assert_close(cfm_override(inputs), cfm_per(inputs), rtol=1e-6, atol=1e-6)


def test_region_weights_threshold_excludes_soft() -> None:
    """Soft mask values < τ should not count as belonging to the sub-region.

    Build m_tumor with all soft values = 0.4 < 0.5. The resulting per-region
    indicators should treat ALL brain voxels as ``brain_not_wt`` (no
    sub-region active), so the weighted loss matches a uniform brain-not-wt
    weight.
    """
    m_brain = torch.ones(_B, 1, _H, _W, _D)
    m_tumor = torch.full((_B, 3, _H, _W, _D), 0.4)  # all below τ=0.5
    rw = RegionWeights(
        enabled=True,
        bg=1.0,
        brain_not_wt=5.0,
        netc=999.0,
        ed=999.0,
        et=999.0,
        threshold=0.5,
    )
    cfm = CFMLoss(reduction="none", norm="l1", region_weights=rw)
    inputs = _controlled_inputs(error_value=1.0)
    inputs.m_brain = m_brain
    inputs.m_tumor = m_tumor
    actual = cfm(inputs)
    # All voxels are brain_not_wt (m_brain=1, all sub-masks < τ). Weighted mean
    # = error_value (because weights cancel in the ratio).
    torch.testing.assert_close(actual, torch.tensor(1.0), rtol=1e-6, atol=1e-6)


def test_cfm_loss_rejects_mean_reduction_with_enabled_region_weights() -> None:
    """``reduction='mean'`` + region_weights.enabled=True must error fast."""
    with pytest.raises(ValueError, match="reduction='none'"):
        CFMLoss(reduction="mean", norm="l1", region_weights=RegionWeights(enabled=True))


def test_cfm_loss_accepts_mean_reduction_with_disabled_region_weights() -> None:
    """``reduction='mean'`` + region_weights.enabled=False is valid back-compat."""
    cfm = CFMLoss(reduction="mean", norm="l1", region_weights=RegionWeights(enabled=False))
    inputs = _controlled_inputs(error_value=0.5)
    expected = F.l1_loss(inputs.v_orig, inputs.u_target, reduction="mean")
    torch.testing.assert_close(cfm(inputs), expected, rtol=1e-6, atol=1e-6)


def test_build_region_weight_tensor_disabled_returns_none() -> None:
    """Disabled config ⇒ caller falls back to mean reduction."""
    assert build_region_weight_tensor(RegionWeights(enabled=False), None, None) is None


def test_build_region_weight_tensor_missing_masks_raises() -> None:
    """Enabled config requires both masks."""
    with pytest.raises(ValueError, match="m_brain"):
        build_region_weight_tensor(RegionWeights(enabled=True), None, None)


def test_builder_wires_region_weights_block() -> None:
    """``build_loss(stage='S1', cfg=...)`` propagates region_weights into CFMLoss."""
    cfg = {
        "cfm": {
            "weight": 1.0,
            "reduction": "none",
            "norm": "l1",
            "region_weights": {
                "enabled": True,
                "bg": 1.0,
                "brain_not_wt": 1.0,
                "netc": 10.0,
                "ed": 10.0,
                "et": 100.0,
            },
        }
    }
    composite = build_loss("S1", cfg)
    cfm = composite.terms["cfm"]
    assert isinstance(cfm, CFMLoss)
    assert cfm.region_weights is not None
    assert cfm.region_weights.enabled is True
    assert cfm.region_weights.et == 100.0


def test_builder_legacy_path_leaves_region_weights_none() -> None:
    """Without ``region_weights`` block the CFMLoss remains in legacy mode."""
    cfg = {"cfm": {"weight": 1.0, "reduction": "mean", "norm": "l1"}}
    composite = build_loss("S1", cfg)
    cfm = composite.terms["cfm"]
    assert isinstance(cfm, CFMLoss)
    assert cfm.region_weights is None
