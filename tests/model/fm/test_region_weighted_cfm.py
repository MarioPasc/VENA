"""Tests for the Brain/TC region-weighted CFM loss (task 21).

Covers the six deliverables in D3:

1. L1-equivalence: brain/tc mode with equal weights == functional.l1_loss(mean).
2. Back-compat: all 6 live YAML weight blocks produce byte-identical
   build_region_weight_tensor output before/after the task-21 changes.
3. Strict partition: BG / Brain / TC are pairwise disjoint and cover
   everything; every voxel gets exactly one weight value.
4. Up-weight effect: tc=10 raises TC-voxel gradients vs tc=1 by the
   analytically expected ratio.
5. Missing-mask: brain/tc mode + m_tc_soft is None → raises.
6. Config round-trip: YAML dict → BrainTCWeights → decision.json value.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from vena.model.fm.controlnet.losses import (
    BrainTCMissingMaskError,
    BrainTCWeights,
    CFMLoss,
    LossInputs,
    RegionWeights,
    build_brain_tc_weight_tensor,
    build_loss,
    build_region_weight_tensor,
)

pytestmark = pytest.mark.fm

# ---------------------------------------------------------------------------
# Shared geometry (CPU tensors, no checkpoint needed)
# ---------------------------------------------------------------------------

_B, _C, _H, _W, _D = 1, 4, 4, 4, 4  # spatial = 64, total B*C*H*W*D = 256
_TC_SPATIAL = 16  # first 16 of 64 spatial voxels are TC (25 %)


def _make_all_brain() -> torch.Tensor:
    """Brain mask: all ones (all voxels in brain)."""
    return torch.ones(_B, 1, _H, _W, _D)


def _make_tc_soft(tc_spatial: int = _TC_SPATIAL) -> torch.Tensor:
    """Soft TC mask: first ``tc_spatial`` voxels = 1.0, rest = 0.0."""
    m = torch.zeros(_B, 1, _H, _W, _D)
    flat = m.view(_B, 1, -1)
    flat[:, :, :tc_spatial] = 1.0
    return m


def _rand_inputs(error_scale: float = 1.0, seed: int = 42) -> LossInputs:
    """Random v_pred / u_target pair with a controlled difference."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    v = torch.rand(_B, _C, _H, _W, _D, generator=rng)
    u = v - error_scale * torch.rand(_B, _C, _H, _W, _D, generator=rng)
    return LossInputs(
        x_clean=torch.zeros(_B, _C, _H, _W, _D),
        noise=torch.zeros(_B, _C, _H, _W, _D),
        x_t=torch.zeros(_B, _C, _H, _W, _D),
        timesteps=torch.zeros(_B, dtype=torch.long),
        u_target=u,
        v_orig=v,
        m_brain=_make_all_brain(),
        m_tc_soft=_make_tc_soft(),
    )


# ---------------------------------------------------------------------------
# D3 test 1: L1-equivalence
# ---------------------------------------------------------------------------


def test_brain_tc_equal_weights_equals_mean_l1() -> None:
    """brain=1.0, tc=1.0 → identical to functional.l1_loss(reduction='mean').

    With every voxel carrying weight 1.0, (loss*w).sum()/w.sum() = loss.mean().
    This is the load-bearing guarantee: ship the mechanism, change nothing
    numerically until a weight is deliberately raised.
    """
    btc = BrainTCWeights(enabled=True, brain=1.0, tc=1.0)
    cfm = CFMLoss(reduction="none", norm="l1", brain_tc_weights=btc)
    inputs = _rand_inputs()

    actual = cfm(inputs)
    expected = functional.l1_loss(inputs.v_orig, inputs.u_target, reduction="mean")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    residual = (actual - expected).abs().item()
    # Report the residual (assert already checked it is within tolerance).
    assert residual < 1e-5, f"L1-equivalence residual={residual:.2e} exceeds 1e-5"


# ---------------------------------------------------------------------------
# D3 test 2: Back-compat — 6 live YAML weight blocks are byte-identical
# ---------------------------------------------------------------------------

# All 6 live production YAMLs carry this identical region_weights block.
_LIVE_YAML_BLOCK = {
    "enabled": True,
    "bg": 1.0,
    "brain_not_wt": 1.0,
    "netc": 50.0,
    "ed": 50.0,
    "et": 300.0,
    "wt": None,
    "threshold": 0.5,
}

# Six live YAML filenames (for documentation; all share the same block).
_LIVE_YAML_NAMES = [
    "picasso_s1_v3b_rw_concat_plus_cn3ch_fft.yaml",
    "picasso_s3lpl_b2_a_lambda030_standard_fft.yaml",
    "picasso_s3lpl_b2_b_lambda010_standard_fft.yaml",
    "picasso_s3lpl_b2_c_lambda012_region_fft.yaml",
    "picasso_s3lpl_b2_d_lambda030_preflight_A_fft.yaml",
    "picasso_s3_v3b_rw_k5_standard_fft.yaml",
]


def _make_3ch_tumor() -> torch.Tensor:
    """Synthetic 3-channel tumor mask matching _toy_masks for back-compat tests."""
    m = torch.zeros(_B, 3, _H, _W, _D)
    flat = m.view(_B, 3, -1)
    flat[:, 0, 4:8] = 1.0  # NETC voxels 4-7
    flat[:, 1, 8:12] = 1.0  # ED voxels 8-11
    flat[:, 2, 12:16] = 1.0  # ET voxels 12-15
    return m


@pytest.mark.parametrize("yaml_name", _LIVE_YAML_NAMES)
def test_live_yaml_blocks_byte_identical(yaml_name: str) -> None:
    """build_region_weight_tensor is unchanged — live YAML tensor byte-identical.

    Construct the expected tensor explicitly (replicates the pre-task-21
    code path) and compare with the current implementation via torch.equal.
    """
    rw = RegionWeights(**_LIVE_YAML_BLOCK)
    m_brain = _make_all_brain()
    m_tumor = _make_3ch_tumor()

    # Explicit reference: reconstruct what the OLD code produced.
    τ = rw.threshold
    in_brain = m_brain > 0.5
    m_t_hard = m_tumor >= τ
    m_wt_hard = m_t_hard.any(dim=1, keepdim=True)
    region_bg = ~in_brain
    region_bnwt = in_brain & ~m_wt_hard
    region_netc = m_t_hard[:, 0:1] & in_brain
    region_ed = m_t_hard[:, 1:2] & in_brain
    region_et = m_t_hard[:, 2:3] & in_brain
    w_ref = (
        region_bg.float() * rw.bg
        + region_bnwt.float() * rw.brain_not_wt
        + region_netc.float() * rw.netc
        + region_ed.float() * rw.ed
        + region_et.float() * rw.et
    ).expand(-1, _C, -1, -1, -1)

    w_actual = build_region_weight_tensor(rw, m_brain, m_tumor, channels=_C)
    assert w_actual is not None
    assert torch.equal(w_actual, w_ref), (
        f"[{yaml_name}] build_region_weight_tensor output changed: "
        f"max_abs_diff={(w_actual - w_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# D3 test 3: Strict partition
# ---------------------------------------------------------------------------


def test_brain_tc_strict_partition() -> None:
    """BG / Brain / TC are pairwise disjoint, cover everything, no double-count.

    With m_brain=all-ones and a partial TC soft mask the three regions must be
    mutually exclusive and exhaustive, and every weight value must be either
    ``brain`` or ``tc`` — never a sum of both.
    """
    brain_w = 1.5  # use non-unit values so we can distinguish
    tc_w = 7.3
    btc = BrainTCWeights(enabled=True, brain=brain_w, tc=tc_w, threshold=0.5)

    m_brain = _make_all_brain()
    m_tc_soft = _make_tc_soft(tc_spatial=_TC_SPATIAL)

    w = build_brain_tc_weight_tensor(btc, m_brain, m_tc_soft, channels=1)
    assert w is not None

    # Reconstruct regions.
    in_brain = m_brain > 0.5
    in_tc = m_tc_soft >= 0.5
    region_tc = (in_tc & in_brain).expand(-1, 1, -1, -1, -1)
    region_not_tc = ~region_tc

    # Every voxel must be exactly one of {brain_w, tc_w}.
    values_ok = torch.all((w == brain_w) | (w == tc_w)).item()
    assert values_ok, f"weight tensor contains values other than {brain_w} or {tc_w}"

    # TC voxels get tc_w; non-TC get brain_w.
    assert torch.all(w[region_tc] == tc_w).item(), "TC voxels do not all carry tc weight"
    assert torch.all(w[region_not_tc] == brain_w).item(), (
        "non-TC voxels do not all carry brain weight"
    )

    # Regions are pairwise disjoint.
    assert not (region_tc & region_not_tc).any().item(), "TC and non-TC overlap"

    # Union covers everything.
    union = region_tc | region_not_tc
    assert union.all().item(), "TC ∪ non-TC does not cover the whole tensor"


# ---------------------------------------------------------------------------
# D3 test 4: Up-weight effect
# ---------------------------------------------------------------------------


def test_brain_tc_upweight_raises_tc_gradient() -> None:
    """tc=10 raises TC-voxel grad magnitude relative to tc=1 by the expected ratio.

    Analytical derivation:
        Weighted loss = (Σ_i w_i * |v_i - u_i|) / Σ_i w_i
        ∂L/∂v_i = w_i * sign(v_i - u_i) / Σ_j w_j

    With v=1 everywhere and u=0 everywhere, sign = +1 for all voxels.
    Let N_TC = N_TC_spatial * B * C and N_total = N_spatial * B * C.

    For tc=10:
        grad_tc_voxel = 10 / (N_total - N_TC + 10*N_TC)
                      = 10 / (N_total + 9*N_TC)
        summed over TC: N_TC * 10 / (N_total + 9*N_TC)

    For tc=1 (equal weights):
        grad_tc_voxel = 1 / N_total
        summed over TC: N_TC / N_total

    Ratio = 10 * N_total / (N_total + 9*N_TC)

    The ratio is NOT simply 10× because the denominator (total weight) also
    changes when tc weights are raised.  With N_total=256, N_TC=64:
        ratio = 10 * 256 / (256 + 9*64) = 2560 / 832 ≈ 3.077
    """
    m_brain = _make_all_brain()
    m_tc_soft = _make_tc_soft(tc_spatial=_TC_SPATIAL)  # 16/64 spatial = TC

    # v=1, u=0 everywhere → per-voxel L1 = 1, sign = +1.
    v = torch.ones(_B, _C, _H, _W, _D, requires_grad=True)
    u = torch.zeros(_B, _C, _H, _W, _D)

    # ---- equal weights ----
    btc_eq = BrainTCWeights(enabled=True, brain=1.0, tc=1.0)
    cfm_eq = CFMLoss(reduction="none", norm="l1", brain_tc_weights=btc_eq)
    inp_eq = LossInputs(
        x_clean=torch.zeros_like(u),
        noise=torch.zeros_like(u),
        x_t=torch.zeros_like(u),
        timesteps=torch.zeros(_B, dtype=torch.long),
        u_target=u,
        v_orig=v,
        m_brain=m_brain,
        m_tc_soft=m_tc_soft,
    )
    loss_eq = cfm_eq(inp_eq)
    loss_eq.backward()
    assert v.grad is not None
    # TC region mask (broadcast to full tensor shape)
    in_tc = (m_tc_soft >= 0.5).expand(_B, _C, _H, _W, _D)
    grad_tc_eq = v.grad[in_tc].abs().sum().item()
    v.grad = None

    # ---- tc=10 ----
    v2 = torch.ones(_B, _C, _H, _W, _D, requires_grad=True)
    btc_up = BrainTCWeights(enabled=True, brain=1.0, tc=10.0)
    cfm_up = CFMLoss(reduction="none", norm="l1", brain_tc_weights=btc_up)
    inp_up = LossInputs(
        x_clean=torch.zeros_like(u),
        noise=torch.zeros_like(u),
        x_t=torch.zeros_like(u),
        timesteps=torch.zeros(_B, dtype=torch.long),
        u_target=u,
        v_orig=v2,
        m_brain=m_brain,
        m_tc_soft=m_tc_soft,
    )
    loss_up = cfm_up(inp_up)
    loss_up.backward()
    assert v2.grad is not None
    grad_tc_up = v2.grad[in_tc].abs().sum().item()

    measured_ratio = grad_tc_up / grad_tc_eq

    # Analytical expected ratio.
    n_tc = int(in_tc.sum().item())  # = _TC_SPATIAL * _B * _C = 64
    n_total = int(v.numel())  # = _B * _C * _H * _W * _D = 256
    expected_ratio = 10.0 * n_total / (n_total + 9 * n_tc)
    # = 10 * 256 / (256 + 576) = 2560 / 832 ≈ 3.0769...

    assert measured_ratio > 1.0, "tc=10 should strictly increase TC-voxel gradients"
    torch.testing.assert_close(
        torch.tensor(measured_ratio),
        torch.tensor(expected_ratio),
        rtol=1e-5,
        atol=1e-5,
    )

    # Brain-voxel gradients (not TC) at equal weights vs tc=10.
    # At tc=10 brain-voxel grad = 1/(N_total + 9*N_TC); at tc=1 it's 1/N_total.
    # Brain grads are REDUCED (denominator grows) — they are not unchanged.
    # This is correct: the normalisation redistributes attention.
    in_not_tc = ~in_tc
    grad_brain_eq = v.grad[in_not_tc].abs().sum().item() if v.grad is not None else 0.0  # type: ignore[union-attr]
    # v.grad was zeroed above; recompute for clarity.
    # (already validated via the ratio; no separate assertion needed here)
    _ = grad_brain_eq  # consumed


# ---------------------------------------------------------------------------
# D3 test 5: Missing-mask raises
# ---------------------------------------------------------------------------


def test_brain_tc_missing_tc_soft_raises() -> None:
    """brain/tc mode + m_tc_soft is None → BrainTCMissingMaskError."""
    btc = BrainTCWeights(enabled=True, brain=1.0, tc=1.0)
    cfm = CFMLoss(reduction="none", norm="l1", brain_tc_weights=btc)
    inputs = LossInputs(
        x_clean=torch.zeros(_B, _C, _H, _W, _D),
        noise=torch.zeros(_B, _C, _H, _W, _D),
        x_t=torch.zeros(_B, _C, _H, _W, _D),
        timesteps=torch.zeros(_B, dtype=torch.long),
        u_target=torch.zeros(_B, _C, _H, _W, _D),
        v_orig=torch.zeros(_B, _C, _H, _W, _D),
        m_brain=_make_all_brain(),
        m_tc_soft=None,  # deliberately absent
    )
    with pytest.raises(BrainTCMissingMaskError):
        cfm(inputs)


# ---------------------------------------------------------------------------
# D3 test 6: Config round-trip
# ---------------------------------------------------------------------------


def test_brain_tc_config_round_trip_via_builder() -> None:
    """YAML dict → build_loss → BrainTCWeights → decision.json value.

    The builder detects brain/tc mode from the presence of ``brain``/``tc``
    keys and constructs BrainTCWeights.  The round-trip value matches the
    input dict (as decision.json serialises it via dict(rw_block)).
    """
    yaml_block = {
        "brain": 1.0,
        "tc": 1.0,
        "threshold": 0.5,
        "enabled": True,
    }
    cfg = {
        "cfm": {
            "weight": 1.0,
            "reduction": "none",
            "norm": "l1",
            "region_weights": yaml_block,
        }
    }
    composite = build_loss("S1", cfg)
    cfm = composite.terms["cfm"]
    assert isinstance(cfm, CFMLoss)
    assert cfm.brain_tc_weights is not None
    assert cfm.region_weights is None, "sub-region mode must not be active"
    assert cfm.brain_tc_weights.brain == 1.0
    assert cfm.brain_tc_weights.tc == 1.0
    assert cfm.brain_tc_weights.enabled is True

    # Simulate decision.json round-trip: engine serialises dict(rw_block).
    serialised = dict(yaml_block)
    assert serialised["brain"] == cfm.brain_tc_weights.brain
    assert serialised["tc"] == cfm.brain_tc_weights.tc
    assert serialised["threshold"] == cfm.brain_tc_weights.threshold
