"""Unit tests for the V3 normalisation variant registry."""

from __future__ import annotations

import pytest
import torch

from vena.preflight.normalization_audit import (
    get_variant_registry,
    joint_modality_percentile_normalise,
)

pytestmark = pytest.mark.unit


def _fake_patient(seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Synthetic 4-modality patient.

    * Brain mask covers the central 12³ region (1728 voxels).
    * T1pre brain intensity follows |N(0, 100)| + 500 (bounded ≥ 500).
    * T1c equals 1.2 × T1pre everywhere (multiplicative inter-modality
      gap — V0 destroys it, V4 preserves it) PLUS a small bright tail of
      ~0.05 % of brain at value 3000 (above the 99.5 %ile, below the
      99.99 %ile so V0/V1/V2/V3/V8 diverge cleanly on the tail).
    * T2 = 0.8 × T1pre, FLAIR = 0.9 × T1pre.
    """
    rng = torch.Generator().manual_seed(seed)
    H, W, D = 16, 16, 16
    brain = torch.zeros((1, 1, H, W, D))
    brain[:, :, 2:14, 2:14, 2:14] = 1.0
    n_brain = int(brain.sum().item())
    t1pre = torch.zeros((1, 1, H, W, D))
    t1pre[brain > 0] = torch.randn(n_brain, generator=rng).abs() * 100 + 500
    t1c = torch.zeros_like(t1pre)
    t1c[brain > 0] = t1pre[brain > 0] * 1.2
    coords = torch.nonzero(brain.squeeze())
    n_bright = max(1, n_brain // 2000)  # ≈ 0.05 %
    for cx, cy, cz in coords[:n_bright]:
        t1c[0, 0, cx, cy, cz] = 3000.0
    t2 = torch.zeros_like(t1pre)
    t2[brain > 0] = t1pre[brain > 0] * 0.8
    flair = torch.zeros_like(t1pre)
    flair[brain > 0] = t1pre[brain > 0] * 0.9
    images = {"t1pre": t1pre, "t1c": t1c, "t2": t2, "flair": flair}
    return images, brain


@pytest.mark.parametrize("variant_id", ["V0", "V1", "V2", "V3", "V4", "V7", "V8"])
def test_variant_apply_preserves_modality_keys(variant_id: str) -> None:
    images, brain = _fake_patient()
    variant = get_variant_registry()[variant_id]
    out = variant.apply(images, {"brain": brain})
    assert set(out.keys()) == set(images.keys())
    for k, x in out.items():
        assert x.shape == images[k].shape
        assert x.dtype == images[k].dtype


@pytest.mark.parametrize("variant_id", ["V0", "V1", "V2", "V3", "V4", "V7", "V8"])
def test_variant_apply_finite(variant_id: str) -> None:
    images, brain = _fake_patient()
    variant = get_variant_registry()[variant_id]
    out = variant.apply(images, {"brain": brain})
    for x in out.values():
        assert torch.isfinite(x).all()


@pytest.mark.parametrize("variant_id", ["V0", "V2", "V3", "V4", "V8"])
def test_clip_variants_stay_in_unit_range(variant_id: str) -> None:
    """V0/V2/V3/V4/V8 all set clip=True so output ∈ [0, 1] on foreground."""
    images, brain = _fake_patient()
    variant = get_variant_registry()[variant_id]
    out = variant.apply(images, {"brain": brain})
    for x in out.values():
        assert x.min().item() >= 0.0 - 1e-6
        assert x.max().item() <= 1.0 + 1e-6


@pytest.mark.parametrize("variant_id", ["V1", "V7"])
def test_no_clip_variants_can_exceed_one(variant_id: str) -> None:
    """V1 and V7 have clip=False — the T1c bright tail must exceed 1.0."""
    images, brain = _fake_patient()
    variant = get_variant_registry()[variant_id]
    out = variant.apply(images, {"brain": brain})
    assert out["t1c"].max().item() > 1.0


def test_v0_v1_agree_on_in_range_voxels() -> None:
    """V0 and V1 share the same (lo, hi); they differ only on the clipped tail."""
    images, brain = _fake_patient()
    v0 = get_variant_registry()["V0"]
    v1 = get_variant_registry()["V1"]
    o0 = v0.apply(images, {"brain": brain})
    o1 = v1.apply(images, {"brain": brain})
    # On voxels where V1 stays in [0, 1] (the non-tail region), V0 and V1 must match.
    for k in images.keys():
        in_range = (o1[k] >= 0.0) & (o1[k] <= 1.0)
        assert torch.allclose(o0[k][in_range], o1[k][in_range], rtol=1e-5, atol=1e-5)


def test_v4_preserves_intermodality_scale() -> None:
    """V4 must keep T1c brighter than T1pre **on average** across brain.

    The synthetic patient has T1c = 1.2 × T1pre everywhere on brain (plus a
    tiny enhancement tail). After per-modality normalisation (V0) each
    modality is independently mapped into [0, 1] — the 1.2× gap is gone.
    Joint-modality (V4) uses one (lo, hi) over the union of all modalities'
    foreground voxels, so the 1.2× multiplicative gap survives.
    """
    images, brain = _fake_patient()
    v0 = get_variant_registry()["V0"]
    v4 = get_variant_registry()["V4"]
    o0 = v0.apply(images, {"brain": brain})
    o4 = v4.apply(images, {"brain": brain})
    b = brain.squeeze() > 0
    gap_v0 = float((o0["t1c"].squeeze() - o0["t1pre"].squeeze())[b].mean().item())
    gap_v4 = float((o4["t1c"].squeeze() - o4["t1pre"].squeeze())[b].mean().item())
    # V0: per-modality clip destroys the multiplicative gap → mean diff ≈ 0.
    assert abs(gap_v0) < 0.05
    # V4: preserves the gap → mean T1c clearly > mean T1pre on brain.
    assert gap_v4 > 0.05


def test_v8_asymmetric_per_modality() -> None:
    """V8: T1c at (0, 99.9), other modalities at (0, 99.5).

    The T1c headroom must be wider than V0's, but T2/FLAIR/T1pre histograms
    are unchanged vs V0.
    """
    images, brain = _fake_patient()
    v0 = get_variant_registry()["V0"]
    v8 = get_variant_registry()["V8"]
    o0 = v0.apply(images, {"brain": brain})
    o8 = v8.apply(images, {"brain": brain})
    # T1pre / T2 / FLAIR identical (or near-identical) between V0 and V8.
    for k in ["t1pre", "t2", "flair"]:
        assert torch.allclose(o0[k], o8[k], rtol=1e-5, atol=1e-5)
    # T1c may differ (V8 uses a higher upper percentile → different scale).
    assert not torch.allclose(o0["t1c"], o8["t1c"])


def test_joint_normaliser_rejects_shape_mismatch() -> None:
    from vena.model.autoencoder.maisi.exceptions import ShapeContractError

    imgs = {
        "a": torch.zeros((1, 1, 8, 8, 8)),
        "b": torch.zeros((1, 1, 8, 8, 4)),  # mismatched D
    }
    with pytest.raises(ShapeContractError):
        joint_modality_percentile_normalise(imgs)


def test_joint_normaliser_empty_dict_rejected() -> None:
    from vena.model.autoencoder.maisi.exceptions import ShapeContractError

    with pytest.raises(ShapeContractError):
        joint_modality_percentile_normalise({})


def test_registry_version_pinned() -> None:
    """Every registered variant has a non-empty variant_version pin."""
    reg = get_variant_registry()
    assert "V0" in reg
    for vid, variant in reg.items():
        assert variant.variant_version, f"variant {vid} missing version pin"
        assert variant.description, f"variant {vid} missing description"
        assert variant.params, f"variant {vid} missing params"
