"""Unit tests for the RegionResolver."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.metrics import RegionResolver, RegionSpec


def _make_batch(B: int = 1) -> dict[str, torch.Tensor]:
    m = torch.zeros(B, 1, 4, 4, 4, dtype=torch.float32)
    m[:, :, 2, 2, 2] = 1.0  # single voxel
    return {"m_wt": m}


@pytest.mark.unit
def test_resolver_derives_wt_dilated_and_bg() -> None:
    specs = {
        "brain": RegionSpec(source="fallback_all_ones"),
        "wt": RegionSpec(source="derived_from_tumor_latent", threshold=0.5),
        "wt_dilated": RegionSpec(source="derived_via_scipy_binary_dilation", structure="ones_3x3x3"),
        "bg": RegionSpec(source="derived"),
        "vessel": RegionSpec(source="skipped"),
    }
    resolver = RegionResolver(specs=specs)
    masks = resolver.resolve(_make_batch())

    assert masks.brain is not None and masks.brain.all()
    assert masks.wt is not None and masks.wt.sum() == 1
    # 3x3x3 dilation of a 1-voxel mask → 27 voxels.
    assert masks.wt_dilated is not None
    assert masks.wt_dilated.sum() == 27
    # bg = brain & ~wt_dilated → 64 - 27 = 37
    assert masks.bg is not None
    assert masks.bg.sum() == 64 - 27
    assert masks.vessel is None


@pytest.mark.unit
def test_resolver_rejects_missing_required_region() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        RegionResolver(specs={"wt": RegionSpec(source="derived_from_tumor_latent")})
