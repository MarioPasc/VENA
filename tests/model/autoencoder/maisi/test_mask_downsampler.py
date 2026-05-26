"""Unit tests for the abstract mask-downsampler contract and the per-class
average-pool model.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.autoencoder.maisi.encode.masks import (
    AbstractMaskDownsampler,
    PerClassAvgPoolDownsampler,
    UnknownDownsamplerError,
    get_downsampler,
)
from vena.model.autoencoder.maisi.encode.masks.shared.exceptions import (
    LabelCodeError,
)
from vena.model.autoencoder.maisi.exceptions import ShapeContractError


@pytest.mark.unit
def test_registry_returns_per_class_avg_pool() -> None:
    ds = get_downsampler("per_class_avg_pool")
    assert isinstance(ds, PerClassAvgPoolDownsampler)
    assert ds.output_channels == 3
    assert ds.channel_names == ("NETC", "ED", "ET")


@pytest.mark.unit
def test_registry_rejects_unknown_name() -> None:
    with pytest.raises(UnknownDownsamplerError):
        get_downsampler("nonexistent")


@pytest.mark.unit
def test_avg_pool_shape_and_dtype() -> None:
    ds = PerClassAvgPoolDownsampler(spatial_compression=4, depth_pad_base=8)
    # 16×16×12 input → pad depth to 16 → pool /4 → (4, 4, 4)
    mask = torch.zeros((1, 1, 16, 16, 12), dtype=torch.int64)
    mask[0, 0, 4:8, 4:8, 4:8] = 4  # ET cube
    out = ds.downsample(mask, target_shape=(4, 4, 4))
    assert out.shape == (1, 3, 4, 4, 4)
    assert out.dtype == torch.float32
    # NETC and ED channels are zero; ET is nonzero in the centre.
    assert out[0, 0].abs().sum().item() == 0.0
    assert out[0, 1].abs().sum().item() == 0.0
    assert out[0, 2, 1:2, 1:2, 1:2].item() == pytest.approx(1.0)


@pytest.mark.unit
def test_avg_pool_label_codes_strict() -> None:
    ds = PerClassAvgPoolDownsampler()
    mask = torch.zeros((1, 1, 8, 8, 8), dtype=torch.int64)
    mask[0, 0, 0, 0, 0] = 7  # unknown
    with pytest.raises(LabelCodeError):
        ds.downsample(mask, target_shape=(2, 2, 2))


@pytest.mark.unit
def test_avg_pool_rejects_wrong_input_rank() -> None:
    ds = PerClassAvgPoolDownsampler()
    with pytest.raises(ShapeContractError):
        ds.downsample(torch.zeros((1, 8, 8, 8), dtype=torch.int64), target_shape=(2, 2, 2))


@pytest.mark.unit
def test_avg_pool_to_attrs_round_trip() -> None:
    ds = PerClassAvgPoolDownsampler(spatial_compression=4, depth_pad_base=8)
    attrs = ds.to_attrs()
    assert attrs["name"] == "per_class_avg_pool"
    assert attrs["output_channels"] == 3
    assert attrs["channel_names"] == ["NETC", "ED", "ET"]
    assert attrs["label_codes"] == {"NETC": 1, "ED": 2, "ET": 4}
    assert attrs["spatial_compression"] == 4


@pytest.mark.unit
def test_concrete_subclass_missing_class_attrs_raises() -> None:
    """A concrete subclass that omits the class-level contract attrs must
    raise ``TypeError`` at class-construction time."""
    with pytest.raises(TypeError):
        class _BadMissingChannels(AbstractMaskDownsampler):
            # output_channels left at default 0 → invalid.
            output_dtype = torch.float32
            channel_names = ()
            name = "bad_missing_channels"

            def downsample(self, mask, target_shape):  # type: ignore[override]
                return mask

            def to_attrs(self):  # type: ignore[override]
                return {}


@pytest.mark.unit
def test_concrete_subclass_channel_count_mismatch_raises() -> None:
    """``len(channel_names)`` must match ``output_channels``."""
    with pytest.raises(TypeError):
        class _BadChannelCount(AbstractMaskDownsampler):
            output_channels = 2
            output_dtype = torch.float32
            channel_names = ("A", "B", "C")  # 3 names for 2 channels
            name = "bad_channel_count"

            def downsample(self, mask, target_shape):  # type: ignore[override]
                return mask

            def to_attrs(self):  # type: ignore[override]
                return {}
