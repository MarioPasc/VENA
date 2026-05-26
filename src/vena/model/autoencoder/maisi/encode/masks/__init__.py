"""Mask downsamplers: image-space → MAISI-latent-space, pluggable by name."""

from .abc_model import AbstractMaskDownsampler
from .models import (
    PerClassAvgPoolDownsampler,
    available_downsamplers,
    get_downsampler,
)
from .shared import (
    LabelCodeError,
    MaskDownsamplerError,
    UnknownDownsamplerError,
    assert_image_space_mask,
    assert_label_codes_subset,
    assert_target_shape,
)

__all__ = [
    "AbstractMaskDownsampler",
    "LabelCodeError",
    "MaskDownsamplerError",
    "PerClassAvgPoolDownsampler",
    "UnknownDownsamplerError",
    "assert_image_space_mask",
    "assert_label_codes_subset",
    "assert_target_shape",
    "available_downsamplers",
    "get_downsampler",
]
