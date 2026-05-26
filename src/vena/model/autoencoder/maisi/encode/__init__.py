"""Image → MAISI latent encoding."""

from .engine import EncodeResult, InferenceMode, MaisiEncoder
from .masks import (
    AbstractMaskDownsampler,
    LabelCodeError,
    MaskDownsamplerError,
    PerClassAvgPoolDownsampler,
    UnknownDownsamplerError,
    available_downsamplers,
    get_downsampler,
)

__all__ = [
    "AbstractMaskDownsampler",
    "EncodeResult",
    "InferenceMode",
    "LabelCodeError",
    "MaisiEncoder",
    "MaskDownsamplerError",
    "PerClassAvgPoolDownsampler",
    "UnknownDownsamplerError",
    "available_downsamplers",
    "get_downsampler",
]
