"""Registry of concrete mask-downsampler models.

Adding a new model means three lines: import it here, list its class in
``_REGISTRY``, and let :func:`get_downsampler` look it up by ``name``.
"""

from __future__ import annotations

from typing import Any

from ..abc_model import AbstractMaskDownsampler
from ..shared.exceptions import UnknownDownsamplerError
from .per_class_avg_pool import PerClassAvgPoolDownsampler

_REGISTRY: dict[str, type[AbstractMaskDownsampler]] = {
    PerClassAvgPoolDownsampler.name: PerClassAvgPoolDownsampler,
}


def get_downsampler(name: str, **params: Any) -> AbstractMaskDownsampler:
    """Look up and instantiate a mask downsampler by registry name.

    Parameters
    ----------
    name : str
        Registry key. Currently registered: ``per_class_avg_pool``.
    **params
        Constructor kwargs forwarded to the resolved class.

    Raises
    ------
    UnknownDownsamplerError
        If ``name`` is not in :data:`_REGISTRY`.
    """
    if name not in _REGISTRY:
        known = sorted(_REGISTRY)
        raise UnknownDownsamplerError(
            f"unknown mask downsampler {name!r}; known: {known}"
        )
    return _REGISTRY[name](**params)


def available_downsamplers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "PerClassAvgPoolDownsampler",
    "available_downsamplers",
    "get_downsampler",
]
