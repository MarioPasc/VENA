"""Inference samplers + timing probes for rectified-flow generation."""

from .base import BaseSampler, SamplerCallable
from .euler import EulerSampler
from .timing import NFETimingProbe

__all__ = ["BaseSampler", "EulerSampler", "NFETimingProbe", "SamplerCallable", "get_sampler"]


def get_sampler(name: str) -> type[BaseSampler]:
    """Registry lookup for sampler classes."""
    name = name.lower()
    if name == "euler":
        return EulerSampler
    raise ValueError(f"unknown sampler '{name}'; choose from {{euler}}")
