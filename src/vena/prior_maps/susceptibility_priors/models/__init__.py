"""Registry of concrete susceptibility-prior models."""

from __future__ import annotations

from vena.prior_maps.susceptibility_priors.abc_model import (
    AbstractSusceptibilityModel,
)

from .magnitude_swan import MagnitudeSwanSusceptibilityModel

MODEL_REGISTRY: dict[str, type[AbstractSusceptibilityModel]] = {
    MagnitudeSwanSusceptibilityModel.name: MagnitudeSwanSusceptibilityModel,
}

__all__ = ["MODEL_REGISTRY", "MagnitudeSwanSusceptibilityModel"]
