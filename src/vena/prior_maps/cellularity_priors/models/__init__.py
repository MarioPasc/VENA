"""Registry of concrete cellularity-prior models."""

from __future__ import annotations

from vena.prior_maps.cellularity_priors.abc_model import AbstractCellularityModel

from .nawm_normalized import NAWMNormalizedCellularityModel

MODEL_REGISTRY: dict[str, type[AbstractCellularityModel]] = {
    NAWMNormalizedCellularityModel.name: NAWMNormalizedCellularityModel,
}

__all__ = ["MODEL_REGISTRY", "NAWMNormalizedCellularityModel"]
