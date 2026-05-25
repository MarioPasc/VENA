"""Registry of concrete vessel-prior models.

Register a new model by importing the class here and adding it to
``MODEL_REGISTRY``. The routine resolves model classes by their string ``name``
attribute as declared in the YAML config.
"""

from __future__ import annotations

from vena.vessel_priors.abc_model import AbstractVesselModel

from .frangi import FrangiVesselModel

MODEL_REGISTRY: dict[str, type[AbstractVesselModel]] = {
    FrangiVesselModel.name: FrangiVesselModel,
}

__all__ = ["MODEL_REGISTRY", "FrangiVesselModel"]
