"""Registry of concrete perfusion-prior models.

Register a new model by importing the class here and adding it to
``MODEL_REGISTRY``. The routine resolves model classes by their string ``name``
attribute as declared in the YAML config.
"""

from __future__ import annotations

from vena.prior_maps.perfusion_priors.abc_model import AbstractPerfusionModel

from .alsop2015 import Alsop2015PerfusionModel

MODEL_REGISTRY: dict[str, type[AbstractPerfusionModel]] = {
    Alsop2015PerfusionModel.name: Alsop2015PerfusionModel,
}

__all__ = ["MODEL_REGISTRY", "Alsop2015PerfusionModel"]
