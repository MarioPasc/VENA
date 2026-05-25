"""Registry of SWI preprocessors used before vessel-prior models.

Register a new preprocessor by importing the class here and adding it to
``PREPROCESSOR_REGISTRY``. The engine resolves preprocessors by their string
``name`` attribute as declared in the YAML config under
``algorithms[*].preprocessing[*].name``.
"""

from __future__ import annotations

from .abc_preprocessor import AbstractPreprocessor, PreprocessingError
from .anisotropic_diffusion import AnisotropicDiffusionPreprocessor
from .clahe import CLAHEPreprocessor

PREPROCESSOR_REGISTRY: dict[str, type[AbstractPreprocessor]] = {
    CLAHEPreprocessor.name: CLAHEPreprocessor,
    AnisotropicDiffusionPreprocessor.name: AnisotropicDiffusionPreprocessor,
}

__all__ = [
    "PREPROCESSOR_REGISTRY",
    "AbstractPreprocessor",
    "AnisotropicDiffusionPreprocessor",
    "CLAHEPreprocessor",
    "PreprocessingError",
]
