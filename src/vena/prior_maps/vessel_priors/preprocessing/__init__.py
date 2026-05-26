"""Registry of SWI preprocessors used before vessel-prior models.

Register a new preprocessor by importing the class here and adding it to
``PREPROCESSOR_REGISTRY``. The engine resolves preprocessors by their string
``name`` attribute as declared in the YAML config under
``algorithms[*].preprocessing[*].name``.
"""

from __future__ import annotations

from vena.prior_maps.vessel_priors.abc_preprocessor import (
    AbstractPreprocessor,
    PreprocessingError,
)

from .clahe import CLAHEPreprocessor

PREPROCESSOR_REGISTRY: dict[str, type[AbstractPreprocessor]] = {
    CLAHEPreprocessor.name: CLAHEPreprocessor,
}

# AnisotropicDiffusionPreprocessor depends on the `itk` (TubeTK) Python package.
# Register it only when the import succeeds so the rest of the subsystem stays
# usable in environments without ITK installed.
try:
    from .anisotropic_diffusion import AnisotropicDiffusionPreprocessor

    PREPROCESSOR_REGISTRY[AnisotropicDiffusionPreprocessor.name] = AnisotropicDiffusionPreprocessor
except ImportError:
    AnisotropicDiffusionPreprocessor = None  # type: ignore[assignment,misc]

__all__ = [
    "PREPROCESSOR_REGISTRY",
    "AbstractPreprocessor",
    "AnisotropicDiffusionPreprocessor",
    "CLAHEPreprocessor",
    "PreprocessingError",
]
