"""BraTS-GLI thin converter wrapper (aliases to the neutral base).

All converter logic lives in :mod:`vena.data.h5.latent_domain.convert`.
This module re-exports :class:`LatentH5Config` and :class:`LatentH5Converter`
under ``BraTSGLI``-prefixed aliases for discoverability.
"""

from __future__ import annotations

from vena.data.h5.latent_domain.convert import (
    LatentH5Config as BraTSGLILatentH5Config,
    LatentH5Converter as BraTSGLILatentH5Converter,
)

__all__ = [
    "BraTSGLILatentH5Config",
    "BraTSGLILatentH5Converter",
]
