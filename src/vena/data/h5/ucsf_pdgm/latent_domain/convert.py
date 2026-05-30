"""UCSF-PDGM thin converter wrapper (back-compat aliases).

All converter logic lives in :mod:`vena.data.h5.latent_domain.convert`.
This module re-exports :class:`LatentH5Config` and :class:`LatentH5Converter`
under their legacy ``UCSF_PDGM_*`` aliases so existing imports continue to work
without any change.
"""

from __future__ import annotations

from vena.data.h5.latent_domain.convert import (
    LatentH5Config as UCSFPDGMLatentH5Config,
    LatentH5Converter as UCSFPDGMLatentH5Converter,
)

__all__ = [
    "UCSFPDGMLatentH5Config",
    "UCSFPDGMLatentH5Converter",
]
