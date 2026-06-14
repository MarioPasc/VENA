"""pGAN-cGAN (Dar et al., 2019) — competitor benchmark routine.

Public exports:

- ``PGANCompetitorConfig`` — frozen Pydantic config.
- ``PGANCompetitorEngine`` — engine with ``run() -> Path``.
"""

from __future__ import annotations

from .engine import PGANCompetitorConfig, PGANCompetitorEngine

__all__ = ["PGANCompetitorConfig", "PGANCompetitorEngine"]
