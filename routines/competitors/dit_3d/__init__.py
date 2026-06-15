"""3D-DiT competitor benchmark routine.

Citation: Peebles & Xie 2023 (DiT) + Eidex *et al.* 2025 §4 (3D adaptation
+ MAISI-latent training recipe).
"""

from __future__ import annotations

from .engine import DiT3DCompetitorConfig, DiT3DCompetitorEngine

__all__ = ["DiT3DCompetitorConfig", "DiT3DCompetitorEngine"]
