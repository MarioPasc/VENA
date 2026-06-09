"""``vena-encode-brain-to-latent`` — add ``masks/brain_latent`` to a latent H5.

Single-cohort post-hoc patcher for the 2026-06-09 training-regime overhaul
(see ``.claude/notes/changes/2026-06-09_training-regime-overhaul.md``
CHANGE 2). Reads the image-domain ``masks/brain`` from the source image H5,
applies the same brain-centred crop box used by the latent encoder, max-pools
4×4×4 to the latent grid, and writes the result as ``masks/brain_latent``
(``int8``, shape ``(1, 48, 56, 48)``) on the latent H5.

Idempotent on re-run: rows whose dataset already carries valid data are
skipped unless ``overwrite: true``. v4 augmented rows are written as all-ones
with a per-dataset ``v4_brain_synthesised_ones`` attr flag (replaying the
elastic+affine transform from ``aug_params_json`` is a follow-up).
"""

from __future__ import annotations

from .engine import (
    BrainToLatentRoutineConfig,
    BrainToLatentRoutineEngine,
)

__all__ = [
    "BrainToLatentRoutineConfig",
    "BrainToLatentRoutineEngine",
]
