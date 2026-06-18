"""Latent Perceptual Loss (LPL) — decoder-feature perceptual stage primitives.

Public surface
--------------
* :class:`LplConfig` — Pydantic recipe schema, round-trippable via
  ``from_yaml``.
* :class:`FeatureStatsEMA` — running per-block per-channel mean/var used
  for shared standardisation (§3.3).
* :class:`LplLoss` — the gated, region-weighted, depth-weighted feature
  distance.
* :func:`decoder_feature_extractor` — context manager that wraps
  ``post_quant_conv`` + :func:`vena.common.partial_decode` into one
  callable.
* :func:`resample_region_to_block`, :func:`soft_wt_from_tumor_latent`,
  :func:`region_weight_map` — mask plumbing.

Implementation lives in:

* :mod:`vena.model.fm.lpl.config`
* :mod:`vena.model.fm.lpl.feature_stats`
* :mod:`vena.model.fm.lpl.region`
* :mod:`vena.model.fm.lpl.hooks`
* :mod:`vena.model.fm.lpl.loss`

PR-1 ships these primitives + the ``decoder_lpl_profile`` preflight that
consumes them. The S3 training-step wiring + CompositeLoss S3 stage lands
in a follow-up after the preflight measures w_l / A / t_min / region
recipe (per ``.claude/notes/changes/decoder_perceptual_loss_s3.md`` §4.7c).
"""

from __future__ import annotations

from .config import LplConfig
from .feature_stats import FeatureStatsEMA
from .hooks import decoder_feature_extractor
from .loss import LplLoss
from .region import (
    region_weight_map,
    resample_region_to_block,
    soft_wt_from_tumor_latent,
)

__all__ = [
    "FeatureStatsEMA",
    "LplConfig",
    "LplLoss",
    "decoder_feature_extractor",
    "region_weight_map",
    "resample_region_to_block",
    "soft_wt_from_tumor_latent",
]
