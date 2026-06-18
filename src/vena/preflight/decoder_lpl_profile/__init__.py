"""``decoder_lpl_profile`` preflight — measure the LPL recipe on real data.

Implements the §4.1 + §4.2 + §4.7b sweeps described in
``.claude/notes/changes/decoder_perceptual_loss_s3.md``. Emits the
``decision.json`` v1.0 contract the future S3 train YAML asserts.

Public surface
--------------
* :class:`DecoderLplProfileConfig` — Pydantic config schema.
* :class:`DecoderLplProfileEngine` — orchestrator (single-shard or
  shard-of-N runs).
* :func:`aggregate` — shard merge + decision.json + figures.
* :class:`DecoderLplDecisionV1` — Pydantic schema for the produced
  ``decision.json``.
* :func:`assert_decoder_lpl_decision_valid` — consumer-side validator.

The engine writes per-(patient, variant) rows into shard CSVs under
``<artifact_dir>/shard_{i}/tables/`` (single-stream runs write directly
under ``<artifact_dir>/tables/`` so aggregation degrades gracefully).
"""

from __future__ import annotations

from .aggregate import aggregate, update_latest_symlink
from .decision import (
    DECISION_PRODUCER,
    DECISION_SCHEMA_VERSION,
    DecoderLplDecisionV1,
    assert_decoder_lpl_decision_valid,
    write_decision_json,
)
from .engine import DecoderLplProfileConfig, DecoderLplProfileEngine

__all__ = [
    "DECISION_PRODUCER",
    "DECISION_SCHEMA_VERSION",
    "DecoderLplDecisionV1",
    "DecoderLplProfileConfig",
    "DecoderLplProfileEngine",
    "aggregate",
    "assert_decoder_lpl_decision_valid",
    "update_latest_symlink",
    "write_decision_json",
]
