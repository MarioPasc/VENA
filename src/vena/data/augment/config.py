"""YAML schema + loader for the augmentation pipeline.

The YAML config is the single artefact the training routine points at to
enable augmentation. Schema (v1.0)::

    schema_version: 1.0
    seed: 0
    augmentations:
      - name: flip_lr
        p: 0.5
      - name: translate
        p: 0.5
        max_voxels: 8
        axes: [h, w, d]
      - name: rotate_yaw
        p: 0.3
        max_deg: 5.0

A preflight ``decision.json`` may be supplied to
:func:`build_pipeline_from_yaml` as a gate: the loader fast-fails if any
requested augmentation is not in the decision's
``latent_safe_augmentations`` list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.augment.base import LatentAugmentationError
from vena.data.augment.pipeline import AugmentationPipeline
from vena.data.augment.transforms import REGISTRY

logger = logging.getLogger(__name__)

# Schema version of the YAML config. Bumped on breaking changes.
SCHEMA_VERSION: str = "1.0"


class AugmentationEntryConfig(BaseModel):
    """One augmentation block in the YAML.

    The ``name`` selects the operator class from
    :data:`vena.data.augment.transforms.REGISTRY`; every other key is
    forwarded as a keyword argument to the operator constructor.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    p: float = 0.5


class AugmentationConfig(BaseModel):
    """Root config object loaded from the augmentation YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    seed: int = 0
    augmentations: list[AugmentationEntryConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> AugmentationConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


def _instantiate_entry(entry: AugmentationEntryConfig) -> Any:
    """Map one YAML entry to a concrete :class:`LatentAugmentation`."""
    if entry.name not in REGISTRY:
        raise LatentAugmentationError(
            f"unknown augmentation {entry.name!r}; available: {sorted(REGISTRY)}"
        )
    cls = REGISTRY[entry.name]
    kwargs = entry.model_dump()
    kwargs.pop("name", None)
    return cls(**kwargs)


def _load_decision_allowlist(decision_path: Path | str) -> set[str]:
    """Read the preflight ``decision.json`` and return the allowlist."""
    decision_path = Path(decision_path)
    if not decision_path.is_file():
        raise LatentAugmentationError(f"preflight decision.json not found: {decision_path}")
    payload = json.loads(decision_path.read_text())
    if "latent_safe_augmentations" not in payload:
        raise LatentAugmentationError(
            f"preflight decision.json missing 'latent_safe_augmentations': {decision_path}"
        )
    return set(payload["latent_safe_augmentations"])


def build_pipeline_from_yaml(
    config_path: Path | str,
    preflight_decision_path: Path | str | None = None,
) -> AugmentationPipeline:
    """Construct a pipeline from the YAML config, with an optional gate.

    Parameters
    ----------
    config_path : Path | str
        Path to the augmentation YAML.
    preflight_decision_path : Path | str | None
        Optional path to the equivariance preflight's ``decision.json``.
        When supplied, every augmentation name in the YAML must appear in the
        decision's ``latent_safe_augmentations`` list; otherwise the loader
        raises :class:`LatentAugmentationError` naming the offending entries.

    Returns
    -------
    AugmentationPipeline
    """
    cfg = AugmentationConfig.from_yaml(config_path)
    if cfg.schema_version != SCHEMA_VERSION:
        raise LatentAugmentationError(
            f"augmentation YAML schema_version={cfg.schema_version!r} "
            f"(expected {SCHEMA_VERSION!r}): {config_path}"
        )
    if not cfg.augmentations:
        raise LatentAugmentationError(f"augmentation YAML lists no augmentations: {config_path}")

    if preflight_decision_path is not None:
        allow = _load_decision_allowlist(preflight_decision_path)
        offenders = [entry.name for entry in cfg.augmentations if entry.name not in allow]
        if offenders:
            raise LatentAugmentationError(
                f"augmentations {offenders} are not in the preflight allowlist "
                f"({sorted(allow)}); see {preflight_decision_path}"
            )

    augs = [_instantiate_entry(entry) for entry in cfg.augmentations]
    logger.info(
        "built AugmentationPipeline from %s with %d augmentation(s): %s",
        Path(config_path).name,
        len(augs),
        [a.name for a in augs],
    )
    return AugmentationPipeline(augs, seed=cfg.seed)
