"""Load and validate a corpus registry JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import CorpusRegistry, RegistryError

logger = logging.getLogger(__name__)


def load_registry(path: Path | str, *, require_latents: bool = True) -> CorpusRegistry:
    """Load a corpus registry and check that referenced H5 files exist.

    Parameters
    ----------
    path
        Path to the registry JSON.
    require_latents
        When ``True`` (training-time default), every cohort's ``latent_h5``
        must exist. Set ``False`` for steps that only need the image caches
        (e.g. before encoding).

    Returns
    -------
    CorpusRegistry
        The validated registry.

    Raises
    ------
    RegistryError
        If the file is missing, malformed, or any referenced H5 is absent.
    """
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"corpus registry not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RegistryError(f"corpus registry is not valid JSON: {path}: {exc}") from exc
    try:
        registry = CorpusRegistry.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise RegistryError(f"corpus registry failed schema validation: {exc}") from exc

    missing: list[str] = []
    for c in registry.cohorts:
        if not c.image_h5.is_file():
            missing.append(f"{c.name}: image_h5 {c.image_h5}")
        if require_latents and not c.latent_h5.is_file():
            missing.append(f"{c.name}: latent_h5 {c.latent_h5}")
    if missing:
        joined = "\n  - ".join(missing)
        raise RegistryError(
            f"corpus registry references missing H5 files:\n  - {joined}"
        )

    logger.info(
        "Loaded corpus '%s': %d cohorts (%d cv, %d test-only)",
        registry.name,
        len(registry.cohorts),
        len(registry.cv_cohorts()),
        len(registry.test_cohorts()),
    )
    return registry
