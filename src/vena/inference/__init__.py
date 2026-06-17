"""Unified inference surface for the VENA validation protocol.

This package implements the abstract contract every benchmarked method
(VENA-S1, VENA-S2, and the 7 competitor families C1..C7 + the C0 identity
floor) satisfies when producing predicted T1c volumes for the validation
H5 schema in ``.claude/notes/validation/validation_proposal.md`` §5.3.

The public surface is:

* :class:`InferenceModel` — the ABC every method implements
  (``setup``/``predict``/``teardown``).
* :class:`InferenceResult` — the immutable result returned by
  ``predict``: harmonised + raw prediction volume, wall-clock seconds,
  peak VRAM.
* :func:`register_inference_model` — decorator that registers an adapter
  class against a string key (used in the YAML ``type`` field).
* :func:`get_inference_factory` — factory lookup that the routine engine
  uses to instantiate an adapter from a registry entry.
* :func:`apply_harmonisation` — the §4.1 intensity-harmonisation
  contract every adapter applies before returning its result.

Adapter classes (one per method family) live under :mod:`vena.inference.adapters`.
Importing :mod:`vena.inference` fires the side-effect that registers every
built-in adapter via the decorator.
"""

from __future__ import annotations

# Importing the adapters subpackage triggers all @register_inference_model
# decorators so the factory dispatch is populated when downstream code does
# ``from vena.inference import get_inference_factory``.
from vena.inference import adapters as _adapters  # noqa: F401
from vena.inference.base import InferenceModel, InferenceResult
from vena.inference.harmonisation import apply_harmonisation
from vena.inference.registry import (
    InferenceRegistryError,
    get_inference_factory,
    list_registered,
    register_inference_model,
)

__all__ = [
    "InferenceModel",
    "InferenceRegistryError",
    "InferenceResult",
    "apply_harmonisation",
    "get_inference_factory",
    "list_registered",
    "register_inference_model",
]
