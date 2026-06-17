"""Decorator-based factory for inference adapters.

The factory is a process-global dict keyed by a short string (``"identity"``,
``"vena_fm"``, ``"pgan"``, ...) that maps to the adapter class. Adapter
modules declare their key with :func:`register_inference_model` at import
time; the routine engine looks classes up via :func:`get_inference_factory`
given the YAML's ``type`` field, then instantiates them with the YAML's
``kwargs`` block.

Re-registering the same key raises :class:`InferenceRegistryError` so a
typo in an adapter file is loud rather than silent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from vena.inference.base import InferenceModel

logger = logging.getLogger(__name__)


class InferenceRegistryError(Exception):
    """Raised on duplicate-key registration or unknown lookup."""


T = TypeVar("T", bound="type[InferenceModel]")


_REGISTRY: dict[str, type[InferenceModel]] = {}

# Lazy-import map for the built-in adapters. Each value is
# ``(module_path, class_name)``; resolution happens on first
# ``get_inference_factory(key)`` call. The point: a process that only runs
# SynDiff should never import the vena.competitors.{t1c_rflow,lddpm_3d,
# dit_3d,lpix2pix_3d,pgan_cgan,resvit} or vena.model.fm.* modules — which
# transitively pull MAISI/Lightning/sklearn deps that may be absent from a
# competitor-specific env (e.g. vena-syndiff has no sklearn / no pandas /
# no datetime.UTC). The lazy resolution keeps each env minimal.
_LAZY_REGISTRY: dict[str, tuple[str, str]] = {
    "identity": ("vena.inference.adapters.identity_adapter", "IdentityAdapter"),
    "pgan": ("vena.inference.adapters.pgan_adapter", "PGANAdapter"),
    "resvit": ("vena.inference.adapters.resvit_adapter", "ResViTAdapter"),
    "syndiff": ("vena.inference.adapters.syndiff_adapter", "SynDiffAdapter"),
    "dit_3d": ("vena.inference.adapters.dit3d_adapter", "DiT3DAdapter"),
    "t1c_rflow": ("vena.inference.adapters.t1c_rflow_adapter", "T1CRFlowAdapter"),
    "lddpm_3d": ("vena.inference.adapters.lddpm3d_adapter", "LDDPM3DAdapter"),
    "lpix2pix_3d": ("vena.inference.adapters.lpix2pix3d_adapter", "LPix2Pix3DAdapter"),
    "vena_fm": ("vena.inference.adapters.vena_fm_adapter", "VenaFMAdapter"),
}


def register_inference_model(model_type: str) -> Callable[[T], T]:
    """Decorator: register an :class:`InferenceModel` subclass under a key.

    Parameters
    ----------
    model_type
        The string the YAML registry uses in its ``type`` field.

    Returns
    -------
    Callable
        The class decorator. The class is returned unchanged so it can
        still be imported and instantiated directly by tests.

    Raises
    ------
    InferenceRegistryError
        If ``model_type`` is already registered with a different class.
    """

    def _wrap(cls: T) -> T:
        existing = _REGISTRY.get(model_type)
        if existing is not None and existing is not cls:
            raise InferenceRegistryError(
                f"inference adapter key {model_type!r} already registered "
                f"to {existing.__module__}.{existing.__name__}; cannot rebind "
                f"to {cls.__module__}.{cls.__name__}"
            )
        cls.model_type = model_type
        _REGISTRY[model_type] = cls
        logger.debug("registered inference adapter %r -> %s", model_type, cls.__name__)
        return cls

    return _wrap


def get_inference_factory(model_type: str) -> type[InferenceModel]:
    """Look up the adapter class for a YAML ``type`` field.

    Resolution order: live registry (set via the decorator at import time —
    used by tests + ad-hoc adapters) → lazy registry (built-ins, resolved
    on demand via :func:`importlib.import_module`).

    Raises
    ------
    InferenceRegistryError
        If ``model_type`` is neither in the live registry nor the lazy one.
    """
    cls = _REGISTRY.get(model_type)
    if cls is not None:
        return cls
    if model_type in _LAZY_REGISTRY:
        import importlib

        module_path, class_name = _LAZY_REGISTRY[model_type]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise InferenceRegistryError(
                f"inference adapter {model_type!r} could not be imported from "
                f"{module_path}: {exc}. The current Python env may be missing a "
                f"dependency this adapter pulls transitively (e.g. SynDiff env "
                f"lacks MAISI / Lightning)."
            ) from exc
        cls = getattr(module, class_name, None)
        if cls is None:
            raise InferenceRegistryError(
                f"inference adapter {model_type!r} resolved to "
                f"{module_path} but class {class_name} is not defined."
            )
        # The module's @register_inference_model decorator should have
        # populated _REGISTRY by now; mirror it defensively.
        _REGISTRY.setdefault(model_type, cls)
        return cls
    known = sorted(set(_REGISTRY) | set(_LAZY_REGISTRY))
    raise InferenceRegistryError(
        f"unknown inference adapter type {model_type!r}; known types: {known}"
    )


def list_registered() -> list[str]:
    """Return all known model_type keys (live + lazy), sorted."""
    return sorted(set(_REGISTRY) | set(_LAZY_REGISTRY))


def _clear_registry_for_tests() -> None:
    """Reset the registry — intended for unit tests only.

    Production callers must rely on import-time registration. Tests that
    register a synthetic ``_Dummy`` adapter call this in a fixture
    teardown so adjacent tests see the clean default registry.
    """
    _REGISTRY.clear()


__all__ = [
    "InferenceRegistryError",
    "get_inference_factory",
    "list_registered",
    "register_inference_model",
]
