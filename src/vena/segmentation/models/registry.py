"""Decorator-based registry mapping model name → builder factory.

The registry is a process-wide singleton.  Model modules register themselves
at import time via :func:`register_segmentation_model`; consumers instantiate
by name via :func:`get_segmentation_model`.

This mirrors the ``@register_cohort`` pattern in
:mod:`vena.data.cohort.registry` — one decorator, one lookup function, one
singleton.

Example
-------
::

    @register_segmentation_model("my_net")
    class MyNet(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None: ...


    model = get_segmentation_model("my_net", cfg)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import torch.nn as nn

    from vena.segmentation.config import ModelConfig

from vena.segmentation.exceptions import SegModelError

logger = logging.getLogger(__name__)

# Type variable for the decorated class (must be an nn.Module subclass).
_M = TypeVar("_M")

# Registry: name → factory callable (model_class or factory function).
# Populated at import time by @register_segmentation_model.
_REGISTRY: dict[str, Callable[[ModelConfig], nn.Module]] = {}


def register_segmentation_model(name: str) -> Callable[[type[_M]], type[_M]]:
    """Class-decorator that registers a segmentation model builder.

    The decorated class must accept a single :class:`~vena.segmentation.config.ModelConfig`
    positional argument and return a ``torch.nn.Module`` instance.

    Parameters
    ----------
    name:
        Registry key (lowercase, snake_case).  Must be unique; re-registering
        the same name raises :class:`~vena.segmentation.exceptions.SegModelError`.

    Returns
    -------
    Callable[[type[_M]], type[_M]]
        Pass-through decorator that leaves the class unchanged.

    Raises
    ------
    SegModelError
        If ``name`` is already registered.
    """

    def _wrap(cls: type[_M]) -> type[_M]:
        if name in _REGISTRY:
            raise SegModelError(
                f"Model '{name}' is already registered. Registered keys: {sorted(_REGISTRY)}"
            )
        _REGISTRY[name] = cls  # type: ignore[assignment]
        logger.debug("Registered segmentation model '%s' -> %s", name, cls.__qualname__)
        return cls

    return _wrap


def get_segmentation_model(name: str, cfg: ModelConfig) -> nn.Module:
    """Instantiate a registered segmentation model.

    Parameters
    ----------
    name:
        Registry key matching a previously decorated class.
    cfg:
        Frozen :class:`~vena.segmentation.config.ModelConfig` passed verbatim
        to the model constructor.

    Returns
    -------
    torch.nn.Module
        A freshly constructed (not yet moved to device) model instance.

    Raises
    ------
    SegModelError
        If ``name`` is not in the registry, listing all registered names.
    """
    if name not in _REGISTRY:
        registered = sorted(_REGISTRY)
        raise SegModelError(f"Unknown segmentation model '{name}'. Registered: {registered}")
    return _REGISTRY[name](cfg)


def registered_model_names() -> list[str]:
    """Return all currently registered model names (sorted).

    Returns
    -------
    list[str]
        Sorted list of registered keys.
    """
    return sorted(_REGISTRY)


__all__ = [
    "get_segmentation_model",
    "register_segmentation_model",
    "registered_model_names",
]
