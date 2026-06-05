"""Decorator-based registry for PEFT variants.

A variant module advertises itself with::

    from .registry import register_peft


    @register_peft("lora")
    class LoRA(BasePEFT): ...

Callers build a configured adapter from the YAML block via
:func:`build_peft`. Variants are imported eagerly from
:mod:`vena.model.fm.maisi.peft.__init__` so the registry is fully populated
before any YAML is parsed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BasePEFT
from .exceptions import PeftConfigError, UnknownVariantError

_REGISTRY: dict[str, type[BasePEFT]] = {}


def register_peft(name: str) -> Callable[[type[BasePEFT]], type[BasePEFT]]:
    """Register a :class:`BasePEFT` subclass under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is already registered with a different class.
    """

    def _decorator(cls: type[BasePEFT]) -> type[BasePEFT]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"PEFT variant {name!r} already registered to {existing.__name__}; "
                f"cannot register {cls.__name__}"
            )
        cls.variant = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def list_variants() -> list[str]:
    """Return the sorted list of registered PEFT variant names."""
    return sorted(_REGISTRY)


def get_variant(name: str) -> type[BasePEFT]:
    """Return the :class:`BasePEFT` subclass registered under ``name``."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise UnknownVariantError(f"unknown PEFT variant {name!r}; available: {list_variants()}")
    return cls


def build_peft(variant: str, params: dict[str, Any] | None) -> BasePEFT:
    """Build a configured PEFT adapter from a YAML block.

    Parameters
    ----------
    variant : str
        Registry key, e.g. ``"lora"``.
    params : dict | None
        Variant-specific parameter block (``peft.params`` in the YAML).
        ``None`` is treated as an empty dict, letting the variant fall back
        to its own defaults.

    Raises
    ------
    UnknownVariantError
        If ``variant`` is not registered.
    PeftConfigError
        If the params block is malformed for this variant.
    """
    cls = get_variant(variant)
    try:
        return cls.from_dict(params or {})
    except PeftConfigError:
        raise
    except (KeyError, ValueError, TypeError) as exc:
        raise PeftConfigError(
            f"failed to build PEFT variant {variant!r} from params {params!r}: {exc}"
        ) from exc
