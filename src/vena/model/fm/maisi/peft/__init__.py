"""PEFT adapter layer for the MAISI rectified-flow trunk.

Public surface:

* :class:`BasePEFT` — the abstract contract every variant implements.
* :func:`build_peft` — factory that consumes the YAML ``peft`` block
  (``{"variant": str, "params": dict}``) and returns a configured adapter.
* :func:`register_peft`, :func:`list_variants` — registry helpers.
* :class:`LoRA` — the default low-rank attention adapter.

Variants register themselves via the decorator at import time; importing
this package guarantees the registry is populated before any YAML is
parsed.
"""

from __future__ import annotations

from .base import BasePEFT
from .exceptions import PeftConfigError, PeftError, UnknownVariantError
from .lora import LoRA
from .registry import build_peft, get_variant, list_variants, register_peft

__all__ = [
    "BasePEFT",
    "LoRA",
    "PeftConfigError",
    "PeftError",
    "UnknownVariantError",
    "build_peft",
    "get_variant",
    "list_variants",
    "register_peft",
]
