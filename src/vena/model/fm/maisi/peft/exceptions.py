"""Exceptions raised by the PEFT adapter layer."""

from __future__ import annotations


class PeftError(Exception):
    """Base class for PEFT adapter failures."""


class UnknownVariantError(PeftError):
    """Raised when a PEFT variant string is not registered."""


class PeftConfigError(PeftError):
    """Raised when the YAML ``peft.params`` block is malformed."""
