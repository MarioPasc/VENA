"""Exceptions raised by the H5 converter / validator stack.

A small hierarchy keeps `except` clauses precise: callers can catch a single
``H5Error`` to swallow anything from this subsystem, or narrow down to
``H5SchemaError`` / ``H5ConvertError`` / ``H5ValidationError`` as needed.
"""

from __future__ import annotations


class H5Error(Exception):
    """Base for every exception raised by ``vena.data.h5``."""


class H5SchemaError(H5Error):
    """A manifest or schema construction is internally inconsistent."""


class H5ConvertError(H5Error):
    """A converter could not turn a source NIfTI into a writable H5 chunk.

    Typically raised when a per-patient volume violates the shape contract
    declared in the manifest (e.g. ``(240, 240, 155)`` for UCSF-PDGM).
    """


class H5ValidationError(H5Error):
    """An H5 file failed post-write validation against its declared manifest."""
