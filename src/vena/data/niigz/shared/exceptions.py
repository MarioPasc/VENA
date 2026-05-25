"""Exceptions raised by the NIfTI cohort loaders."""

from __future__ import annotations


class NiigzLoadError(Exception):
    """Failure to load a NIfTI file (missing path, unreadable, corrupt header)."""


class ModalityNotFoundError(NiigzLoadError):
    """A requested modality does not exist for a given patient."""


class PatientNotFoundError(NiigzLoadError):
    """A requested patient ID is not present in the cohort root."""
