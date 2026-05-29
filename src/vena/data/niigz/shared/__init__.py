"""Shared NIfTI I/O and geometry primitives reused across cohort loaders."""

from __future__ import annotations

from .exceptions import ModalityNotFoundError, NiigzLoadError, PatientNotFoundError
from .geometry import (
    array_axcodes,
    brain_z_extent,
    evenly_spaced_indices,
    non_empty_indices,
    pick_evenly_from,
    reorient_to_axcodes,
)
from .io import NiftiVolume, load_nii, save_nii

__all__ = [
    "ModalityNotFoundError",
    "NiftiVolume",
    "NiigzLoadError",
    "PatientNotFoundError",
    "array_axcodes",
    "brain_z_extent",
    "evenly_spaced_indices",
    "load_nii",
    "non_empty_indices",
    "pick_evenly_from",
    "reorient_to_axcodes",
    "save_nii",
]
