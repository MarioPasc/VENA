"""NIfTI cohort loaders. Each cohort has a dedicated module under this package."""

from __future__ import annotations

from .shared import (
    ModalityNotFoundError,
    NiftiVolume,
    NiigzLoadError,
    PatientNotFoundError,
    brain_z_extent,
    evenly_spaced_indices,
    load_nii,
    non_empty_indices,
    pick_evenly_from,
    save_nii,
)
from .ucsf_pdgm import UCSFPDGMDataset, UCSFPDGMPatient

__all__ = [
    "ModalityNotFoundError",
    "NiftiVolume",
    "NiigzLoadError",
    "PatientNotFoundError",
    "UCSFPDGMDataset",
    "UCSFPDGMPatient",
    "brain_z_extent",
    "evenly_spaced_indices",
    "load_nii",
    "non_empty_indices",
    "pick_evenly_from",
    "save_nii",
]
