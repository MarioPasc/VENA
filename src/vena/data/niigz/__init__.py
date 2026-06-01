"""NIfTI cohort loaders. Each cohort has a dedicated module under this package."""

from __future__ import annotations

from .brats_africa import (
    BraTSAfricaGliomaDataset,
    BraTSAfricaOtherDataset,
    BraTSAfricaPatient,
)
from .brats_ped import BraTSPedDataset, BraTSPedPatient
from .ivy_gap import IvyGAPDataset, IvyGAPPatient
from .lumiere import LUMIEREDataset, LUMIERESession
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
    "BraTSAfricaGliomaDataset",
    "BraTSAfricaOtherDataset",
    "BraTSAfricaPatient",
    "BraTSPedDataset",
    "BraTSPedPatient",
    "IvyGAPDataset",
    "IvyGAPPatient",
    "LUMIEREDataset",
    "LUMIERESession",
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
