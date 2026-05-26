"""Builders for :class:`SubjectInputs` from on-disk UCSF-PDGM + derived priors.

Reuses :class:`vena.data.niigz.UCSFPDGMDataset` for raw modalities + masks
and loads each requested derived prior from
``<derived_root>/<patient_id>/<channel>.nii.gz``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from vena.data.niigz import NiftiVolume, UCSFPDGMDataset, UCSFPDGMPatient
from vena.data.niigz.shared.exceptions import ModalityNotFoundError
from vena.data.niigz.shared.io import load_nii

from ..core.dataclasses import SubjectInputs, SubjectMetadata

logger = logging.getLogger(__name__)


def _safe_load_modality(
    dataset: UCSFPDGMDataset, p: UCSFPDGMPatient, name: str
) -> NiftiVolume | None:
    try:
        return dataset.load_modality(p, name)  # type: ignore[arg-type]
    except ModalityNotFoundError:
        return None


def _safe_load_mask(dataset: UCSFPDGMDataset, p: UCSFPDGMPatient, kind: str) -> NiftiVolume | None:
    loader = {
        "brain": dataset.load_brain_mask,
        "parenchyma": dataset.load_brain_parenchyma_mask,
        "tumour": dataset.load_tumor_seg,
    }[kind]
    try:
        return loader(p)
    except ModalityNotFoundError:
        return None


def _load_derived_priors(
    patient_id: str,
    derived_roots: dict[str, Path],
) -> dict[str, NiftiVolume]:
    """Load each declared derived prior channel for one patient.

    ``derived_roots`` is keyed by channel name (``"cbf"``, ``"cell"``, …) and
    points to the directory that contains one subdirectory per patient. A
    channel is silently skipped when its file is absent — the test layer
    handles the missing prior via ``applicable(inputs)``.
    """
    out: dict[str, NiftiVolume] = {}
    for channel, root in derived_roots.items():
        path = Path(root) / patient_id / f"{channel}.nii.gz"
        if not path.exists():
            logger.warning("derived prior missing for %s: %s", patient_id, path)
            continue
        out[channel] = load_nii(path)
    return out


def _row_to_metadata(patient_id: str, row: dict[str, Any] | None) -> SubjectMetadata:
    if row is None:
        return SubjectMetadata(subject_id=patient_id)

    # UCSF-PDGM metadata CSV columns vary; tolerate the common ones.
    def _get(*keys: str) -> Any:
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None

    age = _get("Age at MRI", "age", "Age")
    sex = _get("Sex", "sex")
    pathology = _get("Final pathologic diagnosis (WHO 2021)", "pathology")
    grade_raw = _get("WHO CNS Grade", "who_grade")
    try:
        grade: int | None = int(grade_raw) if grade_raw is not None else None
    except (ValueError, TypeError):
        grade = None
    return SubjectMetadata(
        subject_id=patient_id,
        age=float(age) if age is not None else None,
        sex=str(sex) if sex is not None else None,
        scanner="GE 3T",  # UCSF-PDGM single-scanner cohort
        field_strength_t=3.0,
        pathology=str(pathology) if pathology is not None else None,
        who_grade=grade,
        extras=dict(row) if row else {},
    )


def load_metadata_csv(csv_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the UCSF-PDGM metadata CSV keyed by 4-digit patient id.

    Returns an empty dict on missing path.
    """
    if csv_path is None or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[str, dict[str, Any]] = {}
    id_col = df.columns[0]
    for _, row in df.iterrows():
        raw = str(row[id_col]).strip()
        m = re.match(r"^UCSF-PDGM-(\d+)$", raw)
        if m is None:
            continue
        padded = f"UCSF-PDGM-{int(m.group(1)):04d}"
        out[padded] = row.to_dict()
    return out


def build_subject_inputs(
    dataset: UCSFPDGMDataset,
    patient: UCSFPDGMPatient,
    derived_roots: dict[str, Path],
    metadata_rows: dict[str, dict[str, Any]] | None = None,
) -> SubjectInputs:
    """Assemble a :class:`SubjectInputs` from on-disk volumes.

    ``t1pre`` is the bias-corrected T1 (``T1_bias``); ``t1gd`` is ``T1c_bias``.
    Falls back to the non-bias-corrected variants if the bias versions are
    missing.
    """
    pid = patient.patient_id
    t1pre = _safe_load_modality(dataset, patient, "T1_bias")
    if t1pre is None:
        t1pre = _safe_load_modality(dataset, patient, "T1")
    t1gd = _safe_load_modality(dataset, patient, "T1c_bias")
    if t1gd is None:
        t1gd = _safe_load_modality(dataset, patient, "T1c")
    if t1pre is None or t1gd is None:
        raise FileNotFoundError(f"Mandatory T1pre/T1c missing for {pid}")
    brain_mask = _safe_load_mask(dataset, patient, "brain")
    if brain_mask is None:
        raise FileNotFoundError(f"Mandatory brain_segmentation missing for {pid}")
    parenchyma = _safe_load_mask(dataset, patient, "parenchyma")
    tumour = _safe_load_mask(dataset, patient, "tumour")
    cbf = _safe_load_modality(dataset, patient, "ASL")
    adc = _safe_load_modality(dataset, patient, "ADC")
    swan_mag = _safe_load_modality(dataset, patient, "SWI_bias")
    if swan_mag is None:
        swan_mag = _safe_load_modality(dataset, patient, "SWI")
    derived = _load_derived_priors(pid, derived_roots)
    meta = _row_to_metadata(pid, (metadata_rows or {}).get(pid))
    return SubjectInputs(
        subject_id=pid,
        t1pre=t1pre,
        t1gd=t1gd,
        brain_mask=brain_mask,
        parenchyma_mask=parenchyma,
        tumour_mask=tumour,
        cbf=cbf,
        adc=adc,
        chi=None,  # UCSF-PDGM has no phase data; QSM is out of v0 scope
        swan_mag=swan_mag,
        derived_priors=derived,
        metadata=meta,
    )
