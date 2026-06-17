"""Image-domain helpers for inference (per-patient reads from the image H5).

The training-time data path is the multi-cohort *latent* data module; the
2D image-tier competitors (C1 pGAN, C2 ResViT, C3 SynDiff) and the C0
identity baseline need *image* domain reads, and the latent-tier
adapters also need the image H5 for the crop_spec + reference T1c. The
helpers in this module fill that gap with a small, dependency-light
API — they are not a ``torch.utils.data.Dataset`` because the inference
engine drives one patient at a time.

Functions
---------
* :func:`resolve_test_patient_ids` — list patient IDs from the test
  split (``splits/test`` for ``role=cv`` cohorts, every patient for
  ``role=test_only``).
* :func:`load_image_modalities` — read native intensities + masks for
  one patient from the image H5 (no normalisation applied).
* :func:`harmonised_modalities_for_record` — read + apply §4.1
  harmonisation to T1pre, T2, FLAIR, T1c; derive a WT mask from
  ``masks/tumor > 0``. Used to fill the predictions-H5 reference
  block.
* :func:`row_index_for_patient` — map a patient_id to its row in the
  H5 by scanning ``/ids``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from vena.inference.harmonisation import apply_harmonisation

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


class ImageH5LookupError(Exception):
    """Raised on missing IDs or schema deviations in an image H5."""


def _decode_str(values: object) -> list[str]:
    out: list[str] = []
    for b in values:  # type: ignore[union-attr]
        out.append(b.decode() if isinstance(b, bytes) else str(b))
    return out


def row_index_for_patient(image_h5: Path | str, patient_id: str) -> int:
    """Return the integer row in ``/ids`` for ``patient_id``."""
    image_h5 = Path(image_h5)
    with h5py.File(image_h5, "r") as f:
        ids = _decode_str(f["ids"][:])
    try:
        return ids.index(patient_id)
    except ValueError as exc:
        raise ImageH5LookupError(f"patient '{patient_id}' not in {image_h5}/ids") from exc


def resolve_test_patient_ids(cohort: CohortEntry, fold: int = 0) -> list[str]:
    """Resolve the test SCAN IDs for one cohort (CSR-expanded from patient IDs).

    For ``role=cv`` cohorts ``splits/test`` is a list of *patient* IDs;
    for ``role=test_only`` cohorts every patient in ``patients/keys`` is
    a test subject. In both cases the IDs in those datasets are
    **patient-level**: a longitudinal patient (LUMIERE, BraTS-GLI) maps
    to multiple scans, each with its own ID in ``/ids``. Image and
    latent reads (which index by row in ``/ids``) therefore need the
    *scan* IDs, not the patient IDs.

    This function performs the CSR expansion ``patient_id`` →
    ``[scan_id, ...]`` via the ``patients/keys`` + ``patients/offsets``
    layout, mirroring ``ExhaustiveValEngine._cohort_val_patients``.

    Parameters
    ----------
    cohort
        Cohort entry from the corpus registry.
    fold
        Unused for ``role=test_only`` cohorts. For ``role=cv`` cohorts
        the test split is *fold-independent* in our H5 schema
        (``splits/test`` lives at the H5 root), so this is currently
        ignored. Kept on the signature so the engine can stay
        fold-aware without later breakage.

    Returns
    -------
    list[str]
        Scan IDs in deterministic patient-then-scan order.
    """
    del fold  # fold-independent test partition; arg kept for future-proofing
    image_h5 = Path(cohort.image_h5)
    if not image_h5.is_file():
        raise ImageH5LookupError(f"image H5 not found for cohort {cohort.name}: {image_h5}")
    with h5py.File(image_h5, "r") as f:
        if cohort.role == "test_only":
            patient_keys = _decode_str(f["patients/keys"][:])
        else:
            if "splits/test" not in f:
                raise ImageH5LookupError(
                    f"cohort {cohort.name}: image H5 lacks 'splits/test' (CV cohort)"
                )
            patient_keys = _decode_str(f["splits/test"][:])
        # CSR expand patient_id → scan_ids. If `patients/*` is absent
        # (rare; some old converters skipped the CSR), fall back to the
        # raw patient_keys assuming patient == scan.
        if "patients/keys" not in f or "patients/offsets" not in f or "ids" not in f:
            return sorted(set(patient_keys))
        csr_keys = _decode_str(f["patients/keys"][:])
        offsets = f["patients/offsets"][:]
        all_ids = _decode_str(f["ids"][:])
        key_to_pos = {k: i for i, k in enumerate(csr_keys)}
        scan_ids: list[str] = []
        for pk in patient_keys:
            if pk not in key_to_pos:
                # Patient ID is in splits/test but absent from patients/keys.
                # This signals a converter bug; surface it on read.
                continue
            pos = key_to_pos[pk]
            start, end = int(offsets[pos]), int(offsets[pos + 1])
            for row in range(start, end):
                scan_ids.append(all_ids[row])
        return scan_ids


def load_image_modalities(
    image_h5: Path | str,
    patient_id: str,
    modalities: tuple[str, ...] = ("t1pre", "t1c", "t2", "flair"),
) -> dict[str, np.ndarray]:
    """Read native intensities + masks for one patient.

    Returns a dict with one key per modality plus ``"brain"`` and
    ``"tumor"`` (binary). Volumes are float32, masks are int8. No
    normalisation is applied.
    """
    row = row_index_for_patient(image_h5, patient_id)
    out: dict[str, np.ndarray] = {}
    with h5py.File(image_h5, "r") as f:
        for mod in modalities:
            ds = f[f"images/{mod}"]
            out[mod] = np.ascontiguousarray(ds[row], dtype=np.float32)
        out["brain"] = np.ascontiguousarray(f["masks/brain"][row], dtype=np.int8)
        out["tumor"] = np.ascontiguousarray(f["masks/tumor"][row], dtype=np.int8)
    return out


def harmonised_modalities_for_record(
    image_h5: Path | str,
    patient_id: str,
) -> tuple[
    np.ndarray,  # t1pre_harmonised
    np.ndarray,  # t2_harmonised
    np.ndarray,  # flair_harmonised
    np.ndarray,  # t1c_real_harmonised
    np.ndarray,  # brain_mask
    np.ndarray,  # wt_mask
]:
    """Read modalities + masks and apply §4.1 harmonisation.

    Returns the six fixed-modality arrays the predictions H5 schema
    needs for one scan, in float32/int8. The harmonisation matches what
    each adapter applies to its own predicted T1c, so the predicted and
    reference volumes live in the same intensity space.
    """
    mods = load_image_modalities(image_h5, patient_id, ("t1pre", "t1c", "t2", "flair"))
    brain = mods["brain"]
    brain_t = torch.from_numpy(brain).to(torch.float32)

    def _harm(mod: str) -> np.ndarray:
        vol = torch.from_numpy(mods[mod])
        harm = apply_harmonisation(vol, brain_mask=brain_t)
        return harm.detach().cpu().numpy().astype(np.float32)

    t1pre_h = _harm("t1pre")
    t2_h = _harm("t2")
    flair_h = _harm("flair")
    t1c_h = _harm("t1c")
    wt = (mods["tumor"] > 0).astype(np.int8)
    return t1pre_h, t2_h, flair_h, t1c_h, brain, wt


__all__ = [
    "ImageH5LookupError",
    "harmonised_modalities_for_record",
    "load_image_modalities",
    "resolve_test_patient_ids",
    "row_index_for_patient",
]
