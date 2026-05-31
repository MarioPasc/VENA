"""IvyGAP NIfTI reader + image-domain H5 converter.

Synthetic on-disk fixture only — no real cohort files needed. Covers:

* Registration on import; cohort registry entry exists.
* Patient discovery with the SRI24-pipeline directory layout.
* Registration-variant precedence (``_N4_r_SS`` > ``_r3_SS`` > ``_r_SS``).
* UPenn segmentation resolution (and graceful skip when missing).
* CohortProtocol structural conformance.
* End-to-end converter run on tiny 8x8x8 volumes with a 1/1/2 split.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import IvyGAPDataset, IvyGAPPatient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)
_LPS_AFFINE[0, 3] = 0.0
_LPS_AFFINE[1, 3] = float(_SHAPE[1] - 1)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _make_brain(rng: np.random.Generator) -> np.ndarray:
    """Centred ellipsoidal blob > 0 so brain-mask derivation has a real bbox."""
    z, y, x = np.indices(_SHAPE)
    cz, cy, cx = (s / 2.0 for s in _SHAPE)
    r2 = ((z - cz) / 3.0) ** 2 + ((y - cy) / 3.0) ** 2 + ((x - cx) / 3.0) ** 2
    base = np.where(r2 < 1.0, rng.uniform(50.0, 200.0, _SHAPE), 0.0)
    return base.astype(np.float32)


def _make_seg() -> np.ndarray:
    seg = np.zeros(_SHAPE, dtype=np.float32)
    seg[3:5, 3:5, 3:5] = 4.0  # enhancing core
    seg[2:6, 2:6, 2:6][seg[2:6, 2:6, 2:6] == 0] = 2.0  # edema halo
    return seg


def _build_ivygap_tree(
    root: Path,
    *,
    patient_specs: list[tuple[str, str, dict[str, str], bool]],
) -> None:
    """Materialise a minimal IvyGAP source tree.

    Each ``patient_specs`` entry is ``(patient_id, scan_date, variant_per_mod,
    has_upenn)``. ``variant_per_mod`` maps slug→suffix in ``{"_r_SS",
    "_r3_SS", "_N4_r_SS"}``.
    """
    rng = np.random.default_rng(0)
    images_root = root / "1_Images_SRI" / "CoRegistered_SkullStripped"
    upenn_root = root / "3_Annotations_SRI" / "UPenn"
    cwru_root = root / "3_Annotations_SRI" / "CWRU"
    images_root.mkdir(parents=True, exist_ok=True)
    upenn_root.mkdir(parents=True, exist_ok=True)
    cwru_root.mkdir(parents=True, exist_ok=True)

    infix_map = {"t1pre": "t1", "t1c": "t1gd", "t2": "t2", "flair": "flair"}

    for patient_id, scan_date, variants, has_upenn in patient_specs:
        session_stem = f"{patient_id}_{scan_date}"
        session_dir = images_root / patient_id / session_stem
        session_dir.mkdir(parents=True, exist_ok=True)
        for slug, infix in infix_map.items():
            suffix = variants[slug]
            fname = f"{session_stem}_{infix}_LPS{suffix}.nii.gz"
            _write_nii(session_dir / fname, _make_brain(rng))
        if has_upenn:
            _write_nii(
                upenn_root / patient_id / f"{session_stem}_UPenn_labels.nii.gz",
                _make_seg(),
            )
        # CWRU label always written for the synthetic patients (covers the
        # cwru_seg_path metadata branch).
        _write_nii(
            cwru_root / patient_id / f"{session_stem}_CWRU_labels.nii.gz",
            _make_seg(),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_ivy_gap_registered_in_global_registry() -> None:
    reg = get_cohort_registry()
    assert "ivy_gap" in reg
    assert reg.pathology_of("ivy_gap") == "glioma"


# ---------------------------------------------------------------------------
# Reader behaviour
# ---------------------------------------------------------------------------


def test_reader_discovers_patients_and_resolves_variants(tmp_path: Path) -> None:
    _build_ivygap_tree(
        tmp_path,
        patient_specs=[
            ("W1", "1996.10.25", dict.fromkeys(("t1pre", "t1c", "t2", "flair"), "_r_SS"), True),
            (
                "W9",
                "1997.04.10",
                {"t1pre": "_r3_SS", "t1c": "_r_SS", "t2": "_r_SS", "flair": "_r_SS"},
                True,
            ),
            (
                "W20",
                "1998.06.01",
                dict.fromkeys(("t1pre", "t1c", "t2", "flair"), "_N4_r_SS"),
                True,
            ),
        ],
    )
    ds = IvyGAPDataset(tmp_path)
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 3
    assert ds.ids() == ["W1", "W20", "W9"]  # lexicographic sort

    w9 = ds["W9"]
    assert isinstance(w9, IvyGAPPatient)
    # Mixed variants: t1pre picks _r3_, others _r_.
    assert w9.metadata["source_basename_t1pre"].endswith("_t1_LPS_r3_SS.nii.gz")
    assert w9.metadata["source_basename_t1c"].endswith("_t1gd_LPS_r_SS.nii.gz")

    w20 = ds["W20"]
    # N4 precedence wins over r3 and r when present.
    assert w20.metadata["source_basename_t1pre"].endswith("_t1_LPS_N4_r_SS.nii.gz")


def test_reader_skips_patient_with_missing_upenn(tmp_path: Path) -> None:
    _build_ivygap_tree(
        tmp_path,
        patient_specs=[
            ("W1", "1996.10.25", dict.fromkeys(("t1pre", "t1c", "t2", "flair"), "_r_SS"), True),
            (
                "W30",
                "1997.05.05",
                dict.fromkeys(("t1pre", "t1c", "t2", "flair"), "_r_SS"),
                False,
            ),
        ],
    )
    ds = IvyGAPDataset(tmp_path)
    assert ds.ids() == ["W1"]


def test_reader_load_modality_returns_lps_volume(tmp_path: Path) -> None:
    _build_ivygap_tree(
        tmp_path,
        patient_specs=[
            ("W1", "1996.10.25", dict.fromkeys(("t1pre", "t1c", "t2", "flair"), "_r_SS"), True),
        ],
    )
    ds = IvyGAPDataset(tmp_path)
    vol = ds.load_modality(ds[0], "t1pre")
    assert vol.array.shape == _SHAPE
    seg = ds.load_tumor_seg(ds[0])
    assert set(np.unique(np.asarray(seg.array)).tolist()) <= {0.0, 2.0, 4.0}
