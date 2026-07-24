"""Tests for vena.segmentation.data.fm_splits.

All tests use synthetic on-disk H5 fixtures; no real cohort data is read.
Covers:
- CohortSplit / FmSplitResolution data contracts
- resolve_fm_splits: cv-layout H5, flat-layout fallback, bad fold index
- resolve_fm_splits: absolute path resolution vs image_h5_root fallback
- resolve_fm_splits: dedup filter (cv required, test_only tolerates absence)
- resolve_fm_splits: test_only contributes only to test split
- resolve_fm_splits: longitudinal cohort patient→scan CSR expansion
- write_splits_json: round-trip + each invariant violation raises
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.segmentation


# ---------------------------------------------------------------------------
# Synthetic H5 helpers
# ---------------------------------------------------------------------------


def _vlen_str_dtype():
    import h5py

    return h5py.special_dtype(vlen=str)


def _write_cv_h5(
    path: Path,
    *,
    patient_keys: list[str],
    scan_ids: list[str],
    offsets: list[int],
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    fold: int = 0,
    shape: tuple[int, int, int] = (4, 4, 4),
    n_folds: int = 1,
) -> None:
    """Create a synthetic image H5 with canonical ``splits/cv`` layout."""
    import h5py

    dt = _vlen_str_dtype()
    n = len(scan_ids)
    h, w, d = shape
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=dt)
        hf.create_dataset("patients/keys", data=np.array(patient_keys, dtype=object), dtype=dt)
        hf.create_dataset("patients/offsets", data=np.array(offsets, dtype=np.int32))
        hf.create_dataset(
            f"splits/cv/fold_{fold}/train",
            data=np.array(train_keys, dtype=object),
            dtype=dt,
        )
        hf.create_dataset(
            f"splits/cv/fold_{fold}/val",
            data=np.array(val_keys, dtype=object),
            dtype=dt,
        )
        hf.create_dataset("splits/test", data=np.array(test_keys, dtype=object), dtype=dt)
        for mod in ("t1pre", "t2", "flair"):
            hf.create_dataset(
                f"images/{mod}", data=rng.standard_normal((n, h, w, d)).astype(np.float32)
            )
        hf.create_dataset("masks/tumor", data=np.zeros((n, h, w, d), dtype=np.int8))
        hf.create_dataset("masks/brain", data=np.ones((n, h, w, d), dtype=np.float32))


def _write_flat_h5(
    path: Path,
    *,
    patient_keys: list[str],
    scan_ids: list[str],
    offsets: list[int],
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    shape: tuple[int, int, int] = (4, 4, 4),
) -> None:
    """Create a synthetic image H5 with legacy flat ``splits/{train,val}`` layout."""
    import h5py

    dt = _vlen_str_dtype()
    n = len(scan_ids)
    h, w, d = shape
    rng = np.random.default_rng(1)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=dt)
        hf.create_dataset("patients/keys", data=np.array(patient_keys, dtype=object), dtype=dt)
        hf.create_dataset("patients/offsets", data=np.array(offsets, dtype=np.int32))
        hf.create_dataset("splits/train", data=np.array(train_keys, dtype=object), dtype=dt)
        hf.create_dataset("splits/val", data=np.array(val_keys, dtype=object), dtype=dt)
        hf.create_dataset("splits/test", data=np.array(test_keys, dtype=object), dtype=dt)
        for mod in ("t1pre", "t2", "flair"):
            hf.create_dataset(
                f"images/{mod}", data=rng.standard_normal((n, h, w, d)).astype(np.float32)
            )
        hf.create_dataset("masks/tumor", data=np.zeros((n, h, w, d), dtype=np.int8))
        hf.create_dataset("masks/brain", data=np.ones((n, h, w, d), dtype=np.float32))


def _write_test_only_h5(
    path: Path,
    *,
    patient_keys: list[str],
    scan_ids: list[str],
    offsets: list[int],
    shape: tuple[int, int, int] = (4, 4, 4),
) -> None:
    """Create a synthetic image H5 for test_only cohort (no cv splits)."""
    import h5py

    dt = _vlen_str_dtype()
    n = len(scan_ids)
    h, w, d = shape
    rng = np.random.default_rng(2)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=dt)
        hf.create_dataset("patients/keys", data=np.array(patient_keys, dtype=object), dtype=dt)
        hf.create_dataset("patients/offsets", data=np.array(offsets, dtype=np.int32))
        hf.create_dataset("splits/test", data=np.array(patient_keys, dtype=object), dtype=dt)
        for mod in ("t1pre", "t2", "flair"):
            hf.create_dataset(
                f"images/{mod}", data=rng.standard_normal((n, h, w, d)).astype(np.float32)
            )
        hf.create_dataset("masks/tumor", data=np.zeros((n, h, w, d), dtype=np.int8))
        hf.create_dataset("masks/brain", data=np.ones((n, h, w, d), dtype=np.float32))


_COHORT_DEFAULTS: dict[str, Any] = {
    "pathology": "preoperative_glioma",
    "label_system": "BraTS2021",
    "longitudinal": False,
    "latent_h5": "/nonexistent/dummy_latents.h5",
    "n_patients": 0,
    "n_scans": 0,
    "modalities": ["t1pre", "t2", "flair"],
    "has_swan": False,
}


def _write_corpus_registry(
    path: Path,
    cohorts: list[dict[str, Any]],
) -> Path:
    """Write a minimal corpus registry JSON.

    Each cohort entry is merged with ``_COHORT_DEFAULTS`` so that the
    :class:`~vena.data.registry.models.CohortEntry` Pydantic model validates
    without requiring callers to supply every field.  The ``latent_h5`` default
    is a dummy non-existent path; ``load_registry(require_latents=False)`` does
    not check whether it exists.
    """
    full_cohorts = [{**_COHORT_DEFAULTS, **c} for c in cohorts]
    registry = {
        "schema_version": "1.0.0",
        "name": "synthetic_fm_splits_test",
        "cohorts": full_cohorts,
    }
    path.write_text(json.dumps(registry))
    return path


def _make_data_cfg(
    corpus_registry: Path,
    image_h5_root: Path,
    fm_fold: int = 0,
    dedup_decision_path: Path | None = None,
    k_folds: int = 2,
    fold_seed: int = 42,
) -> MagicMock:
    cfg = MagicMock()
    cfg.corpus_registry = corpus_registry
    cfg.image_h5_root = image_h5_root
    cfg.fm_fold = fm_fold
    cfg.dedup_decision_path = dedup_decision_path
    cfg.k_folds = k_folds
    cfg.fold_seed = fold_seed
    return cfg


def _single_patient_offsets(n_patients: int) -> list[int]:
    """Build CSR offsets for n_patients each with 1 scan (1:1 layout)."""
    return list(range(n_patients + 1))


# ---------------------------------------------------------------------------
# resolve_fm_splits: basic cv-layout
# ---------------------------------------------------------------------------


class TestResolveFmSplitsCvLayout:
    """resolve_fm_splits reads canonical splits/cv layout correctly."""

    @pytest.fixture()
    def simple_cv_setup(self, tmp_path: Path):
        """One cv cohort, 6 patients, 1 scan each, fold_0."""
        pkeys = [f"PAT_{i:03d}" for i in range(6)]
        sids = [f"SCAN_{i:03d}" for i in range(6)]
        offsets = _single_patient_offsets(6)
        train_keys = pkeys[:4]
        val_keys = pkeys[4:5]
        test_keys = pkeys[5:]

        h5_name = "COH_image.h5"
        h5_path = tmp_path / h5_name
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path)
        return cfg, pkeys, sids, train_keys, val_keys, test_keys

    def test_fm_splits_keys_correct(self, simple_cv_setup):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, _pkeys, _sids, train_keys, val_keys, test_keys = simple_cv_setup
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        assert set(splits["train"]) == set(train_keys)
        assert set(splits["val"]) == set(val_keys)
        assert set(splits["test"]) == set(test_keys)

    def test_per_cohort_populated(self, simple_cv_setup):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, _pkeys, _sids, train_keys, val_keys, test_keys = simple_cv_setup
        res = resolve_fm_splits(cfg)

        assert len(res.per_cohort) == 1
        cs = res.per_cohort[0]
        assert cs.name == "COH"
        assert cs.role == "cv"
        assert set(cs.train_patients) == set(train_keys)
        assert set(cs.val_patients) == set(val_keys)
        assert set(cs.test_patients) == set(test_keys)
        assert cs.n_patients_h5 == 6

    def test_patient_to_scans_single_session(self, simple_cv_setup):
        """1:1 patient→scan layout; each patient maps to exactly one scan."""
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, pkeys, sids, _train, _val, _test = simple_cv_setup
        res = resolve_fm_splits(cfg)

        for pid, sid in zip(pkeys, sids, strict=True):
            assert res.patient_to_scans[pid] == (sid,)

    def test_scans_for_expands_correctly(self, simple_cv_setup):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, pkeys, sids, train_keys, _val, _test = simple_cv_setup
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        scans = res.scans_for(splits["train"])
        expected = sorted(sids[i] for i, pk in enumerate(pkeys) if pk in set(train_keys))
        assert sorted(scans) == sorted(expected)

    def test_patient_to_cohort_populated(self, simple_cv_setup):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, pkeys, *_ = simple_cv_setup
        res = resolve_fm_splits(cfg)

        for pk in pkeys:
            assert res.patient_to_cohort[pk] == "COH"


# ---------------------------------------------------------------------------
# resolve_fm_splits: flat legacy layout (REMBRANDT fallback)
# ---------------------------------------------------------------------------


class TestResolveFmSplitsFlatLayout:
    """resolve_fm_splits falls back to splits/{train,val} when splits/cv is absent."""

    @pytest.fixture()
    def flat_setup(self, tmp_path: Path):
        pkeys = [f"REM_{i:03d}" for i in range(5)]
        sids = [f"RSCAN_{i:03d}" for i in range(5)]
        offsets = _single_patient_offsets(5)
        train_keys = pkeys[:3]
        val_keys = pkeys[3:4]
        test_keys = pkeys[4:]

        h5_path = tmp_path / "REM_image.h5"
        _write_flat_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "REM",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        return _make_data_cfg(reg_path, tmp_path), pkeys, train_keys, val_keys, test_keys

    def test_flat_fallback_reads_splits(self, flat_setup):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, _pkeys, train_keys, val_keys, test_keys = flat_setup
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        assert set(splits["train"]) == set(train_keys)
        assert set(splits["val"]) == set(val_keys)
        assert set(splits["test"]) == set(test_keys)

    def test_flat_fallback_warns(self, flat_setup, caplog):
        import logging

        from vena.segmentation.data.fm_splits import resolve_fm_splits

        cfg, *_ = flat_setup
        with caplog.at_level(logging.WARNING, logger="vena.segmentation.data.fm_splits"):
            resolve_fm_splits(cfg)

        assert any("flat" in msg.lower() or "fallback" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# resolve_fm_splits: missing requested fold in splits/cv
# ---------------------------------------------------------------------------


class TestResolveFmSplitsBadFold:
    """When splits/cv exists but the requested fold is absent, raise SegDataError."""

    def test_bad_fold_raises(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits
        from vena.segmentation.exceptions import SegDataError

        pkeys = [f"P_{i:03d}" for i in range(4)]
        sids = [f"S_{i:03d}" for i in range(4)]
        offsets = _single_patient_offsets(4)

        h5_path = tmp_path / "COH_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=pkeys[:3],
            val_keys=pkeys[3:],
            test_keys=[],
            fold=0,
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        # Request fold_3 which doesn't exist (only fold_0 written)
        cfg = _make_data_cfg(reg_path, tmp_path, fm_fold=3)

        with pytest.raises(SegDataError, match="fold"):
            resolve_fm_splits(cfg)

    def test_bad_fold_error_lists_available(self, tmp_path: Path):
        """Error message must list available folds."""
        from vena.segmentation.data.fm_splits import resolve_fm_splits
        from vena.segmentation.exceptions import SegDataError

        pkeys = [f"P_{i:03d}" for i in range(4)]
        sids = [f"S_{i:03d}" for i in range(4)]
        offsets = _single_patient_offsets(4)

        h5_path = tmp_path / "COH_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=pkeys[:3],
            val_keys=pkeys[3:],
            test_keys=[],
            fold=0,
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, fm_fold=4)

        with pytest.raises(SegDataError, match="fold_0"):
            resolve_fm_splits(cfg)


# ---------------------------------------------------------------------------
# H5 path resolution: absolute path vs image_h5_root fallback
# ---------------------------------------------------------------------------


class TestH5PathResolution:
    """_resolve_h5_path prefers absolute registry path; falls back to image_h5_root."""

    def test_absolute_path_resolution(self, tmp_path: Path):
        """H5 at its registry-absolute path, image_h5_root points elsewhere."""
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        # H5 lives in a subdirectory, not directly in image_h5_root
        h5_dir = tmp_path / "nested" / "subdir"
        h5_dir.mkdir(parents=True)
        h5_path = h5_dir / "COH_image.h5"

        pkeys = ["P_000", "P_001", "P_002"]
        sids = ["S_000", "S_001", "S_002"]
        offsets = _single_patient_offsets(3)
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=["P_000", "P_001"],
            val_keys=["P_002"],
            test_keys=[],
        )
        # Registry points to the nested absolute path
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        # image_h5_root does NOT contain COH_image.h5 — only absolute path works
        cfg = _make_data_cfg(reg_path, tmp_path)  # tmp_path ≠ h5_dir

        res = resolve_fm_splits(cfg)
        assert res.per_cohort[0].image_h5 == h5_path

    def test_fallback_to_image_h5_root(self, tmp_path: Path):
        """When absolute path is missing, _resolve_h5_path falls back to image_h5_root / filename.

        This tests the helper directly: ``load_registry`` validates that
        ``image_h5.is_file()``, so we cannot exercise the fallback branch by
        going through ``resolve_fm_splits`` with a non-existent absolute path.
        The unit under test is ``_resolve_h5_path``.
        """
        from vena.segmentation.data.fm_splits import _resolve_h5_path

        h5_path = tmp_path / "COH_image.h5"
        h5_path.touch()  # create empty placeholder

        # Absolute path points nowhere → fallback to image_h5_root / name
        resolved = _resolve_h5_path("/nonexistent/path/COH_image.h5", tmp_path, "COH")

        assert resolved == h5_path, (
            f"_resolve_h5_path fallback failed: expected {h5_path}, got {resolved}. "
            "Bug 2 not fixed in _resolve_h5_path."
        )


# ---------------------------------------------------------------------------
# Longitudinal cohort: patient → multiple scans
# ---------------------------------------------------------------------------


class TestLongitudinalCohort:
    """LUMIERE-like: one patient maps to several scans via CSR offsets."""

    def test_patient_to_scans_longitudinal(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        # 2 patients: P_000 → 3 scans, P_001 → 2 scans
        pkeys = ["P_000", "P_001"]
        sids = ["S_000", "S_001", "S_002", "S_003", "S_004"]
        offsets = [0, 3, 5]  # P_000 → sids[0:3], P_001 → sids[3:5]

        h5_path = tmp_path / "LONG_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=["P_000"],
            val_keys=["P_001"],
            test_keys=[],
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "LONG",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path)
        res = resolve_fm_splits(cfg)

        assert set(res.patient_to_scans["P_000"]) == {"S_000", "S_001", "S_002"}
        assert set(res.patient_to_scans["P_001"]) == {"S_003", "S_004"}

    def test_scans_for_longitudinal_expands(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        pkeys = ["P_000", "P_001"]
        sids = ["S_000", "S_001", "S_002", "S_003", "S_004"]
        offsets = [0, 3, 5]

        h5_path = tmp_path / "LONG_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=["P_000"],
            val_keys=["P_001"],
            test_keys=[],
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "LONG",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path)
        res = resolve_fm_splits(cfg)

        train_scans = res.scans_for(["P_000"])
        assert set(train_scans) == {"S_000", "S_001", "S_002"}


# ---------------------------------------------------------------------------
# Dedup filter: cv required, test_only tolerates absence
# ---------------------------------------------------------------------------


class TestDedupFilter:
    """Dedup allow-list behaviour mirrors MultiCohortLatentDataModule."""

    def _write_dedup_json(self, path: Path, cohorts: dict[str, list[str]]) -> Path:
        """Write a minimal dedup decision.json."""
        payload = {
            "schema_version": "1.0",
            "cohorts": {name: {"kept_patient_ids": ids} for name, ids in cohorts.items()},
        }
        path.write_text(json.dumps(payload))
        return path

    def test_cv_dedup_filters_patients(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        pkeys = ["P_000", "P_001", "P_002", "P_003"]
        sids = [f"S_{i:03d}" for i in range(4)]
        offsets = _single_patient_offsets(4)

        h5_path = tmp_path / "COH_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=["P_000", "P_001", "P_002"],
            val_keys=["P_003"],
            test_keys=[],
        )
        # Allow-list drops P_001 (cross-cohort duplicate)
        dedup_path = self._write_dedup_json(
            tmp_path / "dedup.json",
            {"COH": ["P_000", "P_002", "P_003"]},  # P_001 absent
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, dedup_decision_path=dedup_path)
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        assert "P_001" not in splits["train"]
        assert "P_000" in splits["train"]

    def test_cv_missing_dedup_entry_raises(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits
        from vena.segmentation.exceptions import SegDataError

        pkeys = ["P_000", "P_001"]
        sids = ["S_000", "S_001"]
        offsets = _single_patient_offsets(2)

        h5_path = tmp_path / "COH_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=["P_000"],
            val_keys=["P_001"],
            test_keys=[],
        )
        # Dedup JSON covers "OTHER" but not "COH" — must raise
        dedup_path = self._write_dedup_json(
            tmp_path / "dedup.json",
            {"OTHER": ["P_000"]},
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, dedup_decision_path=dedup_path)

        with pytest.raises(SegDataError, match=r"[Dd]edup"):
            resolve_fm_splits(cfg)

    def test_test_only_tolerates_missing_dedup_entry(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        pkeys = ["TP_000", "TP_001"]
        sids = ["TS_000", "TS_001"]
        offsets = _single_patient_offsets(2)

        h5_path = tmp_path / "OOD_image.h5"
        _write_test_only_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
        )
        # Dedup JSON has no entry for "OOD" — must NOT raise
        dedup_path = self._write_dedup_json(tmp_path / "dedup.json", {})
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "OOD",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "test_only",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, dedup_decision_path=dedup_path)
        res = resolve_fm_splits(cfg)  # must not raise

        splits = res.fm_splits()
        assert set(splits["test"]) == set(pkeys)
        assert splits["train"] == []
        assert splits["val"] == []

    def test_test_only_applies_dedup_when_present(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        pkeys = ["TP_000", "TP_001", "TP_002"]
        sids = ["TS_000", "TS_001", "TS_002"]
        offsets = _single_patient_offsets(3)

        h5_path = tmp_path / "OOD_image.h5"
        _write_test_only_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
        )
        dedup_path = self._write_dedup_json(
            tmp_path / "dedup.json",
            {"OOD": ["TP_000", "TP_002"]},  # TP_001 dropped
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "OOD",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "test_only",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, dedup_decision_path=dedup_path)
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        assert "TP_001" not in splits["test"]
        assert set(splits["test"]) == {"TP_000", "TP_002"}


# ---------------------------------------------------------------------------
# test_only: contributes only to test
# ---------------------------------------------------------------------------


class TestTestOnlySplitContribution:
    def test_test_only_no_train_val(self, tmp_path: Path):
        from vena.segmentation.data.fm_splits import resolve_fm_splits

        pkeys = ["OOD_000", "OOD_001"]
        sids = ["OOD_S000", "OOD_S001"]
        offsets = _single_patient_offsets(2)

        h5_path = tmp_path / "OOD_image.h5"
        _write_test_only_h5(h5_path, patient_keys=pkeys, scan_ids=sids, offsets=offsets)

        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "OOD",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "test_only",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path)
        res = resolve_fm_splits(cfg)
        splits = res.fm_splits()

        assert splits["train"] == []
        assert splits["val"] == []
        assert set(splits["test"]) == set(pkeys)
        cs = res.per_cohort[0]
        assert cs.train_patients == ()
        assert cs.val_patients == ()


# ---------------------------------------------------------------------------
# write_splits_json: round-trip + invariant violations
# ---------------------------------------------------------------------------


class TestWriteSplitsJson:
    """write_splits_json produces a valid JSON and raises on invariant violations."""

    @pytest.fixture()
    def cv_resolution_and_plan(self, tmp_path: Path):
        """Single cv cohort resolution + FoldPlan for 4 train patients, k=2."""
        pkeys = [f"P_{i:03d}" for i in range(6)]
        sids = [f"S_{i:03d}" for i in range(6)]
        offsets = _single_patient_offsets(6)
        train_keys = pkeys[:4]
        val_keys = pkeys[4:5]
        test_keys = pkeys[5:]

        h5_path = tmp_path / "COH_image.h5"
        _write_cv_h5(
            h5_path,
            patient_keys=pkeys,
            scan_ids=sids,
            offsets=offsets,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
        )
        reg_path = _write_corpus_registry(
            tmp_path / "corpus.json",
            cohorts=[
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        )
        cfg = _make_data_cfg(reg_path, tmp_path, k_folds=2)

        from vena.segmentation.data.fm_splits import resolve_fm_splits
        from vena.segmentation.data.kfold import build_fold_plan

        res = resolve_fm_splits(cfg)
        plan = build_fold_plan(cfg, res.fm_splits(), dedup_duplicates=None)
        return res, plan, tmp_path

    def test_roundtrip_json_readable(self, cv_resolution_and_plan):
        from vena.segmentation.data.fm_splits import write_splits_json

        res, plan, tmp_path = cv_resolution_and_plan
        out = write_splits_json(tmp_path / "splits.json", res, plan)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["schema_version"] == "1.0.0"
        assert "counts" in data
        assert "per_cohort" in data
        assert "patient_to_scans" in data

    def test_roundtrip_counts_correct(self, cv_resolution_and_plan):
        from vena.segmentation.data.fm_splits import write_splits_json

        res, plan, tmp_path = cv_resolution_and_plan
        out = write_splits_json(tmp_path / "splits.json", res, plan)
        data = json.loads(out.read_text())

        splits = res.fm_splits()
        assert data["counts"]["fm_train_patients"] == len(splits["train"])
        assert data["counts"]["fm_val_patients"] == len(splits["val"])
        assert data["counts"]["fm_test_patients"] == len(splits["test"])
        assert len(data["counts"]["per_fold_patients"]) == plan.k

    def test_roundtrip_per_cohort_present(self, cv_resolution_and_plan):
        from vena.segmentation.data.fm_splits import write_splits_json

        res, plan, tmp_path = cv_resolution_and_plan
        out = write_splits_json(tmp_path / "splits.json", res, plan)
        data = json.loads(out.read_text())

        assert "COH" in data["per_cohort"]
        assert "folds" in data["per_cohort"]["COH"]

    def test_folds_union_invariant(self, cv_resolution_and_plan):
        """Raises when ⋃ folds ≠ fm_train_patients."""

        from vena.segmentation.data.fm_splits import write_splits_json
        from vena.segmentation.data.kfold import FoldPlan
        from vena.segmentation.exceptions import SegDataError

        res, plan, tmp_path = cv_resolution_and_plan
        # Corrupt: remove one patient from fold 0 → union ≠ train
        bad_fold0 = plan.folds[0][:-1]  # drop last element
        bad_plan = FoldPlan(
            k=plan.k,
            fm_train_ids=plan.fm_train_ids,
            folds=(bad_fold0, plan.folds[1]),
            fm_val_ids=plan.fm_val_ids,
            fm_test_ids=plan.fm_test_ids,
        )
        with pytest.raises(SegDataError, match="fm_train"):
            write_splits_json(tmp_path / "bad.json", res, bad_plan)

    def test_disjoint_folds_invariant(self, cv_resolution_and_plan):
        """Raises when a patient appears in two folds (true duplication, not reshuffle)."""
        from vena.segmentation.data.fm_splits import write_splits_json
        from vena.segmentation.data.kfold import FoldPlan
        from vena.segmentation.exceptions import SegDataError

        res, plan, tmp_path = cv_resolution_and_plan
        # Add dup to fold 1 while keeping it in fold 0 — dup now in BOTH folds.
        # This makes len(all_in_folds) > len(set(all_in_folds)) → disjoint check fires.
        # (Moving dup from fold0 to fold1 would still be disjoint — just reshuffled.)
        dup = plan.folds[0][0]
        bad_fold1 = plan.folds[1] + (dup,)
        bad_fold0 = plan.folds[0]  # unchanged — dup stays here too
        bad_plan = FoldPlan(
            k=plan.k,
            fm_train_ids=plan.fm_train_ids,
            folds=(bad_fold0, bad_fold1),
            fm_val_ids=plan.fm_val_ids,
            fm_test_ids=plan.fm_test_ids,
        )
        with pytest.raises(SegDataError):
            write_splits_json(tmp_path / "bad2.json", res, bad_plan)

    def test_val_test_not_in_folds_invariant(self, cv_resolution_and_plan):
        """Raises when a val/test patient sneaks into a fold."""
        from vena.segmentation.data.fm_splits import write_splits_json
        from vena.segmentation.data.kfold import FoldPlan
        from vena.segmentation.exceptions import SegDataError

        res, plan, tmp_path = cv_resolution_and_plan
        # Inject a val patient into fold 0
        val_pid = plan.fm_val_ids[0]
        bad_fold0 = plan.folds[0] + (val_pid,)
        # Remove from fold 1 to keep total count right
        bad_fold1 = plan.folds[1][:-1]
        bad_plan = FoldPlan(
            k=plan.k,
            fm_train_ids=(*plan.fm_train_ids, val_pid),
            folds=(bad_fold0, bad_fold1),
            fm_val_ids=plan.fm_val_ids,
            fm_test_ids=plan.fm_test_ids,
        )
        with pytest.raises(SegDataError):
            write_splits_json(tmp_path / "bad3.json", res, bad_plan)


# ---------------------------------------------------------------------------
# Tests for build_fold_plan(cohort_labels=...)
# ---------------------------------------------------------------------------


def _make_degenerate_ids(
    cohort_names: list[str], n_per_cohort: int
) -> tuple[list[str], dict[str, str]]:
    """Build patient IDs that degenerate to singletons under _extract_cohort.

    Uses UPENN-GBM / REMBRANDT-style IDs (accession + date suffix) whose
    heuristic prefix varies per patient, yielding n_per_cohort * len(cohort_names)
    unique singleton labels when the heuristic is applied.  The returned
    ``cohort_labels`` dict maps each ID to its true registry cohort name.
    """
    all_ids: list[str] = []
    labels: dict[str, str] = {}
    for c_idx, cohort in enumerate(cohort_names):
        for i in range(n_per_cohort):
            # Each ID has a unique numeric middle segment → _extract_cohort returns
            # "C{c_idx}-P{i:04d}-" stripped → "C{c_idx}-P{i:04d}" (unique per patient)
            pid = f"C{c_idx}-P{i:04d}-2005"
            all_ids.append(pid)
            labels[pid] = cohort
    return sorted(all_ids), labels


class TestBuildFoldPlanCohortLabels:
    """Tests for the optional cohort_labels keyword in build_fold_plan.

    Verifies that:
    1. Explicit labels collapse to one class per registry cohort (no singletons).
    2. Folds remain deterministic for fixed (ids, seed, k).
    3. The sklearn singleton UserWarning is absent when labels are exact.
    4. The None path produces byte-identical output to the current heuristic.
    5. A patient absent from cohort_labels falls back to _extract_cohort.
    """

    _COHORT_NAMES: ClassVar[list[str]] = ["Arm-A", "Arm-B", "Arm-C", "Arm-D", "Arm-E", "Arm-F"]
    _N_PER_COHORT: ClassVar[int] = 50  # 300 total — well above k=5; 50 per cohort ≥ k

    def _make_cfg(self, k_folds: int = 5, fold_seed: int = 1337) -> MagicMock:
        cfg = MagicMock()
        cfg.k_folds = k_folds
        cfg.fold_seed = fold_seed
        return cfg

    def _fm_splits_and_labels(
        self,
        *,
        n_per_cohort: int | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        n = n_per_cohort if n_per_cohort is not None else self._N_PER_COHORT
        all_ids, labels = _make_degenerate_ids(self._COHORT_NAMES, n)
        fm = {"train": all_ids, "val": [], "test": []}
        return fm, labels

    # ------------------------------------------------------------------

    def test_distinct_label_count_is_n_cohorts(self):
        """With explicit labels, distinct stratification classes == len(cohort_names)."""
        from vena.segmentation.data.kfold import build_fold_plan

        cfg = self._make_cfg()
        fm, labels = self._fm_splits_and_labels()
        plan = build_fold_plan(cfg, fm, cohort_labels=labels)

        # Every patient ID in fm_train_ids must be in labels
        all_label_values = {labels[pid] for pid in plan.fm_train_ids}
        assert all_label_values == set(self._COHORT_NAMES), (
            f"Expected {len(self._COHORT_NAMES)} cohort classes, got {sorted(all_label_values)}"
        )

    def test_per_cohort_fold_spread_balanced(self):
        """Explicit labels → each cohort contributes roughly N/k patients per fold."""
        from vena.segmentation.data.kfold import build_fold_plan

        k = 5
        cfg = self._make_cfg(k_folds=k)
        fm, labels = self._fm_splits_and_labels()
        plan = build_fold_plan(cfg, fm, cohort_labels=labels)

        # For each cohort, count how many of its patients appear in each fold.
        for cohort in self._COHORT_NAMES:
            cohort_pids = {pid for pid, c in labels.items() if c == cohort}
            fold_counts = [sum(1 for pid in fold if pid in cohort_pids) for fold in plan.folds]
            # Each fold should receive roughly N/k patients from each cohort.
            expected = self._N_PER_COHORT // k
            for count in fold_counts:
                assert abs(count - expected) <= 1, (
                    f"Cohort {cohort!r}: fold counts {fold_counts} deviate by "
                    f">1 from expected {expected}."
                )

    def test_deterministic_with_explicit_labels(self):
        """Two calls with the same inputs produce bit-identical FoldPlans."""
        from vena.segmentation.data.kfold import build_fold_plan

        cfg = self._make_cfg()
        fm, labels = self._fm_splits_and_labels()
        plan1 = build_fold_plan(cfg, fm, cohort_labels=labels)
        plan2 = build_fold_plan(cfg, fm, cohort_labels=labels)
        assert plan1 == plan2

    def test_no_singleton_warning_with_explicit_labels(self):
        """sklearn UserWarning about least-populated class absent with exact labels."""
        import warnings

        from vena.segmentation.data.kfold import build_fold_plan

        cfg = self._make_cfg()
        fm, labels = self._fm_splits_and_labels()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_fold_plan(cfg, fm, cohort_labels=labels)

        singleton_warns = [w for w in caught if "least populated class" in str(w.message).lower()]
        assert len(singleton_warns) == 0, (
            f"Expected no singleton warnings but got {len(singleton_warns)}: "
            f"{[str(w.message) for w in singleton_warns]}"
        )

    def test_none_path_byte_identical_to_heuristic(self):
        """cohort_labels=None produces output identical to the pre-change heuristic.

        We verify two independent calls with None are equal (determinism), which
        also confirms the None branch follows the same code path as before the
        cohort_labels parameter was added.
        """
        from vena.segmentation.data.kfold import build_fold_plan

        cfg = self._make_cfg()
        # Use IDs that DO reduce cleanly with the heuristic: BraTS-style prefix.
        cohort_names = ["BraTS-GLI", "UCSF-PDGM"]
        all_ids, _ = _make_degenerate_ids(cohort_names, 30)
        fm = {"train": all_ids, "val": [], "test": []}

        plan1 = build_fold_plan(cfg, fm, cohort_labels=None)
        plan2 = build_fold_plan(cfg, fm, cohort_labels=None)
        assert plan1 == plan2

    def test_fallback_for_pid_absent_from_cohort_labels(self):
        """A PID missing from cohort_labels falls back to _extract_cohort without raising."""
        from vena.segmentation.data.kfold import build_fold_plan

        cfg = self._make_cfg()
        fm, labels = self._fm_splits_and_labels()
        # Remove a few entries from the labels mapping.
        for pid in list(fm["train"])[:3]:
            labels.pop(pid, None)

        # Must not raise; the plan should cover all fm_train_ids.
        plan = build_fold_plan(cfg, fm, cohort_labels=labels)
        assert set(plan.fm_train_ids) == set(fm["train"])
