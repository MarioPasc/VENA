"""Integration tests for MultiCohortLatentDataset and MultiCohortLatentDataModule.

Uses synthetic in-memory temp H5 files conforming to the schema 2.0.0 layout
described in the spec. No GPU, no MAISI checkpoint.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.data.registry.models import CohortEntry, CorpusRegistry
from vena.model.fm.lightning.data import (
    LatentH5Dataset,
    MultiCohortLatentDataModule,
    MultiCohortLatentDataset,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic H5s
# ---------------------------------------------------------------------------

LATENT_SHAPE = (4, 4, 4, 4)  # tiny shape for fast tests (not real 4,48,56,48)
MASK_CHANNELS = 3


def _build_h5(
    path: Path,
    patient_ids: list[str],
    scans_per_patient: list[int],
    split_train_patients: list[str],
    split_val_patients: list[str],
    split_test_patients: list[str],
    cohort_name: str = "TestCohort",
) -> None:
    """Write a minimal schema-2.0.0 H5 to *path*.

    Parameters
    ----------
    patient_ids : list[str]
        All patient keys in the cohort (unique).
    scans_per_patient : list[int]
        Number of scans per patient (same order as patient_ids).
    split_*_patients : list[str]
        Patient-level split keys (subsets of patient_ids).
    """
    scan_ids: list[str] = []
    offsets: list[int] = [0]
    for pid, n in zip(patient_ids, scans_per_patient):
        for s_idx in range(n):
            scan_ids.append(f"{pid}_s{s_idx}")
        offsets.append(offsets[-1] + n)

    n_scans = len(scan_ids)
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = cohort_name
        f.attrs["created_at"] = "2026-01-01T00:00:00Z"
        f.attrs["producer"] = "test_fixture"
        f.attrs["config_json"] = "{}"
        f.attrs["git_sha"] = "deadbeef"

        # ids — vlen str
        dt = h5py.special_dtype(vlen=str)
        ids_ds = f.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=dt)

        # latents
        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = rng.random((n_scans, *LATENT_SHAPE), dtype=np.float32).astype(
                np.float16
            )
            f.create_dataset(f"latents/{mod}", data=data, compression="gzip")

        # masks/tumor_latent
        mask_data = rng.random(
            (n_scans, MASK_CHANNELS, *LATENT_SHAPE[1:]), dtype=np.float32
        )
        f.create_dataset("masks/tumor_latent", data=mask_data, compression="gzip")

        # CSR patient grouping
        f.create_dataset(
            "patients/offsets",
            data=np.array(offsets, dtype=np.int32),
        )
        f.create_dataset(
            "patients/keys",
            data=np.array(patient_ids, dtype=object),
            dtype=dt,
        )

        # Splits (patient keys)
        f.create_dataset(
            "splits/test",
            data=np.array(split_test_patients, dtype=object),
            dtype=dt,
        )
        f.create_dataset(
            "splits/cv/fold_0/train",
            data=np.array(split_train_patients, dtype=object),
            dtype=dt,
        )
        f.create_dataset(
            "splits/cv/fold_0/val",
            data=np.array(split_val_patients, dtype=object),
            dtype=dt,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_cohort_registry(tmp_path_factory):
    """Build two cohort H5s and a CorpusRegistry pointing at them.

    Cohort A (cross-sectional): 6 patients × 1 scan.
        train=[P0..P3], val=[P4], test=[P5]
    Cohort B (longitudinal): 4 patients, P6/P7 have 1 scan, P8/P9 have 2 scans.
        train=[P6,P7,P8], val=[P9], test=[P8]   (P8 in both train & test is OK
        for this unit test — in production splits are disjoint at patient level).

    Note: we keep test split disjoint from train/val for correct no-straddle tests.
    """
    tmp = tmp_path_factory.mktemp("h5")

    # Cohort A
    h5_a = tmp / "cohort_a.h5"
    patients_a = [f"PA{i}" for i in range(6)]
    scans_a = [1] * 6
    _build_h5(
        h5_a,
        patient_ids=patients_a,
        scans_per_patient=scans_a,
        split_train_patients=["PA0", "PA1", "PA2", "PA3"],
        split_val_patients=["PA4"],
        split_test_patients=["PA5"],
        cohort_name="CohortA",
    )

    # Cohort B: 4 patients; P0,P1 → 1 scan; P2,P3 → 2 scans
    h5_b = tmp / "cohort_b.h5"
    patients_b = [f"PB{i}" for i in range(4)]
    scans_b = [1, 1, 2, 2]
    _build_h5(
        h5_b,
        patient_ids=patients_b,
        scans_per_patient=scans_b,
        split_train_patients=["PB0", "PB1", "PB2"],
        split_val_patients=["PB3"],
        split_test_patients=["PB3"],
        cohort_name="CohortB",
    )

    # Dummy image h5 paths (must exist for CorpusRegistry construction in tests
    # that call load_registry; here we pass paths directly so just touch them).
    img_a = tmp / "img_a.h5"
    img_b = tmp / "img_b.h5"
    img_a.touch()
    img_b.touch()

    registry = CorpusRegistry(
        schema_version="1.0.0",
        name="test_corpus",
        cohorts=[
            CohortEntry(
                name="CohortA",
                pathology="glioma",
                label_system="BraTS2021",
                role="cv",
                longitudinal=False,
                image_h5=img_a,
                latent_h5=h5_a,
                n_patients=6,
                n_scans=6,
                modalities=["t1pre", "t1c", "t2", "flair"],
                has_swan=False,
            ),
            CohortEntry(
                name="CohortB",
                pathology="glioma",
                label_system="BraTS2021",
                role="cv",
                longitudinal=True,
                image_h5=img_b,
                latent_h5=h5_b,
                n_patients=4,
                n_scans=6,
                modalities=["t1pre", "t1c", "t2", "flair"],
                has_swan=False,
            ),
        ],
    )
    return registry


# ---------------------------------------------------------------------------
# Tests: MultiCohortLatentDataset
# ---------------------------------------------------------------------------


class TestMultiCohortLatentDataset:
    def _build_tiny_dataset(self, h5_path: Path, ids: list[str]) -> LatentH5Dataset:
        return LatentH5Dataset(h5_path, ids)

    def test_len_and_cohort_key(self, two_cohort_registry) -> None:
        reg = two_cohort_registry
        cohort_a = reg.by_name("CohortA")
        cohort_b = reg.by_name("CohortB")
        ds_a = LatentH5Dataset(cohort_a.latent_h5, [f"PA{i}_s0" for i in range(6)])
        ds_b = LatentH5Dataset(cohort_b.latent_h5, ["PB0_s0", "PB1_s0", "PB2_s0", "PB2_s1"])

        multi = MultiCohortLatentDataset([("CohortA", ds_a), ("CohortB", ds_b)])
        assert len(multi) == 10
        assert multi.cohort_of(0) == "CohortA"
        assert multi.cohort_of(5) == "CohortA"
        assert multi.cohort_of(6) == "CohortB"
        assert multi.cohort_of(9) == "CohortB"

    def test_item_has_cohort_key(self, two_cohort_registry) -> None:
        reg = two_cohort_registry
        cohort_a = reg.by_name("CohortA")
        ds_a = LatentH5Dataset(cohort_a.latent_h5, ["PA0_s0", "PA1_s0"])
        cohort_b = reg.by_name("CohortB")
        ds_b = LatentH5Dataset(cohort_b.latent_h5, ["PB0_s0"])

        multi = MultiCohortLatentDataset([("CohortA", ds_a), ("CohortB", ds_b)])
        item_a = multi[0]
        item_b = multi[2]
        assert item_a["cohort"] == "CohortA"
        assert item_b["cohort"] == "CohortB"

    def test_item_tensor_shapes(self, two_cohort_registry) -> None:
        reg = two_cohort_registry
        cohort_a = reg.by_name("CohortA")
        ds_a = LatentH5Dataset(cohort_a.latent_h5, ["PA0_s0"])
        multi = MultiCohortLatentDataset([("CohortA", ds_a)])
        item = multi[0]
        for key in ("z_t1pre", "z_t1c", "z_t2", "z_flair"):
            assert key in item
            assert item[key].shape == torch.Size([4, 4, 4, 4])
        assert "m_wt" in item
        assert item["m_wt"].shape == torch.Size([1, 4, 4, 4])

    def test_cohort_ranges(self, two_cohort_registry) -> None:
        reg = two_cohort_registry
        ds_a = LatentH5Dataset(reg.by_name("CohortA").latent_h5, ["PA0_s0", "PA1_s0"])
        ds_b = LatentH5Dataset(reg.by_name("CohortB").latent_h5, ["PB0_s0"])
        multi = MultiCohortLatentDataset([("CohortA", ds_a), ("CohortB", ds_b)])
        ranges = {name: (start, length) for name, start, length in multi.cohort_ranges()}
        assert ranges["CohortA"] == (0, 2)
        assert ranges["CohortB"] == (2, 1)

    def test_out_of_bounds_raises(self, two_cohort_registry) -> None:
        ds_a = LatentH5Dataset(
            two_cohort_registry.by_name("CohortA").latent_h5, ["PA0_s0"]
        )
        multi = MultiCohortLatentDataset([("CohortA", ds_a)])
        with pytest.raises(IndexError):
            _ = multi[1]
        with pytest.raises(IndexError):
            _ = multi[-1]


# ---------------------------------------------------------------------------
# Tests: MultiCohortLatentDataModule
# ---------------------------------------------------------------------------


class TestMultiCohortLatentDataModule:
    def test_setup_and_non_empty(self, two_cohort_registry) -> None:
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        assert dm._train_ds is not None
        assert dm._val_ds is not None
        assert dm._test_ds is not None
        assert len(dm._train_ds) > 0
        assert len(dm._val_ds) > 0
        assert len(dm._test_ds) > 0

    def test_train_scan_count(self, two_cohort_registry) -> None:
        """Train: A has 4 patients × 1 scan = 4; B has 3 patients (1+1+2) = 4 scans."""
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        # Cohort A train: PA0-PA3 → 4 scans. Cohort B train: PB0,PB1,PB2 → 1+1+2=4 scans.
        assert len(dm._train_ds) == 8

    def test_val_scan_count(self, two_cohort_registry) -> None:
        """Val: A has 1 patient × 1 scan; B has 1 patient PB3 with 2 scans."""
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        # A: PA4 → 1 scan; B: PB3 → 2 scans → total 3
        assert len(dm._val_ds) == 3

    def test_item_cohort_key_in_dataset(self, two_cohort_registry) -> None:
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        item = dm._train_ds[0]
        assert "cohort" in item
        assert item["cohort"] in {"CohortA", "CohortB"}

    def test_no_patient_straddles_train_val(self, two_cohort_registry) -> None:
        """No patient_id can appear in both train and val within a cohort."""
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()

        def patient_ids_from(ds: MultiCohortLatentDataset) -> set[str]:
            return {ds[i]["patient_id"] for i in range(len(ds))}

        train_pids = patient_ids_from(dm._train_ds)
        val_pids = patient_ids_from(dm._val_ds)
        assert train_pids.isdisjoint(val_pids), (
            f"Patients in both train and val: {train_pids & val_pids}"
        )

    def test_multiscan_patient_contributes_two_rows(self, two_cohort_registry) -> None:
        """PB2 has 2 scans; both scan rows must appear in the train dataset."""
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        # patient_id in the LatentH5Dataset is the scan_id (passed as patient_ids).
        scan_ids_in_train = {dm._train_ds[i]["patient_id"] for i in range(len(dm._train_ds))}
        assert "PB2_s0" in scan_ids_in_train
        assert "PB2_s1" in scan_ids_in_train

    def test_train_dataloader_batch_keys(self, two_cohort_registry) -> None:
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        for key in ("z_t1pre", "z_t1c", "z_t2", "z_flair", "m_wt"):
            assert key in batch, f"Missing key: {key}"
        assert "cohort" in batch
        assert isinstance(batch["cohort"], list)
        assert len(batch["cohort"]) == 2
        # z_t1c shape: (batch_size, 4, H, W, D)
        assert batch["z_t1c"].shape[0] == 2

    def test_val_dataloader_sequential(self, two_cohort_registry) -> None:
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        loader = dm.val_dataloader()
        batches = list(loader)
        # All 3 val scans with batch_size=2 → 2 batches (last may have 1 item,
        # no drop_last for val).
        total = sum(b["z_t1c"].shape[0] for b in batches)
        assert total == 3

    def test_sampler_patient_indices_global_bounds(self, two_cohort_registry) -> None:
        """Every global index in the sampler must be within [0, len(train_ds))."""
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry, fold=0, batch_size=2, num_workers=0
        )
        dm.setup()
        all_indices = {
            scan
            for cohort_patients in dm._train_patient_scan_indices
            for patient in cohort_patients
            for scan in patient
        }
        n = len(dm._train_ds)
        assert all(0 <= idx < n for idx in all_indices), (
            f"Out-of-bounds sampler index. max={max(all_indices)}, n={n}"
        )

    def test_max_train_patients_per_cohort(self, two_cohort_registry) -> None:
        dm = MultiCohortLatentDataModule(
            registry=two_cohort_registry,
            fold=0,
            batch_size=2,
            num_workers=0,
            max_train_patients_per_cohort=2,
        )
        dm.setup()
        # Cohort A capped at 2 patients × 1 scan; Cohort B capped at 2 patients.
        # B: if PB0+PB1 chosen (each 1 scan) → 2 scans; if PB2 chosen → 2 scans.
        # Max total = 2+2=4 (depends on which patients chosen, but ≤ 4+4 original).
        assert len(dm._train_ds) <= 8  # original uncapped size
        # Also: each cohort contributes ≤ max_train_patients_per_cohort patients
        for cohort_patients in dm._train_patient_scan_indices:
            assert len(cohort_patients) <= 2


# ---------------------------------------------------------------------------
# Tests: CSR expansion helper
# ---------------------------------------------------------------------------


def test_expand_patients_to_scans_basic() -> None:
    offsets = np.array([0, 1, 3, 4], dtype=np.int32)  # patients: 1, 2, 1 scan
    keys = ["P0", "P1", "P2"]
    ids = ["P0_s0", "P1_s0", "P1_s1", "P2_s0"]
    scan_ids, p2l = MultiCohortLatentDataModule._expand_patients_to_scans(
        offsets, keys, ids, ["P1", "P2"]
    )
    assert scan_ids == ["P1_s0", "P1_s1", "P2_s0"]
    assert p2l == [[0, 1], [2]]


def test_expand_patients_missing_key_raises() -> None:
    offsets = np.array([0, 1], dtype=np.int32)
    keys = ["P0"]
    ids = ["P0_s0"]
    with pytest.raises(KeyError, match="MISSING"):
        MultiCohortLatentDataModule._expand_patients_to_scans(
            offsets, keys, ids, ["MISSING"]
        )
