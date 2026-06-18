"""Unit tests for :func:`vena.preflight.decoder_lpl_profile.patient_sampler.select_patients_by_strata`.

Builds a synthetic latent H5 with known WT volumes and asserts the
sampler returns one patient per tertile (§4.7b strata sanity).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.preflight.decoder_lpl_profile.patient_sampler import (
    PatientPick,
    select_patients_by_strata,
)

pytestmark = pytest.mark.unit


def _build_synthetic_latent_h5(
    path: Path,
    *,
    patient_volumes: dict[str, float],
) -> None:
    """Construct a minimal latent H5: ids + masks/tumor_latent only.

    The sampler reads ``ids`` for the row→patient_id mapping and
    ``masks/tumor_latent`` to compute the per-row soft-sum volume. We
    encode the requested per-patient volume by setting all three
    channels uniformly so ``sum_channels`` integrates to the target.
    """
    ids = list(patient_volumes)
    n = len(ids)
    shape = (n, 3, 4, 4, 4)  # tiny latent grid for fast volume integration
    tumor = np.zeros(shape, dtype=np.float32)
    n_vox = shape[2] * shape[3] * shape[4]  # 64 voxels
    for i, pid in enumerate(ids):
        # Uniformly fill each channel so clip(sum_C, 0, 1) integrates to
        # patient_volumes[pid] (the soft-volume proxy).
        v = patient_volumes[pid]
        per_voxel = float(v) / float(n_vox)  # per-voxel WT mass
        # sum across 3 channels = per_voxel → each channel = per_voxel / 3
        tumor[i, ...] = max(per_voxel / 3.0, 0.0)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=np.array(ids, dtype=h5py.string_dtype()))
        f.create_dataset("masks/tumor_latent", data=tumor)


def test_picks_one_per_tertile(tmp_path: Path) -> None:
    """A 9-patient cohort with volumes spanning small/median/large
    yields one patient per stratum."""
    volumes = {
        f"P{i:02d}": float(i)  # 0, 1, 2, ..., 8
        for i in range(9)
    }
    h5_path = tmp_path / "cohort.h5"
    _build_synthetic_latent_h5(h5_path, patient_volumes=volumes)

    picks = select_patients_by_strata(h5_path, n_per_cohort=3, seed=0)
    assert len(picks) == 3
    assert {p.stratum for p in picks} == {"small", "median", "large"}

    by_stratum = {p.stratum: p for p in picks}
    # Small must come from the lower tertile (values 0..2).
    assert by_stratum["small"].wt_volume <= 3.0
    # Large must come from the upper tertile (values 6..8).
    assert by_stratum["large"].wt_volume >= 5.0


def test_deterministic_under_same_seed(tmp_path: Path) -> None:
    volumes = {f"P{i:02d}": float(i) for i in range(20)}
    h5_path = tmp_path / "cohort.h5"
    _build_synthetic_latent_h5(h5_path, patient_volumes=volumes)
    picks_a = select_patients_by_strata(h5_path, n_per_cohort=3, seed=42)
    picks_b = select_patients_by_strata(h5_path, n_per_cohort=3, seed=42)
    assert [p.patient_id for p in picks_a] == [p.patient_id for p in picks_b]


def test_empty_cohort_returns_empty(tmp_path: Path) -> None:
    h5_path = tmp_path / "cohort.h5"
    _build_synthetic_latent_h5(h5_path, patient_volumes={"only_one": 1.0})
    # Restrict to non-existent ids → empty result.
    picks = select_patients_by_strata(
        h5_path, n_per_cohort=3, eligible_ids=["does-not-exist"], seed=0
    )
    assert picks == []


def test_eligible_ids_filter(tmp_path: Path) -> None:
    volumes = {f"P{i:02d}": float(i) for i in range(9)}
    h5_path = tmp_path / "cohort.h5"
    _build_synthetic_latent_h5(h5_path, patient_volumes=volumes)
    eligible = {"P00", "P04", "P08"}
    picks = select_patients_by_strata(h5_path, n_per_cohort=3, eligible_ids=eligible, seed=0)
    assert {p.patient_id for p in picks}.issubset(eligible)


def test_patient_pick_dataclass_has_row_index(tmp_path: Path) -> None:
    volumes = {f"P{i:02d}": float(i) for i in range(9)}
    h5_path = tmp_path / "cohort.h5"
    _build_synthetic_latent_h5(h5_path, patient_volumes=volumes)
    picks = select_patients_by_strata(h5_path, n_per_cohort=3, seed=0)
    for p in picks:
        assert isinstance(p, PatientPick)
        assert p.row_index >= 0
        assert p.patient_id.startswith("P")
