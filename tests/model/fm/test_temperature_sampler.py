"""Unit tests for TemperatureBalancedSampler.

Tests cohort probability distributions at different temperatures,
within-batch diversity, __len__ correctness, and determinism.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from vena.model.fm.lightning.data import TemperatureBalancedSampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cps(
    n_patients_per_cohort: list[int],
    scans_per_patient: int = 1,
    global_offset: int = 0,
) -> list[list[list[int]]]:
    """Build a cohort_patient_scan_indices structure for tests."""
    cps: list[list[list[int]]] = []
    idx = global_offset
    for n_patients in n_patients_per_cohort:
        patients: list[list[int]] = []
        for _ in range(n_patients):
            patients.append(list(range(idx, idx + scans_per_patient)))
            idx += scans_per_patient
        cps.append(patients)
    return cps


# Two cohorts: 6 patients (A) and 4 patients (B), 1 scan each.
@pytest.fixture
def two_cohort_cps() -> list[list[list[int]]]:
    return _make_cps([6, 4])


# Two cohorts with multi-scan patients: A=6 scans (6×1), B=6 scans (4 patients,
# some with 2 scans). Global indices: A occupies 0-5, B occupies 6-11.
@pytest.fixture
def two_cohort_cps_multiscan() -> list[list[list[int]]]:
    # cohort A: 6 patients, 1 scan each → indices 0-5
    cohort_a: list[list[int]] = [[i] for i in range(6)]
    # cohort B: 4 patients — 2 have 1 scan, 2 have 2 scans → indices 6-11
    cohort_b: list[list[int]] = [[6], [7], [8, 9], [10, 11]]
    return [cohort_a, cohort_b]


# ---------------------------------------------------------------------------
# Tests: __len__
# ---------------------------------------------------------------------------


def test_len_default(two_cohort_cps) -> None:
    sampler = TemperatureBalancedSampler(two_cohort_cps, batch_size=4)
    total_scans = sum(sum(len(s) for s in p) for p in two_cohort_cps)
    expected = math.ceil(total_scans / 4) * 4
    assert len(sampler) == expected


def test_len_explicit_override(two_cohort_cps) -> None:
    sampler = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, length_in_batches=10
    )
    assert len(sampler) == 40


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed(two_cohort_cps) -> None:
    s1 = TemperatureBalancedSampler(two_cohort_cps, batch_size=4, seed=7)
    s2 = TemperatureBalancedSampler(two_cohort_cps, batch_size=4, seed=7)
    first_batch_1 = list(iter(s1))[:4]
    first_batch_2 = list(iter(s2))[:4]
    assert first_batch_1 == first_batch_2


def test_different_seeds_differ(two_cohort_cps) -> None:
    s1 = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, seed=7, length_in_batches=20
    )
    s2 = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, seed=99, length_in_batches=20
    )
    assert list(iter(s1)) != list(iter(s2))


def test_epochs_differ(two_cohort_cps) -> None:
    """Two successive iterations over the same sampler must produce different sequences."""
    sampler = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, seed=42, length_in_batches=20
    )
    epoch0 = list(iter(sampler))
    epoch1 = list(iter(sampler))
    assert epoch0 != epoch1


# ---------------------------------------------------------------------------
# Tests: tau=0 (uniform over cohorts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_cohort_frequencies(tau: float) -> None:
    """Empirical cohort frequencies must be close to the analytical expectation.

    The sampler draws the first ``min(B, C)`` slots WITHOUT replacement (one per
    cohort) and the remaining ``B - C`` slots WITH replacement ∝ p_c.  With
    ``batch_size=4`` and ``n_cohorts=2`` this yields:

        E[freq_i] = (1/C * n_distinct + p_c[i] * n_with_repl) / B
                  = (0.5 * 2 + p_c[i] * 2) / 4
                  = (0.5 + p_c[i]) / 2

    This is a strict consequence of the sampling algorithm; the test is asserting
    the correct analytical expectation rather than p_c directly.
    """
    n_patients = [6, 4]
    cps = _make_cps(n_patients)
    n_cohorts = len(n_patients)
    batch_size = 4
    n_batches = 2000
    sampler = TemperatureBalancedSampler(
        cps, batch_size=batch_size, tau=tau, seed=0, length_in_batches=n_batches
    )

    # Base cohort probabilities (before the distinct-first correction).
    nc = np.array(n_patients, dtype=np.float64)
    if tau == 0.0:
        weights = np.ones(n_cohorts, dtype=np.float64)
    else:
        weights = nc**tau
    p_c = weights / weights.sum()

    # Analytical expectation accounting for distinct-first slots.
    n_distinct = min(batch_size, n_cohorts)   # = 2
    n_with_repl = batch_size - n_distinct      # = 2
    p_expected = (1.0 / n_cohorts * n_distinct + p_c * n_with_repl) / batch_size

    # Count cohort occurrences in the full sample stream.
    # Cohort A: indices 0-5, Cohort B: indices 6-9.
    counts = Counter(
        0 if idx < n_patients[0] else 1 for idx in iter(sampler)
    )
    total = sum(counts.values())
    p_empirical = np.array([counts[0] / total, counts[1] / total])

    np.testing.assert_allclose(p_empirical, p_expected, atol=0.03)


# ---------------------------------------------------------------------------
# Tests: within-batch diversity
# ---------------------------------------------------------------------------


def test_within_batch_diversity(two_cohort_cps) -> None:
    """With batch_size=4 and 2 cohorts, every batch must have both cohorts."""
    sampler = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, tau=0.5, seed=1, length_in_batches=500
    )
    indices = list(iter(sampler))
    batch_size = 4
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    # cohort A: 0-5, cohort B: 6-9
    all_both = 0
    for batch in batches:
        cohorts_in_batch = {0 if idx < 6 else 1 for idx in batch}
        if len(cohorts_in_batch) == 2:
            all_both += 1

    # With batch_size=4 >= n_cohorts=2, every batch is guaranteed to have both.
    assert all_both == len(batches), (
        f"Expected all {len(batches)} batches to contain both cohorts; "
        f"got {all_both}"
    )


def test_min_distinct_cohorts_per_batch(two_cohort_cps_multiscan) -> None:
    """Each batch must have ≥ min(batch_size, n_cohorts) distinct cohorts."""
    n_cohorts = 2
    batch_size = 4
    sampler = TemperatureBalancedSampler(
        two_cohort_cps_multiscan,
        batch_size=batch_size,
        tau=0.5,
        seed=2,
        length_in_batches=200,
    )
    indices = list(iter(sampler))
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    min_distinct = min(batch_size, n_cohorts)

    for batch in batches:
        # cohort A: 0-5, cohort B: 6-11
        cohorts_in_batch = {0 if idx < 6 else 1 for idx in batch}
        assert len(cohorts_in_batch) >= min_distinct, (
            f"Batch {batch} has only {len(cohorts_in_batch)} distinct cohorts"
        )


# ---------------------------------------------------------------------------
# Tests: all indices within bounds
# ---------------------------------------------------------------------------


def test_indices_in_bounds(two_cohort_cps) -> None:
    total_scans = sum(sum(len(s) for s in p) for p in two_cohort_cps)
    sampler = TemperatureBalancedSampler(
        two_cohort_cps, batch_size=4, seed=0, length_in_batches=100
    )
    for idx in iter(sampler):
        assert 0 <= idx < total_scans, f"Out-of-bounds index: {idx}"


def test_multiscan_patient_scans_included(two_cohort_cps_multiscan) -> None:
    """All scan indices (including multi-scan patients) must appear in the stream."""
    # Run for many batches so every scan has a reasonable chance of appearing.
    sampler = TemperatureBalancedSampler(
        two_cohort_cps_multiscan, batch_size=4, seed=3, length_in_batches=5000
    )
    seen = set(iter(sampler))
    all_indices = {
        scan
        for cohort in two_cohort_cps_multiscan
        for patient in cohort
        for scan in patient
    }
    assert all_indices.issubset(seen), f"Missing indices: {all_indices - seen}"
