"""Tests for ``clean_brain_mask`` — see ``.claude/notes/data/2026-06-18_data_audit.md``."""

from __future__ import annotations

import numpy as np
import pytest

from vena.data.h5.shared.brain_mask import clean_brain_mask

pytestmark = pytest.mark.unit


def _planted_mask() -> np.ndarray:
    m = np.zeros((40, 40, 40), dtype=np.int8)
    m[5:35, 5:35, 5:35] = 1  # big cerebrum-ish block, 27 000 voxels
    m[1:3, 1:3, 1:3] = 1  # 8-voxel noise CC
    m[38, 38, 38] = 1  # 1-voxel noise
    m[20:24, 0:4, 0:4] = 1  # 64-voxel noise (cerebellum-sized? no — too small)
    return m


def test_drops_small_cc_keeps_large() -> None:
    m = _planted_mask()
    out = clean_brain_mask(m, min_component_voxels=1000)
    # Big block survives.
    assert out[5:35, 5:35, 5:35].sum() == 30 * 30 * 30
    # All small CCs dropped.
    assert out[1:3, 1:3, 1:3].sum() == 0
    assert out[38, 38, 38] == 0
    assert out[20:24, 0:4, 0:4].sum() == 0


def test_preserves_cerebellum_sized_secondary_cc() -> None:
    """A second large CC (~simulated cerebellum) must survive alongside the cerebrum."""
    m = np.zeros((50, 50, 50), dtype=np.int8)
    m[5:25, 5:25, 5:25] = 1  # 8000-voxel "cerebrum"
    m[30:40, 30:40, 30:40] = 1  # 1000-voxel "cerebellum"
    m[0, 0, 0] = 1  # 1-voxel noise

    out = clean_brain_mask(m, min_component_voxels=500)
    assert out[5:25, 5:25, 5:25].sum() == 20 * 20 * 20
    assert out[30:40, 30:40, 30:40].sum() == 10 * 10 * 10
    assert out[0, 0, 0] == 0


def test_passthrough_when_single_component() -> None:
    m = np.zeros((20, 20, 20), dtype=np.int8)
    m[5:15, 5:15, 5:15] = 1
    out = clean_brain_mask(m, min_component_voxels=1000)
    assert np.array_equal(out, m)


def test_all_zero_input() -> None:
    m = np.zeros((10, 10, 10), dtype=np.int8)
    out = clean_brain_mask(m)
    assert out.sum() == 0


def test_below_threshold_falls_back_to_largest() -> None:
    """When every CC is sub-threshold, keep the biggest one rather than wiping."""
    m = np.zeros((10, 10, 10), dtype=np.int8)
    m[0:3, 0:3, 0:3] = 1  # 27-voxel
    m[7:9, 7:9, 7:9] = 1  # 8-voxel
    out = clean_brain_mask(m, min_component_voxels=10_000)
    # The 27-voxel block survives; the 8-voxel one is dropped.
    assert out[0:3, 0:3, 0:3].sum() == 27
    assert out[7:9, 7:9, 7:9].sum() == 0


def test_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        clean_brain_mask(np.zeros((10, 10), dtype=np.int8))


def test_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError):
        clean_brain_mask(np.zeros((10, 10, 10), dtype=np.int8), min_component_voxels=0)


def test_preserves_input_dtype() -> None:
    m = _planted_mask()
    out = clean_brain_mask(m)
    assert out.dtype == m.dtype


def test_does_not_mutate_input() -> None:
    m = _planted_mask()
    snapshot = m.copy()
    _ = clean_brain_mask(m)
    assert np.array_equal(m, snapshot)
