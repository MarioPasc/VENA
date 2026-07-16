"""Tests for vena.validation.regions.

Covers: region_masks key set, dilation exactness, bg vs bg_undilated distinction.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.validation


def test_region_masks_keys() -> None:
    """region_masks returns exactly the required keys."""
    from vena.validation.regions import region_masks

    h, w, d = 10, 10, 10
    brain = np.ones((h, w, d), dtype=bool)
    wt = np.zeros((h, w, d), dtype=bool)
    wt[5, 5, 5] = True

    masks = region_masks(brain, wt)
    assert set(masks.keys()) == {"brain", "wt", "wt_dilated", "bg", "bg_undilated"}


def test_region_masks_dilate_k5_radius2() -> None:
    """dilate_k=5 gives exactly a radius-2 (5×5×5) dilation.

    A single foreground voxel at (cx, cy, cz) in an otherwise-zero volume
    should dilate into a (2k+1)=5×5×5 cube centred on it with kernel_size=5.
    """
    from vena.validation.regions import region_masks

    # Large enough volume that boundary effects don't matter.
    h, w, d = 20, 20, 20
    cx, cy, cz = 10, 10, 10

    brain = np.ones((h, w, d), dtype=bool)
    wt = np.zeros((h, w, d), dtype=bool)
    wt[cx, cy, cz] = True

    masks = region_masks(brain, wt, dilate_k=5)
    wt_dilated = masks["wt_dilated"]

    # Expected: a 5×5×5 cube from (cx-2, cy-2, cz-2) to (cx+2, cy+2, cz+2).
    expected = np.zeros((h, w, d), dtype=bool)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                expected[cx + dx, cy + dy, cz + dz] = True

    np.testing.assert_array_equal(
        wt_dilated.astype(np.uint8),
        expected.astype(np.uint8),
        err_msg="dilate_k=5 must give exactly a 5×5×5 dilation (radius 2)",
    )


def test_region_masks_bg_is_brain_minus_dilated_wt() -> None:
    """bg = brain AND NOT wt_dilated."""
    from vena.validation.regions import region_masks

    h, w, d = 12, 12, 12
    brain = np.ones((h, w, d), dtype=bool)
    brain[0, :, :] = False  # one face outside brain

    wt = np.zeros((h, w, d), dtype=bool)
    wt[6, 6, 6] = True

    masks = region_masks(brain, wt, dilate_k=3)

    expected_bg = brain & ~masks["wt_dilated"]
    np.testing.assert_array_equal(masks["bg"], expected_bg)


def test_region_masks_bg_undilated_is_brain_minus_wt() -> None:
    """bg_undilated = brain AND NOT wt (exact non-tumour region, §4.2).

    This is DISTINCT from bg which excludes the dilated tumour margin.
    Conflating them is trap #8 in SHARED_CONTRACTS §11.
    """
    from vena.validation.regions import region_masks

    h, w, d = 12, 12, 12
    brain = np.ones((h, w, d), dtype=bool)
    brain[0, :, :] = False

    wt = np.zeros((h, w, d), dtype=bool)
    wt[6, 6, 6] = True

    masks = region_masks(brain, wt, dilate_k=5)

    expected = brain & ~wt
    np.testing.assert_array_equal(masks["bg_undilated"], expected)

    # bg_undilated must include voxels in the dilation ring (bg excludes them).
    dilation_ring = masks["wt_dilated"] & ~wt
    assert np.any(dilation_ring), "test setup: dilation ring must be non-empty"
    # bg_undilated keeps the ring; bg excludes it
    assert np.any(masks["bg_undilated"] & dilation_ring), "bg_undilated includes dilation ring"
    assert not np.any(masks["bg"] & dilation_ring), "bg excludes dilation ring"


def test_region_masks_dilate_k1_identity() -> None:
    """dilate_k=1 is a no-op (kernel_size=1, padding=0)."""
    from vena.validation.regions import region_masks

    h, w, d = 10, 10, 10
    wt = np.zeros((h, w, d), dtype=bool)
    wt[5, 5, 5] = True
    brain = np.ones((h, w, d), dtype=bool)

    masks = region_masks(brain, wt, dilate_k=1)
    np.testing.assert_array_equal(masks["wt_dilated"], wt)
