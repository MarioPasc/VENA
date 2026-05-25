"""Unit tests for :mod:`vena.preflight.vessel_mask.analysis`."""

from __future__ import annotations

import numpy as np
import pytest

from vena.preflight.vessel_mask import (
    binary_fraction,
    connected_components_stats,
    cylinder_volume,
    dice,
    jaccard,
    otsu_threshold_brainmasked,
    pick_threshold_by_anatomical_fraction,
    skeleton_length,
    sweep_thresholds,
)
from vena.preflight.vessel_mask.analysis import aggregate_per_tag


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestBinaryFraction:
    def test_empty_brain(self) -> None:
        soft = np.zeros((4, 4, 4), dtype=np.float32)
        brain = np.zeros_like(soft, dtype=np.uint8)
        assert binary_fraction(soft, brain, 0.5) == 0.0

    def test_all_above_threshold(self) -> None:
        soft = np.ones((4, 4, 4), dtype=np.float32)
        brain = np.ones_like(soft, dtype=np.uint8)
        assert binary_fraction(soft, brain, 0.5) == pytest.approx(1.0)

    def test_half_above_threshold(self) -> None:
        soft = np.zeros((4, 4, 4), dtype=np.float32)
        soft[:, :, :2] = 1.0  # half voxels at 1.0
        brain = np.ones_like(soft, dtype=np.uint8)
        assert binary_fraction(soft, brain, 0.5) == pytest.approx(0.5)

    def test_brain_restricts_count(self) -> None:
        """Voxels outside the brain do not count toward the fraction."""
        soft = np.ones((4, 4, 4), dtype=np.float32)
        brain = np.zeros_like(soft, dtype=np.uint8)
        brain[:2, :, :] = 1  # 32 brain voxels of 64 total
        assert binary_fraction(soft, brain, 0.5) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestConnectedComponents:
    def test_single_cylinder(self) -> None:
        """A single cylinder yields exactly one connected component."""
        img = cylinder_volume(size=20, radius_mm=2.0, foreground=1.0, background=0.0)
        binary = img > 0.5
        stats = connected_components_stats(binary)
        assert stats["n_components"] == 1
        assert stats["largest_fraction"] == pytest.approx(1.0)

    def test_two_disjoint_blobs(self) -> None:
        binary = np.zeros((16, 16, 16), dtype=bool)
        binary[2:5, 2:5, 2:5] = True
        binary[10:13, 10:13, 10:13] = True
        stats = connected_components_stats(binary)
        assert stats["n_components"] == 2
        assert stats["largest_voxels"] == 27

    def test_brain_restriction_drops_external_components(self) -> None:
        binary = np.zeros((16, 16, 16), dtype=bool)
        binary[0:3, 0:3, 0:3] = True  # blob outside brain
        binary[8:11, 8:11, 8:11] = True  # blob inside
        brain = np.zeros_like(binary)
        brain[6:, 6:, 6:] = True
        stats = connected_components_stats(binary, brain)
        assert stats["n_components"] == 1


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestSkeletonLength:
    def test_empty_input(self) -> None:
        binary = np.zeros((8, 8, 8), dtype=bool)
        assert skeleton_length(binary) == 0

    def test_thick_cylinder_skeletonises_to_centreline(self) -> None:
        """A finite solid cylinder of length L skeletonises to ~L voxels.

        The cylinder must NOT touch the cube boundary — Lee's 3D thinning
        (the algorithm under :func:`skimage.morphology.skeletonize`) treats
        boundary-touching voxels as removable, which erodes a cylinder that
        runs the full length of the cube. We pad with at least one zero
        voxel on every face.
        """
        size = 40
        length = 30  # voxels along z
        radius_vox = 4
        cz, cy, cx = size // 2, size // 2, size // 2
        zz, yy, xx = np.mgrid[:size, :size, :size]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        binary = (r2 < radius_vox**2) & (zz >= cz - length // 2) & (zz < cz + length // 2)
        n_skel = skeleton_length(binary)
        # The skeleton is the centreline; expect roughly the cylinder length
        # within a small tolerance (Lee thinning trims a few endpoint voxels).
        assert length - 6 <= n_skel <= length + 2, (
            f"Skeleton voxel count {n_skel} not close to cylinder length {length}"
        )


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestOtsu:
    def test_bimodal_distribution(self) -> None:
        """Otsu cutoff lies between the two modes of a bimodal histogram."""
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0.2, scale=0.05, size=8000)
        b = rng.normal(loc=0.8, scale=0.05, size=8000)
        soft = np.concatenate([a, b]).reshape(20, 20, 40).astype(np.float32)
        soft = np.clip(soft, 0.0, 1.0)
        brain = np.ones_like(soft, dtype=np.uint8)
        t = otsu_threshold_brainmasked(soft, brain)
        assert 0.3 < t < 0.7, f"Otsu cutoff {t} should sit between the two modes"


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestSetOverlap:
    def test_identical_masks(self) -> None:
        binary = np.zeros((8, 8, 8), dtype=bool)
        binary[2:6, 2:6, 2:6] = True
        brain = np.ones_like(binary)
        assert jaccard(binary, binary, brain) == pytest.approx(1.0)
        assert dice(binary, binary, brain) == pytest.approx(1.0)

    def test_disjoint_masks(self) -> None:
        a = np.zeros((8, 8, 8), dtype=bool)
        b = np.zeros_like(a)
        a[:4] = True
        b[4:] = True
        brain = np.ones_like(a)
        assert jaccard(a, b, brain) == 0.0
        assert dice(a, b, brain) == 0.0

    def test_known_overlap(self) -> None:
        a = np.zeros((4, 4, 4), dtype=bool)
        b = np.zeros_like(a)
        a[:, :, :2] = True   # 32 voxels
        b[:, :, 1:3] = True  # 32 voxels, 16 overlap with a (those at z=1)
        brain = np.ones_like(a)
        assert jaccard(a, b, brain) == pytest.approx(16 / 48)
        assert dice(a, b, brain) == pytest.approx(2 * 16 / 64)


@pytest.mark.unit
@pytest.mark.preflight_vessel
class TestSweepAndPick:
    def test_sweep_records_one_per_threshold(self) -> None:
        soft = np.random.RandomState(0).uniform(0, 1, (8, 8, 8)).astype(np.float32)
        brain = np.ones_like(soft, dtype=np.uint8)
        recs = sweep_thresholds(
            tag="x", patient_id="P", soft=soft, brain=brain,
            thresholds=(0.1, 0.3, 0.5),
        )
        assert len(recs) == 3
        assert [r.threshold for r in recs] == [0.1, 0.3, 0.5]
        # Binary fraction is monotone non-increasing in threshold.
        bf = [r.binary_fraction for r in recs]
        assert bf[0] >= bf[1] >= bf[2]

    def test_aggregate_combines_patients(self) -> None:
        soft = np.random.RandomState(0).uniform(0, 1, (8, 8, 8)).astype(np.float32)
        brain = np.ones_like(soft, dtype=np.uint8)
        recs = []
        for pid in ("P1", "P2", "P3"):
            recs.extend(
                sweep_thresholds(
                    tag="x", patient_id=pid, soft=soft, brain=brain,
                    thresholds=(0.1, 0.5),
                )
            )
        summary = aggregate_per_tag(recs)
        # 1 tag × 2 thresholds.
        assert len(summary) == 2
        assert all(s.n_patients == 3 for s in summary)

    def test_pick_in_band(self) -> None:
        """Picker chooses the in-band threshold closest to the midpoint."""
        soft = np.zeros((10, 10, 10), dtype=np.float32)
        # Construct three thresholds whose binary fractions land on either side
        # and at the midpoint of the target band [0.04, 0.07].
        brain = np.ones_like(soft, dtype=np.uint8)
        recs: list = []
        for pid in ("P",):
            # Build a deterministic soft so the fractions match the design.
            tmpl = np.zeros_like(soft)
            tmpl.flat[:30] = 0.9   # 0.030 fraction
            tmpl.flat[30:55] = 0.5  # cumulative 0.055
            tmpl.flat[55:120] = 0.2  # cumulative 0.120
            recs.extend(
                sweep_thresholds(
                    tag="x", patient_id=pid, soft=tmpl, brain=brain,
                    thresholds=(0.1, 0.3, 0.7),
                )
            )
        summary = aggregate_per_tag(recs)
        picked = pick_threshold_by_anatomical_fraction(
            summary, target_fraction_range=(0.04, 0.07)
        )
        # Only the second threshold (0.3) lands in [0.04, 0.07] (bf=0.055).
        assert picked["x"]["in_band"] is True
        assert picked["x"]["threshold"] == pytest.approx(0.3)

    def test_pick_outside_band_flags_extension(self) -> None:
        recs = []
        soft = np.zeros((10, 10, 10), dtype=np.float32)
        brain = np.ones_like(soft, dtype=np.uint8)
        for pid in ("P",):
            soft.flat[:2] = 0.9  # 0.002 fraction
            recs.extend(
                sweep_thresholds(
                    tag="x", patient_id=pid, soft=soft, brain=brain,
                    thresholds=(0.5,),
                )
            )
        summary = aggregate_per_tag(recs)
        picked = pick_threshold_by_anatomical_fraction(
            summary, target_fraction_range=(0.04, 0.07)
        )
        assert picked["x"]["in_band"] is False
        assert "EXTEND THE SWEEP" in picked["x"]["rationale"]
