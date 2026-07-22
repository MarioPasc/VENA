"""Tests for vena.segmentation.targets (task 12).

All tests are pure-numpy — no GPU, no model weights required.

Coverage:
- harmonise_labels: BraTS-2021 ({0,1,2,4}) and BraTS-2023 ({0,1,2,3}) conventions.
- signed_distance: sphere geometry (SDT values and sigmoid), multi-component
  load-bearing test, nesting enforcement.
- make_soft_targets: shape, dtype, range, nesting, hard-target guard.
- geodesic sanity: high-intensity barrier forces larger effective distance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import distance_transform_edt

from vena.segmentation.config import TargetConfig
from vena.segmentation.exceptions import SegTargetError
from vena.segmentation.targets import (
    harmonise_labels,
    make_soft_targets,
    signed_distance,
    soft_target,
)

pytestmark = pytest.mark.segmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_mask(
    shape: tuple[int, int, int], centre: tuple[int, int, int], radius: float
) -> np.ndarray:
    """Boolean 3-D sphere mask."""
    gz, gy, gx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt((gz - centre[0]) ** 2 + (gy - centre[1]) ** 2 + (gx - centre[2]) ** 2)
    return dist < radius


def _sigmoid_np(x: float, sigma: float) -> float:
    return 1.0 / (1.0 + np.exp(-x / sigma))


# ---------------------------------------------------------------------------
# harmonise_labels
# ---------------------------------------------------------------------------


class TestHarmoniseLabels:
    """Label harmonisation for BraTS-2021 and BraTS-2023 conventions."""

    def test_brats2021_wt_and_netc(self) -> None:
        """BraTS-2021: {0,1,2,4} — WT=(label>0), NETC=(label==1)."""
        label = np.array([[[0, 1, 2, 4]]], dtype=np.int16)
        out = harmonise_labels(label)

        expected_wt = np.array([[[False, True, True, True]]])
        expected_netc = np.array([[[False, True, False, False]]])
        np.testing.assert_array_equal(out["wt"], expected_wt)
        np.testing.assert_array_equal(out["netc"], expected_netc)

    def test_brats2023_wt_and_netc(self) -> None:
        """BraTS-2023: {0,1,2,3} — same derived masks (code-agnostic)."""
        label = np.array([[[0, 1, 2, 3]]], dtype=np.int16)
        out = harmonise_labels(label)

        # Same semantics: WT=all non-zero, NETC=label==1
        expected_wt = np.array([[[False, True, True, True]]])
        expected_netc = np.array([[[False, True, False, False]]])
        np.testing.assert_array_equal(out["wt"], expected_wt)
        np.testing.assert_array_equal(out["netc"], expected_netc)

    def test_tc_brats2021_excludes_edema(self) -> None:
        """BraTS-2021: {0,1,2,4} — TC=(label==1)|(label==4), excludes ED=2."""
        label = np.array([[[0, 1, 2, 4]]], dtype=np.int16)
        out = harmonise_labels(label)

        # TC: NETC(1) + ET(4), NOT edema(2)
        expected_tc = np.array([[[False, True, False, True]]])
        np.testing.assert_array_equal(out["tc"], expected_tc)
        # Edema voxel (index 2, label=2) must NOT be in TC
        assert not out["tc"].flat[2], "edema (label=2) must be excluded from TC"

    def test_tc_brats2023_excludes_edema(self) -> None:
        """BraTS-2023: {0,1,2,3} — TC=(label==1)|(label==3), excludes ED=2."""
        label = np.array([[[0, 1, 2, 3]]], dtype=np.int16)
        out = harmonise_labels(label)

        # TC: NETC(1) + ET(3), NOT edema(2)
        expected_tc = np.array([[[False, True, False, True]]])
        np.testing.assert_array_equal(out["tc"], expected_tc)
        assert not out["tc"].flat[2], "edema (label=2) must be excluded from TC"

    def test_tc_dict_has_tc_key(self) -> None:
        """harmonise_labels returns a dict with a 'tc' key."""
        label = np.array([[[0, 1, 2, 4]]], dtype=np.int16)
        out = harmonise_labels(label)
        assert "tc" in out, "harmonise_labels must return a 'tc' key"

    def test_netc_subset_of_tc(self) -> None:
        """By construction NETC ⊆ TC (NETC = label==1 ⊆ (label>0)&(label!=2))."""
        rng = np.random.default_rng(0)
        label = rng.choice([0, 1, 2, 4], size=(10, 10, 10)).astype(np.int16)
        out = harmonise_labels(label)
        # Every NETC voxel must also be TC (label==1 implies label!=2 and label>0)
        assert np.all(out["tc"] | ~out["netc"]), "NETC must be a subset of TC"

    def test_tc_subset_of_wt(self) -> None:
        """TC ⊆ WT (TC excludes edema, but WT includes it)."""
        rng = np.random.default_rng(1)
        label = rng.choice([0, 1, 2, 4], size=(10, 10, 10)).astype(np.int16)
        out = harmonise_labels(label)
        # Every TC voxel is also WT
        assert np.all(out["wt"] | ~out["tc"]), "TC must be a subset of WT"

    def test_both_conventions_agree_on_wt_and_netc(self) -> None:
        """WT and NETC rules are identical for shared label values {0,1,2}."""
        label_2021 = np.array([[[0, 1, 2, 4]]], dtype=np.uint8)
        label_2023 = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)

        out_2021 = harmonise_labels(label_2021)
        out_2023 = harmonise_labels(label_2023)

        # WT: non-zero; both have 3 non-zero → agree
        assert out_2021["wt"].sum() == out_2023["wt"].sum() == 3
        # NETC: label==1 in both → single True
        assert out_2021["netc"].sum() == out_2023["netc"].sum() == 1

    def test_all_background(self) -> None:
        """All-zero label: WT, TC, and NETC are all False."""
        label = np.zeros((4, 4, 4), dtype=np.int32)
        out = harmonise_labels(label)
        assert not out["wt"].any()
        assert not out["tc"].any()
        assert not out["netc"].any()

    def test_return_dtype_is_bool(self) -> None:
        label = np.array([[[1, 2]]], dtype=np.int32)
        out = harmonise_labels(label)
        assert out["wt"].dtype == bool
        assert out["tc"].dtype == bool
        assert out["netc"].dtype == bool

    def test_empty_label_raises(self) -> None:
        with pytest.raises(SegTargetError, match="empty"):
            harmonise_labels(np.array([], dtype=np.int32).reshape(0, 1, 1))

    def test_netc_is_subset_of_wt(self) -> None:
        """By construction NETC ⊆ WT (label==1 implies label>0)."""
        rng = np.random.default_rng(0)
        label = rng.choice([0, 1, 2, 4], size=(10, 10, 10)).astype(np.int16)
        out = harmonise_labels(label)
        # Every NETC voxel must also be WT
        assert np.all(out["wt"] | ~out["netc"])


# ---------------------------------------------------------------------------
# signed_distance — sphere geometry
# ---------------------------------------------------------------------------


class TestSignedDistanceSphere:
    """Known-geometry tests on a sphere: SDT values and boundary sigmoid."""

    SHAPE = (51, 51, 51)
    CENTRE = (25, 25, 25)
    RADIUS = 10.0
    SIGMA = 2.0
    CLIP = 20.0

    @pytest.fixture(scope="class")
    def sphere_mask(self) -> np.ndarray:
        return _sphere_mask(self.SHAPE, self.CENTRE, self.RADIUS)

    @pytest.fixture(scope="class")
    def sdt(self, sphere_mask: np.ndarray) -> np.ndarray:
        return signed_distance(
            sphere_mask,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )

    def test_sdt_dtype(self, sdt: np.ndarray) -> None:
        assert sdt.dtype == np.float32

    def test_centre_is_positive_and_near_radius(self, sdt: np.ndarray) -> None:
        """SDT at centre ≈ radius (clipped to CLIP if r > CLIP)."""
        cx, cy, cz = self.CENTRE
        val = float(sdt[cx, cy, cz])
        assert val > 0, "Centre of sphere must have positive SDT"
        expected = min(self.RADIUS, self.CLIP)
        assert_allclose(val, expected, atol=1.5, err_msg="Centre SDT should be ~radius")

    def test_boundary_sdt_near_zero(self, sphere_mask: np.ndarray, sdt: np.ndarray) -> None:
        """Voxels exactly on the boundary shell have |SDT| ≈ 0."""
        # Boundary = inside but adjacent to outside
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(sphere_mask)
        boundary = sphere_mask & ~eroded
        boundary_vals = sdt[boundary]
        # All boundary voxels should be close to 0 (within ~sqrt(3) for discrete grids)
        assert np.all(np.abs(boundary_vals) < 2.0), (
            f"Max |SDT| at boundary shell: {np.abs(boundary_vals).max():.3f}"
        )

    def test_boundary_sigmoid_near_half(self, sphere_mask: np.ndarray, sdt: np.ndarray) -> None:
        """Mean sigmoid(SDT/σ) ≈ 0.5 across the boundary shell.

        The discrete boundary spans the SDT sign change: inner ring (inside,
        SDT≈+1) and outer ring (outside, SDT≈-1).  Symmetric averaging gives
        (sigmoid(+1/σ) + sigmoid(-1/σ)) / 2 ≈ 0.5 by the antisymmetry of
        sigmoid around 0.
        """
        from scipy.ndimage import binary_dilation, binary_erosion

        eroded = binary_erosion(sphere_mask)
        dilated = binary_dilation(sphere_mask)
        # Inner ring: inside voxels adjacent to outside (SDT ≈ +1)
        boundary_inner = sphere_mask & ~eroded
        # Outer ring: outside voxels adjacent to inside (SDT ≈ -1)
        boundary_outer = (~sphere_mask) & dilated
        boundary = boundary_inner | boundary_outer
        boundary_soft = 1.0 / (1.0 + np.exp(-sdt[boundary].astype(np.float64) / self.SIGMA))
        mean_soft = float(boundary_soft.mean())
        assert_allclose(mean_soft, 0.5, atol=0.08, err_msg="Mean boundary sigmoid should be ~0.5")

    def test_interior_sigmoid_above_half(self, sphere_mask: np.ndarray, sdt: np.ndarray) -> None:
        """All interior voxels have sigmoid(SDT/σ) > 0.5."""
        interior_sdt = sdt[sphere_mask]
        interior_soft = 1.0 / (1.0 + np.exp(-interior_sdt.astype(np.float64) / self.SIGMA))
        assert np.all(interior_soft > 0.5)

    def test_exterior_sigmoid_below_half(self, sphere_mask: np.ndarray, sdt: np.ndarray) -> None:
        """All exterior voxels have sigmoid(SDT/σ) < 0.5."""
        exterior_sdt = sdt[~sphere_mask]
        exterior_soft = 1.0 / (1.0 + np.exp(-exterior_sdt.astype(np.float64) / self.SIGMA))
        assert np.all(exterior_soft < 0.5)

    def test_monotonic_inward(self, sphere_mask: np.ndarray, sdt: np.ndarray) -> None:
        """SDT increases monotonically toward the sphere centre along the x-axis."""
        cy, cz = self.CENTRE[1], self.CENTRE[2]
        cx = self.CENTRE[0]
        # Sample along x from exterior (0) to centre (cx) at fixed y,z=centre
        vals = sdt[:cx, cy, cz]
        # Must be non-decreasing from exterior toward centre
        assert np.all(np.diff(vals) >= -0.5), (
            "SDT should be monotonically non-decreasing toward sphere centre"
        )


# ---------------------------------------------------------------------------
# signed_distance — multi-component (load-bearing test)
# ---------------------------------------------------------------------------


class TestMultiComponent:
    """Per-component SDT must NOT bridge the gap between disconnected lesions.

    Load-bearing test: two disjoint cubes with a gap between them.

    Per-component (correct):
        Gap voxels have negative SDT (clearly outside both components).
        sigmoid(SDT/sigma) < 0.5  — correctly identified as exterior.

    Naive global (wrong baseline):
        Simply using ``distance_transform_edt(mask)`` (the unsigned EDT, which
        returns 0 for all non-mask voxels) gives sigmoid(0/sigma) = 0.5 at the
        gap — spuriously treating gap voxels as if they were on the boundary.

    The per-component version is strictly lower (more "outside") at the gap.
    """

    # Volume shape
    SHAPE = (40, 10, 10)
    SIGMA = 2.0
    CLIP = 50.0

    @pytest.fixture(scope="class")
    def two_cube_mask(self) -> np.ndarray:
        mask = np.zeros(self.SHAPE, dtype=bool)
        mask[2:12, 2:8, 2:8] = True  # cube 1
        mask[28:38, 2:8, 2:8] = True  # cube 2 — gap at x=[12:28]
        return mask

    @pytest.fixture(scope="class")
    def gap_point(self) -> tuple[int, int, int]:
        """Midpoint of the gap, equidistant from both cubes."""
        return (20, 5, 5)

    def test_per_component_outside_at_gap(
        self, two_cube_mask: np.ndarray, gap_point: tuple[int, int, int]
    ) -> None:
        """Per-component soft value at gap centre must be < 0.5."""
        soft = soft_target(
            two_cube_mask,
            sigma_vox=self.SIGMA,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )
        val = float(soft[gap_point])
        assert val < 0.5, f"Per-component soft at gap centre = {val:.4f}; expected < 0.5"

    def test_naive_global_gives_spurious_boundary_at_gap(
        self, two_cube_mask: np.ndarray, gap_point: tuple[int, int, int]
    ) -> None:
        """Naive unsigned EDT gives sigmoid=0.5 at gap (spuriously high).

        Computed inline as contrast to per-component.  The naive approach
        uses only the unsigned EDT (= 0 for non-mask voxels) and feeds it
        directly into sigmoid, mistakenly treating all exterior voxels as if
        they lie on the boundary.
        """
        # Naive: unsigned EDT for non-mask voxels = 0 → sigmoid(0) = 0.5
        unsigned_edt = distance_transform_edt(two_cube_mask).astype(np.float32)
        naive_soft = float(1.0 / (1.0 + np.exp(-unsigned_edt[gap_point] / self.SIGMA)))
        # Gap voxels not in mask → unsigned_edt = 0 → sigmoid = 0.5
        assert_allclose(
            naive_soft, 0.5, atol=1e-6, err_msg="Naive unsigned EDT must give 0.5 at gap"
        )

    def test_per_component_strictly_lower_than_naive(
        self, two_cube_mask: np.ndarray, gap_point: tuple[int, int, int]
    ) -> None:
        """Per-component soft < naive global soft at gap centre (load-bearing)."""
        soft_pc = soft_target(
            two_cube_mask,
            sigma_vox=self.SIGMA,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )
        unsigned_edt = distance_transform_edt(two_cube_mask).astype(np.float32)
        naive_soft = float(1.0 / (1.0 + np.exp(-unsigned_edt[gap_point] / self.SIGMA)))
        val_pc = float(soft_pc[gap_point])
        assert val_pc < naive_soft, (
            f"Per-component ({val_pc:.4f}) must be strictly less than naive "
            f"global ({naive_soft:.4f}) at gap centre"
        )

    def test_gap_sdt_is_negative(
        self, two_cube_mask: np.ndarray, gap_point: tuple[int, int, int]
    ) -> None:
        """Per-component SDT at gap is negative (exterior)."""
        sdt = signed_distance(
            two_cube_mask,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )
        assert float(sdt[gap_point]) < 0.0

    def test_cube_interiors_are_positive(self, two_cube_mask: np.ndarray) -> None:
        """Interior voxels of each cube have positive SDT after union max."""
        sdt = signed_distance(
            two_cube_mask,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )
        # Interior of cube 1 (away from boundaries)
        assert float(sdt[7, 5, 5]) > 0.0, "Interior of cube 1 must be positive"
        # Interior of cube 2
        assert float(sdt[33, 5, 5]) > 0.0, "Interior of cube 2 must be positive"


# ---------------------------------------------------------------------------
# make_soft_targets — shape, dtype, range, nesting
# ---------------------------------------------------------------------------


class TestMakeSoftTargets:
    """Integration tests for the full make_soft_targets pipeline."""

    @pytest.fixture(scope="class")
    def cfg(self) -> TargetConfig:
        return TargetConfig(soft=True, sdt_sigma_vox=2.0, clip_vox=10.0)

    def _simple_label(self, shape: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
        label = np.zeros(shape, dtype=np.int16)
        # WT region (ED + NETC + ET)
        label[5:15, 5:15, 5:15] = 2  # ED
        label[7:13, 7:13, 7:13] = 4  # ET
        label[8:12, 8:12, 8:12] = 1  # NETC (innermost)
        return label

    def test_output_shape(self, cfg: TargetConfig) -> None:
        label = self._simple_label()
        out = make_soft_targets(label, cfg)
        assert out.shape == (2, 20, 20, 20), f"Expected (2,20,20,20), got {out.shape}"

    def test_output_dtype(self, cfg: TargetConfig) -> None:
        label = self._simple_label()
        out = make_soft_targets(label, cfg)
        assert out.dtype == np.float32

    def test_output_range(self, cfg: TargetConfig) -> None:
        label = self._simple_label()
        out = make_soft_targets(label, cfg)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_nesting_enforced(self, cfg: TargetConfig) -> None:
        """Channel 1 (NETC) must be ≤ channel 0 (WT) everywhere after make_soft_targets."""
        label = self._simple_label()
        out = make_soft_targets(label, cfg)
        wt_soft = out[0]
        netc_soft = out[1]
        assert np.all(netc_soft <= wt_soft + 1e-6), (
            "NETC soft must not exceed WT soft anywhere (nesting violated)"
        )

    def test_nesting_holds_on_random_labels(self, cfg: TargetConfig) -> None:
        """Nesting holds for randomly generated BraTS-style labels."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            # Random labels with values in {0,1,2,4}
            raw = rng.choice([0, 1, 2, 4], size=(16, 16, 16)).astype(np.int16)
            out = make_soft_targets(raw, cfg)
            assert np.all(out[1] <= out[0] + 1e-6)

    def test_wt_larger_than_netc_for_nested_anatomy(self, cfg: TargetConfig) -> None:
        """WT soft target is larger than NETC soft target in the WT region."""
        label = self._simple_label()
        out = make_soft_targets(label, cfg)
        # At the WT core (far from boundary), WT should be high
        wt_core = float(out[0, 10, 10, 10])
        netc_core = float(out[1, 10, 10, 10])
        assert wt_core >= netc_core

    def test_all_background_gives_low_values(self, cfg: TargetConfig) -> None:
        """All-background label → both channels near 0 (far outside)."""
        label = np.zeros((20, 20, 20), dtype=np.int16)
        out = make_soft_targets(label, cfg)
        # Interior voxels have no foreground → all clipped to -clip_vox → sigmoid ≈ 0
        # At least the centre should be well below 0.5
        assert float(out[:, 10, 10, 10].max()) < 0.5

    def test_default_tumor_region_is_tc(self) -> None:
        """TargetConfig default produces TC (not WT) in channel 0."""
        cfg_default = TargetConfig()
        assert cfg_default.tumor_region == "tc", (
            f"TargetConfig.tumor_region default must be 'tc'; got {cfg_default.tumor_region!r}"
        )

    def test_tc_channel0_excludes_edema(self) -> None:
        """With tumor_region='tc' (default), channel-0 soft is ~0 over ED-only voxels.

        Layout: a large ED-only block far from the core; a small NETC+ET core.
        TC excludes edema → channel-0 soft ≈ 0 (≤ 0.5) in the ED region.
        """
        cfg_tc = TargetConfig(soft=True, sdt_sigma_vox=2.0, clip_vox=10.0, tumor_region="tc")
        # (40, 20, 20) volume
        label = np.zeros((40, 20, 20), dtype=np.int16)
        # Large ED-only block at x=[0:15] — no NETC or ET here
        label[0:15, :, :] = 2
        # Small TC core at x=[30:38] — NETC+ET only, no edema
        label[30:38, 6:14, 6:14] = 4  # ET
        label[32:36, 8:12, 8:12] = 1  # NETC nested

        out = make_soft_targets(label, cfg_tc)
        # Centre of the ED-only region — should have low channel-0 with TC
        ed_center_val = float(out[0, 7, 10, 10])
        assert ed_center_val < 0.5, (
            f"TC channel-0 must be < 0.5 in edema-only region; got {ed_center_val:.4f}"
        )

    def test_wt_channel0_includes_edema(self) -> None:
        """With tumor_region='wt', channel-0 soft is high over the same ED region."""
        cfg_wt = TargetConfig(soft=True, sdt_sigma_vox=2.0, clip_vox=10.0, tumor_region="wt")
        label = np.zeros((40, 20, 20), dtype=np.int16)
        label[0:15, :, :] = 2  # large ED block
        label[30:38, 6:14, 6:14] = 4
        label[32:36, 8:12, 8:12] = 1

        out = make_soft_targets(label, cfg_wt)
        # Centre of ED region — should be high with WT
        ed_center_val = float(out[0, 7, 10, 10])
        assert ed_center_val > 0.5, (
            f"WT channel-0 must be > 0.5 in edema region; got {ed_center_val:.4f}"
        )

    def test_nesting_holds_for_tc(self) -> None:
        """NETC ≤ channel-0 everywhere with tumor_region='tc'."""
        cfg_tc = TargetConfig(soft=True, sdt_sigma_vox=2.0, clip_vox=10.0, tumor_region="tc")
        label = self._simple_label()
        out = make_soft_targets(label, cfg_tc)
        assert np.all(out[1] <= out[0] + 1e-6), "NETC must not exceed TC channel anywhere"

    def test_hard_target_mode_raises(self) -> None:
        cfg_hard = TargetConfig(soft=False)
        label = np.zeros((10, 10, 10), dtype=np.int16)
        with pytest.raises(SegTargetError, match="soft=True"):
            make_soft_targets(label, cfg_hard)

    def test_wrong_ndim_raises(self, cfg: TargetConfig) -> None:
        label_2d = np.zeros((10, 10), dtype=np.int16)
        with pytest.raises(SegTargetError, match="3-D"):
            make_soft_targets(label_2d, cfg)


# ---------------------------------------------------------------------------
# soft_target — sigma validation
# ---------------------------------------------------------------------------


class TestSoftTargetValidation:
    def test_zero_sigma_raises(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        with pytest.raises(SegTargetError, match="sigma_vox"):
            soft_target(mask, sigma_vox=0.0, mode="euclidean_percomponent", clip_vox=10.0)

    def test_negative_sigma_raises(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        with pytest.raises(SegTargetError, match="sigma_vox"):
            soft_target(mask, sigma_vox=-1.0, mode="euclidean_percomponent", clip_vox=10.0)


# ---------------------------------------------------------------------------
# signed_distance — geodesic sanity
# ---------------------------------------------------------------------------


class TestGeodesicSanity:
    """Geodesic distance > euclidean when a high-intensity barrier blocks the path.

    Geometry (1-D along x in a (25, 3, 3) volume):
        Blob mask at x=[0:5].
        High-intensity barrier in the image at x=[8:12].
        Test point at x=18 (outside blob, barrier lies between blob and point).

    Because the only path from x=18 to the nearest blob voxel (x=4) must cross
    the barrier (x=[8:12]), the MCP cost exceeds the bare Euclidean distance.
    Result: |sdt_geodesic| > |sdt_euclidean| → sdt_geodesic < sdt_euclidean
    (both negative; geodesic more negative = "further away").
    """

    SHAPE = (25, 3, 3)
    BLOB_X = slice(0, 5)  # mask True
    BARRIER_X = slice(8, 12)  # image high intensity
    BARRIER_INTENSITY = 10.0
    TEST_POINT = (18, 1, 1)
    CLIP = 200.0

    @pytest.fixture(scope="class")
    def mask(self) -> np.ndarray:
        m = np.zeros(self.SHAPE, dtype=bool)
        m[self.BLOB_X, :, :] = True
        return m

    @pytest.fixture(scope="class")
    def image(self) -> np.ndarray:
        img = np.zeros(self.SHAPE, dtype=np.float32)
        img[self.BARRIER_X, :, :] = self.BARRIER_INTENSITY
        return img

    def test_geodesic_sdt_more_negative_than_euclidean_at_barrier(
        self, mask: np.ndarray, image: np.ndarray
    ) -> None:
        """Geodesic distance > Euclidean at test point across the barrier."""
        sdt_euc = signed_distance(
            mask,
            mode="euclidean_percomponent",
            clip_vox=self.CLIP,
        )
        sdt_geo = signed_distance(
            mask,
            mode="geodesic",
            image=image,
            clip_vox=self.CLIP,
        )
        val_euc = float(sdt_euc[self.TEST_POINT])
        val_geo = float(sdt_geo[self.TEST_POINT])

        assert val_euc < 0.0, f"Test point must be outside blob; sdt_euc={val_euc}"
        assert val_geo < 0.0, f"Test point must be outside blob; sdt_geo={val_geo}"
        assert val_geo < val_euc, (
            f"Geodesic ({val_geo:.2f}) must be more negative than Euclidean "
            f"({val_euc:.2f}) at the barrier-blocked test point"
        )

    def test_geodesic_requires_image(self, mask: np.ndarray) -> None:
        with pytest.raises(SegTargetError, match="image"):
            signed_distance(mask, mode="geodesic", clip_vox=10.0)

    def test_geodesic_shape_mismatch_raises(self, mask: np.ndarray) -> None:
        wrong_image = np.zeros((99, 99, 99), dtype=np.float32)
        with pytest.raises(SegTargetError, match="shape"):
            signed_distance(mask, mode="geodesic", image=wrong_image, clip_vox=10.0)
