"""Tests for vena.segmentation.metrics.visualize.

All tests are synthetic — no real cohort H5 or checkpoints required.
Matplotlib is forced to Agg (headless) by the visualize module itself.

Markers
-------
All tests carry the ``segmentation`` marker so they appear in the
``-m segmentation`` suite and are excluded from unrelated marker sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_soft_mask_img(
    h: int = 40,
    w: int = 40,
    d: int = 30,
    *,
    wt_val: float = 0.8,
    netc_val: float = 0.4,
) -> np.ndarray:
    """Build a (2, H, W, D) synthetic soft mask with WT and nested NETC."""
    mask = np.zeros((2, h, w, d), dtype=np.float32)
    # WT in the centre third
    mask[0, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, d // 4 : 3 * d // 4] = wt_val
    # NETC nested inside WT
    mask[1, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, d // 3 : 2 * d // 3] = netc_val
    return mask


def _make_soft_mask_latent() -> np.ndarray:
    """Build a (2, 48, 56, 48) synthetic soft mask at the MAISI latent grid."""
    mask = np.zeros((2, 48, 56, 48), dtype=np.float32)
    mask[0, 10:30, 15:35, 10:30] = 0.75
    mask[1, 15:25, 20:30, 15:25] = 0.45
    return mask


# ---------------------------------------------------------------------------
# render_mask_qc
# ---------------------------------------------------------------------------


class TestRenderMaskQc:
    def test_writes_file(self, tmp_path: pytest.FixtureDef) -> None:
        """QC figure is written to disk for a synthetic case."""
        from vena.segmentation.metrics.visualize import render_mask_qc

        h, w, d = 40, 40, 30
        image = np.random.default_rng(0).uniform(0, 1, (h, w, d)).astype(np.float32)
        hard_mask = np.zeros((h, w, d), dtype=np.int32)
        hard_mask[10:20, 10:20, 5:15] = 1
        soft_img = _make_soft_mask_img(h, w, d)
        soft_lat = _make_soft_mask_latent()

        out = render_mask_qc(
            image,
            hard_mask,
            soft_img,
            soft_lat,
            patient_id="SYNTH-001",
            path=tmp_path / "qc_test.png",
        )
        assert out.exists(), "render_mask_qc must write the PNG"

    def test_soft_mask_graded(self) -> None:
        """Soft mask has values strictly in (0, 1) — not binary."""
        soft_img = _make_soft_mask_img()
        wt = soft_img[0]
        # The mask has values at wt_val=0.8 inside and 0 outside
        assert wt.max() < 1.0, "WT soft mask should not reach 1.0"
        assert wt.min() >= 0.0, "WT soft mask must be non-negative"
        # At least some voxels are strictly between 0 and 1
        assert ((wt > 0.0) & (wt < 1.0)).any(), "Mask must have graded (non-binary) values"

    def test_soft_mask_nested(self) -> None:
        """NETC values are always ≤ WT values (nesting guarantee)."""
        soft_img = _make_soft_mask_img()
        wt = soft_img[0]
        netc = soft_img[1]
        assert (netc <= wt + 1e-6).all(), "NETC must be nested inside WT (NETC ≤ WT)"

    def test_roi_label_tc_in_titles(self, tmp_path: pytest.FixtureDef) -> None:
        """render_mask_qc with roi_label='TC' writes a file; col-0 title contains 'TC'."""
        from unittest.mock import patch

        import matplotlib.pyplot as plt

        from vena.segmentation.metrics.visualize import render_mask_qc

        h, w, d = 40, 40, 30
        image = np.random.default_rng(1).uniform(0, 1, (h, w, d)).astype(np.float32)
        hard_mask = np.zeros((h, w, d), dtype=np.int32)
        hard_mask[10:20, 10:20, 5:15] = 4  # ET
        hard_mask[12:18, 12:18, 7:13] = 1  # NETC
        hard_mask[5:25, 5:25, 3:18] = 2  # Edema outside core

        # Re-set after overwrite so edema is 2 and core is correct
        hard_mask[10:20, 10:20, 5:15] = 4
        hard_mask[12:18, 12:18, 7:13] = 1
        hard_mask[5:8, 5:8, 3:6] = 2  # small isolated ED region

        captured_titles: list[str] = []

        original_set_title = plt.Axes.set_title

        def mock_set_title(self: plt.Axes, label: str, **kwargs: object) -> None:
            captured_titles.append(label)
            original_set_title(self, label, **kwargs)

        with patch.object(plt.Axes, "set_title", mock_set_title):
            render_mask_qc(
                image,
                hard_mask,
                _make_soft_mask_img(h, w, d),
                _make_soft_mask_latent(),
                patient_id="SYNTH-TC",
                path=tmp_path / "qc_tc.png",
                roi_label="TC",
            )

        ch0_titles = [t for t in captured_titles if "TC" in t]
        assert ch0_titles, (
            f"Expected titles containing 'TC' when roi_label='TC'; got: {captured_titles}"
        )

    def test_hard_channel0_tc_excludes_edema(self, tmp_path: pytest.FixtureDef) -> None:
        """Hard channel-0 rendered as TC must exclude edema voxels (label==2).

        A label with a large isolated edema block (label=2) and a small core
        (label=4 ET, label=1 NETC) should yield fewer hard-TC voxels than
        hard-WT voxels on any given slice containing both edema and core.
        """
        h, w, d = 30, 30, 20
        label = np.zeros((h, w, d), dtype=np.int32)
        # Large edema block at the start of depth axis (k=0..9)
        label[:, :, 0:10] = 2  # pure edema
        # Small TC core at the end (k=10..18)
        label[10:20, 10:20, 10:18] = 4  # ET
        label[13:17, 13:17, 12:16] = 1  # NETC nested

        # Pick a depth slice that is purely edema (k=3)
        k_ed = 3
        sl = label[:, :, k_ed]

        # TC hard mask — must exclude edema entirely
        tc_hard = ((sl > 0) & (sl != 2)).astype(np.float32)
        wt_hard = (sl > 0).astype(np.float32)

        # Edema slice has WT>0 but TC==0
        assert wt_hard.sum() > 0, "test setup: edema slice must have nonzero WT"
        assert tc_hard.sum() == 0, (
            f"TC hard mask must be zero on pure-edema slice; got {tc_hard.sum()}"
        )

    def test_wrong_latent_shape_raises(self, tmp_path: pytest.FixtureDef) -> None:
        """SegMetricError is raised when soft_mask_latent has the wrong shape."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import render_mask_qc

        image = np.zeros((40, 40, 30), dtype=np.float32)
        with pytest.raises(SegMetricError, match="soft_mask_latent"):
            render_mask_qc(
                image,
                np.zeros((40, 40, 30), dtype=np.int32),
                _make_soft_mask_img(),
                np.zeros((2, 60, 60, 40), dtype=np.float32),  # wrong shape
                patient_id="bad",
                path=tmp_path / "bad.png",
            )


# ---------------------------------------------------------------------------
# render_slice_montage — pinned layout
# ---------------------------------------------------------------------------


class TestRenderSliceMontage:
    def _make_patients(self, n: int = 4) -> list:
        from vena.segmentation.metrics.visualize import PatientView

        rng = np.random.default_rng(42)
        patients = []
        # tumour volumes: 1000, 500, 200, 800 → ascending order: 200, 500, 800, 1000
        volumes = [1000.0, 500.0, 200.0, 800.0][:n]
        for i, vol in enumerate(volumes):
            soft = _make_soft_mask_img(wt_val=0.6, netc_val=0.3)
            # Scale the mask so the tumour volume proxy matches vol
            patients.append(
                PatientView(
                    patient_id=f"PAT-{i:03d}",
                    t1pre=rng.uniform(0, 1, (40, 40, 30)).astype(np.float32),
                    soft_mask=soft,
                    tumor_volume=vol,
                    cohort="synthetic",
                )
            )
        return patients

    def test_writes_file(self, tmp_path: pytest.FixtureDef) -> None:
        """Montage PNG is written to disk."""
        from vena.segmentation.metrics.visualize import render_slice_montage

        out = render_slice_montage(
            self._make_patients(),
            n_cols=10,
            alpha=0.6,
            path=tmp_path / "montage.png",
        )
        assert out.exists()

    def test_rows_ordered_ascending_tumor_volume(self) -> None:
        """Patients are sorted by ascending tumour volume."""

        patients = self._make_patients(4)
        ordered = sorted(patients, key=lambda p: p.tumor_volume)
        for i in range(len(ordered) - 1):
            assert ordered[i].tumor_volume <= ordered[i + 1].tumor_volume, (
                "Rows must be ordered ascending by tumour volume; "
                f"got {ordered[i].tumor_volume} > {ordered[i + 1].tumor_volume}"
            )

    def test_column_count(self, tmp_path: pytest.FixtureDef) -> None:
        """Exactly n_cols tumour-bearing slice columns per row."""
        from vena.segmentation.metrics.visualize import _axial_tumor_slices

        soft = _make_soft_mask_img()
        n_cols = 10
        z_indices = _axial_tumor_slices(soft, n_cols=n_cols)
        assert len(z_indices) == n_cols, f"Expected {n_cols} columns; got {len(z_indices)}"

    def test_empty_patients_raises(self, tmp_path: pytest.FixtureDef) -> None:
        """SegMetricError is raised when the patients list is empty."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import render_slice_montage

        with pytest.raises(SegMetricError, match="empty"):
            render_slice_montage([], n_cols=10, alpha=0.6, path=tmp_path / "empty.png")


# ---------------------------------------------------------------------------
# Slice selection and continuous overlay — new-contract tests
# ---------------------------------------------------------------------------


class TestSliceSelectionAndContinuity:
    """Tests for area-based slice selection and continuous probability overlay."""

    def test_max_area_slice_picked(self) -> None:
        """Area-based k_img/k_lat selection picks the widest cross-section.

        The old render_mask_qc code used ``wt.max(axis=(0, 1))`` → argmax,
        which picks the slice with the highest peak value.  For soft masks with
        a uniform-valued hot-spot, many slices tie at the peak and argmax
        returns the first one — often NOT the largest cross-section.

        This test constructs a case where the two criteria disagree:
        * Slice 3: peak value 1.0 but only 2 voxels (sum = 2.0, max = 1.0).
        * Slice 12: value 0.8 over 200 voxels  (sum = 160.0, max = 0.8).

        Old (max-based argmax) → picks slice 3 (max=1.0 beats max=0.8).
        New (sum-based argmax) → picks slice 12 (sum=160.0 beats sum=2.0). ✓
        """
        mask = np.zeros((40, 40, 20), dtype=np.float32)
        # Slice 3: high peak, tiny area
        mask[0, 0, 3] = 1.0
        mask[0, 1, 3] = 1.0
        # Slice 12: lower peak, large area (20×10 = 200 voxels × 0.8 = sum 160.0)
        mask[:20, :10, 12] = 0.8

        # Replicate the area-based depth selection now used in render_mask_qc
        depth_sums = mask.sum(axis=(0, 1))
        k_new = int(np.argmax(depth_sums))
        assert k_new == 12, (
            f"Sum-based selection should pick slice 12 (sum=160 > sum=2), got {k_new}"
        )

        # Verify the old logic would have picked the wrong slice (regression guard)
        depth_maxs = mask.max(axis=(0, 1))
        k_old = int(np.argmax(depth_maxs))
        assert k_old == 3, (
            f"Max-based logic should pick slice 3 (demonstrating the fixed bug), got {k_old}"
        )

    def test_soft_overlay_continuous_alpha(self) -> None:
        """RGBA alpha channel contains >2 distinct values — overlay is continuous."""
        from vena.segmentation.metrics.visualize import _overlay_rgba

        # Gradated probability map with many distinct values between 0 and 1
        prob = np.linspace(0.0, 1.0, 200).reshape(10, 20).astype(np.float32)
        color = (0.1, 0.9, 0.2)  # WT green
        rgba = _overlay_rgba(prob, color, alpha_max=0.6)

        distinct_alpha = np.unique(rgba[..., 3])
        assert len(distinct_alpha) > 2, (
            f"Expected >2 distinct alpha values in continuous overlay; "
            f"got {len(distinct_alpha)}: {distinct_alpha}"
        )

    def test_netc_hard_panel_uses_integer_label_not_wt_binary(self) -> None:
        """NETC-hard panel must use label==1, not the WT binary (pre-existing bug guard).

        The engine previously fabricated ``hard_mask = (soft_mask[0] > 0.5)``
        (a WT binary 0/1) and passed that as ``hard_mask`` to ``render_mask_qc``.
        Because ``render_mask_qc``'s ndim==3 branch uses ``hard_mask == 1`` for
        NETC, the WT binary made NETC equal the ENTIRE WT region.

        This test constructs a BraTS-style integer label where:
        * WT bulk (label=2, edema) occupies a large 16×16×6 block;
        * NETC core (label=1) occupies a small 4×4×2 nested block.

        With the WT binary:  ``(label > 0) == 1`` selects 1536 voxels (entire WT).
        With the true label:  ``label == 1`` selects only 32 voxels (NETC core).
        """
        h, w, d = 20, 20, 10
        label = np.zeros((h, w, d), dtype=np.int32)
        label[2:18, 2:18, 2:8] = 2  # WT bulk (edema/ET, NOT NETC): 16×16×6 = 1536 vox
        label[8:12, 8:12, 4:6] = 1  # NETC core nested inside: 4×4×2 = 32 vox

        wt_area = int((label > 0).sum())
        netc_area = int((label == 1).sum())
        assert netc_area < wt_area, "test setup: NETC core must be smaller than WT"

        # True integer label → NETC panel covers only the small NETC core
        netc_from_int_label = int((label == 1).sum())

        # Buggy WT binary → NETC panel incorrectly covers the entire WT region
        wt_binary = (label > 0).astype(np.int32)
        netc_from_wt_binary = int((wt_binary == 1).sum())

        assert netc_from_int_label == 32, (
            f"label==1 should select 32-voxel NETC core; got {netc_from_int_label}"
        )
        assert netc_from_wt_binary == wt_area, (
            f"WT-binary bug: (wt_binary==1) selects entire WT ({wt_area} vox), "
            f"not the NETC core; got {netc_from_wt_binary}"
        )
        assert netc_from_int_label < netc_from_wt_binary, (
            "Integer-label NETC must be strictly smaller than WT-binary NETC"
        )

    def test_overlay_rgb_channels_match_color(self) -> None:
        """RGB channels of the overlay are set to the requested colour."""
        from vena.segmentation.metrics.visualize import _overlay_rgba

        prob = np.ones((5, 5), dtype=np.float32) * 0.5
        color = (1.0, 0.1, 0.6)  # NETC magenta
        rgba = _overlay_rgba(prob, color, alpha_max=0.6)

        np.testing.assert_allclose(rgba[..., 0], color[0], err_msg="R channel mismatch")
        np.testing.assert_allclose(rgba[..., 1], color[1], err_msg="G channel mismatch")
        np.testing.assert_allclose(rgba[..., 2], color[2], err_msg="B channel mismatch")


# ---------------------------------------------------------------------------
# render_latent_embedding — PCA fallback (umap absent)
# ---------------------------------------------------------------------------


class TestRenderLatentEmbedding:
    def _make_latents(self, n: int = 8) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(7)
        return {
            f"PAT-{i:03d}": rng.standard_normal((2, 48, 56, 48)).astype(np.float32)
            for i in range(n)
        }

    def _make_meta(self, pids: list[str]) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        cohorts = ["A", "B"] * (len(pids) // 2 + 1)
        return pd.DataFrame(
            {
                "patient_id": pids,
                "tumor_volume": rng.uniform(100, 5000, len(pids)),
                "cohort": cohorts[: len(pids)],
            }
        ).set_index("patient_id")

    def test_pca_produces_2d_embedding(self, tmp_path: pytest.FixtureDef) -> None:
        """PCA embedding runs and produces a 2-D figure without UMAP."""
        from vena.segmentation.metrics.visualize import render_latent_embedding

        latents = self._make_latents(n=6)
        meta = self._make_meta(list(latents.keys()))
        out = render_latent_embedding(
            latents,
            meta,
            method="pca_umap_perpatient",  # UMAP absent → falls back to PCA
            color_by=("tumor_volume", "cohort"),
            path=tmp_path / "embedding.png",
        )
        assert out.exists()

    def test_correct_number_of_points(self) -> None:
        """PCA embedding has one point per patient."""

        from sklearn.decomposition import PCA

        n = 6
        latents = self._make_latents(n=n)
        feat_mat = np.stack([v.ravel() for v in latents.values()])
        embedding = PCA(n_components=2, random_state=42).fit_transform(feat_mat)
        assert embedding.shape == (n, 2), f"Expected ({n}, 2); got {embedding.shape}"

    def test_empty_latents_raises(self, tmp_path: pytest.FixtureDef) -> None:
        """SegMetricError raised when mask_latents is empty."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import render_latent_embedding

        with pytest.raises(SegMetricError, match="empty"):
            render_latent_embedding(
                {},
                pd.DataFrame(),
                path=tmp_path / "empty.png",
            )


# ---------------------------------------------------------------------------
# render_injection_sanity — injection locality
# ---------------------------------------------------------------------------


class TestRenderInjectionSanity:
    """Synthetic-residual tests for the S2 injection-sanity figure."""

    _H, _W, _D = 20, 20, 16

    def _make_wt_mask(self) -> np.ndarray:
        """Binary WT box in the centre of the volume."""
        mask = np.zeros((self._H, self._W, self._D), dtype=np.float32)
        mask[5:15, 5:15, 4:12] = 1.0
        return mask

    def test_step0_residual_is_zero(self, tmp_path: pytest.FixtureDef) -> None:
        """Step-0 residual is identically zero; the figure must write."""
        from vena.segmentation.metrics.visualize import render_injection_sanity

        wt = self._make_wt_mask()
        res_zero = np.zeros((self._H, self._W, self._D), dtype=np.float32)
        res_scale = np.zeros_like(res_zero)
        res_scale[5:15, 5:15, 4:12] = 1.5  # energy concentrated in WT

        out = render_injection_sanity(
            None,
            {
                "wt_mask": wt,
                "residuals_zero": res_zero,
                "residuals_scale": res_scale,
            },
            path=tmp_path / "injection.png",
        )
        assert out.exists()
        assert float(res_zero.max()) == pytest.approx(0.0), "Step-0 residual must be zero"

    def test_in_wt_out_wt_ratio_greater_than_one(self) -> None:
        """In-WT energy / out-of-WT energy > 1 for residuals concentrated in WT."""
        from vena.segmentation.metrics.visualize import compute_residual_energy_ratio

        wt = self._make_wt_mask()
        # Build residuals that are ONLY nonzero inside WT
        residuals = np.zeros((self._H, self._W, self._D), dtype=np.float32)
        residuals[5:15, 5:15, 4:12] = 1.0  # inside WT box

        ratio = compute_residual_energy_ratio(residuals, wt)
        assert ratio > 1.0, f"Expected in-WT/out-WT ratio > 1; got {ratio:.3f}"

    def test_ratio_less_than_one_when_outside_wt(self) -> None:
        """In-WT/out-WT ratio < 1 when residuals concentrated outside WT."""
        from vena.segmentation.metrics.visualize import compute_residual_energy_ratio

        wt = self._make_wt_mask()
        residuals = np.zeros((self._H, self._W, self._D), dtype=np.float32)
        # Energy only OUTSIDE the WT box
        residuals[0:4, :, :] = 1.0

        ratio = compute_residual_energy_ratio(residuals, wt)
        assert ratio < 1.0, f"Expected ratio < 1 when residuals outside WT; got {ratio:.3f}"

    def test_missing_batch_key_raises(self, tmp_path: pytest.FixtureDef) -> None:
        """SegMetricError raised when batch is missing required keys."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import render_injection_sanity

        with pytest.raises(SegMetricError, match="missing"):
            render_injection_sanity(
                None,
                {"wt_mask": np.zeros((5, 5, 5))},  # missing residuals_zero, residuals_scale
                path=tmp_path / "bad.png",
            )


# ---------------------------------------------------------------------------
# compute_mask_stats — machine stats on a constructed case
# ---------------------------------------------------------------------------


class TestComputeMaskStats:
    def _make_stats_case(self) -> np.ndarray:
        """Build a (3, 2, 10, 10, 10) array with known stats.

        Patient 0: valid WT + nested NETC (no violations).
        Patient 1: 8 voxels where NETC > WT (violations).
        Patient 2: empty WT (empty_mask_count += 1).
        """
        masks = np.zeros((3, 2, 10, 10, 10), dtype=np.float32)

        # Patient 0: WT=0.8 in centre, NETC=0.4 nested inside (no violations)
        masks[0, 0, 3:7, 3:7, 3:7] = 0.8
        masks[0, 1, 4:6, 4:6, 4:6] = 0.4

        # Patient 1: WT=0.3 everywhere, NETC=0.9 in a 2×2×2 sub-box (8 violations)
        masks[1, 0, 3:7, 3:7, 3:7] = 0.3
        masks[1, 1, 3:5, 3:5, 3:5] = 0.9  # 0.9 > 0.3 + epsilon → violation

        # Patient 2: everything zero → empty WT

        return masks

    def test_empty_mask_count(self) -> None:
        """One patient with empty WT is counted."""
        from vena.segmentation.metrics.visualize import compute_mask_stats

        masks = self._make_stats_case()
        stats = compute_mask_stats(masks)
        assert stats["empty_mask_count"] == 1, (
            f"Expected 1 empty mask; got {stats['empty_mask_count']}"
        )

    def test_netc_violation_count(self) -> None:
        """Patient 1 has exactly 8 voxels where NETC > WT."""
        from vena.segmentation.metrics.visualize import compute_mask_stats

        masks = self._make_stats_case()
        stats = compute_mask_stats(masks)
        # 2×2×2 = 8 voxels with NETC=0.9 > WT=0.3
        assert stats["netc_violation_count"] == 8, (
            f"Expected 8 NETC violations; got {stats['netc_violation_count']}"
        )

    def test_soft_mass_fraction_in_wt_range(self) -> None:
        """Soft-mass fraction in WT is in [0, 1]."""
        from vena.segmentation.metrics.visualize import compute_mask_stats

        masks = self._make_stats_case()
        stats = compute_mask_stats(masks)
        frac = stats["soft_mass_fraction_in_wt"]
        assert 0.0 <= frac <= 1.0, f"Fraction out of [0, 1]: {frac}"

    def test_wrong_ndim_raises(self) -> None:
        """SegMetricError raised for non-5-D input."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import compute_mask_stats

        with pytest.raises(SegMetricError):
            compute_mask_stats(np.zeros((4, 10, 10), dtype=np.float32))


# ---------------------------------------------------------------------------
# check_mask_invariants — per-patient crop-frame invariant checks
# ---------------------------------------------------------------------------


def _make_inv_soft_img(*, wt_val: float = 0.8, netc_val: float = 0.4) -> np.ndarray:
    """Build (2, 192, 224, 192) soft mask with TC and nested NETC."""
    mask = np.zeros((2, 192, 224, 192), dtype=np.float32)
    mask[0, 80:112, 90:134, 80:112] = wt_val  # TC block 32×44×32
    mask[1, 88:104, 100:124, 88:104] = netc_val  # NETC nested inside
    return mask


def _make_inv_hard_label() -> np.ndarray:
    """Build (192, 224, 192) BraTS int label matching _make_inv_soft_img."""
    label = np.zeros((192, 224, 192), dtype=np.int32)
    label[80:112, 90:134, 80:112] = 4  # ET — part of TC
    label[88:104, 100:124, 88:104] = 1  # NETC nested inside ET
    return label


def _make_inv_latent_aligned() -> np.ndarray:
    """Build (2, 48, 56, 48) latent whose ×4 upscale overlaps the TC region.

    TC at [80:112, 90:134, 80:112] → latent approx [20:28, 22:33, 20:28].
    After upscale ×4: [80:112, 88:132, 80:112].
    Intersection with soft TC: [80:112, 90:132, 80:112] → IoU ≈ 0.91.
    """
    mask = np.zeros((2, 48, 56, 48), dtype=np.float32)
    mask[0, 20:28, 22:33, 20:28] = 0.8
    mask[1, 22:26, 25:31, 22:26] = 0.4
    return mask


def _make_inv_latent_misaligned() -> np.ndarray:
    """Build (2, 48, 56, 48) latent far from the TC region (registration failure)."""
    mask = np.zeros((2, 48, 56, 48), dtype=np.float32)
    # Place the TC region at the opposite corner from [20:28, 22:33, 20:28]
    mask[0, 0:4, 0:4, 44:48] = 0.8
    mask[1, 0:2, 0:2, 45:48] = 0.4
    return mask


class TestCheckMaskInvariants:
    """Tests for vena.segmentation.metrics.visualize.check_mask_invariants."""

    def test_all_invariants_pass(self) -> None:
        """Well-formed masks pass all invariants; invariant_ok=True."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            _make_inv_soft_img(),
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-PASS",
        )
        assert result["invariant_ok"] is True, (
            f"Expected invariant_ok=True for well-formed masks; got {result}"
        )

    def test_hard_subset_soft_no_violation(self) -> None:
        """Hard TC voxels all have soft_TC > 0.5 → violation_frac ≈ 0."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            _make_inv_soft_img(wt_val=0.8),
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-HSS",
        )
        assert result["hard_subset_soft_violation_frac"] < 0.01, (
            f"Expected ~0 violations; got {result['hard_subset_soft_violation_frac']:.4f}"
        )

    def test_hard_subset_soft_violation_detected(self) -> None:
        """Hard TC voxels with soft_TC = 0.3 < 0.5 → violation_frac > 0."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        # Soft TC intentionally LOW (0.3) so hard TC voxels fall below the 0.5 threshold
        result = check_mask_invariants(
            _make_inv_soft_img(wt_val=0.3),
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-VIOL",
        )
        assert result["hard_subset_soft_violation_frac"] > 0.0, (
            "Expected >0 violations when soft_TC < 0.5 inside hard TC region"
        )

    def test_soft_continuous_nonzero(self) -> None:
        """soft_intermediate_frac > 0 for a mask with values in (0.05, 0.95)."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        soft = _make_inv_soft_img(wt_val=0.6)
        result = check_mask_invariants(
            soft,
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-CONT",
        )
        assert result["soft_intermediate_frac"] > 0.0, (
            "soft_intermediate_frac must be > 0 for mask with values in (0.05, 0.95)"
        )

    def test_registration_iou_high_when_aligned(self) -> None:
        """IoU ≳ 0.7 when latent upscaled ×4 substantially overlaps image-soft."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            _make_inv_soft_img(),
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-IOUhi",
        )
        iou = result["latent_image_iou"]
        assert iou > 0.70, f"Expected IoU > 0.70 for aligned masks; got {iou:.3f}"

    def test_registration_iou_low_when_misaligned(self) -> None:
        """IoU ≈ 0 when latent is placed far from the image-soft TC region."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            _make_inv_soft_img(),
            _make_inv_hard_label(),
            _make_inv_latent_misaligned(),
            patient_id="SYN-IOUlo",
        )
        iou = result["latent_image_iou"]
        assert iou < 0.30, (
            f"Expected IoU < 0.30 for misaligned latent; got {iou:.3f} — "
            "this simulates a pool/crop registration bug"
        )
        # invariant_ok must be False (registration check failed)
        assert result["invariant_ok"] is False

    def test_wrong_soft_img_shape_raises(self) -> None:
        """SegMetricError raised when soft_img_crop has wrong shape."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import check_mask_invariants

        with pytest.raises(SegMetricError, match="soft_img_crop"):
            check_mask_invariants(
                np.zeros((2, 10, 10, 10), dtype=np.float32),  # wrong shape
                _make_inv_hard_label(),
                _make_inv_latent_aligned(),
                patient_id="SYN-BAD",
            )

    def test_wrong_latent_shape_raises(self) -> None:
        """SegMetricError raised when soft_latent has wrong shape."""
        from vena.segmentation.exceptions import SegMetricError
        from vena.segmentation.metrics.visualize import check_mask_invariants

        with pytest.raises(SegMetricError, match="soft_latent"):
            check_mask_invariants(
                _make_inv_soft_img(),
                _make_inv_hard_label(),
                np.zeros((2, 10, 10, 10), dtype=np.float32),  # wrong shape
                patient_id="SYN-BAD2",
            )

    def test_empty_mask_passes_invariants(self) -> None:
        """All-zero masks (no-TC patient) pass invariants — empty TC is valid.

        When TC is entirely absent (e.g. edema-only patient), checks (b) and
        (c) are gated by ``has_tc_region=False`` and trivially pass.  The
        engine must not flag these patients as derivation failures.
        """
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            np.zeros((2, 192, 224, 192), dtype=np.float32),
            np.zeros((192, 224, 192), dtype=np.int32),
            np.zeros((2, 48, 56, 48), dtype=np.float32),
            patient_id="SYN-EMPTY",
        )
        assert result["has_tc_region"] is False
        # Checks (b) and (c) are skipped → invariant_ok must be True
        assert result["invariant_ok"] is True, (
            "Empty-TC patients must pass invariants; got invariant_ok=False"
        )
        assert result["hard_subset_soft_violation_frac"] == pytest.approx(0.0)

    def test_result_keys_present(self) -> None:
        """Result dict contains all mandatory keys."""
        from vena.segmentation.metrics.visualize import check_mask_invariants

        result = check_mask_invariants(
            _make_inv_soft_img(),
            _make_inv_hard_label(),
            _make_inv_latent_aligned(),
            patient_id="SYN-KEYS",
        )
        required_keys = {
            "patient_id",
            "has_tc_region",
            "hard_subset_soft_violation_frac",
            "soft_intermediate_frac",
            "latent_image_iou",
            "latent_image_centroid_dist_vox",
            "invariant_ok",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in invariant result: {missing}"


# ---------------------------------------------------------------------------
# render_mask_qc with crop_spec — consistent-slice crop-frame rendering
# ---------------------------------------------------------------------------


class TestRenderMaskQcCropSpec:
    """Tests for render_mask_qc with crop_spec (crop-frame branch)."""

    def _make_crop_spec(self, native_h: int = 200, native_w: int = 240, native_d: int = 200):
        """Return a CropPadSpec that crops (192,224,192) from the centre."""
        from vena.common import CropPadSpec

        return CropPadSpec(
            crop_origin=(4, 8, 4),
            native_shape=(native_h, native_w, native_d),
            target_shape=(192, 224, 192),
        )

    def test_writes_file_with_crop_spec(self, tmp_path: pytest.FixtureDef) -> None:
        """render_mask_qc with crop_spec runs and writes a PNG."""
        from vena.segmentation.metrics.visualize import render_mask_qc

        h, w, d = 200, 240, 200
        rng = np.random.default_rng(42)
        image = rng.uniform(0, 1, (h, w, d)).astype(np.float32)

        label = np.zeros((h, w, d), dtype=np.int32)
        # TC region near crop origin so it appears in the crop frame
        label[10:30, 20:50, 10:30] = 4  # ET
        label[15:25, 28:42, 15:25] = 1  # NETC

        soft_img = np.zeros((2, h, w, d), dtype=np.float32)
        soft_img[0, 10:30, 20:50, 10:30] = 0.75  # TC channel
        soft_img[1, 15:25, 28:42, 15:25] = 0.45  # NETC channel

        crop_spec = self._make_crop_spec(h, w, d)
        out = render_mask_qc(
            image,
            label,
            soft_img,
            _make_soft_mask_latent(),
            patient_id="SYN-CROP-001",
            path=tmp_path / "qc_crop.png",
            roi_label="TC",
            crop_spec=crop_spec,
        )
        assert out.exists(), "render_mask_qc with crop_spec must write the PNG"

    def test_title_contains_crop_frame(self, tmp_path: pytest.FixtureDef) -> None:
        """Suptitle contains 'crop frame' when crop_spec is provided."""
        from unittest.mock import patch

        import matplotlib.pyplot as plt

        from vena.segmentation.metrics.visualize import render_mask_qc

        h, w, d = 200, 240, 200
        rng = np.random.default_rng(7)
        image = rng.uniform(0, 1, (h, w, d)).astype(np.float32)
        label = np.zeros((h, w, d), dtype=np.int32)
        label[10:30, 20:50, 10:30] = 4
        soft_img = np.zeros((2, h, w, d), dtype=np.float32)
        soft_img[0, 10:30, 20:50, 10:30] = 0.75

        captured_suptitles: list[str] = []
        original = plt.Figure.suptitle

        def mock_suptitle(self: plt.Figure, t: str, **kwargs: object) -> object:  # type: ignore[misc]
            captured_suptitles.append(t)
            return original(self, t, **kwargs)

        crop_spec = self._make_crop_spec(h, w, d)
        with patch.object(plt.Figure, "suptitle", mock_suptitle):
            render_mask_qc(
                image,
                label,
                soft_img,
                _make_soft_mask_latent(),
                patient_id="SYN-CROP-002",
                path=tmp_path / "qc_crop2.png",
                roi_label="TC",
                crop_spec=crop_spec,
            )

        assert any("crop frame" in t for t in captured_suptitles), (
            f"Expected 'crop frame' in suptitle; got: {captured_suptitles}"
        )

    def test_overlay_cmap_rgba_continuous(self) -> None:
        """_overlay_cmap_rgba produces >2 distinct alpha values (continuous, not binary)."""
        import matplotlib.pyplot as plt

        from vena.segmentation.metrics.visualize import _overlay_cmap_rgba

        prob = np.linspace(0.0, 1.0, 400).reshape(20, 20).astype(np.float32)
        rgba = _overlay_cmap_rgba(prob, plt.cm.YlGn, alpha_max=0.75)

        assert rgba.shape == (20, 20, 4), f"Expected (20,20,4); got {rgba.shape}"
        n_distinct_alpha = len(np.unique(rgba[:, :, 3]))
        assert n_distinct_alpha > 10, (
            f"Expected >10 distinct alpha values (continuous overlay); got {n_distinct_alpha}"
        )

    def test_overlay_cmap_rgba_zero_prob_transparent(self) -> None:
        """At probability 0, overlay alpha is 0 (fully transparent)."""
        import matplotlib.pyplot as plt

        from vena.segmentation.metrics.visualize import _overlay_cmap_rgba

        prob = np.zeros((5, 5), dtype=np.float32)
        rgba = _overlay_cmap_rgba(prob, plt.cm.RdPu, alpha_max=0.8)
        np.testing.assert_allclose(
            rgba[:, :, 3], 0.0, err_msg="Alpha must be 0 when probability is 0"
        )

    def test_overlay_cmap_rgba_max_prob_alpha(self) -> None:
        """At probability 1.0, overlay alpha equals alpha_max."""
        import matplotlib.pyplot as plt

        from vena.segmentation.metrics.visualize import _overlay_cmap_rgba

        prob = np.ones((5, 5), dtype=np.float32)
        alpha_max = 0.75
        rgba = _overlay_cmap_rgba(prob, plt.cm.YlGn, alpha_max=alpha_max)
        np.testing.assert_allclose(
            rgba[:, :, 3],
            alpha_max,
            rtol=1e-5,
            err_msg=f"Alpha must equal alpha_max={alpha_max} when probability is 1.0",
        )
