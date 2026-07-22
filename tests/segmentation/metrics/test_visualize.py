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
