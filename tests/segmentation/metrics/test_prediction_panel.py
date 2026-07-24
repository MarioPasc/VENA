"""Tests for PanelRow, sort_panel_rows, and render_prediction_panel.

All tests use synthetic arrays (32×32×24 volumes) — no checkpoints, no GPU,
no real cohort data.

Marker: segmentation  (included in the fast non-slow/non-gpu suite)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vena.segmentation.exceptions import SegMetricError
from vena.segmentation.metrics.visualize import (
    PanelRow,
    _axial_gt_slices,
    render_prediction_panel,
    sort_panel_rows,
)

pytestmark = pytest.mark.segmentation

# Synthetic volume dimensions — small enough for fast CPU rendering.
H, W, D = 32, 32, 24
N_COLS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    patient_id: str,
    metric: float,
    *,
    lesion_z_start: int | None = 5,
    lesion_z_end: int | None = 18,
    empty_tc: bool = False,
    metric_name: str = "dice_tc",
) -> PanelRow:
    """Build a synthetic PanelRow with optional lesion in a known z-range."""
    rng = np.random.default_rng(seed=abs(hash(patient_id)) % (2**31))
    anatomy = rng.random((H, W, D)).astype(np.float32)
    gt_hard = np.zeros((2, H, W, D), dtype=np.uint8)
    pred_soft = rng.random((2, H, W, D)).astype(np.float32)

    if not empty_tc and lesion_z_start is not None and lesion_z_end is not None:
        # TC occupies a rectangular box in the known z-range
        gt_hard[0, 10:22, 10:22, lesion_z_start:lesion_z_end] = 1
        # NETC is a subset of TC (as required by the region nesting contract)
        z_mid = (lesion_z_start + lesion_z_end) // 2
        gt_hard[1, 12:20, 12:20, z_mid - 1 : z_mid + 2] = 1

    return PanelRow(
        patient_id=patient_id,
        anatomy=anatomy,
        gt_hard=gt_hard,
        pred_soft=pred_soft,
        metric=metric,
        metric_name=metric_name,
    )


# ---------------------------------------------------------------------------
# render_prediction_panel — basic rendering
# ---------------------------------------------------------------------------


def test_render_5x10_panel_creates_nonempty_file(tmp_path: Path) -> None:
    """5 rows × 10 cols renders; file exists and is non-trivially sized."""
    rows = [
        _make_row("P001", 0.90),
        _make_row("P002", 0.75),
        _make_row("P003", 0.60),
        _make_row("P004", 0.45),
        _make_row("P005", 0.30),
    ]
    out = tmp_path / "panel.png"
    result = render_prediction_panel(rows, out, n_cols=N_COLS)
    assert result == out
    assert out.exists()
    # A 5-row × 10-col panel at dpi=120 should comfortably exceed 10 KB
    assert out.stat().st_size > 10_000, f"File too small: {out.stat().st_size} bytes"


def test_render_creates_parent_directories(tmp_path: Path) -> None:
    """Parent directories are created automatically."""
    out = tmp_path / "nested" / "dir" / "panel.png"
    render_prediction_panel([_make_row("X", 0.5)], out, n_cols=5)
    assert out.exists()


def test_render_returns_out_path(tmp_path: Path) -> None:
    """Return value equals the supplied out_path."""
    out = tmp_path / "p.png"
    returned = render_prediction_panel([_make_row("A", 0.8)], out, n_cols=3)
    assert returned == out


# ---------------------------------------------------------------------------
# sort_panel_rows
# ---------------------------------------------------------------------------


def test_sort_panel_rows_descending_order() -> None:
    """Rows are returned in strictly descending metric order."""
    rows = [
        _make_row("B", 0.3),
        _make_row("A", 0.7),
        _make_row("D", 0.5),
        _make_row("C", 0.9),
    ]
    sorted_rows = sort_panel_rows(rows)
    assert [r.patient_id for r in sorted_rows] == ["C", "A", "D", "B"]
    metrics = [r.metric for r in sorted_rows]
    assert all(metrics[i] >= metrics[i + 1] for i in range(len(metrics) - 1))


def test_sort_panel_rows_stable_tie_break_by_patient_id() -> None:
    """Rows with equal metrics are broken alphabetically by patient_id."""
    rows = [
        _make_row("Z", 0.5),
        _make_row("A", 0.5),
        _make_row("M", 0.5),
    ]
    sorted_rows = sort_panel_rows(rows)
    assert [r.patient_id for r in sorted_rows] == ["A", "M", "Z"]


def test_sort_panel_rows_single_row() -> None:
    """Single-row input passes through unchanged."""
    rows = [_make_row("only", 0.42)]
    assert sort_panel_rows(rows) == rows


def test_render_uses_sorted_order(tmp_path: Path) -> None:
    """The rendered panel places the highest-metric patient first (top row)."""
    # Build rows in worst-first order; render must sort them best-first.
    rows_worst_first = [
        _make_row("WORST", 0.10),
        _make_row("BEST", 0.95),
    ]
    out = tmp_path / "sorted.png"
    # No assertion on pixel content; just verify no crash and correct sort.
    sorted_rows = sort_panel_rows(rows_worst_first)
    assert sorted_rows[0].patient_id == "BEST"
    render_prediction_panel(rows_worst_first, out, n_cols=3)
    assert out.exists()


# ---------------------------------------------------------------------------
# Degenerate cases — empty GT TC (edema-only patient)
# ---------------------------------------------------------------------------


def test_empty_tc_row_renders_without_raising(tmp_path: Path) -> None:
    """A row with all-zero gt_hard[0] (edema-only) renders without raising."""
    rows = [_make_row("edema_only", 0.0, empty_tc=True)]
    out = tmp_path / "empty_tc.png"
    result = render_prediction_panel(rows, out, n_cols=N_COLS)
    assert result.exists()
    assert result.stat().st_size > 1_000


def test_empty_tc_within_mixed_panel_renders(tmp_path: Path) -> None:
    """A panel mixing normal and edema-only patients renders without raising."""
    rows = [
        _make_row("P_normal", 0.80),
        _make_row("P_edema", 0.40, empty_tc=True),
        _make_row("P_normal2", 0.60),
    ]
    out = tmp_path / "mixed.png"
    render_prediction_panel(rows, out, n_cols=5)
    assert out.exists()


# ---------------------------------------------------------------------------
# Lesion spanning fewer than n_cols slices
# ---------------------------------------------------------------------------


def test_narrow_lesion_renders_n_cols_columns(tmp_path: Path) -> None:
    """Lesion spanning 3 slices still produces a panel with n_cols=10 columns."""
    # Lesion confined to z=5..7 (3 slices), n_cols=10 → linspace repeats endpoints
    rows = [_make_row("narrow", 0.5, lesion_z_start=5, lesion_z_end=8)]
    out = tmp_path / "narrow.png"
    render_prediction_panel(rows, out, n_cols=N_COLS)
    assert out.exists()
    assert out.stat().st_size > 1_000


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_rows_raises_seg_metric_error() -> None:
    """Empty rows raises SegMetricError."""
    with pytest.raises(SegMetricError, match="empty"):
        render_prediction_panel([], Path("/tmp/x.png"), n_cols=N_COLS)


def test_gt_hard_wrong_channels_raises() -> None:
    """gt_hard with 3 channels instead of 2 raises SegMetricError."""
    anatomy = np.zeros((H, W, D), dtype=np.float32)
    gt_hard_bad = np.zeros((3, H, W, D), dtype=np.uint8)
    pred_soft = np.zeros((2, H, W, D), dtype=np.float32)
    row = PanelRow(
        patient_id="bad_gt",
        anatomy=anatomy,
        gt_hard=gt_hard_bad,
        pred_soft=pred_soft,
        metric=0.5,
    )
    with pytest.raises(SegMetricError, match="gt_hard"):
        render_prediction_panel([row], Path("/tmp/x.png"), n_cols=N_COLS)


def test_pred_soft_wrong_channels_raises() -> None:
    """pred_soft with 1 channel instead of 2 raises SegMetricError."""
    anatomy = np.zeros((H, W, D), dtype=np.float32)
    gt_hard = np.zeros((2, H, W, D), dtype=np.uint8)
    pred_soft_bad = np.zeros((1, H, W, D), dtype=np.float32)
    row = PanelRow(
        patient_id="bad_pred",
        anatomy=anatomy,
        gt_hard=gt_hard,
        pred_soft=pred_soft_bad,
        metric=0.5,
    )
    with pytest.raises(SegMetricError, match="pred_soft"):
        render_prediction_panel([row], Path("/tmp/x.png"), n_cols=N_COLS)


def test_anatomy_shape_mismatch_raises() -> None:
    """anatomy with wrong spatial shape raises SegMetricError."""
    anatomy_bad = np.zeros((16, 16, D), dtype=np.float32)  # H,W wrong
    gt_hard = np.zeros((2, H, W, D), dtype=np.uint8)
    pred_soft = np.zeros((2, H, W, D), dtype=np.float32)
    row = PanelRow(
        patient_id="bad_anat",
        anatomy=anatomy_bad,
        gt_hard=gt_hard,
        pred_soft=pred_soft,
        metric=0.5,
    )
    with pytest.raises(SegMetricError, match="anatomy"):
        render_prediction_panel([row], Path("/tmp/x.png"), n_cols=N_COLS)


def test_pred_soft_spatial_mismatch_raises() -> None:
    """pred_soft with wrong spatial dims relative to gt_hard raises SegMetricError."""
    anatomy = np.zeros((H, W, D), dtype=np.float32)
    gt_hard = np.zeros((2, H, W, D), dtype=np.uint8)
    pred_soft_bad = np.zeros((2, H, W, D + 4), dtype=np.float32)  # wrong D
    row = PanelRow(
        patient_id="bad_pred_spatial",
        anatomy=anatomy,
        gt_hard=gt_hard,
        pred_soft=pred_soft_bad,
        metric=0.5,
    )
    with pytest.raises(SegMetricError, match="pred_soft"):
        render_prediction_panel([row], Path("/tmp/x.png"), n_cols=N_COLS)


# ---------------------------------------------------------------------------
# _axial_gt_slices — slice selection verification
# ---------------------------------------------------------------------------


def test_slice_indices_within_lesion_z_range() -> None:
    """Slice indices for a lesion in z=[5, 18) all fall inside that range."""
    hard_tc = np.zeros((H, W, D), dtype=np.uint8)
    lesion_z_start, lesion_z_end = 5, 18
    hard_tc[10:22, 10:22, lesion_z_start:lesion_z_end] = 1

    z_indices = _axial_gt_slices(hard_tc, n_cols=N_COLS)

    assert len(z_indices) == N_COLS, f"Expected {N_COLS} indices; got {len(z_indices)}"
    for z in z_indices:
        assert lesion_z_start <= int(z) < lesion_z_end, (
            f"z={z} is outside lesion range [{lesion_z_start}, {lesion_z_end})"
        )


def test_slice_indices_empty_tc_span_full_depth() -> None:
    """Empty GT TC returns indices spanning [0, D-1]."""
    hard_tc = np.zeros((H, W, D), dtype=np.uint8)
    z_indices = _axial_gt_slices(hard_tc, n_cols=N_COLS)
    assert len(z_indices) == N_COLS
    assert int(z_indices[0]) == 0
    assert int(z_indices[-1]) == D - 1


def test_slice_indices_count_matches_n_cols() -> None:
    """_axial_gt_slices always returns exactly n_cols indices."""
    hard_tc = np.zeros((H, W, D), dtype=np.uint8)
    hard_tc[10:15, 10:15, 3:7] = 1
    for n in (1, 5, 10, 15):
        assert len(_axial_gt_slices(hard_tc, n_cols=n)) == n


# ---------------------------------------------------------------------------
# Composite layer-order tests
# ---------------------------------------------------------------------------


def _simulate_composite(
    anat_val: float,
    gt_val: float,
    tc_pred: float,
    gt_alpha: float = 0.6,
    soft_alpha: float = 0.6,
    gt_color: tuple[float, float, float] = (0.55, 0.0, 0.0),
) -> np.ndarray:
    """Simulate the render_prediction_panel layer order for a single pixel.

    Applies Porter-Duff 'over' compositing in the order:
      anatomy  →  GT hard (dark red)  →  TC soft (YlGn)  →  NETC soft (≈0)

    Parameters
    ----------
    anat_val : float
        Greyscale anatomy intensity in [0, 1].
    gt_val : float
        GT TC mask value (0 or 1).
    tc_pred : float
        Predicted TC soft probability in [0, 1].
    gt_alpha, soft_alpha : float
        Overlay alphas matching the defaults in render_prediction_panel.
    gt_color : tuple
        RGB dark-red colour matching the default gt_color.

    Returns
    -------
    np.ndarray
        Shape (3,) RGB pixel in [0, 1].
    """
    bg = np.array([anat_val, anat_val, anat_val], dtype=np.float64)

    # Layer 2: GT hard (dark red)
    a_gt = gt_alpha * gt_val
    bg = bg * (1.0 - a_gt) + np.array(gt_color) * a_gt

    # Layer 3: TC soft (YlGn)
    tc_rgba = plt.get_cmap("YlGn")(tc_pred)
    a_tc = tc_pred * soft_alpha
    bg = bg * (1.0 - a_tc) + np.array(tc_rgba[:3]) * a_tc

    return bg


def test_composite_order_red_dominant_at_zero_tc_prediction() -> None:
    """With GT present and TC pred≈0, composited pixel is red-dominant.

    Rationale: anatomy → GT (dark red) → TC soft (transparent at prob=0).
    Nothing covers the dark red, so R > G.
    """
    pixel = _simulate_composite(anat_val=0.0, gt_val=1.0, tc_pred=0.0)
    assert pixel[0] > pixel[1], (
        f"Expected red-dominant (false-negative region visible); R={pixel[0]:.4f} G={pixel[1]:.4f}"
    )
    # Sanity: R should equal gt_alpha * gt_color[0] = 0.6 * 0.55 = 0.33
    assert abs(pixel[0] - 0.33) < 0.01, f"Unexpected R value: {pixel[0]:.4f}"


def test_composite_order_green_dominant_at_high_tc_prediction() -> None:
    """With GT present and TC pred=1, YlGn cmap covers GT — pixel is green-dominant.

    Rationale: anatomy → GT (dark red) → TC soft (YlGn opaque at prob=1).
    YlGn(1.0) is dark green, so it partially covers the red.  G > R confirms
    the soft prediction is rendered ON TOP of the GT, not underneath it.
    """
    pixel = _simulate_composite(anat_val=0.0, gt_val=1.0, tc_pred=1.0)
    assert pixel[1] > pixel[0], (
        f"Expected green-dominant (YlGn over GT dark red); R={pixel[0]:.4f} G={pixel[1]:.4f}"
    )


def test_composite_order_inverted_if_gt_on_top_would_be_wrong() -> None:
    """Confirm the OLD (wrong) order — GT on top — would yield the opposite result.

    If GT were rendered last (topmost), the pixel at tc_pred=1 would still be
    red-dominant because the dark red GT would cover the YlGn.  This test
    documents why the order was changed.
    """
    # Simulate OLD order: anatomy → TC soft → GT hard
    anat_val = 0.0
    tc_pred = 1.0
    gt_val = 1.0
    gt_alpha = 0.6
    soft_alpha = 0.6
    gt_color = (0.55, 0.0, 0.0)

    bg = np.array([anat_val, anat_val, anat_val], dtype=np.float64)
    # Layer: TC soft first
    tc_rgba = plt.get_cmap("YlGn")(tc_pred)
    a_tc = tc_pred * soft_alpha
    bg = bg * (1.0 - a_tc) + np.array(tc_rgba[:3]) * a_tc
    # Layer: GT on top (old behaviour)
    a_gt = gt_alpha * gt_val
    bg = bg * (1.0 - a_gt) + np.array(gt_color) * a_gt

    # OLD order → red dominant even when prediction is high → diagnostic info hidden
    assert bg[0] > bg[1], (
        "OLD order (GT on top) makes pixel red even at tc_pred=1 — confirms the fix is needed."
    )


def test_composite_order_rendered_png_red_dominant(tmp_path: Path) -> None:
    """End-to-end: rendered PNG center is red-dominant when TC pred is zero.

    Uses a 1-row × 1-col panel with anatomy=0.5 gray, GT TC=1 everywhere,
    and pred_soft=0 (no prediction).  The center region of the PNG should be
    red-dominant because only the GT dark-red layer is visible (the TC soft
    overlay is fully transparent at prob=0).
    """
    import matplotlib.image as mplimg

    ph, pw, pd = 16, 16, 2  # panel-cell volume dimensions (distinct from module-level H,W,D)
    anatomy = np.full((ph, pw, pd), 0.5, dtype=np.float32)
    gt_hard = np.ones((2, ph, pw, pd), dtype=np.uint8)
    gt_hard[1] = 0  # NETC absent
    pred_soft = np.zeros((2, ph, pw, pd), dtype=np.float32)  # no prediction

    row = PanelRow(
        patient_id="order_test",
        anatomy=anatomy,
        gt_hard=gt_hard,
        pred_soft=pred_soft,
        metric=0.5,
    )
    out = tmp_path / "order_red.png"
    render_prediction_panel([row], out, n_cols=1, dpi=80)

    img = mplimg.imread(str(out))  # (H_px, W_px, ≥3) float [0, 1]
    h_px, w_px = img.shape[:2]
    # Sample the central quarter — below the title area, above any bottom margin
    y0, y1 = h_px // 4, 3 * h_px // 4
    x0, x1 = w_px // 4, 3 * w_px // 4
    center = img[y0:y1, x0:x1, :3]
    mean_r = float(center[:, :, 0].mean())
    mean_g = float(center[:, :, 1].mean())

    assert mean_r > mean_g, f"Center region not red-dominant: R={mean_r:.3f} G={mean_g:.3f}"


def test_composite_order_rendered_png_green_dominant(tmp_path: Path) -> None:
    """End-to-end: rendered PNG center is green-dominant when TC pred is one.

    Same setup as above but pred_soft[0]=1 (full TC confidence).  YlGn at
    max probability is dark-green; composited over the GT dark-red, the
    center region should satisfy G > R.
    """
    import matplotlib.image as mplimg

    ph, pw, pd = 16, 16, 2
    anatomy = np.full((ph, pw, pd), 0.5, dtype=np.float32)
    gt_hard = np.ones((2, ph, pw, pd), dtype=np.uint8)
    gt_hard[1] = 0
    pred_soft = np.zeros((2, ph, pw, pd), dtype=np.float32)
    pred_soft[0] = 1.0  # TC prediction = max everywhere

    row = PanelRow(
        patient_id="order_test_cmap",
        anatomy=anatomy,
        gt_hard=gt_hard,
        pred_soft=pred_soft,
        metric=0.5,
    )
    out = tmp_path / "order_green.png"
    render_prediction_panel([row], out, n_cols=1, dpi=80)

    img = mplimg.imread(str(out))
    h_px, w_px = img.shape[:2]
    y0, y1 = h_px // 4, 3 * h_px // 4
    x0, x1 = w_px // 4, 3 * w_px // 4
    center = img[y0:y1, x0:x1, :3]
    mean_r = float(center[:, :, 0].mean())
    mean_g = float(center[:, :, 1].mean())

    assert mean_g > mean_r, (
        f"Center region not green-dominant (YlGn should cover GT red): "
        f"R={mean_r:.3f} G={mean_g:.3f}"
    )
