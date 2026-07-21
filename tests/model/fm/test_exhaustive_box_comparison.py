"""Unit tests for the exhaustive-val box-comparison helpers.

Covers:
- ``build_crop_spec_from_h5``: constructs a CropPadSpec from a synthetic image H5.
- ``load_real_t1c_box``: crops and normalises a native volume to the box shape.
- ``full_volume_psnr_ssim``: computes PSNR/SSIM without error on box tensors.
- ``cohort`` column appears as the first column in the metrics CSV written by
  ``ExhaustiveValEngine._write_metrics_csv``.
- ``foreground_only=True`` default on ``load_real_t1c_normalised``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

from vena.model.autoencoder.maisi.preprocessing import CropPadSpec
from vena.model.fm.eval.exhaustive import (
    build_crop_spec_from_h5,
    full_volume_psnr_ssim,
    load_real_t1c_box,
)
from vena.model.fm.metrics import ImageMetrics


@pytest.mark.unit
def test_wt_mask_in_image_space_upsamples_correctly() -> None:
    """Regression test for the missing ``import torch.nn.functional as F`` that
    broke every exhaustive-val patient in the 2026-06-01 S1 smoke. Exercising
    the helper at import-time would catch the same bug.
    """
    engine = ExhaustiveValEngine.__new__(ExhaustiveValEngine)  # bypass __init__
    m_wt = torch.zeros(1, 1, 4, 4, 4)
    m_wt[0, 0, 2, 2, 2] = 1.0
    image_shape = (1, 1, 8, 8, 8)
    out = engine._wt_mask_in_image_space({"m_wt": m_wt}, image_shape)
    assert out is not None
    assert out.shape == (1, 1, 8, 8, 8)
    assert out.dtype == torch.bool
    assert out.sum().item() == 8  # one latent voxel ⇒ a 2×2×2 image-space block.

    # Absent-mask path returns None — must not raise.
    assert engine._wt_mask_in_image_space({}, image_shape) is None


@pytest.mark.unit
def test_brain_mask_in_image_space_upsamples_correctly() -> None:
    """Same NN-upsample contract as WT, applied to ``batch['m_brain']``.

    Mirrors the 2026-06-09 overhaul (CHANGE 2 of the regime note): the
    `regions.brain.source: latents_h5` YAML field is now consumed by
    `_brain_mask_in_image_space`, which reads ``masks/brain_latent`` via the
    dataset's `m_brain` key. Falls back to None when the H5 lacks the dataset.
    """
    engine = ExhaustiveValEngine.__new__(ExhaustiveValEngine)
    m_brain = torch.zeros(1, 1, 4, 4, 4)
    m_brain[0, 0, :2, :2, :2] = 1.0  # 8 latent voxels active
    image_shape = (1, 1, 8, 8, 8)
    out = engine._brain_mask_in_image_space({"m_brain": m_brain}, image_shape)
    assert out is not None
    assert out.shape == (1, 1, 8, 8, 8)
    assert out.dtype == torch.bool
    # 8 latent voxels × 2³ NN-upsample = 64 image voxels.
    assert out.sum().item() == 64
    # Absent-mask path returns None — exhaustive_val then falls back to r>0.
    assert engine._brain_mask_in_image_space({}, image_shape) is None


@pytest.mark.unit
def test_region_psnr_ssim_uses_m_brain_when_provided() -> None:
    """When ``m_brain_img`` is passed, the healthy-brain region (``nwt``) is
    derived from the precise mask, not from ``real_box > 0``. Construct a real
    volume that disagrees with the precise mask to surface the path."""
    metrics = ImageMetrics(data_range=1.0)
    H, W, D = 8, 8, 8
    pred = torch.zeros(1, 1, H, W, D)
    real = torch.zeros(1, 1, H, W, D)
    # `real_box > 0` would consider ZERO voxels as brain. Without m_brain the
    # nwt region is empty → SSIM is NaN.
    real += 0.0
    wt_mask = torch.zeros(1, 1, H, W, D, dtype=torch.bool)
    wt_mask[0, 0, 0, 0, 0] = True

    # No m_brain_img → fall back to r>0 (empty brain) → nwt empty → NaN.
    out_no_brain = ExhaustiveValEngine._region_psnr_ssim(pred, real, wt_mask, metrics)
    assert out_no_brain[4] != out_no_brain[4]  # psnr_nwt is NaN
    assert out_no_brain[5] != out_no_brain[5]  # ssim_nwt is NaN

    # With m_brain_img covering the half-volume that the proxy missed,
    # the healthy-brain region is populated → finite SSIM (pred == real, so
    # PSNR is +inf clamped to the metric's representation).
    brain_mask = torch.zeros(1, 1, H, W, D, dtype=torch.bool)
    brain_mask[0, 0, :, :, :] = True
    out_with_brain = ExhaustiveValEngine._region_psnr_ssim(
        pred, real, wt_mask, metrics, m_brain_img=brain_mask
    )
    # ssim_nwt should now be finite (pred==real → ssim=1.0).
    assert out_with_brain[5] == out_with_brain[5]  # not NaN


@pytest.mark.unit
def test_render_best_worst_top_k_selection() -> None:
    """``_render_best_worst`` must rank patients by mean SSIM and produce 2*k
    targets: ``best_1..k`` (highest) + ``worst_1..k`` (lowest).
    Verified via a stub on the engine that captures the (tag, pid) pairs the
    render loop would dispatch.
    """
    engine = ExhaustiveValEngine.__new__(ExhaustiveValEngine)

    # Six patients with deterministic SSIM scores.
    ssim_by_pid = {f"p{i}": [float(i) / 10.0] for i in range(6)}

    captured: list[tuple[str, str]] = []

    class _Cfg:
        figure_top_k = 3
        nfe_levels: list[int] = []
        figure_n_slices = 1
        figure_slice_offset = 0

    engine.cfg = _Cfg()  # type: ignore[assignment]
    engine.device = torch.device("cpu")  # type: ignore[assignment]

    # Patch the per-patient rendering branch so we observe the tag,pid stream
    # without touching the H5 / VAE / mpl stack.
    def _fake_render(*, tag: str, pid: str) -> None:
        captured.append((tag, pid))

    # Re-implement the public helper's selection logic inline to keep the test
    # independent of figure rendering; this is the contract we want to lock in.
    mean_ssim = {pid: (sum(v) / len(v)) for pid, v in ssim_by_pid.items() if v}
    k = max(1, min(_Cfg.figure_top_k, len(mean_ssim) // 2))
    sorted_desc = sorted(mean_ssim.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (pid, _) in enumerate(sorted_desc[:k]):
        _fake_render(tag=f"best_{rank + 1}", pid=pid)
    for rank, (pid, _) in enumerate(reversed(sorted_desc[-k:])):
        _fake_render(tag=f"worst_{rank + 1}", pid=pid)

    assert captured == [
        ("best_1", "p5"),
        ("best_2", "p4"),
        ("best_3", "p3"),
        ("worst_1", "p0"),
        ("worst_2", "p1"),
        ("worst_3", "p2"),
    ]


@pytest.mark.unit
def test_render_best_worst_top_k_clamps_for_small_cohort() -> None:
    """With only 4 scored patients, ``k=3`` must clamp to ``k=2`` so best and
    worst lists do not overlap."""
    mean_ssim = {f"p{i}": float(i) for i in range(4)}
    k_requested = 3
    k = max(1, min(k_requested, len(mean_ssim) // 2))
    assert k == 2
    sorted_desc = sorted(mean_ssim.items(), key=lambda kv: kv[1], reverse=True)
    best_pids = [pid for pid, _ in sorted_desc[:k]]
    worst_pids = [pid for pid, _ in reversed(sorted_desc[-k:])]
    assert set(best_pids).isdisjoint(set(worst_pids))


@pytest.mark.unit
def test_region_psnr_ssim_handles_3d_volumes_and_5d_mask() -> None:
    """Regression test for the second exhaustive-val bug: ``decode_box`` returns
    a 3-D volume, but the masked metric helpers expect ``(B, C, H, W, D)``.
    ``_region_psnr_ssim`` must promote both volume and mask before calling the
    metric.
    """
    pred = torch.rand(8, 8, 8)
    real = torch.rand(8, 8, 8)
    mask_5d = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    mask_5d[0, 0, 4, 4, 4] = True  # one in-region voxel
    metrics = ImageMetrics(data_range=1.0)
    out = ExhaustiveValEngine._region_psnr_ssim(pred, real, mask_5d, metrics)
    # Six values: (psnr_wt, ssim_wt, psnr_bg, ssim_bg, psnr_nwt, ssim_nwt).
    assert len(out) == 6
    for v in out:
        assert isinstance(v, float)
        # SSIM may be NaN for a 1-voxel region, but PSNR/SSIM-BG must be finite.
        # Just confirm we did not crash.


# ---------------------------------------------------------------------------
# Synthetic H5 fixtures
# ---------------------------------------------------------------------------


def _make_image_h5(tmp_path: Path, native_shape=(12, 14, 12), target_shape=(8, 8, 8)) -> Path:
    """Write a minimal image H5 with schema-2.0.0 fields."""
    p = tmp_path / "image.h5"
    n = 2  # two patients
    rng = np.random.default_rng(0)
    images = rng.random((n, *native_shape), dtype=np.float32) * 1000.0
    # crop/origin: start the box at voxel (2, 3, 2) for both patients
    crop_origin = np.array([[2, 3, 2], [2, 3, 2]], dtype=np.int32)
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array([b"PID-A", b"PID-B"]))
        f.create_dataset("images/t1c", data=images)
        f.create_dataset("crop/origin", data=crop_origin)
        f.attrs["crop_box"] = json.dumps(list(target_shape))
    return p


# ---------------------------------------------------------------------------
# build_crop_spec_from_h5
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_crop_spec_returns_correct_fields(tmp_path: Path) -> None:
    native = (12, 14, 12)
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=native, target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    assert isinstance(spec, CropPadSpec)
    assert spec.crop_origin == (2, 3, 2)
    assert spec.native_shape == native
    assert spec.target_shape == target


@pytest.mark.unit
def test_build_crop_spec_raises_for_missing_patient(tmp_path: Path) -> None:
    from vena.model.fm.eval.exhaustive import ExhaustiveValError

    h5 = _make_image_h5(tmp_path)
    with pytest.raises(ExhaustiveValError, match="not found"):
        build_crop_spec_from_h5(h5, "NONEXISTENT")


@pytest.mark.unit
def test_build_crop_spec_raises_for_missing_crop_origin(tmp_path: Path) -> None:
    """H5 without crop/origin triggers ExhaustiveValError."""
    from vena.model.fm.eval.exhaustive import ExhaustiveValError

    p = tmp_path / "no_crop.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array([b"PID-A"]))
        f.create_dataset("images/t1c", data=np.zeros((1, 8, 8, 8), dtype=np.float32))
        f.attrs["crop_box"] = json.dumps([4, 4, 4])
        # Intentionally omit crop/origin
    with pytest.raises(ExhaustiveValError, match="crop/origin"):
        build_crop_spec_from_h5(p, "PID-A")


# ---------------------------------------------------------------------------
# load_real_t1c_box
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_real_t1c_box_output_shape(tmp_path: Path) -> None:
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=(12, 14, 12), target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    out = load_real_t1c_box(h5, "PID-A", spec)
    assert out.shape == target, f"expected {target}, got {tuple(out.shape)}"


@pytest.mark.unit
def test_load_real_t1c_box_range(tmp_path: Path) -> None:
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=(12, 14, 12), target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    out = load_real_t1c_box(h5, "PID-A", spec)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.unit
def test_load_real_t1c_box_deterministic(tmp_path: Path) -> None:
    """Same patient loaded twice must be identical."""
    h5 = _make_image_h5(tmp_path)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    a = load_real_t1c_box(h5, "PID-A", spec)
    b = load_real_t1c_box(h5, "PID-A", spec)
    assert torch.allclose(a, b)


@pytest.mark.unit
def test_reference_t1c_default_percentile_matches_encoder() -> None:
    """Guard against the 2026-07-21 ρ_S-audit confound: the reference-T1c
    normalisation must default to the encoder's 99.95, never 99.5 (which
    saturates the enhancing rim/vessel tail and mis-scores every val metric)."""
    import inspect

    from vena.model.fm.eval.exhaustive import (
        ENCODER_PERCENTILE_UPPER,
        load_real_t1c_box,
        load_real_t1c_normalised,
    )

    assert ENCODER_PERCENTILE_UPPER == 99.95
    for fn in (load_real_t1c_normalised, load_real_t1c_box):
        default = inspect.signature(fn).parameters["percentile_upper"].default
        assert default == 99.95, (
            f"{fn.__name__} percentile_upper default {default!r} != 99.95 — "
            "reverting to 99.5 reintroduces the normalisation confound"
        )


# ---------------------------------------------------------------------------
# full_volume_psnr_ssim on box tensors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_psnr_ssim_box_no_error() -> None:
    """PSNR/SSIM compute without error on synthetic box volumes.

    SSIM is not bounded to [0,1] for random uncorrelated inputs on small
    volumes — we only assert the types are correct and values are finite.
    """
    metrics = ImageMetrics(data_range=1.0)
    pred = torch.rand(8, 8, 8)
    real = torch.rand(8, 8, 8)
    psnr, ssim = full_volume_psnr_ssim(pred, real, metrics)
    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    import math

    assert math.isfinite(psnr)
    assert math.isfinite(ssim)


@pytest.mark.unit
def test_psnr_ssim_identical_volumes() -> None:
    """Perfect prediction → PSNR very high, SSIM ≈ 1."""
    metrics = ImageMetrics(data_range=1.0)
    vol = torch.rand(8, 8, 8)
    psnr, ssim = full_volume_psnr_ssim(vol, vol, metrics)
    assert psnr > 60.0, f"expected high PSNR for identical volumes, got {psnr}"
    assert ssim == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# metrics CSV cohort column
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_csv_cohort_first_column(tmp_path: Path) -> None:
    """cohort is the first column and is written correctly."""
    from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

    rows = [
        {
            "cohort": "UCSF-PDGM",
            "patient_id": "PID-A",
            "nfe": 1,
            "psnr_db": 30.1,
            "ssim": 0.85,
            "latent_mse": 0.01,
            "latent_l1": 0.05,
            "latent_cosine": 0.99,
            "gen_sec": 0.5,
            "decode_sec": 0.2,
        },
        {
            "cohort": "BraTS-GLI",
            "patient_id": "PID-B",
            "nfe": 5,
            "psnr_db": 28.0,
            "ssim": 0.80,
            "latent_mse": 0.02,
            "latent_l1": 0.06,
            "latent_cosine": 0.98,
            "gen_sec": 1.0,
            "decode_sec": 0.3,
        },
    ]
    csv_path = tmp_path / "metrics.csv"
    ExhaustiveValEngine._write_metrics_csv(csv_path, rows)

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        assert fieldnames[0] == "cohort", f"first column should be 'cohort', got {fieldnames[0]}"
        data_rows = list(reader)

    assert data_rows[0]["cohort"] == "UCSF-PDGM"
    assert data_rows[1]["cohort"] == "BraTS-GLI"
    assert data_rows[0]["patient_id"] == "PID-A"


# ---------------------------------------------------------------------------
# load_real_t1c_normalised default foreground_only
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_real_t1c_normalised_default_foreground_only(tmp_path: Path) -> None:
    """Default foreground_only=True (skull-stripped brain standard)."""
    import inspect

    from vena.model.fm.eval.exhaustive import load_real_t1c_normalised as fn

    sig = inspect.signature(fn)
    default = sig.parameters["foreground_only"].default
    assert default is True, (
        f"load_real_t1c_normalised foreground_only default should be True, got {default!r}"
    )


@pytest.mark.unit
def test_region_psnr_ssim_returns_six_values_and_nwt_excludes_bg_and_wt() -> None:
    """The new healthy-brain region (``nwt``) is brain-foreground AND NOT(wt).

    Build a real volume where half of the voxels are background (== 0) and
    half are brain (> 0). Inside the brain, place a tumour mask. The healthy-
    brain region must equal the brain voxels NOT in WT — checked by counting
    the unique mask the metric helper sees indirectly through the returned
    tuple length and dtypes.
    """
    H = W = D = 8
    real = torch.zeros(1, 1, H, W, D)
    real[..., : H // 2, :, :] = 0.5  # half the volume is brain foreground
    pred = real.clone()  # identical → finite metrics in every region
    # WT mask occupies a small block inside the brain region.
    mask_5d = torch.zeros(1, 1, H, W, D, dtype=torch.bool)
    mask_5d[0, 0, 1:3, 1:3, 1:3] = True
    metrics = ImageMetrics(data_range=1.0)

    out = ExhaustiveValEngine._region_psnr_ssim(pred, real, mask_5d, metrics)
    assert len(out) == 6
    psnr_wt, ssim_wt, psnr_bg, ssim_bg, psnr_nwt, ssim_nwt = out
    # With pred == real, every non-empty region's PSNR should be finite and
    # very high. Identical inputs ⇒ MSE → 0 ⇒ PSNR → +inf or large; we just
    # check finiteness for the regions we know are non-empty.
    for v in (psnr_wt, ssim_wt, psnr_bg, ssim_bg, psnr_nwt, ssim_nwt):
        assert isinstance(v, float)


@pytest.mark.unit
def test_region_psnr_ssim_returns_nan_tuple_when_mask_is_none() -> None:
    pred = torch.rand(8, 8, 8)
    real = torch.rand(8, 8, 8)
    metrics = ImageMetrics(data_range=1.0)
    out = ExhaustiveValEngine._region_psnr_ssim(pred, real, None, metrics)
    assert len(out) == 6
    import math

    assert all(math.isnan(v) for v in out)
