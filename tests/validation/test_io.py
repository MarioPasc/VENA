"""Tests for vena.validation.io.

Focus: join correctness (by scan_id, not by row index), missing-scan
behaviour, and streaming memory contract.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# Synthetic H5 helpers
# ---------------------------------------------------------------------------

H, W, D = 16, 16, 16  # small shape for fast tests


def _vlen_str(grp: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _make_pred_h5(
    path: Path,
    scan_ids: list[str],
    patient_ids: list[str],
    *,
    method: str = "VENA-test",
    cohort: str = "TestCohort",
    nfe: int = 5,
    ring: str = "A",
    references_h5: str = "references/TestCohort.h5",
) -> None:
    """Write a minimal schema-2.0 prediction H5."""
    n = len(scan_ids)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["method"] = method
        f.attrs["cohort"] = cohort
        f.attrs["nfe"] = nfe
        f.attrs["ring"] = ring
        f.attrs["references_h5"] = references_h5

        g = f.create_group("predictions")
        data = np.random.default_rng(0).random((n, H, W, D), dtype=np.float32)
        g.create_dataset(
            "t1c_synthetic_harmonised",
            data=data,
            chunks=(1, H, W, D),
        )
        g.create_dataset(
            "t1c_synthetic_raw",
            data=data.copy(),
            chunks=(1, H, W, D),
        )

        g_msk = f.create_group("masks")
        masks_int = np.ones((n, H, W, D), dtype=np.int8)
        g_msk.create_dataset("brain", data=masks_int, chunks=(1, H, W, D))
        g_msk.create_dataset("wt", data=masks_int, chunks=(1, H, W, D))

        g_meta = f.create_group("metadata")
        _vlen_str(g_meta, "scan_id", scan_ids)
        _vlen_str(g_meta, "patient_id", patient_ids)
        _vlen_str(g_meta, "cohort", [cohort] * n)
        g_meta.create_dataset(
            "inference_seconds",
            data=np.ones(n, dtype=np.float32),
        )
        g_meta.create_dataset(
            "peak_vram_mb",
            data=np.full(n, 1000.0, dtype=np.float32),
        )
        g_meta.create_dataset("nfe", data=np.full(n, nfe, dtype=np.int32))


def _make_ref_h5(
    path: Path,
    scan_ids: list[str],
    patient_ids: list[str],
    *,
    cohort: str = "TestCohort",
) -> None:
    """Write a minimal schema-2.0 reference H5."""
    n = len(scan_ids)
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["cohort"] = cohort

        g_ref = f.create_group("reference")
        vol = rng.random((n, H, W, D), dtype=np.float32)
        for name in (
            "t1c_real_harmonised",
            "t1pre_harmonised",
            "t2_harmonised",
            "flair_harmonised",
        ):
            g_ref.create_dataset(name, data=vol.copy(), chunks=(1, H, W, D))

        g_msk = f.create_group("masks")
        masks_int = np.ones((n, H, W, D), dtype=np.int8)
        g_msk.create_dataset("brain", data=masks_int, chunks=(1, H, W, D))
        g_msk.create_dataset("wt", data=masks_int, chunks=(1, H, W, D))

        g_meta = f.create_group("metadata")
        _vlen_str(g_meta, "scan_id", scan_ids)
        _vlen_str(g_meta, "patient_id", patient_ids)
        _vlen_str(g_meta, "cohort", [cohort] * n)


# ---------------------------------------------------------------------------
# Tests — join by scan_id
# ---------------------------------------------------------------------------


def test_iter_scans_join_by_scan_id_not_row_order(tmp_path: Path) -> None:
    """The join must be by scan_id even when reference rows are shuffled.

    An index-join would pass a naive test but fail this one.
    """
    from vena.validation.io import ReferenceCache, iter_scans

    # 3 scans in prediction order: A, B, C
    pred_scan_ids = ["scan_A", "scan_B", "scan_C"]
    pred_patient_ids = ["pt_A", "pt_B", "pt_C"]

    # Reference stores them in REVERSED order: C, B, A
    ref_scan_ids = ["scan_C", "scan_B", "scan_A"]
    ref_patient_ids = ["pt_C", "pt_B", "pt_A"]

    shard = tmp_path / "shard"
    pred_dir = shard / "predictions" / "VENA-test" / "TestCohort"
    pred_dir.mkdir(parents=True)
    ref_dir = shard / "references"
    ref_dir.mkdir(parents=True)

    pred_path = pred_dir / "nfe_5.h5"
    ref_path = ref_dir / "TestCohort.h5"

    # Write unique deterministic values per scan so we can assert correct matching.
    n = 3
    with h5py.File(pred_path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["method"] = "VENA-test"
        f.attrs["cohort"] = "TestCohort"
        f.attrs["nfe"] = 5
        f.attrs["ring"] = "A"
        f.attrs["references_h5"] = "references/TestCohort.h5"

        g = f.create_group("predictions")
        # Each scan gets a unique fill value: 0.1, 0.2, 0.3
        data = np.stack([np.full((H, W, D), 0.1 * (i + 1), dtype=np.float32) for i in range(n)])
        g.create_dataset("t1c_synthetic_harmonised", data=data, chunks=(1, H, W, D))
        g.create_dataset("t1c_synthetic_raw", data=data, chunks=(1, H, W, D))
        g_msk = f.create_group("masks")
        m = np.ones((n, H, W, D), dtype=np.int8)
        g_msk.create_dataset("brain", data=m, chunks=(1, H, W, D))
        g_msk.create_dataset("wt", data=m, chunks=(1, H, W, D))
        g_meta = f.create_group("metadata")
        dt = h5py.string_dtype(encoding="utf-8")
        g_meta.create_dataset("scan_id", data=np.asarray(pred_scan_ids, dtype=object), dtype=dt)
        g_meta.create_dataset(
            "patient_id", data=np.asarray(pred_patient_ids, dtype=object), dtype=dt
        )
        g_meta.create_dataset("cohort", data=np.asarray(["TestCohort"] * n, dtype=object), dtype=dt)
        g_meta.create_dataset("inference_seconds", data=np.ones(n, dtype=np.float32))
        g_meta.create_dataset("peak_vram_mb", data=np.ones(n, dtype=np.float32))
        g_meta.create_dataset("nfe", data=np.full(n, 5, dtype=np.int32))

    with h5py.File(ref_path, "w") as f:
        f.attrs["schema_version"] = "2.0"
        f.attrs["cohort"] = "TestCohort"
        g_ref = f.create_group("reference")
        # Reference volumes: unique fill values 0.9, 0.8, 0.7 for C, B, A
        ref_data = np.stack([np.full((H, W, D), 0.9 - 0.1 * i, dtype=np.float32) for i in range(n)])
        g_ref.create_dataset("t1c_real_harmonised", data=ref_data, chunks=(1, H, W, D))
        for name in ("t1pre_harmonised", "t2_harmonised", "flair_harmonised"):
            g_ref.create_dataset(name, data=ref_data, chunks=(1, H, W, D))
        g_msk = f.create_group("masks")
        m = np.ones((n, H, W, D), dtype=np.int8)
        g_msk.create_dataset("brain", data=m, chunks=(1, H, W, D))
        g_msk.create_dataset("wt", data=m, chunks=(1, H, W, D))
        g_meta = f.create_group("metadata")
        dt = h5py.string_dtype(encoding="utf-8")
        g_meta.create_dataset("scan_id", data=np.asarray(ref_scan_ids, dtype=object), dtype=dt)
        g_meta.create_dataset(
            "patient_id", data=np.asarray(ref_patient_ids, dtype=object), dtype=dt
        )
        g_meta.create_dataset("cohort", data=np.asarray(["TestCohort"] * n, dtype=object), dtype=dt)

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache))

    assert len(samples) == 3

    # scan_A is row 0 in pred (pred val ≈ 0.1), row 2 in ref (real val ≈ 0.7).
    # If the join were by row index: scan_A would get real val ≈ 0.9 (row 0 in ref = scan_C).
    sid_to_sample = {s.scan_id: s for s in samples}

    assert "scan_A" in sid_to_sample
    assert "scan_B" in sid_to_sample
    assert "scan_C" in sid_to_sample

    # scan_A: pred fill = 0.1, real fill = 0.7 (row 2 in ref)
    assert abs(float(sid_to_sample["scan_A"].pred.flat[0]) - 0.1) < 1e-5
    assert abs(float(sid_to_sample["scan_A"].real.flat[0]) - 0.7) < 1e-5

    # scan_C: pred fill = 0.3, real fill = 0.9 (row 0 in ref)
    assert abs(float(sid_to_sample["scan_C"].pred.flat[0]) - 0.3) < 1e-5
    assert abs(float(sid_to_sample["scan_C"].real.flat[0]) - 0.9) < 1e-5


def test_iter_scans_missing_reference_warns_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A scan present in pred but absent from ref produces a WARNING, not an error."""
    import logging

    from vena.validation.io import ReferenceCache, iter_scans

    shard = tmp_path / "shard"
    pred_dir = shard / "predictions" / "VENA-test" / "TestCohort"
    pred_dir.mkdir(parents=True)
    ref_dir = shard / "references"
    ref_dir.mkdir(parents=True)

    pred_path = pred_dir / "nfe_5.h5"
    ref_path = ref_dir / "TestCohort.h5"

    # Prediction has 3 scans; reference has only 2 of them.
    _make_pred_h5(
        pred_path,
        ["A", "B", "C"],
        ["pA", "pB", "pC"],
    )
    _make_ref_h5(ref_path, ["A", "C"], ["pA", "pC"])  # B is missing from ref

    cache = ReferenceCache()
    with caplog.at_level(logging.WARNING, logger="vena.validation.io"):
        samples = list(iter_scans(pred_path, reference_cache=cache))

    assert len(samples) == 2
    scan_ids = {s.scan_id for s in samples}
    assert scan_ids == {"A", "C"}

    # A warning must have been emitted for the missing scan.
    assert any("B" in r.message for r in caplog.records)


def test_build_index_discovers_files(tmp_path: Path) -> None:
    """build_index returns one row per nfe_*.h5 file in production shards."""
    import json

    from vena.validation.io import build_index

    # tmp_path = inference root; shard = one production shard inside it.
    shard = tmp_path / "shard"
    shard.mkdir()
    # decision.json without "smoke" key → production shard.
    (shard / "decision.json").write_text(json.dumps({"schema_version": "1.0"}))

    for method in ("VENA-test", "C0-Identity"):
        for cohort in ("CohortA", "CohortB"):
            d = shard / "predictions" / method / cohort
            d.mkdir(parents=True)
            ref_dir = shard / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)
            _make_pred_h5(
                d / "nfe_1.h5",
                ["scan1"],
                ["p1"],
                method=method,
                cohort=cohort,
                nfe=1,
                references_h5=f"references/{cohort}.h5",
            )
            _make_ref_h5(ref_dir / f"{cohort}.h5", ["scan1"], ["p1"], cohort=cohort)

    index = build_index(tmp_path)
    assert len(index) == 4  # 2 methods × 2 cohorts
    assert set(index["method"]) == {"VENA-test", "C0-Identity"}
    assert set(index["cohort"]) == {"CohortA", "CohortB"}


def test_iter_scans_scan_ids_filter(tmp_path: Path) -> None:
    """scan_ids parameter filters the output to only the requested scans."""
    from vena.validation.io import ReferenceCache, iter_scans

    shard = tmp_path / "shard"
    pred_dir = shard / "predictions" / "VENA-test" / "TestCohort"
    pred_dir.mkdir(parents=True)
    (shard / "references").mkdir(parents=True)

    pred_path = pred_dir / "nfe_5.h5"
    ref_path = shard / "references" / "TestCohort.h5"

    _make_pred_h5(pred_path, ["A", "B", "C"], ["pA", "pB", "pC"])
    _make_ref_h5(ref_path, ["A", "B", "C"], ["pA", "pB", "pC"])

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache, scan_ids=["B"]))
    assert len(samples) == 1
    assert samples[0].scan_id == "B"


# ---------------------------------------------------------------------------
# Smoke-shard filtering tests
# ---------------------------------------------------------------------------


def test_discover_shards_excludes_smoke(tmp_path: Path) -> None:
    """discover_shards skips shards where smoke.enabled is true.

    Reproduces the Picasso smoke_loginexa scenario: an inference root with
    one production shard and one smoke shard.  Only the production shard
    must appear in ShardDiscovery.accepted.
    """
    import json

    from vena.validation.io import discover_shards

    # Production shard — no smoke key.
    prod = tmp_path / "picasso_shard_a_cheap"
    prod.mkdir()
    (prod / "decision.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id_tag": "picasso_shard_a_cheap"})
    )

    # Smoke shard — smoke.enabled: true.
    smoke = tmp_path / "smoke_loginexa"
    smoke.mkdir()
    (smoke / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id_tag": "smoke_loginexa",
                "smoke": {"enabled": True, "n_patients_per_cohort": 1},
            }
        )
    )

    discovery = discover_shards(tmp_path)

    assert len(discovery.accepted) == 1
    assert discovery.accepted[0].root == prod
    assert len(discovery.skipped_smoke) == 1
    assert "smoke_loginexa" in discovery.skipped_smoke


def test_discover_shards_missing_smoke_key_is_production(tmp_path: Path) -> None:
    """A shard with no 'smoke' key in decision.json is treated as production.

    Fail-open: older shards and BraTS-PED backfill shards without the
    smoke-flag convention must not be accidentally excluded.
    """
    import json

    from vena.validation.io import discover_shards

    shard = tmp_path / "picasso_ped_a"
    shard.mkdir()
    # decision.json has no "smoke" key at all.
    (shard / "decision.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id_tag": "picasso_ped_a"})
    )

    discovery = discover_shards(tmp_path)

    assert len(discovery.accepted) == 1
    assert discovery.accepted[0].root == shard
    assert len(discovery.skipped_smoke) == 0


def test_build_index_raises_on_duplicate_scan_id_across_shards(tmp_path: Path) -> None:
    """build_index raises ValueError when the same (method, cohort, nfe, scan_id)
    appears in two production shards.

    This catches the Picasso bug where a smoke shard that survived filtering
    (or an unexpected duplicate production shard) would silently score the
    same patient twice.
    """
    import json

    from vena.validation.io import build_index

    for shard_name in ("shard_alpha", "shard_beta"):
        shard = tmp_path / shard_name
        shard.mkdir()
        (shard / "decision.json").write_text(
            json.dumps({"schema_version": "1.0", "run_id_tag": shard_name})
        )
        pred_dir = shard / "predictions" / "VENA-test" / "TestCohort"
        pred_dir.mkdir(parents=True)
        ref_dir = shard / "references"
        ref_dir.mkdir()
        # Both shards contain the SAME scan_id ("dup_scan").
        _make_pred_h5(
            pred_dir / "nfe_5.h5",
            ["dup_scan"],
            ["pt1"],
            method="VENA-test",
            cohort="TestCohort",
            nfe=5,
            references_h5="references/TestCohort.h5",
        )
        _make_ref_h5(ref_dir / "TestCohort.h5", ["dup_scan"], ["pt1"], cohort="TestCohort")

    with pytest.raises(ValueError, match=r"Duplicate.*method.*cohort.*nfe.*scan_id"):
        build_index(tmp_path)


# ---------------------------------------------------------------------------
# select_scoring_volume tests
# ---------------------------------------------------------------------------


def test_select_scoring_volume_already_normalised() -> None:
    """Volume with p99.5 ≈ 0.8 (inside [0,1]) returns mode='raw'."""
    from vena.validation.io import select_scoring_volume

    rng = np.random.default_rng(0)
    brain = np.ones((10, 10, 10), dtype=bool)
    raw = rng.uniform(0.0, 0.8, (10, 10, 10)).astype(np.float32)
    harmonised = rng.uniform(0.0, 1.0, (10, 10, 10)).astype(np.float32)

    vol, mode = select_scoring_volume(raw, harmonised, brain)

    assert mode == "raw"
    assert vol is raw


def test_select_scoring_volume_scanner_units() -> None:
    """Volume with p99.5 ≈ 2000 (scanner units, e.g. C0-Identity) returns mode='harmonised'."""
    from vena.validation.io import select_scoring_volume

    brain = np.ones((10, 10, 10), dtype=bool)
    raw = np.full((10, 10, 10), 2000.0, dtype=np.float32)
    harmonised = np.random.default_rng(1).uniform(0.0, 1.0, (10, 10, 10)).astype(np.float32)

    vol, mode = select_scoring_volume(raw, harmonised, brain)

    assert mode == "harmonised"
    assert vol is harmonised


def test_select_scoring_volume_boundary_at_threshold() -> None:
    """Values on both sides of SCORING_P995_MAX verify the ≤ boundary.

    The threshold is inclusive (≤, not <).  p99.5 = 1.04 passes; 1.06 fails.
    """
    from vena.validation.io import SCORING_P995_MAX, select_scoring_volume

    brain = np.ones((10, 10, 10), dtype=bool)
    harmonised = np.zeros((10, 10, 10), dtype=np.float32)

    # Below threshold (1.04 < 1.05) → "raw"
    raw_below = np.full((10, 10, 10), 1.04, dtype=np.float32)
    _, mode_below = select_scoring_volume(raw_below, harmonised, brain)
    assert mode_below == "raw", f"p99.5 ≈ 1.04 should be 'raw' (threshold={SCORING_P995_MAX})"

    # Above threshold (1.06 > 1.05) → "harmonised"
    raw_above = np.full((10, 10, 10), 1.06, dtype=np.float32)
    _, mode_above = select_scoring_volume(raw_above, harmonised, brain)
    assert mode_above == "harmonised", (
        f"p99.5 ≈ 1.06 should be 'harmonised' (threshold={SCORING_P995_MAX})"
    )


def test_select_scoring_volume_negative_valued() -> None:
    """Volume with min < SCORING_MIN_FLOOR returns mode='harmonised'.

    Even if p99.5 is fine, a negative minimum means the raw volume is not
    in the normalised [0, 1] space and must be harmonised.
    """
    from vena.validation.io import SCORING_MIN_FLOOR, select_scoring_volume

    brain = np.ones((10, 10, 10), dtype=bool)
    raw = np.full((10, 10, 10), -0.1, dtype=np.float32)  # min = -0.1 < -0.05
    harmonised = np.zeros((10, 10, 10), dtype=np.float32)

    _, mode = select_scoring_volume(raw, harmonised, brain)

    assert mode == "harmonised", (
        f"min = -0.1 is below SCORING_MIN_FLOOR={SCORING_MIN_FLOOR}; must return 'harmonised'"
    )


def test_iter_scans_surfaces_pred_mode_and_raw_p995(pred_path: Path) -> None:
    """iter_scans yields ScanSample with pred_mode and raw_p995 populated.

    Synthetic data is in [0, 1] so pred_mode must be 'raw' and raw_p995
    must be in (0, 1].  Both pred_raw and pred_harmonised must be present.
    """
    from vena.validation.io import ReferenceCache, iter_scans

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache))

    assert len(samples) > 0
    for s in samples:
        assert s.pred_mode in ("raw", "harmonised"), f"Unexpected pred_mode={s.pred_mode!r}"
        assert np.isfinite(s.raw_p995), f"raw_p995 should be finite; got {s.raw_p995}"
        assert s.raw_p995 >= 0.0, f"raw_p995 should be non-negative; got {s.raw_p995}"
        assert s.pred is not None
        assert s.pred_raw is not None
        assert s.pred_harmonised is not None
        # Synthetic fixtures are in [0, 1] → pred should alias pred_raw.
        assert s.pred_mode == "raw", "Synthetic volumes in [0,1] should select 'raw' mode"
        assert np.array_equal(s.pred, s.pred_raw)


def _make_scan_sample() -> object:
    """A minimal ScanSample with distinct pred_raw / stored-real for norm tests."""
    from vena.validation.io import ScanSample

    shape = (4, 4, 4)
    ramp = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    brain = np.ones(shape, dtype=bool)
    return ScanSample(
        scan_id="SCAN-A",
        patient_id="PID-A",
        cohort="TestCohort",
        ring="A",
        method="VENA",
        nfe=5,
        pred=np.zeros(shape, np.float32),
        pred_raw=ramp,  # VAE-space raw synthetic
        pred_harmonised=np.full(shape, 0.5, np.float32),  # stored 99.5 field
        pred_mode="raw",
        raw_p995=0.9,
        real=np.zeros(shape, np.float32),  # stored t1c_real_harmonised (99.5)
        brain=brain,
        wt=np.zeros(shape, bool),
        inference_seconds=1.0,
        peak_vram_mb=1.0,
    )


def _make_image_h5(path: Path) -> None:
    """Image H5 with raw scanner-unit t1c (distinct from the stored harmonised real)."""
    ramp = np.linspace(0.0, 2000.0, 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("ids", data=np.array([b"SCAN-A"]))
        f.create_dataset("images/t1c", data=ramp)


def test_harmonise_sample_9995_renormalises_from_image_h5(tmp_path: Path) -> None:
    """99.95 path re-derives real from the raw image H5, not the stored 99.5 field."""
    from vena.common import ENCODER_PERCENTILE_UPPER
    from vena.validation.io import harmonise_sample_to_percentile

    assert ENCODER_PERCENTILE_UPPER == 99.95
    sample = _make_scan_sample()
    img = tmp_path / "img.h5"
    _make_image_h5(img)

    out = harmonise_sample_to_percentile(sample, {"TestCohort": img})  # default 99.95
    # real is re-derived from the image H5 ramp → non-trivial in [0,1], NOT the zeros
    # that were stored in sample.real.
    assert float(out.real.min()) >= 0.0 and float(out.real.max()) <= 1.0
    assert float(out.real.max()) == pytest.approx(1.0, abs=1e-4)
    assert not np.array_equal(out.real, sample.real)
    # pred re-normalised from pred_raw → [0,1].
    assert float(out.pred.min()) >= 0.0 and float(out.pred.max()) <= 1.0
    assert float(out.pred.max()) == pytest.approx(1.0, abs=1e-4)


def test_harmonise_sample_995_returns_stored_fields(tmp_path: Path) -> None:
    """99.5 path returns the stored harmonised fields unchanged (no image H5 needed)."""
    from vena.validation.io import harmonise_sample_to_percentile

    sample = _make_scan_sample()
    out = harmonise_sample_to_percentile(sample, {}, percentile_upper=99.5)
    assert np.array_equal(out.pred, sample.pred_harmonised)
    assert np.array_equal(out.real, sample.real)


def test_harmonise_sample_rejects_bad_percentile_and_missing_map(tmp_path: Path) -> None:
    """Unsupported percentile and a missing image-H5 entry both raise ValueError."""
    from vena.validation.io import harmonise_sample_to_percentile

    sample = _make_scan_sample()
    with pytest.raises(ValueError, match=r"must be 99\.5"):
        harmonise_sample_to_percentile(sample, {}, percentile_upper=95.0)
    with pytest.raises(ValueError, match="image_h5_map entry"):
        harmonise_sample_to_percentile(sample, {}, percentile_upper=99.95)
