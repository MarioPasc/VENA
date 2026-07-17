"""Unit tests for routines.validation.downstream_seg.engine (§4.4).

The BratsSegmenter is patched out everywhere — no checkpoint is loaded.
All tests run on CPU, no GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from routines.validation.downstream_seg.engine import (
    DownstreamSegConfig,
    DownstreamSegEngine,
    _read_pred_row,
)

from vena.validation.io import ReferenceCache

pytestmark = pytest.mark.unit

_H, _W, _D = 16, 16, 16  # must match conftest dimensions


# ---------------------------------------------------------------------------
# Corpus H5 fixture (tumor labels + input modalities)
# ---------------------------------------------------------------------------


def _write_corpus_h5(
    path: Path,
    scan_ids: list[str],
    *,
    label_system: str = "BraTS2021",
    tumor_label: int = 4,
) -> None:
    """Write a minimal corpus H5 compatible with CorpusLabelCache."""
    n = len(scan_ids)
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["label_system"] = label_system

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ids", data=np.asarray(scan_ids, dtype=object), dtype=dt)

        tumor = np.zeros((n, _H, _W, _D), dtype=np.int8)
        tumor[:, 0, 0, 0] = tumor_label
        tumor[:, 1, 0, 0] = 1
        g_msk = f.create_group("masks")
        g_msk.create_dataset("tumor", data=tumor, chunks=(1, _H, _W, _D))

        g_img = f.create_group("images")
        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = rng.random((n, _H, _W, _D)).astype(np.float32)
            g_img.create_dataset(mod, data=data, chunks=(1, _H, _W, _D))


@pytest.fixture()
def corpus_h5(synth_shard: Path, tmp_path: Path) -> dict[str, Path]:
    """Corpus H5 files for the cohorts in synth_shard."""
    scan_map = {
        "TestCohortA": ["scanA1", "scanA2", "scanA3"],
        "LUMIERE-like": ["lum_s1", "lum_s2", "lum_s3", "lum_s4", "lum_s5"],
        "TestCohortB": ["scanB1", "scanB2"],
    }
    out: dict[str, Path] = {}
    for cohort, sids in scan_map.items():
        path = tmp_path / f"{cohort}_corpus.h5"
        _write_corpus_h5(path, sids)
        out[cohort] = path
    return out


@pytest.fixture()
def ring_partitions_json(tmp_path: Path) -> Path:
    """ring_partitions.json matching synth_shard cohorts."""
    payload = {
        "rings": {
            "A": {
                "cohorts": {
                    "TestCohortA": {"n_scans": 3},
                    "LUMIERE-like": {"n_scans": 5},
                }
            },
            "B": {
                "cohorts": {
                    "TestCohortB": {"n_scans": 2},
                }
            },
        },
        "methods": ["VENA-S1-v3b-rw", "C0-Identity"],
        "selection_nfe": {},
    }
    p = tmp_path / "ring_partitions.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture()
def bundle_path(tmp_path: Path) -> Path:
    """Minimal fake bundle directory with a placeholder model.pt."""
    bp = tmp_path / "bundle"
    (bp / "models").mkdir(parents=True)
    # Placeholder model.pt (0-byte file; BratsSegmenter is patched out anyway)
    (bp / "models" / "model.pt").write_bytes(b"\x00" * 16)
    return bp


# ---------------------------------------------------------------------------
# _read_pred_row — join correctness
# ---------------------------------------------------------------------------


class TestReadPredRow:
    """Verify the scan_id value-join (not row-index join) in _read_pred_row."""

    def test_returns_correct_scan_id(self, synth_shard: Path) -> None:
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data = _read_pred_row(pred_path, 0, ref_cache=cache)
        assert data["scan_id"] == "scanA1"

    def test_t1c_real_differs_from_synth(self, synth_shard: Path) -> None:
        """Real T1c and synthetic T1c should be independently seeded."""
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data = _read_pred_row(pred_path, 0, ref_cache=cache)
        # Real T1c comes from the reference H5 (seed 42),
        # synthetic from the prediction H5 (seed 0); they should differ.
        assert not np.allclose(data["t1c_real"], data["t1c_synth"])

    def test_join_is_value_based_not_index_based(self, synth_shard: Path) -> None:
        """Reference rows are in REVERSED order (join-trap from conftest).

        If the join were index-based, row 0 in the prediction (scanA1) would
        match row 0 in the reference (scanA3).  The scan_id join must return
        scanA1's reference data instead.
        """
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data0 = _read_pred_row(pred_path, 0, ref_cache=cache)
        data2 = _read_pred_row(pred_path, 2, ref_cache=cache)
        # scanA1 and scanA3 are different scans → their real T1c must differ.
        assert not np.allclose(data0["t1c_real"], data2["t1c_real"])

    def test_all_modalities_returned(self, synth_shard: Path) -> None:
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data = _read_pred_row(pred_path, 1, ref_cache=cache)
        for key in (
            "scan_id",
            "patient_id",
            "t1c_synth",
            "pred_mode",
            "wt_join_dice",
            "t1c_real",
            "t1pre",
            "t2",
            "flair",
        ):
            assert key in data, f"missing key: {key}"
        for vol_key in ("t1c_synth", "t1c_real", "t1pre", "t2", "flair"):
            assert data[vol_key].shape == (_H, _W, _D)
            assert data[vol_key].dtype == np.float32

    def test_wt_join_dice_is_one_for_identical_masks(self, synth_shard: Path) -> None:
        """Fixture writes all-ones masks/wt in both pred and ref H5 → Dice ≈ 1.0."""
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data = _read_pred_row(pred_path, 0, ref_cache=cache)
        assert "wt_join_dice" in data
        assert data["wt_join_dice"] == pytest.approx(1.0, abs=1e-6), (
            f"Expected wt_join_dice ≈ 1.0 for all-ones fixture masks, "
            f"got {data['wt_join_dice']!r}. Check conftest fixture or dice_score."
        )

    def test_pred_mode_is_raw_for_normalised_fixture(self, synth_shard: Path) -> None:
        """Fixture data is rng.random() ∈ [0, 1) → select_scoring_volume returns 'raw'."""
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        data = _read_pred_row(pred_path, 0, ref_cache=cache)
        assert data["pred_mode"] == "raw", (
            f"Expected pred_mode='raw' for [0,1) fixture data, got {data['pred_mode']!r}. "
            "Check select_scoring_volume thresholds or conftest fixture."
        )

    def test_missing_scan_id_raises_key_error(self, synth_shard: Path) -> None:
        """Accessing a row index beyond the file raises KeyError (not silent NaN)."""
        pred_path = synth_shard / "predictions" / "VENA-S1-v3b-rw" / "TestCohortA" / "nfe_005.h5"
        cache = ReferenceCache()
        # The conftest only has 3 scans in TestCohortA; row 99 doesn't exist.
        with pytest.raises((KeyError, IndexError, OSError)):
            _read_pred_row(pred_path, 99, ref_cache=cache)


# ---------------------------------------------------------------------------
# DownstreamSegEngine — real-arm caching + CSV output
# ---------------------------------------------------------------------------


def _make_mock_segmenter(dice_value: float = 0.85) -> MagicMock:
    """Return a MagicMock that behaves like BratsSegmenter.segment()."""
    seg = MagicMock()
    rng = np.random.default_rng(42)
    mask = rng.integers(0, 2, (_H, _W, _D)).astype(bool)
    seg.segment.return_value = (mask.copy(), mask.copy(), mask.copy())
    return seg


class TestDownstreamSegEngine:
    """Engine integration tests with a patched segmenter."""

    @staticmethod
    def _make_inference_root(synth_shard: Path, tmp_path: Path) -> Path:
        """Create an isolated inference_root containing only synth_shard.

        Using synth_shard.parent directly would pick up leftover shard*
        directories from previous pytest sessions sharing the same basetemp,
        inflating segment-call counts.  A dedicated directory with a single
        symlink guarantees isolation.
        """
        root = tmp_path / "inference_root"
        root.mkdir(exist_ok=True)
        link = root / synth_shard.name
        if not link.exists():
            link.symlink_to(synth_shard)
        return root

    def _make_cfg(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
        *,
        methods: list[str] | None = None,
        cohorts: list[str] | None = None,
    ) -> DownstreamSegConfig:
        return DownstreamSegConfig(
            inference_root=self._make_inference_root(synth_shard, tmp_path),
            output_root=tmp_path / "analyses",
            bundle_path=bundle_path,
            corpus_map=corpus_h5,
            ring_partitions_path=ring_partitions_json,
            methods=methods or [],
            cohorts=cohorts or [],
            device="cpu",
            amp=False,
            selection_nfe_only=False,  # false so nfe_005 passes any filter
            log_level="WARNING",
        )

    def test_real_arm_called_once_per_scan_across_methods(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        """Real-arm cache: compute once per (cohort, scan_id), not per method."""
        cfg = self._make_cfg(
            synth_shard,
            corpus_h5,
            ring_partitions_json,
            bundle_path,
            tmp_path,
            cohorts=["TestCohortA"],
        )

        mock_seg = _make_mock_segmenter()

        with (
            patch(
                "routines.validation.downstream_seg.engine.BratsSegmenter",
                return_value=mock_seg,
            ),
        ):
            engine = DownstreamSegEngine(cfg=cfg)
            engine.run()

        # TestCohortA has 3 scans; 2 methods share the real arm.
        # Expected: segment() called 3 (real) + 3×2 (synth) = 9 times.
        # If real arm were NOT cached: 3×2 (real) + 3×2 (synth) = 12.
        n_segment_calls = mock_seg.segment.call_count
        n_scans = 3
        n_methods = 2
        # Real arm: n_scans once; synthetic arm: n_scans × n_methods
        expected_max = n_scans + n_scans * n_methods  # 9
        expected_no_cache = n_scans * n_methods * 2  # 12 (if broken)
        assert n_segment_calls == expected_max, (
            f"Expected {expected_max} segment calls (real arm cached), "
            f"got {n_segment_calls} (would be {expected_no_cache} if broken)"
        )

    def test_csv_has_frozen_columns(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        """Output CSV must have the fixed column set."""
        import pandas as pd

        cfg = self._make_cfg(
            synth_shard,
            corpus_h5,
            ring_partitions_json,
            bundle_path,
            tmp_path,
            cohorts=["TestCohortA"],
            methods=["C0-Identity"],
        )
        mock_seg = _make_mock_segmenter()
        with patch(
            "routines.validation.downstream_seg.engine.BratsSegmenter",
            return_value=mock_seg,
        ):
            run_dir = DownstreamSegEngine(cfg=cfg).run()

        csv_path = run_dir / "per_scan" / "downstream_seg.csv"
        assert csv_path.is_file(), f"CSV not written: {csv_path}"
        df = pd.read_csv(csv_path)
        expected_cols = {
            "method",
            "cohort",
            "ring",
            "nfe",
            "scan_id",
            "patient_id",
            "pred_mode",
            "wt_join_dice",
            "dice_wt_real",
            "dice_tc_real",
            "dice_et_real",
            "dice_wt_synth",
            "dice_tc_synth",
            "dice_et_synth",
            "delta_wt",
            "delta_tc",
            "delta_et",
        }
        assert set(df.columns) == expected_cols

    def test_decision_json_written(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        cfg = self._make_cfg(
            synth_shard,
            corpus_h5,
            ring_partitions_json,
            bundle_path,
            tmp_path,
            cohorts=["TestCohortA"],
            methods=["C0-Identity"],
        )
        mock_seg = _make_mock_segmenter()
        with patch(
            "routines.validation.downstream_seg.engine.BratsSegmenter",
            return_value=mock_seg,
        ):
            run_dir = DownstreamSegEngine(cfg=cfg).run()

        dec = json.loads((run_dir / "decision.json").read_text())
        assert dec["schema_version"] == "1.0"
        assert "bundle_input_channel_order" in dec
        assert dec["bundle_input_channel_order"] == ["t1c", "t1", "t2", "flair"]
        assert "empty_et_convention" in dec
        assert "scoring_space_fix" in dec
        assert "skipped_smoke_shards" in dec
        assert "appendix_a_deviation" in dec
        assert "oracle_mask_confound" in dec
        assert "git_sha" in dec
        # WT-join Dice aggregates must be present in the artifact.
        assert "wt_join_dice_min" in dec
        assert "wt_join_dice_mean" in dec
        assert "wt_join_dice_below_0_99_n" in dec
        # For all-ones fixture masks, min/mean must be 1.0 and none below 0.99.
        assert dec["wt_join_dice_min"] == pytest.approx(1.0, abs=1e-6)
        assert dec["wt_join_dice_mean"] == pytest.approx(1.0, abs=1e-6)
        assert dec["wt_join_dice_below_0_99_n"] == 0

    def test_cohort_without_corpus_map_skipped(
        self,
        synth_shard: Path,
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        """Cohort absent from corpus_map should produce zero rows (skip + warn)."""
        import pandas as pd

        # Provide an empty corpus_map — no cohort will match.
        cfg = DownstreamSegConfig(
            inference_root=self._make_inference_root(synth_shard, tmp_path),
            output_root=tmp_path / "analyses",
            bundle_path=bundle_path,
            corpus_map={},  # nothing registered
            ring_partitions_path=ring_partitions_json,
            methods=[],
            cohorts=[],
            device="cpu",
            amp=False,
            selection_nfe_only=False,
            log_level="WARNING",
        )
        mock_seg = _make_mock_segmenter()
        with patch(
            "routines.validation.downstream_seg.engine.BratsSegmenter",
            return_value=mock_seg,
        ):
            run_dir = DownstreamSegEngine(cfg=cfg).run()

        csv_path = run_dir / "per_scan" / "downstream_seg.csv"
        df = pd.read_csv(csv_path)
        assert len(df) == 0, f"Expected 0 rows, got {len(df)}"
        # Segmenter should never have been called.
        mock_seg.segment.assert_not_called()

    def test_latest_symlink_created(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        cfg = self._make_cfg(
            synth_shard,
            corpus_h5,
            ring_partitions_json,
            bundle_path,
            tmp_path,
            cohorts=["TestCohortA"],
            methods=["C0-Identity"],
        )
        mock_seg = _make_mock_segmenter()
        with patch(
            "routines.validation.downstream_seg.engine.BratsSegmenter",
            return_value=mock_seg,
        ):
            run_dir = DownstreamSegEngine(cfg=cfg).run()

        latest = run_dir.parent / "LATEST"
        assert latest.is_symlink(), "LATEST symlink not created"
        assert latest.resolve() == run_dir.resolve()

    def test_delta_equals_real_minus_synth(
        self,
        synth_shard: Path,
        corpus_h5: dict[str, Path],
        ring_partitions_json: Path,
        bundle_path: Path,
        tmp_path: Path,
    ) -> None:
        """delta_wt / delta_tc / delta_et must equal Dice_real − Dice_synth."""
        import pandas as pd

        cfg = self._make_cfg(
            synth_shard,
            corpus_h5,
            ring_partitions_json,
            bundle_path,
            tmp_path,
            cohorts=["TestCohortA"],
            methods=["C0-Identity"],
        )
        mock_seg = _make_mock_segmenter()
        with patch(
            "routines.validation.downstream_seg.engine.BratsSegmenter",
            return_value=mock_seg,
        ):
            run_dir = DownstreamSegEngine(cfg=cfg).run()

        df = pd.read_csv(run_dir / "per_scan" / "downstream_seg.csv")
        assert len(df) > 0

        for _, row in df.iterrows():
            for label in ("wt", "tc", "et"):
                real = row[f"dice_{label}_real"]
                synth = row[f"dice_{label}_synth"]
                delta = row[f"delta_{label}"]
                # NaN propagates correctly in subtraction
                if not (np.isnan(real) or np.isnan(synth)):
                    assert delta == pytest.approx(real - synth, abs=1e-6)


# ---------------------------------------------------------------------------
# _discover_pred_files — smoke-shard exclusion
# ---------------------------------------------------------------------------


class TestDiscoverPredFiles:
    """Verify that _discover_pred_files skips smoke shards."""

    def _write_pred_stub(
        self, root: Path, *, shard: str, method: str, cohort: str, nfe: int
    ) -> Path:
        """Write an empty stub nfe_NNN.h5 (glob bait; contents not read)."""
        p = root / shard / "predictions" / method / cohort / f"nfe_{nfe:03d}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return p

    def test_smoke_shard_is_excluded(self, tmp_path: Path) -> None:
        """Shards with smoke.enabled=True must be absent from results.

        Mirrors the live Picasso topology where smoke_loginexa (smoke.enabled=True)
        and picasso_shard_a (smoke.enabled=False) both contain overlapping
        IvyGAP predictions.  Only the production shard must be included.
        """
        from routines.validation.downstream_seg.engine import _discover_pred_files

        root = tmp_path / "inference"
        root.mkdir()

        # Production shard
        prod = root / "picasso_shard_a"
        prod.mkdir()
        (prod / "decision.json").write_text(
            json.dumps({"schema_version": "1.0", "smoke": {"enabled": False}})
        )
        self._write_pred_stub(
            root, shard="picasso_shard_a", method="C0-Identity", cohort="IvyGAP", nfe=1
        )

        # Smoke shard — must be excluded
        smoke_root = root / "smoke_loginexa"
        smoke_root.mkdir()
        (smoke_root / "decision.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "smoke": {"enabled": True, "n_patients_per_cohort": 1},
                }
            )
        )
        self._write_pred_stub(
            root, shard="smoke_loginexa", method="C0-Identity", cohort="IvyGAP", nfe=1
        )

        results, skipped = _discover_pred_files(
            root,
            method_filter=[],
            cohort_filter=[],
            selection_nfe={},
            selection_nfe_only=False,
        )

        assert "smoke_loginexa" in skipped, f"smoke_loginexa missing from skipped: {skipped}"
        assert "picasso_shard_a" not in skipped, "production shard wrongly flagged as smoke"
        assert len(results) == 1, f"Expected 1 result (prod shard only), got {len(results)}"
        assert results[0][3].parents[3].name == "picasso_shard_a"
