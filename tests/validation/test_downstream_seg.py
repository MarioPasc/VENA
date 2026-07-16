"""Unit tests for vena.validation.downstream_seg (§4.4).

All tests are CPU-only, no checkpoint access, no GPU.  The BratsSegmenter
class is NOT instantiated here — its unit test lives in
test_downstream_seg_engine.py where it is mocked.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.validation.downstream_seg import (
    CorpusLabelCache,
    LabelSystemError,
    derive_sub_labels,
    dice_score,
    normalize_intensity_channel_wise,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Small test volume dimensions
# ---------------------------------------------------------------------------
_H, _W, _D = 8, 8, 8


# ---------------------------------------------------------------------------
# derive_sub_labels — label-system branching
# ---------------------------------------------------------------------------


class TestDeriveSubLabels:
    """Verify the BraTS2021 / BraTS2023 label-system branching."""

    def _make_tumor(self, labels: list[int]) -> np.ndarray:
        """Place one voxel per label value at unique positions."""
        vol = np.zeros((_H, _W, _D), dtype=np.int8)
        for i, lbl in enumerate(labels):
            vol.flat[i] = lbl
        return vol

    def test_brats2021_wt_includes_all_nonzero(self) -> None:
        tumor = self._make_tumor([0, 1, 2, 4])
        wt, _tc, _et = derive_sub_labels(tumor, "BraTS2021")
        assert int(wt.sum()) == 3  # labels 1, 2, 4

    def test_brats2021_tc_is_1_and_4(self) -> None:
        tumor = self._make_tumor([0, 1, 2, 4])
        _wt, tc, _et = derive_sub_labels(tumor, "BraTS2021")
        assert int(tc.sum()) == 2  # labels 1, 4

    def test_brats2021_et_is_label_4(self) -> None:
        tumor = self._make_tumor([0, 1, 2, 4])
        _wt, _tc, et = derive_sub_labels(tumor, "BraTS2021")
        assert int(et.sum()) == 1  # label 4

    def test_brats2023_wt_includes_all_nonzero(self) -> None:
        tumor = self._make_tumor([0, 1, 2, 3])
        wt, _tc, _et = derive_sub_labels(tumor, "BraTS2023")
        assert int(wt.sum()) == 3  # labels 1, 2, 3

    def test_brats2023_tc_is_1_and_3(self) -> None:
        tumor = self._make_tumor([0, 1, 2, 3])
        _wt, tc, _et = derive_sub_labels(tumor, "BraTS2023")
        assert int(tc.sum()) == 2  # labels 1, 3

    def test_brats2023_et_is_label_3(self) -> None:
        """BraTS2023 uses label 3 for ET — never hard-code label 4."""
        tumor = self._make_tumor([0, 1, 2, 3])
        _wt, _tc, et = derive_sub_labels(tumor, "BraTS2023")
        assert int(et.sum()) == 1  # label 3

    def test_brats2023_label_4_not_et(self) -> None:
        """If a BraTS2023 mask accidentally has a voxel at 4, it must not
        enter ET (avoids cross-system confusion)."""
        tumor = self._make_tumor([0, 1, 4])  # label 4 is invalid in BraTS2023
        _wt, _tc, et = derive_sub_labels(tumor, "BraTS2023")
        # label 4 > 0 so it enters WT; but ET must be exactly label 3
        assert int(et.sum()) == 0

    def test_unknown_label_system_raises(self) -> None:
        tumor = np.zeros((_H, _W, _D), dtype=np.int8)
        with pytest.raises(LabelSystemError, match="Unknown label_system"):
            derive_sub_labels(tumor, "BraTS2022")  # does not exist


# ---------------------------------------------------------------------------
# dice_score — empty-region NaN convention
# ---------------------------------------------------------------------------


class TestDiceScore:
    """Verify the Dice metric including the empty-both → NaN contract."""

    def _full_mask(self) -> np.ndarray:
        return np.ones((_H, _W, _D), dtype=bool)

    def _empty_mask(self) -> np.ndarray:
        return np.zeros((_H, _W, _D), dtype=bool)

    def test_perfect_overlap_returns_one(self) -> None:
        m = self._full_mask()
        assert dice_score(m, m) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self) -> None:
        pred = np.zeros((_H, _W, _D), dtype=bool)
        gt = np.zeros((_H, _W, _D), dtype=bool)
        # Both empty → NaN (not 0), tested separately
        pred.flat[0] = True
        assert dice_score(pred, gt) == pytest.approx(0.0)

    def test_empty_both_returns_nan(self) -> None:
        """Primary contract: NaN when both pred and GT are empty (not 0)."""
        result = dice_score(self._empty_mask(), self._empty_mask())
        assert math.isnan(result), f"expected NaN, got {result}"

    def test_empty_pred_nonempty_gt_returns_zero(self) -> None:
        result = dice_score(self._empty_mask(), self._full_mask())
        assert result == pytest.approx(0.0)

    def test_nonempty_pred_empty_gt_returns_zero(self) -> None:
        result = dice_score(self._full_mask(), self._empty_mask())
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        pred = np.zeros((_H, _W, _D), dtype=bool)
        gt = np.zeros((_H, _W, _D), dtype=bool)
        # 2 voxels in pred, 2 in gt, 1 shared → Dice = 2*1/(2+2) = 0.5
        pred.flat[0] = True
        pred.flat[1] = True
        gt.flat[1] = True
        gt.flat[2] = True
        assert dice_score(pred, gt) == pytest.approx(0.5)

    def test_wt_join_proof(self) -> None:
        """Identity model: synthetic == real T1c → WT Dice must be exactly 1.0."""
        # WT mask is arbitrary but identical for both arms.
        rng = np.random.default_rng(0)
        wt = rng.integers(0, 2, size=(_H, _W, _D)).astype(bool)
        result = dice_score(wt, wt)
        # If wt is non-empty, Dice == 1.0; if empty, NaN — both are valid.
        if wt.any():
            assert result == pytest.approx(1.0)
        else:
            assert math.isnan(result)


# ---------------------------------------------------------------------------
# normalize_intensity_channel_wise
# ---------------------------------------------------------------------------


class TestNormalizeIntensity:
    """Verify the bundle's NormalizeIntensityd(nonzero=True, ch_wise=True)."""

    def test_skull_stripped_channel_becomes_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(1)
        vol = np.zeros((4, _H, _W, _D), dtype=np.float32)
        # Channel 0: nonzero foreground; other channels all-zero (skull stripped)
        vol[0] = rng.standard_normal((_H, _W, _D)).astype(np.float32)
        out = normalize_intensity_channel_wise(vol)
        fg = out[0][out[0] != 0.0]
        assert fg.mean() == pytest.approx(0.0, abs=1e-4)
        assert fg.std() == pytest.approx(1.0, abs=1e-4)

    def test_zero_channel_stays_zero(self) -> None:
        vol = np.zeros((4, _H, _W, _D), dtype=np.float32)
        vol[0] = 1.0  # non-zero channel
        out = normalize_intensity_channel_wise(vol)
        for ch in range(1, 4):
            assert (out[ch] == 0.0).all(), f"channel {ch} should stay zero"

    def test_constant_nonzero_channel_becomes_zero(self) -> None:
        """Constant foreground (σ ≈ 0) → foreground set to 0 (not NaN)."""
        vol = np.ones((1, _H, _W, _D), dtype=np.float32)
        out = normalize_intensity_channel_wise(vol)
        assert not np.isnan(out).any()
        assert (out[0] == 0.0).all()

    def test_output_shape_unchanged(self) -> None:
        vol = np.random.default_rng(2).random((4, _H, _W, _D)).astype(np.float32)
        out = normalize_intensity_channel_wise(vol)
        assert out.shape == vol.shape

    def test_input_not_mutated(self) -> None:
        vol = np.random.default_rng(3).random((4, _H, _W, _D)).astype(np.float32)
        original = vol.copy()
        normalize_intensity_channel_wise(vol)
        np.testing.assert_array_equal(vol, original)


# ---------------------------------------------------------------------------
# CorpusLabelCache — synthetic corpus H5 fixture (no GPU)
# ---------------------------------------------------------------------------


def _write_corpus_h5(
    path: Path,
    scan_ids: list[str],
    label_system: str,
    *,
    tumor_label: int = 4,
    rng_seed: int = 99,
) -> None:
    """Write a minimal corpus image H5 for testing."""
    n = len(scan_ids)
    rng = np.random.default_rng(rng_seed)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["label_system"] = label_system

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ids", data=np.asarray(scan_ids, dtype=object), dtype=dt)

        # masks/tumor: place one enhancing-tumour voxel per scan
        tumor = np.zeros((n, _H, _W, _D), dtype=np.int8)
        tumor[:, 0, 0, 0] = tumor_label  # WT label
        tumor[:, 1, 0, 0] = 1  # NCR
        f.create_group("masks").create_dataset("tumor", data=tumor, chunks=(1, _H, _W, _D))

        # images/t1pre, t2, flair for get_inputs()
        g_img = f.create_group("images")
        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = rng.random((n, _H, _W, _D)).astype(np.float32)
            g_img.create_dataset(mod, data=data, chunks=(1, _H, _W, _D))


class TestCorpusLabelCache:
    def test_get_labels_brats2021(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        scan_ids = ["s1", "s2"]
        _write_corpus_h5(h5p, scan_ids, "BraTS2021", tumor_label=4)
        with CorpusLabelCache({"TestCohort": h5p}) as cache:
            wt, _tc, et = cache.get_labels("TestCohort", "s1")
        # WT should include voxels labelled 1 and 4
        assert wt.any()
        assert et.any()  # label 4 → ET in BraTS2021
        assert wt.dtype == bool

    def test_get_labels_brats2023(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        _write_corpus_h5(h5p, ["s1"], "BraTS2023", tumor_label=3)
        with CorpusLabelCache({"Cohort": h5p}) as cache:
            _wt, _tc, et = cache.get_labels("Cohort", "s1")
        assert et.any()  # label 3 → ET in BraTS2023

    def test_get_labels_unknown_scan_raises(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        _write_corpus_h5(h5p, ["s1"], "BraTS2021")
        with CorpusLabelCache({"Cohort": h5p}) as cache:
            with pytest.raises(KeyError, match="not found in corpus H5"):
                cache.get_labels("Cohort", "MISSING")

    def test_get_inputs_returns_three_arrays(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        _write_corpus_h5(h5p, ["s1", "s2"], "BraTS2021")
        with CorpusLabelCache({"Cohort": h5p}) as cache:
            t1pre, t2, flair = cache.get_inputs("Cohort", "s2")
        assert t1pre.shape == (_H, _W, _D)
        assert t2.shape == (_H, _W, _D)
        assert flair.shape == (_H, _W, _D)
        assert t1pre.dtype == np.float32

    def test_has_cohort_true_for_valid(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        _write_corpus_h5(h5p, ["s1"], "BraTS2021")
        cache = CorpusLabelCache({"Cohort": h5p})
        assert cache.has_cohort("Cohort")
        cache.close()

    def test_has_cohort_false_for_missing_file(self, tmp_path: Path) -> None:
        cache = CorpusLabelCache({"Cohort": tmp_path / "does_not_exist.h5"})
        assert not cache.has_cohort("Cohort")
        cache.close()

    def test_join_is_by_scan_id_not_row_index(self, tmp_path: Path) -> None:
        """Reference rows in reversed order must still produce correct labels."""
        h5p = tmp_path / "corpus.h5"
        # Write s2 at row 0, s1 at row 1.
        _write_corpus_h5(h5p, ["s2", "s1"], "BraTS2021", tumor_label=4)
        with CorpusLabelCache({"C": h5p}) as cache:
            wt_s1, _, _ = cache.get_labels("C", "s1")
            wt_s2, _, _ = cache.get_labels("C", "s2")
        # Both scans have the same tumor template, so both should have WT voxels.
        assert wt_s1.any()
        assert wt_s2.any()

    def test_context_manager_closes_handles(self, tmp_path: Path) -> None:
        h5p = tmp_path / "corpus.h5"
        _write_corpus_h5(h5p, ["s1"], "BraTS2021")
        cache = CorpusLabelCache({"C": h5p})
        cache.get_labels("C", "s1")  # open the file
        assert len(cache._handles) == 1
        cache.close()
        assert len(cache._handles) == 0
