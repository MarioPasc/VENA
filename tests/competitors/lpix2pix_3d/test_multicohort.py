"""Tests for the multi-cohort 3D-Latent-Pix2Pix dataset (corpus_registry path).

Verifies the skip-with-WARNING contract for missing cohorts, empty splits,
and the longitudinal/flat-split fallbacks at the multi-cohort layer.

Citation: arXiv:1611.07004; arXiv:2509.24194.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vena.competitors.lpix2pix_3d.dataset import (
    DatasetError,
    MultiCohortPix2PixLatentDataset,
)

from .test_dataset import _make_synth_latent_h5  # type: ignore[no-redef]

pytestmark = pytest.mark.unit


def _make_registry(
    out: Path,
    cohorts: list[dict],
) -> Path:
    out.write_text(json.dumps({"schema_version": "2.0.0", "cohorts": cohorts}, indent=2))
    return out


def test_multicohort_concats_two_cohorts(tmp_path: Path) -> None:
    h5_a = _make_synth_latent_h5(tmp_path / "a.h5", n=4)
    h5_b = _make_synth_latent_h5(tmp_path / "b.h5", n=4)
    reg = _make_registry(
        tmp_path / "corpus.json",
        [
            {"name": "A", "role": "cv", "latent_h5": str(h5_a)},
            {"name": "B", "role": "cv", "latent_h5": str(h5_b)},
        ],
    )
    ds = MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")
    assert len(ds) == 4  # 2 train ids × 2 cohorts
    assert ds.cohort_names == ["A", "B"]


def test_multicohort_skips_missing_h5_with_warning(tmp_path: Path, caplog) -> None:
    h5_a = _make_synth_latent_h5(tmp_path / "a.h5", n=4)
    reg = _make_registry(
        tmp_path / "corpus.json",
        [
            {"name": "A", "role": "cv", "latent_h5": str(h5_a)},
            {"name": "GHOST", "role": "cv", "latent_h5": str(tmp_path / "nope.h5")},
        ],
    )
    with caplog.at_level("WARNING"):
        ds = MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")
    assert ds.cohort_names == ["A"]
    assert any("GHOST" in r.message for r in caplog.records)


def test_multicohort_skips_entry_without_latent_h5_field(tmp_path: Path) -> None:
    h5_a = _make_synth_latent_h5(tmp_path / "a.h5", n=4)
    reg = _make_registry(
        tmp_path / "corpus.json",
        [
            {"name": "A", "role": "cv", "latent_h5": str(h5_a)},
            {"name": "IMAGE-ONLY", "role": "cv"},  # no latent_h5
        ],
    )
    ds = MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")
    assert ds.cohort_names == ["A"]


def test_multicohort_role_filter_drops_test_only_cohorts(tmp_path: Path) -> None:
    h5_a = _make_synth_latent_h5(tmp_path / "a.h5", n=4)
    h5_b = _make_synth_latent_h5(tmp_path / "b.h5", n=4)
    reg = _make_registry(
        tmp_path / "corpus.json",
        [
            {"name": "A", "role": "cv", "latent_h5": str(h5_a)},
            {"name": "T", "role": "test_only", "latent_h5": str(h5_b)},
        ],
    )
    ds = MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")
    assert ds.cohort_names == ["A"]


def test_multicohort_raises_when_all_cohorts_dropped(tmp_path: Path) -> None:
    reg = _make_registry(
        tmp_path / "corpus.json",
        [{"name": "GHOST", "role": "cv", "latent_h5": str(tmp_path / "nope.h5")}],
    )
    with pytest.raises(DatasetError, match="no usable cohorts"):
        MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")


def test_multicohort_handles_mixed_split_schemas(tmp_path: Path) -> None:
    """One cohort uses k-fold, another uses flat splits — both must work."""
    h5_kfold = _make_synth_latent_h5(tmp_path / "kfold.h5", n=4)
    h5_flat = _make_synth_latent_h5(tmp_path / "flat.h5", n=4, flat_splits=True)
    reg = _make_registry(
        tmp_path / "corpus.json",
        [
            {"name": "KFOLD", "role": "cv", "latent_h5": str(h5_kfold)},
            {"name": "FLAT", "role": "cv", "latent_h5": str(h5_flat)},
        ],
    )
    ds = MultiCohortPix2PixLatentDataset(corpus_registry=reg, fold=0, phase="train")
    assert ds.cohort_names == ["KFOLD", "FLAT"]
    assert len(ds) == 4  # 2 train ids × 2 cohorts


def test_multicohort_path_overrides_redirect_per_cohort(tmp_path: Path) -> None:
    h5_real = _make_synth_latent_h5(tmp_path / "real.h5", n=4)
    bogus = tmp_path / "bogus.h5"  # does not exist
    reg = _make_registry(
        tmp_path / "corpus.json",
        [{"name": "A", "role": "cv", "latent_h5": str(bogus)}],
    )
    ds = MultiCohortPix2PixLatentDataset(
        corpus_registry=reg,
        fold=0,
        phase="train",
        path_overrides={"A": h5_real},
    )
    assert ds.cohort_names == ["A"]
