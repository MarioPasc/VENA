"""End-to-end smoke test of the UCSF-PDGM image-domain converter.

Reads the real source tree (skipped when unavailable in CI / non-dev hosts)
and converts a handful of patients into a temporary H5, then re-validates
the artifact and checks key invariants.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.shared import H5Manifest, assert_h5_valid
from vena.data.h5.ucsf_pdgm.image_domain import (
    UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
    UCSF_PDGM_IMAGE_MANIFEST,
    UCSFPDGMImageH5Config,
    UCSFPDGMImageH5Converter,
)

SOURCE_ROOT = Path("/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/source")
METADATA_CSV = Path("/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/metadata/UCSF-PDGM-metadata_v5.csv")


@pytest.mark.slow
@pytest.mark.skipif(not SOURCE_ROOT.exists(), reason="UCSF-PDGM source not mounted")
@pytest.mark.skipif(not METADATA_CSV.exists(), reason="UCSF-PDGM metadata CSV missing")
def test_smoke_convert(tmp_path: Path) -> None:
    out = tmp_path / "UCSFPDGM_image_smoke.h5"
    cfg = UCSFPDGMImageH5Config(
        source_root=SOURCE_ROOT,
        metadata_csv=METADATA_CSV,
        output_path=out,
        n_jobs=2,
        n_test=2,
        n_folds=3,
        seed=0,
        stratify_by=None,
        overwrite=True,
        limit=6,
        log_level="WARNING",
    )
    UCSFPDGMImageH5Converter(cfg).run()

    assert out.exists()
    assert_h5_valid(out, UCSF_PDGM_IMAGE_MANIFEST)

    with h5py.File(out, "r") as f:
        # Self-description: manifest_json round-trips.
        embedded = H5Manifest.from_json(str(f.attrs["manifest_json"]))
        assert embedded == UCSF_PDGM_IMAGE_MANIFEST

        # Shape contract on every image dataset.
        n = f["ids"].shape[0]
        assert n == 6
        for slug in ("t1pre", "t1c", "t2", "flair"):
            d = f[f"images/{slug}"]
            assert d.shape == (n, *UCSF_PDGM_IMAGE_EXPECTED_SHAPE)
            assert d.dtype == np.float32
            assert d.chunks == (1, *UCSF_PDGM_IMAGE_EXPECTED_SHAPE)

        # Tumour seg int8 + label set.
        seg = f["masks/tumor"][...]
        assert seg.dtype == np.int8
        unique = set(np.unique(seg).tolist())
        assert unique <= {0, 1, 2, 4}

        # Splits exist and are typed as strings (vlen-str on h5py becomes object on read).
        test_ids = f["splits/test"][...]
        assert len(test_ids) == 2
        train0 = f["splits/cv/fold_0/train"][...]
        val0 = f["splits/cv/fold_0/val"][...]
        assert len(train0) + len(val0) == n - 2
