"""End-to-end smoke test of the IvyGAP image-domain converter.

Reads the real source tree (skipped when unavailable in CI / non-dev hosts)
and converts a handful of patients into a temporary H5, then re-validates
the artifact and checks key invariants.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.ivy_gap.image_domain import (
    IVY_GAP_IMAGE_EXPECTED_SHAPE,
    IVY_GAP_IMAGE_MANIFEST,
    IvyGAPImageH5Config,
    IvyGAPImageH5Converter,
)
from vena.data.h5.shared import H5Manifest, assert_h5_valid

SOURCE_ROOT = Path(
    "/media/mpascual/MeningD2/GLIOMA/IVYGAP/PKG - IvyGAP-Radiomics-SRI/"
    "IvyGAP-Radiomics/Multi-Institutional Paired Expert Segmentations SRI "
    "images-atlas-annotations"
)


@pytest.mark.slow
@pytest.mark.skipif(not SOURCE_ROOT.exists(), reason="IvyGAP source not mounted")
def test_smoke_convert(tmp_path: Path) -> None:
    out = tmp_path / "IvyGAP_image_smoke.h5"
    cfg = IvyGAPImageH5Config(
        source_root=SOURCE_ROOT,
        output_path=out,
        n_jobs=2,
        shard_size=4,
        n_val=1,
        n_test=1,
        seed=0,
        overwrite=True,
        limit=4,
        log_level="WARNING",
    )
    IvyGAPImageH5Converter(cfg).run()
    assert out.exists()
    assert_h5_valid(out, IVY_GAP_IMAGE_MANIFEST)

    with h5py.File(out, "r") as f:
        embedded = H5Manifest.from_json(str(f.attrs["manifest_json"]))
        assert embedded == IVY_GAP_IMAGE_MANIFEST

        n = f["ids"].shape[0]
        assert n == 4
        for slug in ("t1pre", "t1c", "t2", "flair"):
            d = f[f"images/{slug}"]
            assert d.shape == (n, *IVY_GAP_IMAGE_EXPECTED_SHAPE)
            assert d.dtype == np.float32
            assert d.chunks == (1, *IVY_GAP_IMAGE_EXPECTED_SHAPE)

        seg = f["masks/tumor"][...]
        assert seg.dtype == np.int8
        unique = set(np.unique(seg).tolist())
        assert unique <= {0, 1, 2, 4}

        # 1/1/2 train/val/test partition.
        train = list(f["splits/train"][...])
        val = list(f["splits/val"][...])
        test = list(f["splits/test"][...])
        assert len(train) + len(val) + len(test) == n

        # CSR trivial 1:1.
        offsets = f["patients/offsets"][...]
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == n
