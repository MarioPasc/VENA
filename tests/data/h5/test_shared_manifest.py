"""Manifest construction and JSON round-trip for the shared H5 layer."""

from __future__ import annotations

import pytest

from vena.data.h5.shared import DatasetSpec, H5Manifest
from vena.data.h5.shared.exceptions import H5SchemaError


def _example_manifest() -> H5Manifest:
    return H5Manifest(
        schema_version="1.0.0",
        cohort="TEST",
        domain="image",
        expected_shape=(2, 2, 2),
        datasets=[
            DatasetSpec(
                path="ids",
                dtype="vlen-str",
                kind="id",
                units="dimensionless",
                description="id",
                leading_dim="n_scans",
            ),
            DatasetSpec(
                path="images/x",
                dtype="float32",
                kind="image",
                units="au",
                description="x",
                leading_dim="n_scans",
            ),
        ],
        splits_spec={"a": "b"},
        extras={"k": "v"},
    )


@pytest.mark.unit
def test_manifest_round_trip() -> None:
    m = _example_manifest()
    s = m.to_json()
    m2 = H5Manifest.from_json(s)
    assert m == m2


@pytest.mark.unit
def test_dataset_spec_rejects_leading_slash() -> None:
    with pytest.raises(ValueError, match="no leading"):
        DatasetSpec(
            path="/images/x",
            dtype="float32",
            kind="image",
            units="au",
            description="x",
        )


@pytest.mark.unit
def test_duplicate_paths_rejected() -> None:
    spec = DatasetSpec(
        path="images/x",
        dtype="float32",
        kind="image",
        units="au",
        description="x",
        leading_dim="n_scans",
    )
    with pytest.raises(ValueError, match="duplicate"):
        H5Manifest(
            schema_version="1.0.0",
            cohort="T",
            domain="image",
            datasets=[spec, spec],
        )


@pytest.mark.unit
def test_get_raises_on_missing() -> None:
    m = _example_manifest()
    with pytest.raises(H5SchemaError):
        m.get("does/not/exist")
    assert m.get("ids").kind == "id"


@pytest.mark.unit
def test_by_kind_filter() -> None:
    m = _example_manifest()
    images = m.by_kind("image")
    assert len(images) == 1
    assert images[0].path == "images/x"


@pytest.mark.unit
def test_ucsf_pdgm_manifest_loads() -> None:
    """The cohort manifest itself must be importable and self-consistent."""
    from vena.data.h5.ucsf_pdgm.image_domain import UCSF_PDGM_IMAGE_MANIFEST

    assert UCSF_PDGM_IMAGE_MANIFEST.schema_version == "2.0.0"
    assert UCSF_PDGM_IMAGE_MANIFEST.expected_shape == (240, 240, 155)
    # Manifest round-trip
    assert H5Manifest.from_json(UCSF_PDGM_IMAGE_MANIFEST.to_json()) == UCSF_PDGM_IMAGE_MANIFEST
    # The four sequences plus schema-2.0.0 additions (brain mask, crop origin, CSR).
    paths = {d.path for d in UCSF_PDGM_IMAGE_MANIFEST.datasets}
    for required in (
        "images/t1pre",
        "images/t1c",
        "images/t2",
        "images/flair",
        "masks/tumor",
        "masks/brain",
        "crop/origin",
        "patients/offsets",
        "patients/keys",
    ):
        assert required in paths
