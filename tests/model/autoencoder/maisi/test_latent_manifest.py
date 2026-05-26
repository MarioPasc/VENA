"""Unit tests for the latent-domain H5 manifest."""

from __future__ import annotations

import pytest

from vena.data.h5.shared import H5Manifest
from vena.data.h5.ucsf_pdgm.latent_domain import (
    UCSF_PDGM_LATENT_DEFAULT_MODALITIES,
    UCSF_PDGM_LATENT_MANIFEST,
    UCSF_PDGM_LATENT_SCHEMA_VERSION,
    build_latent_manifest,
)


@pytest.mark.unit
def test_default_manifest_has_expected_datasets() -> None:
    m = UCSF_PDGM_LATENT_MANIFEST
    paths = {d.path for d in m.datasets}
    assert "ids" in paths
    assert "masks/tumor_latent" in paths
    for slug in UCSF_PDGM_LATENT_DEFAULT_MODALITIES:
        assert f"latents/{slug}" in paths
    assert "metadata/who_grade" in paths
    assert m.schema_version == UCSF_PDGM_LATENT_SCHEMA_VERSION
    assert m.cohort == "UCSF-PDGM"
    assert m.domain == "latent"


@pytest.mark.unit
def test_manifest_subset_modalities_round_trips() -> None:
    m = build_latent_manifest(modalities=["t1c", "flair"], mask_output_channels=3)
    paths = {d.path for d in m.datasets}
    assert "latents/t1c" in paths
    assert "latents/flair" in paths
    assert "latents/t1pre" not in paths


@pytest.mark.unit
def test_unknown_modality_rejected() -> None:
    with pytest.raises(ValueError):
        build_latent_manifest(modalities=["mri_unknown"], mask_output_channels=3)


@pytest.mark.unit
def test_manifest_json_roundtrips() -> None:
    s = UCSF_PDGM_LATENT_MANIFEST.to_json()
    parsed = H5Manifest.from_json(s)
    assert parsed.schema_version == UCSF_PDGM_LATENT_MANIFEST.schema_version
    assert parsed.cohort == UCSF_PDGM_LATENT_MANIFEST.cohort
    assert parsed.domain == UCSF_PDGM_LATENT_MANIFEST.domain
    assert len(parsed.datasets) == len(UCSF_PDGM_LATENT_MANIFEST.datasets)
