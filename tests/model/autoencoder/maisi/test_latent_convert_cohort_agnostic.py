"""Pure-logic unit tests for the cohort-agnostic latent converter.

Covers:
- manifest builder for UCSF-like (with metadata) and BraTS-like (no metadata)
- CSR offsets/keys copy  (full-cohort path)
- splits copy            (full-cohort path)
- CSR/splits are SKIPPED for subset runs
- latent spatial constant is (48, 56, 48)

No MAISI checkpoint is loaded; the encoder is stubbed with a fixed-shape tensor.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from vena.data.h5.ucsf_pdgm.latent_domain.manifest import (
    UCSF_PDGM_LATENT_SPATIAL,
    _UCSF_PDGM_METADATA_FIELDS,
    build_latent_manifest,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal source image H5 fixtures
# ---------------------------------------------------------------------------

_NATIVE_SHAPE = (8, 8, 8)
_BOX = (8, 8, 8)  # same as native so crop is a no-op
_CROP_ORIGIN = (0, 0, 0)
_MODALITIES = ["t1pre", "t1c"]


def _make_source_h5(
    path: Path,
    n_scans: int = 4,
    n_patients: int = 2,
    has_metadata: bool = True,
    has_tumor_mask: bool = True,
) -> None:
    """Write a minimal schema-v2.0.0 image H5 for testing."""
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = "UCSF-PDGM" if has_metadata else "BraTS-GLI"
        f.attrs["crop_box"] = json.dumps(list(_BOX))
        f.attrs["orientation"] = "LPS"
        f.attrs["split_role"] = "train"
        f.attrs["longitudinal"] = False
        f.attrs["label_system"] = "BraTS2024"

        # ids
        ids = [f"patient_{i:03d}" for i in range(n_scans)]
        dt = h5py.special_dtype(vlen=str)
        ds_ids = f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=dt)
        ds_ids.attrs["description"] = "scan IDs"
        ds_ids.attrs["units"] = "dimensionless"
        ds_ids.attrs["dtype"] = "vlen-str"
        ds_ids.attrs["leading_dim"] = "n_scans"

        # images
        rng = np.random.default_rng(0)
        for slug in _MODALITIES:
            data = rng.random((n_scans, *_NATIVE_SHAPE), dtype=np.float32)
            f.create_dataset(f"images/{slug}", data=data)

        # tumor mask
        if has_tumor_mask:
            seg = np.zeros((n_scans, *_NATIVE_SHAPE), dtype=np.int8)
            f.create_dataset("masks/tumor", data=seg)

        # brain mask
        brain = np.ones((n_scans, *_NATIVE_SHAPE), dtype=np.int8)
        f.create_dataset("masks/brain", data=brain)

        # crop/origin
        origins = np.zeros((n_scans, 3), dtype=np.int32)
        f.create_dataset("crop/origin", data=origins)

        # patients CSR (n_patients patients, scans evenly distributed)
        scans_per_patient = n_scans // n_patients
        offsets = np.array(
            [i * scans_per_patient for i in range(n_patients)] + [n_scans],
            dtype=np.int32,
        )
        patient_keys = [f"pat_{i:03d}" for i in range(n_patients)]
        f.create_dataset("patients/offsets", data=offsets)
        pk_ds = f.create_dataset(
            "patients/keys",
            data=np.array(patient_keys, dtype=object),
            dtype=dt,
        )
        pk_ds.attrs["description"] = "patient keys"

        # splits
        f.create_dataset(
            "splits/test",
            data=np.array(patient_keys[:1], dtype=object),
            dtype=dt,
        )
        f.create_group("splits/cv")
        f.create_dataset(
            "splits/cv/fold_0/train",
            data=np.array(patient_keys[1:], dtype=object),
            dtype=dt,
        )
        f.create_dataset(
            "splits/cv/fold_0/val",
            data=np.array(patient_keys[:1], dtype=object),
            dtype=dt,
        )
        f["splits"].attrs["n_folds"] = 1
        f["splits"].attrs["description"] = "CV splits"

        # metadata (UCSF only)
        if has_metadata:
            f.create_dataset(
                "metadata/who_grade",
                data=np.ones(n_scans, dtype=np.int8) * 2,
            )
            meta_dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "metadata/sex",
                data=np.array(["M"] * n_scans, dtype=object),
                dtype=meta_dt,
            )

        # manifest_json attr so the converter can detect metadata fields
        if has_metadata:
            from vena.data.h5.ucsf_pdgm.latent_domain.manifest import _UCSF_PDGM_METADATA_FIELDS
            from vena.data.h5.ucsf_pdgm.latent_domain.manifest import build_latent_manifest
            from vena.data.h5.shared import DatasetSpec, H5Manifest

            # Build a minimal image-domain manifest (just the metadata fields)
            meta_specs = [
                DatasetSpec(
                    path=field["path"],
                    dtype=field["dtype"],  # type: ignore[arg-type]
                    kind="metadata",
                    units=field["units"],
                    description=field["description"],
                    leading_dim="n_scans",
                )
                for field in _UCSF_PDGM_METADATA_FIELDS
            ]
            img_manifest = H5Manifest(
                schema_version="2.0.0",
                cohort="UCSF-PDGM",
                domain="image",
                expected_shape=None,
                datasets=meta_specs,
            )
            f.attrs["manifest_json"] = img_manifest.to_json()
        else:
            # BraTS-GLI: no metadata fields in manifest
            from vena.data.h5.shared import H5Manifest

            brats_manifest = H5Manifest(
                schema_version="2.0.0",
                cohort="BraTS-GLI",
                domain="image",
                expected_shape=None,
                datasets=[],
            )
            f.attrs["manifest_json"] = brats_manifest.to_json()


# ---------------------------------------------------------------------------
# Manifest builder tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_latent_spatial_is_48_56_48() -> None:
    assert UCSF_PDGM_LATENT_SPATIAL == (48, 56, 48)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cohort,metadata_fields,expect_metadata",
    [
        ("UCSF-PDGM", _UCSF_PDGM_METADATA_FIELDS, True),
        ("BraTS-GLI", [], False),
        ("BraTS-GLI", None, False),
    ],
)
def test_manifest_metadata_fields(
    cohort: str,
    metadata_fields: list[dict[str, str]] | None,
    expect_metadata: bool,
) -> None:
    m = build_latent_manifest(
        modalities=["t1pre", "t1c"],
        mask_output_channels=3,
        cohort=cohort,
        metadata_fields=metadata_fields,
    )
    meta_paths = {d.path for d in m.datasets if d.path.startswith("metadata/")}
    if expect_metadata:
        assert len(meta_paths) > 0, "Expected metadata datasets for UCSF cohort"
        assert "metadata/who_grade" in meta_paths
    else:
        assert meta_paths == set(), f"Expected no metadata for {cohort}; got {meta_paths}"


@pytest.mark.unit
def test_manifest_always_has_csr_specs() -> None:
    """CSR specs must always be declared regardless of cohort."""
    for cohort, mf in [
        ("UCSF-PDGM", _UCSF_PDGM_METADATA_FIELDS),
        ("BraTS-GLI", []),
    ]:
        m = build_latent_manifest(
            modalities=["t1pre"],
            mask_output_channels=3,
            cohort=cohort,
            metadata_fields=mf,
        )
        paths = {d.path for d in m.datasets}
        assert "patients/offsets" in paths, f"patients/offsets missing for {cohort}"
        assert "patients/keys" in paths, f"patients/keys missing for {cohort}"


@pytest.mark.unit
def test_manifest_cohort_propagates() -> None:
    m = build_latent_manifest(
        modalities=["t1pre"],
        mask_output_channels=2,
        cohort="MyCustomCohort",
        metadata_fields=[],
    )
    assert m.cohort == "MyCustomCohort"


# ---------------------------------------------------------------------------
# CSR + splits copy logic (full-cohort vs subset)
# ---------------------------------------------------------------------------


def _make_fake_encoder(latent_channels: int = 4) -> Any:
    """Return a stubbed MaisiEncoder whose encode() returns a fixed latent."""
    from vena.model.autoencoder.maisi.preprocessing import CropPadSpec
    from vena.model.autoencoder.maisi.encode.engine import EncodeResult

    def _fake_encode(
        x: torch.Tensor,
        mode: str = "auto",
        crop_spec: CropPadSpec | None = None,
        normalise: bool = True,
    ) -> EncodeResult:
        # Return a (1, C, 48, 56, 48) latent regardless of input shape.
        z = torch.zeros(1, latent_channels, 48, 56, 48)
        return EncodeResult(latent=z, pad=None, inference_mode="full", crop=crop_spec)

    mock = MagicMock()
    mock.encode.side_effect = _fake_encode
    mock.handle.device = torch.device("cpu")
    mock.to_attrs.return_value = {}
    return mock


def _make_fake_downsampler() -> Any:
    """Return a stubbed AbstractMaskDownsampler."""
    mock = MagicMock()
    mock.output_channels = 3
    mock.to_attrs.return_value = {"name": "per_class_avg_pool"}

    def _fake_downsample(seg: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        return torch.zeros(1, 3, *target_shape)

    mock.downsample.side_effect = _fake_downsample
    return mock


@pytest.mark.unit
def test_full_cohort_copies_csr_and_splits(tmp_path: Path) -> None:
    """Full-cohort run: CSR + splits must be present in the output latent H5."""
    src_h5 = tmp_path / "src.h5"
    _make_source_h5(src_h5, n_scans=4, n_patients=2, has_metadata=True)

    out_h5 = tmp_path / "latents.h5"
    ckpt = tmp_path / "fake_ckpt.pt"
    ckpt.touch()

    from vena.data.h5.ucsf_pdgm.latent_domain.convert import (
        UCSFPDGMLatentH5Config,
        UCSFPDGMLatentH5Converter,
    )

    cfg = UCSFPDGMLatentH5Config(
        source_image_h5=src_h5,
        output_path=out_h5,
        autoencoder_checkpoint=ckpt,
        modalities=list(_MODALITIES),
        inference_mode="full",
        overwrite=False,
        resume=False,
        checkpoint_every=10,
        limit=None,
        patient_ids=None,
    )

    encoder = _make_fake_encoder()
    downsampler = _make_fake_downsampler()

    # Patch sha256_file + handle attrs so no real file hashing occurs
    with (
        patch(
            "vena.data.h5.latent_domain.convert.sha256_file",
            return_value="deadbeef",
        ),
        patch.object(encoder, "handle", create=True) as mock_handle,
    ):
        mock_handle.device = torch.device("cpu")
        mock_handle.checkpoint_sha256 = "abc123"
        mock_handle.arch_kwargs = {}
        encoder.handle = mock_handle

        converter = UCSFPDGMLatentH5Converter(
            cfg=cfg, encoder=encoder, mask_downsampler=downsampler
        )
        result = converter.run()

    assert result.is_file()
    with h5py.File(result, "r") as f:
        assert "patients/offsets" in f, "CSR offsets missing in full-cohort run"
        assert "patients/keys" in f, "CSR keys missing in full-cohort run"
        assert "splits/test" in f, "splits/test missing in full-cohort run"
        assert "splits/cv/fold_0/train" in f, "splits/cv/fold_0/train missing"
        offsets = np.asarray(f["patients/offsets"][:])
        assert offsets[0] == 0
        assert offsets[-1] == 4  # n_scans


@pytest.mark.unit
def test_subset_run_skips_csr_and_splits(tmp_path: Path) -> None:
    """Subset run (limit < n_all): CSR + splits must NOT be copied."""
    src_h5 = tmp_path / "src.h5"
    _make_source_h5(src_h5, n_scans=4, n_patients=2, has_metadata=False)

    out_h5 = tmp_path / "latents_subset.h5"
    ckpt = tmp_path / "fake_ckpt.pt"
    ckpt.touch()

    from vena.data.h5.ucsf_pdgm.latent_domain.convert import (
        UCSFPDGMLatentH5Config,
        UCSFPDGMLatentH5Converter,
    )

    cfg = UCSFPDGMLatentH5Config(
        source_image_h5=src_h5,
        output_path=out_h5,
        autoencoder_checkpoint=ckpt,
        modalities=list(_MODALITIES),
        inference_mode="full",
        overwrite=False,
        resume=False,
        checkpoint_every=10,
        limit=2,  # < n_all=4 → subset
        patient_ids=None,
    )

    encoder = _make_fake_encoder()
    downsampler = _make_fake_downsampler()

    with (
        patch(
            "vena.data.h5.latent_domain.convert.sha256_file",
            return_value="deadbeef",
        ),
        patch.object(encoder, "handle", create=True) as mock_handle,
    ):
        mock_handle.device = torch.device("cpu")
        mock_handle.checkpoint_sha256 = "abc123"
        mock_handle.arch_kwargs = {}
        encoder.handle = mock_handle

        converter = UCSFPDGMLatentH5Converter(
            cfg=cfg, encoder=encoder, mask_downsampler=downsampler
        )
        result = converter.run()

    assert result.is_file()
    with h5py.File(result, "r") as f:
        # Structural completeness: CSR paths must exist (empty placeholders).
        assert "patients/offsets" in f, "CSR offsets placeholder missing in subset run"
        assert "patients/keys" in f, "CSR keys placeholder missing in subset run"
        # Empty placeholder: zero-length arrays.
        assert len(f["patients/offsets"][:]) == 0, "Expected empty CSR offsets for subset run"
        assert len(f["patients/keys"][:]) == 0, "Expected empty CSR keys for subset run"
        # Splits are not copied for subset runs.
        assert "splits" not in f, "splits should not be copied in subset run"


@pytest.mark.unit
def test_brats_latent_manifest_has_no_metadata_datasets() -> None:
    """BraTS-style manifest (metadata_fields=[]) produces no metadata/* datasets."""
    m = build_latent_manifest(
        modalities=["t1pre", "t1c", "t2", "flair"],
        mask_output_channels=3,
        cohort="BraTS-GLI",
        metadata_fields=[],
    )
    meta_paths = [d.path for d in m.datasets if d.path.startswith("metadata/")]
    assert meta_paths == [], f"Expected no metadata datasets; got {meta_paths}"


@pytest.mark.unit
def test_latent_manifest_latent_datasets_have_correct_paths() -> None:
    modalities = ["t1pre", "t1c", "t2", "flair"]
    m = build_latent_manifest(
        modalities=modalities,
        mask_output_channels=3,
        cohort="UCSF-PDGM",
        metadata_fields=_UCSF_PDGM_METADATA_FIELDS,
    )
    paths = {d.path for d in m.datasets}
    for slug in modalities:
        assert f"latents/{slug}" in paths
    assert "masks/tumor_latent" in paths
    assert "ids" in paths
