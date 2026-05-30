"""Error-path tests for ``vena.data.registry.loader.load_registry``."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.registry import load_registry
from vena.data.registry.models import RegistryError

pytestmark = pytest.mark.unit


def _make_empty_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["placeholder"] = True
        f.create_dataset("dummy", data=np.zeros(1, dtype=np.float32))


def _valid_registry_dict(image_h5: Path, latent_h5: Path) -> dict:
    return {
        "schema_version": "1.0.0",
        "name": "test_corpus",
        "cohorts": [
            {
                "name": "TestCohort",
                "pathology": "preoperative_glioma",
                "label_system": "BraTS2021",
                "role": "cv",
                "longitudinal": False,
                "image_h5": str(image_h5),
                "latent_h5": str(latent_h5),
                "n_patients": 1,
                "n_scans": 1,
                "modalities": ["t1pre", "t1c"],
                "has_swan": False,
            }
        ],
    }


def test_load_registry_happy_path(tmp_path: Path) -> None:
    image_h5 = tmp_path / "image.h5"
    latent_h5 = tmp_path / "latent.h5"
    _make_empty_h5(image_h5)
    _make_empty_h5(latent_h5)
    reg_path = tmp_path / "corpus.json"
    reg_path.write_text(json.dumps(_valid_registry_dict(image_h5, latent_h5)))

    reg = load_registry(reg_path)
    assert reg.name == "test_corpus"
    assert len(reg.cohorts) == 1
    assert reg.cohorts[0].name == "TestCohort"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "does_not_exist.json")


def test_malformed_json_raises(tmp_path: Path) -> None:
    reg_path = tmp_path / "bad.json"
    reg_path.write_text("{not json at all")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_registry(reg_path)


def test_schema_violation_raises(tmp_path: Path) -> None:
    """Missing required keys → pydantic ValidationError wrapped as RegistryError."""
    reg_path = tmp_path / "schema_bad.json"
    reg_path.write_text(json.dumps({"schema_version": "1.0.0", "name": "x"}))  # no cohorts
    with pytest.raises(RegistryError, match="schema validation"):
        load_registry(reg_path)


def test_missing_image_h5_raises(tmp_path: Path) -> None:
    """Referenced image_h5 must exist on disk."""
    latent_h5 = tmp_path / "latent.h5"
    _make_empty_h5(latent_h5)
    reg_path = tmp_path / "corpus.json"
    reg_path.write_text(json.dumps(_valid_registry_dict(tmp_path / "absent_image.h5", latent_h5)))
    with pytest.raises(RegistryError, match="references missing H5"):
        load_registry(reg_path)


def test_missing_latent_h5_raises_when_required(tmp_path: Path) -> None:
    image_h5 = tmp_path / "image.h5"
    _make_empty_h5(image_h5)
    reg_path = tmp_path / "corpus.json"
    reg_path.write_text(json.dumps(_valid_registry_dict(image_h5, tmp_path / "absent_latent.h5")))
    with pytest.raises(RegistryError, match="references missing H5"):
        load_registry(reg_path, require_latents=True)


def test_missing_latent_h5_ok_when_not_required(tmp_path: Path) -> None:
    """The encoding routine sets require_latents=False; latent_h5 may be missing."""
    image_h5 = tmp_path / "image.h5"
    _make_empty_h5(image_h5)
    reg_path = tmp_path / "corpus.json"
    reg_path.write_text(json.dumps(_valid_registry_dict(image_h5, tmp_path / "absent_latent.h5")))
    reg = load_registry(reg_path, require_latents=False)
    assert reg.name == "test_corpus"
