"""Synthetic image-H5 fixtures shared by the unified-inference tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def _vlen_str(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _write_image_h5(
    path: Path,
    *,
    cohort: str = "TEST",
    role: str = "cv",
    patient_ids: list[str],
    shape: tuple[int, int, int] = (16, 16, 16),
    crop_box: tuple[int, int, int] = (16, 16, 16),
    seed: int = 0,
) -> None:
    """Write a minimal but schema-valid image H5.

    The schema mirrors ``.claude/rules/h5-design-principles.md`` and the
    image-domain manifests under ``vena/data/h5/<cohort>/image_domain/`` — we
    only fill the fields the inference routine reads.
    """
    rng = np.random.default_rng(seed)
    n = len(patient_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = cohort
        f.attrs["crop_box"] = json.dumps(list(crop_box))

        f.create_dataset("ids", data=_vlen_str(patient_ids), dtype=h5py.string_dtype("utf-8"))

        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = rng.uniform(0.0, 1000.0, size=(n, *shape)).astype(np.float32)
            f.create_dataset(
                f"images/{mod}",
                data=data,
                chunks=(1, *shape),
                compression="gzip",
                compression_opts=4,
            )
        # Brain mask: central cube of ones to give percentile_normalise a
        # well-defined foreground.
        brain = np.zeros((n, *shape), dtype=np.int8)
        h, w, d = shape
        brain[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, d // 4 : 3 * d // 4] = 1
        f.create_dataset(
            "masks/brain", data=brain, chunks=(1, *shape), compression="gzip", compression_opts=4
        )
        # Tumour: smaller central cube
        tumor = np.zeros((n, *shape), dtype=np.int8)
        tumor[:, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, d // 3 : 2 * d // 3] = 1
        f.create_dataset(
            "masks/tumor", data=tumor, chunks=(1, *shape), compression="gzip", compression_opts=4
        )

        f.create_dataset("crop/origin", data=np.zeros((n, 3), dtype=np.int32))

        # CSR patient layout: 1 scan per patient.
        f.create_dataset(
            "patients/keys", data=_vlen_str(patient_ids), dtype=h5py.string_dtype("utf-8")
        )
        f.create_dataset("patients/offsets", data=np.arange(n + 1, dtype=np.int32))

        if role == "cv":
            # Splits: train/val/test = first/middle/last sample
            split_key = "splits/test"
            f.create_dataset(
                split_key, data=_vlen_str(patient_ids), dtype=h5py.string_dtype("utf-8")
            )


@pytest.fixture
def synthetic_cohort(tmp_path: Path):
    """A single-patient image H5 + a CohortEntry-shaped object.

    Returns a (cohort_entry, image_h5_path) tuple where ``cohort_entry`` is a
    Pydantic ``CohortEntry`` with the right role + paths so the engine and
    adapters can consume it directly.
    """
    from vena.data.registry import CohortEntry

    image_h5 = tmp_path / "cohort_image.h5"
    latent_h5 = tmp_path / "cohort_latent.h5"  # unused by image-tier tests
    patient_ids = ["P001"]
    _write_image_h5(image_h5, cohort="TEST", role="cv", patient_ids=patient_ids, shape=(16, 16, 16))
    # Latent H5 is a placeholder file so CohortEntry validation passes.
    latent_h5.write_bytes(b"\x00")  # not a real H5; adapters that need it will fail

    cohort = CohortEntry(
        name="TEST",
        pathology="glioma",
        label_system="BraTS2021",
        role="cv",
        longitudinal=False,
        image_h5=image_h5,
        latent_h5=latent_h5,
        n_patients=len(patient_ids),
        n_scans=len(patient_ids),
        modalities=["t1pre", "t1c", "t2", "flair"],
        has_swan=False,
    )
    return cohort, image_h5


@pytest.fixture
def two_cohort_registry(tmp_path: Path):
    """A 2-cohort corpus registry JSON + the matching image H5 fixtures.

    Returns ``(registry_path, models_yaml_path)`` — both written to disk; the
    models YAML lists only the identity adapter so the test does not need
    competitor checkpoints.
    """
    import yaml

    cohort_a_dir = tmp_path / "cohort_a"
    cohort_b_dir = tmp_path / "cohort_b"
    cohort_a_dir.mkdir()
    cohort_b_dir.mkdir()
    image_a = cohort_a_dir / "image.h5"
    image_b = cohort_b_dir / "image.h5"
    latent_a = cohort_a_dir / "latent.h5"
    latent_b = cohort_b_dir / "latent.h5"
    _write_image_h5(
        image_a, cohort="A", role="cv", patient_ids=["A001"], shape=(12, 12, 12), seed=1
    )
    _write_image_h5(
        image_b, cohort="B", role="test_only", patient_ids=["B001"], shape=(12, 12, 12), seed=2
    )
    # Latent H5 placeholders — only required to exist.
    latent_a.write_bytes(b"\x00")
    latent_b.write_bytes(b"\x00")

    registry_payload = {
        "schema_version": "1.0.0",
        "name": "test_corpus",
        "cohorts": [
            {
                "name": "A",
                "pathology": "glioma",
                "label_system": "BraTS2021",
                "role": "cv",
                "longitudinal": False,
                "image_h5": str(image_a),
                "latent_h5": str(latent_a),
                "n_patients": 1,
                "n_scans": 1,
                "modalities": ["t1pre", "t1c", "t2", "flair"],
                "has_swan": False,
            },
            {
                "name": "B",
                "pathology": "glioma",
                "label_system": "BraTS2021",
                "role": "test_only",
                "longitudinal": False,
                "image_h5": str(image_b),
                "latent_h5": str(latent_b),
                "n_patients": 1,
                "n_scans": 1,
                "modalities": ["t1pre", "t1c", "t2", "flair"],
                "has_swan": False,
            },
        ],
    }
    registry_path = tmp_path / "corpus.json"
    registry_path.write_text(json.dumps(registry_payload))

    models_payload = {
        "schema_version": "1.0",
        "methods": [
            {
                "name": "C0-Identity",
                "type": "identity",
                "kwargs": {"nfe_list": [1], "selection_nfe": 1, "device": "cpu"},
            }
        ],
    }
    models_yaml_path = tmp_path / "models.yaml"
    models_yaml_path.write_text(yaml.safe_dump(models_payload))

    return registry_path, models_yaml_path
