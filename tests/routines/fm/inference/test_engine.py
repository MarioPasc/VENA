"""Integration test for the unified inference engine.

Runs the engine with only the C0-Identity adapter against a synthetic
2-cohort corpus registry (1 CV cohort + 1 test_only cohort, 1 patient
each), then asserts the on-disk layout matches the spec:

* ``<run_dir>/predictions/C0-Identity/<cohort>/nfe_001.h5`` per cohort
* ``<run_dir>/figures/<cohort>.png`` per cohort (figure enabled)
* ``<run_dir>/decision.json`` with the resolved registries + git_sha
* every H5 passes ``assert_predictions_valid``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


# Import the conftest fixtures by adding the inference tests dir.
@pytest.fixture
def two_cohort_corpus(tmp_path: Path):
    """Replicates ``tests/inference/conftest.py::two_cohort_registry``.

    Cross-test-tree fixture access is brittle; we re-implement the same
    payload here so this test module is self-contained.
    """
    import h5py
    import numpy as np

    def _write_image(path: Path, cohort: str, role: str, pid: str, shape) -> None:
        rng = np.random.default_rng(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = "2.0.0"
            f.attrs["cohort"] = cohort
            f.attrs["crop_box"] = json.dumps(list(shape))
            f.create_dataset(
                "ids", data=np.asarray([pid], dtype=object), dtype=h5py.string_dtype("utf-8")
            )
            for mod in ("t1pre", "t1c", "t2", "flair"):
                data = rng.uniform(0.0, 1000.0, size=(1, *shape)).astype(np.float32)
                f.create_dataset(
                    f"images/{mod}",
                    data=data,
                    chunks=(1, *shape),
                    compression="gzip",
                    compression_opts=4,
                )
            brain = np.zeros((1, *shape), dtype=np.int8)
            h, w, d = shape
            brain[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, d // 4 : 3 * d // 4] = 1
            f.create_dataset(
                "masks/brain",
                data=brain,
                chunks=(1, *shape),
                compression="gzip",
                compression_opts=4,
            )
            tumor = np.zeros((1, *shape), dtype=np.int8)
            tumor[:, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, d // 3 : 2 * d // 3] = 1
            f.create_dataset(
                "masks/tumor",
                data=tumor,
                chunks=(1, *shape),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("crop/origin", data=np.zeros((1, 3), dtype=np.int32))
            f.create_dataset(
                "patients/keys",
                data=np.asarray([pid], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            f.create_dataset("patients/offsets", data=np.asarray([0, 1], dtype=np.int32))
            if role == "cv":
                f.create_dataset(
                    "splits/test",
                    data=np.asarray([pid], dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )

    shape = (12, 12, 12)
    image_a = tmp_path / "A" / "image.h5"
    image_b = tmp_path / "B" / "image.h5"
    latent_a = tmp_path / "A" / "latent.h5"
    latent_b = tmp_path / "B" / "latent.h5"
    _write_image(image_a, "A", "cv", "A001", shape)
    _write_image(image_b, "B", "test_only", "B001", shape)
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


def test_engine_end_to_end_smoke(two_cohort_corpus, tmp_path: Path) -> None:
    from routines.fm.inference.engine import InferenceEngine, InferenceJobConfig

    from vena.inference.h5_writer import assert_predictions_valid

    registry_path, models_yaml_path = two_cohort_corpus
    output_root = tmp_path / "out"

    cfg_payload = {
        "run_id_tag": "engine_test",
        "output_root": str(output_root),
        "corpus_registry": str(registry_path),
        "models_yaml": str(models_yaml_path),
        "fold": 0,
        "device": "cpu",
        "warmup_passes": 0,
        "cohorts": {"cv_test": ["A"], "test_only": ["B"], "exclude": []},
        "methods": {"include": None, "exclude": []},
        "smoke": {"enabled": True, "n_patients_per_cohort": 1, "use_selection_nfe_only": True},
        "nfe_override": None,
        "figure": {"enabled": True, "n_slices": 5, "slice_offset": 1},
        "log_level": "WARNING",
    }
    cfg_path = tmp_path / "job.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_payload))

    cfg = InferenceJobConfig.from_yaml(cfg_path)
    engine = InferenceEngine(cfg)
    run_dir = engine.run()

    # 1. Predictions H5s exist per (cohort × NFE).
    h5_a = run_dir / "predictions" / "C0-Identity" / "A" / "nfe_001.h5"
    h5_b = run_dir / "predictions" / "C0-Identity" / "B" / "nfe_001.h5"
    assert h5_a.is_file(), "missing predictions H5 for cohort A"
    assert h5_b.is_file(), "missing predictions H5 for cohort B"
    assert_predictions_valid(h5_a)
    assert_predictions_valid(h5_b)

    # 2. Per-cohort comparison PNG exists.
    fig_a = run_dir / "figures" / "A.png"
    fig_b = run_dir / "figures" / "B.png"
    assert fig_a.is_file(), "missing figure for cohort A"
    assert fig_b.is_file(), "missing figure for cohort B"

    # 3. decision.json carries the resolved provenance.
    decision = json.loads((run_dir / "decision.json").read_text())
    assert decision["schema_version"] == "1.0"
    assert decision["run_id_tag"] == "engine_test"
    assert set(decision["cohorts"].keys()) == {"A", "B"}
    assert decision["cohorts"]["A"]["patient_ids"] == ["A001"]
    assert decision["cohorts"]["B"]["patient_ids"] == ["B001"]
    method_names = [m["name"] for m in decision["methods"]]
    assert method_names == ["C0-Identity"]
