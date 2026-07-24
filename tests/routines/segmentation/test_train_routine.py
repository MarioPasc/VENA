"""Tests for routines.segmentation.train (task 18).

Coverage
--------
1. Config round-trip: ``from_yaml`` parses every shipped YAML without error.
2. CLI contract: ``cli.py`` accepts exactly one positional argument.
3. Import-side-effect guard: importing the engine triggers no CUDA init,
   no checkpoint load.
4. ``decision.json`` schema: after a stubbed run, all required keys present
   with correct types; Ring-B cohorts appear; **no temperature keys**.
5. ``temperatures.json`` absence: never written.
6. Smoke end-to-end (``slow``): < 5 min wall-clock; artifacts exist and are
   parseable on readback.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.segmentation

# Repo root: tests/routines/segmentation/<this file> -> up three levels.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Paths to all shipped YAML configs
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parents[3]  # VENA/
_TRAIN_CONFIGS = _REPO / "routines/segmentation/train/configs"

_ALL_YAMLS = [
    _TRAIN_CONFIGS / "default.yaml",
    _TRAIN_CONFIGS / "smoke.yaml",
    _TRAIN_CONFIGS / "smoke/loginexa_seg_ukb_fold0.yaml",
    _TRAIN_CONFIGS / "runs/picasso_seg_ukb.yaml",
    _TRAIN_CONFIGS / "runs/picasso_seg_brats.yaml",
    _TRAIN_CONFIGS / "runs/picasso_seg_segresnet.yaml",
]

# ---------------------------------------------------------------------------
# Ring-B cohort names (must appear in decision.json)
# ---------------------------------------------------------------------------

_RING_B_NAMES = {"BraTS-Africa-Glioma", "BraTS-Africa-Other", "BraTS-PED"}

# ---------------------------------------------------------------------------
# 1. Config round-trip: all shipped YAMLs parse without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_path", _ALL_YAMLS, ids=[p.name for p in _ALL_YAMLS])
def test_config_round_trip(yaml_path: Path) -> None:
    """Each shipped YAML parses via SegmentationConfig.from_yaml without error.

    An unknown key causes Pydantic to raise ``ValidationError`` (extra='forbid'),
    catching typos in production configs before a 3-day queue wait.
    """
    from vena.segmentation.config import SegmentationConfig

    cfg = SegmentationConfig.from_yaml(yaml_path)
    assert cfg.model.name in {"bsf_swinunetr_ukb", "bsf_swinunetr_brats", "segresnet"}
    # Print for verification
    fold_val = cfg.run.fold if cfg.run is not None else "N/A"
    print(f"{yaml_path.relative_to(_REPO)} OK  model={cfg.model.name}  fold={fold_val}")


# ---------------------------------------------------------------------------
# 2. CLI contract: exactly one positional argument
# ---------------------------------------------------------------------------


def test_cli_accepts_one_positional_arg() -> None:
    """cli.py sys.argv must have exactly 2 elements (script + yaml path).

    Asserts the script exits with an error when argc != 2 — without actually
    running training.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "routines.segmentation.train.cli"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_REPO / "src") + ":" + str(_REPO)},
    )
    # Should print usage to stderr and exit non-zero
    assert result.returncode != 0
    assert "Usage" in result.stderr or "usage" in result.stderr or "positional" in result.stderr


def test_cli_module_main_exits_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing cli and calling main() with no argv exits non-zero."""
    monkeypatch.setattr(sys, "argv", ["cli.py"])  # no yaml arg

    from routines.segmentation.train import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# 3. Import-side-effect guard: no CUDA init, no checkpoint load
# ---------------------------------------------------------------------------


def test_engine_import_no_cuda() -> None:
    """Importing the engine module must not initialise CUDA.

    Exercises the ``preflight-pattern.md`` rule "No heavy work at import time".

    The check runs in a **fresh subprocess**.  ``torch.cuda.is_initialized()`` is
    process-global, so asserting it in-process only reports whether *anything*
    earlier in the session touched CUDA — the test then passes or fails on test
    ordering rather than on this module's behaviour, which is exactly the kind of
    result that looks like a real signal and is not one.
    """
    import subprocess
    import sys

    probe = (
        "import torch;"
        "import importlib;"
        "importlib.import_module('routines.segmentation.train.engine');"
        "importlib.import_module('routines.segmentation.train.engine.train_engine');"
        "print('INITIALISED' if torch.cuda.is_initialized() else 'CLEAN')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, f"probe failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert proc.stdout.strip().endswith("CLEAN"), (
        f"Importing the engine module initialised CUDA. probe stdout={proc.stdout!r}"
    )


# ---------------------------------------------------------------------------
# 4 & 5. decision.json schema + temperatures.json absence
# ---------------------------------------------------------------------------


def _make_fake_fit_result(run_dir: Path) -> Any:
    """Build a minimal FitResult-like object for stubbing."""
    # Create a dummy checkpoint file
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    ckpt_path.write_bytes(b"FAKE_CKPT")

    @dataclass
    class _FakeFitResult:
        run_dir: Path
        checkpoint: Path
        best_epoch: int = 0
        best_score: float = 0.5
        initial_train_loss: float = 1.0
        final_train_loss: float = 0.8
        history: tuple = ()

    return _FakeFitResult(run_dir=run_dir, checkpoint=ckpt_path)


def _make_mock_resolution(ring_b_names: set[str]) -> Any:
    """Build a mock FmSplitResolution that includes Ring-B cohorts."""

    @dataclass
    class _MockCohortSplit:
        name: str
        role: str
        image_h5: Path = Path("/fake/h5")
        train_patients: tuple = ()
        val_patients: tuple = ()
        test_patients: tuple = ()
        n_patients_h5: int = 0
        n_kept_after_dedup: int = 0

    @dataclass
    class _MockResolution:
        per_cohort: tuple
        patient_to_cohort: dict
        patient_to_scans: dict
        fm_fold: int = 0
        corpus_registry: Path = Path("/fake/registry.json")
        corpus_registry_sha256: str = "deadbeef"
        dedup_decision_path: Any = None

        def fm_splits(self) -> dict:
            return {"train": [], "val": [], "test": []}

    cv_cohorts = [
        _MockCohortSplit(name="UCSF-PDGM", role="cv"),
        _MockCohortSplit(name="BraTS-GLI", role="cv"),
    ]
    ring_b = [_MockCohortSplit(name=n, role="test_only") for n in sorted(ring_b_names)]
    return _MockResolution(
        per_cohort=tuple(cv_cohorts + ring_b),
        patient_to_cohort={},
        patient_to_scans={},
    )


def _make_mock_plan() -> Any:
    """Build a minimal FoldPlan-like object."""

    @dataclass
    class _MockPlan:
        k: int = 3
        fm_train_ids: tuple = ()
        folds: tuple = ((), (), ())
        fm_val_ids: tuple = ()
        fm_test_ids: tuple = ()

        def to_dict(self) -> dict:
            return {
                "k": self.k,
                "fm_train_ids": list(self.fm_train_ids),
                "folds": [list(f) for f in self.folds],
                "fm_val_ids": list(self.fm_val_ids),
                "fm_test_ids": list(self.fm_test_ids),
            }

    return _MockPlan()


@pytest.fixture()
def stubbed_run_dir(tmp_path: Path) -> Path:
    """Run the engine with all external calls stubbed; return the run_dir."""
    from vena.segmentation.config import SegmentationConfig

    smoke_yaml = _TRAIN_CONFIGS / "smoke.yaml"
    cfg = SegmentationConfig.from_yaml(smoke_yaml)

    # Patch run.experiments_root to tmp_path

    raw = cfg.model_dump()
    raw["run"]["experiments_root"] = str(tmp_path)
    raw["run"]["tag"] = "test_stub"
    raw["run"]["fold"] = 0

    cfg_patched = SegmentationConfig.model_validate(raw)

    run_dir_holder: list[Path] = []

    def _fake_resolve_fm_splits(data_cfg: Any) -> Any:
        return _make_mock_resolution(_RING_B_NAMES)

    def _fake_build_fold_plan(
        data_cfg: Any, fm_splits: Any, *, dedup_duplicates: Any = None, cohort_labels: Any = None
    ) -> Any:
        return _make_mock_plan()

    def _fake_write_splits_json(path: Path, res: Any, plan: Any, *, extra: Any = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_fit(self_trainer: Any) -> Any:
        # Create run_dir structure (trainer would do this)
        rd = self_trainer._run_dir
        rd.mkdir(parents=True, exist_ok=True)
        result = _make_fake_fit_result(rd)
        run_dir_holder.append(rd)
        return result

    def _fake_evaluate(
        *, cfg: Any, result: Any, resolution: Any, plan: Any
    ) -> dict[str, dict[str, float]]:
        """Return per-cohort metrics including Ring-B cohorts."""
        out: dict[str, dict[str, float]] = {}
        for cs in resolution.per_cohort:
            out[cs.name] = {
                "tc": 0.72,
                "netc": 0.51,
                "tc_ahd": 5.0,
                "netc_ahd": 6.0,
                "tc_ece": 0.05,
                "netc_ece": 0.06,
                "tc_brier": 0.10,
                "netc_brier": 0.12,
                "et_dice": 0.65,
                "mean_et_soft": 0.45,
            }
        return out

    with (
        patch(
            # resolve_fm_splits is imported inside _run_inner → patch source module
            "vena.segmentation.data.fm_splits.resolve_fm_splits",
            side_effect=_fake_resolve_fm_splits,
        ),
        patch(
            "vena.segmentation.data.kfold.build_fold_plan",
            side_effect=_fake_build_fold_plan,
        ),
        patch(
            "vena.segmentation.data.fm_splits.write_splits_json",
            side_effect=_fake_write_splits_json,
        ),
        patch(
            "vena.segmentation.engine.train.SegTrainer.fit",
            new=_fake_fit,
        ),
        patch(
            # _evaluate_per_cohort is a module-level function → patch here
            "routines.segmentation.train.engine.train_engine._evaluate_per_cohort",
            side_effect=_fake_evaluate,
        ),
    ):
        from routines.segmentation.train.engine import SegTrainEngine

        engine = SegTrainEngine(cfg_patched)
        returned_run_dir = engine.run()

    return Path(returned_run_dir)


def test_decision_json_schema(stubbed_run_dir: Path) -> None:
    """decision.json must have all required keys with correct types.

    Assertions:
    - Required top-level keys present.
    - Ring-B cohorts appear in gseg.per_cohort.
    - No temperature-related keys anywhere in the document.
    """
    decision_path = stubbed_run_dir / "decision.json"
    assert decision_path.exists(), f"decision.json not found at {decision_path}"

    with decision_path.open() as fh:
        payload = json.load(fh)

    # Required top-level keys
    required_keys = [
        "schema_version",
        "produced_at",
        "producer",
        "run_id",
        "run_dir",
        "backbone_arm",
        "model_checkpoint",
        "model_checkpoint_sha256",
        "encoder_load_coverage",
        "fold",
        "k_folds",
        "fold_seed",
        "fm_fold",
        "seed",
        "corpus_registry",
        "corpus_registry_sha256",
        "dedup_decision_path",
        "ckpt_sha256",
        "selection_metric",
        "best_epoch",
        "best_score",
        "tumor_region",
        "gseg",
        "git_sha",
        "config_json",
    ]
    for key in required_keys:
        assert key in payload, f"Required key '{key}' missing from decision.json"

    # Type checks
    assert isinstance(payload["schema_version"], str)
    assert isinstance(payload["fold"], (int, str))
    assert isinstance(payload["best_epoch"], int)
    assert isinstance(payload["best_score"], (int, float))
    assert isinstance(payload["gseg"], dict)
    assert isinstance(payload["gseg"]["passed"], bool)
    assert isinstance(payload["gseg"]["per_cohort"], dict)

    # Ring-B cohorts must appear
    per_cohort = payload["gseg"]["per_cohort"]
    for ring_b in _RING_B_NAMES:
        assert ring_b in per_cohort, (
            f"Ring-B cohort '{ring_b}' missing from gseg.per_cohort. "
            f"Found: {sorted(per_cohort.keys())}"
        )
        row = per_cohort[ring_b]
        assert row["role"] == "test_only", f"{ring_b} role must be 'test_only'"

    # Per-cohort rows must have expected keys (including status/n_evaluated
    # added in fix-2 so null-metric rows are distinguishable from zero-scoring)
    required_cohort_keys = {
        "role",
        "n",
        "status",
        "n_evaluated",
        "tc_dice",
        "netc_dice",
        "tc_ahd",
        "netc_ahd",
        "tc_ece",
        "netc_ece",
        "tc_brier",
        "netc_brier",
        "et_dice",
        "mean_et_soft",
    }
    for cohort_name, row in per_cohort.items():
        missing = required_cohort_keys - set(row.keys())
        assert not missing, f"Cohort '{cohort_name}' missing keys: {missing}"

    # No temperature-related keys in the top-level payload or gseg section.
    # Note: config_json is an embedded config dump and legitimately contains
    # "derivation.temperature: none"; only the *decision* keys are checked.
    gseg_str = json.dumps(payload["gseg"])
    top_keys_str = json.dumps({k: v for k, v in payload.items() if k != "config_json"})
    for forbidden in ("T_TC", "T_NETC"):
        assert forbidden not in top_keys_str, (
            f"Forbidden temperature key '{forbidden}' found in decision.json top-level. "
            "Temperature scaling was dropped in iter-9 Q5."
        )
        assert forbidden not in gseg_str, (
            f"Forbidden temperature key '{forbidden}' found in decision.json gseg section. "
            "Temperature scaling was dropped in iter-9 Q5."
        )
    # Top-level decision.json must not have a standalone "temperature" key
    assert "temperature" not in payload, (
        "decision.json must not have a top-level 'temperature' key. "
        "Temperature scaling was dropped in iter-9 Q5."
    )
    # gseg section must not have temperature keys either
    assert "temperature" not in gseg_str or "T_TC" not in gseg_str, (
        "decision.json gseg section must not contain temperature keys."
    )


def test_no_temperatures_json(stubbed_run_dir: Path) -> None:
    """temperatures.json must NOT be written — iter-9 decision Q5."""
    temps_path = stubbed_run_dir / "temperatures.json"
    assert not temps_path.exists(), (
        "temperatures.json was found but must not be created "
        "(temperature scaling dropped in iter-9 Q5)."
    )


# ---------------------------------------------------------------------------
# 6. Smoke end-to-end (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_end_to_end(tmp_path: Path) -> None:
    """Smoke: 2-epoch segresnet on synthetic data, < 5 min, artifacts readable.

    Uses the smoke.yaml but redirects experiments_root to tmp_path.
    Patches resolve_fm_splits and SegTrainer with a synthetic 4-patient stub
    so no real H5 files or checkpoints are needed.
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    from vena.segmentation.config import SegmentationConfig

    smoke_yaml = _TRAIN_CONFIGS / "smoke.yaml"
    cfg_base = SegmentationConfig.from_yaml(smoke_yaml)

    raw = cfg_base.model_dump()
    raw["run"]["experiments_root"] = str(tmp_path)
    raw["run"]["tag"] = "smoke_e2e"
    raw["run"]["fold"] = 0
    raw["run"]["device"] = "cpu"
    raw["train"]["amp"] = False
    raw["viz"]["enabled"] = False
    raw["data"]["num_workers"] = 0
    cfg = SegmentationConfig.model_validate(raw)

    # ---- synthetic dataset factory (4 patients per split) ---------------
    patch = (32, 32, 32)
    n_patients = 4

    class _SyntheticDataset(Dataset):
        def __init__(self, ids: list, cfg: Any, *, augment: bool, target_cfg: Any) -> None:
            self._n = len(ids)

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int) -> dict:
            rng = np.random.default_rng(idx)
            image = rng.standard_normal((3, *patch)).astype(np.float32)
            target = (rng.random((2, *patch)) > 0.7).astype(np.float32)
            return {
                "image": torch.from_numpy(image),
                "target": torch.from_numpy(target),
            }

    # ---- mock resolution ------------------------------------------------
    resolution = _make_mock_resolution(set())  # no Ring-B in smoke
    plan = _make_mock_plan()
    # Override plan to have 4 patients in folds
    plan.fm_train_ids = tuple(f"P{i:03d}" for i in range(n_patients))
    plan.folds = (*[tuple(f"P{i:03d}" for i in range(n_patients))], *[(), ()])

    def _fake_resolve(data_cfg: Any) -> Any:
        return resolution

    def _fake_plan(
        data_cfg: Any, fm_splits: Any, *, dedup_duplicates: Any = None, cohort_labels: Any = None
    ) -> Any:
        return plan

    def _fake_write_splits(path: Path, res: Any, plan_: Any, *, extra: Any = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def _fake_eval(
        *, cfg: Any, result: Any, resolution: Any, plan: Any
    ) -> dict[str, dict[str, float]]:
        return {}  # empty — no cohorts in the smoke registry

    t0 = time.monotonic()

    with (
        patch(
            "vena.segmentation.data.fm_splits.resolve_fm_splits",
            side_effect=_fake_resolve,
        ),
        patch(
            "vena.segmentation.data.kfold.build_fold_plan",
            side_effect=_fake_plan,
        ),
        patch(
            "vena.segmentation.data.fm_splits.write_splits_json",
            side_effect=_fake_write_splits,
        ),
        patch(
            "routines.segmentation.train.engine.train_engine._evaluate_per_cohort",
            side_effect=_fake_eval,
        ),
        patch(
            "vena.segmentation.engine.train.SegTrainer._build_dataset",
            side_effect=lambda self, ids, augment: _SyntheticDataset(
                ids, cfg.data, augment=augment, target_cfg=cfg.targets
            ),
        ),
    ):
        from routines.segmentation.train.engine import SegTrainEngine

        engine = SegTrainEngine(cfg)
        run_dir = engine.run()

    elapsed = time.monotonic() - t0
    assert elapsed < 300, f"Smoke took {elapsed:.1f}s > 300s limit"

    run_dir = Path(run_dir)

    # ---- verify artifacts exist and are parseable -----------------------
    decision_path = run_dir / "decision.json"
    assert decision_path.exists(), "decision.json not created"
    with decision_path.open() as fh:
        payload = json.load(fh)
    assert payload["schema_version"] == "1.0.0"
    assert "gseg" in payload

    splits_path = run_dir / "splits.json"
    assert splits_path.exists(), "splits.json not created"
    with splits_path.open() as fh:
        json.load(fh)  # parseable

    config_path = run_dir / "config.resolved.yaml"
    assert config_path.exists(), "config.resolved.yaml not created"

    log_path = run_dir / "logs" / "train.log"
    assert log_path.exists(), "logs/train.log not created"

    ckpt_dir = run_dir / "checkpoints"
    assert ckpt_dir.exists(), "checkpoints/ directory not created"
    assert (ckpt_dir / "best.pt").exists() or (ckpt_dir / "last.pt").exists(), (
        "Neither best.pt nor last.pt found in checkpoints/"
    )

    # temperatures.json must not exist
    assert not (run_dir / "temperatures.json").exists(), (
        "temperatures.json was created — must not exist (Q5 decision)."
    )

    print(f"Smoke wall-clock: {elapsed:.1f} s")
    print(f"Run dir: {run_dir}")


# ---------------------------------------------------------------------------
# 7. Real inference test (non-mocked checkpoint + dataset)
# ---------------------------------------------------------------------------


def test_real_inference_metrics(tmp_path: Path) -> None:
    """_infer_cohort_metrics returns real (non-null) metrics from a SegTrainer checkpoint.

    Trains segresnet for 1 epoch on 4 synthetic patients via dataset_factory
    (no H5 access), then calls _infer_cohort_metrics against the produced
    checkpoint.  Asserts status='ok', tc/netc are floats in [0, 1].
    """
    import numpy as np
    import torch
    from routines.segmentation.train.engine.train_engine import _infer_cohort_metrics
    from torch.utils.data import Dataset

    from vena.segmentation.config import SegmentationConfig
    from vena.segmentation.data.kfold import FoldPlan
    from vena.segmentation.engine.train import SegTrainer

    smoke_yaml = _TRAIN_CONFIGS / "smoke.yaml"
    cfg_base = SegmentationConfig.from_yaml(smoke_yaml)
    raw = cfg_base.model_dump()
    raw["run"]["experiments_root"] = str(tmp_path)
    raw["run"]["device"] = "cpu"
    raw["run"]["tag"] = "real_inference_test"
    raw["train"]["amp"] = False
    raw["train"]["max_epochs"] = 1
    raw["train"]["val_every_epochs"] = 1
    raw["train"]["early_stop_patience"] = 10
    raw["data"]["num_workers"] = 0
    raw["data"]["patch_size"] = [16, 16, 16]
    raw["viz"]["enabled"] = False
    cfg = SegmentationConfig.model_validate(raw)

    tiny_patch = tuple(cfg.data.patch_size)
    train_ids = ("P0", "P1", "P2")
    val_ids = ("P3",)
    all_ids = train_ids + val_ids

    class _TinyDS(Dataset):
        def __init__(self, ids: Any, **_kwargs: Any) -> None:
            self._n = len(ids)

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, i: int) -> dict:
            rng = np.random.default_rng(i)
            return {
                "image": torch.from_numpy(rng.standard_normal((3, *tiny_patch)).astype(np.float32)),
                "target": torch.from_numpy((rng.random((2, *tiny_patch)) > 0.7).astype(np.float32)),
                "patient_id": f"P{i}",
            }

    # k=3: folds[i] = 1 held-out patient; for fold=0, train = P1+P2, val = P0
    plan = FoldPlan(
        k=3,
        fm_train_ids=train_ids,
        folds=(("P0",), ("P1",), ("P2",)),
        fm_val_ids=val_ids,
        fm_test_ids=(),
    )
    trainer_run_dir = tmp_path / "trainer_run"
    trainer_run_dir.mkdir()

    trainer = SegTrainer(
        cfg,
        fold=0,
        plan=plan,
        run_dir=trainer_run_dir,
        patient_to_scans=None,
        dataset_factory=_TinyDS,
    )
    result = trainer.fit()
    assert result.checkpoint.exists(), f"No checkpoint written at {result.checkpoint}"

    # ---- mock resolution and cohort-split (no H5 access required) --------
    @dataclass
    class _MockRes:
        patient_to_scans: dict = field(default_factory=dict)

    @dataclass
    class _MockCS:
        name: str = "UCSF-PDGM"
        role: str = "cv"
        image_h5: Path = Path("/fake/h5")

    # Patch SegImageDataset at its source — _run_inference_loop imports it there
    with patch("vena.segmentation.data.dataset.SegImageDataset", new=_TinyDS):
        metrics = _infer_cohort_metrics(
            cfg=cfg,
            best_ckpt=result.checkpoint,
            patient_ids=all_ids,
            resolution=_MockRes(),
            cohort_name="UCSF-PDGM",
            cs=_MockCS(),
        )

    assert metrics["status"] == "ok", f"Expected status='ok', got {metrics['status']!r}"
    assert isinstance(metrics["tc"], float), f"tc is {type(metrics['tc'])}, expected float"
    assert isinstance(metrics["netc"], float), f"netc is {type(metrics['netc'])}, expected float"
    assert 0.0 <= metrics["tc"] <= 1.0, f"tc={metrics['tc']} outside [0, 1]"
    assert 0.0 <= metrics["netc"] <= 1.0, f"netc={metrics['netc']} outside [0, 1]"
    assert metrics["n_evaluated"] > 0, "n_evaluated must be > 0"
    # Confirm no _zero_metrics sentinel survived
    assert metrics["tc"] is not None
    assert metrics["netc"] is not None


# ---------------------------------------------------------------------------
# 8. Missing-H5 / null-metrics gate failure
# ---------------------------------------------------------------------------


def test_null_metrics_fail_gseg_gate(tmp_path: Path) -> None:
    """Null metrics from a failed inference set gseg.passed=False with missing-data reason.

    Part A: _infer_cohort_metrics returns null + status='error:...' on a
    non-existent checkpoint (OSError).
    Part B: engine writes gseg.passed=False with a 'missing-data' entry in
    the failures list when _evaluate_per_cohort returns null metrics.
    """
    from routines.segmentation.train.engine.train_engine import _infer_cohort_metrics

    from vena.segmentation.config import SegmentationConfig

    smoke_yaml = _TRAIN_CONFIGS / "smoke.yaml"
    cfg = SegmentationConfig.from_yaml(smoke_yaml)

    # ---- Part A: missing checkpoint → null metrics ----------------------
    @dataclass
    class _MockResA:
        patient_to_scans: dict = field(default_factory=dict)

    @dataclass
    class _MockCSA:
        name: str = "UCSF-PDGM"
        role: str = "cv"
        image_h5: Path = Path("/nonexistent/h5")

    metrics = _infer_cohort_metrics(
        cfg=cfg,
        best_ckpt=tmp_path / "nonexistent.pt",  # file does not exist → OSError
        patient_ids=("P0", "P1"),
        resolution=_MockResA(),
        cohort_name="UCSF-PDGM",
        cs=_MockCSA(),
    )

    assert metrics["status"].startswith("error:"), (
        f"Expected status starting with 'error:', got {metrics['status']!r}"
    )
    assert metrics["tc"] is None, f"tc should be None for failed inference, got {metrics['tc']}"
    assert metrics["netc"] is None, "netc should be None for failed inference"

    # ---- Part B: engine gate fails with missing-data reason -------------
    raw = cfg.model_dump()
    raw["run"]["experiments_root"] = str(tmp_path)
    raw["run"]["tag"] = "null_gate_test"
    raw["run"]["fold"] = 0
    cfg_patched = SegmentationConfig.model_validate(raw)

    resolution_mock = _make_mock_resolution(_RING_B_NAMES)
    plan_mock = _make_mock_plan()

    def _null_evaluate(*, cfg: Any, result: Any, resolution: Any, plan: Any) -> dict:
        return {
            cs.name: {
                "status": "error: OSError: /fake/h5 not found",
                "n_evaluated": 0,
                "tc": None,
                "netc": None,
                "tc_ahd": None,
                "netc_ahd": None,
                "tc_ece": None,
                "netc_ece": None,
                "tc_brier": None,
                "netc_brier": None,
                "et_dice": None,
                "mean_et_soft": None,
            }
            for cs in resolution.per_cohort
        }

    def _fake_fit_null(self_trainer: Any) -> Any:
        rd = self_trainer._run_dir
        rd.mkdir(parents=True, exist_ok=True)
        return _make_fake_fit_result(rd)

    def _fake_write_splits_null(path: Path, *_args: Any, **_kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    with (
        patch(
            "vena.segmentation.data.fm_splits.resolve_fm_splits",
            side_effect=lambda _d: resolution_mock,
        ),
        patch(
            "vena.segmentation.data.kfold.build_fold_plan",
            side_effect=lambda _d, _s, **_kw: plan_mock,
        ),
        patch(
            "vena.segmentation.data.fm_splits.write_splits_json",
            side_effect=_fake_write_splits_null,
        ),
        patch(
            "vena.segmentation.engine.train.SegTrainer.fit",
            new=_fake_fit_null,
        ),
        patch(
            "routines.segmentation.train.engine.train_engine._evaluate_per_cohort",
            side_effect=_null_evaluate,
        ),
    ):
        from routines.segmentation.train.engine import SegTrainEngine

        engine = SegTrainEngine(cfg_patched)
        run_dir = Path(engine.run())

    with (run_dir / "decision.json").open() as fh:
        payload = json.load(fh)

    assert payload["gseg"]["passed"] is False, "Gate must fail when all cohort metrics are null"
    failures = payload["gseg"]["failures"]
    assert any("missing-data" in str(f) for f in failures), (
        f"Expected a 'missing-data' failure entry in gseg.failures, got: {failures}"
    )
