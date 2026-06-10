"""End-to-end integration test for the three resume modes.

Drives ``FMTrainRoutineEngine.run`` with a minimal config while monkey-patching
the heavy pieces (trunk load, DataModule, ``trainer.fit``) so the test runs in
a couple of seconds on CPU. The point is to verify the *integration* between
``_classify_resume_from`` → ``_resolve_resume_ckpt`` → ``_resolve_run_dir`` →
``_build_decision_payload`` → ``trainer.fit``/``WarmStartCallback`` dispatch,
which the unit tests cover only in isolation.

Each test:

1. Builds a minimal YAML config (no augmentation, no exhaustive_val).
2. Stubs out ``FMLightningModule._setup_trunk_and_controlnet``,
   ``load_registry`` (small synthetic), and ``pl.Trainer.fit`` so nothing
   touches the disk or GPU beyond the run dir itself.
3. Calls ``engine.run()``.
4. Inspects the resulting ``run_dir`` / ``decision.json`` / log-file output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from routines.fm.train.engine import (
    FMTrainRoutineConfig,
    FMTrainRoutineEngine,
    ResumeMode,
    _WarmStartCallback,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic registry + fixtures
# ---------------------------------------------------------------------------


class _FakeCohort:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegistry:
    """Mimics the surface used by ``Engine._build_decision_payload``."""

    cohorts = [_FakeCohort("UCSF-PDGM"), _FakeCohort("BraTS-GLI")]

    def cv_cohorts_with_aug(self):
        return []


def _write_registry(path: Path) -> Path:
    path.write_text(json.dumps({"cohorts": []}))
    return path


def _cfg_dict(
    experiments_root: Path,
    registry: Path,
    *,
    stage: str = "s1",
    tag: str = "smoke_resume",
    resume_from: str | None = None,
) -> dict:
    return {
        "run": {
            "stage": stage,
            "tag": tag,
            "resume_from": resume_from,
            "seed": 1337,
            "device": "cpu",
            "precision": "32",
            "full_determinism": False,
        },
        "data": {
            "corpus_registry": str(registry),
            "fold": 0,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "tau": 0.5,
        },
        "model": {
            "trunk": {
                "checkpoint": "/nonexistent/trunk.pt",
                "arch_json": None,
                "arch_overrides": {},
                "class_token": 9,
                "spacing_mm": [1.0, 1.0, 1.0],
                "trainable": True,
                "regime": "fft",
                "peft": None,
            },
            "controlnet": {
                "conditioning_inputs": ["latent:t1pre", "mask:wt:identity"],
                "arch_overrides": {},
                "perturb_keys": [],
            },
            "vae_checkpoint": "/nonexistent/vae.pt",
        },
        "loss": {"cfm": {"weight": 1.0, "reduction": "mean", "norm": "l2"}},
        "rflow": {
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": True,
            "sample_method": "uniform",
        },
        "optim": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "scheduler": "cosine",
        },
        "ema": {
            "decay": 0.9999,
            "update_after_step": 0,
            "update_every": 1,
            "inv_gamma": 10.0,
            "power": 1.0,
            "min_value": 0.0,
        },
        "training": {
            "total_steps": 4,
            "max_epochs": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "checkpoint_every_epochs": 1,
            "log_train_every_steps": 1,
            "best_metric_name": "mse_latent",
            "best_metric_region": "bg",
            "best_metric_nfe": 5,
            "gradient_clip_val": 1.0,
        },
        "validation": {
            "every_epochs": 0,
            "per_epoch_nfe": 5,
            "full_sweep_every_epochs": 999999,
            "sweep_nfes": [1, 5],
            "qualitative_every_epochs": 999999,
            "image_metrics": False,
        },
        "exhaustive_val": {"enabled": False},
        "output": {
            "experiments_root": str(experiments_root),
            "retention_n_checkpoints": 3,
            "tensorboard": False,
            "wandb": False,
        },
        "regions": {
            "brain": {"source": "fallback_all_ones"},
            "wt": {"source": "derived_from_tumor_latent", "threshold": 0.5},
            "wt_dilated": {
                "source": "derived_via_scipy_binary_dilation",
                "structure": "ones_3x3x3",
            },
            "bg": {"source": "derived"},
            "vessel": {"source": "skipped"},
        },
    }


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Patch the heavy bits so ``engine.run`` is a fast resume-logic harness.

    Yields a factory ``make(...) -> (engine, run_dir_after)`` that builds a
    config and drives ``Engine.run()`` once, returning the engine and the
    artifact path.
    """

    captured: dict[str, object] = {}

    # 1. Stub trunk + controlnet setup so the LightningModule constructs.
    from vena.model.fm.lightning.module import FMLightningModule

    monkeypatch.setattr(
        FMLightningModule, "_setup_trunk_and_controlnet", lambda self: None
    )
    monkeypatch.setattr(FMLightningModule, "setup", lambda self, stage=None: None)

    # 2. Stub the registry loader.
    monkeypatch.setattr(
        "routines.fm.train.engine.load_registry", lambda p: _FakeRegistry()
    )

    # 3. Stub MultiCohortLatentDataModule so it doesn't open any H5 files.
    class _FakeDM:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        "routines.fm.train.engine.MultiCohortLatentDataModule", _FakeDM
    )

    # 4. Stub trainer.fit — just record what ckpt_path was passed.
    import pytorch_lightning as pl

    def _fake_fit(self, model=None, datamodule=None, ckpt_path=None, **kw):
        captured["ckpt_path"] = ckpt_path
        captured["callbacks"] = list(self.callbacks)
        # When CONTINUE asks Lightning to resume from a real ckpt, write a
        # minimal file (Lightning would normally read it; we don't open it).
        return None

    monkeypatch.setattr(pl.Trainer, "fit", _fake_fit, raising=True)

    # 5. Stub VENACheckpointCallback so it doesn't try to write real ckpts.
    class _NoopCkpt:
        def __init__(self, *args, **kwargs) -> None:
            self.dirpath = kwargs.get("dirpath")

    monkeypatch.setattr("routines.fm.train.engine.VENACheckpointCallback", _NoopCkpt)
    monkeypatch.setattr("routines.fm.train.engine.BestCheckpointCallback", lambda *a, **k: _NoopCkpt())
    monkeypatch.setattr("routines.fm.train.engine.SigtermHandler", lambda *a, **k: _NoopCkpt())
    monkeypatch.setattr("routines.fm.train.engine.TrainMetricsCSV", lambda *a, **k: _NoopCkpt())
    monkeypatch.setattr("routines.fm.train.engine.ExhaustiveValLauncher", lambda *a, **k: _NoopCkpt())
    monkeypatch.setattr("routines.fm.train.engine.AugmentationTracker", lambda *a, **k: _NoopCkpt())
    monkeypatch.setattr("routines.fm.train.engine.VariantTracker", lambda *a, **k: _NoopCkpt())

    # 6. Skip the post-train plotter (writes to disk; not needed).
    monkeypatch.setattr("routines.fm.train.engine._run_post_train", lambda run_dir, formats: None)

    # 7. Stub the dedup gate (its absence makes _assert_preflight_gates a no-op).
    # Nothing to patch — gate is already a no-op when augmentation_config_path is None.

    def make(cfg_dict: dict) -> tuple[FMTrainRoutineEngine, Path]:
        cfg = FMTrainRoutineConfig.model_validate(cfg_dict)
        eng = FMTrainRoutineEngine(cfg)
        out = eng.run()
        return eng, out

    yield make, captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _read_decision(run_dir: Path) -> dict:
    return json.loads((run_dir / "decision.json").read_text())


def test_baseline_creates_new_dir(stub_engine, tmp_path: Path) -> None:
    make, captured = stub_engine
    root = tmp_path / "experiments"
    registry = _write_registry(tmp_path / "registry.json")
    cfg = _cfg_dict(root, registry, resume_from="baseline")
    _eng, run_dir = make(cfg)

    # Run dir was minted fresh.
    assert run_dir.parent == root
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_s1_smoke_resume_[0-9a-f]{8}$",
        run_dir.name,
    ), run_dir.name

    # No prior-run resume artifact; only the fresh provenance set.
    assert not list(run_dir.glob("config.resume_*.yaml"))
    assert (run_dir / "config.yaml").exists()

    # Decision.json fields.
    d = _read_decision(run_dir)
    assert d["schema_version"] == "0.8.0"
    assert d["tag"] == "smoke_resume"
    assert d["resume_mode"] == "baseline"
    assert d["resume_source"] == "baseline"
    assert d["resume_source_run_id"] is None

    # trainer.fit got no ckpt_path; no WarmStartCallback.
    assert captured["ckpt_path"] is None
    assert not any(isinstance(cb, _WarmStartCallback) for cb in captured["callbacks"])


def test_continue_reuses_dir(stub_engine, tmp_path: Path) -> None:
    make, captured = stub_engine
    root = tmp_path / "experiments"
    registry = _write_registry(tmp_path / "registry.json")

    # Phase A: BASELINE seeds a sibling.
    cfg_a = _cfg_dict(root, registry, resume_from="baseline")
    _, run_a = make(cfg_a)
    # Plant a fake last.ckpt so CONTINUE has something to resume from.
    fake_ckpt = run_a / "checkpoints" / "last.ckpt"
    fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
    fake_ckpt.write_text("x")

    # Phase B: CONTINUE should reuse the same dir.
    cfg_b = _cfg_dict(root, registry, resume_from="latest")
    _, run_b = make(cfg_b)

    assert run_b == run_a, "CONTINUE must reuse the prior run's dir"
    # A config.resume_*.yaml was written for the audit trail.
    assert list(run_b.glob("config.resume_*.yaml"))

    # decision.json is NOT overwritten on resume (the engine intentionally
    # leaves it; we only check the file exists and still carries BASELINE).
    d = _read_decision(run_b)
    assert d["resume_mode"] == "baseline"

    # trainer.fit received the prior ckpt path.
    assert captured["ckpt_path"] == str(fake_ckpt)
    assert not any(isinstance(cb, _WarmStartCallback) for cb in captured["callbacks"])


def test_warm_start_creates_new_dir_and_injects_callback(
    stub_engine, tmp_path: Path
) -> None:
    make, captured = stub_engine
    root = tmp_path / "experiments"
    registry = _write_registry(tmp_path / "registry.json")

    # Phase A: BASELINE seeds the source.
    cfg_a = _cfg_dict(root, registry, resume_from="baseline")
    _, run_a = make(cfg_a)
    fake_ckpt = run_a / "checkpoints" / "last.ckpt"
    fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
    fake_ckpt.write_text("x")

    # Phase C: WARM_START from the source run_id. The destination uses a
    # *different* tag so the new run lands in its own dir (the s1→s2 use case
    # with stage swap; here we stay in s1 but change recipe).
    cfg_c = _cfg_dict(root, registry, tag="smoke_warm_dst", resume_from=run_a.name)
    _, run_c = make(cfg_c)

    # Brand-new dir, different name.
    assert run_c != run_a
    assert run_c.parent == root
    assert "_smoke_warm_dst_" in run_c.name

    # decision.json records the warm-start lineage.
    d = _read_decision(run_c)
    assert d["resume_mode"] == "warm_start"
    assert d["resume_source"] == run_a.name
    assert d["resume_source_run_id"] == run_a.name
    assert d["tag"] == "smoke_warm_dst"

    # trainer.fit did NOT receive a ckpt_path; the WarmStartCallback was
    # injected to perform the weights-only load at on_fit_start.
    assert captured["ckpt_path"] is None
    warm_cbs = [cb for cb in captured["callbacks"] if isinstance(cb, _WarmStartCallback)]
    assert len(warm_cbs) == 1
    assert warm_cbs[0].ckpt_path == str(fake_ckpt)
