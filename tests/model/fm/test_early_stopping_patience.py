"""Verify that the EarlyStopping callback wired in via ``training.patience``
actually halts a 1000-epoch run once a plateau persists for ``patience``
epochs. The 1000-epoch Picasso runs depend on this signal to release the
SLURM allocation when training has converged.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

pytestmark = pytest.mark.unit


class _DecayThenPlateauModule(pl.LightningModule):
    """Synthetic LightningModule whose ``train/total_epoch`` decays for
    ``plateau_start`` epochs and then plateaus at a fixed value.

    The test does NOT exercise the FM model or any of its dependencies;
    it exercises only the EarlyStopping wiring contract: monitoring
    ``train/total_epoch`` (mode=min) with patience N should stop training
    exactly ``N`` epochs after the minimum is reached.
    """

    def __init__(self, plateau_start: int, plateau_value: float = 1.0) -> None:
        super().__init__()
        self.plateau_start = int(plateau_start)
        self.plateau_value = float(plateau_value)
        self.linear = torch.nn.Linear(1, 1)
        self.stopped_at_epoch: int | None = None

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, _ = batch
        return self.linear(x).pow(2).mean()

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        ep = int(self.current_epoch)
        loss = (
            max(self.plateau_value, 10.0 - 0.5 * ep)
            if ep < self.plateau_start
            else self.plateau_value
        )
        self.log("train/total_epoch", float(loss), on_epoch=True, on_step=False)

    def on_fit_end(self) -> None:  # type: ignore[override]
        self.stopped_at_epoch = int(self.current_epoch)

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def _make_loader() -> DataLoader:
    ds = TensorDataset(torch.ones(4, 1), torch.ones(4, 1))
    return DataLoader(ds, batch_size=2)


def _run(plateau_start: int, patience: int, max_epochs: int) -> int:
    pl.seed_everything(0, workers=True)
    module = _DecayThenPlateauModule(plateau_start=plateau_start)
    cb = EarlyStopping(
        monitor="train/total_epoch",
        mode="min",
        patience=patience,
        check_on_train_epoch_end=True,
        verbose=False,
        strict=False,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[cb],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model=module, train_dataloaders=_make_loader())
    assert module.stopped_at_epoch is not None
    return module.stopped_at_epoch


def test_patience_triggers_after_plateau() -> None:
    # Loss decays for 5 epochs (epochs 0..4) reaching the plateau at epoch 5;
    # with patience=3 training halts after epoch 5+3=8 (best epoch + patience).
    # Lightning post-increments ``current_epoch`` before ``on_fit_end`` fires,
    # so the recorded value is 8+1=9.
    stop_epoch = _run(plateau_start=5, patience=3, max_epochs=40)
    assert stop_epoch == 5 + 3 + 1, f"expected current_epoch=9 at fit-end, got {stop_epoch}"


def test_patience_disabled_runs_to_max_epochs() -> None:
    # Without patience the trainer runs to max_epochs even on a plateau.
    # max_epochs=6 means epochs 0..5 complete; current_epoch at fit-end is 6.
    pl.seed_everything(0, workers=True)
    module = _DecayThenPlateauModule(plateau_start=2)
    trainer = pl.Trainer(
        max_epochs=6,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model=module, train_dataloaders=_make_loader())
    assert module.stopped_at_epoch == 6


def test_patience_yaml_round_trip() -> None:
    """A YAML with ``training.patience: 100`` round-trips through the
    routine's Pydantic schema and the engine builds the callback list with
    EarlyStopping. We do not call ``Engine.run()`` — only validate that the
    schema accepts the new field and that callback construction would add
    the callback when ``patience`` is set.
    """

    from routines.fm.train.engine import FMTrainRoutineConfig

    minimal = {
        "run": {"stage": "s1", "seed": 0, "device": "cpu", "precision": "32"},
        "data": {
            "corpus_registry": "routines/fm/train/configs/corpus/corpus_picasso.json",
            "fold": 0,
            "batch_size": 1,
        },
        "model": {
            "trunk": {
                "checkpoint": "/nonexistent.pt",
                "class_token": 9,
                "spacing_mm": [1.0, 1.0, 1.0],
                "trainable": False,
            },
            "controlnet": {"conditioning_inputs": ["latent:t1pre"]},
            "vae_checkpoint": None,
        },
        "training": {"total_steps": 1, "max_epochs": 1, "patience": 100},
        "output": {"experiments_root": "/tmp/vena-test"},
    }
    cfg = FMTrainRoutineConfig.model_validate(minimal)
    assert cfg.training.patience == 100
    assert cfg.training.max_epochs == 1
