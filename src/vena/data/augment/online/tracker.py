"""Lightning callback that counts per-epoch augmentation combinations.

The dataset stores each sample's combination tag under the key
``_aug_combo``. PyTorch's default collate batches strings into a list of
strings, so the callback's :meth:`on_train_batch_end` can simply iterate the
list and increment a ``(epoch, combo)`` counter.

At :meth:`on_fit_end` the counter is flushed to
``metrics/augmentations_per_epoch.csv`` with columns ``epoch,combo,count``;
combinations are written in lexicographic order within each epoch so the CSV
is reproducible across runs.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Callback

from vena.data.augment.online.pipeline import NO_AUG_TAG

logger = logging.getLogger(__name__)

_COMBO_KEY: str = "_aug_combo"
_VARIANT_KEY: str = "_aug_variant"
_CSV_NAME: str = "augmentations_per_epoch.csv"
_VARIANT_CSV_NAME: str = "variants_per_epoch.csv"
_CLEAN_VARIANT: str = "v0"


class AugmentationTracker(Callback):
    """Count, per epoch, how many samples got each augmentation combination.

    The callback is silent until at least one batch carries the
    ``_aug_combo`` key — so training runs without augmentation cost nothing
    and produce no spurious CSV.

    Parameters
    ----------
    out_dir : Path | str
        Destination directory (typically ``run_dir / "metrics"``); the CSV is
        written under ``out_dir / "augmentations_per_epoch.csv"`` at
        :meth:`on_fit_end` and also flushed at every epoch end so a
        SIGTERM-killed run keeps partial data.
    """

    def __init__(self, out_dir: Path | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self._counts: Counter[tuple[int, str]] = Counter()
        self._saw_any: bool = False

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        combos = batch.get(_COMBO_KEY) if isinstance(batch, dict) else None
        if combos is None:
            return
        if isinstance(combos, str):
            combos = [combos]
        epoch = int(trainer.current_epoch)
        for c in combos:
            self._counts[(epoch, str(c))] += 1
        self._saw_any = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Incremental flush — keeps partial data on SIGTERM mid-run.
        if self._saw_any:
            self._flush()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._saw_any:
            self._flush()

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        path = self.out_dir / _CSV_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "combo", "count"])
            for (epoch, combo), count in rows:
                # Replace empty combos defensively — should never happen since
                # the pipeline writes NO_AUG_TAG, but the column should not
                # contain blank strings even on operator error.
                if not combo:
                    combo = NO_AUG_TAG
                w.writerow([epoch, combo, count])
        logger.info("AugmentationTracker: wrote %d rows to %s", len(rows), path)


class VariantTracker(Callback):
    """Count, per epoch, how many samples got each offline-aug bank variant.

    Companion to :class:`AugmentationTracker`. The
    :class:`vena.model.fm.lightning.data.OfflineAugmentedLatentH5Dataset`
    writes the sampled variant tag (``"v0"`` for the clean source, or
    ``"v1".."vN"`` for a bank variant) into each sample dict under
    ``_aug_variant``. The callback is silent on runs that do not enable the
    offline bank.

    Parameters
    ----------
    out_dir : Path | str
        Destination directory (typically ``run_dir / "metrics"``). The CSV
        is written to ``out_dir / "variants_per_epoch.csv"`` with columns
        ``epoch,variant,count``.
    """

    def __init__(self, out_dir: Path | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self._counts: Counter[tuple[int, str]] = Counter()
        self._saw_any: bool = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        variants = batch.get(_VARIANT_KEY) if isinstance(batch, dict) else None
        if variants is None:
            return
        if isinstance(variants, str):
            variants = [variants]
        epoch = int(trainer.current_epoch)
        for v in variants:
            self._counts[(epoch, str(v))] += 1
        self._saw_any = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._saw_any:
            self._flush()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._saw_any:
            self._flush()

    def _flush(self) -> None:
        path = self.out_dir / _VARIANT_CSV_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "variant", "count"])
            for (epoch, variant), count in rows:
                if not variant:
                    variant = _CLEAN_VARIANT
                w.writerow([epoch, variant, count])
        logger.info("VariantTracker: wrote %d rows to %s", len(rows), path)
