"""HDF5-backed dataset + LightningDataModule for FM training.

Reads ``UCSFPDGM_latents.h5`` (schema 0.1.0, produced by
``vena.data.h5.ucsf_pdgm.latent_domain.convert``). Per-patient single-chunk
reads keep streaming cheap.

Each item is a dict::

    {
        "patient_id": str,
        "z_t1pre":    Tensor (4, 60, 60, 40),
        "z_t2":       Tensor (4, 60, 60, 40),
        "z_flair":    Tensor (4, 60, 60, 40),
        "z_t1c":      Tensor (4, 60, 60, 40),    # target
        "m_wt":       Tensor (1, 60, 60, 40),    # binary WT mask
    }

The WT mask is derived from ``masks/tumor_latent[i]`` (3 soft NETC/ED/ET maps)
by ``m_wt = (clip(c0+c1+c2, 0, 1) >= 0.5).float()`` per proposal §2.2.

Additional modalities (ADC, SWI) or priors (vessel, perfusion) are loaded on
demand: pass their names through ``extra_latents`` / ``extra_priors`` and the
dataset will produce the matching keys (``z_<name>`` / ``prior_<name>``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class LatentH5Dataset(Dataset):
    """Per-patient view over ``UCSFPDGM_latents.h5``.

    Parameters
    ----------
    h5_path : Path | str
        Path to the latents H5.
    patient_ids : Sequence[str]
        IDs (as found under ``/ids``) that this dataset will serve. Order is
        preserved.
    latents : Sequence[str]
        Modalities to read from ``/latents/<name>``. Default
        ``("t1pre", "t2", "flair", "t1c")``; the first three become the
        conditioning inputs and ``t1c`` is the target.
    wt_threshold : float
        Threshold applied to the soft tumour-mask union to obtain the binary
        WT mask. Default 0.5 (proposal §2.2).
    extra_priors : Sequence[str] | None
        Names under ``/priors/<name>``; emitted as ``prior_<name>``. Empty by
        default since ``priors/`` is currently empty in the H5.
    """

    DEFAULT_LATENTS: tuple[str, ...] = ("t1pre", "t2", "flair", "t1c")

    def __init__(
        self,
        h5_path: Path | str,
        patient_ids: Sequence[str],
        latents: Sequence[str] = DEFAULT_LATENTS,
        wt_threshold: float = 0.5,
        extra_priors: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"latents H5 not found: {self.h5_path}")
        self.patient_ids: list[str] = list(patient_ids)
        self.latents: tuple[str, ...] = tuple(latents)
        self.wt_threshold = float(wt_threshold)
        self.extra_priors: tuple[str, ...] = tuple(extra_priors or ())
        self._h5: h5py.File | None = None
        self._idx_by_id: dict[str, int] | None = None

    def _ensure_open(self) -> h5py.File:
        """Lazily open the H5 on first access; safe for forked workers."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            ids = self._h5["ids"][:]
            ids_str = [b.decode() if isinstance(b, bytes) else str(b) for b in ids]
            self._idx_by_id = {pid: i for i, pid in enumerate(ids_str)}
        return self._h5

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        h5 = self._ensure_open()
        assert self._idx_by_id is not None  # populated by _ensure_open
        pid = self.patient_ids[idx]
        if pid not in self._idx_by_id:
            raise KeyError(f"patient_id '{pid}' not found in {self.h5_path}/ids")
        row = self._idx_by_id[pid]

        out: dict[str, torch.Tensor | str] = {"patient_id": pid}
        for name in self.latents:
            arr = h5[f"latents/{name}"][row]  # (4, 60, 60, 40) float32
            out[f"z_{name}"] = torch.from_numpy(np.ascontiguousarray(arr)).float()

        tumor_lat = h5["masks/tumor_latent"][row]  # (3, 60, 60, 40), soft NETC/ED/ET
        soft_union = np.clip(tumor_lat.sum(axis=0, keepdims=True), 0.0, 1.0)
        m_wt = (soft_union >= self.wt_threshold).astype(np.float32)
        out["m_wt"] = torch.from_numpy(np.ascontiguousarray(m_wt))

        for prior in self.extra_priors:
            arr = h5[f"priors/{prior}"][row]
            out[f"prior_{prior}"] = torch.from_numpy(np.ascontiguousarray(arr)).float()

        return out

    def __getstate__(self) -> dict:
        # Drop the open H5 handle so the dataset survives DataLoader worker pickling.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_idx_by_id"] = None
        return state


class LatentH5DataModule(pl.LightningDataModule):
    """LightningDataModule reading the UCSF-PDGM latents H5.

    Splits come directly from the H5: ``splits/cv/fold_<k>/{train,val}`` for
    cross-validation, and ``splits/test`` for the held-out test set.

    Parameters
    ----------
    h5_path : Path | str
    fold : int
        Cross-validation fold (0..4 in UCSFPDGM_latents.h5).
    batch_size : int
    num_workers : int
    pin_memory : bool
    max_train_subjects : int | None
        If set, randomly subsamples the train split to this size with a
        deterministic seed. Used for the S1 smoke (4 subjects).
    seed : int
    """

    def __init__(
        self,
        h5_path: Path | str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 2,
        pin_memory: bool = True,
        max_train_subjects: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_train_subjects = max_train_subjects
        self.seed = int(seed)
        self._train_ids: list[str] = []
        self._val_ids: list[str] = []
        self._test_ids: list[str] = []

    def _decode_ids(self, ds: h5py.Dataset) -> list[str]:
        return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

    def setup(self, stage: str | None = None) -> None:
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"latents H5 not found: {self.h5_path}")
        with h5py.File(self.h5_path, "r", swmr=True) as f:
            self._train_ids = self._decode_ids(f[f"splits/cv/fold_{self.fold}/train"])
            self._val_ids = self._decode_ids(f[f"splits/cv/fold_{self.fold}/val"])
            self._test_ids = self._decode_ids(f["splits/test"])
        if self.max_train_subjects is not None and self.max_train_subjects < len(
            self._train_ids
        ):
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(
                len(self._train_ids), size=self.max_train_subjects, replace=False
            )
            self._train_ids = [self._train_ids[int(i)] for i in sorted(chosen)]
            logger.info(
                "LatentH5DataModule: subsampled train to %d subjects (seed=%d)",
                len(self._train_ids),
                self.seed,
            )
        logger.info(
            "LatentH5DataModule.setup: fold=%d train=%d val=%d test=%d",
            self.fold,
            len(self._train_ids),
            len(self._val_ids),
            len(self._test_ids),
        )

    def _make_dataset(self, ids: list[str]) -> LatentH5Dataset:
        return LatentH5Dataset(self.h5_path, ids)

    def _make_loader(self, ids: list[str], shuffle: bool) -> DataLoader:
        dataset = self._make_dataset(ids)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ids, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ids, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ids, shuffle=False)
