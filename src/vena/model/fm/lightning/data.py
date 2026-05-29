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
import math
import random
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    """Deterministic per-worker seeding (training_routine.md §9).

    Lightning seeds the main process via ``seed_everything(workers=True)``,
    but per-worker NumPy / Python ``random`` state still needs an explicit
    re-seed because some libraries (e.g. h5py via swmr) call ``random``
    during file-open. Mirrors PyTorch's ``DataLoader`` documentation example.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


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
        try:
            return self._read_one(idx)
        except (OSError, KeyError) as exc:
            # training_routine.md §10: skip corrupt-H5 reads, try the next patient.
            pid = self.patient_ids[idx]
            logger.warning("H5 read failed for patient '%s' (%s); skipping.", pid, exc)
            next_idx = (idx + 1) % len(self.patient_ids)
            if next_idx == idx:
                raise
            return self._read_one(next_idx)

    def _read_one(self, idx: int) -> dict[str, torch.Tensor | str]:
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
        max_val_subjects: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_train_subjects = max_train_subjects
        self.max_val_subjects = max_val_subjects
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
        if self.max_val_subjects is not None and self.max_val_subjects < len(self._val_ids):
            rng = np.random.default_rng(self.seed + 1)
            chosen = rng.choice(
                len(self._val_ids), size=self.max_val_subjects, replace=False
            )
            self._val_ids = [self._val_ids[int(i)] for i in sorted(chosen)]
            logger.info(
                "LatentH5DataModule: subsampled val to %d subjects (seed=%d)",
                len(self._val_ids),
                self.seed + 1,
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
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ids, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ids, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_ids, shuffle=False)


# ---------------------------------------------------------------------------
# Multi-cohort pooling
# ---------------------------------------------------------------------------


class MultiCohortLatentDataset(Dataset):
    """Flat global index over a list of per-cohort ``LatentH5Dataset`` objects.

    Parameters
    ----------
    cohorts : list[tuple[str, LatentH5Dataset]]
        Ordered ``(cohort_name, dataset)`` pairs. The global flat index is
        built by concatenating per-cohort indices in this order.
    """

    def __init__(self, cohorts: list[tuple[str, LatentH5Dataset]]) -> None:
        super().__init__()
        if not cohorts:
            raise ValueError("MultiCohortLatentDataset requires at least one cohort")
        self._cohort_names: list[str] = [name for name, _ in cohorts]
        self._datasets: list[LatentH5Dataset] = [ds for _, ds in cohorts]

        # Precompute cumulative offsets for O(log N) global→local mapping.
        self._offsets: list[int] = [0]
        for ds in self._datasets:
            self._offsets.append(self._offsets[-1] + len(ds))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def cohort_of(self, global_idx: int) -> str:
        """Return the cohort name for a global index."""
        cohort_idx, _ = self._resolve(global_idx)
        return self._cohort_names[cohort_idx]

    def cohort_ranges(self) -> list[tuple[str, int, int]]:
        """Return ``(cohort_name, start, length)`` for each cohort."""
        return [
            (self._cohort_names[i], self._offsets[i], len(self._datasets[i]))
            for i in range(len(self._datasets))
        ]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, global_idx: int) -> dict[str, torch.Tensor | str]:
        cohort_idx, local_idx = self._resolve(global_idx)
        item = self._datasets[cohort_idx][local_idx]
        item["cohort"] = self._cohort_names[cohort_idx]
        return item

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, global_idx: int) -> tuple[int, int]:
        """Map *global_idx* to ``(cohort_idx, local_idx)``."""
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(
                f"global_idx {global_idx} out of range [0, {len(self)})"
            )
        # Binary search over offsets.
        lo, hi = 0, len(self._datasets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        cohort_idx = lo
        local_idx = global_idx - self._offsets[cohort_idx]
        return cohort_idx, local_idx


class TemperatureBalancedSampler(torch.utils.data.Sampler):
    """Temperature-balanced patient-aware sampler for multi-cohort training.

    Cohort sampling probabilities are proportional to ``N_c^tau``, where
    ``N_c`` is the number of *patients* in cohort c and ``tau`` is the
    temperature. ``tau=0`` → uniform over cohorts; ``tau=1`` → proportional
    to patient count.

    Within each cohort a patient is drawn uniformly, then one of that
    patient's scans is drawn uniformly — producing an unbiased per-scan
    estimate over the cohort's distribution.

    Parameters
    ----------
    cohort_patient_scan_indices : list[list[list[int]]]
        Outer dimension = cohorts (same order as the ``MultiCohortLatentDataset``
        cohorts); middle = patients; inner = that patient's GLOBAL scan indices.
    batch_size : int
        Number of samples per batch yielded.
    tau : float
        Temperature for cohort weighting.
    seed : int
        Base RNG seed; epoch counter is mixed in so successive epochs differ.
    length_in_batches : int | None
        Override the default epoch length (default: ``ceil(total_train_scans /
        batch_size)``).
    """

    def __init__(
        self,
        cohort_patient_scan_indices: list[list[list[int]]],
        batch_size: int,
        tau: float = 0.5,
        seed: int = 42,
        length_in_batches: int | None = None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
        if not cohort_patient_scan_indices:
            raise ValueError("cohort_patient_scan_indices must be non-empty")

        self._cps = cohort_patient_scan_indices
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.seed = int(seed)
        self._epoch: int = 0

        n_cohorts = len(self._cps)
        n_patients_per_cohort = [len(patients) for patients in self._cps]
        total_scans = sum(
            sum(len(scans) for scans in patients)
            for patients in self._cps
        )

        # Cohort probabilities via temperature scaling.
        nc = np.array(n_patients_per_cohort, dtype=np.float64)
        if self.tau == 0.0:
            weights = np.ones(n_cohorts, dtype=np.float64)
        else:
            weights = nc ** self.tau
        self._p_cohort: np.ndarray = weights / weights.sum()

        if length_in_batches is not None:
            self._length_in_batches = int(length_in_batches)
        else:
            self._length_in_batches = max(1, math.ceil(total_scans / self.batch_size))

    def __len__(self) -> int:
        return self._length_in_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Advance the internal epoch so successive epochs differ."""
        self._epoch = int(epoch)

    def __iter__(self):  # type: ignore[override]
        # Mix seed and epoch so the same sampler produces different sequences
        # across epochs while remaining fully reproducible.
        rng = np.random.default_rng(self.seed + self._epoch * 1_000_003)
        self._epoch += 1  # auto-advance

        n_cohorts = len(self._cps)

        for _ in range(self._length_in_batches):
            batch_indices: list[int] = []
            # Draw cohort slots for this batch.
            # First draw min(B, n_cohorts) distinct cohorts (diversity guarantee).
            n_distinct = min(self.batch_size, n_cohorts)
            distinct_cohorts = list(
                rng.choice(
                    n_cohorts,
                    size=n_distinct,
                    replace=False,
                    p=self._p_cohort,
                )
            )
            cohort_slots: list[int] = distinct_cohorts
            # Fill remaining slots with replacement.
            remaining = self.batch_size - n_distinct
            if remaining > 0:
                extra = list(
                    rng.choice(
                        n_cohorts,
                        size=remaining,
                        replace=True,
                        p=self._p_cohort,
                    )
                )
                cohort_slots = cohort_slots + extra

            for cohort_idx in cohort_slots:
                patients = self._cps[cohort_idx]
                patient_idx = int(rng.integers(0, len(patients)))
                scans = patients[patient_idx]
                scan_pos = int(rng.integers(0, len(scans)))
                batch_indices.append(scans[scan_pos])

            yield from batch_indices


# ---------------------------------------------------------------------------
# Multi-cohort LightningDataModule
# ---------------------------------------------------------------------------


class MultiCohortLatentDataModule(pl.LightningDataModule):
    """LightningDataModule pooling multiple cohort latent H5s for FM training.

    Reads split keys from each cohort's H5 (``splits/cv/fold_<fold>/train``,
    ``splits/cv/fold_<fold>/val``, ``splits/test``), expands patient keys to
    scan rows via the CSR layout, and builds a ``MultiCohortLatentDataset``
    for train / val / test. The train loader uses ``TemperatureBalancedSampler``
    for cohort-balanced sampling.

    Parameters
    ----------
    registry : CorpusRegistry
        Corpus catalogue; cv cohorts feed train/val/test; test-only cohorts
        feed only the test split.
    fold : int
        CV fold index (0-based, must exist in every cv cohort's H5).
    batch_size : int
    tau : float
        Temperature for cohort weighting in ``TemperatureBalancedSampler``.
    num_workers : int
    pin_memory : bool
    seed : int
    max_train_patients_per_cohort : int | None
        If set, deterministically cap each cohort's train patient list to this
        many (useful for smoke runs).
    """

    def __init__(
        self,
        registry,  # CorpusRegistry — avoid circular import at module level
        fold: int = 0,
        batch_size: int = 2,
        tau: float = 0.5,
        num_workers: int = 2,
        pin_memory: bool = True,
        seed: int = 42,
        max_train_patients_per_cohort: int | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.seed = int(seed)
        self.max_train_patients_per_cohort = max_train_patients_per_cohort

        self._train_ds: MultiCohortLatentDataset | None = None
        self._val_ds: MultiCohortLatentDataset | None = None
        self._test_ds: MultiCohortLatentDataset | None = None
        # Per-cohort patient→[global_indices] for the sampler.
        self._train_patient_scan_indices: list[list[list[int]]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_ids(ds: "h5py.Dataset") -> list[str]:
        return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

    @staticmethod
    def _expand_patients_to_scans(
        offsets: "np.ndarray",
        keys: list[str],
        ids: list[str],
        patient_keys: list[str],
    ) -> tuple[list[str], list[list[int]]]:
        """Expand patient keys to scan-level ids and local-scan-index groups.

        Parameters
        ----------
        offsets : np.ndarray
            CSR offsets array, shape ``(n_patients+1,)``.
        keys : list[str]
            Patient keys in CSR order, length ``n_patients``.
        ids : list[str]
            All scan ids in the H5, length ``n_scans``.
        patient_keys : list[str]
            Requested patient keys (subset of ``keys``).

        Returns
        -------
        scan_ids : list[str]
            Scan ids for the requested patients (order: patients then scans
            within each patient).
        patient_to_local_indices : list[list[int]]
            For each patient in ``patient_keys``, the LOCAL scan indices
            (into ``scan_ids``) belonging to that patient.
        """
        key_to_pos: dict[str, int] = {k: i for i, k in enumerate(keys)}
        missing = [pk for pk in patient_keys if pk not in key_to_pos]
        if missing:
            raise KeyError(
                f"patient keys not found in CSR: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )

        scan_ids: list[str] = []
        patient_to_local: list[list[int]] = []
        for pk in patient_keys:
            pos = key_to_pos[pk]
            start, end = int(offsets[pos]), int(offsets[pos + 1])
            local_start = len(scan_ids)
            for row in range(start, end):
                scan_ids.append(ids[row])
            patient_to_local.append(list(range(local_start, len(scan_ids))))

        return scan_ids, patient_to_local

    def _open_cohort_h5(self, latent_h5: "Path") -> "h5py.File":
        return h5py.File(latent_h5, "r", swmr=True)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:  # noqa: C901
        from vena.data.registry.models import CorpusRegistry  # local import

        train_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        val_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        test_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        self._train_patient_scan_indices = []

        rng_subsample = np.random.default_rng(self.seed)

        # --- cv cohorts (contribute train/val/test) ---
        for cohort in self.registry.cv_cohorts():
            with self._open_cohort_h5(cohort.latent_h5) as f:
                ids = self._decode_ids(f["ids"])
                offsets = f["patients/offsets"][:]
                keys = self._decode_ids(f["patients/keys"])
                train_patient_keys = self._decode_ids(
                    f[f"splits/cv/fold_{self.fold}/train"]
                )
                val_patient_keys = self._decode_ids(
                    f[f"splits/cv/fold_{self.fold}/val"]
                )
                test_patient_keys = self._decode_ids(f["splits/test"])

            # Optional cap for smoke runs.
            if (
                self.max_train_patients_per_cohort is not None
                and len(train_patient_keys) > self.max_train_patients_per_cohort
            ):
                chosen = rng_subsample.choice(
                    len(train_patient_keys),
                    size=self.max_train_patients_per_cohort,
                    replace=False,
                )
                train_patient_keys = [train_patient_keys[int(i)] for i in sorted(chosen)]
                logger.info(
                    "%s: capped train patients to %d",
                    cohort.name,
                    self.max_train_patients_per_cohort,
                )

            # Expand patients → scan ids and local index groups.
            train_scan_ids, train_p2l = self._expand_patients_to_scans(
                offsets, keys, ids, train_patient_keys
            )
            val_scan_ids, _ = self._expand_patients_to_scans(
                offsets, keys, ids, val_patient_keys
            )
            test_scan_ids, _ = self._expand_patients_to_scans(
                offsets, keys, ids, test_patient_keys
            )

            logger.info(
                "%s: train=%d patients / %d scans | val=%d patients / %d scans | "
                "test=%d patients / %d scans",
                cohort.name,
                len(train_patient_keys),
                len(train_scan_ids),
                len(val_patient_keys),
                len(val_scan_ids),
                len(test_patient_keys),
                len(test_scan_ids),
            )

            # Build per-cohort datasets.
            train_ds = LatentH5Dataset(cohort.latent_h5, train_scan_ids)
            val_ds = LatentH5Dataset(cohort.latent_h5, val_scan_ids)
            test_ds = LatentH5Dataset(cohort.latent_h5, test_scan_ids)

            train_cohort_datasets.append((cohort.name, train_ds))
            val_cohort_datasets.append((cohort.name, val_ds))
            test_cohort_datasets.append((cohort.name, test_ds))

            # Build global scan indices for the sampler.
            # At this point, the train MultiCohortLatentDataset is not yet
            # assembled, so we accumulate local indices and resolve them to
            # global indices after building the train dataset.
            self._train_patient_scan_indices.append(train_p2l)

        # --- test-only cohorts (contribute to test only) ---
        for cohort in self.registry.test_cohorts():
            with self._open_cohort_h5(cohort.latent_h5) as f:
                ids = self._decode_ids(f["ids"])
                offsets = f["patients/offsets"][:]
                keys = self._decode_ids(f["patients/keys"])
                # Use all patients in the H5 (no train split for test-only).
                all_patient_keys = list(keys)

            all_scan_ids, _ = self._expand_patients_to_scans(
                offsets, keys, ids, all_patient_keys
            )
            logger.info(
                "%s (test-only): %d patients / %d scans",
                cohort.name,
                len(all_patient_keys),
                len(all_scan_ids),
            )
            test_ds = LatentH5Dataset(cohort.latent_h5, all_scan_ids)
            test_cohort_datasets.append((cohort.name, test_ds))

        if not train_cohort_datasets:
            raise RuntimeError("No cv cohorts in registry; cannot build train dataset.")

        # Assemble multi-cohort datasets.
        self._train_ds = MultiCohortLatentDataset(train_cohort_datasets)
        self._val_ds = MultiCohortLatentDataset(val_cohort_datasets)
        self._test_ds = MultiCohortLatentDataset(test_cohort_datasets)

        # Resolve local scan indices → global scan indices for the sampler.
        # The offsets of train_cohort_datasets in the assembled dataset:
        #   cohort i starts at self._train_ds._offsets[i]
        resolved: list[list[list[int]]] = []
        for cohort_idx, p2l in enumerate(self._train_patient_scan_indices):
            cohort_global_offset = self._train_ds._offsets[cohort_idx]
            global_p2l = [
                [cohort_global_offset + local for local in patient_locals]
                for patient_locals in p2l
            ]
            resolved.append(global_p2l)
        self._train_patient_scan_indices = resolved

        logger.info(
            "MultiCohortLatentDataModule ready: train=%d scans, val=%d scans, "
            "test=%d scans across %d cv cohort(s)",
            len(self._train_ds),
            len(self._val_ds),
            len(self._test_ds),
            len(self.registry.cv_cohorts()),
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "call setup() first"
        sampler = TemperatureBalancedSampler(
            cohort_patient_scan_indices=self._train_patient_scan_indices,
            batch_size=self.batch_size,
            tau=self.tau,
            seed=self.seed,
        )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "call setup() first"
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "call setup() first"
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
        )
