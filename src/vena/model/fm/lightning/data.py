"""HDF5-backed dataset + LightningDataModule for FM training.

Reads ``UCSFPDGM_latents.h5`` (schema 0.1.0, produced by
``vena.data.h5.ucsf_pdgm.latent_domain.convert``). Per-patient single-chunk
reads keep streaming cheap.

Each item is a dict::

    {
        "patient_id": str,
        "z_t1pre": Tensor(4, 60, 60, 40),
        "z_t2": Tensor(4, 60, 60, 40),
        "z_flair": Tensor(4, 60, 60, 40),
        "z_t1c": Tensor(4, 60, 60, 40),  # target
        "m_wt": Tensor(1, 60, 60, 40),  # binary WT mask
    }

The WT mask is derived from ``masks/tumor_latent[i]`` (3 soft NETC/ED/ET maps)
by ``m_wt = (clip(c0+c1+c2, 0, 1) >= 0.5).float()`` per proposal §2.2.

Additional modalities (ADC, SWI) or priors (vessel, perfusion) are loaded on
demand: pass their names through ``extra_latents`` / ``extra_priors`` and the
dataset will produce the matching keys (``z_<name>`` / ``prior_<name>``).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MissingFoldSplitError(RuntimeError):
    """Raised when a cohort's latent H5 lacks the requested CV fold splits.

    Surfaces the cohort name, file path, requested fold, and the splits
    actually present so the operator can either regenerate the H5 (re-run the
    cohort's image-domain converter) or re-copy the canonical artifact from
    a host that has the correct schema.
    """


def _assert_cohort_splits_present(
    f: h5py.File,
    *,
    cohort_name: str,
    latent_h5: Path,
    fold: int,
) -> None:
    """Probe a cohort H5 for the splits required by the FM training data path.

    Raises MissingFoldSplitError naming the cohort, file, requested fold, and
    available alternatives. Without this probe, the operator sees only a raw
    ``h5py KeyError: 'Unable to synchronously open object (component not
    found)'`` traced to a generic h5py line — useless for diagnosis on HPC.
    """
    required = [
        f"splits/cv/fold_{fold}/train",
        f"splits/cv/fold_{fold}/val",
        "splits/test",
    ]
    missing = [k for k in required if k not in f]
    if not missing:
        return

    available_folds: list[str] = []
    if "splits/cv" in f:
        try:
            available_folds = sorted(f["splits/cv"].keys())
        except (AttributeError, TypeError):
            available_folds = []
    splits_root = sorted(f["splits"].keys()) if "splits" in f else []
    raise MissingFoldSplitError(
        f"cohort {cohort_name!r}: latent_h5={latent_h5} is missing "
        f"{missing} (requested fold={fold}). Splits group contains "
        f"{splits_root!r}; available CV folds: {available_folds!r}. "
        f"Re-run the cohort's image-domain converter to write "
        f"'splits/cv/fold_K/{{train,val}}' or replace this H5 with the "
        f"canonical artifact (likely a stale upload predating the cv-splits "
        f"schema)."
    )


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
        transform: AugmentationPipeline | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"latents H5 not found: {self.h5_path}")
        self.patient_ids: list[str] = list(patient_ids)
        self.latents: tuple[str, ...] = tuple(latents)
        self.wt_threshold = float(wt_threshold)
        self.extra_priors: tuple[str, ...] = tuple(extra_priors or ())
        # Optional latent-space augmentation pipeline. When set, the dataset
        # invokes it at the end of ``_read_one`` so augmentation runs inside
        # the DataLoader worker (CPU-parallel, no GPU blocking).
        self.transform = transform
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

        if self.transform is not None:
            out, _ = self.transform(out)

        return out

    def __getstate__(self) -> dict:
        # Drop the open H5 handle so the dataset survives DataLoader worker pickling.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_idx_by_id"] = None
        return state


# NOTE: ``LatentH5DataModule`` was removed in the pre-long-run hardening pass.
# All training now flows through :class:`MultiCohortLatentDataModule`. To run
# a single-cohort experiment, point ``data.corpus_registry`` at a registry
# JSON listing only that cohort (see ``routines/fm/train/configs/corpus/``).
# Old YAML configs carrying ``data.latents_h5`` raise a clear
# ``DataConfigError`` at config load time — see ``_DataCfg`` in
# ``routines/fm/train/engine.py``.


class OfflineAugmentedLatentH5Dataset(Dataset):
    """Train-only dataset that draws a variant per ``__getitem__``.

    Wraps a clean :class:`LatentH5Dataset` (the v0 source) together with a
    per-cohort aug-latent H5 (the v1..vK bank produced by
    :mod:`routines.offline_aug.maisi`). Each ``__getitem__`` call:

    1. Samples a variant tag from ``variant_weights`` (e.g. ``v0`` 20%,
       ``v1`` 20%, ..., ``v4`` 20%).
    2. For ``v0`` reads from the clean H5 via the wrapped
       :class:`LatentH5Dataset` (so the online transform pipeline still
       fires on top — flip/translate are still applied).
    3. For ``v1``..``vK`` reads the row in the aug-latent H5 matching
       ``(patient_id, variant)``; runs the same online transform on top.
    4. Writes ``out["_aug_variant"] = variant`` so
       :class:`vena.data.augment.online.VariantTracker` can count.

    Val and test datasets are NOT wrapped: only the train DataLoader sees
    augmented data, by construction.

    Parameters
    ----------
    clean_h5_path : Path | str
        Cohort's clean ``<COHORT>_latents.h5``.
    aug_h5_path : Path | str
        Cohort's augmented ``<COHORT>_latents_aug.h5``.
    patient_ids : Sequence[str]
        Train scan IDs for this cohort + fold.
    variant_weights : dict[str, float]
        Sampling probabilities for ``{"v0", "v1", ...}``. Normalised on
        construction; missing variants get weight 0.
    latents, wt_threshold, extra_priors, transform : forwarded to
        :class:`LatentH5Dataset` for the v0 read.

    Raises
    ------
    KeyError
        If ``variant_weights`` references a non-``v0`` variant that the
        aug-H5 does not provide.
    """

    def __init__(
        self,
        clean_h5_path: Path | str,
        aug_h5_path: Path | str,
        patient_ids: Sequence[str],
        variant_weights: dict[str, float],
        latents: Sequence[str] = LatentH5Dataset.DEFAULT_LATENTS,
        wt_threshold: float = 0.5,
        extra_priors: Sequence[str] | None = None,
        transform=None,  # AugmentationPipeline | None — string-annotated to skip import
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.aug_h5_path = Path(aug_h5_path)
        if not self.aug_h5_path.is_file():
            raise FileNotFoundError(f"aug-latent H5 not found: {self.aug_h5_path}")
        self.patient_ids: list[str] = list(patient_ids)
        # Hot-path lookup for the fallback in _read_aug; avoids O(N) list.index.
        self._pid_to_idx: dict[str, int] = {p: i for i, p in enumerate(self.patient_ids)}
        # The clean reader handles v0 (and also owns the online transform).
        self._clean = LatentH5Dataset(
            clean_h5_path,
            patient_ids,
            latents=latents,
            wt_threshold=wt_threshold,
            extra_priors=extra_priors,
            transform=transform,
        )
        # For v1..vK we still want the online transform; do it after the
        # aug read via the *same* pipeline. We share self._clean.transform.
        self._latents: tuple[str, ...] = tuple(latents)
        self._wt_threshold = float(wt_threshold)
        self._extra_priors: tuple[str, ...] = tuple(extra_priors or ())

        # Normalise + validate variant weights.
        if not variant_weights:
            raise ValueError("variant_weights must be non-empty")
        weights = {k: float(v) for k, v in variant_weights.items() if float(v) > 0}
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError(f"all variant weights are zero: {variant_weights}")
        self._variant_weights: dict[str, float] = {k: v / total for k, v in weights.items()}

        # Lazy aug-H5 state, populated on first read.
        self._aug_h5: h5py.File | None = None
        self._aug_idx: dict[tuple[str, str], int] | None = None
        self._seed = int(seed)

    def _ensure_aug_open(self) -> h5py.File:
        if self._aug_h5 is None:
            self._aug_h5 = h5py.File(self.aug_h5_path, "r", swmr=True)
            ids_arr = np.asarray(self._aug_h5["ids"][:], dtype=object)
            variants_arr = np.asarray(self._aug_h5["variants"][:], dtype=object)
            self._aug_idx = {}
            for row, (pid_v, var_v) in enumerate(zip(ids_arr, variants_arr)):
                pid_str = pid_v.decode() if isinstance(pid_v, (bytes, bytearray)) else str(pid_v)
                var_str = var_v.decode() if isinstance(var_v, (bytes, bytearray)) else str(var_v)
                self._aug_idx[(pid_str, var_str)] = row
            # Sanity: every variant requested in weights (except v0) must
            # appear in the aug-H5 for at least one patient.
            seen_variants = {key[1] for key in self._aug_idx}
            requested = set(self._variant_weights) - {"v0"}
            missing = requested - seen_variants
            if missing:
                raise KeyError(
                    f"variant_weights requested {missing!r} but aug-H5 "
                    f"{self.aug_h5_path} has variants {sorted(seen_variants)}"
                )
        return self._aug_h5

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.patient_ids[idx]
        # Per-call deterministic seed. Python's built-in `hash()` is
        # PYTHONHASHSEED-randomised, so it would give a different variant draw
        # per process restart — kill that with a stable SHA-256-based digest.
        digest = hashlib.sha256(f"{pid}:{idx}".encode()).digest()[:8]
        seed_mix = int.from_bytes(digest, "big", signed=False)
        rng = random.Random(self._seed ^ seed_mix)
        variants = list(self._variant_weights)
        weights = [self._variant_weights[v] for v in variants]
        variant = rng.choices(variants, weights=weights, k=1)[0]
        if variant == "v0":
            out = self._clean[idx]
        else:
            out = self._read_aug(pid, variant)
            if self._clean.transform is not None:
                out, _ = self._clean.transform(out)
        out["_aug_variant"] = variant
        return out

    def _read_aug(self, pid: str, variant: str) -> dict[str, torch.Tensor | str]:
        h5 = self._ensure_aug_open()
        assert self._aug_idx is not None
        key = (pid, variant)
        row = self._aug_idx.get(key)
        if row is None:
            # Fall back to v0 — this can happen if the dedup allowlist
            # differs between bank-build and training. Log once.
            logger.warning(
                "aug-H5 %s has no row for (patient=%r, variant=%r); falling back to clean v0",
                self.aug_h5_path,
                pid,
                variant,
            )
            return self._clean[self._pid_to_idx[pid]]

        out: dict[str, torch.Tensor | str] = {"patient_id": pid}
        for name in self._latents:
            arr = h5[f"latents/{name}"][row]
            out[f"z_{name}"] = torch.from_numpy(np.ascontiguousarray(arr)).float()
        tumor_lat = h5["masks/tumor_latent"][row]
        soft_union = np.clip(tumor_lat.sum(axis=0, keepdims=True), 0.0, 1.0)
        m_wt = (soft_union >= self._wt_threshold).astype(np.float32)
        out["m_wt"] = torch.from_numpy(np.ascontiguousarray(m_wt))
        # extra_priors are not stored in the aug-H5 (no priors group); skip.
        return out

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_aug_h5"] = None
        state["_aug_idx"] = None
        return state


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
            raise IndexError(f"global_idx {global_idx} out of range [0, {len(self)})")
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
        total_scans = sum(sum(len(scans) for scans in patients) for patients in self._cps)

        # Cohort probabilities via temperature scaling.
        nc = np.array(n_patients_per_cohort, dtype=np.float64)
        if self.tau == 0.0:
            weights = np.ones(n_cohorts, dtype=np.float64)
        else:
            weights = nc**self.tau
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
        train_transform: AugmentationPipeline | None = None,
        dedup_allowlists: dict[str, set[str]] | None = None,
        use_offline_augmented_data: bool = False,
        variant_weights: dict[str, float] | None = None,
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
        # Augmentation runs on training samples only; validation / test never
        # see augmented data so metrics remain comparable across runs.
        self.train_transform = train_transform
        # Per-cohort patient-ID allow-list produced by the cohort_dedup
        # preflight (schema v1.0). When set, the DataModule intersects
        # train/val/test patient keys with the cohort's allow-list before
        # CSR-expansion to scan IDs, so the DataLoader never sees a
        # dropped patient. ``None`` disables the filter entirely.
        self._dedup_allowlists = dedup_allowlists
        # Offline-augmented data: when True the train dataset for each cv
        # cohort is wrapped in OfflineAugmentedLatentH5Dataset, which draws
        # variant ∈ {v0..vK} per __getitem__ with `variant_weights` and
        # reads either from the clean latent H5 (v0) or from
        # `cohort.latent_aug_h5` (v1..vK). Val and test never use augmented
        # data — they stay on the clean H5.
        self.use_offline_augmented_data = bool(use_offline_augmented_data)
        if variant_weights is None:
            variant_weights = {
                "v0": 0.2,
                "v1": 0.2,
                "v2": 0.2,
                "v3": 0.2,
                "v4": 0.2,
            }
        self.variant_weights: dict[str, float] = dict(variant_weights)

        self._train_ds: MultiCohortLatentDataset | None = None
        self._val_ds: MultiCohortLatentDataset | None = None
        self._test_ds: MultiCohortLatentDataset | None = None
        # Per-cohort patient→[global_indices] for the sampler.
        self._train_patient_scan_indices: list[list[list[int]]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_ids(ds: h5py.Dataset) -> list[str]:
        return [b.decode() if isinstance(b, bytes) else str(b) for b in ds[:]]

    @staticmethod
    def _expand_patients_to_scans(
        offsets: np.ndarray,
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

    def _open_cohort_h5(self, latent_h5: Path) -> h5py.File:
        return h5py.File(latent_h5, "r", swmr=True)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:

        train_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        val_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        test_cohort_datasets: list[tuple[str, LatentH5Dataset]] = []
        self._train_patient_scan_indices = []

        rng_subsample = np.random.default_rng(self.seed)

        # --- cv cohorts (contribute train/val/test) ---
        for cohort in self.registry.cv_cohorts():
            with self._open_cohort_h5(cohort.latent_h5) as f:
                _assert_cohort_splits_present(
                    f,
                    cohort_name=cohort.name,
                    latent_h5=cohort.latent_h5,
                    fold=self.fold,
                )
                ids = self._decode_ids(f["ids"])
                offsets = f["patients/offsets"][:]
                keys = self._decode_ids(f["patients/keys"])
                train_patient_keys = self._decode_ids(f[f"splits/cv/fold_{self.fold}/train"])
                val_patient_keys = self._decode_ids(f[f"splits/cv/fold_{self.fold}/val"])
                test_patient_keys = self._decode_ids(f["splits/test"])

            # Cohort deduplication. Keep only patient IDs in the preflight's
            # allow-list. Applied BEFORE the CSR expansion so dropped patients
            # never reach the per-cohort LatentH5Dataset nor the sampler.
            if self._dedup_allowlists is not None:
                allow = self._dedup_allowlists.get(cohort.name)
                if allow is None:
                    raise RuntimeError(
                        f"dedup allow-list missing for cohort {cohort.name!r}; "
                        f"the cohort_dedup preflight must cover every cv cohort "
                        f"in the registry (got cohorts: "
                        f"{sorted(self._dedup_allowlists)})"
                    )
                n_train_before = len(train_patient_keys)
                n_val_before = len(val_patient_keys)
                n_test_before = len(test_patient_keys)
                train_patient_keys = [p for p in train_patient_keys if p in allow]
                val_patient_keys = [p for p in val_patient_keys if p in allow]
                test_patient_keys = [p for p in test_patient_keys if p in allow]
                logger.info(
                    "%s: dedup filter kept train=%d/%d, val=%d/%d, test=%d/%d",
                    cohort.name,
                    len(train_patient_keys),
                    n_train_before,
                    len(val_patient_keys),
                    n_val_before,
                    len(test_patient_keys),
                    n_test_before,
                )

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
            val_scan_ids, _ = self._expand_patients_to_scans(offsets, keys, ids, val_patient_keys)
            test_scan_ids, _ = self._expand_patients_to_scans(offsets, keys, ids, test_patient_keys)

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

            # Build per-cohort datasets. Augmentation is attached to the
            # training dataset only — validation / test datasets always read
            # the raw latent so cross-run metrics stay comparable.
            if self.use_offline_augmented_data:
                if cohort.latent_aug_h5 is None:
                    raise RuntimeError(
                        f"use_offline_augmented_data=True but cohort "
                        f"{cohort.name!r} has no latent_aug_h5 in the registry"
                    )
                train_ds = OfflineAugmentedLatentH5Dataset(
                    clean_h5_path=cohort.latent_h5,
                    aug_h5_path=cohort.latent_aug_h5,
                    patient_ids=train_scan_ids,
                    variant_weights=self.variant_weights,
                    transform=self.train_transform,
                    seed=self.seed,
                )
            else:
                train_ds = LatentH5Dataset(
                    cohort.latent_h5,
                    train_scan_ids,
                    transform=self.train_transform,
                )
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

            # Forward-compatible: apply the dedup filter to test-only cohorts
            # too when a per-cohort allow-list happens to be present. Missing
            # entries are tolerated here (test-only cohorts are out of scope
            # for the current preflight) — no warning needed.
            if self._dedup_allowlists is not None and cohort.name in self._dedup_allowlists:
                allow = self._dedup_allowlists[cohort.name]
                n_before = len(all_patient_keys)
                all_patient_keys = [p for p in all_patient_keys if p in allow]
                logger.info(
                    "%s (test-only): dedup filter kept %d/%d",
                    cohort.name,
                    len(all_patient_keys),
                    n_before,
                )

            all_scan_ids, _ = self._expand_patients_to_scans(offsets, keys, ids, all_patient_keys)
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
                [cohort_global_offset + local for local in patient_locals] for patient_locals in p2l
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
