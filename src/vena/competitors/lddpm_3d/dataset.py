"""Per-cohort and multi-cohort latent-H5 datasets for 3D-LDDPM.

The dataset is deterministic (no augmentation) by contract: VENA owns the
augmentation regime, the competitor's loader does not. Each ``__getitem__``
call returns the same tensors for the same index.

This is **identical** in shape and contract to ``T1CRFlowLatentDataset``
(same latent H5 schema, same conditioning, same target). The wrapper is
duplicated rather than imported from the sibling because skill
anti-pattern 7 forbids reaching across ``vena.competitors.<other>.*``
boundaries — competitor leaves stay independent.

Carry-overs from the reference implementations
----------------------------------------------
- **Longitudinal patient-id resolver.** BraTS-GLI / LUMIERE store scan-level
  ids in ``/ids`` (``BraTS-GLI-00000-000``) but patient-level ids in splits
  (``BraTS-GLI-00000``). Prefix-match recovers both scans per patient.
- **Flat-splits fallback.** REMBRANDT (N=63) uses ``splits/{train,val,test}``
  with no ``splits/cv/fold_<k>``. The reader prefers k-fold, falls back to flat.
- **Lazy h5 + ``__getstate__``.** h5py handles are not picklable across
  ``num_workers > 0``; we drop them at pickle time and reopen in ``_open``.
  Open in plain ``"r"`` mode (no SWMR — multi-cohort ``ConcatDataset`` over
  several H5s deadlocks the no-op SWMR handshake on some h5py builds).

Citation
--------
Eidex *et al.* 2025, "An Efficient 3D Latent Diffusion Model for T1-contrast
Enhanced MRI Generation," arXiv:2509.24194.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Raised on malformed latent H5 or invalid split selection."""


def _decode_ids(arr: np.ndarray) -> list[str]:
    """Decode an H5 vlen-str dataset into a list of Python ``str``."""
    out: list[str] = []
    for s in arr:
        if isinstance(s, bytes):
            out.append(s.decode("utf-8"))
        else:
            out.append(str(s))
    return out


class LDDPM3DLatentDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Per-cohort latent-H5 reader for 3D-LDDPM training.

    Parameters
    ----------
    latent_h5:
        Path to a VENA-produced latent H5 (schema 2.0.0). Must contain
        ``latents/<name>`` for every name in ``input_latents`` and
        ``target_latent``.
    fold:
        CV fold (0..4) for non-test phases.
    phase:
        ``"train"``, ``"val"``, or ``"test"``.
    input_latents:
        Tuple of conditioning latent names. Default ``("t1pre", "flair")``
        — paper-faithful for 3D-LDDPM (T1n + T2-FLAIR; see ``UPSTREAM.md``
        "Scope of use").
    target_latent:
        The target. Default ``"t1c"``.
    max_patients:
        If set, only the first ``N`` patients of the resolved split are used
        (smoke runs). Cap is applied **after** longitudinal expansion to keep
        ≥1 scan per requested patient.
    """

    def __init__(
        self,
        latent_h5: Path | str,
        fold: int,
        phase: str,
        input_latents: Sequence[str] = ("t1pre", "flair"),
        target_latent: str = "t1c",
        max_patients: int | None = None,
    ) -> None:
        self.latent_h5 = Path(latent_h5)
        if not self.latent_h5.is_file():
            raise DatasetError(f"latent H5 not found at {self.latent_h5}")
        if phase not in {"train", "val", "test"}:
            raise DatasetError(f"phase must be one of train/val/test, got {phase!r}")

        self.fold = fold
        self.phase = phase
        self.input_latents = tuple(input_latents)
        self.target_latent = target_latent

        with h5py.File(self.latent_h5, "r") as f:
            all_ids = _decode_ids(np.asarray(f["ids"]))
            if phase == "test":
                candidates = ["splits/test"]
            else:
                # Prefer k-fold CV splits; fall back to the flat
                # ``splits/<phase>`` schema used by small cohorts (REMBRANDT).
                candidates = [
                    f"splits/cv/fold_{fold}/{phase}",
                    f"splits/{phase}",
                ]
            key = next((c for c in candidates if c in f), None)
            if key is None:
                raise DatasetError(f"none of {candidates} present in {self.latent_h5}")
            split_ids = _decode_ids(np.asarray(f[key]))

            # Validate the requested latent datasets exist.
            for name in (*self.input_latents, self.target_latent):
                if f"latents/{name}" not in f:
                    raise DatasetError(
                        f"latents/{name} missing in {self.latent_h5}; "
                        f"available: {sorted(f['latents'].keys())}"
                    )

        # Longitudinal id resolution (BraTS-GLI / LUMIERE) and prefix match.
        id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
        resolved_indices: list[int] = []
        resolved_ids: list[str] = []
        missing: list[str] = []
        for pid in split_ids:
            if pid in id_to_idx:
                resolved_indices.append(id_to_idx[pid])
                resolved_ids.append(pid)
                continue
            matched = False
            prefix_dash = f"{pid}-"
            prefix_uscr = f"{pid}_"
            for full_id, idx in id_to_idx.items():
                if full_id.startswith(prefix_dash) or full_id.startswith(prefix_uscr):
                    resolved_indices.append(idx)
                    resolved_ids.append(full_id)
                    matched = True
            if not matched:
                missing.append(pid)
        if missing:
            raise DatasetError(
                f"split {key!r} references {len(missing)} ids absent from /ids "
                f"(exact and prefix match both failed; e.g. {missing[:3]})"
            )
        if max_patients is not None:
            resolved_ids = resolved_ids[:max_patients]
            resolved_indices = resolved_indices[:max_patients]

        self.patient_ids: list[str] = resolved_ids
        self.patient_indices: list[int] = resolved_indices

        logger.info(
            "LDDPM3DLatentDataset[%s/fold%d]: %d patients, cond=%s → %s",
            phase,
            fold,
            len(self.patient_ids),
            self.input_latents,
            self.target_latent,
        )

        self._h5: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._h5 is None:
            # Plain "r" mode — no SWMR (some h5py builds deadlock on the no-op
            # handshake under multiprocessing + ConcatDataset).
            self._h5 = h5py.File(self.latent_h5, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.patient_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        pidx = self.patient_indices[i]
        pid = self.patient_ids[i]
        f = self._open()

        out: dict[str, torch.Tensor | str] = {"patient_id": pid}
        for name in self.input_latents:
            arr = np.asarray(f[f"latents/{name}"][pidx], dtype=np.float32)
            out[f"z_{name}"] = torch.from_numpy(arr)
        tgt = np.asarray(f[f"latents/{self.target_latent}"][pidx], dtype=np.float32)
        out[f"z_{self.target_latent}"] = torch.from_numpy(tgt)
        return out

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None  # h5py handles are not picklable across workers.
        return state


# ---------------------------------------------------------------------------
# Multi-cohort wrapper — mirrors VENA's training corpus
# ---------------------------------------------------------------------------


def _load_corpus_registry(path: Path | str) -> list[dict]:
    """Read a VENA corpus_registry JSON and return the cohort list."""
    with Path(path).open("r") as f:
        registry = json.load(f)
    if "cohorts" not in registry:
        raise DatasetError(f"corpus registry {path} missing 'cohorts'")
    return registry["cohorts"]


class MultiCohortLDDPM3DLatentDataset(Dataset[dict[str, torch.Tensor | str]]):
    """ConcatDataset over per-cohort ``LDDPM3DLatentDataset`` instances.

    Reads a VENA corpus-registry JSON (the same file consumed by
    ``MultiCohortLatentDataModule``) and assembles one per-cohort dataset for
    every entry with ``role == role_filter`` (default ``"cv"``). Per-cohort
    datasets are concatenated via ``torch.utils.data.ConcatDataset`` so the
    DataLoader sees a flat patient index across the cohort union.

    A cohort whose split is empty for the requested ``fold`` / ``phase`` is
    skipped with a WARNING. A cohort whose ``latent_h5`` is unreachable is also
    skipped — this is the contract that lets server-3 / loginexa / Picasso
    share one corpus JSON when each platform mirrors only a subset locally.
    """

    def __init__(
        self,
        corpus_registry: Path | str,
        fold: int,
        phase: str,
        input_latents: Sequence[str] = ("t1pre", "flair"),
        target_latent: str = "t1c",
        max_patients_per_cohort: int | None = None,
        role_filter: str = "cv",
        path_overrides: dict[str, Path | str] | None = None,
    ) -> None:
        self.corpus_registry = Path(corpus_registry)
        self.fold = fold
        self.phase = phase
        self.input_latents = tuple(input_latents)
        self.target_latent = target_latent
        self.max_patients_per_cohort = max_patients_per_cohort
        self.role_filter = role_filter
        self.path_overrides = {k: Path(v) for k, v in (path_overrides or {}).items()}

        cohorts = _load_corpus_registry(self.corpus_registry)
        datasets: list[LDDPM3DLatentDataset] = []
        self.cohort_names: list[str] = []
        self.cohort_sizes: list[int] = []
        for entry in cohorts:
            name = entry["name"]
            if entry.get("role") != role_filter:
                continue
            if "latent_h5" not in entry:
                logger.warning(
                    "MultiCohortLDDPM3DLatentDataset: skipping cohort %s — "
                    "no 'latent_h5' field in registry entry",
                    name,
                )
                continue
            h5 = self.path_overrides.get(name, Path(entry["latent_h5"]))
            if not h5.is_file():
                logger.warning(
                    "MultiCohortLDDPM3DLatentDataset: skipping cohort %s — latent H5 missing at %s",
                    name,
                    h5,
                )
                continue
            try:
                ds = LDDPM3DLatentDataset(
                    latent_h5=h5,
                    fold=fold,
                    phase=phase,
                    input_latents=input_latents,
                    target_latent=target_latent,
                    max_patients=max_patients_per_cohort,
                )
            except DatasetError as exc:
                logger.warning(
                    "MultiCohortLDDPM3DLatentDataset: skipping cohort %s — %s",
                    name,
                    exc,
                )
                continue
            if len(ds) == 0:
                logger.warning(
                    "MultiCohortLDDPM3DLatentDataset: cohort %s has 0 patients "
                    "for fold=%d phase=%s — skipped",
                    name,
                    fold,
                    phase,
                )
                continue
            datasets.append(ds)
            self.cohort_names.append(name)
            self.cohort_sizes.append(len(ds))

        if not datasets:
            raise DatasetError(
                f"no usable cohorts in {self.corpus_registry} (fold={fold}, "
                f"phase={phase}, role={role_filter}). Override paths via "
                f"path_overrides if your platform mirrors data at non-canonical "
                f"locations."
            )

        self._concat = ConcatDataset(datasets)
        self._datasets = datasets
        logger.info(
            "MultiCohortLDDPM3DLatentDataset[%s/fold%d]: %d cohorts, "
            "%d total patients (per-cohort: %s)",
            phase,
            fold,
            len(datasets),
            len(self._concat),
            ", ".join(f"{n}={s}" for n, s in zip(self.cohort_names, self.cohort_sizes)),
        )

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        return self._concat[i]
