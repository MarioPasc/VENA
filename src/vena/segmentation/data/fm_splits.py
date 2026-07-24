"""FM-split resolution for segmentation training (task 14).

Mirrors ``MultiCohortLatentDataModule.setup`` split logic exactly so the
segmenter trains and validates on the same patient populations as the generator:

- **CV cohorts** (``role="cv"``): read ``splits/cv/fold_{fm_fold}/{train,val}``
  and ``splits/test`` from the image H5.  Dedup allow-list is **required**
  (raises :class:`~vena.segmentation.exceptions.SegDataError` on missing entry).
- **test_only cohorts**: read all ``patients/keys``; dedup allow-list is
  tolerated when absent (mirror FM DataModule forward-compatible behaviour).

The ``patient_to_scans`` CSR expansion is essential for longitudinal cohorts
(e.g. LUMIERE, 91 patients / 599 scans): the generator DataModule operates on
*scan* IDs, so the segmenter must expand patient keys to scan IDs before
constructing :class:`~vena.segmentation.data.dataset.SegImageDataset`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vena.segmentation.exceptions import SegDataError

if TYPE_CHECKING:
    from vena.segmentation.config import DataConfig
    from vena.segmentation.data.kfold import FoldPlan

logger = logging.getLogger(__name__)

__all__ = [
    "CohortSplit",
    "FmSplitResolution",
    "resolve_fm_splits",
    "write_splits_json",
]

_PRODUCER = "vena.segmentation.data.fm_splits:1.0.0"
_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSplit:
    """Per-cohort view of the FM patient splits.

    Attributes
    ----------
    name:
        Cohort name from the registry (e.g. ``"UCSF-PDGM"``).
    role:
        ``"cv"`` or ``"test_only"``.
    image_h5:
        Absolute path that actually resolved (absolute registry path if it
        exists; ``image_h5_root / filename`` fallback otherwise).
    train_patients:
        Sorted tuple of patient keys from ``splits/cv/fold_{fm_fold}/train``
        after dedup filtering.  Empty for test_only cohorts.
    val_patients:
        Sorted tuple of patient keys from ``splits/cv/fold_{fm_fold}/val``
        after dedup filtering.  Empty for test_only cohorts.
    test_patients:
        Sorted tuple of patient keys from ``splits/test`` (cv) or all
        ``patients/keys`` (test_only) after dedup filtering.
    n_patients_h5:
        ``len(patients/keys)`` before any dedup filter.
    n_kept_after_dedup:
        ``len(train_patients) + len(val_patients) + len(test_patients)``
        after dedup filter.  Equals ``n_patients_h5`` when dedup is disabled.
    """

    name: str
    role: str  # "cv" | "test_only"
    image_h5: Path  # the path that actually resolved
    train_patients: tuple[str, ...]
    val_patients: tuple[str, ...]
    test_patients: tuple[str, ...]
    n_patients_h5: int  # len(patients/keys) before dedup
    n_kept_after_dedup: int


@dataclass(frozen=True)
class FmSplitResolution:
    """Complete FM split resolution for all registry cohorts.

    Attributes
    ----------
    fm_fold:
        Which FM cross-validation fold was used to read the generator splits.
    corpus_registry:
        Absolute path to the registry JSON that was read.
    corpus_registry_sha256:
        SHA-256 of the registry file (provenance).
    dedup_decision_path:
        Path to the cohort_dedup preflight ``decision.json``, or ``None``.
    per_cohort:
        One :class:`CohortSplit` per cohort in registry order.
    patient_to_scans:
        Mapping from patient key → sorted tuple of scan IDs (from H5 ``ids``
        CSR expansion via ``patients/keys`` + ``patients/offsets``).
    patient_to_cohort:
        Mapping from patient key → cohort name.  Useful for per-cohort metric
        breakdowns.
    """

    fm_fold: int
    corpus_registry: Path
    corpus_registry_sha256: str
    dedup_decision_path: Path | None
    per_cohort: tuple[CohortSplit, ...]
    patient_to_scans: Mapping[str, tuple[str, ...]]
    patient_to_cohort: Mapping[str, str]

    def fm_splits(self) -> dict[str, list[str]]:
        """Union of splits across all cohorts → sorted patient keys.

        Returns
        -------
        dict
            Keys ``"train"``, ``"val"``, ``"test"`` mapping to sorted lists
            of patient keys (union across all cohorts).
        """
        train: set[str] = set()
        val: set[str] = set()
        test: set[str] = set()
        for cs in self.per_cohort:
            train.update(cs.train_patients)
            val.update(cs.val_patients)
            test.update(cs.test_patients)
        return {
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(test),
        }

    def scans_for(self, patient_ids: Sequence[str]) -> list[str]:
        """Expand patient keys to scan IDs via the CSR-derived map.

        Parameters
        ----------
        patient_ids:
            Sequence of patient keys.

        Returns
        -------
        list[str]
            Flat list of scan IDs, in patient order.  A patient with N scans
            contributes N entries.
        """
        scans: list[str] = []
        for pid in patient_ids:
            scans.extend(self.patient_to_scans.get(pid, ()))
        return scans

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        Returns
        -------
        dict
            All fields as JSON-serialisable Python objects.
        """
        return {
            "fm_fold": self.fm_fold,
            "corpus_registry": str(self.corpus_registry),
            "corpus_registry_sha256": self.corpus_registry_sha256,
            "dedup_decision_path": (
                str(self.dedup_decision_path) if self.dedup_decision_path else None
            ),
            "per_cohort": {
                cs.name: {
                    "role": cs.role,
                    "image_h5": str(cs.image_h5),
                    "n_patients_h5": cs.n_patients_h5,
                    "n_kept_after_dedup": cs.n_kept_after_dedup,
                    "train_patients": list(cs.train_patients),
                    "val_patients": list(cs.val_patients),
                    "test_patients": list(cs.test_patients),
                }
                for cs in self.per_cohort
            },
            "patient_to_scans": {pid: list(scans) for pid, scans in self.patient_to_scans.items()},
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_ids(raw: Any) -> list[str]:
    """Decode h5py dataset elements to plain Python strings."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _resolve_h5_path(
    raw_h5: str,
    image_h5_root: Path,
    cohort_name: str,
) -> Path | None:
    """Resolve the H5 path for one cohort.

    Strategy: try the registry's absolute path first; if not found, fall back
    to ``image_h5_root / basename``.  Mirrors the fix for Bug 2 (path
    resolution flattens nested layouts) in :func:`_build_id_index`.

    Parameters
    ----------
    raw_h5:
        ``"image_h5"`` string from the registry entry.
    image_h5_root:
        Fallback directory (``cfg.image_h5_root``).
    cohort_name:
        Cohort name for log messages.

    Returns
    -------
    Path | None
        Resolved path if the file exists; ``None`` otherwise.
    """
    abs_path = Path(raw_h5)
    if abs_path.exists():
        logger.debug("Cohort '%s': H5 resolved via absolute path: %s", cohort_name, abs_path)
        return abs_path

    fallback = image_h5_root / abs_path.name
    if fallback.exists():
        logger.debug(
            "Cohort '%s': H5 resolved via image_h5_root fallback: %s", cohort_name, fallback
        )
        return fallback

    return None


def _expand_patients_to_scans(
    offsets: Any,
    keys: list[str],
    ids: list[str],
    patient_keys: list[str],
) -> list[str]:
    """Expand a list of patient keys to their scan IDs using CSR offsets.

    Parameters
    ----------
    offsets:
        ``patients/offsets`` array from the H5 — int-like, length ``n_patients + 1``.
    keys:
        ``patients/keys`` decoded strings, length ``n_patients``.
    ids:
        ``ids`` decoded strings (scan level), length ``n_scans``.
    patient_keys:
        Subset of ``keys`` to expand.

    Returns
    -------
    list[str]
        Flat list of scan IDs in the order ``patient_keys`` are given.
    """
    key_to_idx: dict[str, int] = {k: i for i, k in enumerate(keys)}
    scans: list[str] = []
    for pk in patient_keys:
        idx = key_to_idx.get(pk)
        if idx is None:
            logger.warning("Patient key '%s' not found in patients/keys — skipping.", pk)
            continue
        o0, o1 = int(offsets[idx]), int(offsets[idx + 1])
        scans.extend(ids[o0:o1])
    return scans


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_sha(ref_path: Path) -> str | None:
    """Try to get the current HEAD short SHA from the repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ref_path.parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_dedup_allowlists(
    dedup_decision_path: Path,
) -> dict[str, set[str]]:
    """Load per-cohort allow-lists from the cohort_dedup preflight JSON.

    Parameters
    ----------
    dedup_decision_path:
        Path to ``artifacts/preflights/cohort_dedup/LATEST/decision.json``.

    Returns
    -------
    dict[str, set[str]]
        ``{cohort_name: frozenset(kept_patient_ids)}``.

    Raises
    ------
    SegDataError
        If the file is missing or malformed.
    """
    if not dedup_decision_path.exists():
        raise SegDataError(
            f"dedup_decision_path not found: {dedup_decision_path}. "
            "Run the cohort_dedup preflight or set dedup_decision_path=None."
        )
    try:
        raw = json.loads(dedup_decision_path.read_text())
    except json.JSONDecodeError as exc:
        raise SegDataError(
            f"dedup_decision_path is not valid JSON: {dedup_decision_path}: {exc}"
        ) from exc

    cohorts_raw = raw.get("cohorts", {})
    if not isinstance(cohorts_raw, dict):
        raise SegDataError(
            f"dedup_decision_path JSON missing 'cohorts' dict: {dedup_decision_path}"
        )

    return {name: set(entry.get("kept_patient_ids", [])) for name, entry in cohorts_raw.items()}


def _read_cv_splits(
    hf: Any,
    h5_path: Path,
    cohort_name: str,
    fm_fold: int,
) -> tuple[list[str], list[str]]:
    """Read train/val patient keys from an open H5 file, handling two layouts.

    Layout 1 (canonical): ``splits/cv/fold_{fm_fold}/{train,val}``
    Layout 2 (legacy flat — REMBRANDT local): ``splits/{train,val}``

    See ``src/vena/data/h5/shared/splits.py`` (``_FLAT_SPLITS`` constant) for
    the legacy layout reference.

    Parameters
    ----------
    hf:
        Open h5py File object.
    h5_path:
        Path of the H5 file (used in error/warning messages only).
    cohort_name:
        Cohort name for log messages.
    fm_fold:
        Requested FM cross-validation fold index.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(train_patient_keys, val_patient_keys)`` decoded strings.

    Raises
    ------
    SegDataError
        If ``splits/cv`` exists but ``fold_{fm_fold}`` does not (wrong fold
        index), or if neither cv nor flat splits are present.
    """
    cv_group = "splits/cv"
    fold_group = f"{cv_group}/fold_{fm_fold}"

    if cv_group in hf:
        # Canonical layout: splits/cv/fold_{n}/{train,val}
        if fold_group not in hf:
            available_folds = sorted(hf[cv_group].keys())
            raise SegDataError(
                f"H5 '{h5_path}' (cohort '{cohort_name}'): "
                f"'splits/cv' exists but fold '{fold_group}' not found. "
                f"Available folds: {available_folds}. "
                f"Check cfg.fm_fold (requested {fm_fold})."
            )
        train_pkeys = _decode_ids(hf[f"{fold_group}/train"][:])
        val_pkeys = _decode_ids(hf[f"{fold_group}/val"][:])
        return train_pkeys, val_pkeys

    # Legacy flat layout: splits/{train,val}
    if "splits/train" in hf and "splits/val" in hf:
        logger.warning(
            "Cohort '%s' H5 '%s': 'splits/cv' absent; falling back to flat "
            "'splits/{train,val}' layout (legacy pre-2026-06-19 REMBRANDT). "
            "All folds will use the same train/val partition.",
            cohort_name,
            h5_path,
        )
        train_pkeys = _decode_ids(hf["splits/train"][:])
        val_pkeys = _decode_ids(hf["splits/val"][:])
        return train_pkeys, val_pkeys

    raise SegDataError(
        f"H5 '{h5_path}' (cohort '{cohort_name}'): no usable splits found. "
        f"Expected 'splits/cv/fold_{fm_fold}/{{train,val}}' or flat "
        f"'splits/{{train,val}}'. Splits keys present: {list(hf.get('splits', {}).keys())}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_fm_splits(cfg: DataConfig) -> FmSplitResolution:
    """Resolve patient splits for all registry cohorts.

    Reads the corpus registry and the image H5 files to produce a complete
    :class:`FmSplitResolution` that mirrors the exact populations used by
    :class:`~vena.model.fm.lightning.data.MultiCohortLatentDataModule`.

    Parameters
    ----------
    cfg:
        Frozen :class:`~vena.segmentation.config.DataConfig`.  Reads
        ``corpus_registry``, ``image_h5_root``, ``fm_fold``, and
        ``dedup_decision_path``.

    Returns
    -------
    FmSplitResolution
        Complete split resolution for all cohorts.  The returned object's
        :attr:`~FmSplitResolution.patient_to_cohort` mapping is the canonical
        stratification source for K-fold construction; pass it as
        ``cohort_labels`` to :func:`~vena.segmentation.data.kfold.build_fold_plan`
        to replace the heuristic :func:`~vena.segmentation.data.kfold._extract_cohort`
        labels with exact registry cohort names::

            resolution = resolve_fm_splits(cfg)
            plan = build_fold_plan(
                cfg,
                resolution.fm_splits(),
                dedup_duplicates=None,
                cohort_labels=resolution.patient_to_cohort,
            )

    Raises
    ------
    SegDataError
        If the registry is missing or malformed; if a cv cohort's H5 is absent
        or lacks required datasets; or if the dedup allow-list is missing a
        required entry for a cv cohort.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for resolve_fm_splits. Install via: pip install h5py"
        ) from exc

    from vena.data.registry import load_registry

    registry = load_registry(cfg.corpus_registry, require_latents=False)
    registry_sha = _sha256_file(cfg.corpus_registry)

    # Load dedup allow-lists (None when dedup_decision_path is not configured)
    dedup_allowlists: dict[str, set[str]] | None = None
    if cfg.dedup_decision_path is not None:
        dedup_allowlists = _load_dedup_allowlists(cfg.dedup_decision_path)
        logger.info(
            "Loaded dedup allow-lists from %s: %d cohorts covered.",
            cfg.dedup_decision_path,
            len(dedup_allowlists),
        )

    per_cohort: list[CohortSplit] = []
    patient_to_scans: dict[str, tuple[str, ...]] = {}
    patient_to_cohort: dict[str, str] = {}

    # --- cv cohorts ---
    for cohort in registry.cv_cohorts():
        h5_path = _resolve_h5_path(str(cohort.image_h5), cfg.image_h5_root, cohort.name)
        if h5_path is None:
            raise SegDataError(
                f"Image H5 for cv cohort '{cohort.name}' not found. "
                f"Tried: '{cohort.image_h5}' and "
                f"'{cfg.image_h5_root / Path(str(cohort.image_h5)).name}'."
            )

        try:
            with h5py.File(h5_path, "r") as hf:
                ids = _decode_ids(hf["ids"][:])
                keys = _decode_ids(hf["patients/keys"][:])
                offsets = hf["patients/offsets"][:]

                train_pkeys, val_pkeys = _read_cv_splits(hf, h5_path, cohort.name, cfg.fm_fold)
                test_pkeys = _decode_ids(hf["splits/test"][:])

        except SegDataError:
            raise
        except Exception as exc:
            raise SegDataError(
                f"Failed to read cv cohort '{cohort.name}' from '{h5_path}': {exc}"
            ) from exc

        n_patients_h5 = len(keys)

        # Dedup filter — REQUIRED for cv cohorts
        if dedup_allowlists is not None:
            allow = dedup_allowlists.get(cohort.name)
            if allow is None:
                raise SegDataError(
                    f"Dedup allow-list missing for cv cohort '{cohort.name}'. "
                    f"Cohorts covered by dedup preflight: {sorted(dedup_allowlists)}. "
                    "Ensure the cohort_dedup preflight covers every cv cohort."
                )
            n_train_before = len(train_pkeys)
            n_val_before = len(val_pkeys)
            n_test_before = len(test_pkeys)
            train_pkeys = [p for p in train_pkeys if p in allow]
            val_pkeys = [p for p in val_pkeys if p in allow]
            test_pkeys = [p for p in test_pkeys if p in allow]
            logger.info(
                "%s (cv): dedup filter kept train=%d/%d, val=%d/%d, test=%d/%d",
                cohort.name,
                len(train_pkeys),
                n_train_before,
                len(val_pkeys),
                n_val_before,
                len(test_pkeys),
                n_test_before,
            )

        # Build patient→scans map for all patients in this cohort
        all_pkeys_for_cohort = sorted(set(train_pkeys) | set(val_pkeys) | set(test_pkeys))
        for pk in all_pkeys_for_cohort:
            if pk not in patient_to_scans:
                scan_ids = _expand_patients_to_scans(offsets, keys, ids, [pk])
                patient_to_scans[pk] = tuple(sorted(scan_ids))
                patient_to_cohort[pk] = cohort.name

        n_kept = len(set(train_pkeys) | set(val_pkeys) | set(test_pkeys))
        cs = CohortSplit(
            name=cohort.name,
            role="cv",
            image_h5=h5_path,
            train_patients=tuple(sorted(train_pkeys)),
            val_patients=tuple(sorted(val_pkeys)),
            test_patients=tuple(sorted(test_pkeys)),
            n_patients_h5=n_patients_h5,
            n_kept_after_dedup=n_kept,
        )
        per_cohort.append(cs)
        logger.info(
            "%s (cv): H5=%s | train=%d | val=%d | test=%d | patients_h5=%d",
            cohort.name,
            h5_path.name,
            len(cs.train_patients),
            len(cs.val_patients),
            len(cs.test_patients),
            n_patients_h5,
        )

    # --- test_only cohorts ---
    for cohort in registry.test_cohorts():
        h5_path = _resolve_h5_path(str(cohort.image_h5), cfg.image_h5_root, cohort.name)
        if h5_path is None:
            logger.warning("Image H5 for test_only cohort '%s' not found — skipping.", cohort.name)
            continue

        try:
            with h5py.File(h5_path, "r") as hf:
                ids = _decode_ids(hf["ids"][:])
                keys = _decode_ids(hf["patients/keys"][:])
                offsets = hf["patients/offsets"][:]
                all_pkeys = list(keys)
        except Exception as exc:
            raise SegDataError(
                f"Failed to read test_only cohort '{cohort.name}' from '{h5_path}': {exc}"
            ) from exc

        n_patients_h5 = len(all_pkeys)

        # Dedup filter — tolerate absence (forward-compatible, no warning)
        if dedup_allowlists is not None and cohort.name in dedup_allowlists:
            allow = dedup_allowlists[cohort.name]
            n_before = len(all_pkeys)
            all_pkeys = [p for p in all_pkeys if p in allow]
            logger.info(
                "%s (test_only): dedup filter kept %d/%d",
                cohort.name,
                len(all_pkeys),
                n_before,
            )

        for pk in all_pkeys:
            if pk not in patient_to_scans:
                scan_ids = _expand_patients_to_scans(offsets, keys, ids, [pk])
                patient_to_scans[pk] = tuple(sorted(scan_ids))
                patient_to_cohort[pk] = cohort.name

        cs = CohortSplit(
            name=cohort.name,
            role="test_only",
            image_h5=h5_path,
            train_patients=(),
            val_patients=(),
            test_patients=tuple(sorted(all_pkeys)),
            n_patients_h5=n_patients_h5,
            n_kept_after_dedup=len(all_pkeys),
        )
        per_cohort.append(cs)
        logger.info(
            "%s (test_only): H5=%s | test=%d | patients_h5=%d",
            cohort.name,
            h5_path.name,
            len(cs.test_patients),
            n_patients_h5,
        )

    return FmSplitResolution(
        fm_fold=cfg.fm_fold,
        corpus_registry=cfg.corpus_registry.resolve(),
        corpus_registry_sha256=registry_sha,
        dedup_decision_path=(
            cfg.dedup_decision_path.resolve() if cfg.dedup_decision_path else None
        ),
        per_cohort=tuple(per_cohort),
        patient_to_scans=patient_to_scans,
        patient_to_cohort=patient_to_cohort,
    )


def write_splits_json(
    path: Path,
    resolution: FmSplitResolution,
    plan: FoldPlan,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Serialise the split resolution + fold plan to a JSON provenance file.

    Writes atomically (temp file + rename) so partial writes cannot corrupt an
    existing file.

    Parameters
    ----------
    path:
        Destination JSON file path.  Parent directories must exist.
    resolution:
        :class:`FmSplitResolution` from :func:`resolve_fm_splits`.
    plan:
        :class:`~vena.segmentation.data.kfold.FoldPlan` from
        :func:`~vena.segmentation.data.kfold.build_fold_plan`.
    extra:
        Optional extra key-value pairs merged into the top level (useful for
        including the engine config hash, run_id, etc.).

    Returns
    -------
    Path
        The resolved path of the written JSON file.

    Raises
    ------
    SegDataError
        If any structural invariant is violated (see §Invariants below).

    Notes
    -----
    Invariants checked before writing:

    1. ``⋃ plan.folds == set(fm_splits["train"])`` exactly.
    2. Folds are pairwise disjoint (no patient in two folds).
    3. No fm_val/fm_test patient appears in any fold.
    4. Per-cohort fold list ``per_cohort[c]["folds"]["fold_i"]`` ⊆ global
       ``plan.folds[i]``.
    5. ``sum(len(fold) for fold in plan.folds) == len(fm_splits["train"])``.
    """
    fm_splits = resolution.fm_splits()
    fm_train = fm_splits["train"]
    fm_val = fm_splits["val"]
    fm_test = fm_splits["test"]

    fm_train_set = set(fm_train)
    fm_val_set = set(fm_val)
    fm_test_set = set(fm_test)
    fm_excluded_set = fm_val_set | fm_test_set

    # Invariant 1: ⋃ folds == fm_train exactly
    folds_union: set[str] = set()
    for fold in plan.folds:
        folds_union.update(fold)
    if folds_union != fm_train_set:
        extra_in_folds = folds_union - fm_train_set
        missing_from_folds = fm_train_set - folds_union
        raise SegDataError(
            f"write_splits_json invariant violated: ⋃ folds ≠ fm_train_patients. "
            f"Extra in folds: {sorted(extra_in_folds)[:5]}. "
            f"Missing from folds: {sorted(missing_from_folds)[:5]}."
        )

    # Invariant 2: pairwise disjoint folds
    all_in_folds: list[str] = [pid for fold in plan.folds for pid in fold]
    if len(all_in_folds) != len(set(all_in_folds)):
        seen: set[str] = set()
        dups = [p for p in all_in_folds if p in seen or seen.add(p)]  # type: ignore[func-returns-value]
        raise SegDataError(
            f"write_splits_json invariant violated: folds not disjoint. "
            f"Duplicates: {sorted(set(dups))[:5]}."
        )

    # Invariant 3: no val/test patient in any fold
    leaked = folds_union & fm_excluded_set
    if leaked:
        raise SegDataError(
            f"write_splits_json invariant violated: {len(leaked)} val/test patients "
            f"found in folds: {sorted(leaked)[:5]}."
        )

    # Invariant 5: sum of fold sizes == len(fm_train)
    total_fold_patients = sum(len(fold) for fold in plan.folds)
    if total_fold_patients != len(fm_train):
        raise SegDataError(
            f"write_splits_json invariant violated: "
            f"sum(per_fold_patients)={total_fold_patients} ≠ len(fm_train)={len(fm_train)}."
        )

    # Build per-cohort fold lists and check invariant 4
    per_cohort_dict: dict[str, dict] = {}
    for cs in resolution.per_cohort:
        cohort_train_set = set(cs.train_patients)
        cohort_folds: dict[str, list[str]] = {}
        for i, fold in enumerate(plan.folds):
            cohort_fold = sorted(set(fold) & cohort_train_set)
            # Invariant 4: per-cohort fold must be a subset of the global fold
            extra_in_cohort = set(cohort_fold) - set(fold)
            if extra_in_cohort:
                raise SegDataError(
                    f"write_splits_json invariant violated: "
                    f"per-cohort fold_{i} for '{cs.name}' has IDs not in global fold_{i}: "
                    f"{sorted(extra_in_cohort)[:5]}."
                )
            cohort_folds[f"fold_{i}"] = cohort_fold

        per_cohort_dict[cs.name] = {
            "role": cs.role,
            "image_h5": str(cs.image_h5),
            "n_patients_h5": cs.n_patients_h5,
            "n_kept_after_dedup": cs.n_kept_after_dedup,
            "fm_train_patients": list(cs.train_patients),
            "fm_val_patients": list(cs.val_patients),
            "fm_test_patients": list(cs.test_patients),
            "folds": cohort_folds,
        }

    per_fold_patients = [len(fold) for fold in plan.folds]
    fm_train_scans = len(resolution.scans_for(fm_train))

    payload: dict = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "producer": _PRODUCER,
        "git_sha": _git_sha(resolution.corpus_registry),
        "k_folds": plan.k,
        "fold_seed": None,  # fold_seed lives in DataConfig; not on FoldPlan
        "fm_fold": resolution.fm_fold,
        "corpus_registry": str(resolution.corpus_registry),
        "corpus_registry_sha256": resolution.corpus_registry_sha256,
        "dedup_decision_path": (
            str(resolution.dedup_decision_path) if resolution.dedup_decision_path else None
        ),
        "counts": {
            "fm_train_patients": len(fm_train),
            "fm_val_patients": len(fm_val),
            "fm_test_patients": len(fm_test),
            "fm_train_scans": fm_train_scans,
            "per_fold_patients": per_fold_patients,
        },
        "fm_train_patients": fm_train,
        "fm_val_patients": fm_val,
        "fm_test_patients": fm_test,
        "folds": {f"fold_{i}": sorted(fold) for i, fold in enumerate(plan.folds)},
        "per_cohort": per_cohort_dict,
        "patient_to_scans": {
            pid: list(scans) for pid, scans in resolution.patient_to_scans.items()
        },
    }

    if extra:
        for k, v in extra.items():
            payload[k] = v

    # Atomic write: write to a temp file in the same directory, then rename
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.json")
    try:
        with open(tmp_fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        import os

        os.rename(tmp_name, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            import os

            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    logger.info("Splits JSON written to %s", path)
    return path
