"""Patient-level split generation for nested cross-validation.

Produces splits stored as **patient-ID strings**, not as integer indices, so
splits survive any reordering of the underlying image stack. This is the
``splits/{test, cv/fold_*/{train, val}}`` layout described in the project's
H5 design principles (the principle-9 indices-not-IDs default is overridden
here per explicit user choice: the cache must be reorder-stable).

The default strategy is:

* Hold out ``n_test`` patients deterministically (stratified if labels given).
* Run ``n_folds``-fold CV on the remaining patients (also stratified if
  labels given), producing ``(train, val)`` per fold.

Stratification uses :class:`sklearn.model_selection.StratifiedKFold` /
:class:`sklearn.model_selection.StratifiedShuffleSplit` when a label vector is
supplied; otherwise plain :class:`KFold` / random selection.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import h5py
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _FoldSplit(TypedDict):
    train: list[str]
    val: list[str]


class NestedCVSplits(TypedDict):
    """Patient-ID-typed split structure consumed by H5 writers and dataloaders."""

    test: list[str]
    folds: dict[int, _FoldSplit]


def make_nested_cv_splits(
    patient_ids: Sequence[str],
    *,
    n_folds: int = 5,
    n_test: int = 50,
    seed: int = 42,
    stratify_by: Sequence[int] | Sequence[str] | None = None,
) -> NestedCVSplits:
    """Build a held-out test set + ``n_folds``-fold CV on the remainder.

    Parameters
    ----------
    patient_ids
        Patient identifiers (e.g. ``["UCSF-PDGM-0004", ...]``). Order is
        preserved internally; returned splits do not depend on input order
        beyond the seed.
    n_folds
        Number of CV folds.
    n_test
        Size of the held-out test set, shared across folds.
    seed
        Seed for both the test-split and the CV shuffler.
    stratify_by
        Optional per-patient label used to stratify the test split AND the
        CV folds. Length must match ``patient_ids``. Categorical labels are
        accepted (strings or ints).

    Returns
    -------
    NestedCVSplits
        ``{"test": [PIDs], "folds": {0: {"train": [...], "val": [...]}, ...}}``

    Raises
    ------
    ValueError
        If sizes are inconsistent (``n_test >= len(patient_ids)``, ``n_folds < 2``,
        etc.) or ``stratify_by`` length disagrees with ``patient_ids``.
    """
    ids = list(patient_ids)
    n = len(ids)
    if n_test <= 0 or n_test >= n:
        raise ValueError(f"n_test must satisfy 0 < n_test < n_patients={n}, got {n_test}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2, got {n_folds}")
    labels: np.ndarray | None = None
    if stratify_by is not None:
        if len(stratify_by) != n:
            raise ValueError(f"stratify_by length {len(stratify_by)} != n_patients {n}")
        labels = np.asarray(list(stratify_by))

    idx = np.arange(n)

    # ---- held-out test split ------------------------------------------------
    if labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        train_idx, test_idx = next(sss.split(idx, labels))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])

    test_ids = [ids[i] for i in test_idx]
    cv_ids = [ids[i] for i in train_idx]
    cv_labels = labels[train_idx] if labels is not None else None

    # ---- CV folds on the remainder ------------------------------------------
    if cv_labels is not None:
        splitter: KFold | StratifiedKFold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        fold_iter = splitter.split(np.zeros(len(cv_ids)), cv_labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(np.zeros(len(cv_ids)))

    folds: dict[int, _FoldSplit] = {}
    for k, (tr, va) in enumerate(fold_iter):
        folds[k] = {
            "train": [cv_ids[i] for i in tr],
            "val": [cv_ids[i] for i in va],
        }

    logger.info(
        "Splits: n=%d test=%d cv=%d × %d folds (stratified=%s)",
        n,
        len(test_ids),
        len(cv_ids),
        n_folds,
        labels is not None,
    )
    return {"test": test_ids, "folds": folds}


def make_cohort_splits(
    patient_ids: Sequence[str],
    *,
    n_folds: int = 5,
    test_fraction: float = 0.10,
    n_test_min: int = 25,
    seed: int = 42,
    stratify_by: Sequence[int] | Sequence[str] | None = None,
    role: Literal["cv", "test_only"] = "cv",
) -> NestedCVSplits:
    """Per-cohort, leakage-proof split with a quota-based test holdout.

    The held-out test size is ``n_test(c) = max(n_test_min, ceil(test_fraction *
    N))`` so every cohort retains enough cases for a per-cohort metric with a
    usable confidence interval (the ``n_test_min`` floor matters for small
    cohorts). Test-only cohorts assign every patient to ``test`` with no CV folds.

    Parameters
    ----------
    patient_ids
        Patient identifiers. Splitting is patient-level so no patient straddles
        a split (the caller expands patient → scan rows downstream).
    n_folds
        Number of CV folds for ``role == "cv"``.
    test_fraction
        Fraction ``ρ`` of patients held out for test (before the floor).
    n_test_min
        Minimum held-out test size; the floor that protects small cohorts.
    seed
        Seed for the test split and CV shuffler.
    stratify_by
        Optional per-patient label (e.g. WHO grade); ``None`` → random.
    role
        ``"cv"`` for train/val/test cohorts; ``"test_only"`` for held-out cohorts.

    Returns
    -------
    NestedCVSplits
        ``{"test": [...], "folds": {...}}``; ``folds`` is empty for test-only.

    Raises
    ------
    ValueError
        If ``role == "cv"`` but the cohort is too small to hold out the quota
        and still run ``n_folds``-fold CV.
    """
    ids = list(patient_ids)
    n = len(ids)
    if role == "test_only":
        logger.info("Cohort split: role=test_only, all %d patients → test", n)
        return {"test": list(ids), "folds": {}}

    n_test = max(int(n_test_min), math.ceil(test_fraction * n))
    if n_test >= n:
        raise ValueError(
            f"cohort too small for cv role: n={n}, computed n_test={n_test} "
            f"(test_fraction={test_fraction}, n_test_min={n_test_min})"
        )
    if n - n_test < n_folds:
        raise ValueError(
            f"cohort too small for {n_folds}-fold CV after holding out n_test={n_test} from n={n}"
        )
    return make_nested_cv_splits(
        ids,
        n_folds=n_folds,
        n_test=n_test,
        seed=seed,
        stratify_by=stratify_by,
    )


_FLAT_SPLITS: tuple[str, ...] = ("splits/train", "splits/val", "splits/test")


def normalize_splits(
    h5_path: Path | str,
    role: Literal["cv", "test_only"],
    *,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Rewrite splits in place so every cohort H5 carries the canonical layout.

    Canonical layout per role:

    * ``role="cv"`` → keep ``splits/test`` plus ``splits/cv/fold_N/{train,val}``.
      Drop the legacy flat ``splits/{train,val}`` that some cohorts shipped.
    * ``role="test_only"`` → keep only ``splits/test``. Drop the
      ``splits/cv/fold_0/{train,val}`` aliases (val == test, train == empty).

    The function never invents splits — it only removes redundant aliases.
    ``splits/test`` is preserved verbatim in both roles.

    Parameters
    ----------
    h5_path : Path | str
        H5 file to rewrite in place.
    role : Literal["cv", "test_only"]
        Cohort role; controls which legacy nodes are removed.
    dry_run : bool
        When True, compute the diff but do not mutate the file. Useful for
        the per-cohort delta CSV produced by
        ``scripts/normalize_splits_inplace.py``.

    Returns
    -------
    dict[str, list[str]]
        ``{"removed": [...], "kept": [...]}`` listing the H5 paths that were
        (or would be) dropped and the ones that remain.
    """
    path = Path(h5_path)
    removed: list[str] = []
    kept: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"normalize_splits: H5 not found: {path}")

    mode = "r" if dry_run else "a"
    with h5py.File(path, mode) as f:
        if "splits" not in f:
            return {"removed": removed, "kept": kept}
        if role == "cv":
            for legacy in ("splits/train", "splits/val"):
                if legacy in f:
                    removed.append(legacy)
                    if not dry_run:
                        del f[legacy]
            # Keep splits/test + splits/cv/* untouched.
            for key in f["splits"]:
                full = f"splits/{key}"
                if full in removed:
                    continue
                kept.append(full)
        elif role == "test_only":
            # Drop the cv/fold_0/{train,val} aliases (and any cv subtree).
            if "splits/cv" in f:
                removed.append("splits/cv")
                if not dry_run:
                    del f["splits/cv"]
            for key in f["splits"]:
                full = f"splits/{key}"
                if full in removed:
                    continue
                kept.append(full)
        else:  # pragma: no cover — covered by Literal
            raise ValueError(f"unknown role {role!r}; expected 'cv' or 'test_only'")
    return {"removed": removed, "kept": kept}
