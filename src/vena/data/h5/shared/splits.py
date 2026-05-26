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
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

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
