"""K-fold out-of-fold split machinery for the segmentation submodule.

Produces a :class:`FoldPlan` that maps FM training patients to K held-out
folds and guarantees no leakage of FM val/test patients into any training fold.

Leakage vectors guarded:
    L1 (direct)     — no FM-val/test ID appears in any fold.
    L2 (transitive) — a patient deduplicated across cohorts (e.g. UCSF-PDGM
                       and BraTS-GLI share patients) must also be excluded from
                       all folds when any of its aliases is an FM-val/test ID.
                       Suppressed via the optional ``dedup_duplicates`` mapping.

Determinism guarantee: ``folds`` depend *only* on
``(sorted(fm_train_ids), cfg.fold_seed, cfg.k_folds)``.  Two calls with the
same inputs produce bit-identical results on any platform.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from vena.segmentation.exceptions import SegDataError

if TYPE_CHECKING:
    from vena.segmentation.config import DataConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FoldPlan",
    "build_fold_plan",
    "oof_assignment",
]


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldPlan:
    """Frozen, JSON-serialisable K-fold out-of-fold split plan.

    Attributes
    ----------
    k:
        Number of cross-validation folds.
    fm_train_ids:
        Sorted tuple of FM training patient IDs.  Equals ``⋃ folds`` exactly.
    folds:
        Tuple of K disjoint tuples.  ``folds[i]`` contains the IDs held out
        when fold-``i`` model is trained; those IDs are predicted OOF by
        fold-``i``.  Union equals ``fm_train_ids`` exactly.
    fm_val_ids:
        Sorted tuple of FM validation IDs.  ``oof_assignment`` maps these to
        ``"all_train"``.
    fm_test_ids:
        Sorted tuple of FM test IDs.  ``oof_assignment`` maps these to
        ``"all_train"``.
    """

    k: int
    fm_train_ids: tuple[str, ...]
    folds: tuple[tuple[str, ...], ...]
    fm_val_ids: tuple[str, ...]
    fm_test_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for provenance logging."""
        return {
            "k": self.k,
            "fm_train_ids": list(self.fm_train_ids),
            "folds": [list(f) for f in self.folds],
            "fm_val_ids": list(self.fm_val_ids),
            "fm_test_ids": list(self.fm_test_ids),
        }

    def to_json(self, **kwargs) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> FoldPlan:
        """Reconstruct from a :meth:`to_dict` output."""
        return cls(
            k=d["k"],
            fm_train_ids=tuple(d["fm_train_ids"]),
            folds=tuple(tuple(f) for f in d["folds"]),
            fm_val_ids=tuple(d["fm_val_ids"]),
            fm_test_ids=tuple(d["fm_test_ids"]),
        )


# ---------------------------------------------------------------------------
# Cohort label extraction (for stratified splitting)
# ---------------------------------------------------------------------------


def _extract_cohort(patient_id: str) -> str:
    """Heuristically extract a cohort label from a patient ID.

    Strips trailing digit groups and delimiters to recover the cohort prefix.

    Examples
    --------
    >>> _extract_cohort("BraTS-GLI-00001")
    'BraTS-GLI'
    >>> _extract_cohort("UCSF-PDGM-001")
    'UCSF-PDGM'
    >>> _extract_cohort("COHA_001")
    'COHA'

    Parameters
    ----------
    patient_id:
        A patient identifier following any BraTS / UCSF-PDGM naming convention.

    Returns
    -------
    str
        Cohort prefix, or *patient_id* unchanged if extraction yields an empty
        string (degenerate case — treated as its own cohort).
    """
    # Remove trailing digit runs, then strip trailing delimiters
    prefix = re.sub(r"\d+$", "", patient_id).rstrip("-_")
    return prefix if prefix else patient_id


# ---------------------------------------------------------------------------
# Core fold-building logic
# ---------------------------------------------------------------------------


def _assign_folds_stratified(
    sorted_ids: list[str],
    k: int,
    seed: int,
) -> list[list[str]]:
    """Assign patient IDs to K folds with cohort stratification.

    Uses :func:`sklearn.model_selection.StratifiedKFold` with ``shuffle=True``
    and ``random_state=seed``.  Cohort labels are derived heuristically from
    patient ID prefixes via :func:`_extract_cohort`.

    Parameters
    ----------
    sorted_ids:
        Lexicographically sorted list of patient IDs.
    k:
        Number of folds.
    seed:
        RNG seed for reproducible shuffling.

    Returns
    -------
    list[list[str]]
        ``k`` lists, where ``folds[i]`` contains the IDs assigned to fold *i*
        as the held-out set.
    """
    from sklearn.model_selection import StratifiedKFold

    cohort_labels = [_extract_cohort(pid) for pid in sorted_ids]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds: list[list[str]] = [[] for _ in range(k)]
    x_dummy = np.arange(len(sorted_ids))  # dummy features — StratifiedKFold only uses y
    for fold_idx, (_, held_out_indices) in enumerate(skf.split(x_dummy, cohort_labels)):
        folds[fold_idx] = [sorted_ids[i] for i in held_out_indices]

    return folds


def _assign_folds_uniform(
    sorted_ids: list[str],
    k: int,
    seed: int,
) -> list[list[str]]:
    """Assign patient IDs to K folds uniformly (fallback when n < k or 1 cohort).

    Permutes the sorted list with the given RNG seed, then round-robins into K
    buckets.

    Parameters
    ----------
    sorted_ids:
        Lexicographically sorted list of patient IDs.
    k:
        Number of folds.
    seed:
        RNG seed.

    Returns
    -------
    list[list[str]]
        ``k`` lists of (approximately) equal size.
    """
    rng = np.random.default_rng(seed)
    permuted_idx = rng.permutation(len(sorted_ids))
    folds: list[list[str]] = [[] for _ in range(k)]
    for position, original_idx in enumerate(permuted_idx):
        folds[position % k].append(sorted_ids[original_idx])
    return folds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_fold_plan(
    cfg: DataConfig,
    fm_splits: Mapping[str, Sequence[str]],
    *,
    dedup_duplicates: Mapping[str, Sequence[str]] | None = None,
) -> FoldPlan:
    """Build a K-fold out-of-fold split plan from FM corpus splits.

    The segmenter train set is **exactly** the FM train patient set.  No
    independent partition is created (doing so would add an L2 leakage vector
    where a patient's scan appears in both the segmenter train fold and the FM
    train set used to generate the conditioning mask).

    Parameters
    ----------
    cfg:
        Frozen :class:`~vena.segmentation.config.DataConfig`.  Reads
        ``cfg.k_folds`` and ``cfg.fold_seed``.
    fm_splits:
        Mapping with keys ``"train"``, ``"val"``, and ``"test"`` mapping to
        sequences of patient IDs (strings).  These are the FM model's
        patient-level splits read from the H5 ``splits/`` group or the corpus
        registry — **not** independently computed.
    dedup_duplicates:
        Optional mapping from patient ID to a list of its *duplicate* IDs in
        other cohorts.  Example real record shape (derived from the
        UCSF-PDGM ↔ BraTS-GLI dedup preflight)::

            {
                "BraTS-GLI-00001": ["UCSF-PDGM-001"],
                "UCSF-PDGM-001": ["BraTS-GLI-00001"],
            }

        When provided, any FM-val/test ID *and all its transitive dedup
        aliases* are excluded from every fold (transitive leakage guard L2).
        ``None`` disables the transitive check (direct check L1 still runs).

    Returns
    -------
    FoldPlan
        Frozen :class:`FoldPlan` with ``k``, ``fm_train_ids``, ``folds``,
        ``fm_val_ids``, and ``fm_test_ids`` populated.

    Raises
    ------
    SegDataError
        If required keys are missing from *fm_splits*, if any FM-val/test ID
        appears in a fold (direct leakage), or if any dedup-duplicate of an
        FM-val/test ID appears in a fold (transitive leakage).
    """
    # ------------------------------------------------------------------ input
    for key in ("train", "val", "test"):
        if key not in fm_splits:
            raise SegDataError(
                f"fm_splits is missing required key '{key}'. "
                f"Available keys: {sorted(fm_splits.keys())}"
            )

    raw_train = list(fm_splits["train"])
    raw_val = list(fm_splits["val"])
    raw_test = list(fm_splits["test"])

    if not raw_train:
        raise SegDataError("fm_splits['train'] is empty — no patients to fold.")

    # ---------------------------------------------------------------- dedup set
    # Build the transitive closure of FM-val/test IDs that must be excluded.
    # Direct: val/test IDs themselves.
    # Transitive: their dedup aliases across cohorts.
    excluded: set[str] = set(raw_val) | set(raw_test)

    if dedup_duplicates is not None:
        # One pass is sufficient: duplicates are symmetric by construction
        # (the dedup preflight records both directions).
        transitive: set[str] = set()
        for eid in excluded:
            aliases = dedup_duplicates.get(eid, [])
            transitive.update(aliases)
        excluded.update(transitive)
        logger.debug(
            "Transitive exclusion: %d direct + %d alias IDs excluded from folds.",
            len(set(raw_val) | set(raw_test)),
            len(transitive),
        )

    # Remove any train IDs that appear in the exclusion set (should not happen
    # in a well-formed corpus, but guard defensively).
    clean_train = [pid for pid in raw_train if pid not in excluded]
    if len(clean_train) < len(raw_train):
        removed = set(raw_train) - set(clean_train)
        logger.warning(
            "Removed %d IDs from fm_train that overlapped with val/test or their dedup aliases: %s",
            len(removed),
            sorted(removed)[:5],
        )

    sorted_train = sorted(clean_train)
    k = cfg.k_folds
    seed = cfg.fold_seed

    if len(sorted_train) < k:
        raise SegDataError(
            f"Too few training patients ({len(sorted_train)}) for {k} folds. "
            "Reduce cfg.k_folds or add more training patients."
        )

    # ---------------------------------------------------------------- fold assignment
    # Try cohort-stratified; fall back to uniform if all IDs share one cohort
    # (trivial stratification would degenerate StratifiedKFold).
    cohorts = {_extract_cohort(pid) for pid in sorted_train}
    if len(cohorts) >= 2:
        try:
            raw_folds = _assign_folds_stratified(sorted_train, k, seed)
        except Exception as exc:
            logger.warning("Stratified KFold failed (%s); falling back to uniform split.", exc)
            raw_folds = _assign_folds_uniform(sorted_train, k, seed)
    else:
        raw_folds = _assign_folds_uniform(sorted_train, k, seed)

    # ---------------------------------------------------------------- validate
    folds_tuple: tuple[tuple[str, ...], ...] = tuple(
        tuple(sorted(f))
        for f in raw_folds  # sort within each fold for repr stability
    )
    _assert_plan_valid(
        folds=folds_tuple,
        fm_train_ids=tuple(sorted_train),
        excluded=excluded,
    )

    plan = FoldPlan(
        k=k,
        fm_train_ids=tuple(sorted_train),
        folds=folds_tuple,
        fm_val_ids=tuple(sorted(raw_val)),
        fm_test_ids=tuple(sorted(raw_test)),
    )
    logger.info(
        "FoldPlan: k=%d | train=%d | val=%d | test=%d | folds=%s",
        k,
        len(sorted_train),
        len(raw_val),
        len(raw_test),
        [len(f) for f in folds_tuple],
    )
    return plan


def _assert_plan_valid(
    folds: tuple[tuple[str, ...], ...],
    fm_train_ids: tuple[str, ...],
    excluded: set[str],
) -> None:
    """Assert all structural invariants of a fold plan.

    Parameters
    ----------
    folds:
        K-tuple of held-out ID tuples.
    fm_train_ids:
        Sorted tuple of all training IDs.
    excluded:
        Set of IDs (val + test + their dedup aliases) that must not appear in
        any fold.

    Raises
    ------
    SegDataError
        If any invariant is violated.
    """
    all_in_folds: list[str] = [pid for fold in folds for pid in fold]
    all_set = set(all_in_folds)

    # (a) Disjoint folds
    if len(all_in_folds) != len(all_set):
        seen: set[str] = set()
        duplicates = [pid for pid in all_in_folds if pid in seen or seen.add(pid)]  # type: ignore[func-returns-value]
        raise SegDataError(
            f"Folds are not disjoint — duplicate IDs found: {sorted(set(duplicates))[:5]}"
        )

    # (b) Union equals train set
    train_set = set(fm_train_ids)
    if all_set != train_set:
        extra = all_set - train_set
        missing = train_set - all_set
        raise SegDataError(
            f"⋃ folds ≠ fm_train_ids. "
            f"Extra in folds: {sorted(extra)[:5]}. "
            f"Missing from folds: {sorted(missing)[:5]}."
        )

    # (c) No excluded ID in any fold (direct + transitive leakage)
    leaked = all_set & excluded
    if leaked:
        raise SegDataError(
            f"Leakage detected: {len(leaked)} excluded IDs found in folds: {sorted(leaked)[:5]}"
        )


def oof_assignment(plan: FoldPlan, patient_id: str) -> int | Literal["all_train"]:
    """Return the OOF fold index for a patient, or ``"all_train"``.

    FM-val and FM-test patients are predicted by the *all-train* model (trained
    on the entire ``fm_train_ids`` set without held-out folds).  FM-train
    patients are predicted OOF by the fold model that held them out.

    Parameters
    ----------
    plan:
        A :class:`FoldPlan` from :func:`build_fold_plan`.
    patient_id:
        Patient ID to look up.

    Returns
    -------
    int | Literal["all_train"]
        Fold index ``i`` if *patient_id* is in ``plan.folds[i]``,
        or ``"all_train"`` if the patient is in ``plan.fm_val_ids`` or
        ``plan.fm_test_ids``.

    Raises
    ------
    SegDataError
        If *patient_id* is not found in any fold, val, or test set.
    """
    # FM-val / FM-test → all-train prediction
    if patient_id in plan.fm_val_ids or patient_id in plan.fm_test_ids:
        return "all_train"

    # FM-train → find which fold held this patient out
    for fold_idx, fold in enumerate(plan.folds):
        if patient_id in fold:
            return fold_idx

    raise SegDataError(
        f"patient_id '{patient_id}' not found in any fold, val set, or test set of the FoldPlan."
    )
