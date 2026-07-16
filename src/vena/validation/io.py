"""Phase-2 results loader — stream prediction/reference pairs scan by scan.

The join is by ``scan_id``, never by row index.  Reference H5 rows and
prediction H5 rows are **not** guaranteed to be in the same order.

Memory contract: ``iter_scans`` keeps at most one prediction volume and one
reference volume in RAM at a time.  Each array is ~35 MB; peak RSS stays flat
across a 50-scan file because the arrays are yielded to the caller and freed
before the next read.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SUPPORTED_PRED_SCHEMAS: frozenset[str] = frozenset({"2.0"})
SUPPORTED_REF_SCHEMAS: frozenset[str] = frozenset({"2.0"})

_INDEX_COLUMNS = [
    "method",
    "cohort",
    "ring",
    "nfe",
    "shard",
    "path",
    "references_h5",
    "n_scans",
    "schema_version",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInfo:
    """One inference shard root with its ``decision.json`` provenance."""

    root: Path
    decision: dict


@dataclass(frozen=True)
class ScanSample:
    """One scan × one method × one NFE, joined from prediction + reference H5.

    Arrays are ``(H, W, D)``; ``pred`` and ``real`` are ``float32`` in
    ``[0, 1]`` (inside the brain mask); ``brain`` and ``wt`` are ``bool``.
    """

    scan_id: str
    patient_id: str
    cohort: str
    ring: str  # "A" or "B"
    method: str
    nfe: int
    pred: np.ndarray  # (H, W, D) float32  — t1c_synthetic_harmonised
    real: np.ndarray  # (H, W, D) float32  — t1c_real_harmonised
    brain: np.ndarray  # (H, W, D) bool
    wt: np.ndarray  # (H, W, D) bool
    inference_seconds: float
    peak_vram_mb: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_str_arr(values: object) -> list[str]:
    """Decode an h5py vlen-str array to ``list[str]``."""
    out: list[str] = []
    for b in values:  # type: ignore[union-attr]
        out.append(b.decode() if isinstance(b, bytes) else str(b))
    return out


def _build_scan_index(ref_path: Path) -> dict[str, int]:
    """Return ``{scan_id: row_index}`` for a reference H5 (reads only /metadata/scan_id)."""
    with h5py.File(ref_path, "r") as f:
        sids = _decode_str_arr(f["metadata/scan_id"][:])
    return {sid: i for i, sid in enumerate(sids)}


def _resolve_references_h5(pred_h5: h5py.File, pred_path: Path) -> Path:
    """Resolve the ``references_h5`` root attr relative to the shard root.

    Shard root = ``pred_path.parents[3]``  (nfe_*.h5 → COHORT → METHOD →
    predictions → shard-root).
    """
    rel = pred_h5.attrs.get("references_h5")
    shard_root = pred_path.parents[3]
    if rel:
        return (shard_root / str(rel)).resolve()
    # Fallback: canonical position.
    cohort = str(pred_h5.attrs.get("cohort", ""))
    return (shard_root / "references" / f"{cohort}.h5").resolve()


# ---------------------------------------------------------------------------
# ReferenceCache
# ---------------------------------------------------------------------------


class ReferenceCache:
    """LRU cache for reference H5 scan-id → row-index mappings.

    16 methods share one reference file per cohort.  Caching the index avoids
    re-reading ``metadata/scan_id`` up to 45× per cohort sweep.

    Parameters
    ----------
    maxsize :
        Maximum number of reference files to keep in cache.
    """

    def __init__(self, maxsize: int = 40) -> None:
        self._cache: OrderedDict[Path, dict[str, int]] = OrderedDict()
        self._maxsize = maxsize

    def get_scan_index(self, ref_path: Path) -> dict[str, int]:
        """Return ``{scan_id: row_index}``, reading and caching as needed.

        Parameters
        ----------
        ref_path :
            Absolute path to a per-cohort references H5.

        Returns
        -------
        dict[str, int]
            Mapping from scan_id to 0-based row index in that file.
        """
        ref_path = Path(ref_path).resolve()
        if ref_path in self._cache:
            self._cache.move_to_end(ref_path)
            return self._cache[ref_path]
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        idx = _build_scan_index(ref_path)
        self._cache[ref_path] = idx
        return idx


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


def discover_shards(root: Path) -> list[ShardInfo]:
    """Discover inference shards by globbing ``<root>/*/decision.json``.

    Parameters
    ----------
    root :
        Inference tree root (e.g. ``results/fm/inference``).

    Returns
    -------
    list[ShardInfo]
        One entry per shard directory that has a ``decision.json``.
    """
    root = Path(root)
    shards: list[ShardInfo] = []
    for dec_path in sorted(root.glob("*/decision.json")):
        try:
            with dec_path.open() as fh:
                decision = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read %s: %s", dec_path, exc)
            continue
        shards.append(ShardInfo(root=dec_path.parent, decision=decision))
    return shards


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def build_index(root: Path) -> pd.DataFrame:
    """Build a tidy index of every prediction H5 under *root*.

    Discovers by globbing ``<root>/*/predictions/*/*/nfe_*.h5``.
    Never hard-codes method or cohort names — BraTS-PED or future methods
    will flow through with no code change.

    Parameters
    ----------
    root :
        Inference tree root.

    Returns
    -------
    pd.DataFrame
        One row per file, columns: ``method, cohort, ring, nfe, shard,
        path, references_h5, n_scans, schema_version``.
    """
    root = Path(root)
    rows: list[dict] = []
    for pred_path in sorted(root.glob("*/predictions/*/*/nfe_*.h5")):
        try:
            with h5py.File(pred_path, "r") as f:
                schema = str(f.attrs.get("schema_version", ""))
                if schema not in SUPPORTED_PRED_SCHEMAS:
                    logger.warning("skipping %s: unsupported schema_version=%r", pred_path, schema)
                    continue
                method = str(f.attrs.get("method", ""))
                cohort = str(f.attrs.get("cohort", ""))
                nfe = int(f.attrs.get("nfe", 0))
                ring = str(f.attrs.get("ring", ""))
                ref_rel = f.attrs.get("references_h5")
                n_scans = int(f["predictions/t1c_synthetic_harmonised"].shape[0])
        except (OSError, KeyError) as exc:
            logger.warning("skipping %s: %s", pred_path, exc)
            continue

        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "ring": ring,
                "nfe": nfe,
                "shard": pred_path.parents[3].name,
                "path": pred_path,
                "references_h5": str(ref_rel) if ref_rel else None,
                "n_scans": n_scans,
                "schema_version": schema,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_INDEX_COLUMNS)
    df = pd.DataFrame(rows)
    # Ensure correct dtypes
    df["nfe"] = df["nfe"].astype(int)
    df["n_scans"] = df["n_scans"].astype(int)
    return df


# ---------------------------------------------------------------------------
# iter_scans — the streaming reader
# ---------------------------------------------------------------------------


def iter_scans(
    pred_path: Path,
    *,
    reference_cache: ReferenceCache,
    scan_ids: list[str] | None = None,
) -> Iterator[ScanSample]:
    """Stream one :class:`ScanSample` at a time from a prediction H5.

    Join is by ``scan_id``.  The reference file rows are **not** assumed to
    be in the same order as the prediction rows — the join is by content, not
    position.

    Memory stays flat: each iteration reads exactly one row from the
    prediction dataset and one row from the reference dataset (``(1,H,W,D)``
    chunk), then yields.  The arrays are freed after the caller's ``next()``
    returns.

    Parameters
    ----------
    pred_path :
        Path to a ``predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5`` file.
    reference_cache :
        Shared :class:`ReferenceCache` instance.
    scan_ids :
        Optional subset of scan IDs to iterate.  When ``None``, all scans
        in the prediction file are streamed.

    Yields
    ------
    ScanSample
        One sample per scan, in the order of *scan_ids* (or the order the
        prediction file stores them when scan_ids is None).

    Warns
    -----
    A prediction scan missing from the reference file (or vice-versa) is
    logged as WARNING and counted, not raised.
    """
    pred_path = Path(pred_path)

    with h5py.File(pred_path, "r") as f_pred:
        schema = str(f_pred.attrs.get("schema_version", ""))
        if schema not in SUPPORTED_PRED_SCHEMAS:
            raise ValueError(
                f"Unsupported schema_version={schema!r} in {pred_path}; "
                f"expected one of {SUPPORTED_PRED_SCHEMAS}"
            )

        method = str(f_pred.attrs.get("method", ""))
        cohort = str(f_pred.attrs.get("cohort", ""))
        nfe = int(f_pred.attrs.get("nfe", 0))
        ring = str(f_pred.attrs.get("ring", ""))

        ref_path = _resolve_references_h5(f_pred, pred_path)

        # ---- build in-memory indices (metadata only, no volumes) ----
        pred_scan_ids = _decode_str_arr(f_pred["metadata/scan_id"][:])
        pred_patient_ids = _decode_str_arr(f_pred["metadata/patient_id"][:])
        pred_inf_secs = f_pred["metadata/inference_seconds"][:]
        pred_vram = (
            f_pred["metadata/peak_vram_mb"][:] if "metadata/peak_vram_mb" in f_pred else None
        )
        pred_idx: dict[str, int] = {sid: i for i, sid in enumerate(pred_scan_ids)}

        target_sids = scan_ids if scan_ids is not None else pred_scan_ids

        # Resolve reference scan index (LRU cached)
        ref_idx = reference_cache.get_scan_index(ref_path)

        # Open the reference file once for the whole iteration
        with h5py.File(ref_path, "r") as f_ref:
            ds_real = f_ref["reference/t1c_real_harmonised"]
            ds_brain = f_ref["masks/brain"]
            ds_wt = f_ref["masks/wt"]
            ds_pred_vol = f_pred["predictions/t1c_synthetic_harmonised"]

            n_missing = 0
            for sid in target_sids:
                pi = pred_idx.get(sid)
                ri = ref_idx.get(sid)

                if pi is None:
                    logger.warning(
                        "scan_id %r not in prediction file %s — skipping", sid, pred_path
                    )
                    n_missing += 1
                    continue
                if ri is None:
                    logger.warning("scan_id %r not in reference file %s — skipping", sid, ref_path)
                    n_missing += 1
                    continue

                # Single-chunk reads — each is O(1 × H × W × D)
                pred_vol = np.asarray(ds_pred_vol[pi], dtype=np.float32)
                real_vol = np.asarray(ds_real[ri], dtype=np.float32)
                brain_arr = np.asarray(ds_brain[ri], dtype=bool)
                wt_arr = np.asarray(ds_wt[ri], dtype=bool)

                yield ScanSample(
                    scan_id=sid,
                    patient_id=pred_patient_ids[pi],
                    cohort=cohort,
                    ring=ring,
                    method=method,
                    nfe=nfe,
                    pred=pred_vol,
                    real=real_vol,
                    brain=brain_arr,
                    wt=wt_arr,
                    inference_seconds=float(pred_inf_secs[pi]),
                    peak_vram_mb=float(pred_vram[pi]) if pred_vram is not None else float("nan"),
                )
                # Arrays go out of scope here; GC frees them before the next read.

            if n_missing > 0:
                logger.warning(
                    "%d scan(s) skipped (missing from pred or ref) in %s",
                    n_missing,
                    pred_path,
                )
