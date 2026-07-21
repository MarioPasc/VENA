"""Phase-2 results loader — stream prediction/reference pairs scan by scan.

The join is by ``scan_id``, never by row index.  Reference H5 rows and
prediction H5 rows are **not** guaranteed to be in the same order.

Memory contract: ``iter_scans`` keeps at most one prediction volume and one
reference volume in RAM at a time.  Each array is ~35 MB; peak RSS stays flat
across a 50-scan file because the arrays are yielded to the caller and freed
before the next read.

Smoke-shard filtering
---------------------
Inference trees may contain **smoke shards** left over from pre-launch test
runs.  A smoke shard has ``decision.json["smoke"]["enabled"] == true``.
:func:`discover_shards` skips them at discovery time and logs each skip at
INFO.  :func:`build_index` calls ``discover_shards`` internally, so smoke
files never enter the index.  The skipped tags are returned in
:class:`ShardDiscovery` so that downstream engines can record them in their
own ``decision.json`` for provenance.

Do **not** hard-code shard directory name prefixes as the filter — BraTS-PED
backfill shards named ``picasso_ped_*`` are production shards and must be
included.  ``smoke.enabled`` is the correct discriminator.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch

from vena.common import ENCODER_PERCENTILE_UPPER, percentile_normalise

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SUPPORTED_PRED_SCHEMAS: frozenset[str] = frozenset({"2.0"})
SUPPORTED_REF_SCHEMAS: frozenset[str] = frozenset({"2.0"})

#: Maximum brain p99.5 for a raw prediction to be scored in its native space.
#:
#: Methods trained on percentile-normalised T1c emit raw p99.5 ≤ ~0.97.
#: Scanner-unit methods (e.g. ``C0-Identity``) sit at 1 424–2 466.
#: The threshold is safe by three orders of magnitude; do not widen without
#: evidence — it governs the paper's headline MAE comparison.
SCORING_P995_MAX: float = 1.05

#: Minimum brain voxel value for a raw prediction to be scored in its native space.
#:
#: Methods trained on normalised T1c emit non-negative values.  A minimum
#: below this floor signals scanner-unit or improperly post-processed data
#: that must pass through the Phase-1 harmonised field instead.
SCORING_MIN_FLOOR: float = -0.05

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
class ShardDiscovery:
    """Result of :func:`discover_shards`.

    Parameters
    ----------
    accepted :
        Production shards (``smoke.enabled`` absent or ``false``).
    skipped_smoke :
        ``run_id_tag`` values (or directory names when the tag is absent) of
        smoke shards that were excluded.
    """

    accepted: list[ShardInfo] = field(default_factory=list)
    skipped_smoke: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScanSample:
    """One scan × one method × one NFE, joined from prediction + reference H5.

    ``pred`` is the **selected scoring volume** — see
    :func:`select_scoring_volume`.  For methods trained on
    percentile-normalised T1c (15 of 16 methods), it is
    ``t1c_synthetic_raw``; for scanner-unit methods such as ``C0-Identity``
    it is ``t1c_synthetic_harmonised``.  Downstream metrics must use
    ``pred`` only — never ``pred_harmonised`` for scoring.

    Both raw and harmonised volumes are preserved in ``pred_raw`` /
    ``pred_harmonised`` so that the §4.1 Table-S1 audit can compare them.
    ``pred_mode`` records the selection (``"raw"`` or ``"harmonised"``) per
    row.  ``raw_p995`` is the 99.5th percentile of ``pred_raw`` inside the
    brain mask — it quantifies under-saturation (e.g. C4-3D-DiT p99.5 ≈
    0.38 vs reference ~1.0) which is a reportable finding, not a bug.

    All spatial arrays are ``(H, W, D)``; float arrays are ``float32``;
    mask arrays are ``bool``.
    """

    scan_id: str
    patient_id: str
    cohort: str
    ring: str  # "A" or "B"
    method: str
    nfe: int
    pred: np.ndarray  # (H, W, D) float32 — selected scoring volume
    pred_raw: np.ndarray  # (H, W, D) float32 — t1c_synthetic_raw (audit only)
    pred_harmonised: np.ndarray  # (H, W, D) float32 — t1c_synthetic_harmonised (audit only)
    pred_mode: str  # "raw" | "harmonised"
    raw_p995: float  # np.percentile(raw[brain], 99.5) — §4.1 audit column
    real: np.ndarray  # (H, W, D) float32 — t1c_real_harmonised
    brain: np.ndarray  # (H, W, D) bool
    wt: np.ndarray  # (H, W, D) bool
    inference_seconds: float
    peak_vram_mb: float


# ---------------------------------------------------------------------------
# Scoring space selection
# ---------------------------------------------------------------------------


def select_scoring_volume(
    raw: np.ndarray,
    harmonised: np.ndarray,
    brain: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the correct volume to score and a mode label.

    Decides per-scan whether to use the raw prediction (already in the trained
    normalised space) or the Phase-1 harmonised version.  The decision is a
    **property of the volume**, never a hard-coded method list, so it remains
    correct when new methods are added (e.g. BraTS-PED backfill).

    Parameters
    ----------
    raw :
        ``predictions/t1c_synthetic_raw`` from the prediction H5,
        shape ``(H, W, D)``, float32.
    harmonised :
        ``predictions/t1c_synthetic_harmonised``, same shape.
    brain :
        Binary brain mask, shape ``(H, W, D)``, bool.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(volume, mode)`` where ``mode`` is ``"raw"`` when the raw volume
        is already in the trained normalised space (brain p99.5
        ≤ :data:`SCORING_P995_MAX` **and** brain min ≥
        :data:`SCORING_MIN_FLOOR`), or ``"harmonised"`` otherwise.

    Notes
    -----
    :data:`SCORING_P995_MAX` = 1.05 is safe by three orders of magnitude:
    methods trained on normalised T1c have raw p99.5 ≤ 0.97; scanner-unit
    methods (today: ``C0-Identity``) sit at 1 424–2 466.  Do **not** widen
    the threshold without evidence — it governs the paper's headline MAE.

    When the brain mask is empty (edge-case: all-background volume), returns
    ``(harmonised, "harmonised")`` and logs a WARNING rather than crashing on
    an empty-array percentile call.
    """
    b = raw[brain]
    if b.size == 0:
        logger.warning("select_scoring_volume: empty brain mask — returning harmonised volume.")
        return harmonised, "harmonised"
    p995 = float(np.percentile(b, 99.5))
    if p995 <= SCORING_P995_MAX and float(b.min()) >= SCORING_MIN_FLOOR:
        return raw, "raw"
    return harmonised, "harmonised"


def _decode_ids(f: h5py.File) -> list[str]:
    """Decode an H5 ``ids`` dataset into Python strings."""
    return [s.decode() if isinstance(s, bytes) else str(s) for s in f["ids"][:]]


def harmonise_sample_to_percentile(
    sample: ScanSample,
    image_h5_map: dict[str, Path],
    *,
    percentile_upper: float = ENCODER_PERCENTILE_UPPER,
) -> ScanSample:
    """Return a copy of *sample* with ``pred`` and ``real`` in one intensity space.

    Canonical re-normalisation for the analysis path, mirroring the encoder's
    ``percentile_upper`` (default :data:`ENCODER_PERCENTILE_UPPER` = 99.95). The
    stored ``reference/t1c_real_harmonised`` and
    ``predictions/t1c_synthetic_harmonised`` are baked at **99.5**, while the
    VAE-decoded ``pred_raw`` lives in the encoder's **99.95** space — scoring one
    against the other is the 2026-07-21 ρ_S normalisation confound (99.5 saturates
    the enhancing-rim/vessel tail). This forces both to *percentile_upper* over the
    brain mask:

    * ``99.5`` — return the stored harmonised fields unchanged (already at 99.5).
    * ``99.95`` — re-normalise ``pred_raw`` at 99.95 **and** re-derive ``real`` from
      the **raw** image-H5 ``images/t1c`` at 99.95 (the stored harmonised real is
      clipped at 99.5 and cannot be un-clipped), keyed by ``scan_id`` then
      ``patient_id``.

    This is the single canonical implementation; the ``rho_s_norm_audit`` preflight's
    ``_force_normalise`` mirrors it and should delegate here.

    Parameters
    ----------
    sample :
        A :class:`ScanSample` from :func:`iter_scans`.
    image_h5_map :
        ``cohort -> image-H5 path`` (raw ``images/t1c``). Consulted only for 99.95.
    percentile_upper :
        Target percentile; must be 99.5 or :data:`ENCODER_PERCENTILE_UPPER`.

    Returns
    -------
    ScanSample
        A frozen copy with ``pred`` and ``real`` replaced.

    Raises
    ------
    ValueError
        If *percentile_upper* is unsupported, or an ``image_h5_map`` entry is
        missing for the 99.95 path.
    KeyError
        If neither ``scan_id`` nor ``patient_id`` is present in the image H5.
    """
    if abs(percentile_upper - 99.5) < 1e-3:
        return replace(sample, pred=sample.pred_harmonised, real=sample.real)
    if abs(percentile_upper - ENCODER_PERCENTILE_UPPER) >= 1e-3:
        raise ValueError(
            f"Unsupported percentile_upper {percentile_upper}; must be 99.5 or "
            f"{ENCODER_PERCENTILE_UPPER}."
        )

    brain_t = torch.from_numpy(sample.brain.astype(np.float32))[None, None]
    pred_t = torch.from_numpy(sample.pred_raw)[None, None]
    new_pred = percentile_normalise(pred_t, upper=percentile_upper, mask=brain_t)[0, 0].numpy()

    h5_path = image_h5_map.get(sample.cohort)
    if h5_path is None:
        raise ValueError(
            f"percentile_upper={percentile_upper} requires an image_h5_map entry for "
            f"cohort {sample.cohort!r} (raw images/t1c is needed — the stored "
            "harmonised real is clipped at 99.5)."
        )
    with h5py.File(h5_path, "r") as f:
        ids = _decode_ids(f)
        idx = next(
            (ids.index(lid) for lid in (sample.scan_id, sample.patient_id) if lid in ids),
            None,
        )
        if idx is None:
            raise KeyError(
                f"Neither scan_id={sample.scan_id!r} nor patient_id="
                f"{sample.patient_id!r} found in {h5_path}"
            )
        real_raw = np.asarray(f["images/t1c"][idx], dtype=np.float32)
    real_t = torch.from_numpy(real_raw)[None, None]
    new_real = percentile_normalise(real_t, upper=percentile_upper, mask=brain_t)[0, 0].numpy()
    return replace(sample, pred=new_pred, real=new_real)


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


def discover_shards(root: Path) -> ShardDiscovery:
    """Discover inference shards and filter out smoke shards.

    Enumerates ``<root>/*/decision.json``.  Any shard whose
    ``decision.json["smoke"]["enabled"]`` is ``true`` is excluded from
    ``accepted`` and its tag is appended to ``skipped_smoke``.

    A shard with **no** ``smoke`` key in its ``decision.json`` is treated as a
    production shard (fail-open: older shards and BraTS-PED backfill shards
    that predate the smoke-flag convention must not be accidentally excluded).

    Parameters
    ----------
    root :
        Inference tree root (e.g. ``results/fm/inference``).

    Returns
    -------
    ShardDiscovery
        ``accepted`` production shards + ``skipped_smoke`` tag list.
    """
    root = Path(root)
    accepted: list[ShardInfo] = []
    skipped_smoke: list[str] = []

    for dec_path in sorted(root.glob("*/decision.json")):
        try:
            with dec_path.open() as fh:
                decision = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read %s: %s", dec_path, exc)
            continue

        smoke_block = decision.get("smoke", {})
        if smoke_block.get("enabled", False):
            tag = str(decision.get("run_id_tag", dec_path.parent.name))
            logger.info(
                "skipping smoke shard %r (dir=%s)",
                tag,
                dec_path.parent.name,
            )
            skipped_smoke.append(tag)
            continue

        accepted.append(ShardInfo(root=dec_path.parent, decision=decision))

    return ShardDiscovery(accepted=accepted, skipped_smoke=skipped_smoke)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def build_index(root: Path) -> pd.DataFrame:
    """Build a tidy index of every **production** prediction H5 under *root*.

    Calls :func:`discover_shards` internally to exclude smoke shards, then
    globs ``predictions/*/*/nfe_*.h5`` within each accepted shard only.

    After building the index the function asserts that
    ``(method, cohort, nfe, scan_id)`` is unique across all accepted shards.
    A duplicate that survived smoke-shard filtering means something unexpected
    is wrong (e.g. two production runs covering the same cohort in different
    shard directories) and must be loud rather than silently dropped.

    Parameters
    ----------
    root :
        Inference tree root.

    Returns
    -------
    pd.DataFrame
        One row per file, columns: ``method, cohort, ring, nfe, shard,
        path, references_h5, n_scans, schema_version``.

    Raises
    ------
    ValueError
        If the same ``(method, cohort, nfe, scan_id)`` appears in more than
        one production shard.
    """
    root = Path(root)
    discovery = discover_shards(root)

    # Tracks (method, cohort, nfe, scan_id) → first-seen shard for the raise.
    seen: dict[tuple[str, str, int, str], Path] = {}
    rows: list[dict] = []

    for shard_info in discovery.accepted:
        for pred_path in sorted(shard_info.root.glob("predictions/*/*/nfe_*.h5")):
            try:
                with h5py.File(pred_path, "r") as f:
                    schema = str(f.attrs.get("schema_version", ""))
                    if schema not in SUPPORTED_PRED_SCHEMAS:
                        logger.warning(
                            "skipping %s: unsupported schema_version=%r", pred_path, schema
                        )
                        continue
                    method = str(f.attrs.get("method", ""))
                    cohort = str(f.attrs.get("cohort", ""))
                    nfe = int(f.attrs.get("nfe", 0))
                    ring = str(f.attrs.get("ring", ""))
                    ref_rel = f.attrs.get("references_h5")
                    n_scans = int(f["predictions/t1c_synthetic_harmonised"].shape[0])
                    scan_ids_in_file = _decode_str_arr(f["metadata/scan_id"][:])
            except (OSError, KeyError) as exc:
                logger.warning("skipping %s: %s", pred_path, exc)
                continue

            # Defence in depth: duplicate scan_ids across production shards must raise.
            for sid in scan_ids_in_file:
                key = (method, cohort, nfe, sid)
                first = seen.get(key)
                if first is not None:
                    raise ValueError(
                        f"Duplicate (method, cohort, nfe, scan_id) found in production shards:\n"
                        f"  method={method!r}, cohort={cohort!r}, nfe={nfe}, scan_id={sid!r}\n"
                        f"  first seen in: {first}\n"
                        f"  also in:       {pred_path}\n"
                        "These are not smoke shards (those are already filtered). "
                        "Check for unexpected duplicate shard directories under "
                        f"{root}."
                    )
                seen[key] = pred_path

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
            ds_pred_raw_vol = f_pred["predictions/t1c_synthetic_raw"]

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
                harmonised_vol = np.asarray(ds_pred_vol[pi], dtype=np.float32)
                raw_vol = np.asarray(ds_pred_raw_vol[pi], dtype=np.float32)
                real_vol = np.asarray(ds_real[ri], dtype=np.float32)
                brain_arr = np.asarray(ds_brain[ri], dtype=bool)
                wt_arr = np.asarray(ds_wt[ri], dtype=bool)

                b = raw_vol[brain_arr]
                raw_p995 = float(np.percentile(b, 99.5)) if b.size > 0 else float("nan")
                pred_vol, pred_mode = select_scoring_volume(raw_vol, harmonised_vol, brain_arr)

                yield ScanSample(
                    scan_id=sid,
                    patient_id=pred_patient_ids[pi],
                    cohort=cohort,
                    ring=ring,
                    method=method,
                    nfe=nfe,
                    pred=pred_vol,
                    pred_raw=raw_vol,
                    pred_harmonised=harmonised_vol,
                    pred_mode=pred_mode,
                    raw_p995=raw_p995,
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
