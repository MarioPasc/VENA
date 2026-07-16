"""Thin engine for the downstream-seg routine (§4.4).

Wires :mod:`vena.validation.downstream_seg` to a YAML config, streams
through the shard predictions, and writes the artifact to disk.

The real arm (segmenter on real T1c) is computed **once per (cohort, scan_id)**
and cached across all 16 methods — not repeated per method.  The synthetic arm
only swaps the T1c channel; the other three harmonised volumes are identical
between arms.

Shardability: set ``methods`` and/or ``cohorts`` in the YAML to restrict a
single job to a subset; fan out to Picasso by (method, cohort) pairs.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from vena.validation.artifacts import (
    make_run_dir,
    symlink_latest,
    write_decision_json,
    write_per_scan_csv,
)
from vena.validation.downstream_seg import (
    BRATS_BUNDLE_VERSION,
    BRATS_INPUT_CHANNELS,
    BRATS_OUTPUT_CHANNELS,
    BratsSegmenter,
    CorpusLabelCache,
    SegResult,
    dice_score,
)
from vena.validation.io import (
    ReferenceCache,  # runtime use: instantiated in Engine.run()
    _decode_str_arr,
    _resolve_references_h5,
)
from vena.validation.registry import SELECTION_NFE

logger = logging.getLogger(__name__)


class DownstreamSegError(Exception):
    """Raised when the engine cannot proceed (missing gate, schema mismatch)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownstreamSegConfig:
    """Frozen configuration for the downstream-seg engine.

    Parameters
    ----------
    inference_root :
        Root of the inference shard tree.
    output_root :
        Parent directory for the artifact folder.
    bundle_path :
        Path to the downloaded ``brats_mri_segmentation`` bundle root.
    corpus_map :
        ``{cohort_name: /abs/path/to/<cohort>_image.h5}`` mapping used
        to look up multi-label tumour GT.
    ring_partitions_path :
        Path to ``ring_partitions.json`` written by ``vena-validation-preregister``.
    methods :
        Optional list of method names to run (default: all discovered).
    cohorts :
        Optional list of cohort names to run (default: all with corpus H5).
    device :
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    amp :
        Use AMP.  Ignored on CPU.
    selection_nfe_only :
        When True (default), run only at each method's ``SELECTION_NFE``.
    log_level :
        Python logging level name.
    """

    inference_root: Path
    output_root: Path
    bundle_path: Path
    corpus_map: dict[str, Path]
    ring_partitions_path: Path
    methods: list[str] = field(default_factory=list)
    cohorts: list[str] = field(default_factory=list)
    device: str = "cpu"
    amp: bool = False
    selection_nfe_only: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> DownstreamSegConfig:
        """Load from a YAML config file."""
        raw = yaml.safe_load(Path(path).read_text())
        corpus_raw: dict[str, str] = raw.get("corpus_map", {})
        corpus_map = {k: Path(v) for k, v in corpus_raw.items()}
        return cls(
            inference_root=Path(raw["inference_root"]),
            output_root=Path(raw["output_root"]),
            bundle_path=Path(raw["bundle_path"]),
            corpus_map=corpus_map,
            ring_partitions_path=Path(raw["ring_partitions_path"]),
            methods=list(raw.get("methods", [])),
            cohorts=list(raw.get("cohorts", [])),
            device=str(raw.get("device", "cpu")),
            amp=bool(raw.get("amp", False)),
            selection_nfe_only=bool(raw.get("selection_nfe_only", True)),
            log_level=str(raw.get("log_level", "INFO")),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_pred_files(
    inference_root: Path,
    *,
    method_filter: list[str],
    cohort_filter: list[str],
    selection_nfe: dict[str, int],
    selection_nfe_only: bool,
) -> list[tuple[str, str, int, Path]]:
    """Glob inference_root for prediction H5 files, filtered per config.

    Returns list of (method, cohort, nfe, path) tuples.
    """
    pattern = "*/predictions/*/*/nfe_*.h5"
    files = sorted(inference_root.glob(pattern))
    results: list[tuple[str, str, int, Path]] = []
    for p in files:
        # predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5
        try:
            nfe = int(p.stem.split("_")[-1])
            cohort = p.parent.name
            method = p.parent.parent.name
        except (ValueError, IndexError):
            logger.debug("skipping unparseable path: %s", p)
            continue

        if method_filter and method not in method_filter:
            continue
        if cohort_filter and cohort not in cohort_filter:
            continue
        if selection_nfe_only:
            target_nfe = selection_nfe.get(method)
            if target_nfe is None:
                logger.warning(
                    "no selection_nfe for method %s — skipping (set selection_nfe_only: false to include)",
                    method,
                )
                continue
            if nfe != target_nfe:
                continue

        results.append((method, cohort, nfe, p))

    logger.info("discovered %d prediction files", len(results))
    return results


def _read_pred_row(pred_path: Path, row_idx: int, *, ref_cache: ReferenceCache) -> dict[str, Any]:
    """Read one row from a prediction H5 at *row_idx*.

    Returns dict with keys: scan_id, patient_id, t1c_synth, t1c_real.
    T1pre / T2 / FLAIR are read separately from the corpus H5 via
    :class:`vena.validation.downstream_seg.CorpusLabelCache`
    (they are identical between real and synthetic arms).

    Parameters
    ----------
    pred_path :
        Path to a ``predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5`` file.
    row_idx :
        0-based scan index within the prediction file.
    ref_cache :
        Shared :class:`~vena.validation.io.ReferenceCache` instance for
        caching the reference scan-id → row-index map across method iterations.

    Returns
    -------
    dict
        Keys: ``scan_id``, ``patient_id``, ``t1c_synth``, ``t1c_real``.
        All volumes are ``(H, W, D)`` float32.
    """
    with h5py.File(pred_path, "r") as pf:
        scan_ids = _decode_str_arr(pf["metadata/scan_id"][:])
        patient_ids = _decode_str_arr(pf["metadata/patient_id"][:])
        scan_id = scan_ids[row_idx]
        patient_id = patient_ids[row_idx]
        t1c_synth = pf["predictions/t1c_synthetic_harmonised"][row_idx]  # (H,W,D)

        # _resolve_references_h5 must be called while pf is open (reads attrs)
        ref_h5 = _resolve_references_h5(pf, pred_path)

    if not ref_h5.is_file():
        raise FileNotFoundError(f"reference H5 not found for {pred_path}: {ref_h5}")

    ref_idx_map = ref_cache.get_scan_index(ref_h5)
    ref_row = ref_idx_map.get(scan_id)
    if ref_row is None:
        raise KeyError(f"scan_id {scan_id!r} not in reference H5 {ref_h5}")

    # Reference H5 stores all harmonised input modalities once per cohort
    # (schema 2.0 design — verified in conftest and prod shard writer).
    with h5py.File(ref_h5, "r") as rf:
        t1c_real = rf["reference/t1c_real_harmonised"][ref_row]
        t1pre = rf["reference/t1pre_harmonised"][ref_row]
        t2 = rf["reference/t2_harmonised"][ref_row]
        flair = rf["reference/flair_harmonised"][ref_row]

    return {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "t1c_synth": t1c_synth.astype(np.float32),
        "t1c_real": t1c_real.astype(np.float32),
        "t1pre": t1pre.astype(np.float32),
        "t2": t2.astype(np.float32),
        "flair": flair.astype(np.float32),
    }


def _seg_to_dice(
    segmenter: BratsSegmenter,
    t1c: np.ndarray,
    t1pre: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    gt_wt: np.ndarray,
    gt_tc: np.ndarray,
    gt_et: np.ndarray,
) -> tuple[float, float, float]:
    """Run segmenter and return (dice_wt, dice_tc, dice_et)."""
    tc_pred, wt_pred, et_pred = segmenter.segment(t1c, t1pre, t2, flair)
    d_wt = dice_score(wt_pred, gt_wt)
    d_tc = dice_score(tc_pred, gt_tc)
    d_et = dice_score(et_pred, gt_et)
    return d_wt, d_tc, d_et


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class DownstreamSegEngine:
    """Execute the downstream-seg routine and write the artifact."""

    cfg: DownstreamSegConfig

    def run(self) -> Path:
        """Run the routine and return the artifact directory.

        Returns
        -------
        Path
            The run directory containing ``per_scan/downstream_seg.csv``.
        """
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )

        # Load ring partitions (drives COHORT_RING, used to annotate CSV)
        from vena.validation.registry import load_partitions

        load_partitions(self.cfg.ring_partitions_path)
        from vena.validation.registry import COHORT_RING

        # Build effective selection_nfe (registry + user override falls through)
        sel_nfe: dict[str, int] = dict(SELECTION_NFE)

        # Discover prediction files
        pred_files = _discover_pred_files(
            self.cfg.inference_root,
            method_filter=self.cfg.methods,
            cohort_filter=self.cfg.cohorts,
            selection_nfe=sel_nfe,
            selection_nfe_only=self.cfg.selection_nfe_only,
        )
        if not pred_files:
            raise DownstreamSegError(f"No prediction files found under {self.cfg.inference_root}")

        # Instantiate the corpus label cache (lazy H5 opens)
        corpus_cache = CorpusLabelCache(self.cfg.corpus_map)

        # Instantiate the segmenter (loads checkpoint — no network access)
        segmenter = BratsSegmenter(
            self.cfg.bundle_path,
            device=self.cfg.device,
            amp=self.cfg.amp,
        )

        # Shared reference-scan-index cache: avoids re-reading metadata/scan_id
        # from the same reference H5 for each of the 16 methods.
        ref_cache = ReferenceCache()

        # Real-arm cache: (cohort, scan_id) → (dice_wt, dice_tc, dice_et)
        # Computed ONCE per scan, reused across all methods.
        real_arm_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        real_arm_call_count = 0

        run_dir = make_run_dir(self.cfg.output_root, "downstream_seg")
        logger.info("artifact dir: %s", run_dir)

        rows: list[dict[str, Any]] = []
        skipped_no_corpus: set[str] = set()
        skipped_no_scan: list[str] = []

        t_start = time.monotonic()

        for method, cohort, nfe, pred_path in pred_files:
            ring = COHORT_RING.get(cohort, "?")

            if not corpus_cache.has_cohort(cohort):
                if cohort not in skipped_no_corpus:
                    logger.warning("cohort %s has no corpus H5 with masks/tumor — skipping", cohort)
                    skipped_no_corpus.add(cohort)
                continue

            # Count scans in this prediction file
            with h5py.File(pred_path, "r") as pf:
                n_scans = pf["metadata/scan_id"].shape[0]

            logger.info(
                "processing method=%s cohort=%s nfe=%d (%d scans)",
                method,
                cohort,
                nfe,
                n_scans,
            )

            for row_idx in range(n_scans):
                try:
                    data = _read_pred_row(pred_path, row_idx, ref_cache=ref_cache)
                except (KeyError, FileNotFoundError) as exc:
                    logger.warning("skipping scan idx %d in %s: %s", row_idx, pred_path, exc)
                    skipped_no_scan.append(f"{method}/{cohort}/idx{row_idx}: {exc}")
                    continue

                scan_id = data["scan_id"]
                patient_id = data["patient_id"]

                # Fetch GT labels from corpus H5.
                try:
                    gt_wt, gt_tc, gt_et = corpus_cache.get_labels(cohort, scan_id)
                except KeyError as exc:
                    logger.warning("GT labels missing for %s/%s: %s", cohort, scan_id, exc)
                    skipped_no_scan.append(f"{method}/{cohort}/{scan_id}: no GT")
                    continue

                # T1pre / T2 / FLAIR come from the reference H5 (identical for
                # both arms; only T1c differs between real and synthetic).
                t1pre = data["t1pre"]
                t2 = data["t2"]
                flair = data["flair"]

                # Real arm — compute once per (cohort, scan_id)
                cache_key = (cohort, scan_id)
                if cache_key not in real_arm_cache:
                    real_arm_call_count += 1
                    logger.debug(
                        "real arm #%d: cohort=%s scan=%s", real_arm_call_count, cohort, scan_id
                    )
                    d_wt_r, d_tc_r, d_et_r = _seg_to_dice(
                        segmenter,
                        data["t1c_real"],
                        t1pre,
                        t2,
                        flair,
                        gt_wt,
                        gt_tc,
                        gt_et,
                    )
                    real_arm_cache[cache_key] = (d_wt_r, d_tc_r, d_et_r)
                else:
                    d_wt_r, d_tc_r, d_et_r = real_arm_cache[cache_key]

                # Synthetic arm — always recompute (only T1c channel differs)
                d_wt_s, d_tc_s, d_et_s = _seg_to_dice(
                    segmenter,
                    data["t1c_synth"],
                    t1pre,
                    t2,
                    flair,
                    gt_wt,
                    gt_tc,
                    gt_et,
                )

                result = SegResult(
                    method=method,
                    cohort=cohort,
                    ring=ring,
                    nfe=nfe,
                    scan_id=scan_id,
                    patient_id=patient_id,
                    dice_wt_real=d_wt_r,
                    dice_tc_real=d_tc_r,
                    dice_et_real=d_et_r,
                    dice_wt_synth=d_wt_s,
                    dice_tc_synth=d_tc_s,
                    dice_et_synth=d_et_s,
                )
                rows.append(
                    {
                        "method": result.method,
                        "cohort": result.cohort,
                        "ring": result.ring,
                        "nfe": result.nfe,
                        "scan_id": result.scan_id,
                        "patient_id": result.patient_id,
                        "dice_wt_real": result.dice_wt_real,
                        "dice_tc_real": result.dice_tc_real,
                        "dice_et_real": result.dice_et_real,
                        "dice_wt_synth": result.dice_wt_synth,
                        "dice_tc_synth": result.dice_tc_synth,
                        "dice_et_synth": result.dice_et_synth,
                        "delta_wt": result.delta_wt,
                        "delta_tc": result.delta_tc,
                        "delta_et": result.delta_et,
                    }
                )

        corpus_cache.close()
        wall_clock_s = time.monotonic() - t_start

        logger.info(
            "real arm calls: %d (cache hits: %d)",
            real_arm_call_count,
            len(rows) - real_arm_call_count,
        )
        logger.info("wall clock: %.1f s", wall_clock_s)

        if not rows:
            logger.warning("no rows produced — check corpus_map and inference_root")

        df = pd.DataFrame(
            rows,
            columns=[
                "method",
                "cohort",
                "ring",
                "nfe",
                "scan_id",
                "patient_id",
                "dice_wt_real",
                "dice_tc_real",
                "dice_et_real",
                "dice_wt_synth",
                "dice_tc_synth",
                "dice_et_synth",
                "delta_wt",
                "delta_tc",
                "delta_et",
            ],
        )

        csv_path = write_per_scan_csv(run_dir, df, name="downstream_seg.csv")
        logger.info("wrote %s (%d rows)", csv_path, len(df))

        # Build summary stats for decision.json
        n_scans_processed = len(set((r["cohort"], r["scan_id"]) for r in rows))
        n_real_arm_unique = real_arm_call_count

        import hashlib

        bundle_sha = hashlib.sha256(
            (self.cfg.bundle_path / "models" / "model.pt").read_bytes()
        ).hexdigest()

        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "producer": "routines.validation.downstream_seg:1.0",
            "inference_root": str(self.cfg.inference_root),
            "output_root": str(self.cfg.output_root),
            "bundle_path": str(self.cfg.bundle_path),
            "bundle_version": BRATS_BUNDLE_VERSION,
            "bundle_model_sha256": bundle_sha,
            "bundle_input_channel_order": list(BRATS_INPUT_CHANNELS),
            "bundle_output_channel_order": list(BRATS_OUTPUT_CHANNELS),
            "bundle_preprocessing": "NormalizeIntensityd(nonzero=True, channel_wise=True)",
            "bundle_inferer": "SlidingWindowInferer(roi_size=[240,240,160], overlap=0.5)",
            "device": self.cfg.device,
            "amp": self.cfg.amp,
            "selection_nfe_only": self.cfg.selection_nfe_only,
            "methods_requested": self.cfg.methods or "all",
            "cohorts_requested": self.cfg.cohorts or "all",
            "n_pred_files": len(pred_files),
            "n_scans_processed": n_scans_processed,
            "n_real_arm_unique_calls": n_real_arm_unique,
            "n_rows_csv": len(df),
            "skipped_cohorts_no_corpus": sorted(skipped_no_corpus),
            "skipped_scans_n": len(skipped_no_scan),
            "wall_clock_s": round(wall_clock_s, 1),
            "empty_et_convention": "NaN when both pred and GT are empty (not 0)",
            "appendix_a_deviation": (
                "Used fixed pretrained BraTS segmenter instead of per-cohort "
                "nnU-Net from scratch. Level confounder cancels in paired Δ; "
                "see 05_downstream_seg.md §2 and report.md."
            ),
            "corpus_map": {k: str(v) for k, v in self.cfg.corpus_map.items()},
        }

        write_decision_json(run_dir, payload)
        symlink_latest(run_dir)

        # Copy config for reproducibility
        shutil.copy(
            self.cfg.ring_partitions_path,
            run_dir / "ring_partitions.json",
        )

        logger.info("downstream_seg done: %s", run_dir)
        return run_dir
