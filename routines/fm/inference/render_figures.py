"""Re-render per-cohort multi-method comparison PNGs from an existing
predictions tree.

The unified inference engine renders the cross-method PNG in-process at
the end of `Engine.run()` using the in-memory `per_cohort_selection_pred`
dict. When the benchmark is split across multiple SLURM jobs (vena +
vena-syndiff envs, or per-method jobs in the future), each job sees
only its own methods, so the in-process figure misses the cross-env
methods. This standalone script reads every method's predictions H5
from disk and re-renders the full multi-method figure per cohort.

Usage::

    python -m routines.fm.inference.render_figures \\
        --predictions-root /abs/path/to/<run_id_tag>/predictions \\
        --out-dir          /abs/path/to/<run_id_tag>/figures \\
        --selection-nfe-yaml routines/fm/inference/configs/models/benchmark_full.yaml \\
        [--patient-id PID]      # per cohort, defaults to first in metadata
        [--n-slices 7]          # axial slices per row
        [--methods C0 C1 ...]   # order; default = alphabetical
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _decode_str(ds: object) -> list[str]:
    out: list[str] = []
    for b in ds:  # type: ignore[union-attr]
        out.append(b.decode() if isinstance(b, bytes) else str(b))
    return out


def _selection_nfe_map(models_yaml: Path) -> dict[str, int]:
    """Return {method_name: selection_nfe} from the models registry YAML."""
    raw = yaml.safe_load(models_yaml.read_text())
    out: dict[str, int] = {}
    for m in raw.get("methods", []):
        name = m["name"]
        kwargs = m.get("kwargs", {})
        out[name] = int(kwargs.get("selection_nfe", 1))
    return out


def _discover_methods(predictions_root: Path) -> list[str]:
    return sorted(p.name for p in predictions_root.iterdir() if p.is_dir())


def _discover_cohorts(predictions_root: Path, method: str) -> list[str]:
    method_dir = predictions_root / method
    if not method_dir.is_dir():
        return []
    return sorted(p.name for p in method_dir.iterdir() if p.is_dir())


def _h5_for_method_cohort_nfe(
    predictions_root: Path, method: str, cohort: str, nfe: int
) -> Path | None:
    f = predictions_root / method / cohort / f"nfe_{int(nfe):03d}.h5"
    return f if f.is_file() else None


def _load_synth(
    h5_path: Path, patient_id: str | None
) -> tuple[str, np.ndarray, np.ndarray, int, float] | None:
    """Return (patient_id, synth_volume, real_t1c_volume, nfe, seconds).

    Returns ``None`` on any read failure (corrupt H5, missing patient,
    schema deviation) so the caller can skip the method-row without
    aborting the whole figure batch.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            pids = _decode_str(f["metadata/patient_id"][:])
            if not pids:
                return None
            if patient_id is None:
                idx = 0
                pid = pids[0]
            elif patient_id in pids:
                idx = pids.index(patient_id)
                pid = patient_id
            else:
                return None
            synth = np.asarray(f["predictions/t1c_synthetic_harmonised"][idx], dtype=np.float32)
            real = np.asarray(f["reference/t1c_real_harmonised"][idx], dtype=np.float32)
            nfe = int(f["metadata/nfe"][idx])
            seconds = float(f["metadata/inference_seconds"][idx])
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("skipping %s: %s", h5_path, exc)
        return None
    return pid, synth, real, nfe, seconds


def render_for_cohort(
    predictions_root: Path,
    cohort: str,
    out_path: Path,
    method_order: list[str],
    selection_nfe: dict[str, int],
    patient_id: str | None,
    n_slices: int,
    slice_offset: int,
) -> Path | None:
    """Render one cohort's unified comparison PNG and return its path."""
    from vena.inference.figure import render_multi_method_figure

    # Pick the patient: take the first method's predictions H5 (at its
    # selection NFE), read patient ids, and use the requested pid or the
    # first one. The reference T1c is shared across methods so any method
    # works for the patient discovery.
    real_t1c: np.ndarray | None = None
    chosen_pid: str | None = patient_id
    method_predictions: list[tuple[str, np.ndarray, int, float]] = []
    for method in method_order:
        nfe = selection_nfe.get(method, 1)
        h5_path = _h5_for_method_cohort_nfe(predictions_root, method, cohort, nfe)
        if h5_path is None:
            logger.warning("missing H5 for %s/%s @ NFE=%d", method, cohort, nfe)
            continue
        loaded = _load_synth(h5_path, chosen_pid)
        if loaded is None:
            logger.warning("patient %r not in %s", chosen_pid, h5_path)
            continue
        pid, synth, real, h5_nfe, seconds = loaded
        if real_t1c is None:
            real_t1c = real
            chosen_pid = pid  # lock in for later methods (in case patient_id was None)
        method_predictions.append((method, synth, h5_nfe, seconds))

    if real_t1c is None or not method_predictions or chosen_pid is None:
        logger.warning("no usable methods for cohort %s; skipping figure", cohort)
        return None

    return render_multi_method_figure(
        cohort=cohort,
        patient_id=chosen_pid,
        real_t1c=real_t1c,
        method_predictions=method_predictions,
        out_path=out_path,
        n_slices=n_slices,
        slice_offset=slice_offset,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-nfe-yaml", type=Path, required=True)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--n-slices", type=int, default=7)
    parser.add_argument("--slice-offset", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="explicit method order; default = alphabetical from predictions/",
    )
    parser.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help="explicit cohort filter; default = every cohort with at least one method",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")

    selection_nfe = _selection_nfe_map(args.selection_nfe_yaml)
    methods = args.methods or _discover_methods(args.predictions_root)
    logger.info("methods (%d): %s", len(methods), methods)

    if args.cohorts:
        cohorts = list(args.cohorts)
    else:
        seen: set[str] = set()
        for m in methods:
            seen.update(_discover_cohorts(args.predictions_root, m))
        cohorts = sorted(seen)
    logger.info("cohorts (%d): %s", len(cohorts), cohorts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cohort in cohorts:
        out_path = args.out_dir / f"{cohort}.png"
        result = render_for_cohort(
            args.predictions_root,
            cohort,
            out_path,
            methods,
            selection_nfe,
            args.patient_id,
            args.n_slices,
            args.slice_offset,
        )
        if result is not None:
            logger.info("wrote %s", result)
            written += 1
    logger.info("done — wrote %d / %d figures", written, len(cohorts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
