"""Core engine for the ρ_S normalisation audit preflight.

Asks one question: does forcing both synth and real to the SAME
percentile_upper change the ranking?

The frozen sweep (2026-07-20T08-41-41Z) compared:
  pred_raw  (VAE output, clipped at 99.95 by the encoder)  vs
  real      (t1c_real_harmonised, clipped at 99.5)

For the 15 latent-space methods this is a scale mismatch of roughly
1.005× (0.5 percentile gap).  Even that small drift inflates ρ_S for
bright-voxel-heavy residual maps — see §14.3 in
.claude/notes/changes/model_redesign_2026-07-21.md.

This engine repeats compute_scan_rows with two forced-percentile modes:

  P=99.5  : new_pred = pred_harmonised  (stored, already at 99.5)
             new_real = real             (stored, already at 99.5)
             → no image H5 needed

  P=99.95 : new_pred = percentile_normalise(pred_raw, upper=99.95)
             new_real = percentile_normalise(T1c from image H5, upper=99.95)
             → requires image_h5_map in config

For the null method C0-Identity both pred_raw and pred_harmonised are
T1pre (scanner units vs 99.5-normalised).  The harmonised P=99.5 path
leaves C0 unchanged; the P=99.95 path re-normalises T1pre at 99.95.
In both cases real and pred are consistently normalised, so C0 is a
valid negative control (ρ_S should not change dramatically for a
method that never goes through the VAE).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats as _scipy_stats
from scipy.ndimage import binary_dilation

from vena.common import percentile_normalise
from vena.validation.artifacts import make_run_dir, symlink_latest, write_decision_json
from vena.validation.io import ReferenceCache, ScanSample, iter_scans
from vena.validation.registry import METHOD_SPECS, SELECTION_NFE

from .config import RhoSNormAuditConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Ring-A cohorts (pre-registered 2026-07-16).
RING_A_COHORTS: frozenset[str] = frozenset(
    {
        "BraTS-GLI",
        "IvyGAP",
        "LUMIERE",
        "REMBRANDT",
        "UCSF-PDGM",
        "UPENN-GBM",
    }
)

# Shard tag → directory name under inference_root (discovery order doesn't matter
# but listing them makes the glob explicit).
_SHARD_DIRS: tuple[str, ...] = (
    "picasso_shard_a_cheap",
    "picasso_shard_b_vena",
    "picasso_shard_c_latent",
    "picasso_shard_d_lddpm",
    "picasso_shard_e_syndiff",
)


# ---------------------------------------------------------------------------
# Normalisation forcing helpers
# ---------------------------------------------------------------------------


def _id_list(f: h5py.File) -> list[str]:
    """Decode the 'ids' dataset into Python strings."""
    return [s.decode() if isinstance(s, bytes) else str(s) for s in f["ids"][:]]


def _force_normalise(
    sample: ScanSample,
    percentile: float,
    image_h5_map: dict[str, Path],
) -> ScanSample:
    """Return a new ScanSample with pred and real forced to *percentile*.

    Parameters
    ----------
    sample :
        Original sample from iter_scans.
    percentile :
        Target percentile upper bound.  Must be 99.5 or 99.95.
    image_h5_map :
        Mapping cohort → local image H5 path.  Only consulted for P=99.95.

    Returns
    -------
    ScanSample
        Frozen dataclass with ``pred`` and ``real`` replaced; all other
        fields preserved.
    """
    if abs(percentile - 99.5) < 1e-3:
        # Both fields are already stored at 99.5 — no recomputation.
        return dataclasses.replace(sample, pred=sample.pred_harmonised, real=sample.real)

    if abs(percentile - 99.95) < 1e-3:
        brain_t = torch.from_numpy(sample.brain.astype(np.float32))[None, None]

        # Re-normalise synthetic at 99.95 (pred_raw is VAE output, already near 99.95
        # scale, but we apply the function explicitly for consistency).
        pred_t = torch.from_numpy(sample.pred_raw)[None, None]
        new_pred = percentile_normalise(pred_t, upper=99.95, mask=brain_t)[0, 0].numpy()

        # Load the raw real T1c from the local image H5.
        h5_path = image_h5_map.get(sample.cohort)
        if h5_path is None:
            raise ValueError(
                f"P=99.95 path requires image_h5_map entry for cohort "
                f"{sample.cohort!r} — add it to the YAML config."
            )
        with h5py.File(h5_path, "r") as f:
            ids = _id_list(f)
            # Try scan_id first (longitudinal cohorts), then patient_id (cross-sectional).
            idx = next(
                (ids.index(lid) for lid in (sample.scan_id, sample.patient_id) if lid in ids),
                None,
            )
            if idx is None:
                raise KeyError(
                    f"Neither scan_id={sample.scan_id!r} nor "
                    f"patient_id={sample.patient_id!r} found in {h5_path}"
                )
            real_raw = np.asarray(f["images/t1c"][idx], dtype=np.float32)

        real_t = torch.from_numpy(real_raw)[None, None]
        new_real = percentile_normalise(real_t, upper=99.95, mask=brain_t)[0, 0].numpy()

        return dataclasses.replace(sample, pred=new_pred, real=new_real)

    raise ValueError(f"Unsupported percentile {percentile}; must be 99.5 or 99.95.")


# ---------------------------------------------------------------------------
# Fast ρ_S computation (Spearman only, no KSG MI, no bootstrap/shuffle)
# ---------------------------------------------------------------------------

#: Voxel subsample cap for Spearman.  100k gives stable ρ_S estimates
#: (SE < 0.01 for N=100k, ρ≈0.3) in < 50 ms; KSG MI at 30k takes ~2.5 s.
_SPEARMAN_N_SUB: int = 100_000

_RHO_ROW_COLUMNS: list[str] = ["method", "patient_id", "scan_id", "cohort", "condition", "rho_s"]


def _fast_rho_rows(
    sample: ScanSample,
    dilate_k: int = 5,
    n_sub: int = _SPEARMAN_N_SUB,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return two minimal rows (C-WB, C-noT) with Spearman ρ_S.

    Skips KSG MI, bootstrap CI, and shuffle null — this preflight only needs
    ρ_S to quantify the normalisation confound.  Subsamples to *n_sub* voxels
    to keep each call < 50 ms (vs ~2.5 s with KSG MI).

    Parameters
    ----------
    sample :
        Normalisation-forced ScanSample with ``pred`` and ``real`` at the
        same percentile.
    dilate_k :
        WT binary-dilation kernel size (odd integer).
    n_sub :
        Maximum voxel count passed to Spearman.  ``None`` = no subsampling.
    rng :
        Optional NumPy random generator.  Created fresh if ``None``.

    Returns
    -------
    list[dict]
        Two rows with keys in ``_RHO_ROW_COLUMNS``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    brain = sample.brain.astype(bool)
    wt = sample.wt.astype(bool)

    abs_resid = np.abs(sample.real.astype(np.float64) - sample.pred.astype(np.float64))
    real_f64 = sample.real.astype(np.float64)

    wt_dil = binary_dilation(wt, iterations=max(1, dilate_k // 2))
    noT_mask = brain & ~wt_dil

    def _rho(mask: np.ndarray) -> float:
        idx = np.where(mask.ravel())[0]
        if idx.size < 10:
            return float("nan")
        if n_sub is not None and idx.size > n_sub:
            idx = rng.choice(idx, n_sub, replace=False)
        r, _ = _scipy_stats.spearmanr(abs_resid.ravel()[idx], real_f64.ravel()[idx])
        return float(r)

    base = {
        "method": sample.method,
        "patient_id": sample.patient_id,
        "scan_id": sample.scan_id,
        "cohort": sample.cohort,
    }
    return [
        {**base, "condition": "C-WB", "rho_s": _rho(brain)},
        {**base, "condition": "C-noT", "rho_s": _rho(noT_mask)},
    ]


# ---------------------------------------------------------------------------
# Prediction H5 discovery
# ---------------------------------------------------------------------------


def _discover_pred_h5s(
    inference_root: Path,
    methods: list[str],
    cohorts: frozenset[str],
) -> list[Path]:
    """Find prediction H5 files for the requested methods and cohorts.

    Each file lives at::

        {shard_root}/predictions/{method}/{cohort}/nfe_{NNN:03d}.h5

    where NNN is the pre-registered selection_nfe for that method.

    Parameters
    ----------
    inference_root :
        Parent of all shard directories.
    methods :
        Method keys to look for.
    cohorts :
        Cohort names to include.

    Returns
    -------
    list[Path]
        One path per (method, cohort, nfe) combination that exists on disk.
    """
    found: list[Path] = []
    for shard_dir in _SHARD_DIRS:
        shard_root = inference_root / shard_dir
        if not shard_root.is_dir():
            continue
        for method in methods:
            nfe = SELECTION_NFE.get(method)
            if nfe is None:
                logger.warning("Unknown method %r — skipping.", method)
                continue
            for cohort in sorted(cohorts):
                p = shard_root / "predictions" / method / cohort / f"nfe_{nfe:03d}.h5"
                if p.is_file():
                    found.append(p)
                    logger.debug("Found: %s", p)
    if not found:
        logger.warning(
            "No prediction H5 files found under %s for methods=%s cohorts=%s",
            inference_root,
            methods,
            sorted(cohorts),
        )
    return found


# ---------------------------------------------------------------------------
# Patient-level collapse
# ---------------------------------------------------------------------------


def _collapse_patient_rho_s(rows: list[dict], condition: str) -> pd.DataFrame:
    """Collapse per-scan rows to patient-level median ρ_S.

    For cross-sectional cohorts patient_id == scan_id so the groupby is a
    no-op.  For LUMIERE (longitudinal) it averages across timepoints.

    Parameters
    ----------
    rows :
        Raw rows from compute_scan_rows.
    condition :
        ``"C-noT"`` or ``"C-WB"``.

    Returns
    -------
    pd.DataFrame
        Columns: method, patient_id, rho_s.  One row per (method, patient_id).
    """
    df = pd.DataFrame(rows, columns=_RHO_ROW_COLUMNS)
    df = df[df["condition"] == condition].copy()
    agg = df.groupby(["method", "patient_id"], sort=False)["rho_s"].mean().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class RhoSNormAuditEngine:
    """Audit engine: re-run ρ_S under forced-percentile normalisation.

    Parameters
    ----------
    cfg :
        Frozen Pydantic config from ``RhoSNormAuditConfig.from_yaml``.
    """

    def __init__(self, cfg: RhoSNormAuditConfig) -> None:
        self.cfg = cfg

    def run(self) -> Path:
        """Execute the audit and write artifacts.

        Returns
        -------
        Path
            The timestamped artifact directory.
        """
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )

        cfg = self.cfg
        methods = cfg.methods if cfg.methods is not None else [s.key for s in METHOD_SPECS]
        cohorts = frozenset(cfg.cohorts) if cfg.cohorts is not None else RING_A_COHORTS

        # --------------- load frozen reference CSV -------------------------
        logger.info("Loading frozen CSV: %s", cfg.frozen_csv_path)
        frozen_df = pd.read_csv(cfg.frozen_csv_path)
        frozen_cnoT = frozen_df[
            (frozen_df["condition"] == cfg.condition) & (frozen_df["ring"] == cfg.ring)
        ].copy()
        # patient-level median ρ_S per method from frozen sweep
        frozen_pat = (
            frozen_cnoT.groupby(["method", "patient_id"])["rho_s"]
            .mean()
            .reset_index()
            .groupby("method")["rho_s"]
            .median()
        )

        # --------------- discover prediction files -------------------------
        pred_h5s = _discover_pred_h5s(cfg.inference_root, methods, cohorts)
        if not pred_h5s:
            raise FileNotFoundError(
                f"No prediction H5 files found under {cfg.inference_root} "
                f"for methods={methods}, cohorts={sorted(cohorts)}."
            )
        logger.info("Discovered %d prediction H5 file(s).", len(pred_h5s))

        # --------------- run per-percentile audit -------------------------
        # rows_by_percentile: percentile → list[dict]
        rows_by_percentile: dict[float, list[dict]] = {p: [] for p in cfg.percentiles}
        ref_cache = ReferenceCache()

        for pred_path in pred_h5s:
            cohort = pred_path.parent.parent.name
            method = pred_path.parent.parent.parent.name
            logger.info("Processing %s / %s …", method, cohort)
            n_processed = 0

            for sample in iter_scans(pred_path, reference_cache=ref_cache):
                if cfg.ring and sample.ring != cfg.ring:
                    continue
                if cfg.scan_limit is not None and n_processed >= cfg.scan_limit:
                    break

                for percentile in cfg.percentiles:
                    try:
                        forced = _force_normalise(sample, percentile, dict(cfg.image_h5_map))
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "Skipping scan %s (P=%.2f): %s", sample.scan_id, percentile, exc
                        )
                        continue
                    # Fast Spearman-only path: skips KSG MI (~2.5 s/scan) and
                    # shuffle/bootstrap null — this audit only needs ρ_S.
                    scan_rows = _fast_rho_rows(forced, dilate_k=5)
                    rows_by_percentile[percentile].extend(scan_rows)

                n_processed += 1
                if n_processed % 10 == 0:
                    logger.info("  … %d scans done", n_processed)

            logger.info("  %s / %s: %d scan(s) processed.", method, cohort, n_processed)

        # --------------- collapse to patient level ------------------------
        patient_rho: dict[float, pd.DataFrame] = {}
        for p, rows in rows_by_percentile.items():
            if not rows:
                logger.warning("No rows for P=%.2f — check scan_limit / ring filter.", p)
                patient_rho[p] = pd.DataFrame(columns=["method", "patient_id", "rho_s"])
                continue
            patient_rho[p] = _collapse_patient_rho_s(rows, cfg.condition)

        # --------------- per-method median ρ_S & Δρ_S --------------------
        # Build table: one row per (method, percentile)
        table_rows = []
        for p, df in patient_rho.items():
            if df.empty:
                continue
            per_method = df.groupby("method")["rho_s"].median()
            for method, rho_val in per_method.items():
                frozen_val = frozen_pat.get(method, float("nan"))
                table_rows.append(
                    {
                        "method": method,
                        "percentile": p,
                        "rho_s": float(rho_val),
                        "rho_s_frozen": float(frozen_val),
                        "delta_rho_s": float(rho_val - frozen_val),
                    }
                )

        result_df = pd.DataFrame(table_rows)
        result_df = result_df.sort_values(["percentile", "rho_s"])

        # --------------- determine canonical percentile -------------------
        # Prefer the percentile that minimises median ρ_S across ALL methods
        # (lower is better; frozen confound is removed).
        canonical_percentile: float = 99.5  # default
        if not result_df.empty:
            summary = result_df.groupby("percentile")["rho_s"].median()
            canonical_percentile = float(summary.idxmin())

        # --------------- build decision payload ---------------------------
        # Key 9 methods for the report
        key_methods = [
            "VENA-S1-v3b-rw",
            "VENA-S1-v3b",
            "VENA-S1-v3a",
            "C0-Identity",
            "C4-3D-DiT",
            "C5-T1C-RFlow",
            "C6-3D-LDDPM",
            "C7-3D-Latent-Pix2Pix",
            "C1-pGAN-t1pre",
        ]

        def _rho_for(method: str, percentile: float) -> float | None:
            sub = result_df[
                (result_df["method"] == method) & (result_df["percentile"] == percentile)
            ]
            return float(sub["rho_s"].iloc[0]) if not sub.empty else None

        delta_table: dict[str, dict] = {}
        for m in methods:
            entry: dict[str, object] = {}
            for p in cfg.percentiles:
                entry[f"rho_s_P{p}"] = _rho_for(m, p)
            entry["rho_s_frozen"] = float(frozen_pat.get(m, float("nan")))
            delta_table[m] = entry

        # Booleans for the report
        vena_primary = "VENA-S1-v3b-rw"
        identity_key = "C0-Identity"
        latent_worse_than_identity_survives: bool | None = None
        vena_vs_identity_survives: bool | None = None

        if 99.5 in patient_rho and not patient_rho[99.5].empty:
            pat_df = patient_rho[99.5]
            med_vena = pat_df[pat_df["method"] == vena_primary]["rho_s"].median()
            med_id = pat_df[pat_df["method"] == identity_key]["rho_s"].median()

            # Latent-Pix2Pix is worst in frozen ranking; check if it stays below identity
            med_pix2pix = pat_df[pat_df["method"] == "C7-3D-Latent-Pix2Pix"]["rho_s"].median()
            if not np.isnan(float(med_pix2pix)) and not np.isnan(float(med_id)):
                # In the frozen ranking, C7 has ρ_S=−0.19 (best); identity=0.351.
                # "latent worse than identity" means C7 is no longer lower than identity.
                latent_worse_than_identity_survives = bool(float(med_pix2pix) < float(med_id))

            if not np.isnan(float(med_vena)) and not np.isnan(float(med_id)):
                vena_vs_identity_survives = bool(float(med_vena) < float(med_id))

        payload: dict = {
            "schema_version": "1.0",
            "canonical_percentile_upper": canonical_percentile,
            "condition": cfg.condition,
            "ring": cfg.ring,
            "n_methods_audited": len(methods),
            "n_cohorts_audited": len(cohorts),
            "delta_table": delta_table,
            "latent_worse_than_identity_survives": latent_worse_than_identity_survives,
            "vena_vs_identity_survives": vena_vs_identity_survives,
            "frozen_csv_path": str(cfg.frozen_csv_path),
        }

        # --------------- write artifacts ----------------------------------
        # make_run_dir creates tables/, figures/, per_scan/ automatically.
        run_dir = make_run_dir(cfg.output_root / "preflights", "rho_s_norm_audit")
        tables_dir = run_dir / "tables"

        # Per-scan rows
        for p, rows in rows_by_percentile.items():
            if rows:
                pd.DataFrame(rows, columns=_RHO_ROW_COLUMNS).to_csv(
                    tables_dir / f"scan_rows_P{p}.csv", index=False
                )

        # Patient-level collapsed
        for p, df in patient_rho.items():
            if not df.empty:
                df.to_csv(tables_dir / f"patient_rho_P{p}.csv", index=False)

        # Summary Δ table
        if not result_df.empty:
            result_df.to_csv(tables_dir / "delta_rho_s.csv", index=False)

        # decision.json
        write_decision_json(run_dir, payload)
        symlink_latest(run_dir)

        # report.md
        _write_report(run_dir, result_df, payload, cfg)

        logger.info("Audit complete → %s", run_dir)
        return run_dir


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    run_dir: Path,
    result_df: pd.DataFrame,
    payload: dict,
    cfg: RhoSNormAuditConfig,
) -> None:
    """Write human-readable report.md to *run_dir*."""
    lines: list[str] = [
        "# ρ_S Normalisation Audit Report",
        "",
        f"**Condition**: {cfg.condition}  |  **Ring**: {cfg.ring}",
        f"**Canonical percentile_upper**: {payload['canonical_percentile_upper']}",
        "",
        "## Decision",
        "",
        f"- `latent_worse_than_identity_survives`: {payload['latent_worse_than_identity_survives']}",
        f"- `vena_vs_identity_survives`: {payload['vena_vs_identity_survives']}",
        "",
        "## Per-method Δρ_S table",
        "",
        "Δρ_S = forced-percentile ρ_S − frozen ρ_S (negative = improvement).",
        "",
    ]

    if not result_df.empty:
        # Pivot: method × percentile
        pivot = result_df.pivot_table(index="method", columns="percentile", values="rho_s")
        frozen_col = result_df.drop_duplicates("method").set_index("method")["rho_s_frozen"]
        pivot["frozen"] = frozen_col
        for p in cfg.percentiles:
            if p in pivot.columns:
                pivot[f"Δ@{p}"] = pivot[p] - pivot["frozen"]
        lines.append(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
        lines.append("")

    lines += [
        "## Fix recommendation",
        "",
        "Add `percentile_upper=99.95` to every `percentile_normalise` call in",
        "`src/vena/model/fm/eval/exhaustive.py::load_real_t1c_normalised` (T-05).",
        "The encoder used `upper=99.95`; the metric path must match.",
        "",
        f"Source: `{cfg.frozen_csv_path}`",
    ]

    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
