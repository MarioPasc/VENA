"""Engine for the vessel-mask preflight.

Consumes the soft maps already on disk under ``<vessel_priors_output>/niigz/
<tag>/<patient>/vessel_soft.nii.gz`` (produced by the ``routines/vessel_priors``
routine) and computes a tentative threshold recommendation per method plus a
cross-method agreement summary. No vesselness operator is re-run.

The output schema is documented in ``.claude/rules/preflight-pattern.md`` —
this engine writes the ``decision.json`` keys the schema mandates and adds
extension keys ``method_agreement`` and ``threshold_sweep`` that downstream
training routines may also consume.

Two outputs survive the run:

* ``artifacts/preflights/vessel_mask/<UTC>/`` — timestamped, immutable.
* ``artifacts/preflights/vessel_mask/LATEST`` — symlink, follows the most
  recent run.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from vena.data.niigz import UCSFPDGMDataset
from vena.data.niigz.shared.io import load_nii

from ._visualize import render_consensus_overlay, render_threshold_curves
from .analysis import (
    PerPatientSweepRecord,
    PerTagSummary,
    ThresholdSweepResult,
    aggregate_per_tag,
    dice,
    jaccard,
    otsu_threshold_brainmasked,
    pick_threshold_by_anatomical_fraction,
    sweep_thresholds,
)

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"


class VesselMaskPreflightError(Exception):
    """Raised when the preflight cannot complete (missing inputs, bad config)."""


@dataclass(frozen=True)
class VesselMaskPreflightConfig:
    """Resolved configuration for one execution of the preflight."""

    vessel_priors_output_root: Path
    dataset_root: Path
    metadata_csv: Path | None
    tags: tuple[str, ...]
    threshold_sweep: tuple[float, ...]
    target_binary_fraction_range: tuple[float, float]
    output_root: Path
    n_patients_max: int | None = None
    log_level: str = "INFO"
    n_slices: int = 5

    @classmethod
    def from_yaml(cls, path: Path | str) -> VesselMaskPreflightConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        for k in (
            "vessel_priors_output_root",
            "dataset_root",
            "output_root",
            "tags",
            "threshold_sweep",
            "target_binary_fraction_range",
        ):
            if k not in raw:
                raise VesselMaskPreflightError(f"missing required key in {path}: {k}")
        tags = tuple(raw["tags"])
        if len(tags) < 2:
            raise VesselMaskPreflightError(
                "tags must list at least two methods for cross-method agreement"
            )
        thresholds = tuple(float(t) for t in raw["threshold_sweep"])
        if not thresholds:
            raise VesselMaskPreflightError("threshold_sweep must be non-empty")
        rng = raw["target_binary_fraction_range"]
        if len(rng) != 2:
            raise VesselMaskPreflightError(
                "target_binary_fraction_range must be a 2-element list"
            )
        metadata_csv = raw.get("metadata_csv")
        return cls(
            vessel_priors_output_root=Path(raw["vessel_priors_output_root"]),
            dataset_root=Path(raw["dataset_root"]),
            metadata_csv=Path(metadata_csv) if metadata_csv else None,
            tags=tags,
            threshold_sweep=thresholds,
            target_binary_fraction_range=(float(rng[0]), float(rng[1])),
            output_root=Path(raw["output_root"]),
            n_patients_max=(
                int(raw["n_patients_max"]) if raw.get("n_patients_max") else None
            ),
            log_level=str(raw.get("log_level", "INFO")).upper(),
            n_slices=int(raw.get("n_slices", 5)),
        )


# ---------------------------------------------------------------- helpers


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _update_latest_symlink(latest: Path, target: Path) -> None:
    """Atomically point ``latest`` at ``target`` (best effort on POSIX)."""
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(target.name, latest)
    except OSError as exc:
        logger.warning("Could not update LATEST symlink %s -> %s: %s", latest, target, exc)


# ---------------------------------------------------------------- engine


class VesselMaskPreflightEngine:
    """Runs the threshold-sweep + cross-method agreement preflight."""

    def __init__(self, cfg: VesselMaskPreflightConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public

    def run(self) -> Path:
        cfg = self.cfg
        timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = cfg.output_root / timestamp
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)

        patients = self._discover_patients()
        if cfg.n_patients_max is not None:
            patients = patients[: cfg.n_patients_max]
        if not patients:
            raise VesselMaskPreflightError(
                f"No patients with soft maps under tags {cfg.tags} in "
                f"{cfg.vessel_priors_output_root}/niigz/"
            )
        logger.info("Discovered %d patients shared across tags %s", len(patients), cfg.tags)

        dataset = UCSFPDGMDataset(cfg.dataset_root, cfg.metadata_csv)
        # UCSFPDGMDataset exposes O(1) string indexing — see `ucsf_pdgm.py`.
        known_ids = set(dataset.ids())

        # ---------------- per-patient sweep + Otsu ----------------
        per_patient_records: list[PerPatientSweepRecord] = []
        otsu_by_tag_patient: dict[str, dict[str, float]] = {tag: {} for tag in cfg.tags}
        per_patient_otsu_binaries: dict[
            tuple[str, str], NDArray[np.uint8]
        ] = {}  # at Otsu — kept for the recommended-threshold consensus later
        brain_cache: dict[str, NDArray[np.bool_]] = {}

        for pid in patients:
            if pid not in known_ids:
                raise VesselMaskPreflightError(
                    f"Patient {pid} present in vessel_priors output but not in "
                    f"UCSF-PDGM dataset at {cfg.dataset_root}"
                )
            brain_vol = dataset.load_brain_mask(dataset[pid])
            brain = (brain_vol.array > 0.5).astype(bool)
            brain_cache[pid] = brain
            for tag in cfg.tags:
                soft_path = self._soft_path(tag, pid)
                soft_vol = load_nii(soft_path)
                soft = soft_vol.array.astype(np.float32, copy=False)
                if soft.shape != brain.shape:
                    raise VesselMaskPreflightError(
                        f"shape mismatch for {pid}/{tag}: soft {soft.shape} vs "
                        f"brain {brain.shape}"
                    )
                recs = sweep_thresholds(
                    tag=tag,
                    patient_id=pid,
                    soft=soft,
                    brain=brain,
                    thresholds=cfg.threshold_sweep,
                )
                per_patient_records.extend(recs)
                otsu_by_tag_patient[tag][pid] = otsu_threshold_brainmasked(soft, brain)
                logger.info(
                    "Sweep tag=%s pid=%s otsu=%.3f bf_at_otsu=%.4f",
                    tag,
                    pid,
                    otsu_by_tag_patient[tag][pid],
                    float(((soft >= otsu_by_tag_patient[tag][pid]) & brain).sum())
                    / max(1, int(brain.sum())),
                )

        # ---------------- aggregate ----------------
        per_tag_summaries = aggregate_per_tag(per_patient_records)
        recommended = pick_threshold_by_anatomical_fraction(
            per_tag_summaries,
            target_fraction_range=cfg.target_binary_fraction_range,
        )

        # ---------------- cross-method agreement at recommended thresholds ----------------
        agreement = self._method_agreement(
            patients=patients,
            recommended=recommended,
            brain_cache=brain_cache,
        )

        # ---------------- figures ----------------
        fig_curves = render_threshold_curves(
            per_tag_summaries,
            target_fraction_range=cfg.target_binary_fraction_range,
            out_path=run_dir / "figures" / "threshold_curves.png",
            recommended_per_tag=recommended,
        )
        consensus_paths = self._render_consensus(
            run_dir=run_dir,
            patients=patients,
            recommended=recommended,
            brain_cache=brain_cache,
            n_slices=cfg.n_slices,
            dataset=dataset,
        )

        # ---------------- tables ----------------
        self._write_tables(
            run_dir=run_dir,
            per_patient=per_patient_records,
            per_tag=per_tag_summaries,
            otsu=otsu_by_tag_patient,
        )

        # ---------------- decision.json + report.md ----------------
        decision = self._build_decision(
            patients=patients,
            recommended=recommended,
            agreement=agreement,
            otsu=otsu_by_tag_patient,
            run_dir=run_dir,
            timestamp=timestamp,
        )
        with (run_dir / "decision.json").open("w") as f:
            json.dump(decision, f, indent=2, sort_keys=True)
        self._write_report(
            run_dir=run_dir,
            decision=decision,
            per_tag=per_tag_summaries,
            agreement=agreement,
            recommended=recommended,
            n_patients=len(patients),
            figures={
                "threshold_curves": fig_curves.relative_to(run_dir),
                "consensus_glob": "figures/consensus_per_patient/*.png",
            },
            consensus_paths=consensus_paths,
        )

        _update_latest_symlink(cfg.output_root / "LATEST", run_dir)
        logger.info("Vessel-mask preflight artifact: %s", run_dir)
        return run_dir

    # ------------------------------------------------------------------ helpers

    def _soft_path(self, tag: str, patient_id: str) -> Path:
        return (
            self.cfg.vessel_priors_output_root
            / "niigz"
            / tag
            / patient_id
            / "vessel_soft.nii.gz"
        )

    def _discover_patients(self) -> list[str]:
        cfg = self.cfg
        per_tag: list[set[str]] = []
        for tag in cfg.tags:
            tag_dir = cfg.vessel_priors_output_root / "niigz" / tag
            if not tag_dir.is_dir():
                raise VesselMaskPreflightError(
                    f"Missing tag directory: {tag_dir}. Run the vessel_priors "
                    f"routine with tag={tag!r} first."
                )
            pids = {
                p.name
                for p in tag_dir.iterdir()
                if p.is_dir() and (p / "vessel_soft.nii.gz").exists()
            }
            if not pids:
                raise VesselMaskPreflightError(
                    f"No vessel_soft.nii.gz files found under {tag_dir}"
                )
            per_tag.append(pids)
        return sorted(set.intersection(*per_tag))

    def _method_agreement(
        self,
        *,
        patients: list[str],
        recommended: dict[str, dict[str, Any]],
        brain_cache: dict[str, NDArray[np.bool_]],
    ) -> dict[str, Any]:
        tags = sorted(recommended)
        if len(tags) < 2:
            return {
                "n_method_pairs": 0,
                "comment": "fewer than two methods — agreement is undefined",
            }
        pairs: list[dict[str, Any]] = []
        for i, ta in enumerate(tags):
            for tb in tags[i + 1 :]:
                jac_list: list[float] = []
                dic_list: list[float] = []
                cons_frac: list[float] = []
                for pid in patients:
                    brain = brain_cache[pid]
                    soft_a = load_nii(self._soft_path(ta, pid)).array
                    soft_b = load_nii(self._soft_path(tb, pid)).array
                    bin_a = soft_a >= float(recommended[ta]["threshold"])
                    bin_b = soft_b >= float(recommended[tb]["threshold"])
                    jac_list.append(jaccard(bin_a, bin_b, brain))
                    dic_list.append(dice(bin_a, bin_b, brain))
                    inter = bin_a & bin_b & brain
                    cons_frac.append(
                        float(inter.sum()) / max(1.0, float(brain.sum()))
                    )
                pairs.append(
                    {
                        "method_a": ta,
                        "method_b": tb,
                        "threshold_a": float(recommended[ta]["threshold"]),
                        "threshold_b": float(recommended[tb]["threshold"]),
                        "jaccard_mean": float(np.nanmean(jac_list)),
                        "jaccard_std": float(np.nanstd(jac_list)),
                        "dice_mean": float(np.nanmean(dic_list)),
                        "dice_std": float(np.nanstd(dic_list)),
                        "consensus_fraction_of_brain_mean": float(np.mean(cons_frac)),
                    }
                )
        return {"n_method_pairs": len(pairs), "pairs": pairs}

    def _render_consensus(
        self,
        *,
        run_dir: Path,
        patients: list[str],
        recommended: dict[str, dict[str, Any]],
        brain_cache: dict[str, NDArray[np.bool_]],
        n_slices: int,
        dataset: UCSFPDGMDataset,
    ) -> list[Path]:
        """Per-patient consensus overlays."""
        out_dir = run_dir / "figures" / "consensus_per_patient"
        out_dir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        tags = sorted(recommended.keys())
        for pid in patients:
            swi = dataset.load_modality(dataset[pid], "SWI_bias").array
            brain = brain_cache[pid]
            binaries: dict[str, NDArray[np.bool_]] = {}
            for tag in tags:
                soft = load_nii(self._soft_path(tag, pid)).array
                binaries[tag] = soft >= float(recommended[tag]["threshold"])
            png = render_consensus_overlay(
                swi=swi,
                brain=brain,
                binaries=binaries,
                out_path=out_dir / f"{pid}.png",
                patient_id=pid,
                n_slices=n_slices,
            )
            out.append(png)
        return out

    def _write_tables(
        self,
        *,
        run_dir: Path,
        per_patient: list[PerPatientSweepRecord],
        per_tag: list[PerTagSummary],
        otsu: dict[str, dict[str, float]],
    ) -> None:
        import csv

        pp = run_dir / "tables" / "per_patient_threshold_sweep.csv"
        with pp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "patient_id",
                    "threshold",
                    "binary_fraction",
                    "n_components",
                    "largest_fraction",
                    "small_cc_count",
                    "skeleton_voxels",
                ]
            )
            for r in per_patient:
                w.writerow(
                    [
                        r.tag,
                        r.patient_id,
                        f"{r.threshold:.6f}",
                        f"{r.binary_fraction:.6f}",
                        r.n_components,
                        f"{r.largest_fraction:.6f}",
                        r.small_cc_count,
                        r.skeleton_voxels,
                    ]
                )
        sm = run_dir / "tables" / "per_tag_summary.csv"
        with sm.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "threshold",
                    "binary_fraction_mean",
                    "binary_fraction_std",
                    "binary_fraction_cv",
                    "n_components_median",
                    "largest_fraction_median",
                    "small_cc_count_median",
                    "skeleton_voxels_median",
                    "n_patients",
                ]
            )
            for s in per_tag:
                w.writerow(
                    [
                        s.tag,
                        f"{s.threshold:.6f}",
                        f"{s.binary_fraction_mean:.6f}",
                        f"{s.binary_fraction_std:.6f}",
                        f"{s.binary_fraction_cv:.6f}",
                        f"{s.n_components_median:.2f}",
                        f"{s.largest_fraction_median:.6f}",
                        f"{s.small_cc_count_median:.2f}",
                        f"{s.skeleton_voxels_median:.2f}",
                        s.n_patients,
                    ]
                )
        ot = run_dir / "tables" / "otsu_thresholds.csv"
        with ot.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "patient_id", "otsu_threshold"])
            for tag, pid_to_val in otsu.items():
                for pid, val in pid_to_val.items():
                    w.writerow([tag, pid, f"{val:.6f}"])

    def _build_decision(
        self,
        *,
        patients: list[str],
        recommended: dict[str, dict[str, Any]],
        agreement: dict[str, Any],
        otsu: dict[str, dict[str, float]],
        run_dir: Path,
        timestamp: str,
    ) -> dict[str, Any]:
        """Assemble the machine-readable preflight contract.

        Keys from ``.claude/rules/preflight-pattern.md §preflights/vessel_mask``
        are populated where derivable from this run; hand-label-dependent keys
        (``dice_vs_hand_label``, ``passes_cmbb_rejection``,
        ``n_hand_label_cases``) remain ``null`` and the ``status`` field flags
        the whole record as tentative until the reader study lands.
        """
        cfg = self.cfg
        otsu_means = {
            tag: float(np.mean(list(vals.values()))) if vals else float("nan")
            for tag, vals in otsu.items()
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "tentative_pending_reader_study",
            "timestamp_utc": timestamp,
            "git_sha": _git_sha(Path(__file__).resolve().parents[4]),
            "n_patients": len(patients),
            "patients": list(patients),
            "vesselness_methods_compared": list(cfg.tags),
            "threshold_sweep": list(cfg.threshold_sweep),
            "target_binary_fraction_range": list(cfg.target_binary_fraction_range),
            "per_tag_recommended": recommended,
            "method_agreement": agreement,
            "otsu_mean_per_tag": otsu_means,
            # Schema-mandated keys that remain unresolved without hand labels.
            "vesselness_method": None,
            "sigma_range_mm": None,
            "soft_mask_threshold": None,
            "dice_vs_hand_label": None,
            "ahd_mm_vs_hand_label": None,
            "n_hand_label_cases": 0,
            "passes_cmbb_rejection": None,
            "vessel_priors_output_root": str(cfg.vessel_priors_output_root),
        }

    def _write_report(
        self,
        *,
        run_dir: Path,
        decision: dict[str, Any],
        per_tag: list[PerTagSummary],
        agreement: dict[str, Any],
        recommended: dict[str, dict[str, Any]],
        n_patients: int,
        figures: dict[str, Any],
        consensus_paths: list[Path],
    ) -> None:
        lines: list[str] = []
        lines.append("# Vessel-mask preflight — tentative\n")
        lines.append(f"**Status**: `{decision['status']}`")
        lines.append(f"**Generated**: {decision['timestamp_utc']}")
        if decision.get("git_sha"):
            lines.append(f"**Git SHA**: `{decision['git_sha']}`")
        lines.append(f"**N patients**: {n_patients}")
        lines.append(f"**Methods compared**: {', '.join(decision['vesselness_methods_compared'])}")
        lines.append("")
        lines.append("## 1. Anatomical prior")
        lo, hi = decision["target_binary_fraction_range"]
        lines.append(
            f"Adult cerebral venous fraction on SWI is ~{lo*100:.1f}-{hi*100:.1f}% by "
            "volume (Bériault 2015, §III.A). Per-method threshold recommendations "
            "are chosen to land the per-patient binary fraction inside this band.\n"
        )

        lines.append("## 2. Per-method recommendation")
        lines.append("")
        lines.append(
            "| Method | Threshold | Mean binary fraction | CV across patients | "
            "Median # CC | Median skeleton voxels | In-band? |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for tag, rec in sorted(recommended.items()):
            lines.append(
                f"| `{tag}` | {rec['threshold']:.3f} | "
                f"{rec['binary_fraction_mean']*100:.2f}% | "
                f"{rec['binary_fraction_cv']*100:.1f}% | "
                f"{rec['n_components_median']:.0f} | "
                f"{rec['skeleton_voxels_median']:.0f} | "
                f"{'yes' if rec['in_band'] else 'NO — extend sweep'} |"
            )
        lines.append("")
        lines.append("Rationale per method:")
        for tag, rec in sorted(recommended.items()):
            lines.append(f"- **{tag}** — {rec['rationale']}")
        lines.append("")

        otsu_means = decision["otsu_mean_per_tag"]
        if otsu_means:
            lines.append("Mean per-patient Otsu cutoff (data-driven baseline):")
            for tag, val in sorted(otsu_means.items()):
                lines.append(f"- `{tag}`: {val:.3f}")
            lines.append("")

        lines.append("## 3. Cross-method agreement")
        if agreement.get("n_method_pairs", 0) == 0:
            lines.append("_No pairwise comparison available._\n")
        else:
            lines.append("")
            lines.append(
                "| Method A | Method B | Threshold A | Threshold B | "
                "Jaccard (mean ± std) | Dice (mean ± std) | Consensus / brain (mean) |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for p in agreement["pairs"]:
                lines.append(
                    f"| `{p['method_a']}` | `{p['method_b']}` | "
                    f"{p['threshold_a']:.3f} | {p['threshold_b']:.3f} | "
                    f"{p['jaccard_mean']:.3f} ± {p['jaccard_std']:.3f} | "
                    f"{p['dice_mean']:.3f} ± {p['dice_std']:.3f} | "
                    f"{p['consensus_fraction_of_brain_mean']*100:.2f}% |"
                )
            lines.append("")

        lines.append("## 4. Figures")
        lines.append(f"![Threshold curves]({figures['threshold_curves']})")
        lines.append("")
        if consensus_paths:
            lines.append("Per-patient consensus overlays at the recommended thresholds:")
            for p in consensus_paths:
                rel = p.relative_to(run_dir)
                lines.append(f"- [{p.stem}]({rel})")
            lines.append("")

        lines.append("## 5. Caveats")
        lines.append(
            "This preflight is **tentative**. The accompanying `decision.json` "
            "cannot be considered closed until:"
        )
        lines.append("")
        lines.append("1. ≥20 patients have hand-labelled vessel masks per `preflight-pattern.md`.")
        lines.append("2. Dice / AHD vs hand-label are computed.")
        lines.append(
            "3. The healthy-control diagnostic (proposal §6.5) confirms the "
            "selected vesselness operator does not learn dark-voxel shortcuts on SWAN."
        )
        lines.append("")
        lines.append(
            "Threshold recommendations rely on a population-level anatomical "
            "prior, not on ground-truth labels; they exclude method bias that is "
            "consistent with the prior (e.g., systematic over-detection of CMBs)."
        )

        (run_dir / "report.md").write_text("\n".join(lines) + "\n")


__all__ = [
    "SCHEMA_VERSION",
    "VesselMaskPreflightConfig",
    "VesselMaskPreflightEngine",
    "VesselMaskPreflightError",
]
