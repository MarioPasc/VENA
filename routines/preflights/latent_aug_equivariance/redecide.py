"""Re-decide latent-augmentation safety from an existing preflight CSV.

The first preflight run produced ``tables/per_patient_metrics.csv`` and a
``decision.json`` evaluated against a single fixed threshold. Reaching a
different decision under a new criterion is then a re-aggregation step: no
decode passes are needed. This CLI takes an existing run directory and an
alternative threshold spec, writes a new timestamped run directory next to
the original, and updates the ``LATEST`` symlink only if explicitly asked.

Two pass criteria are supported:

- ``ssim_only``: ``SSIM >= ssim`` AND ``PSNR >= psnr_db``. Matches the
  baseline preflight; used when only the SSIM threshold needs to change.
- ``ssim_and_recon_floor``: ``SSIM >= ssim`` AND
  ``PSNR >= max(psnr_db_floor, median VAE-recon PSNR across cohorts)``.
  Anchors the PSNR requirement on the VAE's own reconstruction noise so a
  transform whose equivariance gap is BELOW VAE noise is treated as safe.

The new run dir is immutable (preflight-pattern §3); the LATEST symlink is
only retargeted with ``--update-latest`` so the original audit trail stays
intact by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_per_patient(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r") as f:
        return list(csv.DictReader(f))


def _summarise(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_transform: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_transform.setdefault(r["transform"], []).append((float(r["psnr_db"]), float(r["ssim"])))
    out: dict[str, dict[str, float]] = {}
    for label, vals in by_transform.items():
        psnrs = [v[0] for v in vals]
        ssims = [v[1] for v in vals]
        out[label] = {
            "n": len(vals),
            "median_psnr_db": statistics.median(psnrs),
            "median_ssim": statistics.median(ssims),
            "min_psnr_db": min(psnrs),
            "min_ssim": min(ssims),
        }
    return out


def _vae_floor_median_psnr(rows: list[dict[str, str]]) -> float:
    """Median across all rows of the VAE recon floor PSNR.

    The CSV carries ``vae_floor_psnr_db`` per patient (same value repeated
    over all transforms). Aggregating across all rows is fine — we want
    one scalar, and the duplication does not change the median.
    """
    psnrs = [float(r["vae_floor_psnr_db"]) for r in rows if "vae_floor_psnr_db" in r]
    if not psnrs:
        return float("nan")
    return statistics.median(psnrs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Existing preflight run dir.")
    p.add_argument("--criterion", choices=["ssim_only", "ssim_and_recon_floor"], required=True)
    p.add_argument("--ssim", type=float, required=True)
    p.add_argument(
        "--psnr-db",
        type=float,
        required=True,
        help="Strict PSNR threshold (ssim_only) or floor (ssim_and_recon_floor).",
    )
    p.add_argument(
        "--admit-image-domain-only",
        nargs="*",
        default=("gamma",),
        help="Augmentation names that are flagged image-domain-only on failure.",
    )
    p.add_argument(
        "--update-latest",
        action="store_true",
        help="Retarget the cohort's LATEST symlink to the new run dir.",
    )
    p.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-text note recorded inside decision.json for audit.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"source run dir does not exist: {source}")
    csv_path = source / "tables" / "per_patient_metrics.csv"
    src_decision_path = source / "decision.json"
    if not csv_path.is_file() or not src_decision_path.is_file():
        raise SystemExit(
            f"source dir missing tables/per_patient_metrics.csv or decision.json: {source}"
        )

    rows = _load_per_patient(csv_path)
    if not rows:
        raise SystemExit(f"per-patient CSV is empty: {csv_path}")
    summary = _summarise(rows)
    src_decision = json.loads(src_decision_path.read_text())

    if args.criterion == "ssim_only":
        psnr_floor = float(args.psnr_db)
    else:
        vae_floor = _vae_floor_median_psnr(rows)
        psnr_floor = max(float(args.psnr_db), vae_floor)
        logger.info(
            "ssim_and_recon_floor: vae_recon_median_psnr=%.2f dB, requested floor=%.2f dB → effective=%.2f dB",
            vae_floor,
            args.psnr_db,
            psnr_floor,
        )

    latent_safe: set[str] = set()
    rejected_labels: list[str] = []
    per_transform_summary: dict[str, dict[str, float | bool]] = {}
    for label, s in summary.items():
        passes = bool(
            s["n"] > 0
            and s["median_psnr_db"] >= psnr_floor
            and s["median_ssim"] >= float(args.ssim)
        )
        per_transform_summary[label] = {**s, "passes": passes}
        name = label.split("[")[0]
        if passes:
            latent_safe.add(name)
        else:
            rejected_labels.append(label)
    # Names where any param-grid entry fails are excluded.
    all_names_with_failure: set[str] = set()
    for label, ps in per_transform_summary.items():
        if not ps["passes"]:
            all_names_with_failure.add(label.split("[")[0])
    latent_safe -= all_names_with_failure
    image_only = sorted(all_names_with_failure & set(args.admit_image_domain_only))
    truly_rejected = sorted(all_names_with_failure - set(image_only))

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = source.parent / f"{timestamp}_redecided"
    out_dir.mkdir(parents=True, exist_ok=False)
    # Hard-link the source tables so the new dir is self-contained without
    # duplicating bytes.
    (out_dir / "tables").mkdir()
    for src_csv in (source / "tables").glob("*.csv"):
        try:
            (out_dir / "tables" / src_csv.name).hardlink_to(src_csv)
        except OSError:
            (out_dir / "tables" / src_csv.name).write_bytes(src_csv.read_bytes())

    new_decision = {
        "schema_version": src_decision.get("schema_version", "1.0"),
        "produced_at": datetime.now(UTC).isoformat(),
        "producer": "vena.preflight.latent_aug_equivariance.redecide:1.0",
        "source_run_dir": str(source),
        "criterion": args.criterion,
        "pass_threshold": {"psnr_db": psnr_floor, "ssim": float(args.ssim)},
        "note": args.note,
        "n_patients_per_cohort": src_decision.get("n_patients_per_cohort"),
        "cohorts_tested": src_decision.get("cohorts_tested"),
        "modalities_tested": src_decision.get("modalities_tested"),
        "vae_recon_floor": src_decision.get("vae_recon_floor"),
        "vae_checkpoint": src_decision.get("vae_checkpoint"),
        "latent_safe_augmentations": sorted(latent_safe),
        "image_domain_only_augmentations": image_only,
        "rejected_augmentations": truly_rejected,
        "per_transform_summary": per_transform_summary,
    }
    (out_dir / "decision.json").write_text(json.dumps(new_decision, indent=2))
    logger.info(
        "wrote %s (latent_safe=%s, image_only=%s, rejected=%s)",
        out_dir / "decision.json",
        new_decision["latent_safe_augmentations"],
        new_decision["image_domain_only_augmentations"],
        new_decision["rejected_augmentations"],
    )

    if args.update_latest:
        latest = source.parent / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(out_dir.name)
            logger.info("LATEST -> %s", out_dir.name)
        except OSError as exc:
            logger.warning("could not update LATEST symlink: %s", exc)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
