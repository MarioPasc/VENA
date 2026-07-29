#!/usr/bin/env python3
"""Measure enhancement-contrast statistics across tc-weight ablation arms.

Reads pre-computed latent predictions from
  experiments/<run_dir>/exhaustive_val/epoch_NNN/latent_preds.h5,
decodes each prediction with the frozen MAISI VAE, loads the real T1c and
region masks from the cohort latent + image H5 files, then computes per-patient
intensity statistics on the normalised [0, 1] scale.

The ET definition is byte-identical to the one behind psnr_db_et in
exhaustive_val/epoch_*/metrics.csv:
  masks/tumor_latent channel 2 (soft), NN-upsampled to image space, binarised
  at 0.5  (matches ExhaustiveValEngine._per_class_tumor_masks_in_image_space).

Usage (on Picasso, inside VENA-validation with PYTHONPATH set):
    python scripts/measure_intensity_contrast.py

Output:
    ~/execs/vena/analyses/s3_intensity/<UTC>/intensity_stats.csv
    ~/execs/vena/analyses/s3_intensity/LATEST  (symlink)
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PYTHONPATH isolation proof — both modules must resolve inside VENA-validation
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA-validation")

import routines  # noqa: E402
import vena  # noqa: E402

_vena_file = Path(vena.__file__).resolve()
_routines_file = Path(routines.__file__).resolve()
print(f"vena.__file__     = {_vena_file}", flush=True)
print(f"routines.__file__ = {_routines_file}", flush=True)

_repo_resolved = REPO_ROOT.resolve()
if _repo_resolved not in _vena_file.parents:
    raise RuntimeError(
        f"ISOLATION FAIL: vena imports from wrong repo: {_vena_file}\n"
        f"Expected parent: {_repo_resolved}\n"
        f"Set PYTHONPATH={_repo_resolved}/src"
    )
if _repo_resolved not in _routines_file.parents:
    raise RuntimeError(
        f"ISOLATION FAIL: routines imports from wrong repo: {_routines_file}\n"
        f"Expected parent: {_repo_resolved}\n"
        f"Set PYTHONPATH={_repo_resolved}/src"
    )
print("ISOLATION PROOF: OK — both modules resolve inside VENA-validation", flush=True)

from vena.common import MaisiDecoder, load_autoencoder  # noqa: E402
from vena.common.decode import decode_box  # noqa: E402
from vena.model.fm.eval.exhaustive import (  # noqa: E402
    build_crop_spec_from_h5,
    load_real_t1c_box,
)

# ---------------------------------------------------------------------------
# Picasso paths (all verified in brief)
# ---------------------------------------------------------------------------
EXPERIMENTS_BASE = Path("/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments")
VAE_CHECKPOINT = Path(
    "/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints"
    "/NV-Generate-MR/models/autoencoder_v2.pt"
)
ANALYSIS_BASE = Path("/mnt/home/users/tic_163_uma/mpascual/execs/vena/analyses/s3_intensity")
DEVICE = "cuda:0"
NFE_LEVELS = [5, 20]

# ---------------------------------------------------------------------------
# Arm table  (label, run_dir, target_epoch)
# ---------------------------------------------------------------------------
ARMS: list[dict[str, Any]] = [
    {
        "label": "baseline_v3a",
        "run_dir": "2026-07-24_21-04-53_s1_s2_t13_j2_cn2ch_joint_tc5_a742b31b",
        "epoch": 0,
    },
    {
        "label": "J1_tc1",
        "run_dir": "2026-07-24_21-01-08_s1_s2_t13_j1_cn2ch_joint_tc1_003f3366",
        "epoch": 1450,
    },
    {
        "label": "J2_tc5",
        "run_dir": "2026-07-24_21-04-53_s1_s2_t13_j2_cn2ch_joint_tc5_a742b31b",
        "epoch": 1050,
    },
    {
        "label": "J3_tc10",
        "run_dir": "2026-07-24_21-11-14_s1_s2_t13_j3_cn2ch_joint_tc10_bc9314c5",
        "epoch": 1025,
    },
    {
        "label": "J4_tc20",
        "run_dir": "2026-07-25_13-57-32_s1_s2_t13_j4_cn2ch_joint_tc20_0cb4ccd4",
        "epoch": 750,
    },
]

# Output CSV columns (order is the wire format — do not reorder)
CSV_COLS = [
    "label",
    "run_id",
    "epoch",
    "nfe",
    "patient_id",
    "cohort",
    "n_vox_brain",
    "n_vox_et",
    "n_vox_bnwt",
    "p995_pred_brain",
    "p995_real_brain",
    "mean_et_pred",
    "mean_et_real",
    "mean_bnwt_pred",
    "mean_bnwt_real",
    "p99_et_pred",
    "p99_et_real",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_epoch_dir(run_dir: str, target_epoch: int) -> tuple[Path, int]:
    """Return the exhaustive_val epoch dir containing latent_preds.h5.

    Looks for an exact epoch match; falls back to the nearest epoch with a
    latent_preds.h5.  Logs a WARNING when falling back.
    """
    base = EXPERIMENTS_BASE / run_dir / "exhaustive_val"
    if not base.is_dir():
        raise FileNotFoundError(f"No exhaustive_val dir: {base}")

    candidates: list[tuple[int, Path]] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("epoch_"):
            continue
        try:
            ep = int(d.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if (d / "latent_preds.h5").is_file():
            candidates.append((ep, d))

    if not candidates:
        raise FileNotFoundError(f"No epoch dirs with latent_preds.h5 under {base}")

    # Exact match
    for ep, d in candidates:
        if ep == target_epoch:
            return d, ep

    # Nearest
    nearest_ep, nearest_d = min(candidates, key=lambda x: abs(x[0] - target_epoch))
    logger.warning(
        "epoch %d absent in %s — using nearest epoch %d at %s",
        target_epoch,
        run_dir,
        nearest_ep,
        nearest_d,
    )
    return nearest_d, nearest_ep


def _load_job_yaml(epoch_dir: Path) -> dict[str, Any]:
    p = epoch_dir / "job.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"job.yaml missing: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def _build_patient_lookup(
    job: dict[str, Any],
    epoch_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Build patient_id → {cohort, latent_h5, image_h5, row} mapping.

    1. Read metrics.csv (if present) for the cohort column — O(n_rows).
    2. Parse corpus_registry (or single latents_h5/image_h5) to resolve
       per-cohort H5 paths.
    3. Open each latent H5 to build the patient→row index.
    """
    # Step 1: patient → cohort from metrics.csv (optional fast path)
    pid_to_cohort: dict[str, str] = {}
    metrics_csv = epoch_dir / "metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open() as f:
            for row in csv.DictReader(f):
                pid = row.get("patient_id", "")
                coh = row.get("cohort", "")
                if pid and coh:
                    pid_to_cohort[pid] = coh

    # Step 2: cohort_name → {latent_h5, image_h5}
    cohort_paths: dict[str, dict[str, Path]] = {}

    corpus_registry = job.get("corpus_registry")
    if corpus_registry:
        with Path(corpus_registry).open() as f:
            registry = json.load(f)
        for c in registry.get("cohorts", []):
            cohort_paths[c["name"]] = {
                "latent_h5": Path(c["latent_h5"]),
                "image_h5": Path(c["image_h5"]),
            }
    else:
        lh5 = job.get("latents_h5")
        ih5 = job.get("image_h5")
        if lh5 and ih5:
            cohort_name = next(iter(set(pid_to_cohort.values())), "UCSF-PDGM")
            cohort_paths[cohort_name] = {
                "latent_h5": Path(lh5),
                "image_h5": Path(ih5),
            }

    # Step 3: build patient→row from each cohort's latent H5
    result: dict[str, dict[str, Any]] = {}
    for cohort_name, paths in cohort_paths.items():
        lat_h5 = paths["latent_h5"]
        img_h5 = paths["image_h5"]
        if not lat_h5.is_file():
            logger.warning("latent H5 missing — skipping cohort %s: %s", cohort_name, lat_h5)
            continue
        with h5py.File(lat_h5, "r", swmr=True) as lf:
            ids = [b.decode() if isinstance(b, bytes) else str(b) for b in lf["ids"][:]]
        for row_idx, pid in enumerate(ids):
            result[pid] = {
                "cohort": cohort_name,
                "latent_h5": lat_h5,
                "image_h5": img_h5,
                "row": row_idx,
            }

    return result


def _upsample_to_image(
    mask_np: np.ndarray,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    """NN-upsample a latent-space mask to image space.

    Accepts (C, h, w, d) or (1, C, h, w, d); returns (1, C, H, W, D) float.
    """
    t = torch.from_numpy(np.ascontiguousarray(mask_np)).float()
    if t.ndim == 3:
        t = t[None, None]  # (1,1,h,w,d)
    elif t.ndim == 4:
        t = t[None]  # (1,C,h,w,d)
    return F.interpolate(t, size=image_shape, mode="nearest")  # (1,C,H,W,D)


def _compute_stats(
    img_pred: torch.Tensor,  # (H,W,D) float32 [0,1]
    real_box: torch.Tensor,  # (H,W,D) float32 [0,1]
    tumor_lat: np.ndarray,  # (3,h,w,d) soft NETC/ED/ET
    brain_lat: np.ndarray | None,  # (1,h,w,d) or None
) -> dict[str, Any]:
    """Compute per-patient intensity stats on the normalised [0, 1] scale.

    Mask conventions mirror the exhaustive-val engine exactly:

    * WT:  (NETC+ED+ET soft union >= 0.5) at latent level, then NN-upsample
           (matches LatentH5Dataset._read_one → _upsample_latent_mask).
    * ET:  tumor_lat[2] (soft) NN-upsampled to image space, binarised at 0.5
           (matches _per_class_tumor_masks_in_image_space).
    * Brain: masks/brain_latent NN-upsampled, or real_box > 0 fallback.
    * BN-WT: brain & ~WT.

    ET rows are written with NaN in ET columns when n_vox_et == 0.
    """
    image_shape = (img_pred.shape[0], img_pred.shape[1], img_pred.shape[2])

    # --- WT: threshold at latent scale, then upsample (matches LatentH5Dataset) ---
    soft_union = np.clip(tumor_lat.sum(axis=0, keepdims=True), 0.0, 1.0)  # (1,h,w,d)
    m_wt_lat = (soft_union >= 0.5).astype(np.float32)
    m_wt = _upsample_to_image(m_wt_lat, image_shape).squeeze(0).squeeze(0).bool()  # (H,W,D)

    # --- ET: upsample soft first, THEN binarise (matches _per_class_tumor_masks_in_image_space) ---
    m_et_soft = tumor_lat[2:3]  # (1,h,w,d) channel 2 = ET
    m_et_up = _upsample_to_image(m_et_soft, image_shape).squeeze(0).squeeze(0)  # (H,W,D)
    m_et = m_et_up >= 0.5

    # --- Brain ---
    if brain_lat is not None:
        m_brain = _upsample_to_image(brain_lat, image_shape).squeeze(0).squeeze(0).bool()
    else:
        m_brain = real_box > 0  # skull-strip-foreground fallback

    # --- BN-WT ---
    m_bnwt = m_brain & ~m_wt

    # --- Statistics ---
    def _percentile(vol: torch.Tensor, mask: torch.Tensor, pct: float) -> float:
        vox = vol[mask].float()
        if vox.numel() == 0:
            return float("nan")
        return float(torch.quantile(vox, pct / 100.0).item())

    def _mean(vol: torch.Tensor, mask: torch.Tensor) -> float:
        vox = vol[mask].float()
        return float("nan") if vox.numel() == 0 else float(vox.mean().item())

    n_brain = int(m_brain.sum().item())
    n_et = int(m_et.sum().item())
    n_bnwt = int(m_bnwt.sum().item())

    p995_pred_brain = _percentile(img_pred, m_brain, 99.5)
    p995_real_brain = _percentile(real_box, m_brain, 99.5)

    if n_et > 0:
        mean_et_pred = _mean(img_pred, m_et)
        mean_et_real = _mean(real_box, m_et)
        p99_et_pred = _percentile(img_pred, m_et, 99.0)
        p99_et_real = _percentile(real_box, m_et, 99.0)
    else:
        mean_et_pred = mean_et_real = p99_et_pred = p99_et_real = float("nan")

    mean_bnwt_pred = _mean(img_pred, m_bnwt)
    mean_bnwt_real = _mean(real_box, m_bnwt)

    return {
        "n_vox_brain": n_brain,
        "n_vox_et": n_et,
        "n_vox_bnwt": n_bnwt,
        "p995_pred_brain": p995_pred_brain,
        "p995_real_brain": p995_real_brain,
        "mean_et_pred": mean_et_pred,
        "mean_et_real": mean_et_real,
        "mean_bnwt_pred": mean_bnwt_pred,
        "mean_bnwt_real": mean_bnwt_real,
        "p99_et_pred": p99_et_pred,
        "p99_et_real": p99_et_real,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(DEVICE)

    utc_now = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = ANALYSIS_BASE / utc_now
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    # Reproducibility artefacts
    shutil.copy2(__file__, out_dir / Path(__file__).name)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        (out_dir / "git_sha").write_text(git_sha + "\n")
        logger.info("git SHA: %s", git_sha)
    except Exception as exc:
        logger.warning("git SHA unavailable: %s", exc)

    # Load VAE once (shared across all arms)
    logger.info("Loading VAE checkpoint: %s", VAE_CHECKPOINT)
    handle = load_autoencoder(VAE_CHECKPOINT, device=str(device))
    decoder = MaisiDecoder(handle=handle)
    logger.info("VAE loaded.")

    all_rows: list[dict[str, Any]] = []

    for arm in ARMS:
        label = arm["label"]
        run_dir = arm["run_dir"]
        target_epoch = arm["epoch"]
        logger.info("=== ARM %s  run=%s  target_epoch=%d ===", label, run_dir, target_epoch)

        # 1. Locate epoch dir
        try:
            epoch_dir, actual_epoch = _find_epoch_dir(run_dir, target_epoch)
        except FileNotFoundError as exc:
            logger.error("ARM %s: %s — SKIPPING", label, exc)
            continue

        if actual_epoch != target_epoch:
            logger.warning(
                "ARM %s: requested epoch %d; using epoch %d (nearest with preds H5) at %s",
                label,
                target_epoch,
                actual_epoch,
                epoch_dir,
            )
        else:
            logger.info("ARM %s: epoch %d at %s", label, actual_epoch, epoch_dir)

        preds_h5_path = epoch_dir / "latent_preds.h5"
        # (already verified in _find_epoch_dir, but guard again)
        if not preds_h5_path.is_file():
            logger.error("ARM %s: latent_preds.h5 missing — SKIPPING", label)
            continue

        # 2. Parse job.yaml
        try:
            job = _load_job_yaml(epoch_dir)
        except FileNotFoundError as exc:
            logger.error("ARM %s: %s — SKIPPING", label, exc)
            continue
        run_id: str = str(job.get("run_id", run_dir))

        # 3. Build patient lookup
        try:
            patient_lookup = _build_patient_lookup(job, epoch_dir)
        except Exception as exc:
            logger.error("ARM %s: cohort lookup failed: %s — SKIPPING", label, exc)
            continue
        logger.info("ARM %s: %d patients in lookup", label, len(patient_lookup))

        # 4. List patients in preds H5
        with h5py.File(preds_h5_path, "r") as pf:
            patient_ids: list[str] = list(pf["predictions"].keys())
        logger.info("ARM %s: %d patients in latent_preds.h5", label, len(patient_ids))

        n_ok = n_skip = 0

        for pid in patient_ids:
            info = patient_lookup.get(pid)
            if info is None:
                logger.warning("ARM %s: patient %s not in lookup — skipping", label, pid)
                n_skip += 1
                continue

            cohort = info["cohort"]
            lat_h5_path: Path = info["latent_h5"]
            img_h5_path: Path = info["image_h5"]
            lat_row: int = info["row"]

            # Build crop spec + load real T1c (once per patient, reused across NFE)
            try:
                crop_spec = build_crop_spec_from_h5(img_h5_path, pid)
                real_box = load_real_t1c_box(img_h5_path, pid, crop_spec).to(device)
            except Exception as exc:
                logger.warning(
                    "ARM %s: patient %s — crop/real failed (%s) — skipping", label, pid, exc
                )
                n_skip += 1
                continue

            # Load masks from latent H5 (once per patient)
            try:
                with h5py.File(lat_h5_path, "r", swmr=True) as lf:
                    tumor_lat = lf["masks/tumor_latent"][lat_row]  # (3, h, w, d)
                    brain_lat = (
                        lf["masks/brain_latent"][lat_row] if "masks/brain_latent" in lf else None
                    )
            except Exception as exc:
                logger.warning(
                    "ARM %s: patient %s — mask read failed (%s) — skipping", label, pid, exc
                )
                n_skip += 1
                continue

            # Read all needed NFE latents in a single H5 open
            latent_by_nfe: dict[int, np.ndarray | None] = {}
            with h5py.File(preds_h5_path, "r") as pf:
                pid_grp = pf["predictions"][pid]
                for nfe in NFE_LEVELS:
                    key = f"nfe_{nfe}"
                    latent_by_nfe[nfe] = pid_grp[key][()] if key in pid_grp else None

            # Decode each NFE and compute statistics
            for nfe, latent_np in latent_by_nfe.items():
                if latent_np is None:
                    logger.warning(
                        "ARM %s: patient %s — nfe_%d absent — skipping this NFE",
                        label,
                        pid,
                        nfe,
                    )
                    continue

                # latent_np: (4, 48, 56, 48) float16 → decode_box needs (1,C,h,w,d) float32
                latent_t = (
                    torch.from_numpy(latent_np.astype(np.float32)).unsqueeze(0).to(device)
                )  # (1, 4, 48, 56, 48)

                try:
                    img_pred = decode_box(decoder, latent_t, crop_spec)  # (H,W,D) float32
                except Exception as exc:
                    logger.warning(
                        "ARM %s: patient %s nfe=%d — decode failed (%s) — skipping",
                        label,
                        pid,
                        nfe,
                        exc,
                    )
                    continue

                # Move to CPU for statistics (masks are already CPU numpy)
                img_pred_cpu = img_pred.cpu()
                real_cpu = real_box.cpu()

                with torch.no_grad():
                    stats = _compute_stats(img_pred_cpu, real_cpu, tumor_lat, brain_lat)

                all_rows.append(
                    {
                        "label": label,
                        "run_id": run_id,
                        "epoch": actual_epoch,
                        "nfe": nfe,
                        "patient_id": pid,
                        "cohort": cohort,
                        **stats,
                    }
                )
                n_ok += 1

        logger.info("ARM %s: %d patient×NFE rows written, %d patients skipped", label, n_ok, n_skip)

    # Write CSV
    csv_path = out_dir / "intensity_stats.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(all_rows)

    n_data_rows = len(all_rows)
    logger.info(
        "Written: %s  (%d data rows + 1 header = %d lines)", csv_path, n_data_rows, n_data_rows + 1
    )

    # LATEST symlink
    latest_link = ANALYSIS_BASE / "LATEST"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(out_dir)
    logger.info("LATEST -> %s", out_dir)

    # --- Deliverables print block (for the job log) ---
    print("\n" + "=" * 70, flush=True)
    print("DELIVERABLES", flush=True)
    print(f"  Artifact path (readlink): {out_dir}", flush=True)
    print(f"  wc -l: {n_data_rows + 1}  (header + {n_data_rows} data rows)", flush=True)
    print(f"  Header: {','.join(CSV_COLS)}", flush=True)

    # Per-label row counts
    label_counts: dict[str, int] = {}
    for r in all_rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print("  Row counts per label:", flush=True)
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl}: {cnt}", flush=True)

    # Sanity check: real stats must be label-independent for same patient
    # Pick first patient that appears in at least 2 arms
    pid_to_rows: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        pid_to_rows.setdefault(r["patient_id"], []).append(r)

    sanity_pid: str | None = None
    for pid, rows in pid_to_rows.items():
        labels_present = {r["label"] for r in rows}
        if len(labels_present) >= 2:
            sanity_pid = pid
            break

    print(
        "\n  SANITY CHECK (real stats must be identical across arms for same patient):", flush=True
    )
    if sanity_pid is not None:
        sanity_rows = [r for r in pid_to_rows[sanity_pid] if r["nfe"] == NFE_LEVELS[0]]
        print(f"  Patient: {sanity_pid}  (NFE={NFE_LEVELS[0]})", flush=True)
        seen_reals: dict[str, tuple[float, float]] = {}
        mismatch = False
        for sr in sanity_rows:
            lbl = sr["label"]
            p995r = sr["p995_real_brain"]
            met_r = sr["mean_et_real"]
            print(f"    {lbl}: p995_real_brain={p995r:.6f}  mean_et_real={met_r}", flush=True)
            key = f"{p995r:.8f}|{met_r}"
            if seen_reals and key not in seen_reals.values():
                mismatch = True
            seen_reals[lbl] = (p995r, float(met_r) if met_r == met_r else float("nan"))
        if mismatch:
            print("  *** MISMATCH DETECTED — real loading path is wrong ***", flush=True)
        else:
            print("  OK — real stats agree across arms.", flush=True)
    else:
        print("  No patient found in >= 2 arms — cannot verify.", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
