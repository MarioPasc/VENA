"""Per-cohort audit of the cached oracle soft mask ``masks/tumor_latent_soft``.

Runs one cohort per invocation (SLURM array task) and writes a per-scan metric
CSV plus a cohort summary JSON.  Every metric is recomputed from the **actual GT
labels** in the image-domain H5; nothing is taken on trust from the cache.

Invariants covered (see ``README.md`` in this directory for the rationale):

Structural
    shape/dtype, NaN/Inf, value range ``[0, 1]``, ``tumor_region`` attr,
    far-field floor ``sigmoid(-clip_vox / sdt_sigma_vox)``.
Semantic
    nesting ``NETC <= TC``; tumour-present ⇒ mask has structure;
    tumour-absent ⇒ mask is flat at the floor (no phantom tumour).
GT ↔ soft overlap
    hard GT ⊆ ``{soft > 0.5}``; ``Dice(hard, soft > 0.5)``; volume calibration.
Continuity
    fraction of voxels strictly between the floor and 1 (a binary mask scores
    ~0 here — precisely the defect an earlier QC round missed).
Registration / exactness
    ``recompute_mae``: re-deriving from GT and re-pooling with the canonical
    ``apply_crop_pad`` → ``avg_pool3d(4)`` path must reproduce the cached
    latent **bit-exactly**.  Plus latent↔image IoU/centroid, and an independent
    cross-check against the pre-existing oracle ``masks/tumor_latent``
    (``NETC + ET``), produced by a different pipeline.
Geometry / containment
    crop ``(192, 224, 192)`` must not clip the tumour; TC mass must lie inside
    ``masks/brain_latent``; mass conservation under 4x avg-pool (64 image
    voxels per latent voxel).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from vena.common import CropPadSpec, apply_crop_pad
from vena.segmentation.config import DerivationConfig, TargetConfig
from vena.segmentation.targets.soft_targets import make_soft_targets

logger = logging.getLogger("mask_audit")

LATENT_CROP_BOX: tuple[int, int, int] = (192, 224, 192)
POOL_STRIDE: int = 4
VOX_PER_LATENT: int = POOL_STRIDE**3  # 64
SOFT_GROUP = "masks/tumor_latent_soft"
ORACLE_GROUP = "masks/tumor_latent"
BRAIN_GROUP = "masks/brain_latent"

# Cohort table: (name, image_h5_rel, latent_h5_rel) — index == SLURM_ARRAY_TASK_ID.
BASE = "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena"
COHORTS: list[tuple[str, str, str]] = [
    ("UCSF-PDGM", "UCSF_PDGM/h5/UCSFPDGM_image.h5", "UCSF_PDGM/h5/UCSFPDGM_latents.h5"),
    (
        "BraTS-GLI",
        "BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5",
        "BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_latents.h5",
    ),
    ("UPENN-GBM", "upenn_gbm/h5/UPENN-GBM_image.h5", "upenn_gbm/h5/UPENN-GBM_latents.h5"),
    ("IvyGAP", "ivy_gap/h5/IvyGAP_image.h5", "ivy_gap/h5/IvyGAP_latents.h5"),
    (
        "BraTS-Africa-Glioma",
        "brats_africa_glioma/h5/BraTS_Africa_glioma_image.h5",
        "brats_africa_glioma/h5/BraTS_Africa_glioma_latents.h5",
    ),
    (
        "BraTS-Africa-Other",
        "brats_africa_other/h5/BraTS_Africa_other_image.h5",
        "brats_africa_other/h5/BraTS_Africa_other_latents.h5",
    ),
    ("LUMIERE", "lumiere/h5/LUMIERE_image.h5", "lumiere/h5/LUMIERE_latents.h5"),
    ("REMBRANDT", "rembrandt/h5/REMBRANDT_image.h5", "rembrandt/h5/REMBRANDT_latents.h5"),
    ("BraTS-PED", "brats_ped/h5/BraTS_PED_image.h5", "brats_ped/h5/BraTS_PED_latents.h5"),
]

CSV_FIELDS: list[str] = [
    "cohort",
    "scan_id",
    "lat_row",
    "img_row",
    # --- GT volumes (native + crop frame) ---
    "gt_tc_vox",
    "gt_netc_vox",
    "gt_et_vox",
    "gt_ed_vox",
    "gt_wt_vox",
    "gt_tc_vox_crop",
    "gt_netc_vox_crop",
    "crop_clip_frac_tc",
    # --- image-domain soft (crop frame) ---
    "soft_tc_min",
    "soft_tc_max",
    "soft_tc_mean",
    "soft_netc_max",
    "hard_subset_soft_viol_frac_tc",
    "hard_subset_soft_viol_frac_netc",
    "dice_tc_img",
    "dice_netc_img",
    "volratio_tc_img",
    "soft_intermediate_frac_tc",
    "nesting_viol_frac_img",
    # --- cached latent (the artifact under audit) ---
    "lat_tc_min",
    "lat_tc_max",
    "lat_tc_mean",
    "lat_netc_max",
    "lat_nan_count",
    "lat_range_ok",
    "lat_nesting_viol_frac",
    "recompute_mae",
    "recompute_max_abs",
    "lat_iou_tc",
    "lat_centroid_dist_tc",
    "lat_iou_netc",
    "lat_tc_mass_ratio",
    # --- independent oracle cross-check ---
    "oracle_tc_iou",
    "oracle_tc_centroid_dist",
    # --- brain containment ---
    "tc_outside_brain_frac",
    "has_brain_mask",
    # --- errors ---
    "error",
]


def _decode(raw: Any) -> str:
    """Decode an H5 vlen-str id to ``str``."""
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient of two boolean arrays; 1.0 when both are empty."""
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / (sa + sb + 1e-12)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean arrays; 1.0 when both are empty."""
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(a, b).sum()) / union


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    """Centre of mass of a boolean mask, or ``None`` when empty."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return idx.mean(axis=0)


def _upscale(latent: np.ndarray) -> np.ndarray:
    """Nearest-neighbour upscale ``(2,48,56,48)`` -> ``(2,192,224,192)``."""
    t = torch.from_numpy(np.ascontiguousarray(latent)).unsqueeze(0)
    up = F.interpolate(t, scale_factor=float(POOL_STRIDE), mode="nearest")
    return up.squeeze(0).numpy()


def _crop_to_box(vol: np.ndarray, spec: CropPadSpec) -> np.ndarray:
    """Apply the canonical crop/pad to a ``(C,H,W,D)`` float volume."""
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).unsqueeze(0)
    return apply_crop_pad(t, spec).squeeze(0).numpy()


def audit_scan(
    *,
    label: np.ndarray,
    crop_origin: np.ndarray,
    cached: np.ndarray,
    oracle: np.ndarray | None,
    brain: np.ndarray | None,
    target_cfg: TargetConfig,
    floor: float,
) -> dict[str, Any]:
    """Compute every audited metric for one scan.

    Parameters
    ----------
    label
        Native integer BraTS label volume ``(H, W, D)``.
    crop_origin
        Length-3 crop origin from the image H5 ``crop/origin``.
    cached
        Cached latent soft mask ``(2, 48, 56, 48)``.
    oracle
        Pre-existing latent oracle ``(3, 48, 56, 48)`` = soft ``[NETC, ED, ET]``,
        or ``None`` when absent.
    brain
        Latent brain mask ``(1, 48, 56, 48)`` or ``None``.
    target_cfg
        Soft-target settings (must match the ones used to build the cache).
    floor
        Expected far-field value ``sigmoid(-clip_vox / sdt_sigma_vox)``.

    Returns
    -------
    dict
        One row of :data:`CSV_FIELDS` (without ``cohort``/``scan_id``/rows).
    """
    row: dict[str, Any] = {}

    # ---- hard GT (both BraTS conventions: ED == 2, NETC == 1) ----
    tc_hard = (label > 0) & (label != 2)
    netc_hard = label == 1
    wt_hard = label > 0
    row["gt_tc_vox"] = int(tc_hard.sum())
    row["gt_netc_vox"] = int(netc_hard.sum())
    row["gt_et_vox"] = int(tc_hard.sum() - netc_hard.sum())
    row["gt_ed_vox"] = int((label == 2).sum())
    row["gt_wt_vox"] = int(wt_hard.sum())

    spec = CropPadSpec(
        crop_origin=(int(crop_origin[0]), int(crop_origin[1]), int(crop_origin[2])),
        native_shape=(label.shape[0], label.shape[1], label.shape[2]),
        target_shape=LATENT_CROP_BOX,
    )

    # ---- soft targets at image res, then canonical crop ----
    soft_native = make_soft_targets(label, target_cfg)  # (2, H, W, D) float32
    soft_crop = _crop_to_box(soft_native, spec)  # (2, 192, 224, 192)
    hard_crop = _crop_to_box(
        np.stack([tc_hard, netc_hard]).astype(np.float32), spec
    )  # (2, 192, 224, 192)
    tc_hard_c = hard_crop[0] > 0.5
    netc_hard_c = hard_crop[1] > 0.5

    row["gt_tc_vox_crop"] = int(tc_hard_c.sum())
    row["gt_netc_vox_crop"] = int(netc_hard_c.sum())
    row["crop_clip_frac_tc"] = (
        float(1.0 - row["gt_tc_vox_crop"] / row["gt_tc_vox"]) if row["gt_tc_vox"] > 0 else 0.0
    )

    stc, snetc = soft_crop[0], soft_crop[1]
    row["soft_tc_min"] = float(stc.min())
    row["soft_tc_max"] = float(stc.max())
    row["soft_tc_mean"] = float(stc.mean())
    row["soft_netc_max"] = float(snetc.max())

    # hard ⊆ soft, Dice, volume calibration
    row["hard_subset_soft_viol_frac_tc"] = (
        float((stc[tc_hard_c] <= 0.5).mean()) if tc_hard_c.any() else 0.0
    )
    row["hard_subset_soft_viol_frac_netc"] = (
        float((snetc[netc_hard_c] <= 0.5).mean()) if netc_hard_c.any() else 0.0
    )
    row["dice_tc_img"] = _dice(tc_hard_c, stc > 0.5)
    row["dice_netc_img"] = _dice(netc_hard_c, snetc > 0.5)
    row["volratio_tc_img"] = (
        float(int((stc > 0.5).sum()) / row["gt_tc_vox_crop"]) if row["gt_tc_vox_crop"] > 0 else 1.0
    )
    # continuity: strictly between the floor and saturation
    row["soft_intermediate_frac_tc"] = float(((stc > floor + 0.02) & (stc < 0.98)).mean())
    row["nesting_viol_frac_img"] = float((snetc > stc + 1e-6).mean())

    # ---- cached latent ----
    ltc, lnetc = cached[0], cached[1]
    row["lat_tc_min"] = float(ltc.min())
    row["lat_tc_max"] = float(ltc.max())
    row["lat_tc_mean"] = float(ltc.mean())
    row["lat_netc_max"] = float(lnetc.max())
    row["lat_nan_count"] = int((~np.isfinite(cached)).sum())
    row["lat_range_ok"] = bool(cached.min() >= -1e-6 and cached.max() <= 1.0 + 1e-6)
    row["lat_nesting_viol_frac"] = float((lnetc > ltc + 1e-6).mean())

    # ---- exactness: re-pool the cropped soft and compare to the cache ----
    repooled = (
        F.avg_pool3d(
            torch.from_numpy(np.ascontiguousarray(soft_crop)),
            kernel_size=POOL_STRIDE,
            stride=POOL_STRIDE,
        )
        .numpy()
        .astype(np.float32)
    )
    diff = np.abs(repooled - cached)
    row["recompute_mae"] = float(diff.mean())
    row["recompute_max_abs"] = float(diff.max())

    # ---- latent <-> image agreement (crop frame) ----
    up = _upscale(cached)
    up_tc_b, up_netc_b = up[0] > 0.5, up[1] > 0.5
    img_tc_b, img_netc_b = stc > 0.5, snetc > 0.5
    row["lat_iou_tc"] = _iou(up_tc_b, img_tc_b)
    row["lat_iou_netc"] = _iou(up_netc_b, img_netc_b)
    c_up, c_img = _centroid(up_tc_b), _centroid(img_tc_b)
    row["lat_centroid_dist_tc"] = (
        float(np.linalg.norm(c_up - c_img)) if c_up is not None and c_img is not None else -1.0
    )
    # mass conservation: sum over the latent grid x 64 should equal the crop-frame sum
    mass_img = float((stc - floor).clip(min=0).sum())
    mass_lat = float((ltc - floor).clip(min=0).sum()) * VOX_PER_LATENT
    row["lat_tc_mass_ratio"] = float(mass_lat / mass_img) if mass_img > 1e-6 else 1.0

    # ---- independent cross-check vs the pre-existing oracle group ----
    if oracle is not None:
        oracle_tc = np.clip(oracle[0] + oracle[2], 0.0, 1.0)  # NETC + ET
        o_b, l_b = oracle_tc > 0.5, ltc > 0.5
        row["oracle_tc_iou"] = _iou(o_b, l_b)
        c_o, c_l = _centroid(o_b), _centroid(l_b)
        row["oracle_tc_centroid_dist"] = (
            float(np.linalg.norm(c_o - c_l)) if c_o is not None and c_l is not None else -1.0
        )
    else:
        row["oracle_tc_iou"] = -1.0
        row["oracle_tc_centroid_dist"] = -1.0

    # ---- brain containment ----
    if brain is not None:
        b = brain[0] > 0.5 if brain.ndim == 4 else brain > 0.5
        excess = (ltc - floor).clip(min=0)
        total = float(excess.sum())
        row["tc_outside_brain_frac"] = float(excess[~b].sum() / total) if total > 1e-6 else 0.0
        row["has_brain_mask"] = True
    else:
        row["tc_outside_brain_frac"] = 0.0
        row["has_brain_mask"] = False

    return row


def run_cohort(name: str, image_h5: Path, latent_h5: Path, out_dir: Path) -> Path:
    """Audit every scan of one cohort and write the CSV + summary JSON."""
    target_cfg = TargetConfig()  # defaults must match the cache (tumor_region='tc')
    _ = DerivationConfig()  # documents the pooling contract used below
    floor = 1.0 / (1.0 + math.exp(target_cfg.clip_vox / target_cfg.sdt_sigma_vox))

    with h5py.File(latent_h5, "r") as fl:
        if SOFT_GROUP not in fl:
            raise SystemExit(f"{name}: {SOFT_GROUP} absent — run mask_derive first")
        lat_ids = [_decode(x) for x in fl["ids"][:]]
        soft_all = fl[SOFT_GROUP]
        region = soft_all.attrs.get("tumor_region", b"?")
        region = region.decode() if isinstance(region, bytes) else str(region)
        has_oracle = ORACLE_GROUP in fl
        has_brain = BRAIN_GROUP in fl
        cached_all = soft_all[:]
        oracle_all = fl[ORACLE_GROUP][:] if has_oracle else None
        brain_all = fl[BRAIN_GROUP][:] if has_brain else None

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}__metrics.csv"
    rows: list[dict[str, Any]] = []

    with h5py.File(image_h5, "r") as fi:
        img_ids = [_decode(x) for x in fi["ids"][:]]
        img_index = {sid: i for i, sid in enumerate(img_ids)}
        for i, sid in enumerate(lat_ids):
            base = {"cohort": name, "scan_id": sid, "lat_row": i, "img_row": -1, "error": ""}
            j = img_index.get(sid)
            if j is None:
                base["error"] = "id_not_in_image_h5"
                rows.append(base)
                continue
            base["img_row"] = j
            try:
                label = fi["masks/tumor"][j].astype(np.int32)
                origin = fi["crop/origin"][j]
                metrics = audit_scan(
                    label=label,
                    crop_origin=origin,
                    cached=cached_all[i].astype(np.float32),
                    oracle=None if oracle_all is None else oracle_all[i].astype(np.float32),
                    brain=None if brain_all is None else brain_all[i].astype(np.float32),
                    target_cfg=target_cfg,
                    floor=floor,
                )
                base.update(metrics)
            except Exception as exc:
                base["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("%s %s failed: %s", name, sid, exc)
            rows.append(base)
            if (i + 1) % 100 == 0:
                logger.info("%s: %d/%d", name, i + 1, len(lat_ids))

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if not r["error"]]
    summary = {
        "cohort": name,
        "n_scans": len(rows),
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "tumor_region_attr": region,
        "has_oracle_group": has_oracle,
        "has_brain_mask": has_brain,
        "expected_floor": floor,
        "n_tc_empty": sum(1 for r in ok if r["gt_tc_vox"] == 0),
        "max_recompute_mae": max((r["recompute_mae"] for r in ok), default=None),
        "median_dice_tc_img": float(np.median([r["dice_tc_img"] for r in ok])) if ok else None,
        "median_lat_iou_tc": float(np.median([r["lat_iou_tc"] for r in ok])) if ok else None,
    }
    (out_dir / f"{name}__summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("%s done: %s", name, summary)
    return csv_path


def main() -> None:
    """CLI entry: audit one cohort selected by index or name."""
    p = argparse.ArgumentParser(description="Audit cached oracle soft masks for one cohort.")
    p.add_argument("--cohort-index", type=int, default=None, help="index into COHORTS (0-8)")
    p.add_argument("--cohort", type=str, default=None, help="cohort name")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--base", type=str, default=BASE)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cohort_index is not None:
        name, img_rel, lat_rel = COHORTS[args.cohort_index]
    elif args.cohort is not None:
        match = [c for c in COHORTS if c[0] == args.cohort]
        if not match:
            raise SystemExit(f"unknown cohort {args.cohort!r}")
        name, img_rel, lat_rel = match[0]
    else:
        raise SystemExit("give --cohort-index or --cohort")

    run_cohort(name, Path(args.base) / img_rel, Path(args.base) / lat_rel, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
