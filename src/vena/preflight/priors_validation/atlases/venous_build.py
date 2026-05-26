"""In-house dural-sinus mask construction from UCSF-PDGM T1Gd MIPs.

Spec §3 procedure (i–v):

  (i) Select subjects with a midline-shift proxy < 3 mm and (optionally)
      WHO grade ≤ ``who_grade_max``. UCSF-PDGM has no explicit MLS field;
      we use ``tumour_volume_ml + lateral_centroid_offset`` as a proxy and
      pick the lowest-asymmetry N subjects.
 (ii) Register each subject's T1Gd → MNI152NLin2009c via ANTs SyN.
(iii) Per subject, extract top-``intensity_percentile`` voxels within a
      superior axial slab (default MNI z ∈ ``axial_slab_z_mni``).
 (iv) Average the indicator across subjects (probability map).
  (v) Threshold at ``voting_threshold``; drop connected components below
      ``min_component_size`` voxels; save NIfTI in MNI152 space.

The build is computationally heavy (~5 min SyN per subject ⇒ ~2.5 h for
30 subjects on a 32-core CPU node). Each subject's transform is cached
under ``cache_root/<subject_sha>_syn/`` so re-runs are nearly free.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, label

from vena.data.niigz import UCSFPDGMDataset, UCSFPDGMPatient

from .fetch import ensure_atlases

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VenousBuildConfig:
    dataset_root: Path
    metadata_csv: Path | None
    output_root: Path
    atlases_root: Path
    cache_root: Path
    n_subjects: int = 30
    who_grade_max: int | None = None
    midline_shift_mm_max: float = 3.0
    intensity_percentile: float = 95.0
    axial_slab_z_mni: tuple[float, float] = (25.0, 80.0)
    voting_threshold: float = 0.5
    min_component_size: int = 100
    seed: int = 1337
    n_workers: int = 1
    log_level: str = "INFO"


def _midline_offset_mm(brain_mask_arr: np.ndarray, spacing_mm: tuple[float, ...]) -> float:
    """Distance (mm) between the brain centroid's X coordinate and the
    geometric mid-plane. Used as a midline-shift proxy in absence of an
    explicit MLS field in UCSF-PDGM metadata.
    """
    if brain_mask_arr.sum() == 0:
        return float("inf")
    cx, _, _ = center_of_mass(brain_mask_arr > 0)
    return abs(cx - (brain_mask_arr.shape[0] / 2.0)) * float(spacing_mm[0])


def _tumour_volume_ml(tumour_arr: np.ndarray, spacing_mm: tuple[float, ...]) -> float:
    if tumour_arr is None:
        return 0.0
    return float((tumour_arr > 0).sum()) * float(np.prod(spacing_mm)) / 1000.0


def _select_subjects(
    dataset: UCSFPDGMDataset,
    cfg: VenousBuildConfig,
) -> list[UCSFPDGMPatient]:
    """Spec §3 step (i) — pick the lowest-midline-shift subjects."""
    candidates: list[tuple[float, UCSFPDGMPatient]] = []
    rng = random.Random(cfg.seed)
    # Shuffle to break ties non-deterministically beyond the cohort default
    pool = list(dataset)
    rng.shuffle(pool)
    for p in pool:
        meta = p.metadata or {}
        grade_raw = meta.get("WHO CNS Grade") or meta.get("who_grade")
        if cfg.who_grade_max is not None and grade_raw is not None:
            try:
                if int(grade_raw) > cfg.who_grade_max:
                    continue
            except (ValueError, TypeError):
                pass
        try:
            brain = dataset.load_brain_mask(p)
            tumour = dataset.load_tumor_seg(p)
        except Exception:
            continue
        offset = _midline_offset_mm(np.asarray(brain.array), brain.spacing_mm)
        tvol = _tumour_volume_ml(np.asarray(tumour.array), tumour.spacing_mm)
        if offset > cfg.midline_shift_mm_max * 2.0:
            # Hard cap — too eccentric a brain centroid suggests gross deformity
            continue
        # Combined score: smaller is better.
        score = offset + 0.05 * tvol
        candidates.append((score, p))
    candidates.sort(key=lambda kv: kv[0])
    picks = [p for _, p in candidates[: cfg.n_subjects]]
    logger.info(
        "Selected %d subjects (score range %.2f – %.2f mm-equivalent)",
        len(picks),
        candidates[0][0] if candidates else 0.0,
        candidates[len(picks) - 1][0] if candidates and len(picks) > 0 else 0.0,
    )
    return picks


def _register_t1gd_to_mni(
    patient: UCSFPDGMPatient,
    dataset: UCSFPDGMDataset,
    mni_template_path: Path,
    cache_dir: Path,
) -> Path | None:
    """SyN-register the subject's T1Gd to the MNI152 template; return the
    warped T1Gd path (or None on failure)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    warped_path = cache_dir / f"{patient.patient_id}_t1gd_in_mni.nii.gz"
    if warped_path.exists():
        logger.info("[%s] cached warped T1Gd: %s", patient.patient_id, warped_path)
        return warped_path
    try:
        t1gd = dataset.load_modality(patient, "T1c_bias")
    except Exception:
        try:
            t1gd = dataset.load_modality(patient, "T1c")
        except Exception:
            return None
    moving_path = cache_dir / f"{patient.patient_id}_t1gd_native.nii.gz"
    nib.save(
        nib.Nifti1Image(np.asarray(t1gd.array), t1gd.affine, t1gd.header),
        str(moving_path),
    )
    fixed = ants.image_read(str(mni_template_path))
    moving = ants.image_read(str(moving_path))
    t0 = time.time()
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        outprefix=str(cache_dir / f"{patient.patient_id}_ants_"),
    )
    logger.info("[%s] SyN T1Gd→MNI in %.1fs", patient.patient_id, time.time() - t0)
    warped = reg["warpedmovout"]
    ants.image_write(warped, str(warped_path))
    return warped_path


def _build_indicator(
    warped_t1gd_path: Path,
    mni_shape: tuple[int, int, int],
    mni_affine: np.ndarray,
    cfg: VenousBuildConfig,
) -> np.ndarray:
    """Spec §3 step (iii) — top-percentile bright voxels in the superior slab."""
    img = nib.load(str(warped_t1gd_path))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    if arr.shape != mni_shape:
        # Resample to MNI shape if needed (very rare; ANTs SyN already lands
        # in the template grid). Fail loudly.
        raise RuntimeError(f"Warped T1Gd has shape {arr.shape} ≠ MNI shape {mni_shape}")
    # MNI z slab: convert mm range to voxel indices via the affine. For
    # MNI152NLin2009c at 1 mm isotropic the affine is diagonal so z_vox = z_mm − origin_z.
    inv = np.linalg.inv(mni_affine)
    z_lo_vox = int(np.round(inv[2, 3] + cfg.axial_slab_z_mni[0] * inv[2, 2]))
    z_hi_vox = int(np.round(inv[2, 3] + cfg.axial_slab_z_mni[1] * inv[2, 2]))
    z_lo, z_hi = sorted((z_lo_vox, z_hi_vox))
    z_lo = max(0, z_lo)
    z_hi = min(arr.shape[2], z_hi)
    slab_mask = np.zeros_like(arr, dtype=bool)
    slab_mask[:, :, z_lo:z_hi] = True
    in_slab = arr[slab_mask]
    if in_slab.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    threshold = float(np.percentile(in_slab, cfg.intensity_percentile))
    indicator = ((arr >= threshold) & slab_mask).astype(np.float32)
    return indicator


def build_venous_atlas(cfg: VenousBuildConfig) -> dict[str, Any]:
    """Construct the in-house dural-sinus mask. See module docstring."""
    cache_dir = Path(cfg.cache_root) / "venous_build_warps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = ensure_atlases(Path(cfg.atlases_root))
    mni_path = bundle.mni152_t1w

    dataset = UCSFPDGMDataset(cfg.dataset_root, cfg.metadata_csv)
    picks = _select_subjects(dataset, cfg)
    if len(picks) < 5:
        raise RuntimeError(f"Only {len(picks)} subjects pass selection — need >= 5 to average.")

    mni_img = nib.load(str(mni_path))
    mni_affine = mni_img.affine
    mni_shape = mni_img.shape

    accumulator = np.zeros(mni_shape, dtype=np.float32)
    n_contributing = 0
    used_subjects: list[str] = []
    for p in picks:
        warped = _register_t1gd_to_mni(p, dataset, mni_path, cache_dir)
        if warped is None:
            continue
        try:
            ind = _build_indicator(warped, mni_shape, mni_affine, cfg)
        except Exception as exc:
            logger.exception("[%s] indicator build failed: %s", p.patient_id, exc)
            continue
        accumulator += ind
        n_contributing += 1
        used_subjects.append(p.patient_id)

    if n_contributing < 5:
        raise RuntimeError(f"Only {n_contributing} subjects contributed; aborting.")

    probability = accumulator / float(n_contributing)
    binary = (probability >= cfg.voting_threshold).astype(np.uint8)

    # Connected-components clean-up — drop small islands
    cc_labels, n_cc = label(binary)
    if n_cc > 0:
        counts = np.bincount(cc_labels.ravel())
        keep = np.zeros_like(counts, dtype=bool)
        for i in range(1, len(counts)):
            if counts[i] >= cfg.min_component_size:
                keep[i] = True
        binary = keep[cc_labels].astype(np.uint8)

    # Save outputs under a UTC-timestamped sub-directory; update LATEST symlink.
    import datetime as _dt

    timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = Path(cfg.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    nib.save(
        nib.Nifti1Image(binary, mni_affine, mni_img.header),
        str(out_dir / "venous_mask_MNI152.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(probability, mni_affine, mni_img.header),
        str(out_dir / "venous_probability_MNI152.nii.gz"),
    )
    voxel_vol_ml = float(np.prod(mni_img.header.get_zooms()[:3])) / 1000.0
    mask_volume_ml = float(binary.sum()) * voxel_vol_ml

    manifest = {
        "schema_version": "1.0",
        "timestamp_utc": timestamp,
        "n_subjects_contributing": n_contributing,
        "used_subject_ids": used_subjects,
        "voting_threshold": cfg.voting_threshold,
        "intensity_percentile": cfg.intensity_percentile,
        "axial_slab_z_mni": list(cfg.axial_slab_z_mni),
        "min_component_size": cfg.min_component_size,
        "mask_volume_ml": mask_volume_ml,
        "mni_template_path": str(mni_path),
    }
    import json

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    latest = Path(cfg.output_root) / "LATEST"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    logger.info(
        "Venous atlas built: %s  (mask volume %.1f ml from %d subjects)",
        out_dir,
        mask_volume_ml,
        n_contributing,
    )
    return manifest
