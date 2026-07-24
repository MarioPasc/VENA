"""Verify bit-exactness of the derived aug-bank soft masks vs clean-cache entries.

D3 correctness proof for TASK-19b (mask_derive_aug).

For v1/v2/v3 aug rows (intensity-only variants; label == source label, unchanged):
  max |derived_aug[i] - cached_clean[source_row_index[i]]| must equal 0.0

v4 rows (geometric variants; label is warped) are expected to differ and are
skipped.

Usage (on Picasso, after mask_derive_aug has written all 6 cohorts):
    python scripts/verify_aug_bit_exactness.py

Expected output (one line per cohort):
    UCSF-PDGM   v1/v2/v3 rows=N  max_abs_diff=0.000000  PASS
    BraTS-GLI   v1/v2/v3 rows=N  max_abs_diff=0.000000  PASS
    ...

If max_abs_diff > 0 for any cohort the script exits with code 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Cohort registry — Picasso paths (must match the per-cohort YAML configs)
# ---------------------------------------------------------------------------

FSCRATCH = Path("/mnt/home/users/tic_163_uma/mpascual/fscratch")
DATASETS = FSCRATCH / "datasets" / "vena"

_COHORTS: list[dict] = [
    {
        "name": "UCSF-PDGM",
        "image_aug_h5": DATASETS / "UCSF_PDGM" / "h5" / "ucsf_pdgm_image_aug.h5",
        "latent_aug_h5": DATASETS / "UCSF_PDGM" / "h5" / "ucsf_pdgm_latents_aug.h5",
        "latent_clean_h5": DATASETS / "UCSF_PDGM" / "h5" / "ucsf_pdgm_latents.h5",
    },
    {
        "name": "BraTS-GLI",
        "image_aug_h5": DATASETS / "BRATS_GLI" / "PRE_OPERATIVE" / "h5" / "brats_gli_image_aug.h5",
        "latent_aug_h5": DATASETS
        / "BRATS_GLI"
        / "PRE_OPERATIVE"
        / "h5"
        / "brats_gli_latents_aug.h5",
        "latent_clean_h5": DATASETS / "BRATS_GLI" / "PRE_OPERATIVE" / "h5" / "brats_gli_latents.h5",
    },
    {
        "name": "UPENN-GBM",
        "image_aug_h5": DATASETS / "upenn_gbm" / "h5" / "upenn_gbm_image_aug.h5",
        "latent_aug_h5": DATASETS / "upenn_gbm" / "h5" / "upenn_gbm_latents_aug.h5",
        "latent_clean_h5": DATASETS / "upenn_gbm" / "h5" / "upenn_gbm_latents.h5",
    },
    {
        "name": "IvyGAP",
        "image_aug_h5": DATASETS / "ivy_gap" / "h5" / "ivy_gap_image_aug.h5",
        "latent_aug_h5": DATASETS / "ivy_gap" / "h5" / "ivy_gap_latents_aug.h5",
        "latent_clean_h5": DATASETS / "ivy_gap" / "h5" / "ivy_gap_latents.h5",
    },
    {
        "name": "LUMIERE",
        "image_aug_h5": DATASETS / "lumiere" / "h5" / "lumiere_image_aug.h5",
        "latent_aug_h5": DATASETS / "lumiere" / "h5" / "lumiere_latents_aug.h5",
        "latent_clean_h5": DATASETS / "lumiere" / "h5" / "lumiere_latents.h5",
    },
    {
        "name": "REMBRANDT",
        "image_aug_h5": DATASETS / "rembrandt" / "h5" / "rembrandt_image_aug.h5",
        "latent_aug_h5": DATASETS / "rembrandt" / "h5" / "rembrandt_latents_aug.h5",
        "latent_clean_h5": DATASETS / "rembrandt" / "h5" / "rembrandt_latents.h5",
    },
]

_SOFT_MASK_GROUP = "masks/tumor_latent_soft"
# Rows with variant in this set are intensity-only: label unchanged from source.
_V1V2V3_VARIANTS = {"v1", "v2", "v3"}
# Maximum number of v1/v2/v3 rows to audit per cohort (set to None for all).
_AUDIT_ROWS_CAP: int | None = None


def _verify_cohort(cohort: dict) -> tuple[int, float]:
    """Verify bit-exactness for one cohort's v1/v2/v3 rows.

    Returns
    -------
    int
        Number of v1/v2/v3 rows audited.
    float
        Maximum absolute difference across all audited rows.
    """
    name = cohort["name"]
    aug_h5_path = Path(cohort["latent_aug_h5"])
    clean_h5_path = Path(cohort["latent_clean_h5"])

    if not aug_h5_path.exists():
        raise FileNotFoundError(f"[{name}] aug latent H5 not found: {aug_h5_path}")
    if not clean_h5_path.exists():
        raise FileNotFoundError(f"[{name}] clean latent H5 not found: {clean_h5_path}")

    with h5py.File(aug_h5_path, "r") as f_aug:
        if _SOFT_MASK_GROUP not in f_aug:
            raise KeyError(
                f"[{name}] {_SOFT_MASK_GROUP!r} not found in {aug_h5_path}. "
                "Run mask_derive_aug first."
            )
        aug_masks = f_aug[_SOFT_MASK_GROUP]  # (N_aug, 2, 48, 56, 48)
        aug_variants_raw = f_aug["variants"][:]  # (N_aug,)
        aug_src_idx = f_aug["source_row_index"][:].astype(np.int32)  # (N_aug,)

        aug_variants = [v.decode() if isinstance(v, bytes) else str(v) for v in aug_variants_raw]

        with h5py.File(clean_h5_path, "r") as f_clean:
            clean_masks = f_clean[_SOFT_MASK_GROUP]  # (N_clean, 2, 48, 56, 48)

            v123_indices = [i for i, var in enumerate(aug_variants) if var in _V1V2V3_VARIANTS]
            if _AUDIT_ROWS_CAP is not None:
                v123_indices = v123_indices[:_AUDIT_ROWS_CAP]

            if not v123_indices:
                return 0, 0.0

            max_abs = 0.0
            for i in v123_indices:
                src = int(aug_src_idx[i])
                aug_row = aug_masks[i]  # (2, 48, 56, 48)
                clean_row = clean_masks[src]  # (2, 48, 56, 48)
                diff = float(np.abs(aug_row - clean_row).max())
                if diff > max_abs:
                    max_abs = diff

    return len(v123_indices), max_abs


def main() -> None:
    """Run the bit-exactness audit for all 6 cohorts."""
    failures: list[str] = []
    print(f"{'Cohort':<16} {'v1/v2/v3 rows':>15} {'max_abs_diff':>14}  {'Status':>6}")
    print("-" * 60)

    for cohort in _COHORTS:
        name = cohort["name"]
        try:
            n_rows, max_abs = _verify_cohort(cohort)
        except (FileNotFoundError, KeyError) as exc:
            print(f"{name:<16} {'N/A':>15} {'N/A':>14}  ERROR: {exc}")
            failures.append(name)
            continue

        status = "PASS" if max_abs == 0.0 else "FAIL"
        print(f"{name:<16} {n_rows:>15} {max_abs:>14.6f}  {status}")
        if max_abs > 0.0:
            failures.append(name)

    print("-" * 60)
    if failures:
        print(f"FAIL: bit-exactness violated for cohorts: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("PASS: all cohorts are bit-exact for v1/v2/v3 rows.")
        sys.exit(0)


if __name__ == "__main__":
    main()
