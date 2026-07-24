"""Pre-write bit-exactness probe for the aug-bank soft-mask derivation.

Independent of ``mask_derive_aug`` and of ``verify_aug_bit_exactness.py``: this
derives a handful of rows directly, BEFORE any H5 is written, so the whole
plan can be gated on a few seconds of compute instead of an hours-long array.

The claim under test
--------------------
``v1``/``v2``/``v3`` aug variants are intensity-only, so their stored
``masks/tumor`` label is a verbatim copy of the source scan's label. Deriving
the soft ``[TC, NETC]`` mask from an aug row must therefore reproduce the
already-cached ``masks/tumor_latent_soft`` of the CLEAN latent H5 at row
``source_row_index`` **exactly** (``max |Δ| == 0.0``).

A non-zero result means the aug bank's crop convention differs from the clean
path's and the "reuse the same operator" premise is false — report it, do not
work around it.

``v4`` is the geometric variant; its label is warped, so it is expected to
differ and is only checked for the structural invariants (range, nesting).

Also reports per-cohort row counts and measured seconds-per-row so the SLURM
``--time`` can be sized from a measurement rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from vena.common import CropPadSpec
from vena.segmentation.config import DerivationConfig, TargetConfig
from vena.segmentation.derivation.derive import derive_latent_soft_mask

# The aug bank is pre-cropped to the common brain-centred box, so the
# crop/pad step is a no-op and the derivation reduces to SDT -> sigmoid ->
# avg_pool3d(4). Keep this in sync with vena.data.h5.augmented.
LATENT_CROP_BOX = (192, 224, 192)

REGISTRY = Path("routines/fm/train/configs/corpus/corpus_picasso.json")


def _cohorts_from_registry(registry: Path) -> dict[str, tuple[Path, Path]]:
    """Resolve {cohort: (aug_image_h5, clean_latent_h5)} from the corpus registry.

    Paths are read from the registry rather than reconstructed: the nine
    cohorts live in nine distinct directories with inconsistent casing, and a
    hand-written path table silently resolves for some cohorts and not others.
    """
    raw = json.loads(registry.read_text())
    entries = raw["cohorts"] if isinstance(raw, dict) and "cohorts" in raw else raw
    entries = entries if isinstance(entries, list) else list(entries.values())
    out: dict[str, tuple[Path, Path]] = {}
    for c in entries:
        aug = c.get("latent_aug_h5")
        if not aug:  # test_only cohorts have no aug bank
            continue
        # The aug IMAGE h5 is the aug LATENT h5 with the domain token swapped.
        out[c["name"]] = (
            Path(str(aug).replace("_latents_aug.h5", "_image_aug.h5")),
            Path(c["latent_h5"]),
        )
    return out


def _derive(label: np.ndarray, targets: TargetConfig, derivation: DerivationConfig) -> np.ndarray:
    """Derive the soft ``(2, 48, 56, 48)`` mask from a pre-cropped label."""
    spec = CropPadSpec(
        crop_origin=(0, 0, 0),
        native_shape=(label.shape[0], label.shape[1], label.shape[2]),
        target_shape=LATENT_CROP_BOX,
    )
    out = derive_latent_soft_mask(
        source="gt",
        label=label.astype(np.int32),
        crop_spec=spec,
        cfg=derivation,
        target_cfg=targets,
    )
    return out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)


def _probe_cohort(name: str, aug_path: Path, clean_path: Path, n_rows: int) -> dict[str, object]:
    res: dict[str, object] = {"cohort": name}
    if not aug_path.exists():
        return {**res, "status": f"MISSING aug h5: {aug_path}"}
    if not clean_path.exists():
        return {**res, "status": f"MISSING clean h5: {clean_path}"}

    targets, derivation = TargetConfig(), DerivationConfig()

    with h5py.File(aug_path, "r") as fa, h5py.File(clean_path, "r") as fc:
        if "masks/tumor_latent_soft" not in fc:
            return {**res, "status": "clean cache has no masks/tumor_latent_soft"}
        variants = np.asarray(
            [v.decode() if isinstance(v, bytes) else v for v in fa["variants"][:]]
        )
        src_idx = fa["source_row_index"][:]
        n_total = len(variants)
        res["n_rows_total"] = int(n_total)
        res["variant_counts"] = {v: int((variants == v).sum()) for v in sorted(set(variants))}

        intensity_rows = np.where(np.isin(variants, ["v1", "v2", "v3"]))[0]
        geom_rows = np.where(variants == "v4")[0]

        # --- the load-bearing check: intensity-only rows must be bit-exact ---
        # When they are NOT, localise the disagreement: the expected mechanism
        # is that apply_crop_pad ZERO-pads the clean path outside the native
        # volume (so cached == 0 there) while the pre-cropped aug label yields
        # the SDT far-field floor. If every differing voxel sits where
        # cached == 0, the discrepancy is confined to that pad margin.
        max_abs, checked, t0 = 0.0, 0, time.time()
        diff_frac, in_pad_frac, interior_max = 0.0, 1.0, 0.0
        hot_max, dice_min = 0.0, 1.0
        for i in intensity_rows[:n_rows]:
            derived = _derive(fa["masks/tumor"][i], targets, derivation)
            cached = np.asarray(fc["masks/tumor_latent_soft"][int(src_idx[i])])
            delta = np.abs(derived - cached)
            max_abs = max(max_abs, float(delta.max()))
            differing = delta > 0
            if differing.any():
                diff_frac = max(diff_frac, float(differing.mean()))
                pad = cached == 0.0
                in_pad_frac = min(in_pad_frac, float((differing & pad).sum() / differing.sum()))
                interior = delta[~pad]
                interior_max = max(interior_max, float(interior.max()) if interior.size else 0.0)
            # The decision-relevant question is not "is anything different" but
            # "is the region that actually drives the ControlNet different".
            # Conditioning is dominated by the supra-threshold tumour core; a
            # far-field wobble of a few 1e-2 near the SDT floor is cosmetic,
            # a change inside TC is not.
            hot = cached >= 0.5
            if hot.any():
                hot_max = max(hot_max, float(delta[hot].max()))
            both = (derived >= 0.5).astype(np.float64), (cached >= 0.5).astype(np.float64)
            inter = float((both[0] * both[1]).sum())
            denom = float(both[0].sum() + both[1].sum())
            if denom > 0:
                dice_min = min(dice_min, 2.0 * inter / denom)
            checked += 1
        elapsed = time.time() - t0
        res["intensity_rows_checked"] = checked
        res["max_abs_diff"] = max_abs
        res["diff_voxel_fraction"] = diff_frac
        res["frac_of_diffs_in_zero_pad"] = in_pad_frac
        res["max_abs_diff_outside_pad"] = interior_max
        res["sec_per_row"] = round(elapsed / checked, 3) if checked else None
        res["bit_exact"] = bool(max_abs == 0.0)
        res["bit_exact_outside_pad"] = bool(interior_max == 0.0)
        res["max_abs_diff_in_supra_threshold"] = hot_max
        res["min_dice_thresholded"] = dice_min

        # --- v4 (warped): structural invariants only ---
        viol, rng_bad, n_geom = 0, 0, 0
        for i in geom_rows[: max(1, n_rows // 2)]:
            d = _derive(fa["masks/tumor"][i], targets, derivation)
            n_geom += 1
            if d.min() < 0.0 or d.max() > 1.0:
                rng_bad += 1
            if (d[1] > d[0] + 1e-6).any():  # NETC must nest inside TC
                viol += 1
        res["v4_checked"] = n_geom
        res["v4_range_violations"] = rng_bad
        res["v4_nesting_violations"] = viol

    res["status"] = "ok"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=6, help="intensity-only rows probed per cohort")
    ap.add_argument("--cohorts", nargs="*", default=None, help="subset of cohort names")
    args = ap.parse_args()

    cohorts = _cohorts_from_registry(REGISTRY)
    names = args.cohorts or list(cohorts)
    results = [_probe_cohort(n, *cohorts[n], args.rows) for n in names]

    print(json.dumps(results, indent=2))
    failed = [r for r in results if r.get("status") != "ok" or not r.get("bit_exact")]
    print("\n=== VERDICT ===")
    for r in results:
        print(
            f"  {r['cohort']:<24} status={r.get('status')} "
            f"max_abs={r.get('max_abs_diff')} "
            f"max_abs_outside_pad={r.get('max_abs_diff_outside_pad')} "
            f"hot_max={r.get('max_abs_diff_in_supra_threshold')} dice={r.get('min_dice_thresholded')} "
            f"rows={r.get('n_rows_total')} s/row={r.get('sec_per_row')}"
        )
    print("ALL BIT-EXACT" if not failed else f"NOT BIT-EXACT: {[r['cohort'] for r in failed]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
