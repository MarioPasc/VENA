# Mask audit — oracle soft-mask invariant test

Audits the cached oracle soft mask `masks/tumor_latent_soft` `(N,2,48,56,48)` in every cohort's
latent H5. **Every metric is recomputed from the actual GT labels** in the image-domain H5; nothing
is taken on trust from the cache.

Channel 0 = **TC** (tumour core = NETC+ET, edema excluded), channel 1 = **NETC**
(see `project_channel0_tumor_core_not_wt`).

## Files

| file | role |
|---|---|
| `audit_cohort.py` | per-cohort worker (one SLURM array task) → `<cohort>__metrics.csv` + `__summary.json` |
| `flag_and_visualize.py` | merge → flag outliers → per-cohort summary → per-patient figures |
| `array_mask_audit.sh` | SLURM array `0-8`, CPU-only, 6 h, 32 G |

## Invariants covered

**Structural** — shape/dtype; no NaN/Inf (`lat_nan_count`); values in `[0,1]` (`lat_range_ok`);
`tumor_region` attr; the SDT floor is `sigmoid(-clip_vox/sdt_sigma_vox) = 0.034445`.

> ⚠ **Refuted premise (measured 2026-07-24):** "the cached far-field equals the floor" is **false**,
> so there is deliberately **no `below_floor` check**. `apply_crop_pad` **zero-pads** wherever the
> `(192,224,192)` box extends past the native volume, so `lat_tc_min == 0` on every scan (148/148 in
> the dry run) while a TC-empty scan's `lat_tc_max == floor` *exactly* — i.e. the floor holds in the
> un-padded interior only. An initial `lat_tc_min < floor` rule flagged 100% of scans as FAIL.
> Non-negativity is already covered by `lat_range_ok`.

**Semantic** — nesting `NETC ≤ TC` in both the latent cache (`lat_nesting_viol_frac`) and the
image-domain soft map (`nesting_viol_frac_img`); tumour-present ⇒ the mask has structure
(`lat_tc_max > floor`); tumour-absent ⇒ the mask is flat at the floor (**no phantom tumour**).

**GT ↔ soft overlap** — hard GT ⊆ `{soft > 0.5}` (`hard_subset_soft_viol_frac_*`);
`Dice(hard, soft>0.5)` (`dice_tc_img`, `dice_netc_img`); volume calibration
`|soft>0.5| / |hard|` (`volratio_tc_img`, expect ≈1).

**Continuity** — `soft_intermediate_frac_tc`: fraction of voxels strictly between the floor and
saturation. A *binary* mask scores ≈0 here. This is exactly the defect an earlier QC round missed
(a binary WT was rendered while being labelled "soft"), so it is now measured, not eyeballed.

**Registration / exactness**
- `recompute_mae` / `recompute_max_abs` — re-derive from GT and re-pool through the **canonical**
  path (`apply_crop_pad` → `avg_pool3d(k=4)`); the result must reproduce the cache **bit-exactly**.
  This is the single strongest check: it validates derivation *and* registration *and* the write.
- `lat_iou_tc`, `lat_centroid_dist_tc` — cached latent upscaled ×4 vs the image-domain soft map.
- `oracle_tc_iou`, `oracle_tc_centroid_dist` — independent cross-check against the pre-existing
  `masks/tumor_latent` (`NETC + ET`), which was produced by a *different* pipeline.

**Geometry / containment** — `crop_clip_frac_tc`: the `(192,224,192)` crop must not clip the tumour;
`tc_outside_brain_frac`: TC mass must lie inside `masks/brain_latent`;
`lat_tc_mass_ratio`: mass conservation under 4× avg-pool (64 image voxels per latent voxel).

## Outlier policy

Two tiers, because the two failure classes need different logic.

### Tier 1 — `FAIL` (absolute)

Logical invariants that cannot be violated at *any* tumour size. One breach is a defect regardless
of the population distribution:

| reason | condition |
|---|---|
| `nan_in_cache` | `lat_nan_count > 0` |
| `value_out_of_range` | any value outside `[0,1]` |
| `nesting_NETC>TC` / `nesting_img_NETC>TC` | violation fraction `> 1e-6` |
| `hardTC_not_subset_of_soft` | `> 1%` of hard-TC voxels have `soft ≤ 0.5` |
| `crop_clips_tumour` | `> 1%` of TC lost to the crop box |
| `cache_ne_canonical_rederive` | `recompute_max_abs > 1e-5` |
| `tumour_lost_in_cache` | GT TC > 0 but `lat_tc_max ≤ floor + 0.01` |
| `phantom_tumour` | GT TC == 0 but `lat_tc_max > floor + 0.05` |
| `tc_outside_brain` | `> 25%` of TC mass outside the brain mask |
| `low_img_dice` | `dice_tc_img < 0.90` |

`tc_outside_brain` is deliberately **25%, not 5%**: at the coarse latent grid a graded halo around a
brain-edge tumour legitimately spills past the brain mask (measured median 0.006, max 0.097 with no
other defect present), and there is no defensible absolute basis for a tight cut. Only a clearly
pathological fraction is an absolute failure; relative outliers are caught by the statistical tier,
where `tc_outside_brain_frac` is included as a high-is-bad metric.

### Tier 2 — `WARN` / `INFO` (size-stratified robust z)

For the continuous agreement metrics "bad" is relative, so we use the Iglewicz–Hoaglin modified
z-score, one-sided into the bad tail:

```
z = 0.6745 · (x − median) / MAD          (high-is-bad)
z = 0.6745 · (median − x) / MAD          (low-is-bad)
```

`WARN` if `z > 3.5`, `INFO` if `3.0 < z ≤ 3.5`. **MAD, not σ**, so the outliers cannot inflate
their own cutoff.

### The effect-size gate (required — z alone is not enough)

A scan is flagged only when it is **both** statistically unusual (`z >` threshold) **and** off the
stratum median by at least a practically meaningful amount (`MIN_EFFECT`). Robust z is
*hypersensitive on tightly-concentrated or zero-inflated metrics*: pooled mass ratio clusters at
≈1 and brain-spill at ≈0, so MAD collapses toward zero and a trivial deviation scores `z > 3.5`.
Measured on the dry run, z-alone produced 67 spurious `tc_outside_brain_frac` and 44 spurious
`abs_log_massratio_tc` WARNs; adding the gate removed both entirely (119 → 43 WARN over 503 scans).

| metric | `MIN_EFFECT` |
|---|---|
| `lat_iou_tc` | 0.10 IoU points |
| `lat_centroid_dist_tc` | 1.0 voxel (1 latent vox = 4 image vox) |
| `abs_log_volratio_tc` | 0.05 (~5% volume error) |
| `abs_log_massratio_tc` | 0.05 (~5% pooled-mass error) |
| `oracle_tc_centroid_dist` | 1.0 voxel |
| `soft_intermediate_frac_tc` | 0.005 |
| `tc_outside_brain_frac` | 0.05 (5% of TC mass) |

`max_z` (used to rank WARNs for figure selection) is the max over **gated** z-scores, so the
ranking reflects material deviations only.

Metrics and bad direction: `lat_iou_tc` (low), `lat_centroid_dist_tc` (high),
`abs_log_volratio_tc` (high), `abs_log_massratio_tc` (high), `oracle_tc_centroid_dist` (high),
`soft_intermediate_frac_tc` (low), `tc_outside_brain_frac` (high).

**Stratification is load-bearing, not cosmetic.** Latent↔image IoU degrades for small and
multifocal cores purely from 4× avg-pool quantization — one latent voxel spans 4×4×4 image voxels,
so a lesion only a few latent voxels across cannot score a high IoU no matter how correct it is.
This was established concretely on `UCSF-PDGM-0367` (multifocal small TC: IoU 0.332, centroid
4.79 vox, yet the mask is on the right lesion). An unstratified threshold would flag every small
tumour and bury the real defects.

Strata are **`cohort × size-quartile`**, not size alone. Cohorts differ systematically (voxel size,
acquisition, pathology mix), and with global strata those differences masquerade as per-patient
outliers — measured, small OOD cohorts (BraTS-Africa, BraTS-PED) were flagged en masse. Cohort-level
effects belong in `per_cohort_summary.csv`, not in a per-patient outlier list. Bin count adapts to
cohort size so each stratum keeps a stable MAD: 4 bins at n≥64, 3 at n≥36, 2 at n≥16, else 1.

TC-empty scans form their own stratum and are excluded from Tier 2 (their IoU is undefined); they
remain fully covered by Tier 1 via `phantom_tumour`.

A stratum needs ≥8 valid scans and non-degenerate MAD, else that metric is skipped there.

## Figures

One PNG per flagged patient, **4 rows × up to 10 columns**:

- **Columns** — evenly spaced axial slices spanning the **GT-TC** extent
  (`linspace(first_tumour_z, last_tumour_z, n)`), `n = min(10, n_tumour_slices)` so a lesion
  spanning ≤7 slices simply yields fewer columns. TC-empty scans fall back to the whole box.
- **Rows** — `GT TC (binary)` / `Soft TC` / `GT NETC (binary)` / `Soft NETC`.
  GT rows render as **binary** (flat fill + a crisp 0.5 outline). Soft rows render as **continuous
  probability** (`YlGn` for TC, `RdPu` for NETC, alpha ∝ probability, contours at 0.25/0.5/0.75) —
  the same conventions as `vena.segmentation.metrics.visualize`.
- All volumes are in the **crop frame** `(192,224,192)`; the soft rows show the **cached latent
  upscaled ×4**, i.e. the artifact the model actually consumes (deliberately blocky — that is the
  real resolution of the conditioning signal).
- The title carries the flag, the reasons, and the key metrics so each figure is self-explanatory.

Selection: every `FAIL`, the worst `MAX_WARN_FIGS_PER_COHORT` WARNs per cohort by `max_z`, plus
`N_EXEMPLARS_PER_COHORT` median-Dice `EXEMPLAR`s and one `TC_EMPTY` per cohort — a folder of only
outliers gives no baseline to judge them against. **Every cap applied is logged at WARNING level**
and the full flag list always remains in `audit_flags.csv` (no silent truncation).

**Every cohort is guaranteed ≥1 figure even when nothing is flagged.** The `EXEMPLAR` pick normally
provides this; if a cohort has no eligible OK/TC-present scan (all TC-empty, or all errored) a
`COHORT_FALLBACK` figure is forced from its largest-TC scan and logged. `select_for_figures` raises
if any cohort would end up with zero figures, so a silently-unvisualised cohort is impossible.

## Outputs

`audit_metrics_all.csv` (per scan, all metrics + z-scores + flags) · `audit_flags.csv`
(non-OK only) · `per_cohort_summary.csv` · `audit_summary.json` (counts, thresholds, reason
histogram, caps) · `figures_index.csv` · `figures/`.

## Re-run

```bash
sbatch scripts/mask_audit/array_mask_audit.sh          # 9 CPU tasks, one per cohort
# then MANUALLY (Picasso resolves array dependencies to Dependency=(null) — never chain it):
PYTHONPATH=$REPO/src:$REPO/scripts/mask_audit python scripts/mask_audit/flag_and_visualize.py \
    --in-dir  ~/execs/vena/mask_audit/metrics \
    --out-dir ~/execs/vena/mask_audit/report \
    --base    .../fscratch/datasets/vena
```

**For S6 (predicted masks):** point `SOFT_GROUP` at `masks/tumor_latent_pred`. Tier 1 mostly still
applies, but `recompute_mae` and `dice_tc_img` do **not** — a predicted mask is not expected to
reproduce GT — so those two become the *oracle→predicted gap* measurement rather than pass/fail
gates. Keep Tier 2 as-is.
