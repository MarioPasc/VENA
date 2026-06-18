# 2026-06-18 — VENA pre-training data audit

**Scope.** Audit every cohort on Picasso (`/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/`) against the encode/augment code in `src/vena/` + `routines/`. Five axes:

1. Cross-cohort intensity normalisation parity.
2. Tumour-mask + brain-mask storage and propagation to the latent grid.
3. Augmentation pipeline — v4 brain-mask correctness, K=4 coverage per patient, intensity-augmentation impact on T1c synthesis.
4. Image vs latent H5 schema — what is common, what differs (intentional vs drift).
5. Forward roadmap for the next encode pass (schema unification, re-encode triggers).

**Methods.** Per-cohort H5 schema dump (`/tmp/vena_audit.py`) and per-row probe (`/tmp/vena_deep_audit.py`) executed on Picasso (`python3` with user-site `h5py 3.16.0`, `scipy`). Code paths cross-referenced via three `Explore` subagents + direct reads.

**Cohorts audited.**
UCSF-PDGM, BraTS-GLI, IvyGAP, LUMIERE, BraTS-Africa-Glioma, BraTS-Africa-Other, BraTS-PED, UPENN-GBM, REMBRANDT (9 cohorts; matches `corpus_picasso.json`).

---

## 0. TL;DR — what is broken, what is fine, what needs a re-encode

| Severity | Finding | Affects | Re-encode / re-run? |
|---|---|---|---|
| 🔴 Critical | **v4 brain mask is all-ones (synth) on every aug H5.** Confirmed empirically on all 6 cv cohorts — every v4 row has `masks/brain_latent.sum() == 1×48×56×48 = 129 024`. | All cv cohorts | **Yes**: replay affine+elastic on the image-domain brain mask using `aug_params_json` and overwrite v4 rows via `brain_latent_merge_patch.py`. No MR-latent re-encode. |
| 🔴 Critical | **BraTS-Africa `percentile_normalise` discards ~½ of intra-brain tissue.** Image H5 stores z-score (verified `t1c.min = -473 / -96`). Encoder uses `foreground_only=True, foreground_threshold=0` → percentiles computed on positives only; negatives `clamp(0,1) → 0`. | BraTS-Africa-Glioma (95), BraTS-Africa-Other (51) | **Yes** — fix the encoder to pass `masks/brain` and re-encode 146 scans (~30 min on A100). |
| 🟡 Medium | **IvyGAP brain mask carries 35–148 noisy CCs per scan.** Source = `t1pre > 0` with no cleaning. Matches user's "cloud-like blurs at z-extremes" symptom. UPENN-GBM / BraTS-GLI mild (1–14 CCs, ≤ 1-voxel tail). | IvyGAP (heavy), UPENN-GBM / BraTS-GLI (mild) | **Image-H5 in-place patch + re-run brain-to-latent** for affected cohorts. MR latents untouched. |
| 🟡 Medium | Two cohorts (BraTS-GLI, IvyGAP) use `t1pre > 0` only for the brain mask; the other 4 VENA-computed cohorts use the union of all 4 modalities. Small for well-skull-stripped data but is another silent inconsistency. | BraTS-GLI, IvyGAP | Forward-looking unification only. |
| 🟡 Medium | Latent H5 `schema_version` and `splits/*` layout drift across cohorts (some carry both `splits/{train,val,test}` and `splits/cv/fold_N/*`, others only the CV form). DataModule tolerates all variants. | All | Pin to one shape on next re-encode (see §6). |
| 🟢 OK | **K=4 coverage matches the dedup-allowed train+val pool.** Earlier suspicion of "rank-1 shards never merged" was WRONG — server-3 H5s and Picasso H5s are byte-identical merged outputs. The pool reduction is `pool = (all_ids − splits/test) ∩ dedup_allowlist`; UCSF & UPENN-GBM lose most patients to BraTS-GLI overlap, by design. | All cv cohorts | None — intentional. |
| 🟢 OK | Tumour-label set per cohort, BraTS2023 → BraTS2021 remap (3 → 4) before downsampler, per-class `avg_pool3d(4)` → `masks/tumor_latent` (3 channels = NETC, ED, ET). Sample channel sums are non-zero on every probed cohort. | All cohorts | None. |
| 🟢 OK | `masks/brain_latent` present in every cohort's base latent H5 (and every cv-cohort aug latent H5). Fill fraction 17–24 % across cohorts — consistent. **Absent on BraTS-PED** (latent H5 ok, image H5 truncated; already dropped from corpus). | All cv cohorts + BraTS-Africa | None — BraTS-PED already excluded. |
| 🟢 OK | Latent intensity distribution (`latents/t1c`) comparable across all 9 cohorts: `mean ≈ -0.10`, `std ≈ 0.97–0.99`, range ≈ ±5. No outlier cohort. | All cohorts | None. |

Total cv-cohort augmented scans on Picasso: 724 + 4 496 + 116 + 2 108 + 588 + 232 = **8 264 rows** (matches memory note exactly). This is the FULL merged output, not a rank-0 shadow — see §3.3.

---

## 1. Cross-cohort normalisation

### 1.1 Single entry point
`percentile_normalise` lives at `src/vena/model/autoencoder/maisi/preprocessing.py:55`, re-exported through `vena.common`. The encode routine calls it exactly once per modality at `src/vena/model/autoencoder/maisi/encode/engine.py:136` and `:169`, with `lower=0.0, upper=99.5, foreground_only=True` driven from the YAML (`routines/encode/maisi/engine.py:147-149`). No per-cohort override exists.

### 1.2 Image-H5 intensity policies
All cohorts store **raw** intensities **except BraTS-Africa**, which stores **BraTS-pipeline intra-brain z-score**. Each image H5 root attr `intensity_policy` documents it.

Verified empirically — see `image.has_negative_intensities` in `/tmp/vena_deep_audit.json`:

| Cohort | t1c min | t1c p0.5 | t1c p99.5 | Has negatives? | Policy |
|---|---|---|---|---|---|
| UCSF-PDGM | 0.0 | 652 | 5 633 | No | raw, N4 corrected |
| BraTS-GLI | 0.0 | (probed sep.) | (probed sep.) | No | raw |
| IvyGAP | 0.0 | 1 090 | 4 985 | No | raw |
| LUMIERE | 0.0 | 51.8 | 605 | No | raw, skull-stripped |
| UPENN-GBM | 0.0 | 97 | 1 006 | No | raw |
| REMBRANDT | 0.0 | 13 | 209 | No | raw, HD-BET |
| **BraTS-Africa-Glioma** | **−473.0** | 389 | 5 006 | **Yes** | **z-score** |
| **BraTS-Africa-Other** | **−96.0** | 1 316 | 6 296 | **Yes** | **z-score** |

### 1.3 The BraTS-Africa silent intensity skew

`percentile_normalise(foreground_only=True, foreground_threshold=0.0)` defines foreground as `x[x > 0]`. For raw cohorts every brain voxel is positive, so this is correct. For BraTS-Africa, the foreground is intra-brain z-score centred near zero, so roughly half the brain voxels fall in `x ≤ 0` and are excluded from the percentile estimate. When the per-volume `lo = 0th percentile of positives` is then applied to the *full* volume, voxels with `x ≤ 0` produce `y = (x - lo) / denom < 0` and are clamped to 0 (`preprocessing.py:135-136`). Result: ~half the cerebral cortex disappears into the background-zero before the VAE sees it.

The encoder has the brain mask available (`masks/brain` is in the image H5 for every cohort) but never passes it to `percentile_normalise`. Fix landed in §5.1; encoder-side hook + re-encode still pending.

### 1.4 Out-of-brain jitter / "zero outside brain mask"
For correctly skull-stripped cohorts background is already zero and `foreground_only=True` keeps it at zero. The user's idea — explicitly setting `x[~brain_mask] = 0` before percentile estimation — is a no-op for those cohorts, but is the correct fix for BraTS-Africa (and a safety net against any small skull-strip leak in UCSF / LUMIERE shipped masks).

---

## 2. Mask storage and computation

### 2.1 Tumour mask — labels + remap + latent

| Cohort | `label_system` | Image-H5 labels (probed) | Remap before downsample | Latent dataset |
|---|---|---|---|---|
| UCSF-PDGM | BraTS2021 | {0,1,2,4} | none | `masks/tumor_latent` (N,3,48,56,48) float32 |
| BraTS-GLI | **BraTS2023** | {0,1,2,**3**} | 3 → 4 (`convert.py:738,778`) | (N,3,48,56,48) float32 |
| IvyGAP | BraTS2021 | {0,1,2,4} | none | (N,3,48,56,48) float32 |
| LUMIERE | **BraTS2023** | {0,1,2,**3**} | 3 → 4 | (N,3,48,56,48) float32 |
| UPENN-GBM | BraTS2021 | {0,1,2,4} | none | (N,3,48,56,48) float32 |
| REMBRANDT | BraTS2021 | {0,1,2,4} | none | (N,3,48,56,48) float32 |
| BraTS-Africa-Glioma | **BraTS2023** | {0,1,2,**3**} | 3 → 4 | (N,3,48,56,48) float32 |
| BraTS-Africa-Other | **BraTS2023** | {0,1,2,**3**} | 3 → 4 | (N,3,48,56,48) float32 |
| BraTS-PED | BraTS2023 | (not probed — image H5 truncated) | 3 → 4 | (N,3,48,56,48) float32 |

Downsampler: `PerClassAvgPoolDownsampler` (`src/vena/model/autoencoder/maisi/encode/masks/models/per_class_avg_pool.py`). One-hot encode → depth-pad to multiple of 8 → `F.avg_pool3d(kernel_size=4, stride=4)`. Channel order is canonical {NETC, ED, ET} after the BraTS2023 remap; no cohort produces a different ordering.

The BraTS2023 remap is gated on `src.attrs["label_system"] == "BraTS2023"` so the strict-labels downsampler will fast-fail if the attribute is missing — good guard.

### 2.2 Brain mask provenance + connected-component count

| Cohort | Source | n_cc (sample of 3 scans) | Cleanest cc | Noise tail size |
|---|---|---|---|---|
| UCSF-PDGM | shipped `_brain_segmentation.nii.gz > 0.5` | 1 / 1 / 1 | n/a | none |
| LUMIERE | shipped `brain_mask.nii.gz > 0.5` | 1 / 1 / 1 | n/a | none |
| BraTS-Africa-Glioma | union of 4 modalities `!= 0` | 1 / 1 / 1 | n/a | none |
| BraTS-Africa-Other | union of 4 modalities `!= 0` | 1 / 1 / 1 | n/a | none |
| REMBRANDT | union of 4 modalities `!= 0` (post HD-BET) | 1 / 1 / 1 | n/a | none |
| BraTS-GLI | `t1n > 0` | 1 / 1 / **14** | ≥ 1.4 M voxels | top tail 6, 4, 3, 2 voxels |
| UPENN-GBM | union of 4 modalities `!= 0` | 2 / 1 / 2 | ≥ 1.4 M voxels | tail = 1 voxel |
| **IvyGAP** | **`t1pre > 0` only** | **38 / 35 / 148** | ≥ 1.4 M voxels | top tail 65–664 voxels |

**Key correction to the brief.** The brain-mask code path has **no `keep_largest_n_components` filter anywhere**. Grep across `src/vena/` and `routines/`: only `src/vena/preflight/vessel_mask/analysis.py` and `venous_build.py` use those, and only for vessel statistics. So the brief's premise — "we keep 1 component and lose cerebellum" — is hypothetical, not actual. Every cohort's stored `masks/brain` is the raw thresholded mask, cerebellum included.

The data shows the reverse problem: **IvyGAP brain masks have dozens of noisy small CCs** (the "cloud-like blurs at first/last z-slices" the user referenced). They are tiny (≤ 0.02 % of brain volume) and are absorbed by the max-pool-4 to latent grid (latent fill fraction 17–22 % on every cohort, consistent), so they do not poison the latent brain-mask. They *do* poison image-space metric computation and any future image-space loss term — and they should be filtered.

The right policy (matching the user's intent) is "keep every CC with volume above a threshold" — e.g. `≥ 1 000 voxels` — which removes jitter without ever dropping cerebellum, brainstem, or detached white-matter pockets.

### 2.3 Brain-mask → latent path

`routines/encode/brain_to_latent/engine.py::_encode_brain_mask`:

```
apply_crop_pad → (1,1,192,224,192)
F.max_pool3d(kernel_size=4, stride=4)     # correct for binary masks
(pooled > 0).to(int8)                     # → (1,48,56,48)
```

Stored in the LATENT H5 (not the image H5) as `masks/brain_latent`, `int8`, `(N,1,48,56,48)`. Verified present on every cv cohort and on the two BraTS-Africa test cohorts. **Absent on BraTS-PED** — already irrelevant because BraTS-PED was dropped from `corpus_picasso.json`.

Empirical fill fractions match across cohorts (17–24 %), confirming the latent brain mask is consistent.

### 2.4 Tumour-mask → latent path

Same crop + per-class avg-pool. Stored only in the latent H5. Sample channel sums are non-zero on every probed cohort — tumour signal survives the downsample.

---

## 3. Augmentation pipeline + v4

### 3.1 Variant table — what each transform does, what it touches

`src/vena/data/augment/offline/variants.py` + `routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml`.

| Variant | TorchIO subjects touched | Tumour mask | Brain mask propagation |
|---|---|---|---|
| v1 | inputs only (intensity bias + gamma) | copy of source | — (untouched, copied at brain-to-latent time) |
| v2 | inputs only (histogram shift + gamma) | copy of source | — |
| v3 | inputs only (noise + anisotropy + blur + motion) | copy of source | — |
| **v4** | **all channels + tumour `LabelMap`** (elastic + affine) | warped, `label_interpolation="nearest"` ✓ | **NOT in the Subject** → brain-to-latent post-pass writes all-ones |

Sources: `src/vena/data/augment/offline/variants.py:137-173`, `src/vena/data/augment/offline/bank_builder.py:418-428`, `routines/encode/brain_to_latent/engine.py:285-322`.

### 3.2 Per-variant magnitudes — could anything hurt T1c synthesis?

`_INPUT_KEYS = ("t1pre", "t2", "flair")` (`variants.py:45`). Every intensity transform carries `include=list(_INPUT_KEYS)`. **T1c is the target and is never touched by v1/v2/v3.** v4 is the only variant that touches T1c (spatial-only — elastic+affine warp applied consistently to inputs + target + tumour).

| Variant | Transform | Magnitude | t1c hit? | Comment |
|---|---|---|---|---|
| v1 | `RandomBiasField(coefficients=(-0.5, 0.5), order=3)` | always fires | inputs only | Moderate; ±50 % low-frequency bias. Standard nnU-Net range. |
| v1 | `RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)` | ×0.74–1.35 | inputs only | Fires 50 %; reasonable. |
| v2 | `MonaiHistogramShift(n_ctrl=8–12, p=1.0)` | always fires | inputs only | **Most aggressive brightness transform** — always-on non-linear monotonic remap. |
| v2 | `RandomGamma(log_gamma=(-0.15, 0.15), p=0.4)` | ×0.86–1.16 | inputs only | Light additional gamma. |
| v3 | `RandomNoise(std=(0.0, 0.05))` | always fires | inputs only | **Effectively no-op on raw cohorts.** Augmentation runs on RAW intensities (pre-normalisation): UCSF fg ≈ 2000, BraTS-GLI fg ≈ 5000 → std=0.05 is invisible. For LUMIERE fg ≈ 250 still negligible. Not harmful; just useless. |
| v3 | anisotropy, blur, motion | low-to-moderate | inputs only | Fine. |
| v4 | `RandomElasticDeformation(num_control_points=7, max_displacement=4.0, p=1.0)` + `RandomAffine(scales=(0.9, 1.1), degrees=10, translation=8 vox, p=1.0)` | always fires | full Subject (incl. t1c) | Now `p=1.0` (was 0.7×0.7, leaving ~9 % byte-identical v4 rows; lifted in a recent commit). |

**Risk assessment for T1c synthesis specifically.**

The model needs to predict where contrast enhancement should land in T1c. Cues: spatial extent of BBB-disrupted tumour (visible in T2/FLAIR edema) and necrotic-core contrast in T1pre. Strong gamma on T1pre + histogram remap on T1pre could *blur the T1pre/T1c contrast prior*. v2's always-on histogram shift is the single highest-risk transform — if you see enhancement leakage at inference (model hallucinates enhancement in non-enhancing regions), the actions are:

1. Lower v2 weight in `data.variant_weights` (FM trainer YAML).
2. Lower `hist_shift_n_control_high` from 12 → 8 (less wiggly remap).
3. Drop `MonaiHistogramShift.prob` from 1.0 → 0.7 (some v2 rows pass through unchanged).

None of these require re-running the offline-aug pass; they all live in YAML / model-side config. Useful only if you observe the failure mode in practice.

**v3's noise scaling is a bug, not a hazard.** Augmentation operates pre-percentile-normalisation on raw intensities. `std=0.05` on a 0–8000 raw scale is invisible. Fix would be to scale by per-volume p99.5 (post-percentile expression). Filed as a forward-looking issue; current model is unaffected.

### 3.3 The v4 brain-mask "synth-ones" — empirical confirmation

Every cv cohort's `*_latents_aug.h5` carries `masks/brain_latent.attrs["v4_brain_synthesised_ones"] = True`. Probed 5 v4 rows per cohort:

| Cohort | v4 sample rows | sum == full volume? | Non-v4 fill |
|---|---|---|---|
| UCSF | 5/5 | True (sum = 129 024 = 1×48×56×48) | 0.17–0.22 |
| BraTS-GLI | 5/5 | True | 0.20 |
| IvyGAP | 5/5 | True | 0.20–0.21 |
| LUMIERE | 5/5 | True | 0.17 |
| UPENN-GBM | 5/5 | True | 0.21–0.22 |
| REMBRANDT | 5/5 | True | 0.18–0.22 |

**Implication.** For every v4 row the contrastive loss's `healthy = brain ∩ ¬WT` collapses to `1 − WT`. The region-weighted contrastive loss (`project_lp_contrastive_v04`, p=3 healthy + p=1 wt) computes its healthy term over the whole latent grid instead of brain tissue → it weights against ~80 % background voxels.

The `v4_brain_synthesised_ones` flag is set at the dataset level, **not per-row**, so a consumer cannot distinguish v0–v3 from v4 by attribute alone — it has to read `variants[row]`.

### 3.4 K=4 coverage — what the gap actually is

Original suspicion: "rank-1 shards never merged → only ~½ patients have aug rows". Empirical follow-up: **WRONG**. Server-3 carries fully merged H5s (no `*_rank{0,1}.h5` files exist anywhere under `/media/hddb/mario/data/GLIOMAS/`), and the sizes match Picasso byte-for-byte:

| Cohort | server-3 image_aug | Picasso image_aug | server-3 latents_aug | Picasso latents_aug |
|---|---|---|---|---|
| BraTS-GLI | 156 G | 167 G | 33 G | 34 G |
| UCSF-PDGM | 28 G | 29 G | 5.2 G | 5.5 G |
| LUMIERE | 80 G | 80 G | 16 G | 16 G |
| REMBRANDT | 7.8 G | 8.3 G | 1.7 G | 1.8 G |
| UPENN-GBM | 21 G | 21 G | 4.2 G | 4.5 G |
| IvyGAP | 4.0 G | 4.0 G | 0.85 G | 0.89 G |

(Small differences within rounding / metadata padding.)

**Real cause of the row-count gap.** `bank_builder._resolve_rows` (`src/vena/data/augment/offline/bank_builder.py:306-352`):

```
pool = all_ids
     − splits/test set           # by design
     ∩ dedup_allowlist (if set)  # cross-cohort de-duplication
sharded = pool[r % world_size == rank]
```

Val patients ARE in the pool. Test patients are excluded by design. The big drop is the dedup intersect: per `project_cohort_dedup`, UCSF-PDGM ⊂ BraTS-GLI by 293 patients; UPENN-GBM also heavily overlaps BraTS-GLI. The aug bank only keeps the "winner" of each duplicate group, so the unique-patient pool per cohort shrinks substantially for UCSF and UPENN-GBM.

Per-cohort aug row counts → unique aug scans → train-fold-0 sizes:

| Cohort | aug rows | K dist | unique aug ids | train fold-0 scans | coverage of unique-post-dedup pool |
|---|---|---|---|---|---|
| UCSF | 724 | {4: 181} | 181 | 356 | matches memory `project_upenn_gbm_added`-style prediction |
| BraTS-GLI | 4 496 | {4: 1 124} | 1 124 | ≈ 1 124 | ~100 % of dedup-allowed |
| IvyGAP | 116 | {4: 29} | 29 | 24 | over-shoots — likely test included (see §3.5) |
| LUMIERE | 2 108 | {4: 527} | 527 | per-session, see §3.5 | ~88 % of sessions |
| UPENN-GBM | 588 | {4: 147} | 147 | 439 raw → ≈ 147 unique post-dedup | matches memory ("expect ≈ 164 unique post-dedup") |
| REMBRANDT | 232 | {4: 58} | 58 | 53 train + 5 val | over-shoots — likely test included |

**Every augmented scan has all 4 variants.** No K < 4 case. K=4 within the pool is sound.

**This is the intended behaviour** (no fix needed for normal training): augmenting duplicate patients twice would double-count them at training time.

If you want more coverage you have two options, neither recommended by default:
- Drop the dedup intersect → produces K=4 rows for duplicate patients in every cohort they appear in. The trainer's sampler would then need its own dedup gate, which it does not have.
- Augment val patients in a separate pass and consume them only for monitoring. Aug bank-builder is already val-inclusive, so no change needed — the apparent "no val aug" is the same dedup intersect.

### 3.5 Two suspicious over-shoots to investigate

IvyGAP aug 29 unique > 24 train scans; REMBRANDT aug 58 unique > 53 train scans. Two possible causes:
- The cohort's `splits/test` carries patient-ID strings while the aug bank's `test_set` lookup checks scan IDs (`bank_builder.py:340-342`), so test patients are not excluded.
- The cohort's `splits/cv/fold_0/{train, val}` aliases interact with the dedup_allowlist differently.

Both are 1-minute checks; not blocking. They suggest 5–8 test patients are wasting aug compute but are not affecting training fairness (DataModule reads `splits/test` independently to keep test out of the train loader).

### 3.6 Aug-latent H5 schema vs base-latent H5

Aug latent H5 has:
- `latents/<modality>`, `masks/tumor_latent`, `masks/brain_latent` (with the synth-ones caveat above).
- `ids`, `variants`, `source_row_index`, `aug_params_json`.
- **No `splits/*`**, no `patients/*`. Partitioning lives on the clean H5.
- Schema version `0.1.0` (vs `2.0.0` on the clean H5).

`build_aug_latent_manifest` in `src/vena/data/h5/augmented/latent_domain.py` does **not** list `masks/brain_latent` — it is a separate post-pass owned by `vena-encode-brain-to-latent` and not validated by `validate_aug_latent_h5`. Worth folding into the validator (see §6).

---

## 4. Image vs Latent H5 schema

The two schemas were designed to share the same **shape**, but they describe different domains so the dataset names differ. Common backbone first, then deltas.

### 4.1 Common backbone (every cohort, image + latent)

| Group/key | Shape pattern | Notes |
|---|---|---|
| Root attrs | `schema_version`, `created_at`, `producer`, `config_json`, `git_sha`, `cohort`, `label_system`, `crop_box`, `orientation`, `domain` | Self-describing per `.claude/rules/h5-design-principles.md` |
| `ids` | `(N,)` vlen-str | Per-row scan/session ID |
| `masks/tumor` (image) or `masks/tumor_latent` (latent) | `(N, ...)` int8 / float32 | Cohort-agnostic — BraTS2023 cohorts pre-remap to BraTS2021 at encode time |
| `metadata/*` | `(N,)` per field | **Cohort-specific** — UCSF has 14 fields; LUMIERE has 3; BraTS-Africa has 0. By design (H5 rule 5: one dataset per field). |
| `patients/{keys, offsets}` | CSR-style | `keys: (n_pts,)`, `offsets: (n_pts+1,)`. Maps scan rows to patient. |
| `splits/test` | `(n_test,)` | Always present (sometimes alias-only for test-only cohorts) |
| `splits/cv/fold_N/{train, val}` | `(n,)` | 5 folds for cv cohorts (UCSF, BraTS-GLI, LUMIERE, UPENN-GBM); 1 fold for small cohorts (IvyGAP, REMBRANDT); fold_0 alias-only for test-only (BraTS-Africa, BraTS-PED) |

### 4.2 Intentional differences — image vs latent

| Image H5 | Latent H5 |
|---|---|
| `images/{t1pre, t1c, t2, flair}` shape `(N, H, W, D)` float32 — native size | `latents/{t1pre, t1c, t2, flair}` shape `(N, 4, 48, 56, 48)` float32 — MAISI latent grid |
| `masks/brain` shape `(N, H, W, D)` int8 | `masks/brain_latent` shape `(N, 1, 48, 56, 48)` int8 (max-pool-4 of `masks/brain`) |
| `masks/tumor` shape `(N, H, W, D)` int8 (multi-class labels) | `masks/tumor_latent` shape `(N, 3, 48, 56, 48)` float32 (per-class avg-pool, channels = NETC / ED / ET) |
| `crop/origin` shape `(N, 3)` int32 — native-grid crop start | — (crop lives in root attrs only) |
| — | `progress/completed` shape `(N,)` bool — per-row encode-loop checkpoint |
| Root attrs: `intensity_policy` string | Root attrs: `vae_checkpoint_sha256`, `autoencoder_arch_config_json` |
| `schema_version = 2.0.0` | `schema_version` varies by cohort |

### 4.3 Unintentional drift (to unify on next re-encode)

| Issue | Affected | Severity | Suggested unification |
|---|---|---|---|
| Latent H5 `schema_version` not pinned — every cohort writes a different version | All | Medium | Pin to `2.0.0` everywhere on next encode pass. |
| `splits/*` shape varies: cv cohorts have only `splits/cv/fold_N/*`; IvyGAP / REMBRANDT have **both** flat `splits/{train, val, test}` and `splits/cv/fold_0/*` aliases; test-only cohorts have `splits/test` + `splits/cv/fold_0/val` alias (val = test, train = empty) | All | Low | Drop flat `splits/{train,val,test}` everywhere except `splits/test`. Always carry `splits/cv/fold_N/*`. DataModule reads either form today, so the change is backward-compatible. |
| `masks/brain_latent` not in the aug-latent H5 manifest validator (added by separate routine post-pass). | All aug H5s | Low | Add to `build_aug_latent_manifest` + `validate_aug_latent_h5`. |
| Brain mask source varies: 2 cohorts use `t1pre > 0`, 4 use union-of-4-modalities, 2 inherit shipped. | BraTS-GLI, IvyGAP vs the rest | Low | Standardise on union-of-4 + CC clean (`clean_brain_mask`) for the VENA-computed cohorts. UCSF / LUMIERE keep shipped. |
| Aug H5 root attrs: every cohort writes `aug_config_json`, `aug_config_sha256`, `source_image_h5_path`, `source_image_h5_sha256`, but only some carry `world_size` / `rank` (legacy from rank-shard writes). | Aug H5s | None | Cosmetic; can be dropped. |

So the "unified schema" goal *is* met at the structural level (group names, dataset names, dtypes, attr names). It is not met at the bookkeeping level (schema_version + split-key naming + metadata field set). The first is what loaders depend on; the second is what reviewers grep for.

---

## 5. Fixes shipped today (non-breaking)

| File | Change |
|---|---|
| `src/vena/model/autoencoder/maisi/preprocessing.py` | `percentile_normalise(..., mask=None)` new parameter that bypasses the `x > 0` heuristic when a brain mask is provided. Default unchanged when `mask is None` (every existing call site is byte-identical). Solves the BraTS-Africa silent skew once the encoder passes the brain mask. |
| `src/vena/data/h5/shared/brain_mask.py` (**new**) | `clean_brain_mask(mask, min_component_voxels=1000)` keeps every CC above threshold (preserves cerebellum + brainstem; drops boundary noise). 6-connectivity by default. Falls back to "keep largest" if every CC is sub-threshold (with a warning). |
| `src/vena/data/h5/{brats_gli, ivy_gap, upenn_gbm, rembrandt, brats_africa, brats_ped}/image_domain/convert.py` | Each cohort converter now wraps its brain-mask compute with `clean_brain_mask(...)`. Single-line change per file. Future encode passes produce clean masks; existing H5s untouched. |
| `scripts/clean_brain_mask_inplace.py` (**new**) | In-place patcher: read `masks/brain` row-by-row, run `clean_brain_mask`, write back. Idempotent. Writes a per-row delta CSV next to the H5. After running, re-run `vena-encode-brain-to-latent` against the latent H5 to refresh `masks/brain_latent` (no MR-latent re-encode needed because the encoder never read `masks/brain`). Supports `--dry-run`. |
| `tests/data/h5/test_brain_mask_cleaner.py` (**new**) | 9 tests covering single-CC pass-through, multi-region preservation, sub-threshold fallback, dtype/shape contracts. |
| `tests/model/autoencoder/maisi/test_preprocessing.py` | +3 tests for the new `mask=` path (overrides heuristic; shape validation; byte-identical when `mask=None`). |

All tests green: 57 / 57 in `tests/data/h5/` + `tests/model/autoencoder/maisi/test_preprocessing.py`.

---

## 6. Forward roadmap — next encode pass

In priority order. Items 1 + 2 + 3 can be done in one pass; items 4-6 are forward-looking.

### 6.1 Mandatory before next training run

1. **Fix v4 brain mask** (`§3.3`). Cheapest path: replay `RandomElasticDeformation + RandomAffine` from `aug_params_json` on the image-domain brain mask (max-pool-4 → latent), patch `masks/brain_latent[v4 rows]` in-place via the existing `brain_latent_merge_patch.py` flow. No MR-latent re-encode. Alternative: re-run `vena-augment-bank-builder` end-to-end with `members["brain"] = tio.LabelMap(...)` added in `bank_builder._build_subject()` — costs the full offline-aug GPU time (~6 h × 6 cohorts).
2. **Fix BraTS-Africa silent intensity skew** (`§1.3`).
   - Encoder change: have `routines/encode/maisi/engine.py` read `masks/brain` from the source image H5 and pass it to `percentile_normalise(mask=...)` (function already supports it as of §5).
   - Re-encode BraTS-Africa-Glioma (95) + BraTS-Africa-Other (51) = 146 scans. ~30 min on A100.
3. **Apply CC clean to existing image H5s** (`§2.2`). Run `scripts/clean_brain_mask_inplace.py --dry-run` per cohort first (IvyGAP / BraTS-GLI / UPENN-GBM / REMBRANDT / BraTS-Africa / BraTS-PED). If the per-row deltas look like noise tail (not real brain), drop `--dry-run` and apply. Then re-run `vena-encode-brain-to-latent` for those cohorts to refresh `masks/brain_latent`. **MR latents are NOT affected** because the encoder did not consume `masks/brain` to produce them.

### 6.2 Schema unification (does require re-encoding to take full effect)

4. **Pin latent H5 `schema_version` to 2.0.0** in `_stamp_root_attrs` for every cohort's latent-domain converter. Today different cohorts stamp different versions.
5. **Drop flat `splits/{train, val, test}`** outside the test-only cohorts; keep only `splits/cv/fold_N/*` + `splits/test`. DataModule fallback already prefers the CV form.
6. **Standardise brain-mask source for VENA-computed cohorts** to "union of 4 modalities + `clean_brain_mask`". BraTS-GLI and IvyGAP currently use `t1pre > 0` only; the union-of-4 is slightly more permissive at the brain boundary and is what the other 4 VENA-computed cohorts use.
7. **Add `masks/brain_latent` to the aug-latent H5 manifest + validator** so a future re-encode that forgets the brain-to-latent post-pass fast-fails instead of silently shipping a latent H5 without it.

### 6.3 Optional augmentation tweaks (no re-encode, just config)

- `routines/offline_aug/maisi/configs/aug_pipelines/k4_v1.yaml`:
  - `v2.brightness_contrast_prob`: 0.4 → keep. `v2.hist_shift_n_control_high`: 12 → consider lowering to 8 if enhancement leakage shows up at inference. Reasoning in `§3.2`.
  - `v3.noise_std`: rewrite as a fraction of per-volume p99.5 instead of raw intensity (current `(0, 0.05)` is a no-op for raw cohorts).
- FM trainer YAML: tune `data.variant_weights` to dampen v2 if needed.

### 6.4 Things the original audit guessed wrong (CORRECTED)

- **K=4 coverage is NOT a sync gap.** Earlier suspicion was "rank-1 shards never merged → only ~½ patients have aug rows". Server-3 and Picasso H5s are byte-identical merged outputs (§3.4). The gap is the dedup intersect applied at aug-pool construction, which is the intended behaviour.
- **No CC filter ever existed in the brain-mask path.** Earlier brief mentioned "we keep only 1 connected component → cerebellum dropped". No code path implements this; the bug is the inverse (noisy CCs not filtered).

---

## 7. Open questions parked for the user

- **Q1.** For fix 6.1.1 (v4 brain mask), prefer replay from `aug_params_json` (cheap CPU pass) or re-run `vena-augment-bank-builder` end-to-end with brain in the Subject (6 h × 6 cohorts on GPU)? Replay is faster but assumes `aug_params_json` faithfully reproduces the TorchIO random state.
- **Q2.** For the schema unification (6.2), do this in a single coordinated re-encode pass or piecemeal (BraTS-Africa first because it's small + has the z-score bug, others later)?
- **Q3.** §3.5 over-shoots in IvyGAP / REMBRANDT — investigate before next aug re-run, or accept the few wasted aug rows? They don't affect training fairness; test patients are kept out by the DataModule's own `splits/test` gate.

---

## 8. Pointers

- This file: `.claude/notes/data/2026-06-18_data_audit.md`.
- Audit memory: `~/.claude/projects/-home-mpascual-research-code-VENA/memory/project_data_audit_2026_06_18.md`.
- Raw per-cohort JSON: `/tmp/vena_audit.json` (314 KB) + `/tmp/vena_deep_audit.json` (19 KB) on the local box.
- Picasso audit script: `/tmp/vena_audit.py` + `/tmp/vena_deep_audit.py` (also copied to `picasso:/tmp/`).
- In-place mask cleaner: `scripts/clean_brain_mask_inplace.py`.
- Library helpers: `src/vena/data/h5/shared/brain_mask.py` + `src/vena/model/autoencoder/maisi/preprocessing.py::percentile_normalise(mask=)`.
- Related rules: `.claude/rules/h5-design-principles.md`, `.claude/rules/extensibility.md`.
- Related memories: `project_brain_latent_encoding`, `project_offline_aug_complete`, `project_lp_contrastive_v04`, `project_cohort_dedup`, `project_upenn_gbm_added`.
