# 2026-06-19 — VENA pre-training data audit (v2, post-fix)

Follow-up to `.claude/notes/data/2026-06-18_data_audit.md`. Records the
state of the Picasso production dataset after the fix-up plan at
`/home/mpascual/.claude/plans/context-we-are-planning-sorted-valley.md`
executed. Same scope, same probes — only the per-row numbers change.

## 0. TL;DR — what was fixed, by severity

| Severity in v1 | Finding | v2 status |
|---|---|---|
| 🔴 Critical | v4 brain mask is all-ones on every aug H5 | **CLOSED** — TorchIO seed-replay produced byte-identical brain masks per `_variant_seed`; merged into every cv cohort's aug-latent H5; sample v4 sums now in 17–24 % fill range (matching v1/v2/v3). Verified by `scripts/audit_post_fix.py`. |
| 🔴 Critical | BraTS-Africa percentile_normalise discards ~½ intra-brain tissue | **CLOSED** — `percentile_normalise(mask=)` wired through `LatentH5Config.percentile_use_brain_mask` (default True) into the MAISI encoder; BraTS-Africa-Glioma + Other re-encoded on server-3 RTX 4090; new latent distribution matches UCSF (mean ≈ −0.10, std ≈ 0.94–0.99). |
| 🟡 Medium | IvyGAP carries 35–148 noisy CCs per brain mask | **CLOSED** — `scripts/clean_brain_mask_inplace.py` (project-default 1000-voxel CC floor) ran on every cohort image H5; deltas pushed Picasso via the `brain_mask_extract_patch` / `brain_mask_merge_patch` pair. IvyGAP 33/34 rows changed, -3 903 voxels dropped. |
| 🟡 Medium | BraTS-GLI + IvyGAP use t1pre>0 for brain; others use union-of-4 | **CLOSED** — `scripts/harmonize_brain_source_inplace.py` recomputed `masks/brain` from union-of-4 + clean. BraTS-GLI: 638/1251 rows changed (+755 518 voxels recovered at the boundary). IvyGAP: 33/34 rows changed. |
| 🟡 Medium | Latent `schema_version` drift across cohorts; aug-H5 splits aliases | **CLOSED** — `scripts/pin_latent_schema.py` bumped 6 aug-latent H5s `0.1.0 → 0.2.0`; `scripts/normalize_splits_inplace.py` dropped 10 legacy split nodes across 24 cohort H5s. |
| 🟢 OK | masks/tumor_latent intact; per-class avg-pool channels NETC/ED/ET | unchanged |
| 🟢 OK | masks/brain_latent present on every cohort | now stamped with `produced_by_brain_to_latent=True` for the conditional validator |
| 🟢 OK | Latent intensity distribution comparable across cohorts | confirmed post-fix; BraTS-Africa now matches the rest |
| ✓ new | BraTS-PED back in `corpus_picasso.json` | added as `test_only`; 260 patients, image H5 re-transferred (5.1 GB intact), brain_latent populated |

Total v2 changes:

| Phase | Description | Cohorts affected |
|---|---|---|
| 0 | Code + tests | n/a (15 new tests, 198/198 green) |
| 1 | Image-H5 in-place patches (CC clean + brain-source unify) | 5 (UCSF, BraTS-GLI, IvyGAP, UPENN, BraTS-Africa-Glioma) |
| 2 | Re-run `vena-encode-brain-to-latent` | 5 (same 5 cohorts) |
| 3 | BraTS-Africa re-encode w/ mask= | 2 (Africa-Glioma, Africa-Other) |
| 4 | BraTS-PED re-included | 1 (BraTS-PED) |
| 5 | v4 brain-mask fix via TorchIO seed-replay | 6 cv cohorts (UCSF, BraTS-GLI, IvyGAP, LUMIERE, UPENN, REMBRANDT) |
| 6 | Schema unification in-place | 24 H5s |
| 7 | Validation + visual proof | — |

## 1. Cross-cohort normalisation (re-verified)

§1 of v1 unchanged structurally. Only BraTS-Africa changes:

| Cohort | t1c min (image H5) | latent t1c mean | latent t1c std | mask= used at encode? |
|---|---|---|---|---|
| UCSF-PDGM | 0.0 | -0.103 | 0.993 | yes (no-op on raw cohort) |
| BraTS-GLI | 0.0 | (audit §0 baseline) | ~0.99 | yes (no-op) |
| **BraTS-Africa-Glioma** | **-473.0** | **-0.197** | **0.945** | **yes — fix in effect** |
| **BraTS-Africa-Other** | **-96.0** | **-0.140** | **0.969** | **yes — fix in effect** |
| BraTS-PED | 0.0 | (no re-encode this round) | — | n/a |
| UPENN-GBM / REMBRANDT / IvyGAP / LUMIERE | 0.0 | comparable | comparable | yes (no-op) |

BraTS-Africa latents are now distributionally comparable with the others.
The old z-score-clamping bug is closed at encode time. (The image H5 still
stores the original z-scored intensities; only the encoder consumes them
via `mask=masks/brain`.)

## 2. Mask storage and computation

### 2.1 Tumour mask
Unchanged — same `BraTS2023 → BraTS2021` (3 → 4) remap, same per-class
`avg_pool3d(4)` downsampler. No re-encode needed.

### 2.2 Brain mask provenance — POST FIX

| Cohort | Source (now) | n_cc | brain_cc_cleaned attr | brain_source_unified attr |
|---|---|---|---|---|
| UCSF-PDGM | shipped + clean | 1 | True | n/a |
| LUMIERE | shipped + clean | 1 | True | n/a |
| BraTS-Africa-Glioma | union-of-4 + clean | 1 | True | n/a |
| BraTS-Africa-Other | union-of-4 + clean | 1 | True | n/a |
| BraTS-PED | union-of-4 + clean | 1 | True | n/a |
| REMBRANDT | union-of-4 + clean | 1 | True | n/a |
| UPENN-GBM | union-of-4 + clean | 1 | True | n/a |
| **BraTS-GLI** | **union-of-4 + clean** (was t1pre>0) | 1 | True | **True** |
| **IvyGAP** | **union-of-4 + clean** (was t1pre>0) | 1 | True | **True** |

### 2.3 Brain-mask → latent path
Every cohort's `masks/brain_latent` re-derived on Picasso post-fix.
`produced_by_brain_to_latent=True` set on every latent + every cv-cohort
aug-latent H5 (the validator now requires `masks/brain_latent` when the
flag is set).

## 3. Augmentation pipeline + v4

### 3.1 Variant table — UNCHANGED
v1–v3 still intensity-only on `(t1pre, t2, flair)`; v4 still
elastic+affine on all channels. T1c untouched by v1/v2/v3.

### 3.2 v4 brain-mask — POST FIX

Empirical per-cohort v4 brain-latent voxel sums after Phase 5
(`scripts/patch_v4_brain_latent.py` → merge):

| Cohort | n_v4 | min | mean | max | all_equal_synth_ones |
|---|---|---|---|---|---|
| UCSF-PDGM | 181 | 19 802 | 26 947 | 34 873 | **False** |
| BraTS-GLI | 1 124 | 18 879 | 26 254 | 35 202 | **False** |
| UPENN-GBM | 147 | 18 721 | 25 986 | 37 196 | **False** |
| IvyGAP | 29 | 14 063 | 24 735 | 34 492 | **False** |
| LUMIERE | 527 | 17 358 | 26 272 | 36 438 | **False** |
| REMBRANDT | 58 | 18 827 | 26 104 | 34 075 | **False** |

Total 2 066 v4 rows patched across 6 cv cohorts. Pre-fix every row was
129 024 (full-volume ones). Post-fix mean fill ~20 % of latent volume,
exactly matching the v1/v2/v3 brain-mask fill range — confirming the
TorchIO seed-replay produced byte-identical warped brain masks to what the
bank builder would have produced with the brain LabelMap from day one
(`tests/data/augment/offline/test_v4_seed_replay.py`).

v4 brain masks were reconstructed via `scripts/patch_v4_brain_latent.py`,
which uses the per-row stored seed (`bank_builder._variant_seed`) to
replay the elastic+affine transform in TorchIO. Test
`tests/data/augment/offline/test_v4_seed_replay.py` proves byte-identity
against what the bank builder would have produced if the brain LabelMap
had been part of the Subject from day one.

## 4. Image vs Latent H5 schema (unified)

### 4.1 Schema versions — pinned

| Class | Constant | Value (post-fix) |
|---|---|---|
| Image (any cohort) | per-cohort `*_IMAGE_SCHEMA_VERSION` | 2.0.0 |
| Latent (any cohort) | `LATENT_SCHEMA_VERSION` | 2.0.0 |
| Aug-image | `AUG_IMAGE_SCHEMA_VERSION` | **0.2.0** (bumped; added `masks/brain`) |
| Aug-latent | `AUG_LATENT_SCHEMA_VERSION` | **0.2.0** (bumped; conditional `masks/brain_latent` validator) |

### 4.2 Splits layout — canonical

- `role=cv` cohorts carry only `splits/test` + `splits/cv/fold_N/*`. Flat
  `splits/{train,val}` aliases dropped.
- `role=test_only` cohorts carry only `splits/test`. `splits/cv/*` aliases
  dropped.

10 of 24 cohort H5s had legacy nodes; all dropped in Phase 6.

### 4.3 `produced_by_brain_to_latent` root attr — new

Stamped True on every cohort's latent H5 (and every cv-cohort aug-latent
H5) by the brain-to-latent engine (or `brain_latent_merge_patch.py` for the
patch flow). Drives the new conditional validator in
`validate_aug_latent_h5`.

## 4.4 Downstream loader compatibility check

After the schema unification (Phase 6) dropped the `splits/cv/fold_0/{train,val}` aliases on test-only cohorts, one consumer needed adjustment:

| Consumer | Status | Note |
|---|---|---|
| `MultiCohortLatentDataModule` (`src/vena/model/fm/lightning/data.py`) | OK | iterates `cv_cohorts()` only when reading `splits/cv/fold_<fold>/*`; test-only cohorts only contribute via `splits/test`. |
| Competitor datasets (`src/vena/competitors/{pgan_cgan,dit_3d,lddpm_3d,lpix2pix_3d,resvit,syndiff,t1c_rflow}/dataset.py`) | OK | Every dataset defaults `role_filter="cv"` → skips test-only cohorts. Per-cohort split lookup tries `splits/cv/fold_{fold}/{phase}` first, then the legacy `splits/{phase}` flat fallback (cv cohorts always have the cv form, so the fallback is never needed). |
| `routines/fm/inference/engine.py` | OK | No `splits/cv` reads. |
| `routines/fm/exhaustive_val/engine.py::_cohort_val_patients` | **FIXED** | Was unconditionally reading `splits/cv/fold_<fold>/val` for every cohort (cv + test_only). After Phase 6, this would `KeyError` on test-only cohorts. Patched to fall back to `splits/test` when `splits/cv/fold_<fold>/val` is absent — this gives the OOD evaluation the full test set, which is exactly what the legacy alias provided. Confirmed: `cv_cohorts() + test_cohorts()` enumerates 9 cohorts post-fix, including BraTS-PED (260 patients available via splits/test). |
| `REMBRANDT_image.h5` lost splits/cv to Phase 6 | **FIXED** | REMBRANDT (N=63) had been shipped with ONLY flat `splits/{train,val,test}` (the converter explicitly notes "small cohort, no nested CV"). The latent converter added `splits/cv/fold_0/*` as it was being written, but the image converter never did. Phase 6 dropped the flat splits on cv-role cohorts assuming the cv form was always present — it wasn't for REMBRANDT_image. Restored by copying `splits/cv` from `REMBRANDT_latents.h5` (same patient sets — both came from the same `make_cohort_splits` seed). REMBRANDT_image now has `splits/cv/fold_0/train n=53`, `splits/cv/fold_0/val n=5`, `splits/test n=5`. Verified no other cv-cohort image H5 had the same hidden state. |
| `vena.preflight.latent_aug_equivariance` | OK | Already had a graceful fallback (`pool_patient_keys = ... if val_path in f else list(all_ids)`). |
| New BraTS-PED entry in corpus | OK | BraTS-PED has `role=test_only`; the FM DataModule's `test_cohorts()` path reads only `patients/{offsets,keys}` and `ids`, all present. |

## 5. New code shipped

| File | Purpose |
|---|---|
| `src/vena/model/autoencoder/maisi/encode/engine.py` | `MaisiEncoder.encode(mask=)` kwarg; plumbed through both crop-box and depth-pad paths. |
| `src/vena/data/h5/latent_domain/convert.py` | `LatentH5Config.percentile_use_brain_mask` (default True); `_encode_loop` reads `masks/brain` and passes through to the encoder. |
| `src/vena/data/augment/offline/bank_builder.py` | `_build_subject(brain_boxed=)` adds brain `LabelMap`; `_encode_rows` loads + writes; merge function copies brain. |
| `src/vena/data/h5/augmented/image_domain.py` | manifest declares `masks/brain`; schema bumped to 0.2.0. |
| `src/vena/data/h5/augmented/latent_domain.py` | conditional `masks/brain_latent` check on `produced_by_brain_to_latent=True`; schema bumped to 0.2.0. |
| `src/vena/data/h5/shared/splits.py` | `normalize_splits(h5_path, role)`. |
| `src/vena/data/h5/shared/brain_mask.py` | `recompute_union_of_four(...)` generator. |
| `routines/encode/brain_to_latent/engine.py` | reads warped brain from aug-image H5 when available; stamps `produced_by_brain_to_latent=True`. |
| `routines/encode/maisi/engine.py` | exposes `percentile_use_brain_mask`; round-trips into `decision.json`. |
| `scripts/clean_brain_mask_inplace.py` | already present (§5 of v1). |
| `scripts/harmonize_brain_source_inplace.py` | **new** — recomputes `masks/brain` from union-of-4 + CC clean. |
| `scripts/brain_mask_extract_patch.py` / `brain_mask_merge_patch.py` | **new** — image-domain mask patch flow Picasso-wards. |
| `scripts/patch_v4_brain_latent.py` | **new** — seed-replay v4 brain mask. |
| `scripts/pin_latent_schema.py` | **new** — bump latent schema in place. |
| `scripts/normalize_splits_inplace.py` | **new** — drop legacy split aliases per role. |
| `scripts/audit_post_fix.py` | **new** — re-run §0 audit table. |
| `scripts/figures_post_fix.py` | **new** — per-cohort visual-proof PNGs. |

## 6. Tests

15 new unit tests, all marker-stamped (`pytest.mark.unit`). 198/198 green
in the data + routines/encode + model/autoencoder subset
(`pytest -m "not slow and not gpu"`).

| Test | Subject |
|---|---|
| `tests/model/autoencoder/maisi/test_preprocessing.py` (+3) | `percentile_normalise(mask=)` |
| `tests/model/autoencoder/maisi/test_encoder_mask_kwarg.py` (4) | `MaisiEncoder.encode(mask=)` |
| `tests/data/augment/offline/test_bank_builder_brain_member.py` (4) | brain `LabelMap` in Subject |
| `tests/data/augment/offline/test_v4_seed_replay.py` (2) | seed-replay byte-identity |
| `tests/data/h5/augmented/test_brain_latent_validation.py` (4) | conditional validator |
| `tests/data/h5/test_splits_normalize.py` (5) | `normalize_splits` |
| `tests/data/h5/test_recompute_union_of_four.py` (4) | union-of-4 generator |

## 7. Compute resources used

| Phase | Host | Wall-clock |
|---|---|---|
| 0 — code/tests | local | ~1.5 h |
| 1 — image-H5 patches | server-3 CPU | ~10 min |
| 1 — push to Picasso | local /tmp hop | ~15 min |
| 2 — brain-to-latent (5 cohorts) | loginexa CPU | ~5 min |
| 3 — BraTS-Africa re-encode | server-3 RTX 4090 | ~25 min |
| 4 — BraTS-PED retransfer | local /tmp hop | ~5 min |
| 4 — BraTS-PED brain-to-latent | loginexa CPU | <1 min |
| 5 — v4 patch (4 cohorts on server-3) | server-3 CPU | ~30 min |
| 5 — v4 patch (2 cohorts on loginexa) | loginexa CPU | ~3 min |
| 5 — patch upload + merge | Picasso | <5 min |
| 6 — schema unification | loginexa CPU | <1 min |
| 7 — audit + figures | loginexa | ~2 min |
| **Total** | | **~3 h** |

No Picasso SLURM jobs queued. The A100 queue stays free for downstream
training.

## 8. Pointers

- Plan: `/home/mpascual/.claude/plans/context-we-are-planning-sorted-valley.md`.
- Visual proof PNGs: `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/_audit_post_fix_figures/` (rsync to `/tmp/vena_figures/` locally).
- Post-fix audit JSON: `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/_audit_post_fix.json`.
- v1 audit: `.claude/notes/data/2026-06-18_data_audit.md`.
- Memories: `project_brats_ped_truncated_picasso` (updated 2026-06-18 to "RESOLVED").
