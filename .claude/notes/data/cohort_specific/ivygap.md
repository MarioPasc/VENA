# IvyGAP Dataset — Reconnaissance Notes

**Date:** 2026-05-31  
**Scout:** Sonnet 4.6 subagent (reconnaissance only)

---

## Source path

```
/media/mpascual/MeningD2/GLIOMA/IVYGAP/PKG - IvyGAP-Radiomics-SRI/IvyGAP-Radiomics/
  Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations/
  ├── 1_Images_SRI/
  │   └── CoRegistered_SkullStripped/
  │       └── <patient>/
  │           └── <patient>_<date>/          # e.g. W1_1996.10.25/
  │               ├── <id>_flair_LPS_r_SS.nii.gz
  │               ├── <id>_t1_LPS_r_SS.nii.gz
  │               ├── <id>_t1gd_LPS_r_SS.nii.gz
  │               └── <id>_t2_LPS_r_SS.nii.gz
  ├── 2_Atlas_SRI/
  │   └── spgr_unstrip_lps.nii.gz            # SRI24 SPGR atlas (reference, non-skull-stripped)
  └── 3_Annotations_SRI/
      ├── CWRU/
      │   └── <patient>/
      │       └── <id>_CWRU_labels.nii.gz
      └── UPenn/
          └── <patient>/
              └── <id>_UPenn_labels.nii.gz
```

---

## Patient count and naming

- **34 patients** total in `1_Images_SRI/CoRegistered_SkullStripped/`
- **Naming:** `W<N>` where N is non-contiguous: W1, W2, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W18, W19, W20, W22, W26, W29, W30, W32, W33, W34, W35, W36, W38, W39, W40, W42, W43, W48, W50, W53, W54, W55
- No institution/cohort grouping in the image directory — all patients flat under `CoRegistered_SkullStripped/`
- Each patient has exactly one session subdirectory named `<patient_id>_<YYYY.MM.DD>` (scan date, ~1996–2000)
- All glioblastoma (IvyGAP = Ivy Glioblastoma Atlas Project, TGCA/Stanford)

---

## Per-patient files

Every patient has exactly 4 modality files:

| Token in filename | Modality | VENA role |
|---|---|---|
| `_t1_` | T1 pre-contrast | input `t1pre` |
| `_t1gd_` | T1 post-contrast (gadolinium) | **synthesis target** `t1c` |
| `_t2_` | T2 | input `t2` |
| `_flair_` | FLAIR | input `flair` |

Filename pattern: `<patient_id>_<date>_<modality>_LPS[_N4][_r|_r3]_SS.nii.gz`

Suffix variants observed:
- `_r_SS` — standard co-registration + skull-strip (20/34 patients for all 4 modalities)
- `_r3_SS` — 3rd-order (polynomial) registration + skull-strip (14/34 patients, at least one modality; patients: W18, W19, W26, W30, W32, W38, W39, W4, W42, W5, W53, W7, W8, W9)
- `_N4_r_SS` — N4 bias-field correction + co-registration + skull-strip (W20 only)
- Mixed within patient: T1gd often has `_r_` while T1/T2/FLAIR have `_r3_` (e.g. W9, W30, W7)

**No SWAN/SWI files anywhere in the dataset (0 matches for swan/swi/SWI/SWAN).**

---

## Representative patient file listings

### W1 (first, standard `_r_` suffix)
```
W1_1996.10.25_flair_LPS_r_SS.nii.gz    2.7M
W1_1996.10.25_t1_LPS_r_SS.nii.gz       2.6M
W1_1996.10.25_t1gd_LPS_r_SS.nii.gz     3.2M
W1_1996.10.25_t2_LPS_r_SS.nii.gz       3.3M
```

### W29 (middle, standard `_r_` suffix)
```
W29_1998.05.21_flair_LPS_r_SS.nii.gz   2.8M
W29_1998.05.21_t1_LPS_r_SS.nii.gz      2.9M
W29_1998.05.21_t1gd_LPS_r_SS.nii.gz    3.4M
W29_1998.05.21_t2_LPS_r_SS.nii.gz      3.6M
```

### W9 (last by sort, mixed `_r_` / `_r3_` within patient)
```
W9_1997.04.10_flair_LPS_r_SS.nii.gz    3.4M
W9_1997.04.10_t1_LPS_r3_SS.nii.gz      6.1M    ← r3 variant
W9_1997.04.10_t1gd_LPS_r_SS.nii.gz     4.0M
W9_1997.04.10_t2_LPS_r_SS.nii.gz       3.7M
```

---

## NIfTI header

From `W1_1996.10.25_t1gd_LPS_r_SS.nii.gz` (nibabel):

```
shape: (240, 240, 155)
zooms: (np.float32(1.0), np.float32(1.0), np.float32(1.0))
dtype: float32
affine:
[[ -1.   0.   0.  -0.]
 [  0.  -1.   0. 239.]
 [  0.   0.   1.   0.]
 [  0.   0.   0.   1.]]
```

Confirmed SRI24 atlas space (240×240×155, 1 mm isotropic). Affine is identical to SRI24 `spgr_unstrip_lps.nii.gz` atlas header — all volumes are in the same space. All tested patients (W1, W18, W20, W4) return identical shape and zooms regardless of `_r_`/`_r3_`/`_N4_` suffix.

---

## Intensity ranges (W1)

| Modality | min | max | dtype |
|---|---|---|---|
| T1pre | 0.0 | 2880.0 | float32 |
| T1c (t1gd) | 0.0 | 8052.0 | float32 |
| T2 | 0.0 | 10054.0 | float32 |
| FLAIR | 0.0 | 2684.0 | float32 |

Background is zero (skull-stripped). Intensities are in raw scanner units (not normalized). T1c mean over nonzero voxels ≈ 2607.

---

## Segmentation labels

Files: `<id>_CWRU_labels.nii.gz` and `<id>_UPenn_labels.nii.gz`  
Shape: (240, 240, 155), zooms 1 mm isotropic, dtype float32.

Unique label values across all tested patients (W1 CWRU, W1 UPenn, W29 CWRU, W29 UPenn):
```
{0.0, 1.0, 2.0, 4.0}
```

This is the **BraTS-style label scheme**:
- 0 = background
- 1 = necrotic core / non-enhancing tumour core (NCR/NET)
- 2 = peritumoral edema (ED)
- 4 = enhancing tumour (ET)

(BraTS 2021 convention; label 3 = GD-enhancing tumour in older versions — this dataset uses 4.)

### Annotation coverage

| Annotator | Patients | N |
|---|---|---|
| CWRU | W1–W55 excl. W30, W54, W6 | 31/34 |
| UPenn | All 34 | 34/34 |
| Both annotators | 31 patients | 31 |
| UPenn only | W30, W54, W6 | 3 |

---

## SRI24 atlas

`2_Atlas_SRI/spgr_unstrip_lps.nii.gz`:
- Shape: (240, 240, 155), 1 mm isotropic, float32
- **Non-skull-stripped** (`unstrip` in filename) — the reference template used for registration
- Identical affine to patient images

---

## Preprocessing state

Already applied (inferred from filename suffixes + header uniformity):
1. **Skull stripping** (`_SS`) — all volumes, all patients
2. **Co-registration to SRI24 atlas** (`_r_` or `_r3_`) — all modalities per patient are co-registered to the same space; T1gd consistently uses `_r_`, other modalities sometimes `_r3_`
3. **N4 bias-field correction** (`_N4_`) — W20 only

Not applied (not in filename, not documented):
- Intensity normalisation / percentile normalisation
- Brain mask generation (no separate `_mask_` file exists; background is zero from skull-strip)
- Any histogram matching or z-score normalisation

---

## Documentation

No README, no metadata CSV, no PDF found anywhere in the dataset tree. The parent directory contains only:
```
PKG - IvyGAP-Radiomics-SRI/
├── IvyGAP-Radiomics-SRI.sums    (checksum file)
└── IvyGAP-Radiomics/
    └── Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations/
```

The dataset is from the TCIA collection "IvyGAP-Radiomics" (DOI: 10.7937/K9/TCIA.2018.3RIE30SL). The SRI preprocessing pipeline and label scheme are described in:
- Calabrese et al. (2022) "The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset" — *Radiology: Artificial Intelligence* (related UCSF cohort, same SRI24 pipeline)
- Original IvyGAP: Puchalski et al. (2018) *Science* 359(6380):1253-1261

---

## VENA modality coverage

| VENA input | Required | Present in IvyGAP |
|---|---|---|
| T1pre | yes | yes (`_t1_`) |
| T1c (synthesis target) | yes | yes (`_t1gd_`) |
| T2 | yes | yes (`_t2_`) |
| FLAIR | yes | yes (`_flair_`) |
| **SWAN / SWI** | **yes (core differentiator)** | **NO — absent from all 34 patients** |
| Tumour segmentation | for mask conditioning | yes (BraTS-style, 2 annotators) |
| Brain mask | for skull-strip foreground | implicit (zero background) |

**CRITICAL FLAG: SWAN is absent from IvyGAP.** This dataset cannot be used for the full VENA pipeline (vessel-conditioned synthesis). Its viable roles are:

1. Training the **vessel-unaware baseline** (`fm_baseline`, `fm_mask` with tumour mask only — no `M_v`)
2. Ablation rows that isolate the T1pre / T2 / FLAIR → T1c mapping without vessel conditioning
3. **Not** usable for ablation rows that test SWAN encoding or vessel-conspicuity metrics

---

## Open questions for the user

1. **SWAN absence is a blocker for the core VENA hypothesis.** Is the intent to use IvyGAP as supplemental training data for the vessel-unaware branches only, or is there an expected source of SWI/SWAN for these subjects?
2. **Mixed `_r_` vs `_r3_` registration suffixes within a patient** (e.g. W9: T1gd is `_r_`, T1 is `_r3_`). Are these guaranteed to be in the same voxel space (same affine)? The headers tested confirm identical shape/affine, but the registration procedure difference may cause subtle misalignment between modalities in some patients — warrants a visual QC step.
3. **N4 bias-field correction only for W20.** Is this intentional (W20 had lower SNR) or a preprocessing inconsistency? If the reader reads N4 for W20 and raw for all others, the converter should document this heterogeneity in the H5 metadata.
4. **3 patients (W30, W54, W6) have UPenn annotations only** — not CWRU. If consensus/STAPLE segmentations are needed, these 3 must be handled as UPenn-only or excluded.
5. **No patient-level clinical metadata on disk** (IDH status, MGMT, WHO grade, age, sex). TCIA DICOM metadata may carry this; was it downloaded? A metadata CSV is needed for stratified splits and downstream analysis.
6. **Label 4 vs label 3 ambiguity.** IvyGAP labels use {0,1,2,4} matching BraTS-2021 convention. Confirm the niigz reader maps these to the project's tumour-mask binary or multiclass scheme (BraTS-GLI uses the same convention, so the existing reader may be reusable as a template).
7. **Total N=34 is small** relative to UCSF-PDGM (N=501). Intended use: full train/val/test split, or only train augmentation?
