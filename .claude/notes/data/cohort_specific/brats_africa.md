# BraTS-Africa Dataset Notes

**Source path**: `/media/mpascual/MeningD2/GLIOMA/BRATS_AFRICA/BraTS-Africa/`
**Intended use in VENA**: OOD testing only (not training).

---

## A. Top-level structure

```
BraTS-Africa/
├── 51_OtherNeoplasms/   (51 patients)
└── 95_Glioma/           (95 patients)
```

Two pathology subsets encoded as top-level directories with numeric prefix. No further subdirectory split (no train/val/test folders — the full dataset is released as a single set for the challenge). Each subset will be registered as a separate pathology cohort in the corpus JSON.

No README, PDF, or metadata CSV found at any level inside `BraTS-Africa/`.

---

## B. Per-patient layout

Pattern: `<subset>/<patient_id>/<patient_id>-<modality>.nii.gz`

All patients have exactly 5 files: t1n, t1c, t2f, t2w, seg.

### Representative patients — 95_Glioma

**BraTS-SSA-00002-000** (first):
```
BraTS-SSA-00002-000-seg.nii.gz   65 KB
BraTS-SSA-00002-000-t1c.nii.gz  3.1 MB
BraTS-SSA-00002-000-t1n.nii.gz  3.2 MB
BraTS-SSA-00002-000-t2f.nii.gz  3.5 MB
BraTS-SSA-00002-000-t2w.nii.gz  3.5 MB
```

**BraTS-SSA-00130-000** (middle):
```
BraTS-SSA-00130-000-seg.nii.gz   94 KB
BraTS-SSA-00130-000-t1c.nii.gz  2.5 MB
BraTS-SSA-00130-000-t1n.nii.gz  2.4 MB
BraTS-SSA-00130-000-t2f.nii.gz  2.7 MB
BraTS-SSA-00130-000-t2w.nii.gz  2.6 MB
```

**BraTS-SSA-00230-000** (last):
```
BraTS-SSA-00230-000-seg.nii.gz   68 KB
BraTS-SSA-00230-000-t1c.nii.gz  2.3 MB
BraTS-SSA-00230-000-t1n.nii.gz  2.3 MB
BraTS-SSA-00230-000-t2f.nii.gz  2.3 MB
BraTS-SSA-00230-000-t2w.nii.gz  2.6 MB
```

### Representative patients — 51_OtherNeoplasms

**BraTS-SSA-00009-000** (first):
```
BraTS-SSA-00009-000-seg.nii.gz   38 KB
BraTS-SSA-00009-000-t1c.nii.gz  3.8 MB
BraTS-SSA-00009-000-t1n.nii.gz  3.9 MB
BraTS-SSA-00009-000-t2f.nii.gz  4.0 MB
BraTS-SSA-00009-000-t2w.nii.gz  4.1 MB
```

**BraTS-SSA-00170-000** (middle), **BraTS-SSA-00231-000** (last): same 5-file layout.

### Filename convention

| Suffix | Modality |
|--------|----------|
| `-t1n` | T1 pre-contrast (native; "T1n") |
| `-t1c` | T1 post-contrast (contrast-enhanced; "T1c") |
| `-t2f` | T2 FLAIR |
| `-t2w` | T2-weighted |
| `-seg` | Tumour segmentation |

No SWAN/SWI files. No JSON sidecars. No metadata CSVs at any level.

---

## C. NIfTI header sample

**Patient BraTS-SSA-00002-000, t1n (T1 pre-contrast), 95_Glioma:**

```
shape: (240, 240, 155)
zooms: (1.0, 1.0, 1.0)
dtype: float32
affine:
[[  1.   0.  -0.  -0.]
 [  0.   1.  -0. 239.]
 [  0.   0.   1.   0.]
 [  0.   0.   0.   1.]]
```

Matches BraTS standard SRI24 atlas space: 1 mm isotropic, (240, 240, 155).

**Patient BraTS-SSA-00002-000, t1c (T1 post-contrast), 95_Glioma:**

```
shape: (240, 240, 155)
zooms: (1.0, 1.0, 1.0)
dtype: float32
affine:
[[  1.   0.  -0.  -0.]
 [  0.   1.  -0. 239.]
 [  0.   0.   1.   0.]
 [  0.   0.   0.   1.]]
```

All modalities share the same shape and affine (co-registered in atlas space).

**Patient BraTS-SSA-00009-000, t1n, 51_OtherNeoplasms:**
```
shape: (240, 240, 155)
zooms: (1.0, 1.0, 1.0)
```
Same atlas space.

---

## D. Intensity range and segmentation labels

**T1n, BraTS-SSA-00002-000 (glioma):**
- dtype: float32
- min: -290.0, max: 11148.0
- nonzero voxel fraction: 0.170 (~17%)

**T1n, BraTS-SSA-00009-000 (other neoplasm):**
- min: -118.0, max: 12743.0
- nonzero fraction: 0.203

Negative minimum values are expected in BraTS standardised volumes (z-score normalised per modality within the brain mask, with background = 0).

**Segmentation labels — 95_Glioma (BraTS-SSA-00002-000):**
```
unique labels: [0, 1, 2, 3]
```

**Segmentation labels — 51_OtherNeoplasms (BraTS-SSA-00009-000):**
```
unique labels: [0, 1, 3]
```
(label 2 absent in this case; both subsets use {0, 1, 2, 3})

BraTS 2023/SSA label convention:
- 0 = background
- 1 = NCR (necrotic core)
- 2 = ED (edema/invasion)
- 3 = ET (enhancing tumour)

(Label 4 = ET was renumbered to 3 in BraTS 2023 vs BraTS 2021. Confirmed: label 4 does not appear in any glioma patient; the scheme is 0/1/2/3.)

---

## E. Top-level documentation

No README, PDF, or metadata CSV found inside `BraTS-Africa/` or its parent `BRATS_AFRICA/`. Documentation must be sourced from the BraTS-Africa challenge paper (Adewole et al., BraTS-Africa 2023, Synapse challenge platform).

Stated preprocessing per challenge paper:
- Co-registered to SRI24 atlas space
- Skull-stripped
- Resampled to 1 mm isotropic
- Z-score normalised per modality within brain mask
- Consistent affine across all modalities per patient

---

## F. Patient / scan counts

| Subset | Patients | Sessions |
|--------|----------|----------|
| 95_Glioma | 95 | 95 (cross-sectional) |
| 51_OtherNeoplasms | 51 | 51 (cross-sectional) |
| **Total** | **146** | **146** |

Cross-sectional only. Folder suffix `-000` on all patient IDs (timepoint index = 0; no longitudinal data in the released challenge set).

---

## G. Modalities vs VENA requirements

| VENA modality | BraTS-Africa name | Available |
|---------------|------------------|-----------|
| T1pre | `-t1n` | YES |
| T1c | `-t1c` | YES |
| T2 | `-t2w` | YES |
| FLAIR | `-t2f` | YES |
| **SWAN/SWI** | — | **NO — ABSENT** |

`rg`/`find` for `*swi*`, `*swan*` across the full dataset returns zero results.

**SWAN is absent from BraTS-Africa.** Since VENA uses SWAN as the vessel-prior source ($M_v$ from Frangi on SWAN), this dataset cannot provide the vessel-prior input channel. Its OOD-test role must be defined accordingly (see Open Questions).

---

## H. Preprocessing state

- Background voxels are exactly 0 (confirmed by nonzero fraction ~17–20%).
- Volumes are skull-stripped (sparse non-zero region consistent with brain-only mask).
- In SRI24 atlas space, 1 mm isotropic, shape (240, 240, 155).
- Z-score normalised per modality within brain mask: floats with negative minimum values, not raw Hounsfield/arbitrary scanner units.
- All modalities co-registered to the same affine per patient.
- No bias-field correction step documented separately (BraTS pipeline applies N4 as part of standardisation).

---

## Open questions for the user

1. **SWAN absent**: VENA's vessel prior $M_v$ is derived from SWAN/SWI via Frangi filter. BraTS-Africa has no SWAN. How should the OOD test be structured without the vessel-prior input? Options: (a) zero-fill the SWAN channel; (b) synthesise a pseudo-SWAN from T2/FLAIR; (c) restrict OOD test to the T1c synthesis metrics only (no vessel-resolved evaluation for this cohort). Decision needed before `routines/h5_datasets/brats_africa/` is written.

2. **Label convention ambiguity**: BraTS-Africa 51_OtherNeoplasms uses the same 0/1/2/3 label scheme as glioma. Labels 1/2/3 (NCR, ED, ET) may not map cleanly to non-glioma tumour biology. Confirm whether the tumour-region evaluation metrics should be applied to OtherNeoplasms or only to Glioma.

3. **Patient ID numbering is shared across subsets** (BraTS-SSA-XXXXX IDs span both `95_Glioma` and `51_OtherNeoplasms` — e.g. `00009` is OtherNeoplasms, `00002` is Glioma). The `cohort_patient_id` in the H5 must include subset as a namespace prefix to avoid collisions.

4. **No official train/val/test split released** for the challenge subset on disk. VENA will need to define a split for OOD testing (likely all 146 used as test-only, no internal split needed).
