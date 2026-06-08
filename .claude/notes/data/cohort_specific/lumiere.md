# LUMIERE Dataset Notes

**Source path**: `/media/mpascual/MeningD2/GLIOMA/LUMIERE/`
**Intended use in VENA**: training + validation + testing (longitudinal glioma; each session treated as independent scan for cross-sectional pipeline).
**Reference**: Suter et al., Scientific Data 2022, "The LUMIERE Dataset: Longitudinal Glioblastoma MRI With Expert RANO Evaluation." DOI: 10.1038/s41597-022-01881-7

---

## A. Top-level structure

```
LUMIERE/
├── LUMIERE-Demographics_Pathology.csv
├── LUMIERE-MRinfo.csv
├── LUMIERE-datacompleteness.csv
├── LUMIERE-ExpertRating-v202211.csv
├── LUMIERE-readme.pdf
└── Imaging-v202211/
    └── Imaging/
        ├── Patient-001/
        │   ├── week-000-1/
        │   ├── week-000-2/
        │   ├── week-044/
        │   └── week-056/
        ├── Patient-002/
        │   └── week-000 ... week-047  (7 sessions)
        ...
        └── Patient-091/
```

91 patients, 638 sessions total. No train/val/test subdirectory split in the imaging tree — splits must be defined by VENA. Session naming: `week-NNN` (integer weeks post-surgery), with `-1`/`-2` suffix for same-week repeat acquisitions.

---

## B. Per-patient layout

**Tree structure**: `Patient-NNN/week-NNN[-N]/`

Each session directory contains:
```
CT1.nii.gz                       # T1 post-contrast (raw, native space)
T1.nii.gz                        # T1 pre-contrast (raw, native space)
T2.nii.gz                        # T2-weighted (raw, native space)
FLAIR.nii.gz                     # FLAIR (raw, native space)
DeepBraTumIA-segmentation/
│   ├── atlas/
│   │   ├── segmentation/
│   │   │   └── seg_mask.nii.gz              # 4-label seg in MNI atlas space (182,218,182)
│   │   │       measured_volumes_in_mm3.json
│   │   └── skull_strip/
│   │       ├── brain_mask.nii.gz
│   │       ├── ct1_skull_strip.nii.gz       # T1c skull-stripped, atlas space, 1mm iso
│   │       ├── flair_skull_strip.nii.gz
│   │       ├── t1_skull_strip.nii.gz
│   │       └── t2_skull_strip.nii.gz
│   └── native/
│       ├── segmentation                     # (directory; contents not inspected)
│       ├── skull_strip                      # (directory; contents not inspected)
│       └── transformation                   # (directory; registration matrices)
HD-GLIO-AUTO-segmentation/
    ├── native/
    │   ├── segmentation_CT1_origspace.nii.gz   # 3-label seg in CT1 native space
    │   ├── segmentation_FLAIR_origspace.nii.gz # 3-label seg in FLAIR native space
    │   ├── segmentation_T1_origspace.nii.gz
    │   └── segmentation_T2_origspace.nii.gz
    └── registered/
        ├── CT1_r2s_bet.nii.gz                  # skull-stripped, registered to CT1 space
        ├── CT1_r2s_bet_reg.mat
        ├── CT1_r2s_bet_reg.nii.gz
        ├── FLAIR_r2s_bet.nii.gz
        ├── FLAIR_r2s_bet_reg.mat
        ├── FLAIR_r2s_bet_reg.nii.gz
        ├── segmentation.nii.gz                 # 3-label seg in registered space
        ├── T1_r2s_bet.nii.gz
        ├── T1_r2s_bet_reg.mat
        ├── T1_r2s_bet_reg.nii.gz
        ├── T2_r2s_bet.nii.gz
        ├── T2_r2s_bet_reg.mat
        └── T2_r2s_bet_reg.nii.gz
```

**Representative patients:**

| Patient | Sessions |
|---------|----------|
| Patient-001 | week-000-1, week-000-2, week-044, week-056 (4) |
| Patient-002 | week-000, week-003, week-021, week-037, week-040-1, week-040-2, week-047 (7) |
| Patient-006 | 13 sessions (week-000 to week-135) |
| Patient-046 (middle) | week-000, week-001, week-016, week-064 (4) |
| Patient-091 (last) | week-000, week-001, week-014, week-026, week-036, week-043 (6) |

---

## C. NIfTI header samples

### Raw T1 pre-contrast (Patient-001/week-000-1/T1.nii.gz) — native space

```
shape: (640, 640, 24)
zooms: (0.359375, 0.359375, 6.000003)
dtype: uint16
affine:
[[-3.58051181e-01 -3.05501260e-02  6.76660612e-02  1.19269661e+02]
 [ 2.84099132e-02 -3.45523179e-01 -1.58010066e+00  1.44744431e+02]
 [ 1.19420728e-02 -9.39723775e-02  5.78780937e+00 -4.63451385e+01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```
Anisotropic native space: 0.36 mm in-plane, 6 mm slice thickness. NOT atlas-registered.

### Raw CT1 (T1 post-contrast) (Patient-001/week-000-1/CT1.nii.gz) — native space

```
shape: (256, 256, 192)
zooms: (1.0, 1.0, 1.0)
dtype: uint16
affine:
[[ -1.  0.  0.  126.92533112]
 [  0. -1.  0.  162.38874817]
 [  0.  0.  1.  -94.99064636]
 [  0.  0.  0.    1.        ]]
min: 0.0, max: 1135.0
nonzero fraction: 0.098
```
CT1 for this session is already 1 mm isotropic (256, 256, 192) — this varies by session/scanner (see LUMIERE-MRinfo.csv). The native T1 and FLAIR are not isotropic.

### Skull-stripped CT1 in atlas space (DeepBraTumIA/atlas/skull_strip/ct1_skull_strip.nii.gz)

```
shape: (182, 218, 182)
zooms: (1.0, 1.0, 1.0)
min: 0.0, max: 1005.78
nonzero fraction: 0.178
```
MNI152 space (182, 218, 182), 1 mm isotropic, skull-stripped. Background exactly 0.

### Raw T2 (Patient-001/week-000-1/T2.nii.gz) — native space

```
shape: (512, 512, 24)
zooms: (0.449219, 0.449219, 6.000003)
dtype: uint16
```
Highly anisotropic native space.

---

## D. Intensity range and segmentation labels

### CT1 (raw native): min=0, max=1135, dtype=uint16
### T1 skull-stripped atlas (DeepBraTumIA): min=0.0, max=1005.78, float32

### DeepBraTumIA segmentation labels (atlas space, seg_mask.nii.gz)

```
unique labels: [0, 1, 2, 3]
label 0 (background): 7,147,531 voxels
label 1: 16,751 voxels
label 2: 13,262 voxels
label 3: 43,488 voxels
```
Labels 1/2/3 likely correspond to necrotic core, edema, enhancing tumour — consistent with BraTS-style convention but NOT confirmed from the PDF readme (not machine-readable). Verify against LUMIERE-readme.pdf.

### HD-GLIO-AUTO segmentation labels (native per-modality space)

```
CT1 native unique labels: [0, 1, 2]
FLAIR native unique labels: [0, 1, 2]
T1 native unique labels: [0, 1, 2]
T2 native unique labels: [0, 1, 2]
```
HD-GLIO uses 3-class scheme: 0=background, 1=?, 2=? (HD-GLIO paper: 1=whole tumour, 2=? or 1=core, 2=WT — verify against HD-GLIO-AUTO documentation).

### HD-GLIO registered segmentation (registered/segmentation.nii.gz)

```
shape: (640, 640, 24)
zooms: (0.359375, 0.359375, 6.000003)
dtype: uint8
unique labels: [0, 1, 2]
```
Registered to T1 (or FLAIR?) native space at original resolution — still anisotropic.

---

## E. Top-level documentation

Files at dataset root:
- `LUMIERE-readme.pdf` — dataset description (not machine-readable; inspect manually)
- `LUMIERE-Demographics_Pathology.csv` — columns: Patient, Survival time (weeks), Sex, Age at surgery (years), IDH (WT/mut), IDH method, MGMT qualitative, MGMT quantitative
- `LUMIERE-MRinfo.csv` — columns: Patient, Timepoint, Sequence, Field strength, Manufacturer, Model, Spacing, Slice thickness, Voxel size, Rows, Columns, Scanning sequence, Echo, Inversion, Flip, Frequency, Sampling, Scan option, Acquisition type, SAR, Sex, Age, Repetition, Slice count
- `LUMIERE-datacompleteness.csv` — per-session availability flags (x = present) for CT1, T1, T2, FLAIR, DeepBraTumIA, HD-GLIO-AUTO, CoLlAGe features
- `LUMIERE-ExpertRating-v202211.csv` — per-session RANO ratings (Pre-Op, Post-Op, PD, SD, PR, CR)

Key preprocessing facts from raw data inspection:
- **Raw volumes (CT1.nii.gz, T1.nii.gz, T2.nii.gz, FLAIR.nii.gz) are in native scanner space** — NOT skull-stripped, NOT atlas-registered, NOT resampled.
- **DeepBraTumIA atlas path** provides skull-stripped, MNI152-registered (182, 218, 182) 1 mm isotropic versions for all 4 modalities. This is the usable input path.
- **HD-GLIO-AUTO registered path** provides skull-stripped, inter-modality co-registered versions at native resolution (still anisotropic for T1/T2/FLAIR, but consistent spacing within session).
- Multi-vendor, multi-field-strength: 1.0T, 1.5T, 3.0T; Siemens, Philips, GE.

---

## F. Patient / scan counts

| Metric | Value |
|--------|-------|
| Patients | 91 |
| Total sessions | 638 |
| Average sessions/patient | 7.01 |
| CT1 present | 632 / 638 |
| T1 present | 617 / 638 |
| T2 present | 626 / 638 |
| FLAIR present | 612 / 638 |

All four modalities present in most sessions. Some sessions have one or more modalities missing (6–26 missing per modality). Data completeness flags are in `LUMIERE-datacompleteness.csv`.

---

## G. Modalities vs VENA requirements

| VENA modality | LUMIERE name | Available |
|---------------|-------------|-----------|
| T1pre | T1.nii.gz | YES (617/638 sessions) |
| T1c | CT1.nii.gz | YES (632/638 sessions) |
| T2 | T2.nii.gz | YES (626/638 sessions) |
| FLAIR | FLAIR.nii.gz | YES (612/638 sessions) |
| **SWAN/SWI** | — | **NO — ABSENT** |

`find` for `*swi*`, `*swan*` across the full LUMIERE tree returns zero results.

**SWAN is absent from LUMIERE.** This is the primary pipeline blocker for VENA integration. See Open Questions.

---

## H. Preprocessing state

**Raw files** (CT1.nii.gz, T1.nii.gz, T2.nii.gz, FLAIR.nii.gz):
- Native scanner space, NOT skull-stripped, NOT atlas-registered.
- Highly anisotropic (0.36–0.45 mm in-plane, 4–6 mm slice thickness for T1/T2/FLAIR).
- CT1 is sometimes already isotropic 1 mm (depends on session/scanner).
- dtype: uint16, raw scanner intensities.
- Nonzero fraction ~9–17% (not skull-stripped — skull and head visible).

**DeepBraTumIA atlas path** (`atlas/skull_strip/*.nii.gz`):
- MNI152 space, shape (182, 218, 182), 1 mm isotropic.
- Skull-stripped (background exactly 0, nonzero ~17–18%).
- float32 intensities (not further normalised — raw skull-stripped values, no z-score).
- Co-registered across all 4 modalities to the same atlas space.
- This is the **recommended input path** for VENA's H5 converter.

**HD-GLIO registered path** (`registered/*_r2s_bet_reg.nii.gz`):
- Skull-stripped, co-registered to CT1 native space.
- Retains native anisotropic resolution.
- Segmentation (`registered/segmentation.nii.gz`) in this same space.

---

## Open questions for the user

1. **SWAN absent — critical for VENA**: LUMIERE has no SWAN/SWI. VENA's vessel prior $M_v$ is derived from Frangi-filtered SWAN. Options:
   (a) Zero-fill SWAN channel + use Frangi on T1/T2 as proxy vessel prior.
   (b) Use cerebrovascular atlas vessel template as $M_v$ placeholder.
   (c) Mark LUMIERE as SWAN-absent and skip $M_v$ conditioning (train a LUMIERE-specific variant without vessel conditioning).
   This decision impacts whether LUMIERE sessions can be used for full FM training or only as a T1pre→T1c baseline without vessel conditioning.

2. **Which preprocessed version to use as the canonical input**: DeepBraTumIA atlas path (MNI, 182×218×182, 1 mm iso) vs HD-GLIO registered (native space, anisotropic). UCSF-PDGM uses SRI24 atlas at (240, 240, 155). MNI vs SRI24 mismatch between cohorts is a design choice for `MultiCohortLatentDataModule`. Options: (a) re-register LUMIERE atlas volumes to SRI24 at H5 build time; (b) accept different atlas spaces and handle via crop/pad in the DataModule.

3. **Session-as-independent-scan assumption**: VENA treats each session as a separate cross-sectional sample. Pre-operative (week-000) and post-operative (week-000-1/2) sessions have fundamentally different tumour burden. Confirm whether surgical sessions (week-000-1, week-000-2 with RANO label "Pre-Op"/"Post-Op") should be included or filtered at registry build time. Expert RANO ratings in `LUMIERE-ExpertRating-v202211.csv` could be used as a filter flag.

4. **CT1 ambiguity on low-flip-angle sequences**: Some CT1 sessions have flip angle 9°–15° (MPRAGE-like T1c), others 80–90° (spoiled GRE T1c). Vendor mix (Siemens MPRAGE, Philips FFE, GE SPGR). Confirm whether cross-vendor T1c intensity heterogeneity needs harmonisation before training, or whether percentile normalisation at encoding time is sufficient.

5. **DeepBraTumIA seg label mapping to BraTS convention**: Labels 0/1/2/3 observed; correspondence to NCR/ED/ET not confirmed from machine-readable source. Verify against LUMIERE-readme.pdf before writing the tumour mask H5 layer.

6. **Missing modality handling**: ~2–4% of sessions are missing at least one modality. VENA's H5 converter must handle this — flag missing volumes with a null entry or skip entire session? Decision needed before the converter is written.
