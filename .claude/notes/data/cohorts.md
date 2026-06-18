# VENA Training Cohort Registry — 2026-06-03 (last updated 2026-06-19)

Single source of truth for the multi-cohort corpus VENA trains and evaluates
on. Per-cohort markdown notes (e.g. `ivygap.md`, `lumiere.md`,
`brats_africa.md`) hold deeper detail; this file is the *index*.

The numbers below match the deduplicated `corpus_local.json` /
`corpus_server3.json` / `corpus_picasso.json` (schema 1.0.0). Patient counts
are post-conversion (skipped patients dropped); `n_kept` is the number
surviving the `cohort_dedup` preflight.

## State as of 2026-06-19 (post-audit fix-up)

Three audit-§0 critical findings (v4 brain-mask synth-ones, BraTS-Africa
z-score skew, IvyGAP CC noise) **CLOSED on the Picasso production data**.
See `2026-06-19_data_audit_v2.md` for the full report. Key changes that
affect this registry:

- **BraTS-PED is back in `corpus_picasso.json`** as `role: test_only`
  (260 patients; image H5 re-transferred 2026-06-18, brain_latent encoded
  on loginexa, `produced_by_brain_to_latent=True`).
- **BraTS-GLI + IvyGAP brain masks unified** to union-of-4-modalities +
  `clean_brain_mask` (was `t1pre > 0`). BraTS-GLI gained ~755k voxels per
  scan at the boundary; IvyGAP lost ~118 voxels per scan of skull-strip
  noise.
- **BraTS-Africa-Glioma + Other re-encoded** with `percentile_use_brain_mask=True`;
  latent distributions now match UCSF (mean ≈ −0.14/−0.20, std ≈ 0.95/0.97).
- **v4 brain-mask "synth-ones" CLOSED** via TorchIO seed-replay
  (`scripts/patch_v4_brain_latent.py`); 2 066 v4 rows across the 6 cv
  cohorts now carry real warped brain masks (mean sums 24 735–26 947;
  pre-fix every row was 129 024).
- **Schema versions bumped**: aug-image + aug-latent 0.1.0 → 0.2.0
  (added `masks/brain` to aug-image manifest; conditional `masks/brain_latent`
  validator on aug-latent gated on root attr `produced_by_brain_to_latent`).
  Base image + latent H5s stay at `2.0.0`.
- **Splits normalized**: `splits/cv/fold_N/{train,val}` + `splits/test`
  everywhere on cv cohorts; only `splits/test` on test-only cohorts. Legacy
  flat `splits/{train,val}` aliases dropped on IvyGAP + REMBRANDT.
- **REMBRANDT image H5 caveat**: shipped originally with ONLY flat
  `splits/{train,val,test}` (small cohort, no nested CV). After Phase 6
  drop, `splits/cv` was copied in-place from the latent H5 (identical
  patient sets — same `make_cohort_splits` seed). Both image + latent
  now expose `splits/cv/fold_0/train n=53`, `splits/cv/fold_0/val n=5`,
  `splits/test n=5`.

The "Open follow-ups" §3 (REMBRANDT converter) and the "intensity policy"
note further down are now stale — see the marked sections.

---

## At a glance

| # | Cohort | Pathology | n_patients | n_scans | Long. | Role | Label sys | Atlas | Strip | Dedup outcome | Origin |
|---|---|---|---:|---:|:--:|---|---|---|---|---|---|
| 1 | **UCSF-PDGM** | preoperative_glioma | 495 → **202 kept** | 495 | no | cv | BraTS2021 `{0,1,2,4}` | SRI24 | preprocessed | **293 dropped** via `metadata/brats21_id` (overlap with BraTS-GLI) | UCSF, US |
| 2 | **BraTS-GLI** | preoperative_glioma | 1133 | 1251 | yes | cv | BraTS2023 `{0,1,2,3}` | SRI24 | preprocessed | implicit_brats21 — kept whole (highest priority) | RSNA / ASNR / MICCAI 2021-2024 |
| 3 | **IvyGAP** | glioma | 34 | 34 | no | cv | BraTS2021 | SRI24 | preprocessed | no bridge — kept whole (warn) | Allen Inst. / Ben & Catherine Ivy Foundation |
| 4 | **LUMIERE** | glioma | 91 | 638 | **yes** | cv | BraTS2023 | SRI24 | preprocessed | no bridge — kept whole | Insel Bern, CH |
| 5 | **BraTS-Africa-Glioma** | glioma | 95 | 95 | no | test_only | BraTS2023 | SRI24 | **HD-BET via VENA** | passthrough | Sub-Saharan Africa consortium |
| 6 | **BraTS-Africa-Other** | other_neoplasm | 51 | 51 | no | test_only | BraTS2023 | SRI24 | **HD-BET via VENA** | passthrough | Sub-Saharan Africa consortium |
| 7 | **BraTS-PED** | pediatric_glioma | 261 → 260 | 261 | no | test_only | BraTS2023 | SRI24 | **HD-BET via VENA** (defaced source) | passthrough | CBTN / CHOP / DIPG-pBTC |
| 8 | **REMBRANDT** | glioma | 63 (64 sessions – 1 missing T1pre) | 63 | no | cv (53/5/5) | BraTS2021 | SRI24 | **HD-BET via VENA** (CBICA-preprocessed source) | passthrough — not in BraTS21 mapping | NCI / Henry Ford Hospital |
| 9 | **UPENN-GBM** | preoperative_glioma | 611 → **164 kept** | 611 | no | cv | BraTS2021 `{0,1,2,4}` | SRI24 | preprocessed (skull-stripped at source) | **447 dropped** via `metadata/brats21_id` (overlap with BraTS-GLI's implicit BraTS-21 claim) | UPenn (Hospital of the University of Pennsylvania) |

**Totals (post-dedup, 2026-06-03)**: 2833 patients in across 9 cohorts → **2093 kept** (740 rejected: 293 UCSF-PDGM + 447 UPENN-GBM, both via `metadata/brats21_id` bridge against BraTS-GLI). Cross-verified between local `corpus_local.json` (SHA `621278...`) and server3 `corpus_server3.json` (SHA pinned in `/media/hddb/mario/artifacts/preflights/cohort_dedup/LATEST/decision.json`). Modalities are uniformly `{t1pre, t1c, t2, flair}`; no cohort carries SWAN/SWI (the conditioning prior the proposal's vessel branch needs — a known gap until the Málaga in-house cohort lands).

**Training pool at fold 0 (2026-06-05, post-dedup, post-split):** **1224 train patients / 1664 train scans, 290 val patients / 402 val scans, 173 test patients / 253 test scans** across the 6 cv cohorts. The 3 `test_only` cohorts (BraTS-Africa-Glioma, BraTS-Africa-Other, BraTS-PED) are excluded from training but available as held-out external evaluators (146 + 51 + 260 = 457 patients). Per-cohort split breakdown lives in the offline-augmentation table below.

---

## Per-cohort detail

### 1. UCSF-PDGM (UCSF Preoperative Diffuse Glioma MRI)

* **What**: 501 preoperative adult diffuse-glioma patients from UCSF; the
  internal training cohort for VENA.
* **Patients in corpus**: 495 (six excluded by upstream QC); **202** kept
  after dedup against BraTS-GLI.
* **Splits** (single random patient-level partition):
  `splits/test = 50`, `splits/cv/fold_{0..4}/{train,val}` 5-fold on the
  remaining 445. See `tests/data/h5/test_ucsf_pdgm_image_convert_smoke.py`.
* **Preprocessing**: CBICA pipeline — N4 bias-corrected, co-registered to
  SRI24, skull-stripped at upload time. Intensities are scanner-native
  inside the brain; the H5 stores raw float32; percentile normalisation runs
  at encode time in the MAISI front-end (`foreground_only=true`).
* **Tumour seg**: BraTS-2021 labels `{0=bg, 1=NCR/NET, 2=ED, 4=ET}`.
* **Metadata bridge**: `metadata/brats21_id` (when non-empty → patient is
  also a BraTS21 patient → dropped because BraTS-GLI has higher priority).
* **Paper**: Calabrese et al. (2022). *The UCSF Preoperative Diffuse Glioma
  MRI Dataset.* Radiology AI 4(6):e220058.
  [DOI 10.1148/ryai.220058](https://doi.org/10.1148/ryai.220058).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/h5/UCSFPDGM_image.h5`.

### 2. BraTS-GLI (BraTS Glioma — Pre-Operative subset)

* **What**: The multi-institutional adult glioma cohort assembled across the
  BraTS challenges 2017–2024; VENA uses the pre-operative subset.
* **Patients**: 1133 patients across 1251 scans (longitudinal — some
  patients have multiple pre-operative timepoints).
* **Splits**: held-out test + 5-fold CV. Same schema as UCSF-PDGM.
* **Preprocessing**: full BraTS preprocessing pipeline — co-registered to
  SRI24 1 mm iso, skull-stripped, LPS orientation, tumour seg by the BraTS
  expert panel.
* **Tumour seg**: BraTS-2023 labels `{0, 1, 2, 3}` (encoder remaps to
  BraTS-2021 `{0,1,2,4}` at encode time when
  `label_system="BraTS2023"`).
* **Role in dedup**: `implicit_brats21_cohorts = ["BraTS-GLI"]` — it is
  assumed to contain every BraTS-2021 patient under renumbered IDs, so any
  cohort with a non-empty `metadata/brats21_id` bridge drops the matching
  patient (UCSF-PDGM is the only such cohort today).
* **Source papers**:
  * Menze et al. (2015). *The Multimodal Brain Tumor Image Segmentation
    Benchmark (BraTS).* IEEE TMI 34(10):1993–2024.
    [DOI 10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694).
  * Baid et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain
    Tumor Segmentation and Radiogenomic Classification.*
    [arXiv:2107.02314](https://arxiv.org/abs/2107.02314).
  * Karargyris et al. / Bakas et al. (2023). BraTS 2023 / BraTS-Africa /
    BraTS-PED challenge description.
    [arXiv:2305.17033](https://arxiv.org/abs/2305.17033).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/BRATS_GLI/PRE_OPERATIVE/h5/BraTS_GLI_image.h5`.

### 3. IvyGAP (Ivy Glioblastoma Atlas Project — radiomics release)

* **What**: 34 GBM patients with multi-institutional paired expert
  segmentations released alongside the Ivy GAP radiomics dataset.
* **Splits**: single random **24/5/5** train/val/test (N=34 too small for
  stable nested CV). Mirrors the smoke-test cohort layout.
* **Preprocessing**: CBICA pipeline with registration-variant precedence
  `_N4_r_SS > _r3_SS > _r_SS` (encoded as
  `metadata/source_basename_<modality>` rows so the variant chosen per scan
  is auditable). Skull-stripped at upload.
* **Tumour seg**: BraTS-2021 labels `{0,1,2,4}` — UPenn rater is the
  canonical mask; CWRU annotation path is preserved as metadata only.
* **Dedup**: 21 xlsx rows with portal IDs that do not match this cohort's
  `W<N>` IDs (`on_unresolvable=warn`); kept whole pending an external bridge
  file.
* **Paper**: Puchalski et al. (2018). *An Anatomic Transcriptional Atlas of
  Human Glioblastoma.* Science 360(6389):660–663.
  [DOI 10.1126/science.aaf2666](https://doi.org/10.1126/science.aaf2666).
  (Radiomics release: see `.claude/notes/data/ivygap.md`.)
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/IVYGAP/h5/IvyGAP_image.h5`.

### 4. LUMIERE (Longitudinal MRI Glioblastoma w/ RANO)

* **What**: 91 GBM patients with longitudinal MRI + per-timepoint RANO
  evaluation; 638 sessions total.
* **Splits**: 10 held-out test patients + 5-fold CV on the remaining 81.
  Splits are patient-level (no patient straddles).
* **Preprocessing**: BraTS-style preprocessed, skull-stripped, SRI24
  1 mm iso, LPS.
* **Tumour seg**: BraTS-2023 labels.
* **CSR layout**: `patients/{offsets,keys}` group rows by patient; one row
  per session.
* **Paper**: Suter et al. (2022). *The LUMIERE dataset: Longitudinal
  Glioblastoma MRI with Expert RANO Evaluation.* Scientific Data 9, 768.
  [DOI 10.1038/s41597-022-01881-7](https://doi.org/10.1038/s41597-022-01881-7).
* **Source on disk**: `/media/mpascual/MeningD2/GLIOMA/LUMIERE/h5/LUMIERE_image.h5`.

### 5. BraTS-Africa-Glioma

* **What**: 95 adult-glioma scans from sub-Saharan-African scanners (BraTS
  2023 Sub-Saharan Africa cluster).
* **Role**: test_only (`splits/test` covers every patient; no CV folds).
* **Preprocessing**: BraTS pipeline source is **defaced only**, not
  skull-stripped. VENA preprocess routine
  (`routines/preprocess/brats_africa_skullstrip`) ran HD-BET v2 to produce
  the skull-stripped tree that feeds the image-H5 converter.
* **Tumour seg**: BraTS-2023 labels.
* **Paper**: Adewole et al. (2023). *The Brain Tumor Segmentation (BraTS)
  Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient
  Population (BraTS-Africa).*
  [arXiv:2305.19369](https://arxiv.org/abs/2305.19369).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/BRATS_AFRICA/h5/BraTS_Africa_glioma_image.h5`.

### 6. BraTS-Africa-Other

* **What**: 51 non-glioma scans (other_neoplasm; meningioma, metastasis,
  cysts, ...) from the same BraTS-Africa release; held out as OOD.
* **Role**: test_only.
* **Preprocessing**: HD-BET via VENA, as above.
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/BRATS_AFRICA/h5/BraTS_Africa_other_image.h5`.

### 7. BraTS-PED (Pediatric Glioma — BraTS-PED 2024 Training)

* **What**: 261 pediatric HGG patients (BraTS-PED 2024 Training set).
* **Role**: test_only; OOD with respect to the adult training pool.
* **Preprocessing**: source is **defaced only** (skull retained). VENA
  preprocess routine `routines/preprocess/brats_ped_skullstrip` ran HD-BET
  v2 to strip; the image-H5 converter consumes the stripped tree.
* **Tumour seg**: BraTS-2023 labels.
* **Paper**: Kazerooni et al. (2023). *The BraTS Pediatric Brain Tumor
  Segmentation 2023 (BraTS-PEDs) Challenge: Focus on Pediatrics
  (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs).*
  [arXiv:2305.17033](https://arxiv.org/abs/2305.17033).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/BRATS_PED/h5/BraTS_PED_image.h5`.

### 8. REMBRANDT (REpository for Molecular BRAin Neoplasia DaTa)

* **What**: 64 adult-glioma sessions from the TCIA REMBRANDT collection
  (CBICA-preprocessed mirror); 20 patients with `900-00-XXXX` NCI portal
  IDs, 44 with `HFXXXX` Henry-Ford-Hospital legacy IDs. 1 session
  (HF0920_1991.06.14) quarantined for missing T1pre → final n=63.
* **Splits**: single random **53/5/5** train/val/test (N=63 too small for
  stable nested CV; mirrors IvyGAP). After the
  `splits/cv/fold_0/{train,val}` alias patch the trainer accepts it.
* **Preprocessing**: source is registered to SRI24 (`_LPS_rSRI`) but **not
  stripped**. VENA `routines/preprocess/rembrandt_skullstrip` ran HD-BET v2
  (config-driven regex + filename template since REMBRANDT does not follow
  the `BraTS-*-NNNNN-NNN` convention). Brain-volume fraction 15-19 %
  (smoke), 0 errors across 63/63 patients.
* **Tumour seg**: GLISTRboost output, BraTS-2021 labels
  `{0, 1, 2, 4}` (verified on smoke H5).
* **Dedup outcome**: REMBRANDT is **absent** from
  `BraTS2021_MappingToTCIA.xlsx` (collections present: ACRIN-FMISO,
  CPTAC-GBM, Collection 1/3/4/5/7/9, IvyGAP, TCGA-GBM, TCGA-LGG, UCSF-PDGM,
  UPENN-GBM). 63/63 kept; no bridge field needed.
* **Encode QC**: roundtrip MSE 1.3e-4 – 3.4e-4 per modality (≈ 35–39 dB
  PSNR with `data_range=1.0`), well under the 1e-3 acceptance bound.
* **Paper**: Madhavan et al. (2009). *Rembrandt: helping personalized
  medicine become a reality through integrative translational research.*
  Molecular Cancer Research 7(2):157-167.
  [DOI 10.1158/1541-7786.MCR-08-0435](https://doi.org/10.1158/1541-7786.MCR-08-0435).
  TCIA release: Scarpace et al. (2015). *Data From REMBRANDT* (Cancer Imaging
  Archive). [DOI 10.7937/K9/TCIA.2015.588OZUZB](https://doi.org/10.7937/K9/TCIA.2015.588OZUZB).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/REMBRANDT/h5/REMBRANDT_image.h5`
  (image), `/media/mpascual/MeningD2/GLIOMA/REMBRANDT/source/` (raw).

### 9. UPENN-GBM (University of Pennsylvania Glioblastoma — v2.0)

* **What**: 611 preoperative GBM patients with manual or automated tumour
  segmentation from the UPenn release (TCIA collection UPENN-GBM). The
  source `images_structural/` tree carries 671 patients; the cohort reader
  drops the 60 with **no** segmentation at discovery time (the converter
  never sees them).
* **Patients in corpus**: 611. **Post-dedup keep**: TBD (logged into
  `decision.json`); expectation ≈164 unique post-dedup against BraTS-GLI's
  implicit BraTS-2021 claim (447 of 611 carry a non-empty
  `metadata/brats21_id`).
* **Splits** (5-fold CV + ~10% test):
  `splits/test = ~62`, `splits/cv/fold_{0..4}/{train,val}` 5-fold on the
  remaining ~549. Identical signature to BraTS-GLI / UCSF-PDGM (calls
  `make_cohort_splits(n_folds=5, test_fraction=0.10, n_test_min=25,
  seed=42, role="cv")`).
* **Preprocessing**: source ships skull-stripped (CBICA pipeline; co-registered
  to SRI24 1 mm iso, LPS, `(240, 240, 155)`). No VENA-side HD-BET step needed.
  Intensities are scanner-native inside the brain; the H5 stores raw
  `float32`; percentile normalisation runs at encode time with
  `percentile_upper=99.95, percentile_foreground_only=true`, identical to
  UCSF-PDGM / BraTS-GLI on server3.
* **Tumour seg policy**: **manual seg preferred** (`images_segm/` — 147
  patients), automated fallback (`automated_segm/` — 611 patients, superset
  of manual). The reader records the source per patient under
  `metadata/seg_source ∈ {"manual", "automated"}` and the converter writes it
  into the H5 for auditing. Labels follow BraTS-2021 `{0=bg, 1=NCR/NET, 2=ED,
  4=ET}` (verified from seg headers).
* **Brain mask**: derived as the union of nonzero voxels across the four
  skull-stripped modalities (same pattern as BraTS-PED / REMBRANDT). Brain
  voxel fraction is ~16% of the (240,240,155) volume on smoke samples.
* **Dedup outcome (measured 2026-06-03)**:
  `bridge_fields: {UPENN-GBM: metadata/brats21_id}` — same contract as
  UCSF-PDGM. The lookup CSV
  (`/media/mpascual/MeningD2/GLIOMA/UPENN_GBM/metadata/UPENN-GBM_brats21_lookup_v1.csv`)
  was built once from `BraTS2021_MappingToTCIA.xlsx` by
  `scripts/preprocess/build_upenn_gbm_brats21_lookup.py`: filters rows where
  Data Collection ∈ {UPENN-GBM, UPENN-GBM_Additional} and `portal_id` matches
  `UPENN-GBM-NNNNN_NN`. 447 of 562 candidate xlsx rows match (115 of the
  "_Additional" rows carry `portal_id = "new-not-previously-in-TCIA"` — no
  per-patient bridge available; they fall to the `on_unresolvable=warn`
  path). The dedup `priority` slot is **3rd** (after BraTS-GLI, UCSF-PDGM),
  i.e. UPenn cedes overlapping patients to the higher-priority cohorts.
  **Final**: `n_total=611, n_kept=164, n_rejected=447`.
* **Encode QC (2026-06-03)**: roundtrip MSE on 4 patients × 4 modalities:
  t1pre 4.7–5.1e-4, t1c 1.9–2.6e-4, t2 3.1–4.2e-4, flair 4.2–4.8e-4 —
  all comfortably below the 1e-3 acceptance bound (≈33–37 dB PSNR with
  `data_range=1.0`).
* **S2 smoke (2026-06-03)**: 4-epoch run
  `2026-06-03_10-16-46_s2_81211dac` on server3 used all 9 cohorts incl.
  UPENN-GBM (`cfm_cohort_UPENN-GBM` column present, loss 1.91 → 1.40
  across epochs); exhaustive validation drew 10 UPENN-GBM patients per
  epoch (same quota as IvyGAP/REMBRANDT). No PreflightGateError.
* **Paper**: Bakas et al. (2022). *The University of Pennsylvania
  Glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, &
  radiomics.* Scientific Data 9, 453.
  [DOI 10.1038/s41597-022-01560-7](https://doi.org/10.1038/s41597-022-01560-7).
  TCIA release: [DOI 10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49).
* **Source on disk**:
  `/media/mpascual/MeningD2/GLIOMA/UPENN_GBM/PKG - UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/`
  (raw); `/media/mpascual/MeningD2/GLIOMA/UPENN_GBM/h5/UPENN-GBM_image.h5`
  (post-conversion).

---

## Cross-cohort invariants

* **Shape**: every cohort lives at native SRI24 `(240, 240, 155)`, 1 mm iso,
  LPS, `float32`. Common crop box `(192, 224, 192)`.
* **Modalities**: `{t1pre, t1c, t2, flair}` for every cohort. No cohort
  carries SWAN/SWI — the vessel-prior input the proposal needs is still
  pending the Málaga in-house cohort (`role: external`, expected 2026-Q3).
* **Brain mask** (as of 2026-06-19): derived in the converter as the union
  of nonzero voxels across the four modalities + `clean_brain_mask`
  (1000-voxel CC floor; preserves cerebellum + brainstem, drops boundary
  noise). UCSF-PDGM + LUMIERE inherit the shipped CBICA / Bern mask;
  BraTS-GLI + IvyGAP were harmonized in-place from the legacy `t1pre > 0`
  to the union-of-4 policy (2026-06-18). Every cohort's image H5 carries
  `masks/brain.attrs["brain_cc_cleaned"] = True`; harmonized cohorts also
  carry `brain_source_unified = True` + `brain_source_modalities =
  "t1pre,t1c,t2,flair"`.
* **Latent brain mask**: `masks/brain_latent` shape `(N, 1, 48, 56, 48)`
  int8 (max-pool-4 of the image-domain brain mask). Aug-latent rows for
  v4 are TorchIO-seed-replayed warps of the image-domain brain mask, not
  synth-ones. Every latent H5 carries root attr
  `produced_by_brain_to_latent = True` (post-2026-06-18); the conditional
  aug-latent validator gates on this.
* **Latent shape** (MAISI-V2 VAE): `(C=4, H=48, W=56, D=48)`; 4× spatial
  compression of the 192×224×192 brain box.
* **Intensity policy**: H5 stores native scanner intensities (BraTS-Africa
  still stores intra-brain z-score; the encoder accommodates via
  `mask=masks/brain`); per-modality percentile normalisation `[0, 99.95]`
  runs at encode time. Two equivalent encoder knobs:
  `percentile_foreground_only=True` (legacy: `x > 0` heuristic) **plus**
  `percentile_use_brain_mask=True` (default since 2026-06-18: passes
  `masks/brain` to `percentile_normalise`, bypassing the heuristic).
  `mask` overrides `foreground_only` when both are set; for raw cohorts the
  two paths produce byte-identical output, for z-score cohorts (BraTS-Africa)
  the mask path is required to preserve intra-brain negatives.

## Dedup invariants (as of 2026-06-03)

* Priority list: `["BraTS-GLI", "UCSF-PDGM", "UPENN-GBM", "IvyGAP", "LUMIERE"]`.
  Cohorts not in the list inherit lowest priority (passthrough, in
  insertion order) — currently `BraTS-Africa-{Glioma,Other}`, `BraTS-PED`,
  `REMBRANDT`.
* `implicit_brats21_cohorts = ["BraTS-GLI"]` — only BraTS-GLI is treated as
  the umbrella for BraTS-2021 IDs.
* `bridge_fields = {UCSF-PDGM: metadata/brats21_id, UPENN-GBM: metadata/brats21_id}`
  — two cohorts now carry an explicit cross-cohort bridge. IvyGAP has 21
  candidate overlaps in the xlsx but no matching ID space; the warn-mode lets
  it pass. UPenn's 115 "_Additional" rows with placeholder portal IDs also
  fall to the warn path.
* Decision file: `artifacts/preflights/cohort_dedup/LATEST/decision.json` v1.0.
  Current: 9 cohorts, **2833 in / 2093 kept / 740 rejected**
  (293 UCSF-PDGM + 447 UPENN-GBM). Trainer gates on
  `data.dedup_decisions_path` when that key is set; both
  `routes/preflights/cohort_dedup/configs/default.yaml` (local) and
  `default_server3.yaml` (server3 paths) are SHA-distinct decisions because
  the corpus registry paths differ.

---

## Offline augmentation bank (2026-06-05)

Bank produced by `routines/offline_aug/maisi/` over the 6 cv cohorts (test-only cohorts are not augmented by design). One image-domain H5 + one MAISI-encoded latent H5 per cohort, both schema-versioned (`vena.data.h5.augmented` v0.1.0). Each scan in scope is replicated as **K = 4 variants**:

| variant | image-domain transform (TorchIO + MONAI) | applies to | per-transform `p` |
|---|---|---|---|
| `v1` field/scanner | `RandomBiasField(order=3, coeffs=±0.5)` + `RandomGamma(log_γ ∈ ±0.3)` | inputs only | bias `p=1.0`, gamma `p=1.0` |
| `v2` contrast-shape | MONAI `RandHistogramShift(n_ctrl=(8,12))` + `RandomGamma` (tight) | inputs only | hist `p=1.0`, brightness `p=0.4` |
| `v3` SNR/resolution | `RandomNoise(std ∈ [0, 0.05])` + `RandomAnisotropy` + `RandomBlur` + `RandomMotion(num_transforms ∈ [1,3])` (low-p) | inputs only | noise `p=1.0`, aniso `p=0.7`, blur `p=0.4`, motion `p=0.1` |
| `v4` anatomy | `RandomElasticDeformation` + `RandomAffine(scale 0.9–1.1, rot ±10°, trans 8 vox)` | **inputs + target + WT mask** (joint) | elastic `p=1.0`, affine `p=1.0` |

`v0` is the unaugmented clean latent (re-used from the per-cohort `*_latents.h5`, not stored a second time). At training time the trainer samples `v0..v4` per `__getitem__` with `variant_weights={v0:0.2, v1:0.2, v2:0.2, v3:0.2, v4:0.2}` (uniform), then runs the *online* latent flip + translate tier on top.

### Per-cohort augmented-row counts (server 3)

| Cohort | source scans in bank | aug rows per variant (× 4) | **total aug rows** | merged-bank size (image + latent) |
|---|---:|---:|---:|---:|
| **UCSF-PDGM** | 181 | 181 × 4 | **724** | 27.3 GB + 5.2 GB |
| **BraTS-GLI** | 1124 | 1124 × 4 | **4496** | 155 GB + 32 GB |
| **UPENN-GBM** | 147 | 147 × 4 | **588** | 20.0 GB + 4.2 GB |
| **IvyGAP** | 29 | 29 × 4 | **116** | 4.0 GB + 0.83 GB |
| **LUMIERE** | 527 | 527 × 4 | **2108** | 79.8 GB + 15.0 GB |
| **REMBRANDT** | 58 | 58 × 4 | **232** | 7.8 GB + 1.65 GB |
| **Totals** | **2066 unique source scans** | — | **8264 aug rows** | **294 GB image + 59 GB latent = 353 GB** |

Bank coverage = "all non-`splits/test` patients × all CV folds" per cohort, written once and re-usable across every fold. The "source scans in bank" column therefore exceeds any single fold's `train_scans` count (which is itself a strict subset). The same bank is mirrored at `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<COHORT>/h5/` on Picasso (synced 2026-06-04T22:08 UTC).

### Per-cohort fold-0 split sizes (post-dedup-allow-list × `splits/cv/fold_0`)

| Cohort | train patients | train scans | val patients | val scans | test patients | test scans |
|---|---:|---:|---:|---:|---:|---:|
| UCSF-PDGM | 147 | 147 | 34 | 34 | 21 | 21 |
| BraTS-GLI | 815 | 899 | 204 | 225 | 114 | 127 |
| UPENN-GBM | 121 | 121 | 26 | 26 | 17 | 17 |
| IvyGAP | 24 | 24 | 5 | 5 | 5 | 5 |
| LUMIERE | 64 | 420 | 16 | 107 | 11 | 72 |
| REMBRANDT | 53 | 53 | 5 | 5 | 5 | 5 |
| **Totals (6 cv)** | **1224** | **1664** | **290** | **402** | **173** | **253** |

### Leakage verification — *no augmented patient reaches val or test during training*

Audited 2026-06-05 with `scripts/`-style probe against `corpus_server3.json` × `LATEST/decision.json`:

- The bank-builder excludes `splits/test` patients at write time (`_resolve_rows` in `vena.data.augment.offline.bank_builder`) — verified: **`aug_ids ∩ test_scan_ids = ∅` for all 6 cohorts**.
- `OfflineAugmentedLatentH5Dataset` is instantiated with `patient_ids=train_scan_ids` (line 784 in `vena/model/fm/lightning/data.py`). `__getitem__(idx)` reads `self.patient_ids[idx]`, so the train iterator never queries val patients — even though the bank file itself does store val-patient rows (this is by design so the same bank works for any fold).
- The val and test DataLoaders are built from `LatentH5Dataset(cohort.latent_h5, ...)` against the **clean** H5 only (lines 798–799). They never open the aug bank.
- The dedup allow-list is intersected with each of train / val / test patient keys before CSR expansion (lines 725–727), so dedup-rejected patients cannot reappear in any partition.

**Conclusion:** at the data-flow level, no augmented patient is presented to the validation or test loaders. The aug-bank's storage scope (all non-test patients) is a superset of any single fold's train set — that is intentional for fold reuse, not a leak.

### Batch composition strategy

`TemperatureBalancedSampler` at `tau = 0.5` ("square-root sampling" on patient counts). At fold 0:

- Cohort probabilities ∝ `N_c^0.5`: BraTS-GLI 0.40, UCSF-PDGM 0.17, UPENN-GBM 0.15, LUMIERE 0.11, REMBRANDT 0.10, IvyGAP 0.07.
- Within a cohort: uniform patient → uniform scan within that patient (so a 6-session LUMIERE patient does not over-contribute relative to a 1-session UCSF-PDGM patient).
- Per-batch diversity guarantee: first `min(batch_size, n_cohorts)` slots drawn **without replacement**, then remaining slots with replacement.

The √N choice is the standard multi-domain compromise (uniform = over-represents the smallest cohort; proportional = drowns the smallest under the largest; with a 815:24 ≈ 34× max:min patient-count ratio here, `τ=0.5` shrinks the effective ratio to ≈ √34 ≈ 5.8×).

**Literature support for `τ=0.5`** (Tier 1 — direct precedent):

- **Conneau & Lample (2019)** *Cross-lingual Language Model Pretraining (XLM)*. NeurIPS 2019. arXiv:1901.07291. The seminal reference: $p_i \propto n_i^{\alpha}$ with α=0.5 on multilingual corpora; the heuristic VENA inherits.
- **Arivazhagan et al. (2019)** *Massively Multilingual NMT in the Wild*. arXiv:1907.05019. Best $T = 5$ on 103 languages ($\tau = 1/T = 0.2$ in our parameterisation); shows the optimal exponent depends on imbalance severity.
- **Salmani et al. (2025)** *Sampling and Loss Weights in Multi-Domain Training*. arXiv:2511.06913. Derives $p_i \propto \sqrt{n_i \sigma_i^2}$ from gradient-variance minimisation → $\tau = 0.5$ is variance-optimal **under the assumption that within-cohort gradient variance is uniform across cohorts**. Direct theoretical grounding for our choice.
- **Patil et al. (2025)** *Temperature Sampling for Robot Learning on Imbalanced Datasets*. arXiv:2510.19373. The closest empirical analogue to VENA (5–20× imbalance, non-NLP): $\tau = 0.5$ is the most robust choice across two benchmarks vs $\tau \in \{0.25, 0.75, 1.0\}$.
- **Glocker et al. (2019)** *Machine Learning with Multi-Site Imaging Data*. arXiv:1910.04597. Shows scanner-site effects dominate multi-site MRI representations after standard harmonisation — motivates cross-site mixing within each batch (= the force-distinct-cohort policy).

**Open critique** (load-bearing):

- **Wang et al. (2020)** *Balancing Training for Multilingual NMT*. ACL 2020, arXiv:2004.06748. **No fixed τ is universally optimal** — the best τ varies per experimental setting; the paper learns per-domain weights via min-max optimisation and consistently beats every fixed-τ baseline. For VENA this means: **ablate $\tau \in \{0.3, 0.5, 0.7, 1.0\}$ on held-out synthesis PSNR before fixing $\tau = 0.5$ in any paper submission**.
- **LUMIERE longitudinal caveat.** Salmani's variance-optimality assumes within-cohort gradient variance ≈ proportional to cohort size. LUMIERE's 91 patients × ~7 scans/patient violate this (within-patient correlation drives the effective sample size below the scan count). VENA's two-stage uniform draw (patient-then-scan) partially compensates by treating each LUMIERE patient as one cohort unit, but the temperature exponent itself was derived under an i.i.d. assumption that does not hold for LUMIERE. Worth re-deriving the optimal $\tau$ for the longitudinal case, or running the ablation above.

**Force-distinct-cohort batch policy:**

- **Yu et al. (2020) PCGrad** *Gradient Surgery for Multi-Task Learning*. NeurIPS 2020, arXiv:2001.06782. Within-batch task diversity is necessary for detecting and resolving gradient conflicts between cohorts. VENA does not implement PCGrad, but the policy is consistent with it.
- **Zhao et al. (2020)** *Training Confounder-Free Deep Learning Models for Medical Applications*. Nature Comms 11:6010. DOI 10.1038/s41467-020-19784-9. Balanced within-batch site representation directly reduces site-confounder leakage in brain MRI — closest medical-imaging precedent for our policy.
- **Zhao & Feng (2025)** *Validation of FL Strategies for Multi-Contrast MRI Synthesis*. bioRxiv 2025.02.09.637305. Closest task analogue (multi-institutional brain MRI synthesis). Finds FedBN > FedAvg because per-batch BN statistics from a single dominant cohort degrade cross-cohort synthesis — directly relevant to VENA's policy of mixing cohort-specific BN statistics within each batch.

**Patient-then-scan vs scan-uniform:**

The patient-then-scan two-stage draw (instead of scan-uniform) is the correct call for the longitudinal cohorts: it gives each LUMIERE patient an equal voice rather than letting a single patient's seven sessions dominate the cohort's contribution to the loss. There is no direct paper on this in multilingual NLP (sentences ≠ documents distinction is not made), but the practitioner consensus in longitudinal medical imaging (Zhao 2020 above; nnU-Net's uniform case sampling — Isensee et al. 2021 Nature Methods 18:203 — although nnU-Net does not have multi-site batch balancing at all) is patient-level first.

**Summary verdict** — $\tau = 0.5$ + force-distinct-cohort + patient-then-scan is a well-supported default with two open follow-ups: (i) the τ ablation, and (ii) if (i) shows τ-instability, the longitudinal-aware variant of the variance-optimal exponent.

---

## Useful related notes

* [[cohort_dedup_incident]] — first-time dedup playbook (the audit trail).
* [[reference-icai-server]] — running smoke tests on server3 (dual
  `vena/` ↔ `src/vena/` layout caveat).
* [[project-cohort-dedup]] — current dedup state (2026-06-02).
* [[project-s2-smoke-validated]] — S2 + Lp-contrastive smoke baseline.
* `.claude/notes/data/ivygap.md`, `lumiere.md`, `brats_africa.md` — deeper
  per-cohort notes.

## Open follow-ups

1. ~~**REMBRANDT converter** does not write `splits/cv/fold_0/{train,val}`
   natively~~ — **CLOSED 2026-06-19**: post the Phase-6 schema unification
   the latent H5 carries `splits/cv/fold_0/*` natively and the image H5 was
   patched in-place via `h5py copy` from the latent (same patient sets;
   verified `splits/cv/fold_0/train n=53`, `splits/cv/fold_0/val n=5`).
   Converter still emits flat splits as the source-of-truth for REMBRANDT;
   a longer-term fix would add `splits/cv/fold_0/*` natively at conversion
   time so future re-runs do not need the copy step.
2. **IvyGAP bridge file** — 21 unresolvable xlsx overlaps; supply an
   external bridge to actually dedup IvyGAP against BraTS-GLI.
3. **Málaga in-house cohort** — multi-vendor + SWAN/SWI (the
   vessel-prior input the proposal needs); `role: external`, expected later
   in 2026. Until then VENA conditions on an empty vessel map at training
   time and the proposal §6 external-validation pipeline cannot run on real
   SWAN data.
4. **No SWAN anywhere yet** — all 9 current cohorts have
   `has_swan = false`. The vessel-aware conditioning branch trains on a
   constant zero-tensor placeholder; the discriminator design hinges on
   this being a fast no-op when the modality is absent.
5. **BraTS-Africa "z-score" cohort flag is heuristic, not authoritative.**
   The image H5 has no `intensity_policy` root attr and per-scan empirical
   stats vary (most scans look raw with `mean ≈ 3000`; a minority have
   `min < 0` with 0.0% intra-brain negatives). The
   `percentile_use_brain_mask=True` default in the encoder is a safety net;
   any future cohort that ships z-score / standardised intensities should
   set the same flag explicitly.
