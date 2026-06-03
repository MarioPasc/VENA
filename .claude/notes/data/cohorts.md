# VENA Training Cohort Registry — 2026-06-03

Single source of truth for the multi-cohort corpus VENA trains and evaluates
on. Per-cohort markdown notes (e.g. `ivygap.md`, `lumiere.md`,
`brats_africa.md`) hold deeper detail; this file is the *index*.

The numbers below match the deduplicated `corpus_local.json` /
`corpus_server3.json` / `corpus_picasso.json` (schema 1.0.0). Patient counts
are post-conversion (skipped patients dropped); `n_kept` is the number
surviving the `cohort_dedup` preflight.

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
* **Brain mask**: derived in the converter as the union of nonzero voxels
  across the four modalities (background is exactly 0 post HD-BET / CBICA
  skull-strip).
* **Latent shape** (MAISI-V2 VAE): `(C=4, H=48, W=56, D=48)`; 4× spatial
  compression of the 192×224×192 brain box.
* **Intensity policy**: H5 stores native scanner intensities; per-modality
  percentile normalisation `[0, 99.95]` with `foreground_only=true` runs at
  encode time. Cross-cohort latent comparability requires every cohort to
  share these settings — change only in lockstep across `corpus_*.json`.

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

## Useful related notes

* [[cohort_dedup_incident]] — first-time dedup playbook (the audit trail).
* [[reference-icai-server]] — running smoke tests on server3 (dual
  `vena/` ↔ `src/vena/` layout caveat).
* [[project-cohort-dedup]] — current dedup state (2026-06-02).
* [[project-s2-smoke-validated]] — S2 + Lp-contrastive smoke baseline.
* `.claude/notes/data/ivygap.md`, `lumiere.md`, `brats_africa.md` — deeper
  per-cohort notes.

## Open follow-ups

1. **REMBRANDT converter** does not write `splits/cv/fold_0/{train,val}`
   natively — the trainer requires that path. Currently patched in-place on
   the latent H5; needs a converter fix (see task #21 in current session).
2. **IvyGAP bridge file** — 21 unresolvable xlsx overlaps; supply an
   external bridge to actually dedup IvyGAP against BraTS-GLI.
3. **Málaga in-house cohort** — multi-vendor + SWAN/SWI (the
   vessel-prior input the proposal needs); `role: external`, expected later
   in 2026. Until then VENA conditions on an empty vessel map at training
   time and the proposal §6 external-validation pipeline cannot run on real
   SWAN data.
4. **No SWAN anywhere yet** — all 8 current cohorts have
   `has_swan = false`. The vessel-aware conditioning branch trains on a
   constant zero-tensor placeholder; the discriminator design hinges on
   this being a fast no-op when the modality is absent.
