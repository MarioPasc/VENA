# VENA Validation Proposal — Competitor Benchmarking and Evaluation Protocol

*Mario Pascual González, 2026-06-08; revised 2026-06-17. Target venues:
Medical Image Analysis, IEEE TMI, MICCAI 2026 main conference.*

Companion to [`literature.md`](literature.md) (competitor selection rationale,
§11b) and the project proposal at
`/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` (model
specification). This document pins **how** we measure success, **which**
competitors we measure against, **on which data**, and **how we test for
statistical and clinical significance**.

The 2026-06-17 revision: (i) drops the vessel-segmentation-operator
ensemble (Frangi/Jerman/OOF/VesselFM) entirely from the protocol and
replaces it with a label-free **spatial residual analysis** that tests
the vessel-fidelity claim directly from the intensity residual map;
(ii) demotes the perceptual family (LPIPS-3D, RadImageNet-LPIPS) — both
are 2D feature extractors with no validated 3D extension; (iii) adds an
explicit **intensity-harmonisation prerequisite** (§4.1) so paired
metrics are comparable across methods; (iv) adds a new
**§5 operational protocol** pinning checkpoint selection, inference
timing, and the per-method predictions H5 schema; (v) makes the 2D→3D
inference protocol for the slice-wise competitors explicit (§2 rule 10)
and adds an inter-slice consistency diagnostic (§4.7); (vi) **adds
C6 (3D-Latent-DDPM) and C7 (3D-Latent-Pix2Pix)** to the mandatory
tier so the 3D-latent quartet C4–C7 reproduces in full the four-method
3D benchmark T1C-RFlow themselves published (Eidex et al. 2025 §4),
holding the MAISI U-Net trunk and MAISI-family VAE fixed and varying
only the generative formulation across C4–C7 (transformer-diffusion /
RFlow / pure-DDPM / conditional-GAN). The competitor matrix is now
**seven external competitors + one internal ablation (A1-VENA-S1) +
the C0 identity null floor = ten methods** in the predictions H5
inventory; statistical comparisons are over the eight non-VENA rows.

---

## 1. Design principles

A validation protocol for a high-impact methodological paper has to defend
four positions:

**P1 — Paired ground truth dictates paired metrics.** The task is
$\{T_{1\text{pre}}, T_2, \text{FLAIR}\} \to T_{1c}$, where the real $T_{1c}$
is on disk for every test scan. The relevant question is *"how close is
the predicted $\widehat{T_{1c}}$ to the ground-truth $T_{1c}$, voxel- and
structure-wise?"*, not *"is the distribution of predictions
distributionally close to the distribution of real images?"*. Distributional
metrics (FID, KID) answer the second question and are appropriate for
*unconditional* generation; they are demoted to *secondary plausibility
checks* here, never primary endpoints (justification in §7).
Reinke et al. (2024) *Common limitations of image processing metrics: A
picture story*, **Nature Methods** 21(2):182–194,
DOI:10.1038/s41592-023-02151-z, frames this distinction explicitly:
"distribution-level metrics are a complement to, not a substitute for,
sample-level metrics when the task is paired."

**P2 — Out-of-distribution is tested by construction, not by hope.** Four
distribution shifts are *built into* the test set (geographic, age, vendor,
pathology). Each shift is evaluated separately so the failure-mode is
attributable.

**P3 — Statistical conclusions require a pre-registered analysis plan.** We
specify the primary endpoint, the family of comparisons, the per-family
multiplicity correction, the test statistic, and the effect-size estimator
*before* unblinding the test set. Maier-Hein et al. (2018) *Why rankings of
biomedical image analysis competitions should be interpreted with care*,
**Nature Communications** 9:5217, DOI:10.1038/s41467-018-07619-7, is the
canonical reference: rankings flip under reasonable perturbations of the
analysis plan unless the plan is fixed in advance.

**P4 — Counterfactual evidence is required to defend the synthesis claim.**
The model must (a) not hallucinate enhancement on healthy controls,
(b) not hallucinate enhancement on non-enhancing gliomas, and (c) lose
enhancement when the WT-mask conditioning is removed (the §6.5 shortcut
diagnostic from the proposal). These three counterfactuals are mandatory
before submission.

---

## 2. Competitor selection

Seven external competitors + one internal controlled comparator
(VENA-S1). The mandatory tier was revised on 2026-06-08 after dropping
CFM (no public code) and TumorFlow (different task scope — longitudinal
physics-guided growth, not paired CE translation); CFM's slot in the
FM/diffusion tier is taken by 3D-DiT, and TumorFlow's role as
"no-contrastive controlled comparator" is replaced by VENA-S1 (a
strictly tighter ablation because no architectural delta exists). The
2026-06-17 revision added the **3D-Latent-DDPM (C6)** and
**3D-Latent-Pix2Pix (C7)** competitors so the 3D-latent quartet
(C4–C7) reproduces in full the four-method 3D benchmark T1C-RFlow
itself published (Eidex et al. 2025 §4: DiT-3D, DDPM, Pix2Pix, RFlow)
— see `literature.md` §11b for the framing argument. Re-listed here
with the *training protocol* — what is held fixed when re-training
each on our cohort split:

| Tag | Method | Reference | Code | Re-train inputs | 2D / 3D | Wall-clock (4× A100 est.) |
|---|---|---|---|---|---|---|
| **C0-Identity** | $\widehat{T_{1c}} \equiv T_{1\text{pre}}$ | — | — | T1pre → T1c (no learning) | n/a | 0 |
| **C1-pGAN** | pGAN (paired) | Dar et al. 2019, IEEE TMI | icon-lab/pGAN-cGAN | T1pre, T2, FLAIR → T1c | 2D axial (slice-stacked) | ~1 d |
| **C2-ResViT** | ResViT (paired) | Dalmaz et al. 2022, IEEE TMI | icon-lab/ResViT | T1pre, T2, FLAIR → T1c | 2D axial (slice-stacked) | ~2 d |
| **C3-SynDiff** | SynDiff (paired) | Özbey et al. 2023, IEEE TMI | icon-lab/SynDiff | T1pre, T2, FLAIR → T1c | 2D axial (slice-stacked) | ~3 d |
| **C4-3D-DiT** | Latent diffusion with DiT backbone | Peebles & Xie 2023; Mo et al. 2023; **3D latent adaptation per Eidex 2025** (`dit3d.py / dit3d_wrapper.py` in T1C-RFlow repo) | T1C-RFlow repo | T1pre, T2, FLAIR → T1c | 3D MAISI latent | ~4 d |
| **C5-T1C-RFlow** | T1C-RFlow | Eidex et al. 2025, arXiv:2509.24194 | github.com/zacheidex/... | T1pre, T2, FLAIR → T1c | 3D MAISI latent (custom retrained VAE) | ~4 d |
| **C6-3D-Latent-DDPM** | Pure conditional DDPM in 3D latents | Ho, Jain & Abbeel 2020 DDPM formulation; **3D latent adaptation per Eidex 2025** (`train_ddpm.py` in T1C-RFlow repo) | T1C-RFlow repo | T1pre, T2, FLAIR → T1c | 3D MAISI latent (same MAISI U-Net trunk as C5) | ~2.5 d |
| **C7-3D-Latent-Pix2Pix** | Conditional GAN in 3D latents | Isola et al. 2017 Pix2Pix formulation; **3D latent adaptation per Eidex 2025** (`train_pix2pix_*.py` in T1C-RFlow repo) | T1C-RFlow repo | T1pre, T2, FLAIR → T1c | 3D MAISI latent | ~2 d |
| **A1-VENA-S1*** | VENA's own CFM-only checkpoint | this work, `picasso_s1_1000ep.yaml` | this repo | T1pre, T2, FLAIR → T1c | 3D MAISI latent | already trained |

*A1 = "ablation 1", not a competitor. Same MAISI VAE + NV-Generate-MR
trunk + ControlNet + ControlNet conditioning + training cohort + epoch
budget as VENA; the *only* delta is the $L^p$-aware contrastive term in
the loss. Any S2-vs-S1 gap on the §4.3 spatial-residual metrics is
therefore attributable to the contrastive term, isolating the headline
contribution under a zero-architectural-delta ablation.

**Specifically excluded** (with reasoning):
- **CFM** (Chang et al., MICCAI 2025) — no public code or weights;
  re-implementation cost dominates the marginal value.
- **TumorFlow** (Biller et al. 2026) — the published task is longitudinal
  physics-guided glioblastoma growth, not paired CE translation. Cited as
  the architectural precedent that established the MAISI+ControlNet+RF
  scaffold we adopt (see `literature.md` §5).
- **Pix2Pix-DDPM / Palette-Med** — methodologically subsumed by C3
  (SynDiff = paired conditional diffusion + adversarial bridge; the
  adversarial term is a *strict addition* to pure conditional DDPM).
  Inclusion would inflate compute by ~4 d A100 for marginal architectural
  diversity. The discussion will note the omission with this argument.

C4-3D-DiT replaces C4-CFM in the diffusion/FM tier because (i) T1C-RFlow
themselves use DiT-3D as a published baseline (Eidex et al. 2025 §4), so
reviewers expect it; (ii) it tests the *transformer-backbone diffusion*
axis that T1C-RFlow's U-Net does not.

**Training rules pinned for every competitor.**

1. **Same training cohort** as VENA: 1,224 patients / 1,664 scans pooled
   across the 6 CV cohorts at fold 0 (`corpus_picasso.json`,
   `splits/cv/fold_0/train`). No competitor sees a single test-partition
   patient.
2. **Same intensity normalisation upstream of the model**: percentile
   clipping $[0, 99.95]$ inside the brain mask, identical to VENA's encode
   step. Per-method internal preprocessing (e.g. icon-lab's z-score) is
   layered on top of this shared normalised volume so the competitor's
   architecture sees the contrast it was trained on, but the source
   intensities are pinned. *Downstream* harmonisation of the predicted
   $\widehat{T_{1c}}$ to a common output scale is §4.1; this rule pins the
   *input* side.
3. **Same skull-stripping**: the HD-BET / CBICA brain mask shipped with each
   cohort H5 is used. No competitor re-runs skull-stripping.
4. **Same compute budget**: each competitor runs for the wall-clock above
   *or* until validation loss plateaus, whichever comes first. Early
   stopping is enabled with patience 100 epochs on val MAE (paired). This
   guards against unfair under-training of competitors but caps the GPU
   bill.
5. **No oracle test-set tuning**: hyperparameter selection uses
   `splits/cv/fold_0/val` only. Test partitions are unblinded once, at the
   end, for the final table.
6. **Released weights as supplementary check**: when authors' weights are
   public, we additionally evaluate them on our test partitions
   *without* re-training, and report the gap. This is a separate
   supplementary table — never the primary comparison.
7. **Pretraining is method-faithful, not compute-matched.** Each
   competitor is trained the way its authors trained it, *including*
   any pretraining recipe the original paper used. VENA's
   NV-Generate-MR warm-start is therefore *not* matched on the
   competitor side — pGAN, ResViT, SynDiff train from scratch as in
   their original papers; T1C-RFlow uses its custom VAE; 3D-DiT (C4)
   is the *only* exception, trained over the same frozen MAISI VAE as
   VENA so the VAE-choice confound is controlled. The discussion must
   state explicitly that any VENA-vs-competitor gap conflates
   (a) the $L^p$-contrastive loss, (b) the MAISI+NV-Generate-MR
   foundation-model warm-start, and (c) the latent-FM formulation. The
   internal A1-VENA-S1 ablation disentangles (a) from (b)+(c); the
   proposal's H1 trunk-init ablation (§8) disentangles (b) from (c).
8. **Sampler equivalence at the patient level.** Every competitor's
   data loader respects the same patient-then-scan two-stage draw with
   $\tau = 0.5$ temperature-balanced cohort sampling that VENA uses
   (per `cohorts.md` §"Batch composition strategy"). For the 2D
   slice-wise competitors (C1, C2, C3), this means *patient* sampling
   under $\tau = 0.5$, then a random axial slice from the sampled
   patient — not a flat slice loader. Without this rule, the
   longitudinal LUMIERE patients (~7 sessions × ~155 slices each)
   would dominate any flat-slice 2D loader, advantaging or
   disadvantaging the 2D tier in a way unrelated to architecture.
9. **Hyperparameters: published defaults + a small val sweep.** Each
   competitor's HPs start at the authors' published values. We run a
   *small* sweep (≤ 5 configs, ≤ 1 A100-day per competitor) over the
   2–3 HPs the original paper reports as most sensitive (typically
   learning rate, batch size, number of timesteps for diffusion-tier
   competitors). The sweep is on `splits/cv/fold_0/val`; test partitions
   are unblinded once. Light HP discipline prevents the "they
   under-trained the competitors" reviewer comment and the "they
   over-tuned the competitors into the ground" comment simultaneously.
10. **2D→3D inference protocol for slice-wise competitors (C1, C2, C3).**
    pGAN, ResViT, and SynDiff are 2D axial models. To produce a 3D
    $\widehat{T_{1c}}$ volume, each is applied **slice-by-slice along
    the axial axis** (the orientation in which they were trained), then
    the per-slice predictions are stacked into a 3D volume with the
    shape of the real $T_{1c}$. **No** inter-slice smoothing, post-hoc
    3D refinement, sagittal/coronal re-application, or 2.5D
    multi-view fusion is performed — any such treatment is a *model*
    modification and would not be a faithful re-implementation of the
    published method. Implication: 3D-aggregate metrics (PSNR-3D,
    SSIM-3D, MS-SSIM-3D) on the slice-stacked output are penalised by
    *inter-slice discontinuity* (banding along z), which is a real
    architectural property of the 2D tier and not an evaluation bug.
    We attribute the contribution of this artefact via the z-gradient
    diagnostic in §4.7, and we also report per-slice 2D metrics (mean
    across axial slices) for C1–C3 as a supplementary table so the
    reader can distinguish "lost on per-slice quality" from "lost on
    slice-stacking incoherence". C4 (3D-DiT), C5 (T1C-RFlow), C6
    (3D-Latent-DDPM), C7 (3D-Latent-Pix2Pix), VENA, and A1-VENA-S1 are
    3D-native and not subject to this artefact.

C0 (identity) is the **null-model floor**. Any non-trivial method must
beat C0 on every region-restricted metric; if a method does not beat C0
inside the WT, the model has failed at the *one* thing it is supposed to
do (synthesise enhancement).

**Caveat on competitor specificity (must be reflected in the paper's
discussion).** Of the seven external competitors:

- **C5 (T1C-RFlow)** is the only one that targets T1c synthesis as its
  headline task — it defines the current numerical SOTA. A win over C5
  on the primary endpoint *is* the SOTA claim.
- **C4 (3D-DiT), C6 (3D-Latent-DDPM), C7 (3D-Latent-Pix2Pix)** are the
  remaining three rows of *T1C-RFlow's own four-method 3D benchmark*
  (Eidex et al. 2025 §4: DiT-3D / DDPM / Pix2Pix / RFlow). They are
  generic 3D latent generative architectures, not T1c-specific, and
  they were released by Eidex et al. precisely as the comparison
  population the RFlow main result beat. By reproducing this full
  quartet on our cohort split, we make the SOTA comparison head-to-head
  on T1C-RFlow's own terms.
- **C1, C2, C3** do *not* target T1c synthesis — they are *generic*
  paired cross-contrast MR translation methods (pGAN, ResViT, SynDiff)
  re-purposed for the T1pre+T2+FLAIR→T1c direction. They are included
  because reviewers at MedIA / IEEE TMI consistently expect the icon-lab
  baseline triad — a methods paper that does not beat the strongest
  *general* cross-contrast architecture is dismissable as "just a task
  paper". *Architectural* baselines, not *task* baselines.

The discussion must report architectural-baseline wins (C1/C2/C3 — 2D
generic cross-contrast; C4/C6/C7 — 3D-latent generic-formulation
baselines from T1C-RFlow's benchmark) and task-baseline wins (C5)
**separately**: a small win over C5 is what defines SOTA on the task;
wins over C1–C4 and C6–C7 are what define a methods contribution that
generalises beyond the task. **The C4–C7 sub-table is the
generative-formulation isolation experiment**: same MAISI U-Net trunk
class wherever applicable, same MAISI-family VAE, same 3D latent
grid — *only the generative formulation differs* (transformer-diffusion
/ U-Net rectified flow / U-Net pure DDPM / U-Net conditional GAN). The
A1-VENA-S1 ablation isolates the $L^p$-contrastive contribution at
zero architectural delta against C5.

C1–C3 are re-trained with the T1pre+T2+FLAIR→T1c direction selected
from their many-to-one (ResViT, SynDiff) or many-to-many (pGAN-cGAN)
APIs. C4-3D-DiT, C6-3D-Latent-DDPM, and C7-3D-Latent-Pix2Pix are
trained over MAISI latents to match VENA's latent space (controls for
the VAE-choice confound). The training pairs are identical to VENA's.

---

## 3. Test-cohort partitioning

Three concentric rings. Each ring tests a distinct distribution shift.

### Ring A — In-distribution (ID) test

Held-out **`splits/test`** partitions of the six CV cohorts.

| Cohort | n patients | n scans | Stratification within ID |
|---|---:|---:|---|
| UCSF-PDGM | 21 | 21 | single-vendor (GE 3T), adult, preoperative glioma |
| BraTS-GLI | 114 | 127 | multi-vendor, multi-institution, adult, preoperative |
| IvyGAP | 5 | 5 | adult GBM |
| LUMIERE | 11 | 72 | longitudinal (treated GBM at multiple timepoints — *partially ID* because post-treatment imaging deviates from preoperative training distribution) |
| REMBRANDT | 5 | 5 | mixed adult glioma; CBICA-mirror |
| UPENN-GBM | 17 | 17 | adult GBM, BraTS-2021 implicit subset (deduplicated against BraTS-GLI) |
| **Total** | **173** | **253** | |

Ring A is the *primary endpoint*. The headline table in the paper is
the eight competitors (C0–C7 + A1-VENA-S1, minus C0 from the
multiplicity family but reported in the table as the null floor) × VENA
on Ring-A pooled (with per-cohort breakdown in supplementary).

### Ring B — Demographic / geographic OOD

All three `role=test_only` cohorts: BraTS-Africa-Glioma (95), BraTS-Africa-
Other (51 — non-glioma, *pathology* OOD on top of geographic OOD),
BraTS-PED (260 pediatric).

Ring B answers *"does the model generalise beyond the multi-vendor adult
Western glioma distribution it was trained on?"*. Per Adewole et al.
(2023, arXiv:2305.19369) on BraTS-Africa and Kazerooni et al. (2023,
arXiv:2305.17033) on BraTS-PED, the scanner field strengths, contrast
protocols, and patient anatomies in these cohorts differ substantially from
the adult-Western training pool — Adewole et al. specifically demonstrate
that BraTS-2023 winning models lose 5–15 Dice points on Africa data.

### Ring C — Acquisition / vendor OOD

The HRUM (Hospital Universitario Regional de Málaga) in-house cohort,
**pending data-sharing agreement and IRB pseudonymisation**
(`role=external`, expected 2026-Q3 — per `cohorts.md` open follow-up #3
and proposal §2.3). Tier specification (revised 2026-06-08 — the previous
SWAN-conditioned Tier 3 was dropped from this paper's scope; VENA in this
formulation **does not consume SWAN as an input**):

- **Tier 1** (mandatory T1pre, T1c, T2, FLAIR — preoperative glioma and
  meningioma). This is the vendor-OOD primary endpoint for Ring C.
- **Tier 2** (multi-vendor: Siemens / Philips alongside GE; non-enhancing
  glioma as a counterfactual — see §8.2).
- **Tier 3** (optional, exploratory: longitudinal follow-up timepoints).
  Reported as supplementary; not part of the primary or secondary endpoint
  family.

A SWAN-conditioned follow-up paper is *parked* for after HRUM Tier 3
lands — that variant requires re-training and is out of scope here. The
vessel-fidelity claim in this submission is purely loss-side (the $p_b=3$
background term of the $L^p$-aware contrastive upweights non-tumour
errors); it is testable on every ring through the spatial residual
analysis in §4.3, without ever needing SWAN as input and without any
vessel-segmentation operator.

### Pre-registration

The patient identifiers in each ring are frozen on submission day and
written, as plain JSON, to
`artifacts/validation/<UTC-timestamp>/ring_partitions.json`.
The hash of that file is written into every competitor's
`decision.json` so the analysis cannot retroactively choose its test set.

---

## 4. Metric suite

Seven subsections, organised so that §4.1 must be applied before any
metric in §4.2–§4.7 — without a common output intensity scale, the
voxel-wise metrics across competitors are not comparable.

### 4.1 Intensity harmonisation (mandatory pre-processing)

Each competitor emits $\widehat{T_{1c}}$ in its own intensity convention:
pGAN / ResViT / SynDiff produce z-scored axial slabs that are re-mapped
to a positive scale per the icon-lab inference code; T1C-RFlow and VENA
decode through a frozen VAE whose latent-space scaling differs from the
input image scale; 3D-DiT inherits VENA's VAE intensity contract.
Without downstream harmonisation, an absolute-error metric (MAE) would
trivially favour whichever method's default output scale happens to align
with the target — Reinke et al. (2024 §3.4) flag this as the
**data-range confound**, the same trap that silently inflated PSNR/SSIM
across the synthesis literature.

The goal of harmonisation is **two-fold**:
1. Place each method's $\widehat{T_{1c}}$ in the *same dynamic range* as
   the input $T_{1\text{pre}}, T_2, \text{FLAIR}$ sequences of that scan
   (the volumes the model conditioned on), so the prediction sits in the
   same scale as the inputs it derives from.
2. Place every method's $\widehat{T_{1c}}$ in the *same dynamic range
   across methods* on a given scan, so PSNR/SSIM/MAE numbers are
   commensurable between architectures.

**Protocol (applied once per prediction, before any §4.2 metric).** For
every method's $\widehat{T_{1c}}$ output volume on every scan:

1. Restrict to the brain mask (HD-BET / CBICA mask shipped with each
   cohort H5).
2. Compute the per-volume percentile range
   $[p_{0.5}, p_{99.5}]$ inside the brain foreground.
3. Clip to $[p_{0.5}, p_{99.5}]$, then linearly rescale to $[0, 1]$.
   Background voxels (outside the brain mask) are forced to 0.

This is exactly the `percentile_normalise(lower=0.5, upper=99.5,
foreground_only=True)` contract VENA applies to every encoded sequence
during training (per `.claude/rules/model-coding-standards.md` rule 15
and `multi-cohort training and encoding always use foreground_only=True
because all stored volumes are skull-stripped`). Applying the *same*
contract to (a) the input $T_{1\text{pre}}, T_2, \text{FLAIR}$
references, (b) the real $T_{1c}$, and (c) every method's
$\widehat{T_{1c}}$ harmonises all six volumes per scan to a single
$[0, 1]$ scale.

**Cross-method intensity audit (mandatory in supplementary).** After
harmonisation, report a per-method × per-cohort table (Table S1) of the
post-harmonisation distribution moments — mean, std, 1st / 50th / 99th
percentiles — across Ring A scans. Diagnostic rules:

- Cross-method spread of *mean* post-harmonisation intensity within a
  cohort must be < 0.05 on the $[0, 1]$ scale; a method whose mean
  drifts further has retained a residual scale artefact and the
  harmonisation pipeline is re-derived for it (typically: a different
  brain-mask convention, leakage of background voxels into the
  percentile estimate).
- The *real $T_{1c}$* row is the reference; the report shows each
  method's per-moment deviation from it. A method that has its 99th
  percentile systematically lower than the real $T_{1c}$ is producing
  under-saturated enhancement — a *real* failure mode that we report
  alongside MAE rather than absorb into the harmonisation.

*Why percentile-normalisation over alternatives.* Nyúl-Udupa histogram
standardisation (Nyúl & Udupa 1999, *Magn. Reson. Med.* 42(6):1072–1081,
DOI: 10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M)
fits a piecewise-linear deformation between scan histograms; applying it
*per method* would *absorb* the model's intensity-domain failure mode
into the fitted transform and hide the bug we want to measure. A
per-scan affine fit (slope-intercept regression of $\widehat{T_{1c}}$
onto $T_{1c}$) would do the same — and also invite the "you fitted to
the answer" reviewer objection. Per-volume percentile clipping is the
contract VENA was trained against; applying it identically to every
method preserves the harmonisation goal without masking systematic
intensity-domain errors.

### 4.2 Primary — paired voxel-wise fidelity (per region)

All metrics computed in **image space**, on the percentile-normalised
intensity volumes from §4.1, with `data_range=1.0` fixed everywhere
(Reinke et al. 2024 §3.2). Three region restrictions per metric:

- **brain** (HD-BET / CBICA mask).
- **WT** (whole-tumour mask, BraTS labels >0).
- **brain ∖ WT** (background; the region the $L^p$-contrastive
  specifically targets — H2/H3 of the proposal).

Four metrics, all reported per region:

- **MAE** — direct intensity error, primary ranking statistic per Reinke
  et al. (2024 §3.1; MSE is dominated by the residual heavy tail; MAE is
  the robust choice).
- **PSNR-3D** — reported alongside SSIM-3D, never alone.
- **SSIM-3D** — `monai.metrics.SSIMMetric` with $k_1 = 0.01$, $k_2 =
  0.03$, Gaussian window 11³, `data_range=1.0` (Wang et al. 2004, DOI:
  10.1109/TIP.2003.819861).
- **MS-SSIM-3D** — multi-scale, four levels, weights $[0.0448, 0.2856,
  0.3001, 0.3633]$ (Wang et al. 2003), correlates better than
  single-scale SSIM with radiologist quality scores on MR synthesis
  (Pamb et al. 2024, **IEEE JBHI**).

**Primary endpoint of the paper**: **MAE on the *brain* region of Ring
A, VENA vs the median of the 8 competitors** (C0 through C7 inclusive
of the null-model floor — see §2 table; statistical apparatus in §6).

*Why 3D SSIM/MS-SSIM with the 2D competitors.* For C1–C3 (2D
axial-slice methods), 3D SSIM/MS-SSIM penalises the inter-slice
discontinuity that they architecturally produce (§4.7 quantifies the
artefact). This is correct: the deliverable to the radiologist is a 3D
volume, and a 3D metric must reflect the volume's structural
coherence. The 2D competitors are *also* reported under 2D slice-wise
PSNR/SSIM (mean across axial slices) as supplementary Table S3, so a
reader can attribute the gap to (a) per-slice prediction quality or
(b) slice-stacking inter-slice incoherence.

### 4.3 Spatial residual analysis — bright-region error concentration

**Replaces the previous draft's eight vessel-segmentation metrics
(M1–M8) entirely.** The Mallio et al. (2023, **Frontiers in
Neuroimaging** 2:1055463, DOI: 10.3389/fnimg.2023.1055463) failure-mode
review identifies *vessel under-enhancement* as the dominant
clinically-relevant failure of CE-MR-synthesis models. The previous draft
addressed it by running Frangi / Jerman / OOF / VesselFM on real and
synthetic $T_{1c}$ and reporting a multi-operator ensemble. That
approach was dropped for three substantive reasons:

1. **No cohort ships ground-truth vessel labels.** Every vessel mask in
   the previous draft was itself a model output. Using a model-derived
   mask as the "truth" makes operator noise a *confounder* of model
   evaluation; the analysis can no longer separate "did the synthesis
   model fail?" from "did the vessel operator fail?".

2. **Standard vessel-segmentation operators are out-of-domain on
   $T_{1c}$.** Hessian-based filters (Frangi 1998, Jerman 2016, OOF
   2008) need bright tubular structure on a dark background — the
   regime of *TOF-MRA* (time-of-flight MR angiography) or CE-MRA, where
   blood flow or angiographic contrast produces dedicated vessel
   conspicuity. Structural $T_{1c}$ at standard Gd dose shows partial,
   spatially-heterogeneous enhancement of vasculature alongside
   competing bright structures (choroid plexus, fat, calcified pineal,
   ring-enhancing tumour); Hessian operators systematically false-
   positive on these (Mallio et al. 2023 §2). Deep vessel-segmentation
   foundation models (VesselFM, Wittmann et al. 2024,
   arXiv:2411.17386) are trained on TOF-MRA / CT-angiography / OCTA /
   vEM corpora; the authors do not evaluate structural CE MRI and do
   not claim zero-shot transfer to it. A pre-protocol spot-check would
   *itself* be a noisy and contested exercise, with two of three
   plausible outcomes (fail, pass-with-reservations) leaving the
   analysis on questionable footing. **The fundamental constraint** is
   that the canonical input for cerebral vessel segmentation is
   TOF-MRA — which our cohorts do not have — and using $T_{1c}$ as a
   stand-in injects an unquantifiable physical-physics mismatch.

3. **Operator suites multiply the multiplicity burden** and shift the
   headline claim from "VENA preserves contrast uptake" to "VENA
   preserves Frangi-detectable contrast uptake" (or VesselFM-detectable,
   or any other ensemble-dependent thing). The claim becomes operator-
   conditional rather than anatomy-conditional.

We therefore test the *same hypothesis* — does the model
under-synthesise the bright contrast-uptake regions outside the
tumour? — using only the intensity residual map and the real $T_{1c}$
itself. No auxiliary segmenter, no atlas, no operator-choice ensemble.

#### 4.3.1 Setup

Define the per-scan residual map
$r(x) = T_{1c}(x) - \widehat{T_{1c}}(x)$
after the §4.1 harmonisation. The residual is stored per scan in the
predictions H5 (§5.3) and reused by every analysis below — one decode
pass per scan, no re-computation.

**Two conditions** are reported, with different interpretations:

- **C-WB** (whole brain): residual analysed over the brain mask. Tests
  *global contrast-uptake fidelity*: the model is penalised for
  under-synthesising bright regions of any origin, including the
  enhancing tumour. This is the headline single-number test of "does
  the residual concentrate where the real $T_{1c}$ is brightest?".
- **C-noT** (background, tumour excluded): residual analysed over
  $R_{\text{bg}} = \text{brain} \setminus \mathrm{dilate}(M_{\text{WT}},
  k=5)$ — brain minus the 5-voxel-dilated tumour mask. Bright voxels
  inside $R_{\text{bg}}$ correspond exclusively to **vessels, dural
  venous sinuses, choroid plexus, pituitary, pineal gland, and dural
  enhancement** — the normally-enhancing anatomy of clinical interest.
  **C-noT is the vessel-fidelity claim of the paper.**

#### 4.3.2 Mathematical apparatus

The question "does $|r(x)|$ depend systematically on $T_{1c}(x)$ over a
region $R$?" is a *conditional-dependence* test between two scalar
fields on the same voxel grid. We report two complementary statistics
per scan, plus an intensity-stratified visualisation.

**S1 — Spearman rank correlation** $\rho_S(|r|, T_{1c})$ over $R$, per
scan. Yin & Carroll (1990) *Statistics & Probability Letters*
10(1):69–76, DOI: 10.1016/0167-7152(90)90114-M, establish Spearman ρ as
*the* heteroscedasticity diagnostic when the residual is non-Gaussian —
exactly our regime. Bishara & Hittner (2012) *Psychological Methods*
17(3):399–417, DOI: 10.1037/a0028087, show that under heavy-tailed
non-normal data (skew > 2, matching the bimodal-with-heavy-tail
distribution of $T_{1c}$ inside the brain), Pearson r inflates Type-I
error to ~20% at α = 0.05 while Spearman ρ retains nominal Type-I and
greater power. Spearman ρ is a single scalar per scan and aggregates
cleanly via paired Wilcoxon across patients per competitor. A large
positive $\rho_S$ means errors are systematically larger in bright
voxels of the real volume — the clinical failure mode.

**S2 — Top-q% bright-voxel error mass concentration ratio.** For
$q \in \{1\%, 5\%, 10\%\}$, define the top-$q$ bright set
$B_q = \{x \in R : T_{1c}(x) > Q^{1-q}_{\text{real}}\}$ where
$Q^{1-q}$ is the $(1-q)$-th intra-volume intensity quantile of
$T_{1c}$ over $R$. The *concentration ratio* is

$$
\mathrm{Conc}(q) = \frac{\sum_{x \in B_q} |r(x)|}{q \cdot \sum_{x \in R} |r(x)|}.
$$

Under spatial independence of $|r|$ and $T_{1c}$ over $R$,
$\mathbb{E}[\mathrm{Conc}(q)] = 1$. $\mathrm{Conc}(q) > 1$: errors
concentrate in bright voxels above chance. $\mathrm{Conc}(q) < 1$:
errors avoid bright voxels. This is a fixed-quantile point on the
Lorenz curve of the joint $(|r|, T_{1c})$ distribution; the precedent
in image analysis is the Gini-style morphology statistic of Abraham,
van den Bergh & Nair (2003) *Astrophysical Journal* 588:218–229, DOI:
10.1086/373919, which adopts the Lorenz/Gini machinery to characterise
pixel-ranked light distributions in unlabelled astronomical images —
the closest published analogue to our vessel-label-free setting.

$\mathrm{Conc}(5\%)$ is the most communicable single number: "the top
5% of background bright voxels carry $X\times$ their fair share of the
model's error". Dimensionless, scale-invariant, falsifiable against
$\mathrm{Conc} = 1$.

**S3 — Intensity-stratified residual visualisation (Bland-Altman
adaptation).** Partition $R$ into intensity deciles of $T_{1c}$; per
decile, plot mean $|r|$ and its 95% CI across Ring-A patients per
competitor. This is the discretised Bland-Altman plot (Bland & Altman
1986, **The Lancet** 327(8476):307–310, DOI:
10.1016/S0140-6736(86)90837-8; 1999, *Statistical Methods in Medical
Research* 8(2):135–160, DOI: 10.1177/096228029900800204) where the
$x$-axis bin is the $T_{1c}$ decile and the $y$-axis is the absolute
residual. The slope of the curve is the proportional-bias diagnostic
the 1999 extension formalises. Reported as one figure per Ring (Fig.
3a–c) overlaying VENA and the eight competitors.

**Exploratory: mutual information.** As a non-monotonic-dependence
companion, report MI($|r|$, $T_{1c}$) over $R$ using the
Kraskov-Stögbauer-Grassberger k-NN estimator (Kraskov, Stögbauer &
Grassberger 2004, **Phys. Rev. E** 69:066138, DOI:
10.1103/PhysRevE.69.066138; $k = 5$, the standard neuroscience MI
default). MI catches U-shaped or plateau-shaped failure modes that
Spearman ρ misses — for example, a model that fits parenchymal
mid-intensities but fails *both* at the dark CSF tail and the bright
vessel tail. Reported in supplementary only; not in the multiplicity
family.

#### 4.3.3 Reporting structure

- **Table 3 (in main text)** — per-method rows × per-region columns ×
  per-statistic blocks. Each cell contains $\mathrm{Conc}(5\%) \pm
  \mathrm{CI}_{95}$ and $\rho_S \pm \mathrm{CI}_{95}$. Bootstrap CI
  (10,000 patient-stratified resamples). Regions: C-WB (brain) and
  C-noT ($R_{\text{bg}}$). Methods: C0, C1, C2, C3, C4, C5, C6, C7,
  A1-VENA-S1, VENA. *Both stats reported side by side*, per your
  request — the reader can frame the claim with either.
- **Table S2 (supplementary)** — $\mathrm{Conc}(1\%)$,
  $\mathrm{Conc}(10\%)$, and the KSG MI for non-monotonic diagnostics.
- **Figure 3 (in main text)** — Spearman ρ as a heat-map on a
  *cohort × method* grid (one cell per Ring-A cohort, one column per
  method, colour = $\rho_S$ under C-noT). Companion correlation map for
  the reader.
- **Figure 4 (in main text)** — intensity-stratified residual plot
  (S3): mean $|r|$ per decile, overlaid across the seven methods, one
  panel per ring.

#### 4.3.4 Statistical apparatus

- Per competitor, **paired Wilcoxon** on per-patient
  $\mathrm{Conc}(5\%)$ under C-noT against the same statistic for
  VENA. Two-sided, α = 0.05.
- Per competitor, **paired Wilcoxon** on per-patient $\rho_S$ under
  C-noT against VENA's. Two-sided, α = 0.05.
- **Multiplicity inside the spatial-residual family**: Holm-Bonferroni
  over $2 \text{ stats} \times 8 \text{ competitors} = 16$ paired
  tests. This is a family separate from the §4.2 primary-endpoint
  family; per §6.3 it counts as a secondary endpoint family and does
  *not* inflate the §4.2 multiplicity, but it carries its own
  Holm-Bonferroni correction so neither S1 nor S2 reports inflated
  Type-I error.

#### 4.3.5 Defence against the "trivial ρ" reviewer attack

A reviewer may argue that a positive $\rho_S(|r|, T_{1c})$ is *trivially*
expected because the bright tail of $T_{1c}$ is anatomy-driven
(myelinated white matter, fat partial-volume, calcium, plus the
contrast-uptake we care about), so any model with imperfect
reconstruction at the bright tail produces a positive correlation.
Three defences, baked into the protocol:

1. **C-noT restricts $R$ to the brain minus a 5-voxel-dilated tumour
   mask.** This removes the dominant enhancement source and the bulk
   of partial-volume myelin contrast at the GM/WM boundary. What
   remains in the bright tail of $T_{1c}$ over $R_{\text{bg}}$ *is*
   the normally-enhancing anatomy and the cerebral vasculature.
2. **Per-scan intensity-shuffle null.** For each scan, shuffle
   $T_{1c}(x)$ values uniformly at random within the brain mask
   (breaking spatial correspondence with $|r|$ while preserving the
   marginal intensity distribution). Recompute S1 and S2 on the
   shuffled volume. The shuffle-null mean and 95% range establish the
   *expected* $\rho_S$ and $\mathrm{Conc}(5\%)$ under "any monotonic
   correlation with the bright tail is trivial". Report the
   *delta-to-shuffle* — observed minus shuffle-mean — as the primary
   effect size. A delta close to zero means the correlation *was*
   trivial; a delta significantly greater than zero is real signal.
   This is the standard spatial-null construction in functional
   neuroimaging (Alexander-Bloch et al. 2018, **NeuroImage**
   178:540–551, DOI: 10.1016/j.neuroimage.2018.05.070).
3. **C0-Identity row in Tables 3 and S2**: $\widehat{T_{1c}} \equiv
   T_{1\text{pre}}$ is the *no-synthesis* baseline. C0 by construction
   has $|r|$ exactly equal to $|T_{1c} - T_{1\text{pre}}|$, which is
   maximally concentrated in bright contrast-uptake regions by
   definition. VENA's S1/S2 must beat C0's; if VENA cannot, the model
   has added no enhancement information beyond identity.

A win on $\rho_S$ alone is dismissable. A win on $\mathrm{Conc}(5\%)$
under C-noT, against the intensity-shuffle null, with C0 as the upper
bound on bright-region error concentration, is not.

### 4.4 Secondary — task-based downstream segmentation

The previous draft's **perceptual family (LPIPS-3D + RadImageNet-LPIPS)
is demoted out of the metric suite entirely.** Reasons:

- **LPIPS-3D does not exist.** Zhang et al. (2018, **CVPR**, DOI:
  10.1109/CVPR.2018.00068) defined LPIPS as a 2D perceptual distance
  over AlexNet / VGG / SqueezeNet features extracted from individual
  RGB images. No 3D extension has been validated against radiologist
  agreement; the 2.5-D slab average that has crept into the synthesis
  literature is *not* a published metric and does not correlate
  reliably with clinical quality (Dohmen et al. 2025, *Scientific
  Reports*, arXiv:2405.08431 — "Similarity and Quality Metrics for MR
  Image-to-Image Translation" — finds LPIPS poorly aligned with
  clinical-quality judgement on MR synthesis).
- **RadImageNet ResNet-50 is 2D.** Mei et al. (2022, *Radiology AI*
  4(5):e210315) trained the RadImageNet backbones on 2D radiology
  slices. A 2.5-D slab approximation of RadImageNet-LPIPS inherits the
  2D / 3D inconsistency that disadvantages the 3D-native tier (C4, C5,
  C6, C7, VENA) against the 2D tier (C1, C2, C3), the opposite of the
  comparability the harmonisation in §4.1 aims for.

The perceptual-quality claim — to the extent it matters at all — is
absorbed into the reader study (§10) where the judgement is human, not
2D-feature-based.

**Task-based downstream segmentation is retained as the sole §4.4
metric** (the Preetha et al. 2021 protocol, **Lancet Digital Health**
3(12):e784–e794, DOI: 10.1016/S2589-7500(21)00205-3). Train an nnU-Net
(Isensee et al. 2021, **Nature Methods** 18:203–211, DOI:
10.1038/s41592-020-01008-z) on real $\{T_{1\text{pre}}, T_{1c}, T_2,
\text{FLAIR}\} \to$ BraTS labels using the train partitions of every
Ring-A cohort that has labels (UCSF-PDGM, BraTS-GLI, IvyGAP, LUMIERE,
REMBRANDT, UPENN-GBM). Evaluate on $\{T_{1\text{pre}},
\widehat{T_{1c}}, T_2, \text{FLAIR}\} \to$ predicted labels.

Report Dice for whole-tumour (WT), tumour-core (TC), and enhancing-
tumour (ET) sub-labels. **ET-Dice is the clinical-impact metric**: the
only sub-label whose accuracy directly depends on $\widehat{T_{1c}}$
quality (WT and TC use the T2/FLAIR/T1pre channels as well; ET depends
predominantly on the T1c contrast geometry). A model that wins on MAE
but loses on ET-Dice has produced contrast that *looks* right voxel-wise
but is *anatomically wrong* for the downstream task — the discussion
must call this out and the §9 failure-mode taxonomy must include the
qualitative example.

Appendix A (unchanged from the previous draft) explains why nnU-Net
rather than a BraTS-challenge winner is the right segmenter here.

### 4.5 Inference cost (mandatory for the FM/RF claim)

- **Wall-clock per volume** on a single A100 40 GB at NFE ∈ {1, 5, 20}
  for diffusion/FM-tier competitors (C3, C4, C5, C6, VENA, A1); fixed-cost
  per volume for the GAN-tier (C1, C2, C7). Mean ± std over 50 Ring-A
  volumes; measurement protocol in §5.2. **C6 (3D-Latent-DDPM) is
  expected to be the most expensive per-volume** because pure DDPM
  sampling at NFE = 20 still under-samples the standard 250- or
  1000-step DDPM schedule; we additionally report C6 at NFE = 50 and
  NFE = 250 in a supplementary table, since the C6-vs-C5 cost-quality
  curve is itself the published advantage of rectified flow over DDPM
  (Liu, Gong & Liu 2023; Eidex et al. 2025 §5).
- **Peak VRAM** at inference, same 50-volume run, GPU-resident peak
  from `torch.cuda.max_memory_allocated()` after warmup.
- **Number of forward-backward passes per training step** (1 for C0–C2
  and C7; 1 for C3–C6 modulo the diffusion/FM training-step cost;
  2 for VENA's contrastive paths). Reported for cost-quality framing
  in the discussion.

The proposal's <10 s/volume target inherits from Eidex et al. 2025
(T1C-RFlow); VENA must match it at NFE = 5.

### 4.6 Distributional plausibility (sanity check, not primary)

- **FID-3D** with RadImageNet features (2.5-D axial-slab approximation).
  *Reported, not used for ranking* — see §7 for the justification.
  Purpose: flag mode collapse or systematic over-/under-saturation in
  the synthetic volumes. A competitor that loses to VENA on paired MAE
  but matches it on FID is an *interesting* finding (we generate more
  realistic but less accurate samples) — worth supplementary
  discussion, never a headline.
- **KID-3D** — same purpose, less biased on small samples (Bińkowski
  et al. 2018, arXiv:1801.01401). Reported alongside FID.

### 4.7 2D→3D inter-slice consistency diagnostic

For C1, C2, C3 (and reported for context on C4, C5, C6, C7, VENA): per-volume
**mean absolute z-gradient** of the harmonised $\widehat{T_{1c}}$ over
the brain mask,

$$
\overline{|\partial_z I|} = \mathbb{E}_{x \in \text{brain}} \big|I(x, y, z+1) - I(x, y, z)\big|,
$$

and the same quantity on the real $T_{1c}$ for the same scan. The
**z-gradient discontinuity ratio** is

$$
\mathrm{ZGD} = \frac{\overline{|\partial_z \widehat{T_{1c}}|}}{\overline{|\partial_z T_{1c}|}}.
$$

$\mathrm{ZGD} > 1$: the synthesised volume has *more* inter-slice
variation than the real $T_{1c}$, attributable to per-slice prediction
noise that did not exist before stacking. $\mathrm{ZGD} \approx 1$: the
volume's z-axis statistics match the real one's. $\mathrm{ZGD} < 1$: the
volume is *over-smoothed* in z relative to the real volume.

For the 2D tier (C1, C2, C3) the diagnostic isolates the
slice-stacking artefact from per-slice quality, letting the discussion
write: "C1 lost 1.7 dB of PSNR-3D versus VENA; of that, 0.9 dB is
attributable to the inter-slice discontinuity quantified by
$\mathrm{ZGD} = 1.42$, and 0.8 dB to per-slice prediction quality
(supplementary Table S3 2D slice-wise comparison)". For the 3D-native
tier (C4, C5, C6, C7, VENA, A1-VENA-S1) $\mathrm{ZGD}$ should hover near 1 by construction;
a 3D-native method whose ZGD deviates has a different problem (z-axis
blurring or noise injection) than the slice-stacking failure mode.

Reported in Table S3 alongside the 2D slice-wise PSNR/SSIM for C1–C3.

---

## 5. Evaluation operational protocol

The protocol below pins how predictions are generated, stored, and
audited, so the statistical analysis in §6 operates on a frozen,
verifiable artefact set. Every decision below writes into the
per-method `decision.json` referenced in §11.

### 5.1 Best-checkpoint selection

Each method's "best checkpoint" is selected *before* test unblinding,
using `splits/cv/fold_0/val` only.

| Method | Selection criterion | Source-of-truth artefact |
|---|---|---|
| C0 — Identity | n/a (no training) | — |
| C1 — pGAN | minimum val MAE (paired, harmonised T1c) over the published-recipe schedule | `benchmarks/C1-pGAN/checkpoints/<run>/best_val_mae.pt` |
| C2 — ResViT | minimum val MAE (paired, harmonised) | `benchmarks/C2-ResViT/checkpoints/<run>/best_val_mae.pt` |
| C3 — SynDiff | minimum val MAE (paired, harmonised) measured at the authors' published sampling NFE | as above |
| C4 — 3D-DiT | minimum val MAE (paired, harmonised, VAE-decoded image-space) at NFE = 5 | as above |
| C5 — T1C-RFlow | minimum val MAE (paired, harmonised, VAE-decoded image-space) at NFE = 5 | as above |
| C6 — 3D-Latent-DDPM | minimum val MAE (paired, harmonised, VAE-decoded image-space) at NFE = 20 (the DDPM-tier inference budget reported in Eidex et al. 2025 §5) | `benchmarks/C6-3D-Latent-DDPM/checkpoints/<run>/best_val_mae.pt` |
| C7 — 3D-Latent-Pix2Pix | minimum val MAE (paired, harmonised, VAE-decoded image-space) at the single fixed-cost forward pass | `benchmarks/C7-3D-Latent-Pix2Pix/checkpoints/<run>/best_val_mae.pt` |
| A1 — VENA-S1 | exhaustive-val PSNR / SSIM curve (per `.claude/rules/model-coding-standards.md` §"Exhaustive validation") | `experiments/<run>/checkpoints/ema_best.ckpt` + audit |
| VENA (S2 / FFT / LoRA) | as A1 | as A1 |

The VENA / A1 asymmetry — train-loss-based `ema_best` plus an
exhaustive-val-curve audit — is required because VENA's in-process
validation is offloaded to the async second-GPU job per
`model-coding-standards.md` §"Async, second GPU". The chosen epoch's
`decision.json` records both the train-loss-best step and the
exhaustive-val-best step. **If they diverge by more than 100 epochs,
the exhaustive-val-best wins**, and that fact is documented in the
run's `decision.json`. The competitors do *not* face this asymmetry —
their authors' protocols use synchronous val.

**Pre-registration of checkpoint choice.** Once a method's best
checkpoint is selected on val, its SHA-256 is written into the
predictions H5 (§5.3) and the `decision.json`. The headline table is
computed *exactly once* per method, on the predictions generated by the
selected checkpoint. There is no re-selection on test data.

### 5.2 Inference protocol

- **Hardware**: a single A100 40 GB, sole occupancy. Inference jobs run
  on the same A100 hardware as training, via Picasso `--gres=gpu:1
  --constraint=dgx`. No mixed-hardware runs.
- **Warm-up**: every method receives 5 untimed forward passes before
  the timed run, model already on-GPU. This isolates kernel
  compilation, CUDA graph caching, and one-time cuDNN benchmarking
  from the wall-clock measurement.
- **Per-volume timing**: measured with
  `torch.cuda.synchronize() + time.perf_counter()` around the full
  inference path (input H5 read → ... → harmonised $\widehat{T_{1c}}$
  on CPU as a NumPy array). The timed region *includes* the
  pre-/post-processing the model needs at inference (encode for the
  latent-tier, harmonisation for everyone) and *excludes* output H5
  write.
- **NFE sweep**: diffusion/FM-tier competitors (C3, C4, C5, VENA, A1)
  are timed at NFE ∈ {1, 5, 20}. C6 (3D-Latent-DDPM) is additionally
  timed at NFE ∈ {50, 250} per §4.5. GAN-tier (C1, C2, C7) is timed
  at its fixed-cost path; the NFE column in Table 4 (inference cost)
  shows "—".
- **Sample size**: 50 Ring-A volumes per method, sampled
  patient-stratified across the six CV cohorts (8–9 per cohort). Same
  50 patient IDs across methods.
- **Peak VRAM**: `torch.cuda.reset_peak_memory_stats()` before the
  50-volume run; `torch.cuda.max_memory_allocated()` peak at the end.
  Reported as one number per method per NFE.
- **Failure handling**: an OOM or numerical-NaN failure during the
  50-volume timed run is reported as the failure rate (e.g. "C3 at
  NFE = 20: 47 / 50 volumes completed; 3 OOMs on the patient with
  shape 240×240×220"). The table footnotes the failures explicitly.

### 5.3 Output artefact schema (per-method predictions H5)

Every method writes its Ring A / Ring B / Ring C predictions to a
single HDF5 per ring per method, following the schema below (compatible
with `.claude/rules/h5-design-principles.md`):

```
benchmarks/predictions/<method>/<ring>.h5

  /predictions/t1c_synthetic_harmonised    (N, H, W, D)  float32, gzip 4, chunk (1, H, W, D)
                                                         after §4.1 harmonisation, range [0, 1]
  /predictions/t1c_synthetic_raw           (N, H, W, D)  float32, gzip 4
                                                         method-native output before harmonisation
                                                         (audit only; not used for metrics)
  /reference/t1c_real_harmonised           (N, H, W, D)  float32, gzip 4
  /reference/t1pre_harmonised              (N, H, W, D)  float32, gzip 4
  /reference/t2_harmonised                 (N, H, W, D)  float32, gzip 4
  /reference/flair_harmonised              (N, H, W, D)  float32, gzip 4
  /masks/brain                             (N, H, W, D)  int8,    gzip 4
  /masks/wt                                (N, H, W, D)  int8,    gzip 4
  /residuals/raw                           (N, H, W, D)  float32, gzip 4
                                                         t1c_real_harmonised − t1c_synthetic_harmonised
                                                         (pre-computed once for §4.3 reuse)

  /metadata/patient_id                     (N,)  vlen-str
  /metadata/cohort                         (N,)  vlen-str
  /metadata/inference_seconds              (N,)  float32   # per-volume wall-clock, §5.2
  /metadata/peak_vram_mb                   (N,)  float32   # method-level scalar broadcast to N for convenience
  /metadata/nfe                            (N,)  int32     # for C3/C4/C5/VENA; -1 for GAN-tier
  /metadata/scan_shape                     (N, 3) int32

  attrs/
    schema_version       = "1.0"
    created_at           = ISO-8601-UTC
    producer             = "benchmarks.<method>.predict:v<>"
    method               = "C0|C1|C2|C3|C4|C5|C6|C7|A1|VENA"
    ring                 = "A|B|C"
    harmonisation_recipe = "percentile_normalise(lower=0.5, upper=99.5, foreground_only=True)"
    git_sha              = "<sha>"
    checkpoint_path      = "<abs path>"
    checkpoint_sha256    = "<sha>"
    ring_partition_hash  = "<sha of artifacts/validation/<UTC>/ring_partitions.json>"
```

A validator `benchmarks.predictions.assert_predictions_valid(path)`
asserts the schema and cross-field invariants per
`h5-design-principles.md`:

- `predictions/t1c_synthetic_harmonised.shape ==
  reference/t1c_real_harmonised.shape == masks/brain.shape`.
- `predictions/t1c_synthetic_harmonised` range ⊆ $[0, 1]$ inside
  `masks/brain`, with the brain-mask exterior forced to 0.
- No NaN / Inf in any prediction or residual.
- `residuals/raw == reference/t1c_real_harmonised −
  predictions/t1c_synthetic_harmonised` to within float32 tolerance.
- `metadata/patient_id` has no duplicates and the ID set matches
  `ring_partition_hash`'s pre-registered list.

Every metric script (§4.2, §4.3, §4.4, §4.5, §4.6, §4.7) consumes only
validated H5s. The residual map is stored once and reused by §4.2
(MAE/PSNR/SSIM), §4.3 (spatial residual analysis), and §9 (failure-mode
figures) — single decode pass per scan.

**Storage budget.** Per ring per method: ~3 GB at gzip 4 for the seven
volumetric fields × N ≈ 173 (Ring A) or up to 406 (Ring B) volumes at
240³ × float32. Ten methods × three rings × ~3 GB ≈ 94 GB total
under
`benchmarks/predictions/`. Lives on `/media/mpascual/MeningD2/` locally
and on Picasso fscratch in mirror; the rsync route documented in
`reference_picasso_transfer_route` applies.

### 5.4 Compute environment per method

| Method | Conda env | Singularity image | CUDA / PyTorch |
|---|---|---|---|
| C0 | n/a | n/a | n/a |
| C1, C2, C3 | `icon-lab` | `singularity://icon-lab.sif` | per author defaults |
| C4 | `vena` | `singularity://vena.sif` | 12.1 / 2.4.x |
| C5 | `t1c-rflow` | `singularity://t1c-rflow.sif` | per author defaults |
| C6 (3D-Latent-DDPM) | `t1c-rflow` | `singularity://t1c-rflow.sif` | per author defaults — shares the T1C-RFlow env because the trunk + VAE + data-loader scaffolding is reused from C5 (only the noising schedule and sampler differ) |
| C7 (3D-Latent-Pix2Pix) | `t1c-rflow` | `singularity://t1c-rflow.sif` | per author defaults — same reasoning as C6 |
| A1, VENA | `vena` | `singularity://vena.sif` | 12.1 / 2.4.x |

Each environment is captured as an `environment.yml` + a Singularity
recipe in `benchmarks/<method>/env/`. The submission ships a `make
predictions/<ring>` target that runs every method end-to-end from
checkpoint to validated H5.

---

## 6. Statistical analysis plan

**Pre-registered before unblinding Ring A.**

### 6.1 Primary endpoint and decision rule

- *H_0:* MAE(VENA, whole-brain, Ring A) ≥ median MAE of the 8 competitors
  on the same data (C0 through C7 + A1 = 8 paired comparisons; C0
  identity included as the null-model floor).
- *H_1:* MAE(VENA, whole-brain, Ring A) < median MAE of the 8 competitors.
- Test: **paired Wilcoxon signed-rank** of per-patient MAE differences
  (VENA − competitor), one test per competitor, two-sided, α = 0.05.
- Multiplicity: **Holm-Bonferroni** over the 8 paired tests
  (family-wise error rate at 0.05). Holm-Bonferroni is preferred over
  raw Bonferroni because the tests are not independent (same patients,
  same metric) and Holm is uniformly more powerful (Holm 1979,
  *Scandinavian Journal of Statistics* 6:65–70). Higher-order FWER
  procedures (Hochberg, Hommel) require an assumption of joint normality
  on the test statistics that does not hold for paired Wilcoxon.

### 6.2 Effect size and confidence interval

- **Cliff's δ** (Cliff 1996) for the pairwise effect size — non-parametric,
  no normality assumption.
- **Bootstrap 95% CI** on the per-patient MAE difference (10,000
  resamples, patient-stratified to respect the cohort structure of Ring A).
- **Minimum clinically important difference (MCID)** for MAE — no
  consensus value exists in the CE-synthesis literature; we register a
  threshold of $0.01$ on the $[0,1]$-normalised intensity scale based on
  Preetha 2021's reported between-reader variability (their reader-study
  pairs differed by ≈ 1.5% absolute intensity on average). A win that
  does not exceed MCID is reported as *statistically* but not
  *clinically* significant.

### 6.3 Secondary endpoints and exploratory tests

Secondary endpoints (region-restricted MAE/PSNR/SSIM/MS-SSIM from §4.2,
spatial-residual S1/S2 from §4.3, nnU-Net Dice from §4.4) are tested
**without** primary-endpoint multiplicity correction; each is a
*separate family* with its own internal Holm-Bonferroni across the 8
competitors:

- §4.2 region-restricted: one family per (metric, region) cell — e.g.
  MAE on WT, MAE on brain ∖ WT, SSIM-3D on WT — each with 8 paired
  Wilcoxon tests and Holm-Bonferroni.
- §4.3 spatial-residual: one family covering both S1 ($\rho_S$) and S2
  ($\mathrm{Conc}(5\%)$) under C-noT, 16 paired tests total with
  Holm-Bonferroni inside the family. *Both* statistics are reported as
  equal headline tests (per user direction).
- §4.4 task-based: one family for WT-Dice, TC-Dice, ET-Dice (3 cells ×
  8 competitors with Holm-Bonferroni per cell).

Findings on secondary endpoints are reported as *confirmatory* only
when the §6.1 primary endpoint rejects $H_0$. Otherwise they are
*exploratory* and reported in supplementary with explicit language
(Bender & Lange 2001, *Adjusting for multiple testing — when and how?*,
**J. Clin. Epidemiol.** 54(4):343–349, DOI:
10.1016/S0895-4356(00)00314-0).

### 6.4 Per-ring stratification

Ring-A primary endpoint is *pooled* (one MAE per patient, all 173
patients). Per-cohort sub-analyses are reported but treated as exploratory.
Ring-B (BraTS-Africa, BraTS-PED) and Ring-C (HRUM) endpoints are
*separately* tested, each with its own family of 8 comparisons and
Holm-Bonferroni correction. The Ring-B/C comparisons answer the
generalisation question; they are pre-registered as *secondary*.

### 6.5 Power

We have power to detect an effect size Cliff's δ ≥ 0.15 (small-to-medium)
on Ring-A with n=173 paired observations at α=0.05 / 8 = 0.00625 and
β=0.20. Power computed via the standard Mann-Whitney power approximation
(Noether 1987, *J. Am. Statist. Assoc.* 82:645–647): the asymptotic
relative efficiency of Wilcoxon vs t-test is 0.955 under near-normal
residuals; for n=173 paired, the detectable effect size at α=0.00625
β=0.80 is δ ≈ 0.135 (the previous 6-competitor draft gave δ ≈ 0.13 at
α = 0.0083; tightening to α/8 raises the floor by ≈ 0.005, still well
below the small-to-medium Cliff's δ = 0.15 reference). Sufficient.
Holm-Bonferroni's adaptive nature means the most-significant test
faces α/8 while subsequent rejections face α/7, α/6, …, recovering
much of the power lost to the larger family.

---

## 7. Why FID is demoted to a sanity check

FID measures the Fréchet distance between two distributions of feature
vectors. Three reasons it is inappropriate as a primary endpoint here:

1. **The task has a ground truth.** $T_{1c}$ exists on disk for every test
   patient. The question we want to answer is *"how close is
   $\widehat{T_{1c}}$ to $T_{1c}$?"*, not *"is the *set* of
   $\widehat{T_{1c}}$ predictions distributionally close to the *set* of
   real $T_{1c}$ volumes?"*. A model that swaps two patients' synthetic
   volumes pre-evaluation incurs zero FID penalty but zero paired
   accuracy. FID is *blind* to the alignment we care about.
2. **Sample size sensitivity.** Chong & Forsyth (2020) *Effectively
   Unbiased FID and Inception Score and Where to Find Them*, CVPR,
   DOI:10.1109/CVPR42600.2020.00611, show that FID is biased downward on
   small samples and that the bias depends on the underlying distribution.
   Ring-A has 173 samples; Ring-B sub-rings have ≤ 260. The bias is
   non-negligible.
3. **Localised failures don't show up.** Vessel under-enhancement is a
   *focal* failure on a small voxel fraction. The global feature
   distribution can match the real distribution while every individual
   prediction has the wrong vessels — Borji (2022) *Pros and Cons of GAN
   Evaluation Measures: New Developments*, **CVIU** 215:103329, DOI:
   10.1016/j.cviu.2021.103329, gives constructed counterexamples.

We report FID as a *plausibility* check (does the synthesised distribution
look like the real distribution at the global level?), but we do not use
it to rank methods, do not include it in the multiplicity family, and do
not draw any submission-headline claim from it. This matches the explicit
guidance in Reinke et al. (2024) §3.6.

---

## 8. Counterfactual / shortcut-learning diagnostics

These tests are mandatory before submission. A model that wins on the
primary endpoint but fails them is not publishable in good faith.

### 8.1 Healthy-control hallucination test

Cohort: FOMO-60K subset of healthy controls (or a comparable
publicly-available healthy adult cohort once `preflights/shortcut_diag`
closes). Construct an artificial WT mask centered on a plausible glioma
location for each control; run inference with $\{T_{1\text{pre}}, T_2,
\text{FLAIR}, M_{\text{WT}}^{\text{artificial}}\} \to \widehat{T_{1c}}$.

Metric: false-positive-enhancement volume = number of voxels in $\widehat{
T_{1c}} - T_{1\text{pre}}$ exceeding the 95th percentile of intensity
difference on the *real* preoperative cohort, restricted to the artificial
mask. Decision: **median false-positive volume across the control cohort
< 0.5 mL** is the publication gate; above this, the model has learned a
shortcut "voxel ∈ $M_{\text{WT}}$ → predict bright" (Geirhos et al. 2020,
**Nature Machine Intelligence** DOI: 10.1038/s42256-020-00257-z).

### 8.2 Non-enhancing-glioma test (HRUM Tier 2)

HRUM cohort stratifies glioma into enhancing and non-enhancing per the
radiologist read at acquisition. On the non-enhancing subset, real
$T_{1c}$ shows no Gd uptake; the synthesised $\widehat{T_{1c}}$ must
match. The metric is the same false-positive-enhancement volume restricted
to the BraTS-WT mask provided by the nnU-Net segmenter. Decision: same
gate as §8.1.

### 8.3 Mask-ablation invariance test

For every Ring-A patient, run inference twice: once with the real
$M_{\text{WT}}$, once with an all-zero mask. The difference $\Delta_M =
\widehat{T_{1c}}(\text{real mask}) - \widehat{T_{1c}}(\text{zero mask})$
*should* concentrate inside the dilated WT region and *should not* leak
into the rest of the brain. This is the operational test of H2 of the
proposal at inference time (the loss-side version of the test).
Quantify: $(\sum_{\text{brain} \setminus M_{\text{WT}}} |\Delta_M|) /
(\sum_{\text{brain}} |\Delta_M|)$. Decision: ratio < 0.10 for the
proposed model and not worse than A1-VENA-S1 (the controlled
no-contrastive comparator; the controlled comparator was TumorFlow in
an earlier draft but TumorFlow was dropped — see §2).

---

## 9. Failure-mode taxonomy and qualitative reporting

The supplementary must include per-cohort residual-map figures
($T_{1c} - \widehat{T_{1c}}$) overlaid with $M_{\text{WT}}$ and the
brain mask, for *every* method on the same 8 patients (2 best, 4 median,
2 worst by primary-endpoint MAE). One figure per qualitative failure
category from Mallio et al. 2023:

1. Vessel / sinus / choroid-plexus under-enhancement (the §4.3 C-noT
   failure mode visualised on the residual heat-map).
2. Non-tumour over-enhancement (pituitary, choroid plexus, meninges).
3. Enhancement-pattern blur inside the WT (ring vs solid vs ribbon-of-fire).
4. Vendor-induced contrast drift (Ring-C only; Siemens / Philips vs GE).
5. Pediatric-specific failures (Ring-B BraTS-PED: smaller heads,
   different myelination contrast).
6. Inter-slice banding (C1, C2, C3 only) — paired with the ZGD diagnostic
   from §4.7.

The taxonomy follows the failure-mode-vocabulary tradition of Pamb et al.
(2024, IEEE JBHI) and the SegMen-style consensus in Kervadec et al.
(2021) *Boundary loss for highly unbalanced segmentation*, **Medical
Image Analysis** 67:101851 — semantically labelled failures, not raw
"this looks wrong".

---

## 10. Reader study (deferred to Ring C)

Two-AFC protocol on a HRUM Ring-C subset, three radiologists (target: 2
staff neuroradiologists, 1 senior resident at HRUM). Per the proposal §7.2
and the standard reader-study design in Preetha et al. 2021:

- 60 cases per reader, balanced (glioma vs meningioma, enhancing vs
  non-enhancing).
- Random presentation of paired triples (T1pre, real T1c, synthetic T1c
  from VENA *or* the top competitor per Ring-A primary endpoint —
  blinded).
- Reader answers: (a) which T1c is real, (b) confidence 1–5 Likert,
  (c) free-text region where synthesis fails clinically.
- Analysis: under the null *"reader cannot distinguish synthetic from
  real"*, the proportion of correct identifications follows
  $\mathrm{Binomial}(60, 0.5)$ per reader. We test the *aggregate* fraction
  correct against 0.5 with a one-sample binomial test; the venue-relevant
  claim is "fraction correct ≤ 0.60 (≈ chance + 1σ at n=60) for VENA,
  > 0.60 for the competitor". Sample-size justification: at fraction
  correct 0.5 the binomial 95% CI half-width is ±0.13 — adequate to
  distinguish 0.50 from 0.65.

The reader study is *not* a substitute for §6's quantitative endpoints; it
is the clinical-significance complement and the only remaining
perceptual-quality signal in the protocol (LPIPS-3D / RadImageNet were
demoted in §4.4).

---

## 11. Reproducibility checklist (MICCAI / MedIA expectation)

- All competitor checkpoints + VENA's final checkpoint released under
  Apache 2.0 (model code) + NVIDIA OneWay Non-Commercial (trunk weights;
  forwarded as a `links.txt` per `external-deps.md`).
- Test-set patient ID lists pinned in `artifacts/validation/<UTC>/`,
  hash recorded in every method's `decision.json` (§5.3).
- **Inference wrapper per method**: `python -m benchmarks.<method>.infer
  <input_h5> <output_h5>` with the validator (§5.3) called on the
  output H5 before any metric runs.
- **Metrics wrapper**: `python -m benchmarks.metrics.compute
  <predictions_h5> --regions brain,wt,bg --metrics
  mae,rmse,psnr,ssim,msssim,spearman,conc05,conc01,conc10,zgd
  --ring A|B|C`. The metrics CLI reads the predictions H5 (which
  carries its own real-T1c reference and brain/wt masks), so a single
  argument suffices.
- **Spatial-residual analysis wrapper** (§4.3): `python -m
  benchmarks.spatial_residual.compute <predictions_h5> --condition
  cwb,cnot --shuffle-null 1000 --output <table_csv>` writes per-patient
  Conc(q), Spearman ρ, and the shuffle-null reference values.
- Each method's training command, conda env, container, and
  CUDA / PyTorch versions logged in a `decision.json` per re-trained
  competitor; together they form a single `benchmarks/decision.json`
  artifact at submission time.

---

## 12. Timeline (mapped onto proposal §10)

| Phase | Weeks | Deliverable |
|---|---|---|
| Re-implement / port C1, C2, C3 (icon-lab tier) | 1.0 | Three wrapper scripts that train and infer on our H5 schema; 2D→3D slice-stacking inference protocol per §2 rule 10 |
| Re-implement / port C4 (3D-DiT) on MAISI latents | 1.0 | Wrapper script; same VAE / conditioning interface as VENA |
| Re-implement / port C5 (T1C-RFlow) | 1.0 | Wrapper script |
| Re-implement / port C6 (3D-Latent-DDPM) on MAISI latents | 0.5 | Wrapper script; reuses C5 dataset / runner / infer-CLI scaffolding (only the scheduler and sampler differ) |
| Re-implement / port C7 (3D-Latent-Pix2Pix) on MAISI latents | 0.5 | Wrapper script; reuses C5 dataset scaffolding; adds 3D PatchGAN discriminator |
| Re-train all 7 external competitors on Picasso | 3.0 | Trained checkpoints + per-method `decision.json` (C6 + C7 add ~5 A100-days to the budget; both fit inside the existing 2.5-week wall-clock window with the 4× A100 reservation by interleaving on different nodes) |
| Intensity-harmonisation module (§4.1) + cross-method audit table | 0.3 | `benchmarks.harmonise` module + Table S1 (post-harmonisation moments across methods) |
| Predictions H5 generation + validator (§5.3) for all 10 methods × 3 rings | 0.5 | 30 validated H5s under `benchmarks/predictions/` |
| Inference-cost + VRAM measurement pass (§5.2) | 0.3 | Table 4 (inference-cost) populated |
| Spatial residual analysis pipeline (§4.3) — Conc(q) + Spearman ρ + KSG MI + intensity-shuffle null | 0.5 | `benchmarks.spatial_residual` module + Tables 3 / S2 + Figures 3 / 4 |
| nnU-Net training on real T1c per Ring-A cohort + Dice eval (§4.4) | 0.7 | Per-cohort nnU-Net checkpoints + WT/TC/ET Dice table |
| Ring-A primary + secondary endpoint pass | 0.5 | `benchmarks/ring_a/results.csv`, all §4.2–§4.7 tables |
| Ring-B evaluation pass | 0.5 | `benchmarks/ring_b/results.csv` |
| Counterfactual / shortcut diagnostics (§8) | 1.0 | False-positive enhancement tables; mask-ablation ratios |
| Statistical analysis pass (pre-registered plan, §6) | 0.5 | Headline table + per-metric breakdowns + figures |
| Failure-mode qualitative pass (§9) | 0.5 | Supplementary residual-map figures |
| **HRUM Ring-C (deferred until data lands)** | 2.0 | Ring-C results + reader-study results |
| Writing | 4.0 | MedIA / IEEE TMI submission |

Compresses to ~11.5 weeks of effective work before HRUM, ~17.5 weeks
end-to-end once HRUM Tier 1+2 lands. (Was ~10 weeks in the 2026-06-17
morning draft and ~9 weeks in the 2026-06-08 draft; the +1.5 weeks
over the morning draft is the addition of C6 + C7 — 1.0 week porting
(2 × 0.5 since both reuse C5 scaffolding) + 0.5 week marginal training
budget (~5 A100-days added; interleaved into the existing 2.5-week
Picasso window). The new C6 + C7 rows close the 3D-quartet gap
identified in the 2026-06-17 roster review without inflating compute
proportionally, because the integration scaffolding from C5 is
reused.)

---

## Appendix A — Why nnU-Net for the downstream task (and not BraTS-2024
winners)

nnU-Net is the canonical "no-handle-tuning" baseline (Isensee et al. 2021,
*Nature Methods*) and the only segmentation model that *self-configures*
to our cohort and patch size. Using a BraTS-challenge winner (e.g.
Optimised-nnU-Net or BraTS-2023 SOTA) would introduce a confounder:
their training distribution overlaps with our Ring-A cohorts (BraTS-GLI
in particular), so a high Dice on synthetic T1c could reflect the
segmenter's familiarity with the *real* BraTS distribution rather than
the synthesis quality. Standard nnU-Net trained from scratch on the
real-T1c version of each Ring-A cohort decouples this.

---

## Appendix B — Why we do not benchmark Kleesiek 2019 despite the
SWI-input match

Kleesiek 2019 is the only published model with SWI as an input. Its
methodological framework (Bayesian 3D U-Net, pre-FM, pre-latent-space) is
five generations old and was never released as code. A faithful
re-implementation requires the original 10-channel acquisition (which
HRUM may provide on Tier 3) *and* a re-derivation of the dropout-based
uncertainty calibration. Both are tractable but constitute a separate
paper. We cite Kleesiek 2019 as the SWI-input motivation throughout the
related work and treat it as out-of-scope for the benchmark.

---

## Appendix C — Why no vessel-segmentation operator (and what we do
instead)

The 2026-06-08 draft of this protocol used a four-operator vessel-
extraction ensemble (Frangi + Jerman + OOF + VesselFM) as the
load-bearing apparatus for the vessel-fidelity claim. The 2026-06-17
revision drops it entirely. The argument:

1. **No ground truth.** No cohort in our protocol (UCSF-PDGM, BraTS-GLI,
   IvyGAP, LUMIERE, REMBRANDT, UPENN-GBM, BraTS-Africa, BraTS-PED) ships
   hand-annotated cerebral-vessel masks on $T_{1c}$. Using any operator's
   output as "ground truth" couples the synthesis evaluation to the
   operator's failure modes — operator noise becomes a confounder of
   model evaluation.

2. **Operator–input domain mismatch.** Classical vessel-segmentation
   operators (Frangi 1998, Jerman 2016, OOF/Law-Chung 2008) were
   designed and evaluated on **TOF-MRA** (time-of-flight MR angiography),
   CE-MRA, or contrast-enhanced CT angiography — modalities engineered
   to suppress non-vascular signal and highlight tubular flow. Deep
   vessel-segmentation foundation models (VesselFM, Wittmann et al.
   2024) are likewise trained on TOF-MRA / CTA / OCTA / vEM corpora.
   Structural $T_{1c}$ at clinical Gd dose is a different physical
   regime: enhancement is partial, regionally heterogeneous, and shares
   the bright tail with choroid plexus, pituitary, calcified pineal, fat
   partial-volume, and ring-enhancing tumour. The operators systematically
   false-positive on these in $T_{1c}$ — a failure mode published only
   indirectly (Mallio et al. 2023 §2 describes the qualitative trap).

3. **TOF-MRA — which is the canonical vessel input — is not in any of
   our cohorts.** Several cohorts ship SWI, none ship TOF-MRA. Acquiring
   or simulating TOF-MRA is outside the scope of this paper.

4. **No published zero-shot transfer.** VesselFM's paper (Wittmann et al.
   2024 §4) reports zero-shot evaluations on MRA, vEM, OCTA, and CT
   only. The authors do not evaluate or claim zero-shot transfer to
   structural CE MRI. A spot-check on UCSF-PDGM T1c would itself be a
   contested exercise — see the 2026-06-08 draft's spot-check protocol
   for the level of effort that was required just to admit VesselFM as
   one of four operators.

5. **A label-free alternative exists.** §4.3 tests the *same hypothesis*
   — "does the model under-synthesise the bright contrast-uptake
   regions outside the tumour?" — by correlating the residual map with
   the real $T_{1c}$ intensity inside the brain ∖ dilated WT region.
   This requires no segmenter, no atlas, and no operator ensemble. The
   apparatus (Spearman ρ + top-q% mass concentration + Bland-Altman
   intensity-stratified plot + intensity-shuffle null) is the standard
   statistical toolkit for heteroscedasticity / conditional-dependence
   testing translated to the image domain (Yin & Carroll 1990, Bishara
   & Hittner 2012, Bland & Altman 1986/1999, Abraham et al. 2003).

The trade-off: the residual-based test does not produce a "vessel
mask" the reader can inspect visually. The §9 failure-mode supplementary
figures provide the qualitative complement (per-cohort residual heat
maps overlaid with the brain and WT masks, on the same patients used
across all methods). The reader study (§10) provides the human
adjudication of clinical realism. The combination — quantitative
spatial residual analysis + qualitative residual maps + human reader
study — covers the vessel-fidelity claim more rigorously than any
operator ensemble could.

---

## References

(Full bibliography in `literature.md`; this document cites by author-year.
The pre-registered analysis plan and patient-ID partitions are deposited
at `artifacts/validation/<UTC>/` at submission day, hash recorded in
each competitor's `decision.json`.)

- Abraham, van den Bergh & Nair 2003, **ApJ** 588:218–229, DOI:
  10.1086/373919 — Gini-style morphology / pixel-rank concentration.
- Adewole et al. 2023, arXiv:2305.19369 — BraTS-Africa.
- Alexander-Bloch et al. 2018, **NeuroImage** 178:540–551, DOI:
  10.1016/j.neuroimage.2018.05.070 — spatial-null permutation in
  imaging.
- Bender & Lange 2001, **J. Clin. Epidemiol.** 54(4):343–349, DOI:
  10.1016/S0895-4356(00)00314-0 — multiple-testing communication.
- Bińkowski et al. 2018, arXiv:1801.01401 — KID.
- Bishara & Hittner 2012, **Psychological Methods** 17(3):399–417, DOI:
  10.1037/a0028087 — Spearman ρ over Pearson under heavy-tailed data.
- Bland & Altman 1986, **The Lancet** 327(8476):307–310, DOI:
  10.1016/S0140-6736(86)90837-8 — agreement of methods, intensity-
  stratified differences.
- Bland & Altman 1999, *Stat. Methods Med. Res.* 8(2):135–160, DOI:
  10.1177/096228029900800204 — heteroscedastic extension.
- Borji 2022, **CVIU** 215:103329, DOI: 10.1016/j.cviu.2021.103329 — GAN-
  evaluation review.
- Chong & Forsyth 2020, **CVPR**, DOI: 10.1109/CVPR42600.2020.00611 —
  FID bias.
- Cliff 1996 — Cliff's δ; *Ordinal Methods for Behavioral Data Analysis*.
- Dar et al. 2019, **IEEE TMI** 38(10):2375–2388, DOI:
  10.1109/TMI.2019.2901750 — pGAN.
- Dalmaz et al. 2022, **IEEE TMI** 41(10):2598–2614, DOI:
  10.1109/TMI.2022.3167808 — ResViT.
- Dohmen et al. 2024, **MICCAI Workshop DGM4MICCAI** / LNCS, DOI:
  10.1007/978-3-031-72744-3_15, arXiv:2408.06075 — pitfalls of
  reference metrics on synthetic medical images.
- Dohmen et al. 2025, **Scientific Reports**, arXiv:2405.08431 —
  similarity and quality metrics for MR I2I translation.
- Geirhos et al. 2020, **Nature Machine Intelligence** 2:665–673, DOI:
  10.1038/s42256-020-00257-z — shortcut learning.
- Holm 1979, **Scandinavian Journal of Statistics** 6:65–70 — Holm-
  Bonferroni.
- Isensee et al. 2021, **Nature Methods** 18:203–211, DOI:
  10.1038/s41592-020-01008-z — nnU-Net.
- Kazerooni et al. 2023, arXiv:2305.17033 — BraTS-PED.
- Kervadec et al. 2021, **Medical Image Analysis** 67:101851 — boundary
  loss; failure-mode vocabulary precedent.
- Kraskov, Stögbauer & Grassberger 2004, **Phys. Rev. E** 69:066138,
  DOI: 10.1103/PhysRevE.69.066138 — k-NN MI estimator.
- Maier-Hein et al. 2018, **Nature Communications** 9:5217, DOI:
  10.1038/s41467-018-07619-7 — biomedical ranking rigor.
- Mallio et al. 2023, **Frontiers in Neuroimaging** 2:1055463, DOI:
  10.3389/fnimg.2023.1055463 — failure-mode review.
- Mei et al. 2022, **Radiology AI** 4(5):e210315 — RadImageNet (cited
  only in §4.4 demotion rationale).
- Noether 1987, **J. Am. Statist. Assoc.** 82:645–647 — Wilcoxon power.
- Nyúl & Udupa 1999, **Magn. Reson. Med.** 42(6):1072–1081, DOI:
  10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M — MR
  intensity standardisation (cited only in §4.1 alternatives rationale).
- Özbey et al. 2023, **IEEE TMI** 42(12):3524–3539, DOI:
  10.1109/TMI.2023.3290149 — SynDiff.
- Pamb et al. 2024, **IEEE JBHI** — *Beyond PSNR and SSIM* survey.
- Preetha et al. 2021, **Lancet Digital Health** 3(12):e784–e794, DOI:
  10.1016/S2589-7500(21)00205-3 — downstream-task protocol.
- Reinke et al. 2024, **Nature Methods** 21(2):182–194, DOI:
  10.1038/s41592-023-02151-z — image-processing-metric limitations.
- Wang et al. 2003, IEEE — MS-SSIM.
- Wang et al. 2004, **IEEE TIP** 13(4):600–612, DOI:
  10.1109/TIP.2003.819861 — SSIM.
- Wittmann et al. 2024, **MICCAI 2025** / arXiv:2411.17386 — VesselFM
  (cited only in §4.3 / Appendix C drop rationale).
- Yin & Carroll 1990, *Statistics & Probability Letters* 10(1):69–76,
  DOI: 10.1016/0167-7152(90)90114-M — Spearman ρ as a
  heteroscedasticity diagnostic.
