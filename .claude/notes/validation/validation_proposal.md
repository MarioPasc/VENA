# VENA Validation Proposal — Competitor Benchmarking and Evaluation Protocol

*Mario Pascual González, 2026-06-08. Target venues: Medical Image Analysis,
IEEE TMI, MICCAI 2026 main conference.*

Companion to [`literature.md`](literature.md) (competitor selection rationale,
§11b) and the project proposal at
`/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` (model
specification). This document pins **how** we measure success, **which**
competitors we measure against, **on which data**, and **how we test for
statistical and clinical significance**.

---

## 1. Design principles

A validation protocol for a high-impact methodological paper has to defend
four positions:

**P1 — Paired ground truth dictates paired metrics.** The task is
$\{T_{1\text{pre}}, T_2, \text{FLAIR}\} \to T_{1c}$, where the real $T_{1c}$
is on disk for every test scan. The relevant question is *"how close is the
predicted $\widehat{T_{1c}}$ to the ground-truth $T_{1c}$, voxel- and
structure-wise?"*, not *"is the distribution of predictions
distributionally close to the distribution of real images?"*. Distributional
metrics (FID, KID) answer the second question and are appropriate for
*unconditional* generation; they are demoted to *secondary plausibility
checks* here, never primary endpoints (justification in §6).
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

Five external competitors + one internal controlled comparator (VENA-S1).
The mandatory tier was revised on 2026-06-08 after dropping CFM (no public
code) and TumorFlow (different task scope — longitudinal physics-guided
growth, not paired CE translation); CFM's slot in the FM/diffusion tier is
taken by 3D-DiT, and TumorFlow's role as "no-contrastive controlled
comparator" is replaced by VENA-S1 (a strictly tighter ablation because
no architectural delta exists). Re-listed here with the *training
protocol* — what is held fixed when re-training each on our cohort split:

| Tag | Method | Reference | Code | Re-train inputs | 2D / 3D | Wall-clock (4× A100 est.) |
|---|---|---|---|---|---|---|
| **C0-Identity** | $\widehat{T_{1c}} \equiv T_{1\text{pre}}$ | — | — | T1pre → T1c (no learning) | n/a | 0 |
| **C1-pGAN** | pGAN (paired) | Dar et al. 2019, IEEE TMI | icon-lab/pGAN-cGAN | T1pre, T2, FLAIR → T1c | 2D axial | ~1 d |
| **C2-ResViT** | ResViT (paired) | Dalmaz et al. 2022, IEEE TMI | icon-lab/ResViT | T1pre, T2, FLAIR → T1c | 2D axial | ~2 d |
| **C3-SynDiff** | SynDiff (paired) | Özbey et al. 2023, IEEE TMI | icon-lab/SynDiff | T1pre, T2, FLAIR → T1c | 2D axial | ~3 d |
| **C4-3D-DiT** | Latent diffusion with DiT backbone | Peebles & Xie 2023; Mo et al. 2023; Eidex 2025 baseline | facebookresearch/DiT (2D) + 3D adaptation port | T1pre, T2, FLAIR → T1c | 3D MAISI latent | ~4 d |
| **C5-T1C-RFlow** | T1C-RFlow | Eidex et al. 2025, arXiv | (paper repo, contact authors if missing) | T1pre, T2, FLAIR → T1c | 3D latent (custom VAE) | ~4 d |
| **A1-VENA-S1*** | VENA's own CFM-only checkpoint | this work, `picasso_s1_1000ep.yaml` | this repo | T1pre, T2, FLAIR → T1c | 3D MAISI latent | already trained |

*A1 = "ablation 1", not a competitor. Same MAISI VAE + NV-Generate-MR
trunk + ControlNet + ControlNet conditioning + training cohort + epoch
budget as VENA; the *only* delta is the $L^p$-aware contrastive term in
the loss. Any S2-vs-S1 gap on the §4.5 metrics is therefore attributable
to the contrastive term, isolating the headline contribution under a
zero-architectural-delta ablation.

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
   intensities are pinned.
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

C0 (identity) is the **null-model floor**. Any non-trivial method must
beat C0 on every region-restricted metric; if a method does not beat C0
inside the WT, the model has failed at the *one* thing it is supposed to
do (synthesise enhancement).

**Caveat on competitor specificity (must be reflected in the paper's
discussion).** Of the five external competitors:

- **C5 (T1C-RFlow)** is the only one that targets T1c synthesis as its
  headline task — it defines the current numerical SOTA. A win over C5
  on the primary endpoint *is* the SOTA claim.
- **C4 (3D-DiT)** is a *generic latent-diffusion* architecture (the
  transformer-backbone diffusion baseline that the FM/diffusion
  literature uses as its standard reference, including T1C-RFlow
  themselves). Not T1c-specific.
- **C1, C2, C3** do *not* target T1c synthesis — they are *generic*
  paired cross-contrast MR translation methods (pGAN, ResViT, SynDiff)
  re-purposed for the T1pre+T2+FLAIR→T1c direction. They are included
  because reviewers at MedIA / IEEE TMI consistently expect the icon-lab
  baseline triad — a methods paper that does not beat the strongest
  *general* cross-contrast architecture is dismissable as "just a task
  paper". *Architectural* baselines, not *task* baselines.

The discussion must report architectural-baseline wins (C1/C2/C3/C4)
and task-baseline wins (C5) **separately**: a small win over C5 is what
defines SOTA on the task; wins over C1–C4 are what define a methods
contribution that generalises beyond the task. The A1-VENA-S1 ablation
isolates the $L^p$-contrastive contribution.

C1–C3 are re-trained with the T1pre+T2+FLAIR→T1c direction selected from
their many-to-one (ResViT, SynDiff) or many-to-many (pGAN-cGAN) APIs.
C4-3D-DiT is trained over MAISI latents to match VENA's latent space
(controls for the VAE-choice confound). The training pairs are identical
to VENA's.

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

Ring A is the *primary endpoint*. The headline table in the paper is the
six competitors × VENA on Ring-A pooled (with per-cohort breakdown in
supplementary).

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
  glioma as a counterfactual — see §7.2).
- **Tier 3** (optional, exploratory: longitudinal follow-up timepoints).
  Reported as supplementary; not part of the primary or secondary endpoint
  family.

A SWAN-conditioned follow-up paper is *parked* for after HRUM Tier 3
lands — that variant requires re-training and is out of scope here. The
vessel-fidelity claim in this submission is purely loss-side (the $p_b=3$
background term of the $L^p$-aware contrastive upweights non-tumour
errors); it is testable on every ring through the multi-operator
vessel-extraction ensemble in §4.5, without ever needing SWAN as input.

### Pre-registration

The patient identifiers in each ring are frozen on submission day and
written, as plain JSON, to
`artifacts/validation/<UTC-timestamp>/ring_partitions.json`.
The hash of that file is written into every competitor's
`decision.json` so the analysis cannot retroactively choose its test set.

---

## 4. Metric suite

Four families, decreasing in primary-endpoint weight.

### 4.1 Primary — paired voxel-wise fidelity (per region)

All metrics computed in **image space**, after VAE decoding, on the
percentile-normalised intensity volume $\widehat{T_{1c}}, T_{1c} \in [0,
1]$ (this fixes `data_range=1.0`, the bug that silently inflates PSNR/SSIM
across the synthesis literature — see Reinke et al. 2024 §3.2).

Per metric, three region restrictions are evaluated:
- **whole brain** (mask = HD-BET / CBICA brain mask).
- **WT** (whole-tumour mask, BraTS labels >0).
- **brain \ WT** (background; the region the $L^p$-contrastive specifically
  targets — H2/H3 of the proposal).

Metrics:

- **MAE**, **RMSE** — direct intensity error. MAE is preferred over MSE for
  ranking (Reinke et al. 2024 §3.1: MSE is dominated by the heavy tail of
  the residual distribution). Both reported for completeness.
- **PSNR-3D** — standard but susceptible to dynamic-range effects; we
  always report it alongside SSIM and never alone.
- **SSIM-3D** — `monai.metrics.SSIMMetric` with $k_1 = 0.01$, $k_2 = 0.03$,
  Gaussian window 11³, `data_range=1.0`. Wang et al. (2004) DOI:
  10.1109/TIP.2003.819861.
- **MS-SSIM-3D** — multi-scale, four levels, weights $[0.0448, 0.2856,
  0.3001, 0.3633]$ (Wang et al. 2003) for downstream-segmentation-relevant
  structures (Pamb et al. 2024 *Beyond PSNR and SSIM: A Survey of Image
  Quality Metrics for Medical Imaging*, **IEEE JBHI** — confirms MS-SSIM
  correlates better than SSIM with radiologist quality scores on MR
  synthesis).

The primary endpoint is **MAE on the whole-brain region of Ring A,
VENA vs the median of the 6 competitors**.

### 4.2 Secondary — perceptual and downstream-task fidelity

- **LPIPS-3D** (Zhang et al. 2018, CVPR DOI: 10.1109/CVPR.2018.00068) —
  2.5-D approximation, averaging the LPIPS over orthogonal slabs; AlexNet
  backbone for protocol parity with the MAISI-v2 evaluation. *And* a
  second LPIPS pass with **RadImageNet ResNet-50 features** (Mei et al.
  2022, *Radiology AI* 4(5):e210315) for medical-domain perceptual
  alignment. Both reported; the RadImageNet variant is the principal one,
  because ImageNet features have known transfer-failure modes on MR
  intensity statistics (Mei 2022 §Discussion).
- **Task-based: nnU-Net segmentation Dice** on the Preetha protocol
  (Preetha et al. 2021, Lancet Digital Health, DOI:
  10.1016/S2589-7500(21)00205-3). Train an nnU-Net (Isensee et al. 2021,
  Nature Methods, DOI: 10.1038/s41592-020-01008-z) on real
  $\{T_{1\text{pre}}, T_{1c}, T_2, \text{FLAIR}\} \to$ BraTS labels for
  every Ring-A cohort; evaluate on $\{T_{1\text{pre}}, \widehat{T_{1c}},
  T_2, \text{FLAIR}\} \to$ predicted labels. Report Dice for whole-tumour
  (WT), tumour-core (TC), and enhancing-tumour (ET) sub-labels. ET-Dice
  is the *clinical-impact* metric: it directly measures whether the
  synthesised T1c preserves enhancement geometry well enough for
  downstream tumour assessment.
- *Background-contrast / vessel evaluation* — see §4.5 below. The full
  treatment is separated out because we have **no ground-truth vessel
  segmentations** in any of the nine cohorts, and the protocol has to
  be designed around that constraint.

### 4.3 Tertiary — distributional plausibility (sanity check, not primary)

- **FID-3D** with RadImageNet features (2.5-D approximation, axial slabs).
  *Reported, not used for ranking* — see §6 for the explicit justification.
  Its purpose is to flag mode collapse or systematic over-/under-saturation
  in the synthetic volumes; a competitor that loses to VENA on paired MAE
  but matches it on FID is an *interesting* finding (we generate more
  realistic but less accurate samples), worth supplementary discussion.
- **KID-3D** — same purpose, less biased on small samples (Bińkowski et al.
  2018, arXiv:1801.01401).

### 4.4 Inference cost (mandatory for the FM/RF claim)

- Wall-clock per volume on a single A100 40 GB at NFE ∈ {1, 5, 20}, mean
  ± std over 50 Ring-A volumes. The proposal's <10 s target inherits from
  Eidex et al. 2025 (T1C-RFlow); VENA must match it.
- Peak VRAM at inference.
- Number of forward-backward passes per training step (1 for C0–C2; 2 for
  VENA's contrastive paths; matters for cost-quality framing).

### 4.5 Background contrast evaluation (vessels + normally-enhancing anatomy)

**Constraint.** None of the nine cohorts ships ground-truth vessel
segmentations, and the current VENA formulation deliberately **does not
use SWAN/SWI as an input** (the vessel-fidelity claim is loss-side, driven
by the $p_b = 3$ background term of the $L^p$-aware contrastive — proposal
§5.3). The vessel-fidelity test must therefore be conducted under a
no-vessel-label regime, on T1pre/T1c/T2/FLAIR alone.

**Principle.** No vessel-extraction operator is used as ground truth.
Instead, each operator is applied *identically* to real and synthetic
$T_{1c}$, and we measure whether the model preserves the contrast
structures that the *same operator* extracts from the *real* volume. Any
systematic operator bias (false positives on sulcal-CSF boundaries,
white-matter junctions, etc.) cancels in the paired comparison. The same
logic underlies the use of nnU-Net in the Preetha protocol (Preetha et al.
2021) — the segmenter need not be perfect, only consistently applied
across real and synthetic. For robustness, we run **four** complementary
vessel-extraction operators and report per-operator and ensemble results.

**The vessel-extraction operator suite.** Each operator is applied
identically to real and synthetic T1c volumes. None requires SWAN/SWI as
input; all four work on post-contrast structural MRI.

| Tag | Operator | Type | Output | Reference |
|---|---|---|---|---|
| **F** | Frangi vesselness | Classical Hessian, multi-scale | Soft [0,1] | Frangi et al. 1998, DOI: 10.1007/BFb0056195 |
| **J** | Jerman vesselness | Classical Hessian, junction-corrected | Soft [0,1] | Jerman et al. 2016, IEEE TMI, DOI: 10.1109/TMI.2016.2515603 |
| **O** | OOF (Optimally Oriented Flux) | Classical flux-based | Soft scalar | Law & Chung 2008, ECCV, DOI: 10.1007/978-3-540-88693-8_27 |
| **V** | VesselFM (zero-shot on T1c, **conditional on spot-check**) | Foundation model trained on brain MRA + mouse vEM + mouse OCTA + liver CT (paper Table 1) | Binary mask (sigmoid + threshold, whole-volume via sliding window $128^3$ overlap 0.5) | Wittmann et al. 2024, MICCAI 2025, arXiv:2411.17386. Code+weights: github.com/bwittmann/vesselFM |

*Why this four-operator panel.* Frangi is the standard reference and the
fastest. Jerman corrects Frangi's underestimation at vessel junctions and
aneurysms — the exact regions where T1c enhancement is hardest to
synthesise (cf. Mallio et al. 2023 §3.2). OOF resolves both Hessian
operators' branch-point weakness via flux integration. VesselFM is the
foundation-model entry — the only off-the-shelf *deep-learning* segmenter
released with public weights and a documented zero-shot generalisation
claim across modalities; its inclusion gives the ensemble an
architectural-family diversity that all-classical or all-deep panels
would lack. If the four operators agree on a vessel-fidelity verdict,
the verdict is robust to operator-choice — and this is itself a
reportable metric (M8 below).

**Caveat on VesselFM scope (audited 2026-06-08).** VesselFM's training
corpus and its four zero-shot evaluation domains (paper §4.1: MRA, vEM,
OCTA, CT) do *not* include any structural brain MRI. The paper does
not evaluate or claim zero-shot performance on $T_{1c}$. We extrapolate
to T1c because the MRA→T1c analogy is qualitatively reasonable (bright
tubular Gd-enhanced lumen, suppressed background — similar to TOF) but
physically imperfect: Gd-BBB-disruption physics differ from flow-related
enhancement, and choroid plexus / pituitary / dural sinuses / enhancing
tumour are all bright on T1c without an MRA precedent to suppress
mis-segmentation. **The extrapolation must be empirically validated
before VesselFM enters the protocol** — see the spot-check below.

**Pre-protocol spot-check (mandatory gate).** Before VesselFM is
admitted to the four-operator ensemble, we run a one-time validation
pass on 20 randomly-selected real $T_{1c}$ volumes from UCSF-PDGM
(separate from the test partition; drawn from val) under the inference
contract documented above (single-channel T1c, percentile re-tuned to
`(0.5, 99.5)` with `foreground_only=True`, sliding-window $128^3$).
Visual review by the PI and a co-author identifies:

1. *Vessel coverage*: does VesselFM mark the major cerebral vessels
   (MCA M1-M3, ACA, PCA, sigmoid + transverse sinuses, vein of Galen,
   superior sagittal sinus) that Frangi also marks?
2. *Tumour over-segmentation*: does VesselFM erroneously mark enhancing
   tumour as vessel? Quantified as $|V \cap M_{\text{WT}}| / |V|$
   (fraction of VesselFM mask inside the WT). Threshold: > 20% counts
   as systematic tumour confusion.
3. *Non-vessel false positives*: choroid plexus, pituitary, pineal,
   dural sinus regions inspected manually for sensible vs hallucinated
   labelling.

**Decision rule.** If the spot-check is acceptable (≥ 18 of 20 volumes
pass on all three criteria with no systematic failure), VesselFM enters
the ensemble as the fourth operator. If it fails (≥ 3 volumes show
systematic tumour confusion or major-vessel misses), the ensemble drops
to three operators (F, J, O) with a **"≥ 2-of-3 agree"** consensus
rule, and M3/M4/M7/M8 are re-baselined for $|{\text{operators}}| = 3$.
A failed spot-check is itself reported in the paper's methods section
as a methodological finding ("foundation-model vessel segmenters
released in 2024 do not zero-shot to structural CE MRI; the
implications for the broader CE-synthesis evaluation literature are
discussed in §X"). Either outcome is publishable.

**Optional sensitivity experiment.** VesselFM's repo supports one-shot
fine-tuning (`vesselfm/seg/finetune.py`). After the spot-check, fine-
tune VesselFM with 1 annotated T1c slab (the PI's hand-drawn vessel
annotation on a single UCSF-PDGM volume) and re-run the ensemble with
the fine-tuned weights. Reported as supplementary; addresses the "did
you cherry-pick the zero-shot scenario?" reviewer comment.

**Ensemble consensus.** For each volume, define the *consensus vessel
mask*

$$
V_{\text{cons}} = \big\{x : \big|\{O \in \{F^*, J^*, O^*, V\} : x \in O\}\big| \geq 2\big\}
$$

where $F^*, J^*, O^*$ are the percentile-thresholded binary versions of
Frangi/Jerman/OOF (per-volume 95th percentile) and $V$ is VesselFM's
native binary output. The "≥ 2 of 4 agree" rule is a standard ensemble
consensus criterion (Kuncheva 2014, *Combining Pattern Classifiers*,
Wiley); below 2-of-4, the mask is dominated by single-operator
idiosyncrasies; above 2-of-4 (the 3-of-4 or unanimous rules), the mask
collapses to large-vessel-only.

Every M-metric below is reported **per operator** (F, J, O, V) and
**for the ensemble consensus** ($V_{\text{cons}}$).

**Background region $R_{\text{bg}}$.** Defined per scan as
$\text{brain} \setminus \mathrm{dilate}(M_{\text{WT}}, k = 5)$ — the
brain mask minus the WT mask dilated by 5 voxels (to avoid bleed from
ring-enhancement at the tumour boundary). Every metric below is computed
on $R_{\text{bg}}$ unless stated otherwise.

The clinically relevant structures in $R_{\text{bg}}$ that *should*
enhance on T1c are:

1. **Vasculature** — cerebral arteries (M1–M4 MCA branches, ACA, PCA),
   cortical veins, deep venous system, dural venous sinuses (superior
   sagittal, transverse, sigmoid, straight sinus, vein of Galen).
2. **Normally-enhancing anatomy** — choroid plexus (lateral, third, and
   fourth ventricles), pituitary gland and infundibulum, pineal gland.
3. **Meninges** — dural and arachnoid layers (thin, often subtle
   enhancement; the most common false-positive site in synthesised T1c).

A model can fail in two complementary ways: (a) *under-enhance* these
structures (vessel-fidelity loss; the Mallio failure mode), or (b)
*over-enhance* into adjacent parenchyma (the hallucination failure mode).
The metrics below probe both.

#### 4.5.1 Direct vesselness-map comparison (M1)

For each of the three classical operators ($F$ = Frangi, $J$ = Jerman,
$O$ = OOF), apply with $\sigma \in \{0.5, 1.0, 1.5, 2.0, 2.5\}$ mm,
$\alpha = 0.5, \beta = 0.5, \gamma = $ adaptive. Each soft response map
$\mathcal{O}: \text{brain} \to [0, 1]$ is the maximum response over
scales (the Frangi convention, also adopted by Jerman and OOF).
VesselFM ($V$) returns a binary mask directly — M1 is reported only
for the three soft-map operators.

**Metric.** Per operator $\mathcal{O} \in \{F, J, O\}$,
$\text{MAE}_\mathcal{O} = \mathbb{E}_{x \in R_{\text{bg}}}
|\mathcal{O}_{\text{real}}(x) - \mathcal{O}_{\text{syn}}(x)|$ and the
Pearson correlation $\rho_\mathcal{O}$ between the two maps over
$R_{\text{bg}}$. *No thresholding* — calibration-free. A model that
systematically blurs vessels produces a lower-magnitude
$\mathcal{O}_{\text{syn}}$ and a positive MAE bias; the sign of the bias
is reported (under-detection vs over-detection of tubular contrast).

*Justification.* Direct comparison of soft maps avoids the threshold
arbitrariness flagged by Reinke et al. (2024 §3.3) on binary
segmentation-derived metrics. Per-operator reporting (rather than a
single aggregate) lets the discussion attribute disagreements to the
known operator-specific weaknesses (Frangi at branch points, Jerman
near aneurysms, OOF at very small scales) — diagnostic value, not just
ranking value.

#### 4.5.2 Top-percentile contrast-uptake overlap (M2)

The *spatial pattern of contrast uptake outside the tumour* — where the
top X% of intensity sits — is what a clinician implicitly evaluates when
they say "the vessels look right". We test it directly without invoking
vessel labels:

$$
B^{q}_{\text{real}} = \{x \in R_{\text{bg}} : T_{1c}(x) > Q^q_{\text{real}}\},
\quad B^{q}_{\text{syn}} = \{x \in R_{\text{bg}} : \widehat{T_{1c}}(x) > Q^q_{\text{syn}}\},
$$

where $Q^q$ is the $q$-th intra-volume intensity quantile restricted to
$R_{\text{bg}}$. **Per-volume quantiles** (not a fixed intensity
threshold) eliminate the global-intensity-drift confound — a model that
gets the global brightness wrong but the *relative* pattern right is not
penalised. Reported at $q \in \{90, 95, 99\}$:

**Metric.** Dice($B^q_{\text{real}}, B^q_{\text{syn}}$), IoU, and the
Hausdorff-95 distance between their boundaries. Dice at $q = 95$ is the
primary background-contrast endpoint.

*Justification.* Per-volume quantile thresholding is the standard
calibration-free comparator in pathology imaging (Veta et al. 2019,
*Medical Image Analysis* 54:111–121 on mitosis detection) — applied here
because vessel-vs-parenchyma intensity contrast is the meaningful signal,
not the absolute intensity.

#### 4.5.3 clDice on percentile-thresholded operator maps (M3)

For each soft-map operator $\mathcal{O} \in \{F, J, O\}$, threshold *per
volume* at its own 95th percentile:
$V^{0.95}_{\text{real}, \mathcal{O}} =
\mathcal{O}_{\text{real}} > q_{95}(\mathcal{O}_{\text{real}})$,
$V^{0.95}_{\text{syn}, \mathcal{O}} =
\mathcal{O}_{\text{syn}} > q_{95}(\mathcal{O}_{\text{syn}})$. For
VesselFM ($V$), use its native binary mask directly. Finally, for the
ensemble consensus $V_{\text{cons}}$ defined above.

**Metric.** clDice (Shit et al. 2021, CVPR DOI:
10.1109/CVPR46437.2021.01629) between $V^{0.95}_{\text{real}, \mathcal{O}}$
and $V^{0.95}_{\text{syn}, \mathcal{O}}$, reported per operator and for
the ensemble. clDice penalises *topological* failures (broken vessel
chains, false bridges) which voxel-overlap metrics like Dice miss —
designed for tubular structures and used precisely in no-ground-truth
regimes where the comparison is mask-vs-mask, not mask-vs-label.

*Key clarification.* clDice requires *two binary masks*, not a binary
mask and a ground truth. It is symmetric in the two arguments. We use
it as a *consistency* metric between the real-applied and synthetic-
applied operator outputs — not as a vessel-segmentation accuracy
metric.

#### 4.5.4 Vessel-conspicuity ratio on fixed real-vessel mask (M4)

Fix the vessel mask from the *real* volume's ensemble consensus:
$V_{\text{ref}} = V_{\text{cons, real}}$ (the ≥ 2-of-4 mask defined
above, computed from the *real* T1c only). Using the ensemble rather
than a single operator removes operator-choice bias from M4's reference
mask.

On the *same* mask, compute the conspicuity ratio on real and synthetic:

$$
\mathrm{VCR}(I) = \frac{\bar{I}\big|_{V_{\text{ref}}}}{\bar{I}\big|_{\mathrm{dilate}(V_{\text{ref}}, 5) \setminus V_{\text{ref}}}},
$$

i.e. mean intensity inside the reference vessels divided by mean
intensity in a 5-voxel rim around them, where "rim" = dilated vessel
mask minus vessel mask.

**Metric.** $\Delta \mathrm{VCR} = \mathrm{VCR}(\widehat{T_{1c}}) -
\mathrm{VCR}(T_{1c})$ — the difference between synthetic and real
conspicuity *on the same anatomical structures*. Negative
$\Delta \mathrm{VCR}$ = the model blurs vessels into parenchyma
(Mallio's failure mode). Reported per volume; statistical test against
the per-volume null $\Delta \mathrm{VCR} = 0$ via paired Wilcoxon (per
competitor).

*Justification for fixing the mask from real.* If we used
$V_{\text{syn}}$ to define the rim for the synthetic comparison, a
model that simply does not produce vessels at all would have an
undefined VCR; using $V_{\text{ref}}$ for both forces every method to
be evaluated on the same anatomical region. This is the operational
"does the model preserve vessel contrast" test.

#### 4.5.5 Background-intensity Wasserstein distance (M5)

Compute the histogram of intensities in $R_{\text{bg}}$ for real and
synthetic volumes (256 bins, range $[0, 1]$). **Metric:** Wasserstein-1
distance (Earth Mover's, `scipy.stats.wasserstein_distance`) between
the two histograms. This is a *whole-distribution* test: a model that
gets the average background brightness right but flattens the bright
tail (where vessels live) shows a large $W_1$ even when MAE/SSIM are
acceptable. Reported per volume.

*Distinction from FID.* $W_1$ here is computed *per scan*, between the
*real* and *predicted* intensity distribution of *the same patient*'s
background voxels. FID computes a distance between the *cross-patient*
feature distributions of *all* real and *all* synthetic volumes. $W_1$
here is paired; FID is not.

#### 4.5.6 ROI-anchored enhancement preservation (M6)

The four anatomical ROIs that normally enhance on T1c — choroid plexus
(CP), pituitary gland (PIT), pineal gland (PIN), dural venous sinuses
(DVS) — are extracted from $T_{1\text{pre}}$ using FastSurfer-LIT
(Henschel et al. 2022, *Imaging Neuroscience* DOI:
10.1162/imag_a_00005) for CP, PIT, PIN and a Frangi-on-T1pre + atlas-
prior fusion for DVS (the SuperSinus atlas from Bernier et al. 2018,
*Frontiers in Neuroinformatics* 12:55, DOI: 10.3389/fninf.2018.00055,
mapped to SRI24 via ANTs at preprocessing time). Both extractions run
on $T_{1\text{pre}}$ — the *non-contrast* sequence — so the ROI
delineation is identical for real and synthetic T1c and contains no
T1c-derived information that could leak the answer.

**Metric.** Per-ROI mean intensity correlation:
$\rho_r = \mathrm{corr}\big(\bar{T_{1c}}\big|_r, \bar{\widehat{T_{1c}}}\big|_r\big)$
across Ring-A patients, for $r \in \{\text{CP}, \text{PIT}, \text{PIN},
\text{DVS}\}$. And per-ROI MAE:
$\mathrm{MAE}_r = \mathbb{E}_p |\bar{T_{1c}}\big|_{r, p} -
\bar{\widehat{T_{1c}}}\big|_{r, p}|$ over patients $p$.

*Why this matters separately from the global metrics.* The choroid
plexus and pituitary together occupy < 0.5% of brain voxels; their
contribution to whole-brain MAE is invisible. A model that mis-
synthesises pituitary enhancement could pass every global metric and
still be clinically unacceptable. M6 isolates the small but
clinically-load-bearing ROIs.

#### 4.5.7 Centerline density preservation (M7)

For each operator $\mathcal{O} \in \{F, J, O, V\}$ and for the ensemble
consensus $V_{\text{cons}}$, skeletonise the binary mask (the percentile-
thresholded soft maps for $F, J, O$; VesselFM's native mask for $V$) via
`skimage.morphology.skeletonize_3d`. Report the **total skeleton length
ratio** $L_{\text{syn}, \mathcal{O}} / L_{\text{real}, \mathcal{O}}$ per
volume, per operator. A ratio < 1 indicates the model has dropped fine
vasculature; a ratio > 1 indicates spurious vessel-like structures.
Aggregated across Ring A, this is the "does the model preserve the
*amount* of fine vasculature" metric, complementing M3's "does it
preserve the *topology*" question. Per-operator reporting catches the
case where the model fools one operator but not another (e.g. produces
structures that pass Frangi but fail VesselFM's learned vessel
prototype).

#### 4.5.8 Cross-operator consistency (M8)

A model that genuinely preserves vessel structures should be evaluated
the same way by *every* operator in the panel. M8 is a meta-metric on
the agreement between operators applied to the *synthetic* volume,
benchmarked against the same agreement on the *real* volume.

Per volume, compute the pairwise Cohen's $\kappa$ between the four
binary operator masks (F*, J*, O*, V — six pairs) on the real T1c, and
the same six $\kappa$ on the synthetic T1c. The agreement vectors are
$\boldsymbol{\kappa}_{\text{real}}, \boldsymbol{\kappa}_{\text{syn}}
\in \mathbb{R}^6$.

**Metric.** $\Delta \boldsymbol{\kappa} = \boldsymbol{\kappa}_{\text{syn}}
- \boldsymbol{\kappa}_{\text{real}}$ — six per-pair deltas; we report
their mean and the test of $\Delta \bar{\kappa} = 0$ (paired Wilcoxon)
per competitor. A negative mean indicates the synthetic volume is
*easier* for one operator than another in a way the real volume is not —
i.e. the model has fit *one* operator's vessel prototype rather than
producing structures all four operators can agree on. This is the
operationalisation of "the model gamed Frangi but didn't actually
produce real vessels".

*Why this matters.* It catches a failure mode invisible to M1–M7: a
model that perfectly matches Frangi(real) on the synthetic volume but
that VesselFM rejects as non-vessel-like would score high on M1–M3
under Frangi but be detectable only by comparing operators. M8 is the
robustness audit on the rest of the panel.

#### Summary of background-contrast metrics

| Metric | Primary purpose | Requires labels? | Region | Operators reported |
|---|---|---|---|---|
| M1 — Vesselness-map MAE / ρ | Soft-map fidelity, calibration-free | No | $R_{\text{bg}}$ | F, J, O |
| M2 — Top-q% Dice | "Where does contrast go" | No (per-volume q) | $R_{\text{bg}}$ | operator-free (intensity-based) |
| M3 — clDice on percentile-thresholded operator | Topological vessel preservation | No (mask vs mask) | $R_{\text{bg}}$ | F, J, O, V, ensemble |
| M4 — ΔVCR | Vessel-vs-parenchyma contrast | No (uses real-ensemble as ref) | $V_{\text{cons, real}}$ | ensemble only |
| M5 — $W_1$ per scan | Distributional tail of bright voxels | No | $R_{\text{bg}}$ | operator-free (intensity-based) |
| M6 — ROI mean intensity ρ, MAE | Normally-enhancing-anatomy fidelity | Anatomical atlas only (no vessels) | CP, PIT, PIN, DVS | operator-free (atlas-based) |
| M7 — Skeleton-length ratio | Fine-vasculature retention | No | $R_{\text{bg}}$ | F, J, O, V, ensemble |
| M8 — Cross-operator $\Delta\bar\kappa$ | Robustness to operator-choice; detects "gamed one filter" failure | No | $R_{\text{bg}}$ | all six operator pairs |

**Primary background-contrast endpoint:** M2 Dice at $q = 95$ on
Ring-A pooled. **Secondary endpoints:** M3 ensemble clDice, M4 ΔVCR,
M6 ρ per ROI. M1, M5, M7, M8 are exploratory diagnostics for the
failure-mode taxonomy (§8); M8 is the gaming-resistance check that
keeps the protocol honest if a competitor (or VENA itself) over-fits
to one operator's prototype.

---

## 5. Statistical analysis plan

**Pre-registered before unblinding Ring A.**

### 5.1 Primary endpoint and decision rule

- *H_0:* MAE(VENA, whole-brain, Ring A) ≥ median MAE of the 6 competitors
  on the same data.
- *H_1:* MAE(VENA, whole-brain, Ring A) < median MAE of the 6 competitors.
- Test: **paired Wilcoxon signed-rank** of per-patient MAE differences
  (VENA − competitor), one test per competitor, two-sided, α=0.05.
- Multiplicity: **Holm-Bonferroni** over the 6 paired tests
  (family-wise error rate at 0.05). Holm-Bonferroni is preferred over
  raw Bonferroni because the tests are not independent (same patients,
  same metric) and Holm is uniformly more powerful (Holm 1979,
  *Scandinavian Journal of Statistics* 6:65–70). Higher-order FWER
  procedures (Hochberg, Hommel) require an assumption of joint normality
  on the test statistics that does not hold for paired Wilcoxon.

### 5.2 Effect size and confidence interval

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

### 5.3 Secondary endpoints and exploratory tests

Secondary endpoints (region-restricted MAE/SSIM, LPIPS-RadImageNet, nnU-Net
Dice, clDice) are tested **without** primary-endpoint multiplicity
correction; a separate Holm-Bonferroni family is run *per metric* across
the 6 competitors. Findings on secondary endpoints are reported as
*confirmatory* only when the primary endpoint rejects $H_0$. Otherwise
they are *exploratory* and reported in the supplementary with explicit
language (Bender & Lange 2001, *Adjusting for multiple testing — when and
how?*, **J. Clin. Epidemiol.** 54(4):343–349, DOI:
10.1016/S0895-4356(00)00314-0).

### 5.4 Per-ring stratification

Ring-A primary endpoint is *pooled* (one MAE per patient, all 173
patients). Per-cohort sub-analyses are reported but treated as exploratory.
Ring-B (BraTS-Africa, BraTS-PED) and Ring-C (HRUM) endpoints are
*separately* tested, each with its own family of 6 comparisons and
Holm-Bonferroni correction. The Ring-B/C comparisons answer the
generalisation question; they are pre-registered as *secondary*.

### 5.5 Power

We have power to detect an effect size Cliff's δ ≥ 0.15 (small-to-medium)
on Ring-A with n=173 paired observations at α=0.05 / 6 = 0.0083 and
β=0.20. Power computed via the standard Mann-Whitney power approximation
(Noether 1987, *J. Am. Statist. Assoc.* 82:645–647): the asymptotic
relative efficiency of Wilcoxon vs t-test is 0.955 under near-normal
residuals; for n=173 paired, the detectable effect size at α=0.0083
β=0.80 is δ ≈ 0.13. Sufficient.

---

## 6. Why FID is demoted to a sanity check

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

## 7. Counterfactual / shortcut-learning diagnostics

These tests are mandatory before submission. A model that wins on the
primary endpoint but fails them is not publishable in good faith.

### 7.1 Healthy-control hallucination test

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

### 7.2 Non-enhancing-glioma test (HRUM Tier 2)

HRUM cohort stratifies glioma into enhancing and non-enhancing per the
radiologist read at acquisition. On the non-enhancing subset, real
$T_{1c}$ shows no Gd uptake; the synthesised $\widehat{T_{1c}}$ must
match. The metric is the same false-positive-enhancement volume restricted
to the BraTS-WT mask provided by the nnU-Net segmenter. Decision: same
gate as §7.1.

### 7.3 Mask-ablation invariance test

For every Ring-A patient, run inference twice: once with the real
$M_{\text{WT}}$, once with an all-zero mask. The difference $\Delta_M =
\widehat{T_{1c}}(\text{real mask}) - \widehat{T_{1c}}(\text{zero mask})$
*should* concentrate inside the dilated WT region and *should not* leak
into the rest of the brain. This is the operational test of H2 of the
proposal at inference time (the loss-side version of the test).
Quantify: $(\sum_{\text{brain} \setminus M_{\text{WT}}} |\Delta_M|) /
(\sum_{\text{brain}} |\Delta_M|)$. Decision: ratio < 0.10 for the
proposed model and not worse than C6-TumorFlow (the controlled
no-contrastive comparator).

---

## 8. Failure-mode taxonomy and qualitative reporting

The supplementary must include per-cohort residual-map figures
($T_{1c} - \widehat{T_{1c}}$) overlaid with $M_{\text{WT}}$, Frangi
vessel mask, and brain mask, for *every* method on the same 8 patients
(2 best, 4 median, 2 worst by primary-endpoint MAE). One figure per
qualitative failure category from Mallio et al. 2023:

1. Vessel under-enhancement.
2. Non-tumour over-enhancement (pituitary, choroid plexus, meninges).
3. Enhancement-pattern blur inside the WT (ring vs solid vs ribbon-of-fire).
4. Vendor-induced contrast drift (Ring-C only; Siemens / Philips vs GE).
5. Pediatric-specific failures (Ring-B BraTS-PED: smaller heads,
   different myelination contrast).

The taxonomy follows the failure-mode-vocabulary tradition of Pamb et al.
(2024, IEEE JBHI) and the SegMen-style consensus in Kervadec et al.
(2021) *Boundary loss for highly unbalanced segmentation*, **Medical
Image Analysis** 67:101851 — semantically labelled failures, not raw
"this looks wrong".

---

## 9. Reader study (deferred to Ring C)

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

The reader study is *not* a substitute for §5's quantitative endpoints; it
is the clinical-significance complement.

---

## 10. Reproducibility checklist (MICCAI / MedIA expectation)

- All competitor checkpoints + VENA's final checkpoint released under
  Apache 2.0 (model code) + NVIDIA OneWay Non-Commercial (trunk weights;
  forwarded as a `links.txt` per `external-deps.md`).
- Test-set patient ID lists pinned in `artifacts/validation/<UTC>/`.
- Inference script per method (six wrappers + VENA's), each callable as
  `python -m benchmarks.<method>.infer <input_h5> <output_h5>`.
- Metrics script callable as
  `python -m benchmarks.metrics.compute <output_h5> <real_h5> --regions
  brain,wt,bg --metrics mae,rmse,psnr,ssim,msssim,lpips,cldice,vcr`.
- Each method's training command, conda env, container, and
  CUDA / PyTorch versions logged in a `decision.json` per re-trained
  competitor; together they form a single `benchmarks/decision.json`
  artifact at submission time.

---

## 11. Timeline (mapped onto proposal §10)

| Phase | Weeks | Deliverable |
|---|---|---|
| Re-implement / port C1, C2, C3 (icon-lab tier) | 1.0 | Three wrapper scripts that train and infer on our H5 schema |
| Re-implement / port C4 (3D-DiT) on MAISI latents | 1.0 | Wrapper script; same VAE / conditioning interface as VENA |
| Re-implement / port C5 (T1C-RFlow) | 1.0 | Wrapper script |
| Re-train all 5 competitors on Picasso | 2.5 | Trained checkpoints + per-method `decision.json` |
| Stand up vessel-extraction operator suite (F, J, O, V) | 0.5 | Reusable `vena.eval.vesselness` module wrapping `skimage`, `pyvane`, and VesselFM; unit tests on synthetic tubular phantoms |
| **VesselFM spot-check** on 20 UCSF-PDGM real T1c val volumes | 0.3 | Pass/fail report + decision: ensemble size 4 (F, J, O, V) if pass, 3 (F, J, O) if fail; either outcome reportable in paper §methods |
| *(optional)* VesselFM one-shot fine-tuned variant + supplementary re-run | 0.4 | Sensitivity table; addresses the "did you cherry-pick zero-shot?" reviewer comment |
| Ring-A evaluation pass (M1–M8 ×6 methods) | 0.5 | `benchmarks/ring_a/results.csv` |
| Ring-B evaluation pass | 0.5 | `benchmarks/ring_b/results.csv` |
| Counterfactual / shortcut diagnostics | 1.0 | False-positive enhancement tables; mask-ablation ratios |
| Statistical analysis pass (pre-registered plan) | 0.5 | Headline table + per-metric breakdowns + figures |
| Failure-mode qualitative pass | 0.5 | Supplementary residual-map figures |
| **HRUM Ring-C (deferred until data lands)** | 2.0 | Ring-C results + reader-study results |
| Writing | 4.0 | MedIA / IEEE TMI submission |

Compresses to ~9 weeks of effective work before HRUM, ~15 weeks
end-to-end once HRUM Tier 1+2 lands. (Was 6 competitors / ~10 weeks in
the 2026-06-08 morning draft; tightened after dropping CFM and
TumorFlow.)

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

## References

(Full bibliography in `literature.md`; this document cites by author-year.
The pre-registered analysis plan and patient-ID partitions are deposited
at `artifacts/validation/<UTC>/` at submission day, hash recorded in
each competitor's `decision.json`.)

- Adewole et al. 2023, arXiv:2305.19369 — BraTS-Africa.
- Bender & Lange 2001, **J. Clin. Epidemiol.** 54(4):343–349, DOI:
  10.1016/S0895-4356(00)00314-0 — multiple-testing communication.
- Bińkowski et al. 2018, arXiv:1801.01401 — KID.
- Borji 2022, **CVIU** 215:103329, DOI: 10.1016/j.cviu.2021.103329 — GAN-
  evaluation review.
- Chong & Forsyth 2020, **CVPR**, DOI: 10.1109/CVPR42600.2020.00611 —
  FID bias.
- Cliff 1996 — Cliff's δ; *Ordinal Methods for Behavioral Data Analysis*.
- Dar et al. 2019, **IEEE TMI** 38(10):2375–2388, DOI:
  10.1109/TMI.2019.2901750 — pGAN.
- Dalmaz et al. 2022, **IEEE TMI** 41(10):2598–2614, DOI:
  10.1109/TMI.2022.3167808 — ResViT.
- Geirhos et al. 2020, **Nature Machine Intelligence** 2:665–673, DOI:
  10.1038/s42256-020-00257-z — shortcut learning.
- Holm 1979, **Scandinavian Journal of Statistics** 6:65–70 — Holm-
  Bonferroni.
- Isensee et al. 2021, **Nature Methods** 18:203–211, DOI:
  10.1038/s41592-020-01008-z — nnU-Net.
- Kazerooni et al. 2023, arXiv:2305.17033 — BraTS-PED.
- Maier-Hein et al. 2018, **Nature Communications** 9:5217, DOI:
  10.1038/s41467-018-07619-7 — biomedical ranking rigor.
- Mallio et al. 2023, **Frontiers in Neuroimaging** 2:1055463, DOI:
  10.3389/fnimg.2023.1055463 — failure-mode review.
- Mei et al. 2022, **Radiology AI** 4(5):e210315 — RadImageNet.
- Noether 1987, **J. Am. Statist. Assoc.** 82:645–647 — Wilcoxon power.
- Özbey et al. 2023, **IEEE TMI** 42(12):3524–3539, DOI:
  10.1109/TMI.2023.3290149 — SynDiff.
- Pamb et al. 2024, **IEEE JBHI** — *Beyond PSNR and SSIM* survey.
- Preetha et al. 2021, **Lancet Digital Health** 3(12):e784–e794, DOI:
  10.1016/S2589-7500(21)00205-3 — downstream-task protocol.
- Reinke et al. 2024, **Nature Methods** 21(2):182–194, DOI:
  10.1038/s41592-023-02151-z — image-processing-metric limitations.
- Shit et al. 2021, **CVPR**, DOI: 10.1109/CVPR46437.2021.01629 — clDice.
- Wang et al. 2003, IEEE — MS-SSIM.
- Wang et al. 2004, **IEEE TIP** 13(4):600–612, DOI:
  10.1109/TIP.2003.819861 — SSIM.
- Zhang et al. 2018, **CVPR**, DOI: 10.1109/CVPR.2018.00068 — LPIPS.
