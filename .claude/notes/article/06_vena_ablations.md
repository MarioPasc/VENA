# Study 6 — VENA ablations

**Paper §4.7 · Priority: secondary · Data status: ⚠️ trained-variants ready,
retraining ablations planned · Scope: Ring A**

References `00_HUB.md` §2. The **mask on/off ablation is the paper's most
important single result** (it bounds the headline) and is already computed; the
rest of this doc scopes what is ready now vs what needs new compute (your Q3
answer: *scope now, decide per-ablation later*).

---

## 1. The ablation axes

| # | Axis | Comparison | Status | Result (if known) |
|---|---|---|---|---|
| A1 | **Mask conditioning** | v3a (none) → v3b (ControlNet on GT [NETC,ED,ET]) | ✅ trained | **−0.0335 MAE_wt, +0.166 SSIM_wt, ~0 whole-brain.** The central result. |
| A2 | **Region-weighted loss** | v3b → v3b-rw (ET ×300, NETC ×50, ED ×50) | ✅ trained | **Net negative**: costs MAE_brain (p=0.0022) & SSIM_brain (p<1e-5); tumour **n.s.** (MAE_wt p=0.997) despite ET weighted 300×. |
| A3 | **Decoder perceptual loss (LPL)** | v3b-rw → S3-LPL-b2c | ✅ trained | **Inert**: MAE_brain p=1.00, MAE_wt p=0.997. Confirms the stored decision to stop LPL. |
| A4 | **NFE / sampling budget** | v3a & v3b-rw at NFE 1/2/5/10/20 | ✅ trained | flat past NFE=5 (see Study 2). |
| A5 | **Predicted-mask (deployable)** | v3b-rw(GT) → v3b-rw(predicted WT) | 🔨 **planned (T6.5)** | — the honest middle between oracle & no-mask. |
| A6 | **Input modalities** | T1pre / +T2 / +FLAIR / +SWAN | 🔨 candidate (T6.6) | isolates each input's marginal value; **is SWAN worth it?** |
| A7 | **SWAN encoding** | SWAN-as-input vs mask-only vs none | 🔨 candidate | proposal §7 axis; tests the original vessel premise. |
| A8 | **Target parameterisation** | direct $T_{1c}$ vs residual $T_{1c}-T_{1pre}$ | 🔨 candidate | proposal §7 axis. |

**A1–A4 are the shippable ablation table (zero new compute).** A5–A8 are the
"candidate retraining" backlog, prioritised in §5.

## 2. Claims (from trained variants)

- Mask conditioning buys tumour fidelity **only** (whole-brain unchanged) — and
  because that mask is oracle GT, the *buy* is an upper bound (cross-ref Studies
  1, 4). This is the cleanest statement of VENA's mechanism.
- Region weighting and LPL are **honest negatives** — reported, not buried. They
  strengthen the paper's credibility (MedIA rewards this).
- **Sharpest single finding:** weighting the enhancing region **×300** in the loss
  did *not* measurably improve tumour fidelity (MAE_wt p=0.997) while it *cost*
  whole-brain fidelity. The tumour bottleneck is therefore **representational
  (the MAISI latent / conditioning), not loss-weighting** — motivates the VAE-floor
  test (Study 7) and the TC/ET re-scoring (Study 8), not more loss tuning.

## 3. Data — paths (A1–A4)

**Source:** `paired_fidelity/LATEST/per_scan/paired_fidelity_patient.csv` — all of
v3a, v3b, v3b-rw, S3-LPL are present at all NFEs. Ablation **Holm family = 3**
(v3b, v3a, S3-LPL vs v3b-rw), separate from the competitor family (§2.3). Also
pull A1/A2/A3 tumour deltas from Studies 1/4/5 so the ablation table and the
headline table cannot disagree.

## 4. Tables & figures (A1–A4)

**Table 6 — ablation ladder.** Rows: v3a → v3b → v3b-rw → S3-LPL (in build
order). Columns: MAE_brain, SSIM_brain, MAE_wt, SSIM_wt, ΔDice_ET (from Study 4),
ρ_S (from Study 5), each with Wilcoxon vs the previous rung + Cliff's δ. One table
tells the whole "what each design choice bought" story.

**Fig 6 — component-effect bars:** for each axis A1–A3, the signed effect on
{whole-brain, tumour} with CI — shows A1 helps tumour only, A2/A3 help nothing.

## 5. Planned ablations — specs & go/no-go (A5–A8)

### T6.5 — Predicted-mask v3b-rw (your requested addition; **highest-value planned**)
The deployable question: *without the GT mask, how much of v3b-rw's tumour gain
survives?* Pipeline:
1. Run a tumour segmenter on **pre-contrast inputs only** ({T1pre,T2,FLAIR}) →
   predicted WT mask. Candidate: the same BraTS SegResNet (Study 4) restricted to
   pre-contrast channels, or a purpose-trained pre-contrast WT segmenter.
2. **Retrain** VENA v3b-rw with the *predicted* mask as conditioning (not GT).
3. Re-infer on Ring A + Ring B; re-run fidelity + downstream_seg.
Result placement: a third row between v3a (lower) and v3b-rw-GT (upper) in
Studies 1 & 4. **Cost: 1 training + 1 inference sweep. Blocker: user go + segmenter choice (Hub §5.2).**

### T6.6 — Input-count (candidate)
Train {T1pre}, {T1pre,T2}, {T1pre,T2,FLAIR}, {+SWAN}. Isolates each modality's
marginal MAE/SSIM and **directly tests whether SWAN earns its place** — the
residue of the original vessel thesis. Cost: up to 4 trainings. Defer unless the
SWAN question is deemed reviewer-critical.

### T6.7 — SWAN-encoding / target-parameterisation (candidate, proposal §7)
Lower priority; only if a reviewer presses on the design space.

## 6. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "You only ablate what helps you" | We report A2 (rw) and A3 (LPL) as **negatives**; the ablation table shows losses plainly. |
| "GT mask is unrealistic" | A5 predicted-mask arm (if run) gives the deployable number; if deferred, state it as the key limitation + future work with the exact protocol above. |
| "Is SWAN even useful?" | A6 input-count answers directly; if deferred, acknowledge as open and cite the vessel-fidelity (Study 5) as the current SWAN-motivated evidence. |
| "n=3 ablation family, weak correction" | Pre-registered family; Holm within it; effect sizes + CIs reported regardless of significance. |

## 7. Task checklist

- [ ] **T6.1** Table 6 ablation ladder (A1–A4) from per_scan + Studies 4/5 numbers.
- [ ] **T6.2** Fig 6 component-effect bars.
- [ ] **T6.3** Verify A1/A2/A3 numbers match Study 1 & HANDOFF §4a–4e exactly.
- [ ] **T6.5 (planned)** Predicted-mask arm — **await user go + segmenter choice**.
- [ ] **T6.6 (candidate)** Input-count ablations — await compute decision.
- [ ] **T6.7 (candidate)** SWAN-encoding / target parameterisation.
