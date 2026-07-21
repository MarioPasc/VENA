# Study 9 — Shortcut diagnostic / healthy-control safety (§6.5)

**Paper §4.8 · Priority: optional · Data status: ⛔ blocked — control cohort not
on disk · Scope: healthy controls (to source)**

References `00_HUB.md` §2 and proposal §6.5. This is the **safety** experiment: it
answers the question a clinical reviewer *must* ask — *does the model hallucinate
enhancement where there is none?* It is under-specified by design (Q3: scope now,
detail later) and blocked on data, but the protocol is fixed here so it is ready
the moment a cohort lands.

---

## 1. Question & why it matters

A gadolinium-free synthesiser is dangerous if it **invents contrast enhancement**
— a false enhancing focus reads as tumour/recurrence. The failure mode is
learnable: the model may key on SWAN dark voxels or vascular cues and paint
enhancement. The only way to measure it is on subjects with a **known-negative
ground truth**: healthy controls (or confirmed non-enhancing follow-ups), where
the correct synthetic $T_{1c}$ shows **no** pathological enhancement.

**Claim shape:** false-positive enhancement volume on controls ≈ 0 mL (or,
honestly, "X mL, concentrated in vessels/choroid" — which connects to Study 5).

## 2. Protocol (fixed; from proposal §6.5 + `preflights/shortcut_diag` schema)

- **Cohort:** healthy adults (or non-enhancing scans) with pre-contrast
  {T1pre,T2,FLAIR,SWAN} and a real $T_{1c}$ that is **enhancement-negative**.
  `ground_truth_enhancement = "none"`.
- **Metric:** `false_positive_enhancement_volume_ml` — volume of voxels where
  synth $T_{1c}$ exceeds a contrast-enhancement threshold but real $T_{1c}$ does
  not (per proposal §6.5). Report per method; also a spatial map (where do the
  false positives land — vessels? deep GM? matches Study 5's structures?).
- **Comparators:** all 16 methods (the shortcut is not VENA-specific; C0-Identity
  is the trivial null — it *cannot* hallucinate because it copies T1pre, so it is
  the 0-mL reference; any generative method above it is inventing signal).
- **Feasibility gate:** the `shortcut_diag` preflight's `protocol_feasible` flag
  parks the study until a control cohort exists.

## 3. Data — blocker

- **Needs:** a healthy/non-enhancing control cohort **on disk** with the input
  modalities + a negative real $T_{1c}$. **Not currently available** (the
  `shortcut_diag` preflight `control_cohort_path` is unset).
- **Options to unblock (Hub open decision):**
  1. Source a public healthy cohort with post-contrast T1 (rare — most healthy
     datasets have no gadolinium).
  2. Use **non-enhancing tumour follow-ups** from the existing cohorts (e.g.
     LUMIERE longitudinal timepoints radiologically read as non-enhancing) — a
     pragmatic proxy already partially on disk.
  3. Use **contralateral healthy hemispheres** of existing patients as a
     within-subject negative control (cheap; imperfect — global model behaviour
     may differ).
- **Inference:** once a cohort lands, reuse the frozen prediction pipeline (no
  retraining) — same as every other study.

## 4. Placement

A short **safety** subsection (§4.8) or a **Limitations**-adjacent result. Even a
small pilot (n≈10–12, proposal's `n_controls`) is publishable as "we tested for
the enhancement-hallucination failure mode; VENA's false-positive volume is X mL,
concentrated in Y". A **null protocol_feasible** is itself reportable: "the
diagnostic is specified; a control cohort is being sourced (future work)."

## 5. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "Is it safe to deploy?" | This study is the direct answer; report false-positive-enhancement mL per method. |
| "No healthy post-contrast data exists" | Option 2/3 proxies (non-enhancing follow-ups / contralateral) stated with their limitations. |
| "C0 trivially passes" | That's the point — C0 is the 0-mL reference; generative methods are judged against it. |
| "Where do false positives go?" | Spatial map; cross-reference Study 5 (vessels/choroid are the usual bright-structure culprits). |

## 6. Task checklist

- [ ] **T9.1** (user/PI) Decide the control source (public / non-enhancing follow-ups / contralateral).
- [ ] **T9.2** Land the cohort in the corpus (image H5 + registry) via the `add-dataset` recipe.
- [ ] **T9.3** Run the frozen prediction pipeline on controls (all 16 methods).
- [ ] **T9.4** Implement `false_positive_enhancement_volume_ml` + spatial map; run the `shortcut_diag`-style analysis.
- [ ] **T9.5** Report per-method mL + spatial map; tie to Study 5.
