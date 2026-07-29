# Retraining v3a — why, and the exact recipe changes

**Written:** 2026-07-29 · **Author:** orchestrator session (`/orchestrate`, Opus @ xhigh)
**Companion doc:** [`S3_ORACLE_VERDICT.md`](S3_ORACLE_VERDICT.md) — the S3 oracle-matrix analysis this follows from.

**Subject run.**
`/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/`

Every configuration value quoted below was read from that run's own `config.yaml`; every measurement was
re-derived from its `exhaustive_val/epoch_*/metrics.csv` and `metrics/train_epoch.csv`, or from
`/media/mpascual/Sandisk2TB/research/vena/results/article/paired_fidelity/2026-07-20T12-11-10Z/tables/`.
Line references are to the VENA repo at the commit current on 2026-07-29.

> **Read §17 before quoting any absolute number from §1–§3 or §11–§15.** Every epoch-mean in this document is an
> unweighted mean over `metrics.csv` rows, and that aggregate turns out to be **56 % LUMIERE and 6 % UCSF-PDGM**,
> with **16 % drawn from `test_only` cohorts' test splits**. Paired comparisons (v3a vs A6; retained epoch vs peak
> epoch within a run) are unaffected — the confound is identical on both sides and cancels. Absolute values and
> argmax epochs are not.
>
> **If you are implementing this document, start at [§20](#20-bug--defect-register--the-implementation-checklist).**
> It is the consolidated checklist: 16 confirmed defects with `file:line` and fix, 4 verified non-defects that must
> **not** be "fixed", 5 Picasso launch traps, and the assertion block to add to the run log.

---

## 0. Should v3a be retrained at all?

**Yes — but not for the reason first proposed, and not urgently.**

v3a is the warm-start anchor for the entire S3 oracle matrix (all five arms load
`checkpoints/ema_best.ckpt` from the path above). Replacing it invalidates every S3 comparison, so this is not a
free action. The case for doing it rests on four independent findings, in descending order of force:

1. **The checkpoints worth having no longer exist** (§1). This is irreversible and cannot be fixed by re-analysis.
2. **Classifier-free guidance cannot be retrofitted** (§4). The single cheapest untried lever on the actual
   failure mode requires the base model to have been trained with conditioning dropout. It was not.
3. **Roughly half the compute was spent past the run's own quality peak** (§3), so a retrain is cheaper than the
   original — not more expensive.
4. **The evaluation that judged v3a was mis-normalised** (§6), so its archived quality curve understates it by
   1.7–2.1 dB and cannot be used as a baseline.

**Against:** the S3 arms are already trained and analysed; a retrain restarts that. The recommendation is
therefore: retrain **when** the next training generation is launched anyway (i.e. fold these changes into that
launch), **not** as a standalone job to replace v3a in place.

---

## 1. The retention policy destroyed the checkpoints that mattered

### What the config says

`.../2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/config.yaml`:

```yaml
training:
  checkpoint_every_epochs: 25      # line 93
output:
  retention_n_checkpoints: 3       # line 136
```

v3a checkpointed **every 25 epochs** across a 1900-epoch run — about 76 checkpoints were written. `retention_n_checkpoints: 3`
kept three.

### What survives on disk

`.../2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/checkpoints/`:

```
ema_best.ckpt          2755.4M
ema_epoch_1649.ckpt    2755.4M
ema_epoch_1799.ckpt    2755.4M
ema_epoch_1849.ckpt    2755.4M
last.ckpt              2755.4M
trunk_ema_snapshot.pt   688.8M
```

### Why that is the costly part

Re-derived from `.../exhaustive_val/epoch_*/metrics.csv` (NFE=20, mean over 116 patients):

| quantity | peak value | peak epoch | value at 1900 |
|---|---|---|---|
| whole-volume PSNR | 25.331 dB | **525** | 25.145 dB |
| PSNR_ET | 16.920 dB | **900** | 16.539 dB |

**Neither epoch 525 nor epoch 900 exists on disk.** They were written and then deleted. No amount of later
analysis recovers them, and no re-selection of the "right" metric can be applied retrospectively — which is
exactly the situation §2 describes.

### Change

```yaml
output:
  retention_n_checkpoints: 40      # was 3 — keep the every-25-epoch series for the whole run
```

76 checkpoints × 2.755 GB ≈ 209 GB; at 40 ≈ 110 GB. Against ~4 A100-days per run this is the cheapest insurance
in the project. If storage is genuinely tight, raise `checkpoint_every_epochs` to 50 and keep 40 — still full
coverage of the run at half the footprint.

---

## 2. The checkpoint-selection signal is training loss — and the config knob that looks like it controls this is dead

### The correction

An earlier draft of the S3 analysis claimed `ema_best` was selected on *background-region latent MSE*. **That was
wrong**, and the error is recorded here rather than quietly dropped.

v3a's `config.yaml` contains:

```yaml
training:
  best_metric_name: mse_latent     # line 95
  best_metric_region: bg           # line 96
  best_metric_nfe: 5               # line 97
```

These look decisive. They are not. `src/vena/model/fm/lightning/callbacks/checkpointing.py:58` would assemble
them into the monitor key `val/mse_latent_bg_nfe5` — but:

- `routines/fm/train/engine.py:1464` **hard-codes** `ckpt_monitor = "train/total_epoch"` and passes it explicitly
  as `monitor_key=` to both `VENACheckpointCallback` (line 1466) and `BestCheckpointCallback` (line 1475),
  overriding the `best_metric_*` defaults; and
- v3a sets `validation.every_epochs: 0` (line 104) — in-process validation is offloaded to the async job, so **no
  `val/*` metric is ever logged**. The key those three YAML fields build could never have resolved.

So `ema_best` is selected on the **epoch-aggregated training loss**, exactly as `model-coding-standards.md` rule 5
already documents. The three `best_metric_*` keys are vestigial: editing them changes nothing.

### Why it still matters

Training loss is not synthesis quality. The consequence is measurable. Retained checkpoint epochs versus each
run's PSNR_ET-optimal epoch (NFE=20, re-derived from the cadence CSVs):

| run | PSNR_ET-best epoch | retained epochs | PSNR_ET forfeited |
|---|---|---|---|
| J1 | 1450 → 20.251 dB | 1350 / 1400 / 1475 | −0.118 dB |
| J2 | 1050 → 20.562 dB | 1775 / 1800 / 1850 | −0.164 dB |
| J3 | 1025 → 20.524 dB | 1775 / 1800 / 1850 | −0.198 dB |
| J4 | 750 → 20.764 dB | 1775 / 1800 / 1850 | −0.279 dB |

The selector consistently retains epochs 700–1100 **after** the quality peak, on the declining part of the curve.
The forfeited 0.12–0.28 dB is the same order as the entire benefit of raising the tumour loss weight 20×
(+0.41 dB) — i.e. comparable to the headline effect of the whole S3 ablation, lost to a monitor choice.

*(The measurement stands; only its attribution changed. It is train-loss selection, not background-region
selection.)*

### Changes

1. **Delete or wire up `best_metric_{name,region,nfe}`.** A configuration knob that appears live and is inert is a
   trap — it cost this analysis a wrong conclusion. Either remove the three fields, or have the engine consult
   them and fail loudly when the referenced metric is not logged.
2. **Select on an exhaustive-val metric**, not train loss — the async job already computes `psnr_db_et` per
   cadence epoch. Feeding it back requires the training process to read the async artifact; if that coupling is
   unwanted, §1's retention fix is the pragmatic substitute (keep everything, choose afterwards).
3. Note that under the perception–distortion reading (§7) **any** PSNR-type selector prefers the *dimmer* model.
   This is a further reason to keep checkpoints and defer the choice rather than to hunt for a better single
   scalar.

---

## 3. Half the compute ran past the quality peak — and `max_epochs` is not the reason

### The mechanism

```yaml
run:
  max_epochs: 10000                # line 90 — v3a never approached this
training:
  patience: 250                    # line 99
```

`routines/fm/train/engine.py:1526` attaches `EarlyStopping(monitor=ckpt_monitor, ...)` — the **same**
`train/total_epoch`. ~~v3a stopped near epoch 1900 because the *training loss* stopped improving for 250 epochs.~~

> **CORRECTION 2026-07-29 (user-verified from run artifacts):** The §3 mechanism claim above is wrong. v3a ran
> to epoch **1923** and stopped because it hit `training.total_steps: 400000` exactly (`logs/train.log` final line:
> `epoch 1923 done | step=400000`). Grep of the full log finds **no EarlyStopping message anywhere**. Furthermore,
> `metrics/train_epoch.csv` shows `total_mean` reached its minimum at epoch **1862** of 1923 and improved by 0.019
> over the final 250 epochs — patience 250 was never close to firing, and `max_epochs: 10000` was never binding
> either. `total_steps` is the **only knob that has ever terminated a run in this project.**
>
> This actually *strengthens* the §3 finding: if `train/total_epoch` improves monotonically and EarlyStopping never
> fires, it cannot serve as a convergence signal at all — not even in principle. The original text's conclusion
> (checkpoint selection must be post-hoc over validation quality) is correct; the claimed mechanism is not.
> B17 (termination-reason logging added to `decision.json` schema 0.12.0) permanently closes this class of
> misattribution.

Training loss kept creeping downward long after validation quality turned over, so the run continued.

### The cost, across every run measured

| run | epochs run | whole-PSNR peak | PSNR_ET peak | share of run after the ET peak |
|---|---|---|---|---|
| v3a | 1923 (hit total_steps=400000) | 525 | 900 | **53 %** |
| J2 | 1900 | 875 | 1050 | 45 % |
| J3 | 1900 | 1175 | 1025 | 46 % |
| J4 | 1900 | 1000 | 750 | **61 %** |

Every run ended 0.09–0.38 dB PSNR_ET *below* its own peak. This is not merely wasted compute — the extra epochs
made the retained model slightly worse.

### Change

Do **not** simply set `max_epochs: 1000` — that hard-codes a guess. Point EarlyStopping at a validation signal
(as in §2) so the run stops when *quality* plateaus rather than when *training loss* does. If the coupling to the
async job is too invasive for now, `max_epochs: 1100` with retention at 40 is a safe interim: it covers every peak
observed across five runs (525–1175) and frees roughly two A100-days per run.

---

## 4. Conditioning dropout — the one change that cannot be retrofitted

```yaml
training:
  conditioning_dropout_p: 0.0      # line 100
  conditioning_dropout_keys:       # line 101
  - wt
```

The infrastructure exists and is switched off. It has been off in **every run to date** — v3a, all five S3 arms,
and the v3b siblings.

### Why this is the highest-value line in the file

The measured failure is that VENA under-renders enhancement *intensity*. From
`.../paired_fidelity/2026-07-20T12-11-10Z/tables/tableS1_undersaturation.csv` (read verbatim), the 99.5th
percentile of each method's own output:

| method | `raw_p995_mean` |
|---|---|
| C3-SynDiff-flair | 0.9011 |
| **VENA-S1-v3b** | 0.8421 |
| C2-ResViT | 0.8228 |
| **VENA-S1-v3a** | **0.78582** |
| **VENA-S1-v3b-rw** (oracle mask) | **0.78558** |
| C4-3D-DiT | 0.3972 |

A *perfect* tumour mask changes VENA's intensity tail by **−0.03 %** (0.78582 → 0.78558).

Classifier-free guidance is the standard remedy for a model under-using its conditioning: it extrapolates
`v_uncond + w·(v_cond − v_uncond)`, pushing samples away from the unconditional mean. Its canonical *artifact* in
the natural-image literature is **over-saturation** — the exact inverse of VENA's deficit. And once the base is
trained with dropout, `w` is a **free, sweepable inference-time knob**: no retraining per value, and the sweep
itself yields a `raw_p995`-versus-`w` dose-response curve that is a publishable figure whichever way it points.

Adding it later means retraining the base. That is precisely the position this document exists to describe.

### Change

```yaml
training:
  conditioning_dropout_p: 0.1
  conditioning_dropout_keys: [ ... ]   # for concat_only: the modality concat channels
```

Apply guidance on a **middle** interval of the noise schedule at inference, not at high noise
(Kynkäänniemi et al. 2024, NeurIPS, arXiv:2404.07724 — guidance at high noise distorts the prior and degrades
quality). See §7 for why the high-noise regime needs a *training*-side fix instead.

---

## 5. Log intensity statistics in the cadence CSV from day one

v3a's `exhaustive_val/epoch_*/metrics.csv` carries 46 columns: PSNR, SSIM, MAE and MSE for whole/WT/BG/NWT/ET/
NETC/ED/BNWT, latent MSE/L1/cosine, timings, voxel counts. It carries **no intensity statistic** — no percentile,
no signed bias, no region mean.

That single omission is why answering "does the model render the rim at the right brightness?" required a
separate A100 job decoding `latent_preds.h5` (job 1679496, 2026-07-29), rather than a `groupby` over CSVs that
already existed. The decoded volume is in memory at that point in `routines/fm/exhaustive_val/engine.py`, so the
marginal cost of recording these is effectively zero.

### Change — add to the exhaustive-val CSV

```
p995_pred_brain, p995_real_brain,
mean_et_pred,    mean_et_real,
mean_bnwt_pred,  mean_bnwt_real
```

`mean_et − mean_bnwt` is the enhancement contrast: the quantity the whole project is ultimately about, and the
one nothing currently records. **Instrument the quantity you will want to ask about, before you need it.**

---

## 6. The evaluation that judged v3a was mis-normalised

Fixed in commit `01350e0` (2026-07-22), *"fix(validation): canonical 99.95 encoder percentile in exhaustive-val"*.
Its message states the mechanism:

> the frozen MAISI latent caches were encoded at `percentile_upper=99.95`, so every decode-vs-real comparison must
> normalise the reference at 99.95 (**99.5 saturates the enhancing-rim/vessel tail**)

v3a ran **2026-06-24**, before the fix. Its archived cadence curve compared predictions in 99.95-space against a
reference clipped at 99.5 — clipping precisely the enhancing rim under study. Measured on the same weights and
the same 116 patients: the whole-volume gap is **1.70 dB** and the ET gap **2.06 dB** (larger, as the mechanism
predicts).

Consequences for a retrain: **automatic** — any new run inherits `ENCODER_PERCENTILE_UPPER = 99.95` from
`vena.common` (`src/vena/validation/io.py:214`). Assert it in the run log rather than assume it.

**Also affected, and not automatic:**
`/media/mpascual/Sandisk2TB/research/vena/results/article/paired_fidelity/2026-07-20T12-11-10Z/` — the current
`LATEST` — predates the fix by two days. `table1_ring_a_fidelity.csv`, `tableS1_undersaturation.csv` and
`tableS2_zgd.csv` were all produced under the rim-clipping normalisation. Within-artifact contrasts measured
identically on both sides (e.g. v3a vs v3b-rw in §4) survive; **absolute ET-region numbers do not**. Regenerate
before any of it reaches a manuscript.

---

## 7. Keep the base free of region weighting

v3a is already correct here:

```yaml
loss:
  cfm:
    reduction: mean                # line 65
    norm: l1                       # line 66
```

Plain mean L1, no region weighting. **Keep the `reduction: mean`** — the evidence below is that region weighting
downstream is harmful. The `norm: l1` half of this block is *reopened* by §12: the argument here concerns region
weighting, and it does not transfer to the choice of norm.

The two ET-mask-conditioned siblings of v3a differ *only* in region weighting:

- `.../results/fm/vena/2026-06-22_15-22-04_s1_v3b_concat_plus_cn3ch_fft_c698f45a/` — no region weighting
- `.../results/fm/vena/2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd/` — `netc: 50, ed: 50, …`

From `table1_ring_a_fidelity.csv` + `tableS1_undersaturation.csv` (Ring A, N=247):

| metric | v3b (no RW) | v3b-rw (RW=50) |
|---|---|---|
| MS-SSIM_wt | **0.8161** | 0.7559 |
| SSIM_brain | **0.5893** | 0.5537 |
| MAE_brain | **0.0915** | 0.0955 |
| `raw_p995` | **0.8421** | 0.7856 |
| MAE_wt | 0.0965 | **0.0948** |
| SSIM_wt | 0.5700 | **0.5737** |

Region weighting improved **only** the median-type metrics (MAE_wt, marginally SSIM_wt) and degraded everything
perceptual plus the whole intensity tail — v3b-rw's `raw_p995` (0.7856) is indistinguishable from *unconditioned*
v3a (0.7858), meaning the weighting erased the entire brightness gain that conditioning alone had produced.

This is the perception–distortion tradeoff (Blau & Michaeli, CVPR 2018, arXiv:1711.06077) visible inside VENA's
own arms: an L1 objective minimises to the conditional median, and region weighting makes the model fit that dim
median *more accurately*. It also explains why the decoder-perceptual-loss ablation
(`.../results/fm/vena/ablations/lpl_module/2026-06-28_11-16-50_s3_v3b_rw_lpl_k5_standard_fft_s1warm_5dd0fe5b/`)
added nothing: perceptual losses are engineered to be **invariant to global intensity shifts**, so LPL was a tool
built to ignore brightness applied to a brightness deficit.

**Open:** whether the same monotone degradation appears across the S3 arms J1(tc=1) → J4(tc=20). Job 1679496 is
measuring it. Until it reports, this section is supported by the v3b pair only.

---

## 8. Leave the timestep distribution alone for now

```yaml
rflow:
  use_timestep_transform: true     # line 71
  base_img_size_numel: 129024      # line 72
```

There is a plausible argument that low-frequency amplitude — i.e. rim brightness — is fixed at **high** noise
(Rissanen et al., ICLR 2023, arXiv:2206.13397), while the SD3-style transform (Esser et al., ICML 2024,
arXiv:2403.03206) concentrates sampling at α ∈ [0.3, 0.7] and the retired LPL gate sat at `t_dn > 0.4` ⇒ α < 0.6.
If true, the regime that sets brightness is systematically under-supervised from two directions at once.

**This is a hypothesis, not a finding.** Do not retune the schedule on it. The discriminating diagnostic is free:
bin α, and per bin measure velocity error inside the tumour core and the ET intensity of the one-step estimate
`x̂₁ = x_t + α·v`. Run that before touching these two lines.

---

## 9. Pipeline hygiene to fold into the same change

- **Hard-guard `load_warm_start`.** `src/vena/model/fm/lightning/module.py:1536-1539` filters to shape-matching
  keys and calls `load_state_dict(..., strict=False)`; `module.py:435-442` only registers `_trunk_module` when
  `trunk.trainable` is true. A `trainable: false` run warm-started from a `trainable: true` checkpoint therefore
  matches **zero** keys, logs `loaded=0` at INFO, and trains to completion on stock weights. This voided arm J0 of
  the S3 matrix — 1 d 21 h of A100 time. Raise when `loaded == 0`.
- **Assert `ENCODER_PERCENTILE_UPPER == 99.95`** in the run log (§6).
- **`PYTHONPATH` must be `<repo>/src:<repo>`**, not `<repo>/src` — `routines/` lives at the repo root, so the
  shorter form silently loads `routines` from the stale shared checkout. Killed job 1679494 on 2026-07-29.

---

## 11. Input modality set — keep T2. Dropping it costs 0.45 dB and collapses the useful training window

*(Added 2026-07-29 in response to the "is t1pre+FLAIR enough?" question.)*

### The ablation is clean

`.../ablations/input_sequence/2026-07-23_09-20-19_s1_v3a_a6_t1pre_flair_fft_d30214f8/` (**A6**) differs from v3a in
**exactly one field**:

```yaml
model.trunk.input_concat.cond_latents:  [t1pre, flair]          # A6   (in_channels: 12)
model.trunk.input_concat.cond_latents:  [t1pre, t2, flair]      # v3a  (in_channels: 16)
```

Everything else is byte-identical: `seed: 1337`, `corpus_picasso.json`, `tau: 0.5`, `fold: 0`, batch 4 × accum 2,
`fft` trainable trunk, `l1` / `mean`, logit-normal + timestep transform, EMA 0.9999, cosine LR 1e-4,
`use_offline_augmented_data: true` with uniform `v0..v4` weights. Both ran 1900 epochs. Both exhaustive-val passes
scored the **same 116 patients** (`n_patients: 60` is per-cohort, two cohorts).

**The normalisation confound does not apply.** A6's commit `1ad2ba4` is *not* a descendant of `01350e0`
(`git merge-base --is-ancestor` returns false), and `git show 1ad2ba4:src/vena/model/fm/eval/exhaustive.py:55`
still reads `percentile_upper: float = 99.5`. Both runs were scored at 99.5. The head-to-head is apples-to-apples;
§6 applies to both equally and to neither differentially.

### Result (NFE=20, mean over 116 patients, re-derived from the cadence CSVs)

| metric | v3a best | A6 best | Δ (A6−v3a) | v3a @1900 | A6 @1900 | Δ @1900 |
|---|---|---|---|---|---|---|
| PSNR whole | **25.331** | 24.885 | **−0.446** | **25.145** | 24.578 | −0.568 |
| SSIM whole | **0.9116** | 0.9054 | −0.0062 | **0.9105** | 0.9026 | −0.0078 |
| PSNR_WT | **16.750** | 16.589 | −0.161 | **16.532** | 16.084 | −0.448 |
| PSNR_ET | **16.920** | 16.733 | −0.187 | **16.539** | 16.174 | −0.365 |
| SSIM_ET | **0.9379** | 0.9364 | −0.0015 | **0.9319** | 0.9240 | −0.0080 |
| MAE_ET | **0.1609** | 0.1614 | +0.0005 | **0.1662** | 0.1732 | +0.0070 |

A6 is worse on **every** metric at both the peak and the endpoint. The single exception is SSIM_WT at peak
(0.9551 vs 0.9526, +0.0025) — 0.14 × the across-epoch SD of that metric, i.e. noise, and it inverts by epoch 1900.

### The larger finding is the shape of the curve, not the gap

| run | PSNR-whole peak | PSNR_ET peak | mean PSNR_ET over ep ≥ 1500 |
|---|---|---|---|
| v3a | epoch **525** (smoothed 925) | epoch **900** | 16.643 |
| A6  | epoch **150** (smoothed 150) | epoch **150** | 16.114 |

A6 reaches its best at epoch 150 and decays monotonically for the remaining 1750 epochs, shedding 0.56 dB whole /
0.56 dB ET. v3a's peak sits 3.5–6× later and its decay is a third as steep. Removing T2 does not merely lower the
ceiling — it removes the model's ability to keep improving. The most parsimonious reading: with two inputs the
trunk saturates the information available in the conditioning and thereafter fits the offline-augmentation noise;
the third modality supplies enough additional signal to keep the optimisation productive an order of magnitude
longer.

### Why T2 carries non-redundant information

T2 and T2-FLAIR are not interchangeable. FLAIR is T2-weighting with CSF nulled by an inversion pulse; suppressing
free water is exactly what makes FLAIR good at peritumoural oedema against a dark CSF background, and exactly what
destroys the information distinguishing **cystic/necrotic tumour components** (free-fluid-like, bright on T2,
suppressed on FLAIR) from solid non-enhancing tumour. UCSF-PDGM is a GBM cohort in which necrotic core (NETC) is
present in the majority of cases, and NETC is one of the regions the exhaustive-val CSV scores separately. Dropping
T2 removes the only sequence that separates fluid from solid non-enhancing tissue — the tissue class immediately
adjacent to the enhancing rim this project is trying to render.

### Change

**None. Keep `cond_latents: [t1pre, t2, flair]`.** Do not adopt the two-modality "field standard".

Two secondary points:

1. The premise "t1pre + FLAIR is the standard" is true of the *nearest competitor* (see §14) but it is not a
   standard the field converged on for a reason — it is what BraTS ships and what a two-channel U-Net was
   convenient with. VENA has the T2 volume already encoded and cached; the marginal cost of the fourth channel
   block is 4 latent channels on the first conv.
2. **A6 is now a publishable ablation row, not a discarded run.** It is the cleanest single-variable input
   ablation in the project (one field, same seed, same 116 scored patients, 1900 epochs each) and it answers a
   question a reviewer will ask. Report it in the proposal §7 input-modality axis as
   "cond = {t1pre, flair} vs {t1pre, t2, flair}: −0.45 dB PSNR, −0.19 dB PSNR_ET, peak epoch 150 vs 525".

---

## 12. The CFM norm — L1 is field-standard, theoretically wrong, and never cleanly tested here

*(Added 2026-07-29. This section revises §7's "keep `norm: l1`" from "settled" to "the one loss question worth an
arm".)*

### The theory says L2, and the reason is not cosmetic

Flow matching's central guarantee (Lipman et al., ICLR 2023, arXiv:2210.02747, Theorem 2) is that the *tractable*
conditional objective has the same gradient as the *intractable* marginal one:

$$\mathcal{L}_{\text{FM}}(\theta)=\mathbb{E}_{t,\,x\sim p_t}\big\|v_\theta(x,t)-u_t(x)\big\|_2^2,\qquad
\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{t,\,z,\,x\sim p_t(\cdot|z)}\big\|v_\theta(x,t)-u_t(x|z)\big\|_2^2,$$

$$\nabla_\theta\mathcal{L}_{\text{FM}}=\nabla_\theta\mathcal{L}_{\text{CFM}},\qquad
u_t(x)=\mathbb{E}_z\!\left[u_t(x|z)\mid x_t=x\right].$$

The proof expands $\|a-b\|_2^2=\|a\|_2^2-2\langle a,b\rangle+\|b\|_2^2$: the cross term is **linear** in $b$, so
conditional expectation passes through it, and $\|b\|_2^2$ is $\theta$-independent and drops from the gradient.
That expansion is the defining property of the squared norm (more generally, of a Bregman divergence). **It does
not hold for $\|\cdot\|_1$.**

Under an L1 objective the pointwise minimiser is the conditional **median**, componentwise:

$$v^\star_\theta(x,t)=\operatorname{median}_z\!\left[u_t(x|z)\mid x_t=x\right]\neq\mathbb{E}_z\!\left[u_t(x|z)\mid x_t=x\right]=u_t(x).$$

So an L1-trained network does not approximate the marginal velocity field. The ODE $\dot x = v_\theta(x,t)$ is not
the probability-flow ODE of $p_t$, and the transport $p_0\to p_1$ carries no guarantee. VENA has been training a
**median regressor that is integrated as if it were a mean**. (Lehmann & Casella, *Theory of Point Estimation*,
2nd ed., Springer 1998, Thm 4.1.2, for the median/mean minimiser identity.)

### And the bias points in exactly the direction of the measured failure

The bias $\mathbb{E}[U]-\operatorname{median}[U]$ is zero for a symmetric conditional and **positive for a
right-skewed one** — the median sits below the mean. Enhancing-rim intensity given a pre-contrast input is
precisely a right-skewed conditional: most patients enhance moderately, a minority enhance intensely, and none
enhance negatively. L1 therefore systematically under-predicts the upper tail of enhancement.

That is a first-principles derivation of the exact quantity §4 measures as the project's headline defect:
VENA's `raw_p995` = 0.786 against C3-SynDiff's 0.901. It also explains why *every* downstream fix failed — region
weighting (§7) made the model fit the dim median **more accurately**; LPL was invariant to global intensity by
construction; a perfect oracle mask moved the tail by −0.03 %. Each intervention operated downstream of a bias
that is baked into the objective's minimiser.

**This does not prove L2 fixes it.** It establishes that the loss norm is a live, mechanistically-motivated
candidate for the failure mode, and that "L1 because the field uses L1" is not an argument that survives contact
with the theorem.

### The prior L2 experiment does not settle this

The retired L2 run (2026-06-12 S1, plateaued 26.5 dB whole / 18.3 dB WT-PSNR) is cited as evidence L2 fails. It is
not, because it was **confounded on at least three axes simultaneously** — `decoder_perceptual_loss_s3_analysis_2026-06-20.md`
§3 says so explicitly, listing L2→L1 *and* hard-zero-init→scale-ramp *and* the timestep transform as the deltas of
the replacement run. That analysis's own recommended experiment (E1) bundled all three. Its §"Q5" even flags the
outstanding need: *"one row 'loss norm: l2 → l1, all else equal'"* — a row that was never run.

Note also that the retired L2 run's absolute numbers (26.5 dB whole, 18.3 dB WT) are **higher** than v3a's
(25.3 / 16.7). Those are not comparable — different cohort composition, different conditioning route — but they
are certainly not evidence L2 was worse.

### Change — a three-arm loss ablation on the retrain generation

`CFMLoss.__init__` (`src/vena/model/fm/controlnet/losses/cfm.py:76-84`) already validates
`norm ∈ {"l2", "l1"}` and dispatches at lines 139-141. Arms 1 and 2 are a one-line YAML change each.

```yaml
loss:
  cfm:
    norm: l1            # arm A — incumbent, the control
    # norm: l2          # arm B — restores the FM identity; tests the median-bias hypothesis
    # norm: huber       # arm C — requires ~15 lines in cfm.py (see below)
```

Arm C, **pseudo-Huber**, is the compromise worth the small implementation cost: quadratic for
$|r|<\delta$ (so it is mean-seeking, and Bregman, in the small-residual regime where the FM identity actually
binds) and linear beyond (so it keeps L1's robustness to the outlier voxels that motivated the switch away from
L2 in the first place). Song & Dhariwal (ICLR 2024, arXiv:2310.14189, §3.3) adopted pseudo-Huber for consistency
training on exactly this trade-off. Set $\delta$ to a robust scale of the velocity residual — e.g. the median
absolute residual measured on the first 100 steps of the L1 arm — rather than picking a round number.

**Acceptance:** the arms are discriminated by `mean_et_pred` and `p995_pred_brain` (§13), *not* by PSNR. The
prediction under the median-bias hypothesis is specific and falsifiable: **arm B raises `mean_et_pred` toward
`mean_et_real` and may lower whole-volume PSNR** (the perception–distortion trade-off of §7, run forward instead
of backward). If arm B raises PSNR and leaves `mean_et_pred` flat, the hypothesis is dead and L1 is vindicated on
evidence rather than on convention.

**Cost:** two extra 1900-epoch arms is the wrong framing — under §3's `max_epochs ≈ 1100` these are ~2.5 A100-days
each, and §11 shows the informative window may be much shorter still. Run them as siblings of the main retrain on
separate GPUs.

---

## 13. Exhaustive-val — three defects to fix before the retrain

*(Added 2026-07-29. Extends §5, which called for six intensity columns; that recommendation stands unchanged and
is repeated in the table at §10.)*

### 13a. `latent_preds.h5` is written every cadence epoch and **explicitly never pruned**

`src/vena/model/fm/lightning/callbacks/exhaustive_launcher.py:68` states the policy in a comment:

> ``latent_preds.h5`` files are NEVER pruned; only the ~1 GB [snapshots are]

and `routines/fm/train/engine.py:481-484` repeats it on `prune_snapshots_keep`. Measured on v3a: **527 MB per
cadence epoch × 69 epochs = 38 GB for one run.** `routines/fm/exhaustive_val/engine.py:454` writes it
unconditionally.

At the §1 retention of 40 checkpoints (110 GB) plus 38 GB of latents, one run is ~150 GB. Across the §12 loss
arms plus the retrain itself that is ~600 GB — on Picasso `fscratch`, which is the same filesystem the cohort H5
caches live on.

The latents are genuinely useful (they are what job 1679496 decoded), but they are useful at a **coarse** cadence.
Nothing in the analysis so far needed them every 25 epochs.

**Change** — add one field to `_ExhaustiveValCfg` (`routines/fm/train/engine.py`, alongside
`prune_snapshots_keep`) and gate the write:

```yaml
exhaustive_val:
  every_epochs: 25
  latent_preds_every_n: 4      # NEW — write latent_preds.h5 on every 4th cadence pass,
                               # counting from the first (epochs 0, 100, 200, ...).
                               # 0 = never; 1 = current behaviour.
```

Gate at `engine.py:454`. Count in **cadence passes, not epochs**, so the meaning does not silently change when
`every_epochs` is retuned; anchor at the first pass so epoch 0 (the warm-start ceiling) is always captured.
`metrics.csv`, `aggregate.csv`, `timing.csv` and the figures stay every pass — they are ~500 KB and they are the
record §1 and §2 depend on. Footprint drops 38 GB → ~10 GB.

**Also fix the docstrings in the same change** (`coding-standards.md` rule 17): three separate comments currently
assert latents are never pruned.

### 13b. The 99.95 normalisation *is* now correct end-to-end — assert it rather than trust it

Verified at HEAD, contrary to the concern that prompted the question:

- `src/vena/model/fm/eval/exhaustive.py:56,109` — both `load_real_t1c_normalised` and `load_real_t1c_box` default
  to `percentile_upper = ENCODER_PERCENTILE_UPPER` (99.95).
- `routines/fm/exhaustive_val/engine.py:723` (metrics path) and `:879` (figure path) both call
  `load_real_t1c_box(image_h5, pid, crop_spec)` **without overriding the percentile**.

So every number in `metrics.csv`, the `ssim_by_pid` ranking that selects best/worst patients, the
`psnr_ssim_by_pid_nfe` row annotations, and the per-slice display window in `render_comparison_figure` all derive
from the *same* 99.95-normalised reference. Metric and figure cannot disagree, and best-patient selection is on
the corrected scale. **No defect here.**

The residual risk is silent drift, and it has already bitten twice (the 99.5 era; the `input_img_size_numel`
WARNING). Per §9, log it: emit `ENCODER_PERCENTILE_UPPER=%s` into `subprocess.log` at job start and assert
`== 99.95`. A one-line `logger.info` makes every future archived run self-describing about the convention it was
scored under — which is precisely what was missing when §6 had to be reconstructed from commit dates.

### 13c. MS-SSIM is absent from the exhaustive-val CSV

The 46 columns are PSNR / SSIM / MAE / MSE per region + latent metrics + timings + voxel counts. **No MS-SSIM.**
The implementation exists — `src/vena/validation/metrics_paired.py:231` (`ms_ssim_brain`) and `:267`
(`ms_ssim_wt_bbox`), wrapping `monai.metrics.regression.compute_ms_ssim` — but only in the post-hoc article
package, so the cadence curve cannot be read in the metric the manuscript reports.

This matters more than a missing column usually would, because MS-SSIM is where region weighting showed its
largest degradation in §7 (MS-SSIM_wt 0.8161 → 0.7559, a 0.06 drop, versus 0.0037 on single-scale SSIM_wt). The
metric with the most dynamic range on this project's actual failure mode is the one not logged during training.

**Change** — add to `metrics.csv`:

```
ms_ssim_brain, ms_ssim_wt_bbox
```

Reuse `vena.validation.metrics_paired` directly (per `extensibility.md`: import, do not re-implement). Honour the
documented `min_dim = 90` guard — `ms_ssim_wt_bbox` returns NaN on small WT bounding boxes, and the existing
`_write_aggregate_csv` already skips NaN cells.

### Combined change to the exhaustive-val CSV (§5 + §13c)

```
p995_pred_brain, p995_real_brain,
mean_et_pred,    mean_et_real,
mean_bnwt_pred,  mean_bnwt_real,
ms_ssim_brain,   ms_ssim_wt_bbox
```

Eight columns. The decoded prediction and the normalised real volume are both already in memory at
`engine.py:752`. Marginal cost: two MS-SSIM calls and six percentile/mean reductions per (patient, NFE).

---

## 14. What T1C-RFlow actually does differently — read from the vendored source

*(Added 2026-07-29. Source: `src/external/t1c_rflow/upstream/` at SHA `fc8314f6`; Eidex et al., arXiv:2509.24194.)*

Four findings, in descending order of how much they should change the retrain. The first two invalidate
hypotheses currently in circulation.

### 14a. Its loss is byte-identical to VENA's — the "L1 advantage" hypothesis is falsified

`upstream/train_rflow.py:207`:

```python
loss = F.l1_loss(noise_pred, (tgt - noise))
```

L1 on the velocity target $u = z_{\text{T1c}} - z_{\text{noise}}$. Its scheduler (lines 136-140) is
`num_train_timesteps=1000`, `use_discrete_timesteps=True`, `sample_method="logit-normal"`,
`use_timestep_transform=True` — **the same four settings v3a runs**. `decoder_perceptual_loss_s3_analysis_2026-06-20.md`
§1 named L1 as one of the two deltas "that most parsimoniously explain T1C-RFlow's qualitative tumor advantage".
VENA adopted L1 in the S1 v2 recipe. **The advantage, if real, is not the loss** — and §12's case for testing L2
is therefore *strengthened*, not weakened, by this: matching T1C-RFlow's loss did not close the gap.

### 14b. Its `seg` tensor is not a segmentation — it is T2-FLAIR

`upstream/train_rflow.py:64` (`LatentPairDataset`):

```python
mu_cond = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t1n_z_mu.pt"))   # T1 native
mu_seg  = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t2f_z_mu.pt"))   # T2-FLAIR
```

and line 202: `model_in = torch.cat([noisy_latents, cond, seg], 1)`, with
`in_channels = latent_channels * 3 = 12` (line 129). The variable named `seg` holds `-t2f_`, i.e. **T2-FLAIR**.

So T1C-RFlow conditions on **{T1n, T2-FLAIR}, concatenated in the latent channel dimension, no mask, no
segmentation** — which is **exactly the A6 configuration of §11**, down to `in_channels: 12`. And A6 is VENA's
*worst* input arm. Whatever produces T1C-RFlow's tumours, it is not its conditioning set.

*(Correct this wherever the project has recorded T1C-RFlow as mask- or segmentation-conditioned. It is not.)*

### 14c. It resamples the VAE posterior on every `__getitem__` — VENA uses fixed cached latents

`upstream/train_rflow.py:86-89`:

```python
z_tgt  = μ_t + σ_t * torch.randn_like(μ_t)
z_cond = μ_c + σ_c * torch.randn_like(μ_c)
z_seg  = μ_s + σ_s * torch.randn_like(μ_s)
```

Both $\mu$ and $\sigma$ are cached to disk (`*_z_mu.pt`, `*_z_sigma.pt`) and a **fresh posterior sample is drawn
every time an item is loaded** — for the target and both conditions. Every epoch the model sees a different draw
from $q(z|x)$.

VENA stores a single encoded tensor per (patient, modality) — `src/vena/data/h5/latent_domain/convert.py:793`
calls `self.encoder.encode(t, mode=cfg.inference_mode, ...)` and writes one array to `latents/<slug>`; no
$\sigma$ is persisted. Stochasticity comes instead from five *offline* augmented variants (`v0..v4`, uniform
weights) — five fixed points, redrawn never.

This is the most substantive recipe difference found, and it is the one that plausibly bears on §11's finding
that A6 stops improving at epoch 150: continuous posterior resampling is a per-epoch-fresh regulariser, whereas
five frozen variants are exhaustible. It is also a *documented property of the VAE*, not a hyperparameter — the
encoder emits a distribution and VENA is discarding its width.

**Change — scoped as an experiment, not a default.** Persisting $\sigma$ means re-encoding the corpus, so do not
fold this into the retrain blind. Order:

1. Measure first, for free: encode ~20 patients with $\sigma$ retained and report
   $\mathbb{E}[\sigma]/\mathrm{SD}[\mu]$ per latent channel. If the MAISI-V2 posterior is near-deterministic
   (a common outcome for a strongly-regularised VAE-GAN), resampling is a no-op and this closes with one cheap
   measurement.
2. Only if that ratio is non-trivial, add $\sigma$ to the latent H5 (schema bump per `h5-design-principles.md`,
   principle 1) and a `data.posterior_resample: bool` flag.

### 14d. It selects checkpoints on held-out validation loss, and uses no EMA

`train_rflow.py:250-260` saves `latest`, `best` (on `val_l1`, evaluated every `val_interval` epochs — lines
225-237), and an epoch snapshot every 50. `grep -nE "ema|EMA"` over the trainer returns nothing.

VENA selects on `train/total_epoch` (§2) and stops on it (§3). The competitor whose tumour rendering motivated
this whole line of inquiry uses a **held-out** signal for both. That is independent corroboration of §2 change 2
and §3 — from the exact model the project is trying to match.

VENA's EMA is a genuine advantage and should be kept; the selection signal is the part to copy.

### What is *not* the difference

Latent geometry. `decoder_perceptual_loss_s3_analysis_2026-06-20.md` §1 already recorded the "different latent
resolution" hypothesis as **falsified** — same MAISI latent shape. The vendored config differs only in
`base_img_size_numel` (upstream `64·64·48 = 196608`; VENA `129024`), which is the timestep-transform reference
volume, not the data shape. It also fine-tunes its own autoencoder (`upstream/checkpoints/autoencoder_epoch273.pt`)
where VENA uses frozen MAISI-V2 — a real difference, but one the project has deliberately excluded, and
`preflights/maisi_vae` is the routine that owns re-opening it.

---

## 15. Acceptance criteria — the axis that matters is the **mask**, not PSNR-vs-SSIM

*(Added 2026-07-29 in response to the "should SSIM be the base acceptance metric?" question. Measured on v3a,
NFE=20, 65 cadence epochs ≥ 100, 116 patients.)*

### SSIM is not meaningfully more discriminative than PSNR here

Define a discriminative index as (SD of the epoch-mean across cadence epochs) / (mean within-epoch across-patient
SD) — how much of the metric's spread is signal about the checkpoint versus spread across patients:

| metric | across-epoch range | across-epoch SD | within-epoch patient SD | index |
|---|---|---|---|---|
| PSNR whole | 0.770 dB | 0.1196 | 3.114 | 0.0384 |
| SSIM whole | 0.0078 | 0.00108 | 0.02373 | **0.0454** |
| PSNR_WT | 1.179 dB | 0.1752 | 4.596 | 0.0381 |
| SSIM_WT | 0.0178 | 0.00289 | 0.06100 | **0.0473** |
| PSNR_ET | 1.343 dB | 0.1954 | 6.334 | 0.0308 |
| SSIM_ET | 0.0251 | 0.00392 | 0.08499 | **0.0461** |

SSIM's index is 18–50 % higher than PSNR's at matched region — a real but small edge, and nowhere near the
difference that would justify calling one "the base metric". (Caveat: these are *unpaired* SDs; epoch-to-epoch
comparisons are paired on the same 116 patients, so the operative noise is smaller than column 4 suggests. The
ranking between metrics is unaffected.)

### And it selects the same checkpoint

Spearman rank-correlation of the epoch ordering, v3a:

| | PSNR | SSIM | PSNR_WT | SSIM_WT | PSNR_ET | SSIM_ET |
|---|---|---|---|---|---|---|
| **PSNR** | 1.000 | **0.907** | 0.786 | 0.545 | 0.761 | 0.593 |
| **SSIM** | 0.907 | 1.000 | 0.691 | 0.481 | **0.649** | 0.508 |
| **PSNR_ET** | 0.761 | 0.649 | 0.877 | 0.564 | 1.000 | **0.736** |

Whole-volume PSNR and whole-volume SSIM agree at ρ = 0.907 and pick the **same epoch (525)**. Swapping one for the
other changes nothing. The disagreement is between *regions*: whole picks 525, ET picks 900 (PSNR_ET) / 925
(SSIM_ET). Within the ET region the two metrics agree at ρ = 0.736 and land one cadence step apart.

### The asymmetry decides it

| selector | epoch chosen | PSNR_ET obtained | SSIM obtained |
|---|---|---|---|
| whole-volume SSIM | 525 | 16.731 (**−0.189 dB** vs best) | 0.91164 (best) |
| PSNR_ET | 900 | 16.920 (best) | 0.91144 (**−0.00020**) |

Selecting on ET costs 0.0002 SSIM — 2.6 % of that metric's entire across-epoch range, i.e. nothing. Selecting on
whole-volume SSIM costs 0.19 dB PSNR_ET — 14 % of its range, and the same order as the entire measured benefit of
the 20× tumour-weight increase in the S3 matrix (+0.41 dB). **The ET-region selector dominates: it is nearly free
in the whole-volume metric and the whole-volume selector is not free in ET.**

### The raw argmax is a winner's-curse estimate

v3a's whole-volume PSNR peak moves **525 → 925** under a 3-point centred moving average (25.331 → 25.309), while
its PSNR_ET peak stays at 900 (16.920 → 16.841) and A6's stays at 150. Picking the maximum of 65 noisy cadence
points overfits the 116-patient validation set; the whole-volume "peak at 525" in §1 is partly that artefact. This
does not change §1's conclusion — 900/925 are also gone from disk — but it does change the *procedure*.

### But the selector may not be an ET metric — the ET mask is an oracle

*(Revision 2026-07-29, second pass. The draft above recommended early-stopping on `psnr_db_et`. That
recommendation is withdrawn.)*

`psnr_db_et` is computed inside a mask derived from the **ground-truth segmentation**, and "enhancing tumour" is
*defined* by contrast enhancement on the ground-truth T1c. The region only exists because the target image
exists. Selecting a checkpoint with it means the training procedure consumes a label that:

- is unavailable at deployment (Málaga cohort, and any clinical use — there is no reference T1c to segment);
- is unavailable for the healthy-control shortcut diagnostic (proposal §6.5), where there is no tumour at all;
- is the **same oracle dependence the project already recorded as its most damaging finding** —
  `project_vena_oracle_mask_finding`: VENA's apparent tumour win over SOTA was entirely the GT WT mask it alone
  received, and on the pre-registered primary endpoint C2-ResViT wins. Re-introducing a GT mask at the
  checkpoint-selection stage rebuilds the same criticism one layer further back, where it is harder to see.

A reviewer who has read the oracle-mask analysis will ask how the checkpoint was chosen. "On a metric computed
inside the ground-truth enhancing-tumour mask" is not an answer that survives.

**The 0.19 dB is the price of a defensible protocol, and it should be paid.** It is also smaller than it looks:
under §1's retention of 40 checkpoints the *selection* is deferred to post-hoc analysis anyway, and the monitor's
only real job is to decide **when to stop**, not which weights to publish.

### The two admissible selectors, and what they actually measure

| candidate | column | mask | GT-dependent? | admissible? |
|---|---|---|---|---|
| whole-box PSNR / SSIM | `psnr_db`, `ssim` | **none** — `full_volume_psnr_ssim` builds `mask = torch.ones_like(p)` (`exhaustive.py:249`) | No | **Yes** |
| brain-masked MAE / MSE | `mae_whole`, `mse_whole` | `masks/brain` | No — see below | **Yes**, with provenance stated |
| brain-masked PSNR / SSIM | *(does not exist)* | — | No | **Yes, once added** |
| ET / WT / NETC / ED metrics | `*_et`, `*_wt`, … | GT tumour segmentation | **Yes** | **No — reporting only** |

**Note a naming defect while fixing this.** `psnr_db` and `ssim` are computed over the **whole decoded brain box
including intra-box background** (`mask = ones`), while `mae_whole` and `mse_whole` in the *same CSV row* are
computed **inside `masks/brain`** (`engine.py:1271` sets `n_voxels_brain` from that same `whole_mask`). "Whole"
means two different things in adjacent columns. Rename on the retrain: `psnr_db_box` / `ssim_box` for the
mask-free pair, `mae_brain` / `mse_brain` for the masked pair, and add the missing `psnr_db_brain` /
`ssim_brain`. Bump the CSV schema and state it in the module docstring.

### Brain-mask provenance — the detail the selector needs to declare

The chain, verified end to end:

1. **Source.** For UCSF-PDGM, `src/vena/data/h5/ucsf_pdgm/image_domain/convert.py:154-163` reads
   `{patient_id}_brain_segmentation.nii.gz` — the brain segmentation **shipped with the cohort**, produced by the
   UCSF-PDGM preprocessing pipeline, not by VENA — reorients to LPS, and binarises at `> 0.5` into
   `masks/brain`. Each cohort's converter supplies its own equivalent; cohorts without one fall back to a
   nonzero-voxel heuristic (all VENA cohorts are distributed skull-stripped).
2. **Latent encode.** `masks/brain` is encoded to the 4×48×56×48 latent grid as `masks/brain_latent`.
3. **Read-back.** `ExhaustiveValEngine._brain_mask_in_image_space` (`engine.py:1072-1083`) reads
   `batch["m_brain"]` and **upsamples the latent mask back to image space**; `brain_mask_source` records
   `"masks/brain_latent"` (all 580 rows of v3a epoch 900) or `"real_box>0"` when the H5 predates the field.

Two consequences to state in the manuscript rather than discover in review:

- **The mask is not GT-target-derived.** It is a skull-strip boundary obtainable from any structural sequence;
  at deployment it is one HD-BET call on T1pre. Using it costs no oracle information. This is the property that
  makes brain-masked selection admissible and ET-masked selection not.
- **The mask used at scoring time is a latent round-trip**, ÷8 then ×8, so it is a dilated/smoothed version of
  the source mask with ~4-voxel boundary error. Harmless for a whole-brain aggregate; do not reuse this path for
  anything boundary-sensitive. And the `"real_box>0"` fallback *is* derived from the real T1c — never let a
  selection run take that branch. Assert `brain_mask_source == "masks/brain_latent"` for every row, alongside
  the §13b percentile assertion.

### Change

1. **Do not make SSIM the base acceptance metric.** Whole-box SSIM and whole-box PSNR agree at ρ = 0.907 and pick
   the same epoch; the swap buys nothing. The evidence for "SSIM is more robust" is a ~20 % edge in
   discriminative index, not a change of conclusion.
2. **Early-stop and checkpoint on a mask-free or brain-masked metric — never on `*_et` / `*_wt`.** Concretely:
   monitor **`ssim_brain`** (added per the rename above), with `psnr_db_box` as the pre-registered fallback if
   the brain-masked pair is not ready in time. Rationale for preferring the brain mask over the raw box: the box
   contains a large constant-zero background margin that inflates PSNR and dilutes any signal from the tissue,
   and the mask that removes it carries no oracle information. Rationale for SSIM over PSNR *within* that
   choice: SSIM's discriminative index is ~20 % higher at matched region, and since the two agree on the
   selected epoch anyway, taking the slightly better-conditioned one is free.
3. **Select on a smoothed curve** — 3-point centred moving average over cadence epochs — not the raw argmax
   (v3a's whole-box peak moves 525 → 925 under smoothing).
4. **Report ET/WT metrics; never select on them.** They stay in `metrics.csv` and in every figure and table —
   they are the scientific readout. The rule is only that they must not close a loop back into training. State
   this explicitly in the methods section: *"checkpoint selection and early stopping used brain-masked SSIM on
   the cross-validation validation split; no tumour-segmentation-derived metric influenced model selection."*
   That sentence is worth writing precisely because §14d shows the nearest competitor cannot claim it either.
5. **Acceptance criteria ≠ selection criteria.** The stopping monitor is one admissible scalar. The *acceptance*
   criterion for the retrain is the pre-registered tuple `{SSIM_brain, MS-SSIM_wt (§13c), PSNR_ET,
   mean_et_pred vs mean_et_real (§5)}`, evaluated post-hoc over the retained checkpoints. §7 shows any
   distortion scalar prefers the dimmer model and §12 gives the mechanism, so no single distortion metric can be
   the acceptance criterion for a project whose headline defect is an intensity deficit.
6. **Keep every checkpoint (§1) regardless.** This is what makes the whole question low-stakes: with 40
   checkpoints retained, an imperfect monitor costs compute, not results.

---

## 17. The thing that was missing — the exhaustive-val aggregate is 56 % one cohort, and 16 % of it is test data

*(Added 2026-07-29, third pass, in answer to "is there anything I'm missing?". This is the most consequential
finding in the document and it invalidates the aggregation used by every number in §1–§3 and §11–§15 — though not,
as explained below, the comparisons drawn from them.)*

### What the 116 "patients" actually are

Per-cohort unique `patient_id` counts in v3a's `exhaustive_val/epoch_900/metrics.csv`:

| cohort | rows | registry role | longitudinal | n_patients / n_scans |
|---|---|---|---|---|
| **LUMIERE** | **65 (56 %)** | cv | **true** | 91 / **638** |
| BraTS-GLI | 9 | cv | true | 1133 / 1251 |
| UCSF-PDGM | **7** | cv | false | 495 / 495 |
| UPENN-GBM | 7 | cv | false | 611 / 611 |
| BraTS-Africa-Glioma | 6 | **test_only** | false | 95 / 95 |
| BraTS-Africa-Other | 6 | **test_only** | false | 51 / 51 |
| BraTS-PED | 6 | **test_only** | false | 260 / 260 |
| IvyGAP | 5 | cv | false | 34 / 34 |
| REMBRANDT | 5 | cv | false | 63 / 63 |

### Why, mechanically

`ExhaustiveValEngine._split_n_patients(60, 9)` allocates 6–7 **patients** per cohort. Then
`_cohort_val_patients` does what its own docstring says (`engine.py:312-317`):

> draws ``min(budget, |val|)`` keys with a seeded RNG, then **expands each patient to its scan-level IDs via the
> CSR layout — so longitudinal cohorts contribute every scan of a selected patient**.

LUMIERE averages 7.0 scans/patient (638/91), so its 7-patient budget becomes **65 scored rows**. The budget is
enforced in patients; `metrics.csv` is written in scans. The behaviour is intentional and documented; its effect
on the *aggregate* is neither.

### Two consequences, one of them serious

**(a) The unweighted mean is a statement about LUMIERE.** Every epoch-mean in this document — the 25.331 dB peak
at 525, the 16.920 dB PSNR_ET peak at 900, §15's discriminative indices and rank correlations — is 56 % weighted
to seven post-treatment LUMIERE patients and 6 % to UCSF-PDGM, the primary cohort. LUMIERE is longitudinal
recurrent-GBM follow-up: post-surgical cavities, post-radiation change, treatment-related enhancement. Its
enhancement statistics are not those of the pre-operative gliomas the retrain targets. Selecting a checkpoint on
this aggregate optimises for the wrong cohort, and the seven patients are further counted ~9× each, so the
effective sample is far smaller than n=116 suggests.

**(b) 16 % of the early-stopping signal would come from held-out test data.** Three cohorts are `role: test_only`
and carry no `splits/cv/fold_0/val`. `_cohort_val_patients` falls through to `elif "splits/test" in f`
(`engine.py:334`) — correct for the intended OOD-monitoring use, **wrong the moment this aggregate closes a loop
back into training**. BraTS-Africa-Glioma + BraTS-Africa-Other + BraTS-PED contribute 18 of 116 rows. Wire
`train/total_epoch` → an exhaustive-val monitor (§2, §3, §15) without fixing this and the run early-stops on a
signal that is 15.5 % test data. That is a one-sentence reviewer objection and a trivially avoidable one.

### What this does *not* invalidate

The **paired comparisons** survive intact, because the confound is identical on both sides and cancels in the
difference:

- §11 (v3a vs A6) — same registry, same fold, same seeded draw, **same 116 rows**, verified by
  `patient_id` set intersection = 116/116. The −0.446 dB conclusion stands.
- §2 / §3 (retained epoch vs peak epoch, within one run) — same rows throughout.
- §7 (v3b vs v3b-rw) — drawn from the `paired_fidelity` Ring-A tables, a different and properly-constructed
  evaluation.

What does **not** survive is any *absolute* reading, and any claim about *which epoch* is best — the argmax is
taken over a LUMIERE-dominated curve. §1's "the peak epochs are gone from disk" is unaffected in force (they are
gone either way) but the specific epochs 525 / 900 should be re-derived under (a) before being quoted again.

### Change

1. **Aggregate cohort-balanced, and exclude `test_only` cohorts from any signal that feeds back into training.**
   `aggregate.csv` is already written per (cohort, nfe, region) — the machinery exists. Define the monitor as the
   unweighted mean **over cv-role cohorts of the per-cohort mean**, not the mean over rows. Add a
   `role` column to `metrics.csv` so the filter is expressible without re-reading the registry.
2. **Fail loudly if a `test_only` cohort reaches the monitor.** Add the assertion next to the §13b percentile and
   §15 `brain_mask_source` assertions: a run whose monitor set intersects any `splits/test` pool raises, rather
   than logging a warning nobody reads. This is the same class of defect as the `loaded=0` warm-start trap in §9.
3. **Decide the scan-vs-patient unit explicitly and record it.** For a longitudinal cohort, either average
   within patient before aggregating (one row per patient) or keep scan-level rows and state it. The current
   behaviour — patient-budgeted, scan-aggregated — is the one combination that is defensible in neither reading.
   Recommended: average within patient, which also removes the 9× duplication.
4. **Raise the per-cohort budget for the primary cohort.** Seven UCSF-PDGM scans is too few to steer a 1900-epoch
   run. With §13a's `latent_preds_every_n` freeing ~28 GB and most of the pass cost being sampling, a
   `n_patients` of 120–180 with a cv-only cohort set is affordable at the same cadence.
5. **Re-derive §1–§3 and §15 under the corrected aggregation** before those numbers enter a manuscript. They are
   diagnostic-grade as they stand; they are not publication-grade.

---

## 18. Protocol for the `l1` / `l2` / `pseudo-huber` ablation

*(Added 2026-07-29. The ablation is approved; this section is the pre-registration, and the four traps below are
the ones that would silently void it.)*

### Design

Three arms, each `resume_from: baseline` (from the MAISI FM trunk), differing in **exactly one field**:

```yaml
loss:
  cfm:
    reduction: mean
    norm: l1        # arm A — control (incumbent)
    # norm: l2      # arm B — restores the FM identity (§12)
    # norm: huber   # arm C — Bregman near 0, robust in the tail
```

Everything else pinned to the retrain recipe: seed 1337, fold 0, `[t1pre, t2, flair]` (§11), CFG dropout 0.1
(§4), retention 40 (§1), cohort-balanced cv-only monitor (§17), 8 new CSV columns (§13).

**Trap 1 — do not warm-start from v3a.** The instruction "run this ablation for the v3a checkpoint" must mean
*in the v3a retraining generation*, not *initialised from v3a's weights*. `ema_best.ckpt` already encodes an
L1-fitted conditional median; an L2 arm warm-started from it measures "how far can L2 pull an L1 solution in the
remaining budget", not "does L2 give a different solution". All three arms start from the same frozen trunk. This
is the same class of error that voided arm J0 (§9).

### Trap 2 — the early-stopping monitor must not be the training loss

`train/total_epoch` is the **loss value itself**, and the three norms put it on three different scales: v3a's
`cfm` column has mean 0.900 (≈ 𝔼|r| under L1), whereas an L2 arm reports ≈ 𝔼[r²] and Huber a blend. A
patience-based stop on that quantity would halt the three arms at different points for reasons having nothing to
do with quality, and the resulting "L2 trained for 700 epochs, L1 for 1900" comparison would be uninterpretable.

All three arms **must** use the identical §15 monitor (`ssim_brain`, cv-only, cohort-balanced, 3-point smoothed).
This is not a refinement of §2/§3 — for this ablation it is a correctness precondition.

### Trap 3 — gradient scale, and why it is the leading explanation for the retired L2 run

The two objectives differ in gradient magnitude by a factor set by the residual scale:

$$\frac{\partial}{\partial v}\,|v-u| = \operatorname{sign}(v-u)\ \Rightarrow\ \text{magnitude } 1,
\qquad
\frac{\partial}{\partial v}\,(v-u)^2 = 2(v-u)\ \Rightarrow\ \text{magnitude } 2|r|.$$

From v3a's `metrics/train_step.csv` (400 000 steps, L1, `gradient_clip_val: 1.0`):

| quantity | mean | median | p95 | max |
|---|---|---|---|---|
| `cfm` (= 𝔼\|r\|) | 0.900 | 0.906 | 1.004 | 1.679 |
| `grad_norm_*_preclip` | 0.331 | 0.253 | 0.543 | 111.9 |
| `grad_clip_active` | **0.0108** | — | — | — |

With 𝔼|r| ≈ 0.90, an L2 arm's gradients scale by ≈ 2·𝔼|r| ≈ **1.8×**, putting mean pre-clip norm near 0.60 and
p95 near 0.98 — i.e. **gradient clipping goes from firing on ~1 % of steps to roughly half of them**. Clipping is
a nonlinear, arm-dependent modification of the update. The arms would then differ in optimiser behaviour, not
just in estimator, and "L2 is worse" would be unfalsifiable. This is the most parsimonious explanation available
for why the retired 2026-06-12 L2 run underperformed, and it is entirely untested.

**Deriving the new clip value.** Assume the per-voxel residual $r$ is approximately Gaussian; then
$\mathrm{RMS}(r)=\mathbb{E}|r|\sqrt{\pi/2}=0.90\times1.2533=1.128$, and the ratio of L2's to L1's gradient norm is

$$\frac{\|2r\|_2}{\|\operatorname{sign}(r)\|_2}=2\,\mathrm{RMS}(r)\approx\mathbf{2.26\times}.$$

Projecting v3a's measured L1 distribution onto the L2 arm: mean ≈ 0.75, p95 ≈ 1.23, p99 ≈ 2.3. L2's tail is
additionally *heavier* than this linear projection, because an outlier residual scales the gradient linearly
under L2 whereas L1 clamps it to `sign`. Pseudo-Huber sits between the two, its gradient bounded by δ.

**Change — `training.gradient_clip_val: 1.0 → 5.0`, in every arm and in the main retrain.**

| arm | projected pre-clip p95 | headroom at clip = 5.0 |
|---|---|---|
| L1 (measured) | 0.543 | **9.2×** |
| L2 (projected) | ~1.23 | ~4.1× |
| pseudo-Huber | between | — |

5.0 clears every arm's p99 with margin, so clipping stops being a routine part of the update in any of them,
while still neutralising the pathological spikes actually observed (max 111.9 → scaled by 0.045; at the old
threshold it was scaled by 0.009 — both fully suppressed).

**Honest framing of *why* to raise it:** at `1.0`, clipping fires on **1.08 %** of L1 steps, so the current
recipe is *not* meaningfully throttled — "L1 training is being prevented" is not supported by the data. The
reason to raise is **arm comparability**: at `1.0` the L2 arm would clip on roughly half its steps, making the
three arms differ in optimiser behaviour rather than in estimator, and rendering the ablation unfalsifiable.
Raising it in the L1 arm too is what keeps the ablation single-variable.

**Validity criterion and abort rule (pre-registered):**

- `grad_clip_active` mean must be **< 5 % in every arm**, read from `metrics/train_step.csv` after the first
  5 000 steps. An arm that fails is invalid and must be re-run at a higher threshold, not reported.
- **Risk of raising:** a spike now delivers up to 5× the update it did at `1.0`. AdamW's second-moment
  normalisation absorbs most of this, but if any arm produces a non-finite loss or a >10× loss excursion within
  the first 2 000 steps, drop to `2.0` **in all three arms** and restart all three — never one.
- Do **not** normalise the loss per voxel. L2's heteroscedastic weighting toward high-residual voxels *is* the
  mechanism under test (§12); only the global scale may be matched, and raising the clip threshold does exactly
  that without touching the objective.

**Retracted:** an earlier pass of this section called the identical `grad_norm_cn_*` / `grad_norm_trunk_*` columns
a logging bug. **It is not one** — see N1 in §20. Do not "fix" it.

### Trap 4 — implementing `huber` touches three places, and δ is not a free parameter

`CFMLoss` (`src/vena/model/fm/controlnet/losses/cfm.py`) needs:

1. line 83-84 — the validator `if norm not in ("l2", "l1"): raise` must admit `"huber"`;
2. lines 108-111 — the `reduction="none"` branch used by the region-weighted path;
3. lines 139-141 — the mean/sum dispatch.

Missing (2) yields a loss that silently falls back to L1 whenever region weighting is enabled — the exact shape
of the inert-config bug §2 documents.

δ must be pre-registered from measurement, not chosen round: set it to the median absolute velocity residual of
the **arm-A control over its first 100 optimiser steps**. From v3a that is ≈ 0.90, which places the quadratic/
linear transition at the median residual — half the voxels in each regime, the intended compromise. Record δ in
`decision.json` (schema bump; the run's `schema_version` is already at **0.10.0**, not the 0.8.0 the rules file
documents — fix that drift too).

**While in this file, fix the stale docstring** (`coding-standards.md` rule 17). It currently reads *"Per the
MAISI-v2 reference implementation we keep the MSE formulation (proposal §5.2). The MAISI training script actually
uses an L1 default; we follow the proposal's text rather than the upstream script"* — the opposite of the
production recipe since S1 v2. The constructor default is likewise still `norm: str = "l2"` while every
production YAML sets `l1`; leave the default alone if you prefer, but say so.

### Readout — the ablation is judged on intensity, not on PSNR

The discriminating prediction of §12 is specific: **arm B raises `mean_et_pred` toward `mean_et_real` and may
lower whole-box PSNR.** So:

- **Primary endpoint:** `mean_et_pred − mean_et_real` and `p995_pred_brain − p995_real_brain` (§5 columns).
- **Secondary:** `SSIM_brain`, `MS-SSIM_wt` (§13c), `PSNR_ET` — reported, not selected on (§15).
- **Dependency:** the §13 instrumentation must land **before** these arms launch. Without those eight columns the
  ablation cannot be read, and answering it post-hoc costs another decode job (§5).
- **Statistics:** paired per-patient differences over the cohort-balanced cv set (§17), Wilcoxon signed-rank on
  the three pairwise contrasts, Holm correction across contrasts × the pre-registered metric set. Report the
  median paired difference with a bootstrap CI, not p alone.
- **Honest limitation to state:** one seed per arm. The paired design controls patient variance; it does **not**
  control seed variance, and a 0.2 dB difference between single-seed arms is not a result. If the arms separate
  only marginally, the finding is "no detectable difference at n=1 seed", not "L1 wins".

---

## 20. Bug & defect register — the implementation checklist

*(Added 2026-07-29. Consolidates every defect found while writing this document into one actionable list, so an
implementing agent does not have to reconstruct them from prose. Every "verified" row was confirmed by reading
the cited file:line or by re-deriving from run artifacts; nothing here is inferred from documentation.)*

**Severity:** `C` = blocks or invalidates the retrain · `H` = must fix before launch · `M` = fix in the same
change · `L` = hygiene.

### B — Confirmed defects to fix

| # | sev | where | what | fix | § |
|---|---|---|---|---|---|
| **B1** | **C** | `routines/fm/exhaustive_val/engine.py:312-317` (`_cohort_val_patients`) + `:290` (`_split_n_patients`) | Budget is enforced in **patients**, `metrics.csv` is written in **scans**. LUMIERE (7.0 scans/pt) turns a 7-patient budget into **65 of 116 rows (56 %)**; UCSF-PDGM gets 7 (6 %). Every row-mean is a statement about LUMIERE. | Average within patient, then average unweighted across cohorts. Use the per-(cohort, nfe, region) `aggregate.csv`, never a row-mean over `metrics.csv`. | §17 |
| **B2** | **C** | `routines/fm/exhaustive_val/engine.py:334` (`elif "splits/test" in f`) | Cohorts with no `splits/cv/fold_<n>/val` fall through to the **test split**. BraTS-Africa-Glioma + BraTS-Africa-Other + BraTS-PED = 18/116 rows, so **16 % of the aggregate is held-out test data**. Correct for OOD monitoring; leakage the moment it feeds early stopping. | Restrict any training-feedback signal to `role: cv` cohorts. Add a `role` column to `metrics.csv`. **Raise** (do not warn) if a `test_only` cohort reaches the monitor. | §17 |
| **B3** | **C** | `config.yaml training.gradient_clip_val: 1.0` | At 1.0 the L2 arm clips ~half its steps vs L1's 1.08 %, so the §18 arms would differ in optimiser behaviour, not estimator. | **`gradient_clip_val: 5.0`** in all arms + the main retrain. Validity: `grad_clip_active` < 5 % per arm. Abort rule in §18. | §18 |
| **B4** | **H** | `routines/fm/train/engine.py:1464` hard-codes `ckpt_monitor = "train/total_epoch"`, passed as `monitor_key=` at `:1466`/`:1475`; `validation.every_epochs: 0` means no `val/*` key ever exists | `training.best_metric_{name,region,nfe}` look decisive and are **inert**. Editing them changes nothing. This inertness caused a wrong conclusion in an earlier analysis. | Delete the three fields, **or** have the engine resolve them and raise when the referenced metric is not logged. Never leave them live-looking and dead. | §2 |
| **B5** | **H** | `config.yaml output.retention_n_checkpoints: 3` with `checkpoint_every_epochs: 25` over ~1900 epochs | ~76 checkpoints written, 3 kept. The quality-peak checkpoints do not exist on disk and cannot be recovered. | `retention_n_checkpoints: 40`. ~110 GB; cheapest insurance in the project. | §1 |
| **B6** | **H** | `src/vena/model/fm/eval/exhaustive.py:249` vs `routines/fm/exhaustive_val/engine.py:1271` | **`psnr_db`/`ssim` are box-wide** (`mask = torch.ones_like(p)`) while **`mae_whole`/`mse_whole` are brain-masked** (`whole_mask`, which also sets `n_voxels_brain`). "Whole" means two different things in adjacent columns of the same row. | Rename: `psnr_db_box`/`ssim_box` (mask-free) and `mae_brain`/`mse_brain` (masked). Bump the CSV schema; update the module docstring in the same change (rule 17). | §15 |
| **B7** | **H** | `metrics.csv` column set | **No `psnr_db_brain` / `ssim_brain`** — the only admissible early-stopping metric (§15) does not exist. ET/WT metrics are oracle-derived and inadmissible. | Add both, computed inside `m_brain_img`. This blocks item 3 of §19. | §15 |
| **B8** | **H** | `metrics.csv` column set | **No intensity statistics** — no percentile, no signed bias, no region mean. Answering "is the rim bright enough?" needed a separate A100 decode job (1679496). Blocks the §18 readout. | Add `p995_pred_brain, p995_real_brain, mean_et_pred, mean_et_real, mean_bnwt_pred, mean_bnwt_real`. The decoded volume is already in memory at `engine.py:752`. | §5 |
| **B9** | **H** | `src/vena/model/fm/controlnet/losses/cfm.py:83-84`, `:108-111`, `:139-141` | `CFMLoss` **rejects `norm="huber"`** at the validator, and there are **two** dispatch sites. Patching only `:139-141` leaves the `reduction="none"` (region-weighted) path silently falling back to L1 — the same inert-config shape as B4. | Add `huber` to the validator **and both** dispatch sites. δ pre-registered from the control arm's median absolute residual (≈ 0.90), recorded in `decision.json`. | §18 |
| **B10** | **H** | `src/vena/model/fm/lightning/module.py:1536-1539` (`load_state_dict(..., strict=False)`) + `:435-442` (`_trunk_module` only registered when `trunk.trainable`) | A `trainable: false` run warm-started from a `trainable: true` checkpoint matches **zero** keys, logs `loaded=0` at INFO, and trains to completion on stock weights. Voided arm J0 — 1 d 21 h of A100. | **Raise** when `loaded == 0`. | §9 |
| **B11** | **H** | SLURM worker `PYTHONPATH` | `<repo>/src` isolates `vena` but **not** `routines` (it lives at the repo root), so the shorter form silently loads `routines` from a stale shared checkout. Killed job 1679494. | `PYTHONPATH=<repo>/src:<repo>`; verify from a foreign CWD. | §9 |
| **B12** | **M** | `routines/fm/exhaustive_val/engine.py:454`; policy comments at `exhaustive_launcher.py:68` and `train/engine.py:481-484` | `latent_preds.h5` (527 MB) written **every** cadence pass and **explicitly never pruned** → **38 GB per run**, verified on v3a. | Add `exhaustive_val.latent_preds_every_n: int = 4`, counted in **cadence passes** and anchored at the first pass (so epoch 0 is always captured). `0` = never, `1` = current. Fix all three docstrings. | §13a |
| **B13** | **M** | `metrics.csv` column set; implementation exists at `src/vena/validation/metrics_paired.py:231` / `:267` | **No MS-SSIM**, despite it being the metric with the largest dynamic range on this project's failure mode (§7: 0.0602 spread vs 0.0037 for single-scale SSIM_wt). | Add `ms_ssim_brain, ms_ssim_wt_bbox`; import from `vena.validation.metrics_paired` (do not re-implement — `extensibility.md`). Honour the documented `min_dim = 90` NaN guard. | §13c |
| **B14** | **M** | `src/vena/model/fm/controlnet/losses/cfm.py` module docstring | Stale (rule 17): *"we keep the MSE formulation (proposal §5.2) … we follow the proposal's text rather than the upstream script"* — the opposite of the production recipe since S1 v2. | Rewrite to state the production default is `l1` and why; note the constructor default remains `l2`. | §18 |
| **B15** | **L** | `.claude/rules/preflight-pattern.md` | Documents `decision.json` schema **0.8.0**; v3a's actual `decision.json` is **0.10.0**. | Update the rule to 0.10.0 and record the intervening bumps. | §18 |
| **B16** | **L** | `CLAUDE.md` "Documentation source-of-truth" | Names `/media/.../vena/docs/proposal.md` as authoritative and says "the proposal wins". **That file does not exist.** Several rules defer to it. | Repoint at the surviving docs (`literature.md`, `training_routine.md`, …) or restore the file. | Appendix |

### N — Verified NON-defects. Do not "fix" these.

| # | thing that looks wrong | why it is correct |
|---|---|---|
| **N1** | `grad_norm_cn_preclip` == `grad_norm_trunk_preclip` **identically** across all 400 000 v3a steps (mean 0.331, median 0.253, p95 0.543, max 111.902) | **By construction.** `module.py:1061-1063` documents `grad_norm_cn_*` as a historical misnomer for the *combined* ControlNet + unfrozen-trunk norm. v3a sets `controlnet.enabled: false`, so the combined set **is** the trunk set. A naming wart kept for CSV back-compat, not a logging bug. An earlier pass of §18 called this a defect; that call is retracted. |
| **N2** | Whether exhaustive-val normalises the reference at 99.95 | **Already correct at HEAD.** `exhaustive.py:56,109` default to `ENCODER_PERCENTILE_UPPER`; `engine.py:723` (metrics) and `:879` (figures) both call `load_real_t1c_box` **without** overriding it. Metrics, the `ssim_by_pid` best/worst ranking, the row annotations and the figure display window all share one reference. Add the §13b assertion for provenance — do not re-plumb. |
| **N3** | `CFMLoss.__init__(norm="l2")` default while production runs `l1` | Harmless: every production YAML sets `norm` explicitly and it round-trips into `decision.json`. Change it or don't — but decide deliberately rather than as a drive-by. |
| **N4** | A6's exhaustive-val being scored at percentile 99.5 rather than 99.95 | Correct to leave. A6's commit `1ad2ba4` predates `01350e0`, and so does v3a's — **both** runs used 99.5, which is exactly what makes the §11 head-to-head valid. Do not re-score one side only. |

### K — Known operational traps that apply to the Picasso launch

| # | trap | guard |
|---|---|---|
| **K1** | `sbatch --parsable` returns **ANSI-coloured** job IDs; a raw `--dependency=afterok:$ID` is *accepted* and silently recorded as `Dependency=(null)` | Strip the escape codes, then read the dependency back with `scontrol show job` before trusting the chain. |
| **K2** | A100 nodes expose an **untyped** gres | Pin with `--gres=gpu:1 --constraint=a100`. `--gres=gpu:A100:1` matches no node; `--constraint=dgx` also matches B200, which kills the cu124 env. |
| **K3** | `set -u` in a worker that activates the `vena` env | Forbidden — `SYS_SYSROOT` is unbound during `gxx_linux-64` activation. Use `set -eo pipefail`. |
| **K4** | Any new sampler wrapping `RFlowScheduler` with `use_timestep_transform=True` | Must plumb `input_img_size_numel` or MONAI divides `None / int` and **every** per-patient pass fails with a silent WARNING and an empty `metrics.csv`. Patched at `exhaustive_val/engine.py:332`; fallback `48*56*48`. |
| **K5** | Stale-artifact cleanup by `mv` on a live run tree | Killed 10 jobs on a prior occasion (commit `288bf4c`). Never move a directory a running job holds open. |

### Assertions to add to the run log (one place, all of them)

```
ENCODER_PERCENTILE_UPPER == 99.95                    # §13b — N2 provenance
brain_mask_source == "masks/brain_latent"            # §15 — never the real_box>0 fallback
monitor cohort set ∩ {role: test_only} == ∅          # §17 / B2
warm-start loaded > 0                                # §9 / B10
grad_clip_active mean < 0.05 after 5 000 steps       # §18 / B3
PYTHONPATH contains "<repo>/src:<repo>"              # §9 / B11
```

Each guards a defect that has **already** cost this project a run. Raise, do not warn — every one of these was
survivable-looking at WARNING level and that is precisely why it was missed.

---

## 19. Summary — the diff against v3a's `config.yaml`

| # | key | v3a | proposed | force of evidence | § |
|---|---|---|---|---|---|
| 1 | `output.retention_n_checkpoints` | `3` | `40` | **Strong** — the peak epochs are gone from disk | §1 |
| 2 | `training.best_metric_{name,region,nfe}` | `mse_latent`/`bg`/`5` | remove, or wire up + fail loudly | **Strong** — currently inert; caused a wrong conclusion | §2 |
| 3 | checkpoint / EarlyStopping monitor | `train/total_epoch` | **`ssim_brain`**, cv-cohorts only, cohort-balanced, 3-point smoothed | **Strong** — train loss ≠ quality and is not comparable across §18's norms; ET metrics are inadmissible (oracle) | §2, §15, §17, §18 |
| 4 | `run.max_epochs` (effective stop) | ~~\~1900 via patience~~ **CORRECTED: 1923 via `total_steps=400000`; EarlyStopping never fired** | `total_steps: 800000` (binding); `patience: 150` (runaway guard only); `max_epochs: 10000` (non-binding) | **Strong** — 45–61 % of every run past its quality peak; `total_steps` is the actual termination lever | §3, B17 |
| 5 | `training.conditioning_dropout_p` | `0.0` | `0.1` | **Strong** — cannot be retrofitted; enables CFG | §4 |
| 6 | exhaustive-val CSV columns | 46 cols, no intensity stats, **no MS-SSIM** | add **8**; rename the `whole`/box pairs; add `role` | **Strong** — zero cost; "whole" currently means two different things | §5, §13c, §15, §17 |
| 7 | `loss.cfm.reduction` (region weighting) | `mean` | **unchanged** | **Strong** — v3b vs v3b-rw says RW harms | §7 |
| 8 | `rflow.use_timestep_transform` | `true` | **unchanged pending diagnostic** | Hypothesis only | §8 |
| 9 | `model.trunk.input_concat.cond_latents` | `[t1pre, t2, flair]` | **unchanged — keep T2** | **Strong** — A6 worse on every metric; peak epoch 150 vs 525 | §11 |
| 10 | `loss.cfm.norm` | `l1` | **3 arms: `l1` (control) / `l2` / `huber`** | **Approved** — L1 breaks the FM identity; its median bias predicts the measured under-saturation; never cleanly A/B'd | §12, §18 |
| 11 | `exhaustive_val.latent_preds_every_n` | *(field does not exist)* | add; default `4` | **Strong** — 38 GB/run, explicitly never pruned | §13a |
| 12 | run-log assertions | none | `ENCODER_PERCENTILE_UPPER == 99.95`; `brain_mask_source == masks/brain_latent`; no `test_only` cohort in the monitor; `loaded > 0` on warm-start | **Strong** — each guards a defect that has already cost a run | §6, §9, §13b, §15, §17 |
| 13 | posterior resampling `z = μ + σ·ε` | fixed cached latents + 5 offline variants | **measure `E[σ]/SD[μ]` first** | Hypothesis — the one substantive T1C-RFlow delta | §14c |
| 14 | **exhaustive-val cohort mix** | 9 cohorts, row-mean, **56 % LUMIERE, 16 % `test_only`** | cv-only, cohort-balanced, patient-averaged; raise `n_patients` | **Strong** — would leak test data into early stopping | §17 |
| 15 | `training.gradient_clip_val` | `1.0` | **`5.0`** in every arm and the main retrain, `grad_clip_active < 5 %` as a validity criterion | **Strong** — at `1.0` the L2 arm clips ~50 % of steps vs L1's 1.08 %; 5.0 gives 9.2× headroom over L1's p95 and ~4.1× over L2's projected p95 | §18 |
| 16 | `grad_norm_{cn,trunk}_preclip` | identical values | **no change — N1, verified non-defect** | Retracted — the columns are identical by construction (`controlnet.enabled: false`) | §18, §20 N1 |

**An implementing agent should work from §20, not from this table.** §20 carries every defect with its
`file:line`, its fix, and — equally important — the four things that look like bugs and are not.

### What the questions resolved to

| Q | Answer |
|---|---|
| Drop T2 (adopt t1pre+FLAIR)? | **No.** A6 loses 0.45 dB PSNR / 0.19 dB PSNR_ET and peaks at epoch 150 instead of 525. Keep it as a published ablation row. |
| Is L1 correct? | **Unresolved, and that is the finding.** Field-standard, but it violates the FM equivalence theorem and its median bias predicts VENA's exact failure mode. Prior L2 run confounded three ways — most likely by gradient clipping (§18). Ablation approved. |
| Exhaustive-val fixes? | Gate `latent_preds.h5` (38 GB → 10 GB); add MS-SSIM; the 99.95 normalisation is **already correct** end-to-end including best-patient selection — assert it rather than fix it. |
| What does T1C-RFlow do differently? | **Not the loss** (byte-identical L1) and **not the conditioning** (its `seg` is T2-FLAIR, = A6, VENA's worst arm). It resamples the VAE posterior every batch and selects on held-out val loss. |
| SSIM as the base acceptance metric? | **Change the mask, not the metric.** Whole-box SSIM and PSNR agree at ρ=0.907 and pick the same epoch. Monitor brain-masked SSIM; **never** select on ET/WT — those masks are oracles. |
| Anything missing? | **Yes — §17.** The val aggregate is 56 % LUMIERE (7 patients × ~9 scans) and 6 % UCSF-PDGM, and 16 % of it comes from `test_only` cohorts' `splits/test`. Wiring the monitor to it without fixing this leaks test data into early stopping. |

### Order of operations

1. **Fix §17 first.** It is a prerequisite for items 3, 4 and 10 — every one of them wires a decision to this
   aggregate, and all three are unsound until the cohort mix and the `test_only` fall-through are fixed.
2. Land the rest of the instrumentation: §13a gate, §13c MS-SSIM, §5 intensity columns, §15 column rename +
   `psnr_db_brain`/`ssim_brain`, §12-row assertions, §18 grad-norm logging fix, `huber` in `CFMLoss`.
3. Run the free α-binned diagnostic (§8) and let job 1679496 report (§7).
4. Measure `E[σ]/SD[μ]` on ~20 patients (§14c). One cheap encode decides whether posterior resampling is live.
5. Launch the retrain generation: the three §18 loss arms, all `resume_from: baseline`, all carrying items
   1–6, 9, 11, 12, 14, 15. Do not replace v3a in place — it anchors the S3 matrix.
6. Re-derive §1–§3, §11 and §15 under the corrected aggregation; regenerate `paired_fidelity` under the 99.95
   normalisation (§6). Neither set of numbers is publication-grade until both are done.

---

## Appendix — documentation drift found while writing this

`CLAUDE.md` names `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` as the method source-of-truth and
states "whenever this file drifts from the proposal, the proposal wins". **That file does not exist.** The `docs/`
directory contains `literature.md`, `training_routine.md`, `decoder_perceptual_loss_s3.md`,
`proposal_contrastive_loss.md`, and `proposal_deprecated_26052026.md`. The authoritative pointer in the project's
top-level rules resolves to nothing — worth fixing, since several rules defer to it.

---

## 21. Implementation log

*(Append-only. Sections §0–§20 are never edited. Timestamp every entry. If §20 is wrong, say so here.)*

---

### 2026-07-29T00:00Z — GATE 1: Implementation plan (agent, feature/v3a-retrain-instrumentation)

Branch: `feature/v3a-retrain-instrumentation` (local checkout; NOT a git worktree — rsync compatibility).

#### Line-number verification against HEAD

All spec file:line references re-verified before writing this plan. Discrepancies noted inline.

| Spec ref | Claimed line | Verified |
|---|---|---|
| `_cohort_val_patients` | `engine.py:309` | ✓ `def _cohort_val_patients` at 309 |
| `elif "splits/test"` fallthrough | `engine.py:334` | Needs Phase-1 confirm (indexed range cut off) |
| `full_volume_psnr_ssim` mask=ones | `exhaustive.py:249` | ✓ `mask = torch.ones_like(p)` in that function |
| `engine.py:1271` n_voxels_brain | `:1271` | ✓ `n_voxels_brain` set from `whole_mask` |
| `cfm.py:83-84` norm validator | `:83-84` | ✓ `if norm not in ("l2", "l1"): raise` |
| `cfm.py:108-111` reduction="none" | `:108-111` | ✓ `voxel = F.mse_loss(...)` / `F.l1_loss(...)` with `reduction="none"` |
| `cfm.py:139-141` mean/sum dispatch | `:139-141` | ✓ final `if self.norm == "l2"` / `return F.l1_loss(...)` |
| `module.py:1536-1539` load_state_dict | `:~1540` | ✓ `self.load_state_dict(loadable, strict=False)` |
| `train/engine.py:1464` ckpt_monitor | `1464` | ✓ `ckpt_monitor = "train/total_epoch"` |
| `ms_ssim_brain` | `metrics_paired.py:231` | ✓ confirmed |
| `ms_ssim_wt_bbox` min_dim=90 | `metrics_paired.py:267,275` | ✓ confirmed |

#### Orchestrator overrides (three items that deviate from the spec body)

1. **B6 — NO RENAME.** Spec says rename `psnr_db`→`psnr_db_box`, `ssim`→`ssim_box`, `mae_whole`→`mae_brain`, `mse_whole`→`mse_brain`. **OVERRIDE: DO NOT rename.** Add docstring clarification to `full_volume_psnr_ssim` stating that `psnr_db`/`ssim` are box-wide (mask=ones) and that `mae_whole`/`mse_whole` are brain-masked (naming wart retained for archived-run compat). Add the missing `psnr_db_brain` and `ssim_brain` per B7. Record this in `decision.json` schema notes.
2. **No git worktree.** Main checkout only. rsync uses the real `.git` directory; a worktree `.git` file breaks `git rev-parse` on Picasso.
3. **No Picasso submission without explicit user approval (GATE 2).**

#### Per-item plan (B1–B16)

---

**B1 — Aggregation: patient mean then cohort mean (C, blocks monitor)**

*File:* `routines/fm/exhaustive_val/engine.py`  
*Functions changed:* `_write_aggregate_csv`, `_run_multi_cohort`, new helper `_cohort_balanced_aggregate`  
*Aggregation contract:*  
1. Group `metric_rows` by `(cohort, patient_id, nfe)` → compute per-patient mean over their scan-rows.  
2. Group patient means by `(cohort, nfe, region)` → compute cohort mean over patients.  
3. Final aggregate over cv-only cohorts = unweighted mean of cohort means.  
`aggregate.csv` columns: `cohort, nfe, region, metric, value, n_patients, n_scans`.  
`aggregate_cv.csv`: a second file restricted to `role: cv` cohorts, used by the monitor callback.  
*Blast radius:* `_write_aggregate_csv` callers in the same engine only. No external consumers at HEAD.  
*Test:* `tests/model/fm/test_exhaustive_aggregation.py` — unit, no GPU. Synthetic `metric_rows` with known scan-to-patient ratios (mimicking LUMIERE's 7.0 scans/patient vs UCSF-PDGM's 1.0) must produce equal cohort weights after patient-level averaging.

---

**B2 — role column + test_only guard (C, blocks monitor integrity)**

*Files:* `routines/fm/exhaustive_val/engine.py`  
*Changes:*  
- `_cohort_val_patients`: receive `cohort_role: str` from registry; add it to every row written in `metric_rows`.  
- `_run_multi_cohort`: pass `role` through; guard: if `role == "test_only"` and `is_monitor_cohort`, **raise** `ExhaustiveValError("test_only cohort '%s' reached the monitor")`.  
- New `role` column in `metrics.csv` header (append to `_V3_EXTRA_COLS` / `_BASE_COLS`).  
*Test:* extend `test_exhaustive_aggregation.py` — synthetic rows with mixed `role` values; assert that the cv-only aggregate excludes test_only rows; assert the guard raises.

---

**B3 — gradient_clip_val: 1.0 → 5.0 (C, confounds ablation) [CORRECTION: also a Pydantic default]**

*Files:* `routines/fm/train/engine.py` (`TrainingConfig.gradient_clip_val: float` default changed from `1.0` to `5.0` at line ~414); all new production YAML configs; loginexa smoke.  
*Code change:* Pydantic default at `engine.py:414` (confirmed line; field is `gradient_clip_val: float = 1.0`).  
*Assertion:* `grad_clip_active < 0.05` after 5000 steps — logged as an assertion-callback check (see assertion block).  
*Test:* `tests/routines/fm/test_train_config_schema.py` — load `TrainingConfig()` with no YAML overrides; assert `gradient_clip_val == 5.0`.

---

**B4 — dead EarlyStopping fields; fixed-length training (H) [REDESIGNED by orchestrator 2026-07-29]**

*Orchestrator evidence:* EarlyStopping with `strict=False` skips check when key absent but does NOT increment `wait_count`, so patience=250 epochs silently becomes 250 cadence passes = 6250 epochs when the monitor is sparse. Also `mode="min"` hard-coded at `engine.py:1528`; SSIM requires `mode="max"`. Building metric readback would replace one inert-knob-that-looks-live with another.

*Decision:* Fixed-length training for all three arms.

*File:* `routines/fm/train/engine.py`  
*Change:*  
1. Delete `best_metric_name`, `best_metric_region`, `best_metric_nfe` from `TrainingConfig`. Add a guard in `_assert_run_invariants` that raises `ConfigurationError` if any of these keys appear in a loaded YAML (defensive: blocks old-config stale reads).  
2. Ensure no log line in `Engine.run()` claims "EarlyStopping ENABLED" when `patience` is `None`. Existing `if cfg.training.patience is not None:` block already gates this correctly — verify and add a corresponding log line for the disabled case.  
3. `max_epochs: 1100` + `patience: null` in all three arm configs; EarlyStopping never instantiated.  
4. `ckpt_monitor` remains `"train/total_epoch"` (mode=min); checkpoint retention at 40 keeps the quality-peak range.  
5. Post-hoc selection: new standalone script `scripts/select_checkpoint.py` reads `aggregate_cv.csv` per arm, applies patient-mean then cohort-mean aggregation, returns best epoch by `ssim_brain`. Not coupled to the training loop.  
*Blast radius:* `TrainingConfig` Pydantic model (field deletion); `Engine.run()` log path; no Lightning internals touched.  
*Test:* `tests/routines/fm/test_train_config_schema.py` — assert the three dead fields are no longer accepted by `TrainingConfig`.

---

**B5 — retention_n_checkpoints: 3 → 40 (H) [CORRECTION: also a Pydantic default]**

*Files:* `routines/fm/train/engine.py` (`OutputConfig.retention_n_checkpoints: int` default changed from `3` to `40` at line ~501); all new production configs.  
*Code change:* Pydantic default at `engine.py:501`.  
*Test:* `tests/routines/fm/test_train_config_schema.py` — load `OutputConfig()` with no overrides; assert `retention_n_checkpoints == 40`.

---

**B6 — OVERRIDE: no rename; add docstring clarification only (H)**

*File:* `src/vena/model/fm/eval/exhaustive.py`  
*Change:* Add explicit docstring to `full_volume_psnr_ssim` stating: "`psnr_db`/`ssim` returned here are MASK-FREE (mask = torch.ones_like) and placed in the `psnr_db` / `ssim` CSV columns. They include the intra-box background. Brain-masked equivalents are `psnr_db_brain`/`ssim_brain` computed separately in the exhaustive-val engine." No column rename.  
*Test:* none needed (docstring change only).

---

**B7 — add psnr_db_brain / ssim_brain columns (H, unblocks monitor)**

*File:* `routines/fm/exhaustive_val/engine.py`  
*Change:* In `_per_patient_region_metrics` (or equivalent), after computing `mae_whole`/`mse_whole` from `brain` mask, also compute:  
```python
psnr_brain = _scalar(image_metrics.psnr(p[None,None], r[None,None], brain[None,None]))
ssim_brain = _scalar(image_metrics.ssim(p[None,None], r[None,None], brain[None,None]))
out["psnr_db_brain"] = psnr_brain
out["ssim_brain"] = ssim_brain
```
Add both to `_V3_EXTRA_COLS`.  
*Test:* `tests/model/fm/test_exhaustive_new_columns.py` — with a synthetic pred/real/brain_mask triplet, verify `ssim_brain` is NaN when mask is empty and a finite value otherwise.

---

**B8 — intensity statistics columns (H, unblocks §18 readout)**

*File:* `routines/fm/exhaustive_val/engine.py`  
*Change:* At `engine.py:752` (where decoded pred and real volumes are in memory), add:  
```python
out["p995_pred_brain"] = float(torch.quantile(pred[brain], 0.995).item()) if brain.any() else float("nan")
out["p995_real_brain"] = float(torch.quantile(real[brain], 0.995).item()) if brain.any() else float("nan")
# ET mean: requires ET mask (m_et_img)
out["mean_et_pred"] = float(pred[m_et_img].mean().item()) if m_et_img is not None and m_et_img.any() else float("nan")
out["mean_et_real"] = float(real[m_et_img].mean().item()) if m_et_img is not None and m_et_img.any() else float("nan")
# BNWT = brain & ~wt mean (background wrt tumour)
out["mean_bnwt_pred"] = float(pred[bnwt].mean().item()) if bnwt is not None and bnwt.any() else float("nan")
out["mean_bnwt_real"] = float(real[bnwt].mean().item()) if bnwt is not None and bnwt.any() else float("nan")
```
Add 6 columns to `_V3_EXTRA_COLS`.  
*Test:* covered by `test_exhaustive_new_columns.py`.

---

**B9 — CFMLoss huber branch (H, gates arm C)**

*File:* `src/vena/model/fm/controlnet/losses/cfm.py`  
*Three places to change (all three required; missing any one silently misbehaves):*  
1. Validator at `:83-84`: `if norm not in ("l2", "l1", "huber"): raise`  
2. `reduction="none"` branch at `:108-111`: add `elif self.norm == "huber": voxel = _pseudo_huber(inputs.v_orig, inputs.u_target, self.delta)`  
3. Mean/sum dispatch at `:139-141`: add `elif self.norm == "huber": return _pseudo_huber_reduced(inputs.v_orig, inputs.u_target, self.delta, self.reduction)`  

New module-level helper:  
```python
def _pseudo_huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """Pseudo-Huber loss elementwise. Quadratic for |r| < delta, linear beyond."""
    r = pred - target
    return delta**2 * (torch.sqrt(1 + (r / delta)**2) - 1)
```

Add `delta: float = 0.90` parameter to `CFMLoss.__init__` (pre-registered from control arm; ≈ median absolute velocity residual, Song & Dhariwal ICLR 2024). Record `delta` in `decision.json`.  
*Blast radius:* `CFMLoss` constructor callers — all pass `norm=` explicitly from YAML so the new default does not affect them. `region_weights` + `brain_tc_weights` paths both route through the `reduction="none"` dispatch, so both are covered.  
*Test:* `tests/model/fm/test_cfm_loss_huber.py` — unit. Verify: (a) `norm="huber"` accepted; (b) small residuals give quadratic output; (c) large residuals give ~linear output; (d) `reduction="none"` path correct (for region-weighted usage); (e) invalid norm still raises.

---

**B10 — raise when load_warm_start loaded=0 (H)**

*File:* `src/vena/model/fm/lightning/module.py`  
*Change:* After `result = self.load_state_dict(loadable, strict=False)` and the existing INFO log, add:  
```python
if len(loadable) == 0:
    raise RuntimeError(
        f"load_warm_start: loaded 0 keys from {p}. "
        "Most likely cause: 'trainable: false' run warm-started from a 'trainable: true' checkpoint "
        "(trunk keys only exist in state_dict when trainable=True). "
        "Check trunk_config.trainable."
    )
```  
*Blast radius:* `_WarmStartCallback.on_fit_start` — the raise propagates and kills the run before any training step. This is the intended behaviour.  
*Test:* `tests/model/fm/test_load_warm_start_guard.py` — mock a checkpoint with no overlapping keys; assert `RuntimeError` raised.

---

**B11 — module-path assertion (H) [CORRECTION: assert resolved paths, not PYTHONPATH string]**

*File:* `routines/fm/train/engine.py`  
*Change:* At the top of `Engine.run()` (before any side effect), add:  
```python
import routines, vena as _vena_pkg
repo_root = Path(__file__).resolve().parents[3]  # routines/fm/train/engine.py → 3 parents → VENA/
routines_root = Path(routines.__file__).resolve().parents[1]   # VENA/routines/__init__.py → VENA/
vena_root = Path(_vena_pkg.__file__).resolve().parents[2]      # VENA/src/vena/__init__.py → VENA/
if routines_root != repo_root:
    raise EnvironmentError(
        f"'routines' package resolves to {routines_root}, expected {repo_root}. "
        "PYTHONPATH=<repo>/src:<repo> required."
    )
if vena_root != repo_root:
    raise EnvironmentError(
        f"'vena' package resolves to {vena_root}, expected {repo_root}."
    )
```  
This is CWD-independent: it checks where Python actually loaded the modules from, not what is in the env-var string. A PYTHONPATH that sets the string correctly but uses a stale editable install still fails.  
*Also:* fix all SLURM worker scripts under `routines/fm/train/slurm/` and `routines/fm/exhaustive_val/slurm/` to use `PYTHONPATH=${REPO}/src:${REPO}`.  
*Test:* `tests/routines/fm/test_preflight_pythonpath.py` — monkeypatch `routines.__file__` to point outside the repo root; assert `EnvironmentError` raised. Run from a foreign CWD to verify CWD-independence (use `tmp_path` fixture).

---

**B12 — latent_preds_every_n gating (M)**

*Files:* `routines/fm/exhaustive_val/engine.py` (gate at call to `write_latent_preds_h5`), `src/vena/model/fm/lightning/exhaustive_launcher.py` (or wherever launcher YAML is built), `ExhaustiveValConfig` Pydantic model.  
*Change:*  
- Add `latent_preds_every_n: int = 4` to `ExhaustiveValConfig`.  
- Maintain `_cadence_pass_count: int` counter in the engine (incremented each call to `run_epoch`).  
- Gate: `if latent_preds_every_n > 0 and (_cadence_pass_count % latent_preds_every_n == 0 or _cadence_pass_count == 1): write_latent_preds_h5(...)`.  
- Fix the three docstrings that claim "never pruned" to document the new gating policy.  
*Test:* `tests/model/fm/test_latent_preds_gating.py` — unit; verify gate fires on passes 1, 5, 9 (every_n=4) and skips 2, 3, 4, 6, …

---

**B13 — MS-SSIM columns (M)**

*File:* `routines/fm/exhaustive_val/engine.py`  
*Change:* After the existing `_emit` calls, add:  
```python
from vena.validation.metrics_paired import ms_ssim_brain, ms_ssim_wt_bbox
out["ms_ssim_brain"] = ms_ssim_brain(pred[None,None], real[None,None], brain_mask)
out["ms_ssim_wt_bbox"] = ms_ssim_wt_bbox(pred[None,None], real[None,None], wt_mask)
```
Import must be at module top (not inside the call; formatter hook strips unused imports, so add import and use in same edit).  
Honour `min_dim=90` NaN guard — already implemented in the library; the call returns NaN when the WT bbox is too small.  
*Test:* `test_exhaustive_new_columns.py` — verify `ms_ssim_brain` is finite when both volumes are random tensors of shape `(1, 100, 100, 100)` and brain mask is all-True; verify `ms_ssim_wt_bbox` is NaN when WT bbox dims < 90.

---

**B14 — cfm.py module docstring (M)**

*File:* `src/vena/model/fm/controlnet/losses/cfm.py`  
*Change:* Replace the stale paragraph *"Per the MAISI-v2 reference implementation we keep the MSE formulation… we follow the proposal's text rather than the upstream script"* with: *"Production default is `norm='l1'` (all runs since S1 v2, 2026-06-20). Constructor default remains `norm='l2'` for backward compatibility with unit tests that do not set it explicitly; every production YAML sets norm explicitly. The `'huber'` option (pseudo-Huber, Song & Dhariwal ICLR 2024) is available for the §18 ablation arm C; δ is set from the L1 arm's median absolute velocity residual (≈ 0.90)."*

---

**B15 — preflight-pattern.md schema drift (L)**

*File:* `.claude/rules/preflight-pattern.md`  
*Change:* Update the `decision.json` version reference from `0.8.0` to `0.10.0`. Add changelog lines: 0.9.0 (LPL coupling, 2026-06-09), 0.10.0 (recipe tag + resume classification, 2026-06-10).

---

**B16 — CLAUDE.md dead proposal.md pointer (L)**

*File:* `CLAUDE.md` "Documentation source-of-truth" table  
*Change:* Update the "Proposal" row from the non-existent path to: `training_routine.md` (method details) + `proposal_deprecated_26052026.md` (archived original). Add a note: *"proposal.md no longer exists; training_routine.md is now authoritative for architecture and loss details."*

---

#### Assertion block — all six, in `Engine.run()` before any side effect [CORRECTIONS: added 3 missing; raise not warn]

Implemented as `_assert_run_invariants(cfg, registry)` called at the top of `Engine.run()`, before any side effect. All six raise, not warn — every one was survivable-looking at WARNING level and that is why it was missed.

```python
def _assert_run_invariants(cfg: TrainingRoutineConfig, registry: CorpusRegistry) -> None:
    """Hard guards that have each already cost this project a run. Raise, do not warn."""
    # N2 provenance — §13b
    from vena.common import ENCODER_PERCENTILE_UPPER
    if ENCODER_PERCENTILE_UPPER != 99.95:
        raise AssertionError(
            f"ENCODER_PERCENTILE_UPPER={ENCODER_PERCENTILE_UPPER} != 99.95. "
            "The 99.95 percentile normalisation is load-bearing for intensity metrics."
        )
    # §15 — brain mask source must never be the real_box>0 fallback (which IS derived from the real T1c)
    brain_key = getattr(cfg.data, "brain_mask_key", None)
    if brain_key != "masks/brain_latent":
        raise AssertionError(
            f"brain_mask_key={brain_key!r}; must be 'masks/brain_latent'. "
            "The real_box>0 fallback is derived from the real T1c and leaks target information."
        )
    # §17 / B2 — no test_only cohort in monitor set
    monitor_names = {c.name for c in registry.cv_cohorts()}
    test_only_names = {c.name for c in registry.cohorts() if c.role == "test_only"}
    overlap = monitor_names & test_only_names
    if overlap:
        raise AssertionError(
            f"test_only cohorts found in monitor set: {overlap}. "
            "This leaks held-out test data into early stopping."
        )
    # §9 / B11 — resolved module paths (CWD-independent; see B11 above)
    _assert_module_paths_in_repo_root()
```

The `loaded > 0` guard is in `load_warm_start` in `module.py` (B10) — fires after `setup()`, not here.

The `grad_clip_active < 0.05` check is a **callback** (`GradClipValidityCallback`) that raises at step 5000 if the mean fraction of clipped steps exceeds 5 %. This is the §18 validity criterion — the three arms must all pass it before the ablation result is interpretable. Implemented as a new callback in `src/vena/model/fm/lightning/callbacks.py`, attached to the trainer in `Engine.run()` when `cfg.training.gradient_clip_val` is set.

---

#### New configs (§18 ablation arms + updated smoke)

Three production configs (all arms share identical settings except `loss.cfm.norm`):

| File | Arm | Key changes vs picasso_s1_1000ep_fft.yaml |
|---|---|---|
| `routines/fm/train/configs/runs/picasso_s1_v4_l1_fft.yaml` | A (L1 control) | `gradient_clip_val: 5.0`, `retention_n_checkpoints: 40`, `conditioning_dropout_p: 0.1`, `latent_preds_every_n: 4`, `max_epochs: 1100`, `patience: null` (EarlyStopping disabled) |
| `routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml` | B (L2) | same + `loss.cfm.norm: l2` |
| `routines/fm/train/configs/runs/picasso_s1_v4_huber_fft.yaml` | C (Huber) | same + `loss.cfm.norm: huber`, `loss.cfm.delta: 0.90` |

Post-hoc selection script: `scripts/select_checkpoint.py`. Reads the per-arm `aggregate_cv.csv` files, applies patient-mean → cohort-mean, reports best epoch by `ssim_brain` per arm. Not coupled to training loop. Part of this commit.

Updated smoke (exercises all new code paths in 25 min on loginexa V100):

| File | Purpose |
|---|---|
| `routines/fm/train/configs/smoke/loginexa_s1_v4_4ep_l1.yaml` | Smoke arm A; `block_until_complete: true`, `n_patients: 3`, `NFE: [1,5]`, `latent_preds_every_n: 1`, `conditioning_dropout_p: 0.1`, `gradient_clip_val: 5.0`, `patience: null` |

The smoke exercises: new CSV columns (`ssim_brain`, `ms_ssim_brain`, `p995_pred_brain`, etc.), cohort-balanced aggregation, `role` column (cv vs test_only guard), `huber` branch tested via unit test only (not reachable in 4-epoch smoke without reaching steady state). `aggregate_cv.csv` must appear in run dir and have ≥1 row — checked by the loginexa pass criteria.

schema bump: `decision.json` 0.10.0 → 0.11.0 adding `loss_cfm_norm`, `loss_cfm_delta`, `latent_preds_every_n`, `gradient_clip_val`, `exhaustive_val_aggregation: "patient_mean_then_cohort_mean"`. Written in `Engine._build_decision_payload()`. `.claude/rules/preflight-pattern.md` updated in same commit (closes B15).

---

#### Test strategy

All new tests: `pytestmark = pytest.mark.unit` (no GPU). Files:

| Test file | Covers |
|---|---|
| `tests/model/fm/test_exhaustive_aggregation.py` | B1 patient-mean then cohort-mean; B2 role guard |
| `tests/model/fm/test_exhaustive_new_columns.py` | B7 psnr_db_brain/ssim_brain; B8 intensity stats; B13 MS-SSIM/NaN guard |
| `tests/model/fm/test_latent_preds_gating.py` | B12 every_n gate |
| `tests/model/fm/test_cfm_loss_huber.py` | B9 huber, all three dispatch sites |
| `tests/model/fm/test_load_warm_start_guard.py` | B10 loaded=0 raises |
| `tests/routines/fm/test_preflight_module_paths.py` | B11 resolved module path assertion (CWD-independent) |
| `tests/routines/fm/test_train_config_schema.py` | B3/B5 Pydantic defaults; B4 dead fields rejected |
| `tests/routines/fm/test_select_checkpoint.py` | B4 post-hoc selection script: ssim_brain aggregation |

Guard test for no removed public symbols: B4 deletes config fields (`best_metric_*`) that were never in `__all__` — no guard test needed. No other public symbols removed.

---

#### Items flagged as potentially wrong or already fixed in spec

- **B15/B16 line numbers:** Doc-only changes; line numbers not material.
- **`engine.py:334` "elif splits/test":** To be confirmed by direct read in Phase 1. Spec's description of the behaviour is confirmed by indexing; fix applies regardless of exact line.
- **B4 inert fields:** Confirmed — `ckpt_monitor` is hardcoded; `best_metric_*` fields are read only to build a key that is never passed to any callback. Redesigned to fixed-length + post-hoc selection (see redesign block).
- **Monitor design (redesigned):** No readback coupling. Fixed-length training with post-hoc `ssim_brain` selection from `aggregate_cv.csv` is simpler, more auditable, and makes the three arms exactly comparable by construction.

---

#### Scope assessment

Single Phase 1 agent. All code changes in ≤7 files. B1/B2 highest risk (aggregation logic) — localised to `exhaustive_val/engine.py`. B4 simplified by redesign (no Lightning coupling). Estimated delta: ~380 lines production + ~220 lines tests.

---

#### Baseline (to be recorded at Phase 1 start)

```
df -h / | tail -1
PYTHONPATH=$PWD/src:$PWD ~/.conda/envs/vena/bin/python -m pytest \
  -m "not slow and not gpu" -q \
  --basetemp=/home/mpascual/.pytest-tmp-v3aimpl | tail -3
```

Recorded 2026-07-29:
- Disk: `/dev/nvme1n1p2  137G   64G   66G  50% /` (66 G free)
- Baseline: **1602 passed, 1 skipped, 23 deselected, 240 warnings** in 124s (full fast-suite run)

---

### 2026-07-29T21:14Z — GATE 2 APPROVED + SUBMITTED (agent, feature/v3a-retrain-instrumentation)

#### Final HEAD

```
local  HEAD: 0a1d7bf5c2ee670bd2a7842170abdf246210a2ee   branch: feature/v3a-retrain-instrumentation   dirty: 0
Picasso HEAD: 0a1d7bf5c2ee670bd2a7842170abdf246210a2ee   branch: feature/v3a-retrain-instrumentation   dirty: 0
```

Both sides verified clean at submission. **Picasso repo must not be touched until all three jobs are RUNNING** — SLURM reads the repo at start time, not submit time; any commit or rsync before start silently changes the code one or more arms run. If a fix becomes necessary before start, the procedure is: `scancel` all three, apply, resubmit together.

#### Three §18 ablation arms — job IDs

| Job ID | Name | Arm | Config | State at submit |
|---|---|---|---|---|
| 1679902 | vena-s1-v4-l1-fft | A — L1 (control) | `routines/fm/train/configs/runs/picasso_s1_v4_l1_fft.yaml` | PENDING / Priority |
| 1679903 | vena-s1-v4-l2-fft | B — L2 | `routines/fm/train/configs/runs/picasso_s1_v4_l2_fft.yaml` | PENDING / Priority |
| 1679904 | vena-s1-v4-huber-fft | C — Huber (δ=0.90) | `routines/fm/train/configs/runs/picasso_s1_v4_huber_fft.yaml` | PENDING / Priority |

All three: `TimeLimit=6-00:00:00`, `ReqTRES=cpu=16,mem=256G,node=1,gres/gpu=2`, `Features=a100`, `Partition=gpu_partition`, `Dependency=(null)`. Independent — no chaining.

Worker: `routines/fm/train/slurm/runs/worker_fm_train_picasso_v4_ablation.sh`  
Launchers: `routines/fm/train/slurm/runs/launcher_picasso_s1_v4_{l1,l2,huber}.sh`

#### First-start check (do this when each job transitions PENDING → RUNNING)

Read the job's `.out` file:
```
/mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s1-v4-{l1,l2,huber}-fft_167990{2,3,4}.out
```

Confirm line 61 (`Config: ${CONFIG_PATH}`) matches the expected per-arm YAML above, and that:
```
Git commit: 0a1d7bf5c2ee670bd2a7842170abdf246210a2ee
```

If any two arms print the same config, **`scancel` all three immediately** — `--export` propagation failure would make the ablation meaningless.

#### Resource ask rationale

`--mem=256G` is derived from `sacct` on all completed VENA FM training jobs: every completed run shows `ReqMem=256G`. The closest lower boundary that completed was never tested; 48 G (the prior default) OOM-killed inference benchmark shards. Do not "optimise" this without a completed run at the lower value.

#### §18 primary endpoint — where to read it

The §18 primary endpoint is `mean_et_pred − mean_et_real` (mean signed intensity error inside the enhancing-tumour region). It is written per epoch and per NFE to:

```
<run_dir>/exhaustive_val/epoch_NNN/metrics.csv
```

Column: `mean_et_pred` and `mean_et_real` (added by B8, `metrics_paired.py`). The signed difference is **not pre-computed** in the CSV — compute it post-hoc as `mean_et_pred − mean_et_real` per row. A value near zero means the model produces correct enhancement magnitude; a large positive value means over-enhancement; negative means under-enhancement.

Run dirs will be under `/mnt/home/users/tic_163_uma/mpascual/execs/VENA/` on Picasso (format: `YYYY-MM-DD_HH-MM-SS_s1_fft_cfm_{l1,l2,huber}_<token>/`).

The cohort-balanced aggregate (patient mean → cohort mean over CV cohorts only) is in:
```
<run_dir>/exhaustive_val/epoch_NNN/aggregate_cv.csv
```
Columns: `cohort, nfe, region, metric, value, n_patients, n_scans`. Filter `region == "et"` and `metric == "mean_et_pred"` / `"mean_et_real"` to derive the primary endpoint at the aggregate level.

#### Post-hoc checkpoint selection

Script: `scripts/select_checkpoint.py`

Usage (after all three arms complete or at any intermediate epoch):
```bash
~/.conda/envs/vena/bin/python scripts/select_checkpoint.py \
  --run-dirs <l1_run_dir> <l2_run_dir> <huber_run_dir> \
  --metric ssim_brain \
  --region brain
```

Reads each arm's per-epoch `aggregate_cv.csv`, applies the same patient-mean → cohort-mean as the training loop, returns the epoch with the highest `ssim_brain` for each arm. The selected checkpoint is `<run_dir>/checkpoints/epoch=NNN-*.ckpt`. `retention_n_checkpoints: 40` keeps all epochs in the quality-peak window (roughly epochs 600–1100 for a 1100-epoch run at the v3a convergence profile).

#### Test suite at submission HEAD

```
0a1d7bf:  1696 passed, 0 failed, 1 skipped, 23 deselected  (fast suite, no GPU/slow)
```

The one test that moved since Gate-1 baseline (1602 → 1696): all B-series items (B1–B19) landed; the single fix in this Gate-2 window was `test_resume_modes_integration.py::test_baseline_creates_new_dir` asserting `schema_version == "0.12.0"` → `"0.13.0"` after B19 bumped the post-training patch target.

#### Defects found and fixed during Gate-2 window

| ID | Description | Commit |
|---|---|---|
| B18 | `run_id` suffix documented as `<short-sha>` but actually `sha256(timestamp+pid+hostname)[:8]` — docs-only fix in `preflight-pattern.md` | earlier session |
| B19 | `decision.json` had no code provenance; added `git_sha` + `git_dirty`, bumped schema 0.12.0 → 0.13.0 | `5dcc0ab` |
| B19 import | `resolve_git_sha`/`resolve_git_dirty` called but not imported in `engine.py`; caught by first B19 smoke | `c073e14` |
| stale test | `test_baseline_creates_new_dir` asserted schema 0.12.0; fixed to 0.13.0 | `0a1d7bf` |

#### Process note (carry forward)

Re-run the fast suite after every change that touches a schema string, a contract field, or a Pydantic default. A stale green is indistinguishable from a real one until a Gate-2 review catches it. This burned us twice in this session (B19 import missing, schema assertion stale). Rule: `pytest -m "not slow and not gpu"` is a ≤3-minute operation; run it before reporting any gate status.
