# S3 — Oracle-mask injection: results and verdict

**Date:** 2026-07-29 · **Session:** `/orchestrate`, Opus @ xhigh · **Owner:** orchestrator (all numbers below
re-derived from the per-patient CSVs, not transcribed from any agent report).

**Source data.** 357 `exhaustive_val/epoch_*/metrics.csv` harvested from
`picasso:~/execs/vena/experiments/` (5 arms) plus 69 from the local v3a archive
`/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/`.
Every file is 580 rows = 116 patients × 5 NFE × 9 cohorts; 46 columns, schema identical across all six runs;
`brain_mask_source = masks/brain_latent` everywhere. Unless stated, numbers are **NFE = 20**, mean over patients.

---

## 0. Job provenance (checked before any number was read)

| arm | job | trunk | `tc` weight | SLURM state | elapsed | last cadence |
|---|---|---|---|---|---|---|
| J0 | 1643042 | freeze | 1 | COMPLETED | 1-21:40:47 | epoch 1900 |
| J1 | 1643044 | joint | 1 | **NODE_FAIL** | 2-15:14:59 | epoch 1500 |
| J2 | 1643046 | joint | 5 | COMPLETED | 3-17:52:00 | epoch 1900 |
| J3 | 1643048 | joint | 10 | COMPLETED | 3-18:08:36 | epoch 1900 |
| J4 | 1643050 | joint | 20 | COMPLETED | 3-18:22:37 | epoch 1900 |

J1 died on a node failure at epoch 1500. Every J1 comparison below is either within-arm (ep 0 → 1500) or against
the other arms **at matched epoch 1500**, so the truncation is not a confound.

---

## 1. 🔴 J0 IS VOID — it never loaded the baseline

```
J0: load_warm_start ... loaded=0    missing=671  unexpected=1307
J1: load_warm_start ... loaded=1307 missing=671  unexpected=0
```

**Mechanism, confirmed in code (not inferred from the log):**

- `src/vena/model/fm/lightning/module.py:435-442` — `self._trunk_module = self._trunk_handle.model` executes
  **only when `trunk_config.trainable` is true**.
- `src/vena/model/fm/lightning/module.py:1536-1539` — `load_warm_start` keeps only source keys that are already
  present in `self.state_dict()` with a matching shape, then calls `load_state_dict(..., strict=False)`.

v3a is a `concat_only` recipe (conditioning by input concatenation, **no ControlNet**), so its `ema_best.ckpt` is
1307 tensors living under the `_trunk_module.*` prefix. J0 sets `trunk.trainable: false`, never registers that
prefix, and therefore **matches zero keys**. It discarded the entire v3a fine-tuned trunk and trained a fresh
2-channel ControlNet on top of the **stock MAISI trunk** (`sha256=cbf3ee62ec80`, `trainable=False`).

The evidence in the metrics agrees exactly: J0 at epoch 0 scores **6.681 dB** whole-volume PSNR and **SSIM 0.065**,
where J1–J4 from the same nominal warm start score **26.88 dB / 0.911**. J0 climbs to 20.69 dB by epoch 1900 and
never reaches the point the other arms *started* from — it cannot, because its trunk is frozen at stock weights.

**Consequence.** The J0 → J1 "freeze → joint gain", the single reason J0 was in the matrix, **cannot be read**.
J0 is not a ControlNet-only lower bound; it is an unrelated from-scratch experiment. Its `dPSNR_whole = +14.0 dB`
over its own epoch 0 is recovery from a broken initialisation, not learning.

**Class of bug.** Warm-starting a `trainable: false` run from a `trainable: true` checkpoint silently loads
nothing, logs `loaded=0` at INFO, and trains to completion. This is the fourth instance this project has hit of
*an operation completing cleanly while producing an empty or stale artifact* (after S5's all-zero G-SEG table, the
stale-run-dir no-op, and the header-only `metrics.csv`). It needs a hard guard, not a log line.

---

## 2. 🔴 v3a's archived cadence curve is NOT a valid baseline

v3a's own in-run exhaustive-validation never exceeds **25.331 dB** whole-volume (peak at epoch 525) or
**16.920 dB** PSNR_ET (peak at epoch 900). But v3a's own `ema_best.ckpt`, evaluated by the arms' exhaustive-val at
their epoch 0 with a near-null ControlNet (`output_scale = sigmoid(-5) = 0.0067`), scores **26.879 dB / 18.76 dB**
on the same 116 patients with the same brain mask.

A ~1.55 dB (whole) / ~1.85 dB (ET) gap between a checkpoint and its own run's evaluation curve is not a model
property. **Any "gain over v3a" computed against the archived curve is inflated by that amount and must not be
published.** Naively differencing the two gives +3.6 to +4.0 dB PSNR_ET; the honest figure is in §3.

The only internally consistent contrast is **within-arm**: each arm's epoch 0 (= v3a `ema_best` + null ControlNet)
against its own later epochs, computed by one code path on one patient set.

### Root cause — RESOLVED: the reference was normalised at the wrong percentile

Commit **`01350e0`, 2026-07-22**, `fix(validation): canonical 99.95 encoder percentile in exhaustive-val`
(verified myself in `git log`). Its own message states the mechanism:

> the frozen MAISI latent caches were encoded at `percentile_upper=99.95`, so every decode-vs-real comparison must
> normalise the reference at 99.95 (**99.5 saturates the enhancing-rim/vessel tail** — the rho_s confound)

`ENCODER_PERCENTILE_UPPER = 99.95` now lives in `vena.common` and is the default in `vena/validation/io.py:214`.
v3a ran **2026-06-24**, before the fix; the arms ran **2026-07-24**, after it. v3a's cadence evaluations therefore
compared a prediction in 99.95-space against a reference clipped at 99.5 — clipping precisely the enhancing rim,
which is why the ET gap (2.06 dB) is larger than the whole-volume gap (1.70 dB).

Same-weights/same-patients is proved by the latent columns, which are invariant to reference normalisation:
v3a epoch_1875 `latent_mse = 1.5351`, `latent_cosine = 0.18926` vs arms epoch_000 `1.53495`, `0.18970`.

Trunk-EMA at sampling time is **ruled out** — `trunk_finetuned_snapshot` was already in the launcher at commit
`a1dd749` (2026-06-16), before v3a launched. `ema_best` not matching the cadence peak is real but minor:
`argmin(total_mean)` is epoch 1862 (25.180 dB) vs the curve peak at epoch 525 (25.331 dB) — 0.15 dB.

**TRUSTWORTHY BASELINE: the arms' own epoch-0 rows (26.879 dB whole, 18.76 dB PSNR_ET).** v3a's archived cadence
curve is biased low by ~1.7–2.1 dB and must not be used as a comparison point anywhere.

### 🔴 Blast radius beyond S3

`results/article/paired_fidelity/LATEST` resolves to **`2026-07-20T12-11-10Z` — two days before the fix.** Every
number in `table1_ring_a_fidelity.csv`, `tableS1_undersaturation.csv` and `tableS2_zgd.csv` was produced under the
normalisation that saturates the enhancing rim. The §4.1 comparison used here (v3a vs v3b-rw `p99.5`) is
within-artifact and measured identically on both sides, so the confound cancels for *that* contrast — but the
artifact as a whole should be regenerated before any of it reaches a manuscript, and any absolute ET-region claim
from it is suspect.

---

## 3. The honest result: what the oracle mask actually buys

Paired per-patient, arm's own epoch 0 → its best epoch, n = 110 (6 of 116 patients have no ET), NFE = 20,
bootstrap 95 % CI (10 000 resamples, seed 1337), Wilcoxon signed-rank.

| arm | `tc` | best ep | **ΔPSNR_ET** | 95 % CI | win rate | p | ΔPSNR whole |
|---|---|---|---|---|---|---|---|
| J1 | 1 | 1450 | **+1.49 dB** | [+0.93, +1.81] | 84 % | 1.1e-11 | +0.21 |
| J2 | 5 | 1050 | **+1.72 dB** | [+0.90, +1.99] | 76 % | 3.4e-11 | +0.16 |
| J3 | 10 | 1025 | **+1.65 dB** | [+1.10, +2.01] | 77 % | 2.8e-09 | +0.24 |
| J4 | 20 | 750 | **+2.07 dB** | [+1.34, +2.38] | 81 % | 9.5e-13 | +0.16 |

Excluding LUMIERE (which is 65/116 of the validation set and dilutes the effect), the same deltas are
**+1.78 / +2.06 / +2.44 / +2.63 dB** — so the gain is not a LUMIERE artifact; it is *stronger* without it.
Every cohort except BraTS-PED (n = 1 with finite ET) is positive in every arm.

**Whole-volume fidelity is untouched** (+0.16 to +0.24 dB), which is the expected signature of a tumour-local
conditioning effect and matches the earlier finding that MAE_brain is unchanged by the oracle mask (p = 0.97).

### The region-weighting lever is nearly exhausted

Paired against J1 at each arm's final epoch:

| contrast | ΔPSNR_ET | 95 % CI | p | ΔPSNR whole |
|---|---|---|---|---|
| J2 (tc=5) − J1 | +0.16 dB | [−0.13, +0.43] | 0.066 | −0.03 |
| J3 (tc=10) − J1 | +0.28 dB | [−0.05, +0.61] | 0.042 | **−0.10** (p=0.028) |
| J4 (tc=20) − J1 | **+0.41 dB** | [+0.05, +0.74] | 0.0048 | −0.10 |

At matched epoch 1500 the same contrasts are +0.23 / +0.27 / +0.35 dB. **A 20× increase in the tumour-core loss
weight buys +0.41 dB PSNR_ET and costs 0.10 dB of whole-volume PSNR.** The dose-response is real but flat; there
is no operating point further along this axis worth taking.

### All arms are saturated

Mean PSNR_ET over the last 100 cadence epochs minus the same over [final−500, final−400]:
J1 **+0.12**, J2 **+0.03**, J3 **−0.04**, J4 **−0.06** dB. More epochs will not help. J3 and J4 are already
declining; both peak around epoch 750–1025 and drift down for the remaining ~1000 epochs.

---

## 4. What the mask fixed, and what it did not

**Fixed — localisation.** The ET-vs-healthy-tissue error ratio `MAE_ET / MAE_healthy` falls from **1.59 → 1.24**
(J4, ep 0 → 1900), and `PSNR_ET − PSNR_healthy` goes from **−1.44 dB to +0.46 dB**. Relative to healthy tissue,
the enhancing region stops being the model's worst region. NFE behaviour is also repaired: v3a loses 1.45 dB
PSNR_ET going from NFE 2 → 20 (17.99 → 16.54), while every arm is flat at ≈20.5 dB across NFE 1–20.

**Not fixed — intensity.** Three independent lines say the residual failure is a brightness calibration problem,
not a localisation one:

1. **Under-saturation is unchanged by the mask.** From
   `results/article/paired_fidelity/LATEST/tables/tableS1_undersaturation.csv` (verified verbatim):
   VENA-S1-v3a `raw_p995 = 0.78582`, VENA-S1-v3b-rw (oracle mask) `= 0.78558` — a **−0.03 %** difference. The
   99.5th-percentile intensity of the output is identical with and without a perfect mask. For reference
   C3-SynDiff-flair reaches 0.9011 and VENA-S1-v3b (no region weighting) reaches 0.8421 — i.e. **region weighting
   made under-saturation worse** (0.842 → 0.786) in the earlier generation too.
2. **The error's *shape* inside ET never changes.** The error-concentration index `MSE_ET / MAE_ET²` is
   **1.512 at epoch 0 and 1.518 at epoch 1900** for J4 — unmoved by 1900 epochs. It is *lower* than the healthy-tissue
   value (2.03), so ET error is spread broadly and uniformly, not concentrated in a few badly-wrong rim voxels.
   Training shrinks the magnitude of a uniform offset (MAE_ET 0.119 → 0.0925) without altering its structure.
3. **Nothing in the objective supervises intensity.** The training loss is an L1 on the *velocity in latent space*
   with a region weight. Rendering the enhancing rim at 60 % of its true brightness is penalised in exact
   proportion to its contribution to latent L1 — which is small (§5).

**Read together: the model now knows *where* the enhancement is and still does not know *how bright* to make it.**

---

## 5. Why the lever is weak — the size argument

The ET region occupies a median of **431 latent voxels** (q25 192, q75 720) out of the 129 024-voxel latent grid
(48×56×48) — **0.33 %**. Tumour core is 563 voxels, **0.44 %**. Thirteen of 116 validation patients have an ET
smaller than a single 4×4×4 latent block.

The measured gain scales with that footprint. Tertiles of ET size, within-arm ΔPSNR_ET:

| arm | small (≈165 latent vox) | mid (≈456) | large (≈909) |
|---|---|---|---|
| J1 | +0.74 | +1.65 | +1.77 |
| J2 | +0.63 | +1.67 | +2.11 |
| J3 | +0.88 | +2.03 | +1.78 |
| J4 | +1.33 | +2.16 | +2.12 |

Monotone in 3 of 4 arms, but Spearman ρ = +0.09 to +0.16 with **p = 0.09–0.33 — not significant**. This is
*consistent with* a latent-resolution floor and is **not evidence for it**; it is stated here as a hypothesis to
test, not a finding.

---

## 6. FP-safety

Six patients have `n_voxels_et == 0` (BraTS-PED and UCSF-PDGM). Their WT-region MAE at the final epoch:
J1 0.0975, J2 0.0935, J3 0.0946, **J4 0.0817** — monotonically *better* at higher `tc` weight, i.e. up-weighting
the tumour core did **not** induce false enhancement on cases without any. No FP-safety objection to J4.
(This comparison is within the arm family only; the v3a column is contaminated by §2.)

---

## 6b. Ranked mechanisms for the ineffectiveness

Injection-path audit (independent agent), with every load-bearing claim re-verified by me against the **live run
config** on Picasso (`.../j2_.../config.yaml`), not the repo defaults.

**Ruled out first** — the obvious suspects are all clean:

- `output_scale_ramp: {enabled: true, ramp_steps: 5000, steepness: 10}` with **208 optimiser steps/epoch**
  (verified from `metrics/train_epoch.csv`) → `output_scale` hits 1.0 at **epoch 24 of 1900**. The branch ran at
  full strength for 99 % of training.
- `conditioning_dropout_p = 0.0` — the mask was present in every step.
- The hint network applies **no spatial downsampling** (`conditioning_embedding_num_channels: [64]` ⇒ zero
  stride-2 blocks); the 48×56×48 mask meets the trunk features at full latent resolution.
- The mask itself is fine: thresholded Dice 1.000 against the clean path, range [0.03445, 0.96555].

**The mask was delivered, at full strength, at full resolution, for the whole run. It is not a plumbing failure.**

### H1 — The objective has no term that can express "brighter" [STRONG]

The entire S1 loss is an L1 on the **velocity field in latent space**: `‖v_pred − (x_1 − x_0)‖₁`, region-weighted
(`losses/cfm.py:136`, `losses/region_weights.py:268-285`). There is no image-space term — LPL is S3-only,
contrastive is S2-only, and these runs are `stage: s1`. An L1 regression is median-seeking, so it converges to the
**median latent velocity** over patients and timesteps at TC voxels. Rendering the rim at 60 % of true brightness
is penalised only in proportion to its contribution to a latent L1 norm — which is tiny. This is the same
signature the data shows: the error inside ET is a broad uniform offset whose *shape never changes* in 1900
epochs (§4.2), and `p99.5` is identical with and without the mask (§4.1).

### H2 — A perfect mask supplies *where*, and enhancement is a *how much* question [STRONG]

`conditioning_inputs: [mask:tc_soft:identity, mask:netc_soft:identity]` — **verified in the run config**. The
ControlNet receives *only* the two tumour masks. Gadolinium uptake is blood–brain-barrier breakdown; two patients
with identical core geometry can enhance completely differently. The mask is, by construction, uninformative about
enhancement intensity. This is why the oracle mask is not a ceiling-raiser: it was never carrying the missing
information. It also explains the shape of the result exactly — localisation improved (MAE_ET/MAE_healthy
1.59 → 1.24) while intensity did not move at all.

### H3 — The gradient share is too small even at 20× [STRONG]

TC share of the weighted loss is `w·f / (1 + (w−1)·f)`. From my own measurement, TC ≈ 563 latent voxels =
**f ≈ 0.44 %** of the 129 024-voxel grid (the J4 YAML comment claims 0.1 %; I use the measured value). Then

| `tc` weight | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| TC share of gradient | 0.44 % | 2.2 % | 4.2 % | **8.1 %** |

Even at the top weight tested, **>91 % of the gradient is still pushing on non-tumour tissue**. And the measured
return on that 20× is +0.41 dB PSNR_ET for −0.10 dB whole-volume. Pushing further up this axis is not a
promising direction; the flatness of the dose-response is what a nearly-saturated lever looks like.

### H4 — The ControlNet gets no imaging content [PLAUSIBLE, and actionable]

Imaging latents (T1pre, T2, FLAIR) reach the **trunk** via `input_concat` (12 extra channels at `conv_in`); the
ControlNet branch sees masks alone. So the branch that is supposed to inject tumour-specific behaviour has no
access to the very features (T2/FLAIR signal, necrosis texture) that predict which part of the core will enhance.
Whether this matters is testable, not established.

### H5 — 🔴 Checkpoint selection uses a signal that is not synthesis quality [STRONG — quantified below]

> **⚠ CORRECTED 2026-07-29.** An earlier version of this section claimed `ema_best` was selected on
> **background-region latent MSE**, citing these three config keys:
>
> ```
> best_metric_name: mse_latent
> best_metric_region: bg
> best_metric_nfe: 5
> ```
>
> **That attribution was wrong.** The keys are *vestigial*.
> `src/vena/model/fm/lightning/callbacks/checkpointing.py:58` would assemble them into the monitor key
> `val/mse_latent_bg_nfe5`, but (a) `routines/fm/train/engine.py:1464` **hard-codes**
> `ckpt_monitor = "train/total_epoch"` and passes it explicitly as `monitor_key=` to both
> `VENACheckpointCallback` (L1466) and `BestCheckpointCallback` (L1475), overriding those defaults; and
> (b) every run sets `validation.every_epochs: 0` — in-process validation is offloaded — so **no `val/*` metric is
> ever logged** and that key could never have resolved. Editing the three fields changes nothing.
>
> The tell that exposed it: reading `engine.py` to write the retrain recipe, not reading the YAML.

**What is actually true.** `ema_best.ckpt` is selected on **`train/total_epoch`** — the epoch-aggregated *training
loss* — exactly as `model-coding-standards.md` rule 5 documents ("train loss ≠ synthesis quality"). The same
monitor drives `EarlyStopping` (`engine.py:1526`) with `patience: 250`, which is why runs continued ~1000 epochs
past their validation peak: training loss kept creeping down.

Retention compounds it: `checkpoint_every_epochs: 25` with `output.retention_n_checkpoints: 3` means ~76
checkpoints are written per run and 3 survive.

The consequence is unchanged and still applies to every downstream consumer of `ema_best.ckpt`, including the
warm starts in this very matrix. My §3 numbers do **not** depend on it — I selected the best epoch by PSNR_ET
from the cadence CSVs directly. **A config knob that looks decisive and is inert is a defect in its own right:**
either delete `best_metric_{name,region,nfe}` or wire them up and fail loudly when the metric is absent.

**Quantified.** The epochs the selector actually retained, against each arm's PSNR_ET-optimal epoch:

| arm | PSNR_ET-best epoch | retained checkpoint epochs | PSNR_ET forfeited |
|---|---|---|---|
| J1 | 1450 → 20.251 dB | 1350 / 1400 / 1475 | **−0.118 dB** |
| J2 | 1050 → 20.562 dB | 1775 / 1800 / 1850 | **−0.164 dB** |
| J3 | 1025 → 20.524 dB | 1775 / 1800 / 1850 | **−0.198 dB** |
| J4 | 750 → 20.764 dB | 1775 / 1800 / 1850 | **−0.279 dB** |

The selector keeps epochs 700–1100 *after* the PSNR_ET peak, on the declining part of the curve. The forfeited
0.12–0.28 dB is **the same order as the entire benefit of raising the tumour weight 20×** (+0.41 dB). We are
losing to a config line roughly what the whole region-weighting axis buys.

### H6 — Latent resolution floor [HYPOTHESIS ONLY — not established]

ET is a median 431 latent voxels and 13/116 patients have one smaller than a 4×4×4 block (§5). The gain is
monotone across ET-size tertiles in 3 of 4 arms, but Spearman p = 0.09–0.33. Do not state this as a finding until
the §8 experiment is run.

---

## 7. The latent-metric columns — do not read them as model quality

`latent_cosine ≈ 0.19` and `latent_mse ≈ 1.534` are frozen across 1900 epochs (J1: 1.53495 → 1.53425; J3/J4 get
slightly *worse*) while image PSNR_ET gains 1.5–2.1 dB. Operands, confirmed in
`routines/fm/exhaustive_val/engine.py:718,747,770-773`:

- `z_target = batch["z_t1c"]` — the MAISI-encoded ground-truth T1c latent for that patient.
- `z_pred` — a **stochastic flow-matching sample** initialised from `x0 = torch.randn_like(z_target)`
  (`engine.py:811`) and integrated for `nfe` steps.
- `mask = torch.ones_like(z_target, dtype=torch.bool)` — despite the "masked" docstring, the mask is all-ones.
- Cosine is one scalar per patient over all 4 channels flattened together (`metrics/latent.py:62-68`).

So the column measures the distance between one stochastic draw and one specific encoder code. It is not a
reconstruction error and there is no reason for it to fall with training. `MSE = 2(1−ρ)` gives 1.62 against the
observed 1.53, so the numbers are internally coherent — not index-mismatched or NaN-corrupted.

**These columns are not evidence that the flow model fails to fit the T1c distribution, and no claim in this
document rests on them.** But ρ ≈ 0.19 is also *not yet shown to be normal*: the audit established what the
operands are, not what a healthy value looks like. Before anyone cites these columns in either direction, measure
the two reference points — cosine between two **different** patients' T1c latents (the floor) and cosine between
`z_t1c` and the same patient's `z_t1pre` (a same-anatomy upper anchor). Both are free: they need only the cached
latent H5, no sampling. Until then, treat the column as uninterpretable rather than reassuring.

---

## 8. The measurement that does not exist

There is **no VAE round-trip ceiling restricted to the enhancing region** anywhere in the project — not in
`artifacts/`, not on Picasso, not in the article tables. `routines/preflights/maisi_vae/` was specified but never
run. `table1_ring_a_fidelity.csv` has `{mae,psnr,ssim,ms_ssim}_{brain,wt,bg_undilated}` — no ET column, no
VAE-oracle row. The "VAE tax" has been an inference from a tier pattern since 2026-07-17 and is still flagged open
in `HANDOFF.md` §10.

This is the single cheapest experiment that would discriminate between the leading hypotheses, and it needs no
training: encode the **real** T1c through the frozen MAISI VAE, decode it, and score PSNR/MAE/SSIM and `p99.5`
**inside the ET mask**. If the VAE's own reconstruction cannot render the enhancing rim at full brightness, no
conditioning scheme downstream of it ever will, and the whole latent-space approach needs revisiting. Roughly
30 min on one A100 for a 50-volume split.

---

## 8b. Post-verdict revision (2026-07-29, after user feedback)

Three corrections and additions to the analysis above, all from checks made after the first pass.

**The VAE ET ceiling is NOT the problem.** The user measured it: MAISI round-trip fidelity inside ET is high.
H6 (latent-resolution floor) and the §8 experiment are therefore **closed**. Delete them from the plan.

**H5 — my own attribution was wrong; see the corrected §H5 above.** I claimed `ema_best` was selected on
background-region latent MSE. It is selected on **`train/total_epoch`** (training loss); the `best_metric_*` keys
are inert. The measured 0.12–0.28 dB forfeit stands, its cause does not.

*(The sub-point about the `bg` region itself remains factually true and is worth keeping for whoever next reads
those config keys: `MAE_bg = 0.0146` vs `MAE_whole = 0.0754`, `MAE_bnwt = 0.0747`, `MAE_et = 0.0925` — `bg` error
is 5× smaller than any brain region because it is skull-stripped air. So `bg` would have been a poor selector had
it ever been live. It was not. The user's instinct that a good refinement anchor should be unbiased whole-image
fidelity is right; that anchor is `whole` or `bnwt`, and neither is currently wired to checkpoint selection.)*

**H2 sharpened — localisation was effectively perfect, which strengthens the conclusion.** ET/TC by volume is
**median 0.885** (q25 0.776, q75 0.983; median ET 27 584 voxels vs NETC 3 936). The soft TC mask was in practice a
near-perfect ET locator, so "we only gave it TC, not ET" is not available as an explanation. The model had the
location and still did not brighten. *(Corollary, useful for the segmenter question: targeting TC instead of ET
costs almost nothing in localisation.)*

### The real binding constraint: the L1 minimiser is the conditional median

Region weighting did exactly what it was designed to do — it fit its target more accurately (MAE_ET −22 %,
0.119 → 0.0925). The target is the problem. The minimiser of an L1 velocity loss is the **conditional median**
velocity given `(x_t, t, T1pre, T2, FLAIR, mask)`. Enhancement brightness has the highest conditional variance of
any attribute given non-contrast input — BBB breakdown is not visible in T1pre/T2/FLAIR — so its conditional
median is dim. Raising the weight to 100, or moving to full image resolution, converges to the same dim median
faster. This is Blau & Michaeli's perception–distortion tradeoff (CVPR 2018, arXiv:1711.06077): reducing mean
distortion necessarily increases the distributional gap.

**This also explains the LPL null without reopening the STOP decision.** Perceptual losses — VGG-style or
decoder-feature — are *engineered to be invariant to global intensity and contrast shifts*; they respond to
texture and edges. LPL was a tool built to ignore brightness applied to a brightness deficit. The confound the
user flagged (warm-started from the discarded ET-conditioned v3b) is second-order to that.

### Prior art — verified, and a positioning threat

**Brandstötter & Kobler, "Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using
Posterior-Mean Rectified Flow", arXiv:2508.12640 (18 Aug 2025)** — verified via arXiv. Diagnoses *our exact
failure mode* (MSE-trained rectified flow converges to the posterior mean → structurally right, texturally and
amplitude-flat) on *our exact task* (T1c synthesis, BraTS 2023-2025, 360 held-out volumes). Remedy: a two-stage
PMRF — patch-based 3D U-Net for the posterior mean, then a **time-conditioned 3D rectified flow seeded from that
estimate** rather than from noise. Reports axial FID 12.46 (−68.7 % vs posterior mean) at **+27 % volumetric MSE**
— the tradeoff made explicit. Note it is **image-space**, not latent. VENA must cite and position against it.

**Ruffle et al., arXiv:2508.16650 (Aug 2025, rev. June 2026)** — verified. 11 089 studies, 10 datasets, 4
countries. nnU-Net predicts enhancement from non-contrast T1/T2/FLAIR at **83.0 % balanced accuracy** (beats
blinded neuroradiologists at 71.7 %), **enhancement volume R² = 0.859**, Dice ≥ 0.7 in 50.2 % of enhancing cases.
Bears directly on the information-limit question: enhancement *location and volume* are substantially predictable
from non-contrast; the paper says nothing about *intensity*. That is exactly the gap VENA sits in.

### ⚠ A citation the literature agent inverted — do not repeat it

The agent's top recommendation was to apply CFG at α ∈ [0.5, 1.0] (high noise), citing Kynkäänniemi et al. 2024
(NeurIPS, arXiv:2404.07724). That paper's finding is the **opposite**: guidance at high noise *distorts the prior
and hurts*; it is beneficial only in a **middle** interval. Reconcile with Rissanen et al. (ICLR 2023,
arXiv:2206.13397 — low-frequency amplitude is fixed at high noise) as follows: fix the high-noise regime on the
**training** side (timestep curriculum / Min-SNR-γ, arXiv:2303.09556), and use guidance on the **middle**
interval. Do not guide at high noise.

---

## 9. Verdict

**(a) Is ControlNet injection sufficient to place enhancement given a perfect mask? — NO, but it is not the
failure it looks like.** With a ground-truth mask the model gains **+1.5 to +2.1 dB PSNR_ET** (paired, p < 1e-8,
+1.8 to +2.6 dB excluding LUMIERE) and closes the ET-vs-healthy error gap from 1.59× to 1.24×. That is a real,
reproducible localisation gain. What it does **not** do is change the intensity of the synthesised enhancement:
`p99.5` moves by −0.03 %, and the shape of the ET error distribution is bit-for-bit unmoved across 1900 epochs.
**The mask answers "where"; nothing in the S1 objective answers "how bright".**

**(b) Which recipe to carry forward.** ⚠ **This answer was REVERSED on 2026-07-29 — see §10.** The original
reading ("J4 is best on every ET metric, carry it forward") ranked the arms on *distortion* metrics only. Under
the perception–distortion tradeoff the arm that maximises PSNR_ET is the arm that most regresses to the dim
conditional median. Evidence from VENA's own sibling arms now suggests the opposite ranking. **Do not carry J4
forward until §10's test is run.**

**(c) GO / NO-GO for the segmenter phase.** The oracle number is the ceiling a predicted mask can aspire to. That
ceiling is **+1.5 to +2.1 dB PSNR_ET with zero change in enhancement intensity**. A predicted mask will land below
it. Investing a full segmenter phase to chase a fraction of a ceiling this low is not justified *on this
objective*. The blocker is not mask quality — it is that the training signal cannot express enhancement.
**Recommendation: NO-GO on the segmenter phase as currently motivated**; the segmenter is worth building only
after an objective exists that the mask can actually leverage.

### Ordered next steps

1. **Measure the VAE ET ceiling** (§8). Cheap, no training, and it discriminates H1/H2 from H6. Nothing else
   should be launched before this number exists. If the VAE cannot round-trip the rim, that is the paper's
   central finding and it changes the architecture, not the loss.
2. **Fix checkpoint selection** — it monitors `train/total_epoch`, not any region metric (the `best_metric_*`
   keys are inert; see corrected §H5). Recovering the 0.12–0.28 dB needs either a val-driven monitor or, more
   simply, raising `output.retention_n_checkpoints` from `3` so the every-25-epoch series survives and the choice
   can be made afterwards. Full recipe in [`v3a_retraining.md`](v3a_retraining.md).
3. **Add a hard guard on `load_warm_start`**: raise when `loaded == 0`, and warn when `loaded / len(src_state)`
   is below some threshold. The J0 failure mode is silent, structural, and will recur.
4. **Give the objective an intensity term.** H1 is the strongest hypothesis and the only one with a clear remedy:
   a decode-and-compare term in image space inside the ET region (an ET-restricted LPL, or a direct intensity /
   percentile-matching penalty). Note the prior LPL programme was stopped as optimisation-inert — an
   ET-restricted intensity loss is a different object and should be argued as such, not assumed.
5. **Regenerate `results/article/paired_fidelity/`** — the current LATEST (2026-07-20) predates the 99.95
   normalisation fix that specifically un-clips the enhancing rim. No ET-region claim should leave the project
   from that artifact until it is rebuilt.
6. **Re-run J0 correctly** if the freeze→joint gain is still wanted, with the warm start actually loading.
7. Only then reconsider the segmenter phase.

### Proposed refinement objective — "where + how bright + usable gradient"

Ordered by cost. **Step 0 is free and makes everything after it non-speculative.**

**0. Measure where brightness is actually decided (free, no training).** For held-out patients, bin α, and at each
bin compute (i) the velocity error inside TC and (ii) the ET intensity of the one-step estimate `x̂_1 = x_t + α·v`
(the S3 machinery already computes this; the timestep convention is pinned in `CLAUDE.md`). This produces an
α-vs-rim-amplitude curve that decides between "the high-noise regime is undertrained" (Rissanen) and "the
minimiser is dim" (Blau & Michaeli). Note the current setup supervises *against* the high-noise regime from two
directions: SD3 timestep transform concentrates at α ∈ [0.3, 0.7], and the LPL gate was `t_dn > 0.4` ⇒ α < 0.6.
Do not design the next loss before this curve exists.

**1. CFG on the mask channel — cheapest real intervention.** Train with `conditioning_dropout_p ≈ 0.1` on the mask
keys; sweep guidance weight `w` at inference. The infrastructure is already in the configs
(`conditioning_dropout_p: 0.0`, `conditioning_dropout_keys: [wt]`, `decision.json` 0.7.0 carries both fields), so
this is one training run plus a **free** inference sweep. It is the literal implementation of "make the model pay
more attention to the mask": guidance amplifies `v_cond − v_uncond`, pushing samples away from the unconditional
mean. Its canonical *artifact* is over-saturation — the exact inverse of VENA's measured deficit. Restrict
guidance to a middle α interval (Kynkäänniemi 2024), **not** high noise. If flow-specific first-step undershoot
appears, CFG-Zero* (arXiv:2503.18886) fixes it without retraining. The `w`-sweep yields a p99.5 dose-response
curve, which is a publishable figure whichever way it goes.

**2. Change the minimiser, not the weight.** Two options, not mutually exclusive:
   - **(a) PMRF-style second stage** (arXiv:2508.12640). Treat the current model as the posterior-mean stage — it
     is already location-correct — and add a refinement flow **seeded from its output instead of from noise**.
     Architecturally cheap, directly targets amplitude, and has a published result on this exact task. Expect the
     documented tradeoff: better realism, *worse* MSE. Decide in advance that this is acceptable and pre-register
     which metric is primary, or the result will be unreadable.
   - **(b) A distributional intensity term inside the soft mask.** Decode `x̂_1` and penalise a *statistic* of the
     intensity distribution inside the soft TC mask — top-percentile match, or W₁ between predicted and real
     intensity histograms. Two reasons this behaves better than the per-voxel route: its minimiser is not the
     conditional median, and a few scalars per volume are not diluted by the 0.44 % voxel share that caps §H3.
     Uses the soft mask as weights, so it respects the "cover TC, don't overfit ET" design intent. **Explicitly
     not a perceptual loss** — that is why it is not a repeat of LPL.

**3. Reframe the claim.** Ruffle et al. show location and volume are largely predictable from non-contrast;
nobody has shown intensity is. Moya-Sáez et al. (Front. Neuroimaging 2023, DOI:10.3389/fnimg.2023.1055463) record
that the field has no consensus on whether under-enhancement is model failure or an information limit. **The
oracle-mask matrix is precisely the experiment that separates them, and it is already run.** "Even with a perfect
tumour mask, enhancement intensity is not recoverable from non-contrast input" is a stronger and more defensible
contribution than "our conditioning helps a bit" — and it is what the data actually supports.

### Numbers the next session must not re-derive

- Trustworthy v3a baseline: **26.879 dB** whole / **18.76 dB** PSNR_ET (NFE=20, 116 patients). *Not* 25.145/16.539.
- Oracle-mask effect: **+1.5 to +2.1 dB PSNR_ET**, whole-volume flat (+0.16…+0.24 dB), saturated by epoch ~1000.
- Region weighting 1→20: **+0.41 dB PSNR_ET, −0.10 dB whole-volume.** Axis exhausted.
- ET footprint: median **431 latent voxels** = 0.33 % of the 129 024 grid; TC 563 = 0.44 %.
- TC share of gradient at `tc=20`: **≈8 %** (not the 0.1 %/2 % implied by the J4 YAML comment).
- Error-concentration inside ET `MSE/MAE²`: **1.512 → 1.518** over 1900 epochs. Unmoved.
- `p99.5` with vs without oracle mask: **0.78582 vs 0.78558** (−0.03 %).

---

## 10. 🔴 VERDICT REVERSAL (2026-07-29) — region weighting is the harm, not the fix

Prompted by the user's observation that the ET-mask-conditioned siblings of v3a "lead the charts on tumour
SSIM". I verified it, and the within-family comparison it exposes overturns §9(b).

### The two sibling runs

| run | conditioning (`conditioning_inputs`) | region weighting |
|---|---|---|
| `2026-06-22_15-22-04_s1_v3b_concat_plus_cn3ch_fft_c698f45a` | `mask:{netc,ed,et}:identity` (hard, 3 ch, **incl. ED + explicit ET**) | **none** (plain mean L1) |
| `2026-06-22_15-20-57_s1_v3b_rw_concat_plus_cn3ch_fft_320b5ddd` | same | `netc: 50, ed: 50, …` |

Note both differ from the J-arms, which used **soft** masks, 2 channels (`tc_soft`, `netc_soft`), **no edema
channel**, and weights 1–20.

### The user's claim — verified, with the necessary caveat

Ranked by `ssim_wt_mean` (Ring A, N=247), the top three are **v3b-rw 0.5737 · LPL-b2c 0.5731 · v3b 0.5700**, ahead
of C5-T1C-RFlow (0.4646) and C2-ResViT (0.4475). **But all three leaders carry `is_oracle = True`** and the
non-oracle v3a sits at 0.4081. The +0.16 SSIM_wt over v3a is the GT mask, not the objective — consistent with
`[[project_vena_oracle_mask_finding]]`. **The lead over competitors is still not claimable.**

### The comparison that matters is *within* the oracle family

| metric | v3b (no RW) | v3b-rw (RW=50) | winner |
|---|---|---|---|
| `ms_ssim_wt` | **0.8161** | 0.7559 | v3b **+0.060** |
| `ssim_brain` | **0.5893** | 0.5537 | v3b +0.036 |
| `mae_brain` | **0.0915** | 0.0955 | v3b −0.004 |
| **`raw_p995`** (intensity tail) | **0.8421** | 0.7856 | v3b **+0.057** |
| `mae_wt` | 0.0965 | **0.0948** | v3b-rw −0.002 |
| `ssim_wt` | 0.5700 | **0.5737** | v3b-rw +0.004 |

**Region weighting improved only the median-type metrics (MAE_wt, marginally SSIM_wt) and degraded everything
perceptual and the entire intensity tail.** v3b-rw's `raw_p995` (0.7856) is indistinguishable from *unconditioned*
v3a (0.7858) — i.e. weighting erased the whole brightness gain that conditioning alone had produced.

This is the perception–distortion tradeoff (Blau & Michaeli 2018) measured **inside VENA's own arms**, on VENA's
own data. It is a stronger form of the §8b argument than the theory alone.

### Consequences

1. **The user's hypothesis is substantially right.** L1-on-velocity is *not* the blocker; it produced the best
   intensity tail in the project (v3b, 0.8421) when left alone. **Region weighting is the blocker.**
2. **§9(b) is reversed.** The J-arms were ranked on PSNR_ET / MAE_ET / SSIM_ET — all distortion metrics — so the
   ranking J1 < J2 < J3 < J4 is a ranking of *how well each arm fits the dim median*. J1 (`tc = 1`) is
   numerically identical to plain mean-L1 and is the arm most comparable to v3b. **J4 is the likely worst arm on
   brightness, not the best.**
3. **The whole S3 matrix optimised the wrong axis.** Every arm carried `reduction: none` + `region_weights`
   by design ("one loss code path for every arm" — S2 note). There was no true no-RW arm except J1-by-numerical-
   coincidence.

### The decisive test — cheap, no training

Decode one cadence epoch per J-arm from the existing `exhaustive_val/epoch_NNN/latent_preds.h5` and compute
`raw_p995` plus ET-region intensity statistics. **Falsifiable prediction: `raw_p995` decreases monotonically with
`tc` weight (J1 > J2 > J3 > J4)**, mirroring v3b > v3b-rw. If confirmed:

- the recommendation becomes **drop region weighting**, and
- VENA gains a clean two-family mechanistic result (v3b/v3b-rw *and* J1→J4) that explains a null result via an
  established principle — a far better paper than "our conditioning helps a bit".

If refuted, the tradeoff is not the operative mechanism here and §8b needs rewriting.

**Do not launch any new training run before this test.** It uses data already on disk.

### Also settled in this pass

- **Do NOT relaunch v3a for checkpoint selection alone.** The blocker the user raised (defining brain voxels in
  latent space) does not exist: `masks/brain_latent` is already cached and already in use — every `metrics.csv`
  carries `brain_mask_source = masks/brain_latent`, so no new derivation is needed. But v3a retained only
  `ema_best`, `ema_epoch_{1649,1799,1849}`, `last` (`retention_n_checkpoints: 3` against
  `checkpoint_every_epochs: 25`), so the epochs worth re-selecting (cadence peak 525, PSNR_ET peak 900)
  **no longer exist on disk** — retrospective reselection is impossible, and a 4-day rerun buys ~0.19 dB in a
  confounded normalisation. v3a is the warm-start anchor for every S3 arm; replacing it invalidates the matrix.
  **Fix forward**, folded into the next training launch — the full recipe with evidence is in
  [`v3a_retraining.md`](v3a_retraining.md).
- **PMRF is correctly judged out of scope.** The user's reading is accurate: PMRF trains a *second* network whose
  flow starts at the posterior-mean estimate rather than at noise, whereas VENA continues training *one* network
  on the noise→data map with added conditioning. Two networks at inference is a real cost, and if §10's test
  confirms that removing region weighting recovers the intensity tail, PMRF is not needed. Keep it as the
  fallback if the cheap fix fails, and cite it regardless (arXiv:2508.12640) — it is prior art on this exact task.

---

## 11. If v3a were relaunched: the base-model recipe

Asked 2026-07-29. Ordered by value, each with the evidence that motivates it.

### 11.1 Retain periodic checkpoints — the single change that would have prevented this whole situation

v3a retained `ema_best`, `ema_epoch_{1649,1799,1849}`, `last` — five files, all late. Its whole-volume peak was
**epoch 525** and its PSNR_ET peak **epoch 900**. Both are **gone from disk forever**, so no retrospective
reselection is possible no matter what metric we later decide was right.

Keep **every Nth checkpoint (N ≈ 100) for the whole run**, in addition to top-k. 19 checkpoints × 2.75 GB ≈ 52 GB
— against ~4 A100-days per run. The storage is free by comparison, and it converts "we picked the wrong selection
metric" from a lost run into a `ls` command.

### 11.2 Fix the selection metric — but do not rely on it

`best_metric_region: bg` selects on the near-empty region outside the brain (`MAE_bg = 0.0146` vs
`MAE_bnwt = 0.0747`). Change to `whole`. But 11.1 matters more: the right metric was not knowable in advance, and
it still isn't — on the perception–distortion reading, PSNR-type selection actively prefers the dimmer model.

### 11.3 Train ~1000 epochs, not 1900

| run | epochs run | whole-PSNR peak | PSNR_ET peak | % of run after the ET peak |
|---|---|---|---|---|
| v3a | 1900 | 525 | 900 | **53 %** |
| J2 | 1900 | 875 | 1050 | 45 % |
| J3 | 1900 | 1175 | 1025 | 46 % |
| J4 | 1900 | 1000 | 750 | **61 %** |

Every run spent roughly half its budget past its own peak, and ended 0.09–0.38 dB PSNR_ET *below* it. Halving the
schedule costs nothing measurable and frees ~2 A100-days per run.

### 11.4 Add conditioning dropout from the start — this one cannot be retrofitted

`conditioning_dropout_p: 0.0` in every run to date. Training with `p ≈ 0.1` costs nothing and is what makes
classifier-free guidance available at inference — a *free*, sweepable knob on how strongly the model attends to
its conditioning, whose canonical artifact (over-saturation) is the inverse of VENA's measured deficit. Adding it
later means retraining the base, which is precisely the position we are in now. Applies to the modality concat
channels for a `concat_only` base, and to mask channels for any conditioned successor.

### 11.5 Log intensity statistics in the cadence CSV from day one

The reason §10's question needed a separate GPU job is that `metrics.csv` carries PSNR/SSIM/MAE per region but
**no intensity statistic**. Add `p995_pred_brain`, `p995_real_brain`, `mean_et_{pred,real}`,
`mean_bnwt_{pred,real}` to exhaustive-val. The decoded volume is already in memory at that point, so the marginal
cost is ~zero, and every future run answers the perception–distortion question from its own CSV. Instrument the
quantity you will want to ask about, before you need it.

### 11.6 No region weighting in the base

v3a has none (plain mean L1) — keep it. Whether to add it downstream is what §10 tests.

### 11.7 Pipeline hygiene, not recipe

- Hard guard in `load_warm_start`: raise when `loaded == 0` (see §1).
- Confirm exhaustive-val runs at `ENCODER_PERCENTILE_UPPER = 99.95` — fixed in `01350e0`, automatic for any new
  run, but assert it rather than assume.

### 11.8 Left open deliberately

The timestep distribution (`use_timestep_transform`) should not be changed until the §9-step-0 α-binned
diagnostic says where rim amplitude is actually set. Do not tune it on theory.

### §10 addendum — pre-registered decision rule (written BEFORE the data existed)

Job 1679524 was still RUNNING when this was written. The analysis script
(`scratchpad/s3_intensity_agg.py`) was validated against two synthetic controls with the production CSV header:

| control | construction | group-mean span | paired Wilcoxon Δ vs J1 |
|---|---|---|---|
| **positive** | planted monotone −0.004/arm | 13.3 % of mean | −0.004, **p ≈ 1e-26** |
| **null** | identical means, per-patient noise only | **3.2 – 8.0 % of mean** | −0.004, **p = 0.18 – 0.87** |

**The group-mean span cannot discriminate**: pure noise produced up to 8 % against a planted effect's 13 %, and
the two controls produced *identical* delta magnitudes (±0.004). With only four arms, "monotone decreasing" also
occurs by chance with probability 1/24 ≈ 4 %. Neither the span nor the monotonicity boolean is evidence.

**Decision rule, fixed in advance:**

1. The **paired Wilcoxon on per-patient deltas vs J1** is decisive. It cancels per-patient variance, which is what
   separated the two controls by 25 orders of magnitude.
2. §10's prediction is **confirmed** only if `contrast_pred` deltas vs J1 are negative for J2, J3 and J4 *and*
   significant after multiplicity correction across the three comparisons.
3. If the deltas are non-significant, the result is **REFUTED** and reported as such: the perception–distortion
   tradeoff is not the operative mechanism across the J-family, the v3b/v3b-rw difference came from something
   else (candidates: the explicit ED conditioning channel, hard vs soft masks, or weight 50 vs 20), and §8b is
   rewritten accordingly.
4. A monotone-looking mean ordering with non-significant paired tests is reported as **null**, not as a trend.

The user's visual observation (J1 and J4 indistinguishable on the same patient) predicts outcome 3.

---

## 12. 🔴 §10 REFUTED BY DIRECT MEASUREMENT — region weighting *increases* enhancement contrast

Job **1679524**, COMPLETED, artifact
`picasso:~/execs/vena/analyses/s3_intensity/2026-07-29T14-14-41Z/intensity_stats.csv`
(1160 rows = 5 labels × 2 NFE × 116 patients). Decoded from each arm's `latent_preds.h5` with the frozen MAISI
VAE; ET mask byte-identical to the `psnr_db_et` definition; real T1c normalised at `ENCODER_PERCENTILE_UPPER`.

**All five pre-registered gates PASS**, including the decisive one: `p995_real_brain` and `mean_et_real` have
**0.000e+00 spread across arms** (the real image does not depend on which arm predicted it). Epochs used are
exactly 0 / 1450 / 1050 / 1025 / 750 — no silent fallback. The 60 non-finite `mean_et_pred` are exactly the 6
no-ET patients × 5 labels × 2 NFE.

### The result (NFE = 20; NFE = 5 agrees)

| arm | mean_ET pred − real | contrast (ratio of means) | contrast (median per-patient ratio) |
|---|---|---|---|
| baseline v3a | −0.0335 | 0.548 | 0.784 |
| J1 `tc=1` | −0.0161 | 0.723 | 0.915 |
| J2 `tc=5` | −0.0061 | 0.771 | 0.922 |
| J3 `tc=10` | −0.0044 | 0.795 | 0.939 |
| **J4 `tc=20`** | **−0.0004** | **0.825** | **0.953** |

Paired Wilcoxon on `contrast_pred` vs J1 (the pre-registered decisive statistic), n = 110:
**J2 +0.0053 (p = 0.027) · J3 +0.0081 (p = 0.0032) · J4 +0.0115 (p = 0.0025)** — all **positive**, all significant,
monotone. Against the v3a baseline every arm is far ahead: J1 +0.0197 (p = 8.4e-05) … J4 +0.0311 (p = 1.5e-04).

**The prediction in §10 was that contrast falls monotonically J1 → J4. It rises monotonically.** Refuted, and in
reverse.

### Consequences — two of my own conclusions overturned

1. **§9(b) was right the first time; the §10 reversal was wrong.** J4 (`tc = 20`) *is* the arm to carry forward.
   I reversed a correct conclusion on the strength of a **cross-family inference** (v3b vs v3b-rw) instead of a
   direct measurement of the family in question. The tell I should have heeded is the user's: J1 and J4 looked
   alike, which is consistent with a small effect — not with a large reversed one.
2. **Why the v3b pair misled.** `v3b_rw` up-weights `netc: 50, ed: 50` — **edema**, a large dim region — whereas
   the J-arms up-weight the **tumour core** only. Up-weighting a large dim region to fit its median more
   accurately is a different intervention from up-weighting a small bright one, and plausibly does compress
   contrast. Additionally `tableS1`'s `raw_p995` is a **raw-space, whole-volume** statistic from a pre-`01350e0`
   artifact — not the brain-box ET contrast measured here. The two were never measuring the same thing.

### What the numbers actually say about the project's central question

- **Mask conditioning recovers most of the enhancement.** Contrast goes from **54.8 % → 82.5 %** of the real
  value, and J4's mean ET intensity lands **within 0.0004** of ground truth — effectively exact.
- **The residual gap is NOT in the tumour.** J4 renders the enhancing region at the right absolute intensity; the
  shortfall is that **healthy tissue is rendered ~0.0155 too bright** (`mean_bnwt` 0.3314 predicted vs 0.3159
  real), which compresses the ET-minus-background contrast. The remaining ~17 % is a **background-brightness
  calibration** problem, not an enhancement-synthesis one.
- **There is no global under-saturation here.** `p995_pred_brain` (0.820) is *above* `p995_real_brain` (0.788).
  The earlier "VENA under-saturates" reading came from `tableS1`'s raw whole-volume percentile in a pre-fix
  artifact and does not survive this measurement.

### Revised guidance

- **Carry J4 (`tc = 20`, joint trunk) forward.** Restore §9(b)'s original recommendation.
- **The region-weight axis is not exhausted after all** — contrast is still rising at `tc = 20`. Whether `tc = 50`
  or `tc = 100` continues to help is now an open, cheap question, and PSNR_ET would have hidden it entirely.
- **Retarget the remaining effort at healthy-tissue intensity calibration**, not at the rim.
- **§8b's perception–distortion framing is not refuted, but it is not the operative mechanism across the J-family
  either.** Region weighting on a small bright region improved *both* the distortion metric (PSNR_ET) and the
  perceptual/intensity one (contrast) — no tradeoff was visible. Downgrade H1 from [STRONG] to [PLAUSIBLE,
  contradicted for tumour-core weighting].
- **Do not cite `tableS1_undersaturation.csv` for any ET claim** until `paired_fidelity` is regenerated post-fix.

### Method note worth keeping

The two synthetic controls run before the data landed were what made this readable: the null control showed a
group-mean span of up to **8 %** with *no* true effect, against the observed **13 %** — so the span alone would
not have settled it. The paired test did, exactly as pre-registered. Keep that pattern.
