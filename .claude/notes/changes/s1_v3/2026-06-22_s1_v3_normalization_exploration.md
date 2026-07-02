# S1 v3 — Data-Layer Spec: Normalization Audit and Cohort Re-Encode

*Mario Pascual González — VENA, IBIMA-BIONAND.*
*2026-06-22 — child of `.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md` (read **§3.3, §3.4, §4 H4** of that document for context). This spec covers step (3) of the v3 plan; the architectural changes are in the sibling spec `.claude/notes/changes/2026-06-22_s1_v3_model_implementation.md`.*

---

## 0. Goal in one sentence

Select a per-modality intensity normalisation that (a) preserves the T1c
hyperintensity (Gd-enhancement) tail without crushing it against the rest
of the brain, (b) keeps the MAISI-V2 VAE encode → decode round-trip
in-distribution (MAE_whole ≤ 0.01 in normalised space), and then re-encode
all 9 cohort image-H5s + their offline-augmented siblings into latents
that the v3 trainer can consume.

The current normalisation
`percentile_normalise(lo=0, hi=99.5, foreground_only=True, clip=True)`
flattens the enhancement signal (father doc §3.3: ⟨|T1c − T1pre|⟩ is
**identical** in WT and non-WT after normalisation, 0.3837 = 0.3837). We
need a variant that breaks this symmetry while keeping the VAE happy.

---

## 1. Hypothesis space — normalisation variants to test

All variants operate on the **skull-stripped** brain volume (background
voxels stay at 0). All produce a `float32` tensor in approximately `[0, 1]`
(with a small over-shoot for the no-clip variants). All compute on
**foreground voxels only** unless explicitly noted.

| ID | Spec | Rationale |
|---|---|---|
| **V0** | **`percentile_normalise(0, 99.5, fg=True, clip=True)`** (current) | Baseline. Reference for comparisons. |
| **V1** | `percentile_normalise(0, 99.5, fg=True, clip=False)` | Lets enhancement keep its super-percentile values (> 1.0). Tests whether clip is the lossy step. |
| **V2** | `percentile_normalise(0, 99.9, fg=True, clip=True)` | Moves the cut to the top 0.1 % — preserves all but the very brightest voxels. |
| **V3** | `percentile_normalise(0, 99.99, fg=True, clip=True)` | Top 0.01 % only — preserves essentially the entire enhancement distribution. |
| **V4** | **Joint-modality percentile**: compute (0, 99.5) over the **union** of (T1pre, T1c, T2, FLAIR) per patient; use the same `(lo, hi)` to scale all four. | Preserves **inter-modality intensity correspondence**. The T1c enhancement remains *brighter than* T1pre at the same anatomical voxel in the normalised space. Cost: T2/FLAIR end up in a narrower range. |
| **V5** | **Per-modality z-score on brain stats** + softclip to `[0, 1]` via `0.5 * (1 + tanh((x − μ) / (2σ)))` | Information-preserving (`tanh` is monotonic everywhere) but bounded. Keeps the bright tail relative-position correct. |
| **V6** | **WhiteStripe-style** (Shinohara et al. 2014, *NeuroImage*) — normalise by mean of *normal-appearing white matter* via T1pre histogram peak; apply same affine to T1c. | The clinical-radiology standard. Preserves the absolute T1c-vs-T1pre intensity contrast because the scale is determined by *non-enhancing* tissue. Highest fidelity to "enhancement vs not". Implementation cost: needs WM detection. |
| **V7** | `percentile_normalise(0, 99.5, fg=False, clip=True)` — **whole-volume percentile** (background included, as T1C-RFlow does) + `minmax01` | Mirrors T1C-RFlow's normalisation literally. Tests whether VENA's foreground-only restriction is responsible for the squashing. |

**Sweep order**: V0 (baseline, no run) → V1 → V2 → V3 → V7 (cheap one-line
variants) → V4 → V5 (need new function) → V6 (most involved). Stop the
sweep when a variant clears the §3 acceptance criteria *and* is the
cheapest one that does.

---

## 2. Test protocol

### 2.1 Cohort and sample

- **UCSF-PDGM, fold 0 val** — 89 patients available; sample N = 30
  (seed 1337). UCSF is the headline cohort and is the only cohort
  whose VAE encode/decode quality has already been characterised
  (existing `encode_ucsf_pdgm_maisi/LATEST/`); we extend that
  characterisation per variant.
- Cohort restriction: UCSF-PDGM is chosen because its T1c images include
  the full enhancement-distribution range (grades 2 / 3 / 4 all present).
  The variant winner is then verified on a **smoke** sample of N = 5
  patients per *other* cohort (BraTS-GLI, REMBRANDT, LUMIERE, BraTS-PED)
  to catch cohort-specific failure modes before the full re-encode.

### 2.2 Per-variant measurements

For each patient × variant, encode → decode → compute the following.
Implement as a single routine `routines/preflights/normalization_audit/`
(per `.claude/rules/preflight-pattern.md`).

#### A. VAE round-trip quality (the OOD test)

| Metric | Whole brain | ET (label 4) | NETC (label 1) | ED (label 2) | BNWT (brain & ¬WT) | BG |
|---|---|---|---|---|---|---|
| MAE | required | **required (load-bearing)** | required | required | required | should ≈ 0 |
| MSE | required | required | required | required | required | should ≈ 0 |
| PSNR (dB, data_range = max of normalised range) | required | required | required | required | required | n/a |
| SSIM (whole-volume, with brain mask) | required | n/a | n/a | n/a | n/a | n/a |

These are computed on **T1c only** (the target modality). The other
modalities (T1pre, T2, FLAIR) are checked at whole-brain level only —
they don't drive the v3 recipe choice.

#### B. Intensity-signal preservation (the contrast-preservation test)

Per (patient, region ∈ {ET, NETC, ED, BNWT}):

- `mean_intensity_t1c_normalised`
- `mean_intensity_t1pre_normalised`
- `mean_diff = mean(T1c − T1pre)` (signed)
- `mean_abs_diff = mean(|T1c − T1pre|)`
- Aggregate: ratio `⟨|T1c − T1pre|⟩_ET / ⟨|T1c − T1pre|⟩_BNWT` per variant.

This is the metric that V0 fails on (0.62 ≈ ET signal *smaller* than
BNWT signal). The variant winner must invert this ratio.

#### C. Latent-target signal (the loss-signal test)

For each patient × variant, encode T1c and T1pre, then per region:

- `mean_abs_delta_latent = ⟨|z_t1c − z_t1pre|⟩`
- Ratio `⟨|Δ|⟩_ET / ⟨|Δ|⟩_BNWT` per variant.

A normalisation that preserves enhancement at the *latent* level is what
the model actually trains against. Image-space preservation is necessary
but not sufficient.

#### D. Distribution-shape audit (the OOD safety check)

Per variant × modality:

- Histogram of normalised intensities (256 bins, range `[0, 1.1]` to
  catch the no-clip variants' overshoot)
- 1st, 50th, 99th percentile of the foreground distribution
- KL divergence between the variant's foreground distribution and the
  V0 baseline (this is the "how OOD does this push the encoder?" metric)

If `KL(V_i || V0) > 1.0` nats, the variant is too far from the MAISI
pre-training distribution and is rejected regardless of A/B/C results.
The MAISI VAE is *not* re-trained for this project; we are only re-using
it.

### 2.3 Sample code skeleton (the audit routine)

`routines/preflights/normalization_audit/engine.py`:

```python
from vena.common import load_autoencoder
from vena.preflight.normalization_audit import (
    NormalizationVariant, run_round_trip, compute_per_region_stats,
    compute_intensity_signal, compute_latent_signal,
    compute_distribution_shape, render_recon_grid, render_intensity_histogram,
)

class NormalizationAuditEngine:
    def __init__(self, cfg: NormalizationAuditConfig): ...
    def run(self) -> Path:
        vae = load_autoencoder(self.cfg.vae_checkpoint)
        results = []
        for variant in self.cfg.variants:           # list of NormalizationVariant
            for pid in self.cfg.patient_ids:        # 30 UCSF-PDGM ids
                imgs = load_patient_images(pid)     # T1pre/T1c/T2/FLAIR + masks
                normed = variant.apply(imgs)         # dict of normalised float32
                z = vae.encode(normed)               # dict of latents
                recon = vae.decode(z)                # dict of decoded images
                results.append({
                    "variant": variant.id,
                    "patient_id": pid,
                    "round_trip": compute_per_region_stats(normed, recon, imgs["masks"]),
                    "intensity_signal": compute_intensity_signal(normed, imgs["masks"]),
                    "latent_signal": compute_latent_signal(z, imgs["masks"]),
                    "distribution": compute_distribution_shape(normed),
                })
        emit_report(results, self.cfg.out_dir)
        emit_decision_json(results, self.cfg.out_dir)
        return self.cfg.out_dir
```

`NormalizationVariant` is a frozen dataclass with `id: str`, `apply(imgs) -> dict`,
and `sha256: str` (hash of the function-defining code). Each variant V0..V7
is registered via `@register_variant("v3")` in
`src/vena/preflight/normalization_audit/variants.py`.

### 2.4 Implementation files (new)

- `src/vena/preflight/normalization_audit/__init__.py` (exports)
- `src/vena/preflight/normalization_audit/variants.py` (V0..V7 definitions + registry)
- `src/vena/preflight/normalization_audit/round_trip.py` (encode/decode/metric pipeline)
- `src/vena/preflight/normalization_audit/figures.py` (recon-grid, histogram renderers)
- `src/vena/preflight/normalization_audit/decision.py` (emit `decision.json` matching schema below)
- `routines/preflights/normalization_audit/engine.py`
- `routines/preflights/normalization_audit/cli.py`
- `routines/preflights/normalization_audit/configs/{default,smoke}.yaml`
- `tests/preflight/test_normalization_audit_variants.py` (unit tests for each variant; check shape, range, idempotence)

Console script: `vena-preflight-normalization-audit`.

---

## 3. Acceptance criteria (the variant winner)

A variant V_i is the **winner** iff it satisfies *all* of:

| # | Criterion | Threshold | Rationale |
|---|---|---|---|
| C1 | VAE MAE_whole ≤ 0.010 | absolute | Stays close to the V0 baseline (0.0041). Anything 2.5× worse is OOD. |
| C2 | VAE MAE_ET ≤ 0.015 | absolute | The most sensitive region; allows some increase but bounded. |
| C3 | KL(V_i || V0) ≤ 1.0 nats | per modality | Encoder stays in pre-training distribution. |
| C4 | Image-space ⟨\|T1c−T1pre\|⟩_ET / ⟨\|T1c−T1pre\|⟩_BNWT ≥ **1.5** | relative | Currently 0.62 — *must* exceed 1.0 (enhancement clearer than background) and ≥ 1.5 (substantial margin). |
| C5 | Latent ⟨\|Δ\|⟩_ET / ⟨\|Δ\|⟩_BNWT ≥ **1.3** | relative | Latent-space signal is what the model trains against. Current 1.10. |
| C6 | Smoke-cohort verification: C1–C5 satisfied on N=5 patients each for BraTS-GLI, REMBRANDT, LUMIERE, BraTS-PED | per cohort | Catches cohort-specific failure modes. |
| C7 | Round-trip PSNR_whole ≥ 35 dB | absolute | Stricter than C1; catches subtle encoder degradation. |

**Tie-breaking** (multiple variants pass C1–C7):

1. Prefer the variant with the **lowest** KL divergence (most in-distribution).
2. Then the variant with the **highest** C4 ratio (best enhancement preservation).
3. Then the **cheapest** implementation (clip flip < whole-volume percentile < new normalisation function).

**Fallback**: if no V1..V7 variant clears the criteria, *fall back to V0*
and accept the data-layer status quo. The architecture and loss changes
(sibling spec) carry the v3 recipe; the normalisation fix becomes a v3.1
follow-up. Document the failure explicitly in the artifact decision.json
(`winner: null` with reason).

---

## 4. Output artifacts

Per the `preflight-pattern.md` contract, the routine writes to:

```
artifacts/preflights/normalization_audit/<UTC-timestamp>/
├── report.md
├── decision.json
├── figures/
│   ├── recon_grid_v0.png
│   ├── recon_grid_v1.png
│   ├── ... (one per variant)
│   ├── intensity_histogram_t1c.png    # all variants overlaid, T1c only
│   ├── intensity_histogram_t1pre.png
│   ├── per_region_psnr_bar.png        # PSNR_{whole, ET, NETC, ED, BNWT} bars per variant
│   ├── signal_ratio_scatter.png       # C4 ratio vs C5 ratio per variant
│   └── distribution_kl_bar.png        # KL(V_i || V0) per (variant, modality)
└── tables/
    ├── per_patient_per_variant.csv    # one row per (variant × patient × region)
    ├── aggregate_metrics.csv          # one row per (variant × region)
    └── smoke_cohorts.csv              # one row per (variant × cohort × patient × region)
```

### 4.1 `decision.json` schema (v1.0.0)

```json
{
  "schema_version": "1.0.0",
  "produced_at": "<UTC ISO-8601>",
  "producer": "routines.preflights.normalization_audit:0.1.0",
  "vae_checkpoint": "/abs/path/to/autoencoder_v2.pt",
  "vae_checkpoint_sha256": "<sha256>",
  "git_sha": "<repo HEAD sha at audit time>",
  "config_json": "<full resolved YAML as string>",
  "n_patients_main": 30,
  "n_patients_smoke_per_cohort": 5,
  "variants_tested": ["V0", "V1", "V2", "V3", "V4", "V7"],
  "variants_full": {
    "V0": {"sha256": "<sha of variant code>", "params": {"lo": 0, "hi": 99.5, "fg": true, "clip": true}},
    "V1": {...}
  },
  "metrics_per_variant": {
    "V1": {
      "mae_whole": 0.0044,
      "mae_et":    0.0058,
      "mae_netc":  0.0042,
      "mae_ed":    0.0040,
      "mae_bnwt":  0.0041,
      "psnr_whole": 38.2,
      "psnr_et":   37.5,
      "ssim_whole": 0.985,
      "image_signal_ratio_et_over_bnwt": 1.78,
      "latent_signal_ratio_et_over_bnwt": 1.42,
      "kl_divergence_per_modality": {"t1c": 0.12, "t1pre": 0.08, "t2": 0.05, "flair": 0.06},
      "passes_C1_through_C5": true,
      "passes_smoke_cohorts": true
    }
  },
  "winner": "V1",
  "winner_rationale": "Lowest KL while passing C1-C5. C4 ratio 1.78 (best of the cheap variants). Re-encoding tractable in <12h on server3.",
  "fallback_used": false,
  "next_action": "re_encode_all_cohorts",
  "estimated_re_encode_hours_server3_cuda0": 11.5
}
```

### 4.2 `report.md` template

Reuse the structure of
`/media/hddb/mario/results/vena/encode_ucsf_pdgm_maisi/LATEST/report.md`
(per-modality table + roundtrip figures + stabilization sweep). Add three
new sections: **"Per-region MAE"**, **"Intensity signal preservation"**,
**"Distribution shape vs V0"**.

---

## 5. Re-encoding rollout (post-decision)

Triggered by `decision.json["next_action"] == "re_encode_all_cohorts"`. The
**v3 corpus** must be regenerated end-to-end; the v2 corpus stays on disk
as an immutable fallback.

### 5.1 What gets re-encoded

For **each of the 9 cohorts** (UCSF-PDGM, BraTS-GLI, UPENN-GBM, IvyGAP,
BraTS-Africa-Glioma, BraTS-Africa-Other, LUMIERE, REMBRANDT, BraTS-PED):

1. **Source image H5** → **new latent H5** (renamed with `_v3` suffix):
   `<cohort>_image.h5` → `<cohort>_latents_v3.h5`
2. **Augmented image H5** (for cohorts that have one) → **new augmented latent H5**:
   `<cohort>_image_aug.h5` → `<cohort>_latents_aug_v3.h5`

The image H5s themselves do **not** change — they store **raw intensities**
(N4-bias-corrected, not normalised). Only the encoded latents change
because the encoding applies the new normalisation before pushing through
the VAE.

### 5.2 Per-cohort sequence

For each cohort `C`:

1. **Update the encode routine config**:
   `routines/encode/<C>/configs/default.yaml` — replace the
   `normalization:` block with the winner's spec (or keep V0 as default
   and add a v3-suffixed config).
2. **Update the encoder library**:
   `vena.common.percentile_normalise` is the canonical normalisation
   entrypoint. The winner is either an extension of this function (new
   kwargs, e.g. `clip=False`) or a sibling function (e.g.
   `joint_modality_percentile_normalise`). Either way it is registered
   via the variants.py registry and selected by the encode config.
3. **Run the encode routine** on server3:cuda:0 or server3:cuda:1 (one
   cohort per GPU; cohorts in parallel where possible):
   ```bash
   vena-encode-<C> routines/encode/<C>/configs/default_v3.yaml
   ```
   This produces `<cohort>_latents_v3.h5` with bumped schema version.
4. **Run the encode preflight stabilization sweep**: re-emit
   `/media/hddb/mario/results/vena/encode_<C>_maisi_v3/<UTC>/report.md`
   with the same per-modality MAE/MSE/Lp³ table as the existing V0
   artifacts. **Acceptance**: V3 MAE per modality is within ±20 % of V0
   for T1pre/T2/FLAIR, and within ±50 % for T1c (allowing some change due
   to the new normalisation).
5. **Encode the augmented variants** the same way, reading
   `<cohort>_image_aug.h5` and writing `<cohort>_latents_aug_v3.h5`.

### 5.3 Wall-clock budget

UCSF-PDGM took ~7 h on server3:cuda:0 in the V0 sweep (495 patients ×
4 modalities). Scaling to the full corpus:

| Cohort | N patients (incl. aug) | Est. hours on server3:cuda:0 |
|---|---:|---:|
| UCSF-PDGM | 495 + 724 aug | ~10 h |
| BraTS-GLI | ~1300 + aug   | ~14 h |
| UPENN-GBM | ~610 + aug    | ~9 h |
| IvyGAP    | ~280 + aug    | ~5 h |
| BraTS-Africa-Glioma | ~60 (no aug)  | ~1 h |
| BraTS-Africa-Other  | ~95 (no aug)  | ~1.5 h |
| LUMIERE   | ~250 + aug    | ~5 h |
| REMBRANDT | ~133 + aug    | ~3 h |
| BraTS-PED | 260 (test-only, no aug)  | ~3 h |
| **Total** | ~7300 image-modality encodes | **~50 h** |

Parallelisable across cuda:0 and cuda:1 → ~25 h elapsed (~1 day).

### 5.4 Schema bump and registry rewire

After re-encode:

1. Bump **latent H5 schema** version 2.0.0 → **2.1.0**; add a new root
   attribute `normalization_variant_id` (string, e.g. `"V1"`) and
   `normalization_variant_sha256` (string).
2. Update `routines/fm/train/configs/corpus/corpus_picasso.json` to point
   at the v3 latent paths. Keep a parallel `corpus_picasso_v2.json` for
   the immutable v2 corpus. Add a `corpus_version` field (e.g. `"v3"`).
3. Update `routines/fm/train/configs/corpus/corpus_server3.json` similarly
   (server3 paths only).
4. Bump `routines/fm/train` engine to validate `corpus_version == "v3"`
   when loading v3-only configs.
5. Re-run **all four other gating preflights** against the v3 corpus:
   - `routines/preflights/latent_aug_equivariance/` — must re-pass.
   - `routines/preflights/cohort_dedup/` — re-verify (data didn't change,
     should still pass; if not, investigate).
   - `routines/preflights/decoder_lpl_profile/` — re-run for the S3 LPL
     programme (`w_l` may shift slightly).
   - `routines/preflights/brain_to_latent/` — if exists; re-emit
     brain_latent mask in v3 H5.

### 5.5 Picasso transfer plan

Per `reference_picasso_transfer_route.md` memory, the route is
**server3 → local /tmp → picasso** (~15 MB/s effective). v3 latent H5s
total ~150 GB (estimate from per-cohort sizes). Transfer in `tmux` on
server3, then `tmux` on local relaying to picasso, to survive SSH drops.

```bash
# server3 → local /tmp
ssh icai-server "tmux new-session -d -s xfer_to_local \
  'rsync -avz --progress /media/hddb/mario/data/GLIOMAS/*/h5/*latents_v3.h5 \
    mpascual@<local>:/media/mpascual/Sandisk2TB/staging/'"

# local → picasso
tmux new-session -d -s xfer_to_picasso \
  'rsync -avz --progress /media/mpascual/Sandisk2TB/staging/*latents_v3.h5 \
    picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<cohort>/h5/'
```

Wall-clock at 15 MB/s: ~3 h per cohort large H5 (estimate), so ~24-30 h
total transfer. Run in background overnight after re-encode completes.

---

## 6. Visualisations and tables required for sign-off

Beyond the per-artifact figures in §4, the following deliverables are
**load-bearing** for the v3 go/no-go decision and must be inspectable
before any v3 training launch:

### 6.1 Recon-grid figure (per variant winner)

Layout: 4 rows (T1c, T1pre, T2, FLAIR) × 6 columns (3 mid-axial slices ×
2 columns each: original vs recon). One figure per variant. Plus a 7th
"diff" column = `|original − recon|` ×5 for visual amplification.

**Acceptance**: the winner's `T1c` recon must be visually
indistinguishable from the original at typical viewing window. **The
enhancing rim must be present in the recon at the same intensity as in
the original** (within ±10 % visually). The "diff" column should show no
enhancement-shaped structure (i.e. the recon error is texture noise, not
systematic).

### 6.2 Intensity-histogram overlay (per modality, all variants)

Per modality: log-scale y-axis, 256 bins over `[0, 1.1]`. Each variant
gets a coloured line. Mark vertical dashed lines at the 50th, 90th, 99th,
99.5th, 99.9th, 99.99th percentile of V0.

**Acceptance**: the winner's T1c histogram has a **clearly visible right
tail** beyond V0's 99.5%ile (which is where V0 clips to 1.0). This is the
quantitative analogue of "T1c enhancement is preserved".

### 6.3 Per-region PSNR bar chart

Grouped bars per variant: PSNR_{whole, ET, NETC, ED, BNWT} as five bars
per group. V0 included as reference. Threshold C7 (PSNR_whole ≥ 35) drawn
as a horizontal line.

**Acceptance**: winner's PSNR_ET is no worse than V0's PSNR_ET by more
than 3 dB (which the C2 MAE_ET threshold of 0.015 enforces directly).

### 6.4 Signal-ratio scatter

x-axis: image-space C4 ratio (target ≥ 1.5).
y-axis: latent C5 ratio (target ≥ 1.3).
Each variant is a point; V0 at origin-ish; winner upper-right.

**Acceptance**: at least one variant lies in the upper-right quadrant
beyond (1.5, 1.3).

### 6.5 Smoke-cohort summary table

| Cohort | Variant winner C1 (MAE_whole) | C2 (MAE_ET) | C4 (signal ratio) | All pass? |
|---|---|---|---|---|
| BraTS-GLI | … | … | … | ✓/✗ |
| REMBRANDT | … | … | … | ✓/✗ |
| LUMIERE   | … | … | … | ✓/✗ |
| BraTS-PED | … | … | … | ✓/✗ |

**Acceptance**: all four rows are ✓. One ✗ blocks the rollout; investigate
and either pick a different variant or fall back to V0.

### 6.6 Distribution-KL overlay

Bar chart: per (variant × modality), KL(V_i || V0) in nats. Threshold C3
(KL ≤ 1.0) drawn as horizontal line.

**Acceptance**: winner's bars are all below the threshold. Any variant
with a bar above the threshold is automatically rejected.

---

## 7. What downstream changes are NOT in this spec

To avoid scope creep, the following are **explicitly out of scope** for
this audit:

- **No VAE re-training.** MAISI-V2 stays frozen. If no variant clears
  the criteria, we fall back to V0; we do not fine-tune MAISI.
- **No SWAN encoding work.** SWAN is the proposal's vessel-prior
  source; encoding it through MAISI is a separate audit (per CLAUDE.md
  proposal §3.4 risk row). The v3 trainer doesn't consume SWAN; this
  spec covers T1pre / T1c / T2 / FLAIR only.
- **No mask re-encoding.** The `masks/tumor_latent` 3-channel soft mask
  is normalisation-independent (it's a label-derived mask, not an
  intensity). The existing latent H5's mask group is copied unchanged
  into the v3 latent H5.
- **No corpus dedup re-run.** Per `project_cohort_dedup` memory, the
  dedup decision is based on patient IDs and brats21_id bridges, not
  intensities — independent of normalisation.

These are tracked separately and will reference back to this audit's
`decision.json` when the v3 corpus is the basis.

---

## 8. Sign-off requirements before launching v3 training

The model-implementation sibling spec
(`.claude/notes/changes/2026-06-22_s1_v3_model_implementation.md`)
becomes executable when **all** of the following are true:

1. `artifacts/preflights/normalization_audit/LATEST/decision.json` exists
   with `winner != null`, **OR** `winner: null` with explicit
   `fallback_used: true` (V0 retained).
2. All 9 cohorts' v3 latent H5s have schema version 2.1.0 and pass the
   per-cohort encode preflight (§5.2 step 4).
3. `corpus_picasso.json` (and `corpus_server3.json`) has been updated to
   `corpus_version: "v3"` with all paths pointing at v3 latent H5s.
4. The augmentation preflight (`latent_aug_equivariance`) has been
   re-run and re-passed against v3 latents.
5. All v3 latent H5s have been transferred to Picasso and verified by
   sha256 checksum match against the server3 originals.

Sign-off is a single decision-json update to `corpus_version: "v3"` on
both registries. Until then, the v3 trainer refuses to launch (config
validation error).

---

*End of normalization spec. Sibling spec covers the model and trainer changes.*
