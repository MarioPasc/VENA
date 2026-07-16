# Phase-2 Validation — Shared Contracts (READ FIRST)

*Authored by the orchestrator, 2026-07-16. Every Phase-2 agent reads this file
before touching code. It pins the facts that were **verified on disk**, not
inferred from documentation. Where this file and a doc disagree, **this file
wins** — the docs contain stale numbers (see §10).*

Companion docs (read for scientific intent, not for facts):
- `.claude/notes/validation/validation_proposal.md` — the protocol (§4 metrics, §6 stats)
- `.claude/notes/validation/validation_fairness.md` — the Phase-1 fairness audit
- `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/README.md` — the results contract

Project rules that still apply in full: `.claude/rules/coding-standards.md`,
`.claude/rules/preflight-pattern.md`, `.claude/rules/h5-design-principles.md`,
`.claude/rules/extensibility.md`.

---

## 1. What Phase 2 is

Phase 1 (**done**) ran every method over every test scan and froze the
predictions to HDF5. **No metrics were computed.** Phase 2 is the analysis
layer: read those frozen H5s, compute the §4 metric suite, apply the §6
pre-registered statistical plan, and emit self-contained artifact folders.

Phase 2 **never** re-runs a model, never re-harmonises, never writes a
volume back to disk.

---

## 2. Environment and worktree isolation (LOAD-BEARING — read twice)

Conda env `vena`, interpreter `~/.conda/envs/vena/bin/python`. Python 3.11.

`vena` is installed into that env as an **editable install path-pinned to the
main checkout** (`/home/mpascual/research/code/VENA`) via a setuptools
`_EditableFinder`. If you are working in a git worktree, a naive
`~/.conda/envs/vena/bin/python -m pytest` **imports the main checkout's code,
not yours**, and your tests will pass against code you did not write.

**Verified empirically (2026-07-16).** The `_EditableFinder` sits at
`sys.meta_path[4]`, *after* `PathFinder[3]`, so `sys.path` wins. `sys.path` is
`[cwd, *PYTHONPATH, *site-packages]`. Therefore the **only** correct invocation
from a worktree is:

```bash
cd <WORKTREE>
PYTHONPATH=<WORKTREE>/src ~/.conda/envs/vena/bin/python -m pytest ...
```

- `cwd = <WORKTREE>` makes `routines` resolve from your worktree.
- `PYTHONPATH=<WORKTREE>/src` makes `vena` resolve from your worktree.

**The split-brain trap:** omit `PYTHONPATH` and `routines` loads from your
worktree while `vena` loads from the main checkout — half your code, half
someone else's, no error. This is the single most likely way to waste a day.

**Mandatory self-check.** Run this as your FIRST command in the worktree and
paste the output into your closing report. Do not proceed if it fails:

```bash
cd <WORKTREE> && PYTHONPATH=<WORKTREE>/src ~/.conda/envs/vena/bin/python -c "
import pathlib, vena, routines
wt = pathlib.Path('<WORKTREE>').resolve()
for m in (vena, routines):
    p = pathlib.Path(m.__file__).resolve()
    assert p.is_relative_to(wt), f'LEAK: {m.__name__} -> {p}'
print('import isolation OK')"
```

**Do NOT** `pip install -e .` and do NOT clone the conda env. The env is shared
read-only; the PYTHONPATH recipe is sufficient and was verified. Installing
would repoint the shared env and break every other agent.

---

## 3. Where the data is

| Host | Root | Status |
|---|---|---|
| Local | `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/` | 288 G, all 360+40 files. **Dev + smoke here.** |
| Picasso | `/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference/` | 289 G, same content. **Full sweep here.** |

```
<ROOT>/
├── README.md
├── analyses/                       ← Phase-2 output root (currently empty)
├── picasso_shard_a_cheap/
│   ├── decision.json               ← provenance (schema_version "1.0")
│   ├── predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5
│   ├── references/<COHORT>.h5
│   ├── figures/  logs/
├── picasso_shard_b_vena/  picasso_shard_c_latent/
├── picasso_shard_d_lddpm/ picasso_shard_e_syndiff/
```

Shards are an operational split only (write-disjoint). Discover everything by
globbing `<ROOT>/*/predictions/*/*/nfe_*.h5`.

**Local disk is 93% full (134 G free). Phase 2 must persist scalars, CSVs and
PNGs only — NEVER a derived volume** (no residual maps, no probability maps).
Recompute residuals on the fly; they are exactly
`t1c_real_harmonised − t1c_synthetic_harmonised`.

**MeningD2 is NOT mounted locally.** The corpus image H5s (needed only for
§4.4 multi-label tumour GT) exist only on Picasso — see §7.

---

## 4. The method grid (VERIFIED on disk 2026-07-16 — 360 files, 40 references)

Exact on-disk spelling, hyphens throughout. 16 methods × 8 cohorts.
**45 distinct (method, NFE) pairs.**

| Method (exact dir name) | NFE files on disk | selection_nfe | Role |
|---|---|---|---|
| `C0-Identity` | 001 | 1 | **Family** — null floor (copies T1pre) |
| `C1-pGAN-t1pre` | 001 | 1 | **Family** |
| `C1-pGAN-t2` | 001 | 1 | supplementary panel |
| `C1-pGAN-flair` | 001 | 1 | supplementary panel |
| `C2-ResViT` | 001 | 1 | **Family** |
| `C3-SynDiff-t1pre` | 004 | 4 | **Family** |
| `C3-SynDiff-t2` | 004 | 4 | supplementary panel |
| `C3-SynDiff-flair` | 004 | 4 | supplementary panel |
| `C4-3D-DiT` | 001 002 005 010 020 **100** | 5 | **Family** |
| `C5-T1C-RFlow` | 001 002 005 010 020 **100** | 5 | **Family** — task SOTA |
| `C6-3D-LDDPM` | 005 010 020 1000 | 1000 | **Family** |
| `C7-3D-Latent-Pix2Pix` | 001 | 1 | **Family** |
| `VENA-S1-v3a` | 001 002 005 010 020 | 5 | ablation — concat-only, no mask cond. |
| `VENA-S1-v3b` | 001 002 005 010 020 | 5 | ablation — concat + 3-ch ControlNet |
| `VENA-S1-v3b-rw` | 001 002 005 010 020 | 5 | **VENA — THE HEADLINE MODEL** (region-weighted) |
| `VENA-S3-LPL-b2c` | 001 002 005 010 020 | 5 | ablation — the LPL null arm |

> An earlier probe reported NFE=100 missing for C4/C5. **That was wrong** —
> `nfe_100.h5` exists for both, in all 8 cohorts. 45 pairs × 8 = 360. ✔

### 4.1 Pre-registered statistical roles (DECIDED 2026-07-16 — do not re-litigate)

- **VENA** (the reference arm of every paired test) = **`VENA-S1-v3b-rw`**.
- **Competitor family** (n = 8, one Holm-Bonferroni family per metric cell,
  per proposal §6.1):
  `C0-Identity`, `C1-pGAN-t1pre`, `C2-ResViT`, `C3-SynDiff-t1pre`,
  `C4-3D-DiT`, `C5-T1C-RFlow`, `C6-3D-LDDPM`, `C7-3D-Latent-Pix2Pix`.
- **Ablation family** (separate, own Holm correction, n = 3):
  `VENA-S1-v3b` (isolates region-weighting — the proposal's "A1"),
  `VENA-S1-v3a` (isolates mask conditioning),
  `VENA-S3-LPL-b2c` (the LPL null).
- **Supplementary rows** (reported in tables, in NO test family):
  `C1-pGAN-t2`, `C1-pGAN-flair`, `C3-SynDiff-t2`, `C3-SynDiff-flair`.

*Why the t1pre panel:* pGAN/SynDiff ran as one-to-one single-source models
(fairness concern ⑥). Picking the winning panel on test would be oracle
selection and breaks pre-registration (P3). T1pre→T1c is the canonical
pre→post CE-synthesis direction, so it is choosable **a priori**, without
looking at test data. Every table still reports all 16 rows.

**All 16 rows are always computed and always reported.** The family
assignment governs *which p-values get Holm-corrected together*, nothing else.

### 4.2 NFE reporting

- **Headline table**: each method at its own `selection_nfe` (above).
- **Matched-NFE sub-table**: NFE = 5, covering the few-step latent tier
  (`VENA-S1-v3b-rw`, the 3 VENA ablations, `C4-3D-DiT`, `C5-T1C-RFlow`).
  This is the only apples-to-apples generative-formulation comparison.
- **Cost-quality Pareto**: every (method, NFE) pair.

---

## 5. Cohorts and rings (VERIFIED — reference-file leading dims)

| Cohort | Ring | role | scans (N in H5) | patients |
|---|---|---|---:|---:|
| `UCSF-PDGM` | A | cv_test | 50 | 50 |
| `BraTS-GLI` | A | cv_test | 127 | 114 |
| `UPENN-GBM` | A | cv_test | 62 | 62 |
| `IvyGAP` | A | cv_test | 5 | 5 |
| `LUMIERE` | A | cv_test | **72** | **11** |
| `REMBRANDT` | A | cv_test | 5 | 5 |
| `BraTS-Africa-Glioma` | B | test_only | 95 | 95 |
| `BraTS-Africa-Other` | B | test_only | 51 | 51 |
| **Ring A** | | | **321** | **247** |
| **Ring B** | | | **146** | **146** |
| **TOTAL** | | | **467** | **393** |

Ring A = in-distribution (held-out CV test). Ring B = external/OOD, **never
contributed to any CV split**. Keep them separate in every report; they are
separate endpoint families (§6.4).

**BraTS-PED (260, pediatric OOD) was never inferred in Phase 1.** A backfill
job is running in parallel. Phase-2 code must therefore **discover cohorts from
disk, never hard-code the list of 8** — when BraTS-PED lands it must flow
through with no code change. Record the cohorts actually analysed in
`decision.json`.

---

## 6. Predictions H5 schema 2.0 — the join contract

### 6.1 Prediction file `predictions/<METHOD>/<COHORT>/nfe_<NNN>.h5`

| Dataset | Shape | dtype | Use |
|---|---|---|---|
| `predictions/t1c_synthetic_harmonised` | (N,240,240,155) | f32 | **THE PREDICTION. Score this.** |
| `predictions/t1c_synthetic_raw` | (N,240,240,155) | f32 | Method-native. **NEVER score this.** Audit only. |
| `masks/brain` | (N,240,240,155) | i8 | binary {0,1} |
| `masks/wt` | (N,240,240,155) | i8 | binary {0,1} — **whole-tumour only, NOT multi-label** |
| `metadata/scan_id` | (N,) | vlen str | **unique row key — THE JOIN KEY** |
| `metadata/patient_id` | (N,) | vlen str | **repeats** for longitudinal cohorts |
| `metadata/cohort` | (N,) | vlen str | |
| `metadata/nfe` | (N,) | i32 | constant per file |
| `metadata/inference_seconds` | (N,) | f32 | §4.5 cost axis |
| `metadata/peak_vram_mb` | (N,) | f32 | §4.5 cost axis |
| `metadata/scan_shape` | (N,3) | i32 | |

Root attrs: `method`, `cohort`, `nfe`, `ring`, `schema_version="2.0"`,
`git_sha`, `harmonisation_recipe`, `references_h5`, `run_id_tag`,
`created_at`, `producer`, `config_json`.

### 6.2 Reference file `references/<COHORT>.h5`

`reference/{t1c_real_harmonised, t1pre_harmonised, t2_harmonised,
flair_harmonised}` (M,240,240,155) f32; `masks/{brain,wt}` (M,…) i8;
`metadata/{scan_id, patient_id, cohort, scan_shape}`.

Written once per cohort **per shard** → 40 files, 5 byte-identical copies of
each of the 8 cohorts. Use any copy; prefer the one the prediction's
`references_h5` attr names.

### 6.3 The join — do this, not the naive thing

1. Read the prediction's root attr `references_h5` (e.g. `references/UCSF-PDGM.h5`).
   Resolve it **relative to the shard root = `pred_path.parents[3]`**
   (`nfe_x.h5` → `<COHORT>` → `<METHOD>` → `predictions` → shard root).
2. **Join on `metadata/scan_id`. NEVER on row index.** A prediction file may
   drop a scan that failed at inference while the reference still lists it.
   (Verified: for UCSF-PDGM the sets *are* equal and *are* in the same order —
   **that is luck, not a contract**. Build the `scan_id → row` map and use it.)
3. The residual is **not stored**. Recompute:
   `r = t1c_real_harmonised − t1c_synthetic_harmonised`.

---

## 7. Intensity contract — the double-normalisation trap

Harmonisation was **already applied in Phase 1**, identically to every method
*and* to the real T1c:

```
percentile_normalise(lower=0.0, upper=99.5, foreground_only=True)   # then exterior forced to 0
```

Verified on disk: inside brain ⊆ [0,1]; outside brain ≡ 0, for both the
prediction and the reference.

**Rules:**
1. **DO NOT re-normalise.** No z-scoring, no min-max, no histogram matching.
   Re-applying `percentile_normalise` silently changes every number.
   (Exception: §4.4's segmenter applies its own bundle preprocessing —
   identically to the real and synthetic arms. That is a fixed instrument, not
   a re-harmonisation.)
2. **`data_range=1.0`** everywhere for PSNR/SSIM/MS-SSIM. Never derive it from
   the data.
3. **Restrict to `masks/brain`** for whole-volume metrics. The exterior is a
   constant 0 in both volumes and would inflate SSIM/PSNR.
4. The `lower=0.0` (vs the proposal §4.1's `0.5`) is deliberate and documented
   in `harmonisation.py`; it matches the encoder contract and is applied
   identically to every method and the reference, so it introduces no
   between-method bias. Do not "fix" it.

---

## 8. Reuse mandates (per `coding-standards.md` rule 6 — libraries first)

**Everything below already exists and is tested. Reimplementing any of it is a
review failure.**

| Need | Use exactly this | Notes |
|---|---|---|
| PSNR / SSIM / MAE / MSE, masked | `vena.model.fm.metrics.ImageMetrics` | `__init__(data_range=1.0, ssim_window_size=7)`; `psnr/ssim/mae/mse(pred, target, mask) -> (B,)`; NaN on empty region |
| **MS-SSIM-3D** | `monai.metrics.MultiScaleSSIMMetric` | monai **1.5.2 installed**. Do NOT add `pytorch-msssim`. Weights per Wang 2003: `[0.0448,0.2856,0.3001,0.3633]` |
| SSIM-3D | `monai.metrics.SSIMMetric` | k1=0.01, k2=0.03, gaussian, `data_range=1.0` |
| Spearman ρ + bootstrap CI | `vena.preflight.priors_validation.statistics.correlation.spearman_with_bootstrap_ci` | `(x, y, *, n_boot=1000, ci=0.95, seed=1337, nan_policy="omit") -> SpearmanResult`. **Canonical.** |
| Holm-Bonferroni | `statsmodels.stats.multitest.multipletests(pvals, method="holm")` | statsmodels 0.14.6 installed |
| Paired Wilcoxon | `scipy.stats.wilcoxon` | scipy 1.15.3 |
| KSG mutual information | `sklearn.feature_selection.mutual_info_regression(..., n_neighbors=5)` | **This IS the Kraskov-Stögbauer-Grassberger estimator** the proposal cites. k=5 = the proposal's default. |
| Binary dilation on GPU | `F.max_pool3d(x.float(), kernel_size=k, stride=1, padding=k//2) > 0.5` | The technique used by `vena.model.fm.metrics.regions._resolve_wt_dilated`. Exact for an all-ones element. **No scipy/CPU round-trip.** |
| Qualitative figure conventions | `vena.model.fm.eval.exhaustive.render_comparison_figure` + `select_content_slices` | See §9 — **read it, match its conventions, do not modify it** |
| Percentile normalise (audit only) | `vena.common.percentile_normalise` | Do NOT apply to data. Only to verify Phase-1's recipe. |
| Validate an input H5 | `vena.inference.h5_writer.validate_predictions` / `validate_references` | Returns a list of violations |
| Resolve test scans → patients | `vena.inference.image_dataset.resolve_test_scan_patient_pairs(cohort, fold=0) -> list[(scan_id, patient_id)]` | For pre-registration |
| Decode helpers, MAISI primitives | `vena.common` only | Never reach into `vena.model.autoencoder.maisi.*` |

`vena.model.fm.metrics.RegionResolver` is **not** directly reusable — it is
bound to the training batch dict. Reuse its *dilation semantics*, not the class.

**Private `_psnr`/`_ssim` helpers exist in every `src/vena/competitors/*/inference.py`
and in `src/external/*`. They are numpy, untested, and have no `data_range`
contract. DO NOT COPY OR CALL THEM.**

---

## 9. Artifact contract — every subroutine emits a self-contained folder

```
<output_root>/<routine>/<UTC-stamp>/          # UTC stamp: "%Y-%m-%dT%H-%M-%SZ"
├── decision.json      # machine-readable contract (schema_version, produced_at,
│                      #   producer, git_sha, input root + shard shas, cohorts
│                      #   analysed, methods analysed, family assignment, every
│                      #   parameter that influenced a number, n_scans, n_patients)
├── report.md          # human-readable; inlines the figures; states what was skipped
├── tables/            # CSVs — aggregated per (method × cohort × nfe), per ring
├── figures/           # PNGs
└── per_scan/          # THE SAMPLE-WISE TIDY CSV — one row per (method, cohort, nfe, scan_id)
                       #   carrying patient_id. This is the input to all downstream
                       #   statistics. Long/tidy format, never wide.
<output_root>/<routine>/LATEST -> <UTC-stamp>   # RELATIVE symlink
```

`output_root` defaults to
`/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses`
and **must be a YAML parameter** (Picasso points it at
`~/execs/vena/inference/analyses`).

**Copy the canonical `_make_run_dir` verbatim** (from
`routines/preflights/latent_aug_equivariance`):

```python
@staticmethod
def _make_run_dir(parent: Path) -> Path:
    parent = Path(parent).resolve()
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = parent / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = parent / "LATEST"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)      # RELATIVE
    except OSError as exc:
        logger.warning("could not update LATEST symlink: %s", exc)
    return run_dir
```

### 9.1 The per_scan CSV is the deliverable that matters most

Everything downstream (the §6 statistical pass, the paper tables) reads it.
Mandatory columns on every routine's `per_scan/*.csv`:

```
method, cohort, ring, nfe, scan_id, patient_id, <metric columns...>
```

Freeze the header once; every row fully populated; no white cells. If a value
is genuinely undefined (empty region), write `NaN` and count it in
`report.md` — never silently drop the row.

### 9.2 Figures — required properties

- **Aggregated figures across methods**, with **statistical significance
  marked** (Holm-corrected, within the correct family; annotate `*`/`**`/`***`
  or `n.s.`, and say in the caption which family and which correction).
- **Tables aggregated per cohort × method**, Ring A and Ring B separate.
- **A qualitative visualisation**: black background, the actual prediction
  rendered, overlaid with the relevant structure (WT contour / brain / residual
  heat-map) where it makes sense.

Match the house conventions in
`vena.model.fm.eval.exhaustive.render_comparison_figure` (black facecolor;
rows sorted by the quality metric **descending**; per-slice `vmin/vmax`
anchored to the **real** slice's `(min,max)` so panels are visually
comparable; metric values in the row ylabel; gray colormap).

**Do NOT modify `render_comparison_figure`** — it is keyed by NFE, is consumed
by `exhaustive_val`, and is guarded by
`tests/model/fm/test_render_figure_signature_dropped_mean_ssim`. Write a
**sibling** method-keyed renderer in your own module and reuse
`select_content_slices`.

---

## 10. Known doc drift — do not "fix" these, they are already known

| Doc claim | Reality on disk |
|---|---|
| proposal §3: Ring A = 173 patients / 253 scans; UCSF-PDGM 21; UPENN-GBM 17 | Ring A = **247 patients / 321 scans**; UCSF-PDGM 50; UPENN-GBM 62. The cohorts grew. Fairness doc §4 is correct. |
| proposal §4.1: `percentile_normalise(lower=0.5, ...)` | `lower=0.0`. Deliberate; matches the encoder contract; applied to every method + the reference. |
| proposal §5.3: H5 schema 1.0, single file, `/reference/*` and `/residuals/raw` inside the prediction file | **Schema 2.0**: predictions and references are **separate files**; the residual is **not stored**. |
| proposal §5.1: C5-T1C-RFlow uses its own custom VAE | C5 decodes through the **same frozen `autoencoder_v2.pt`** as the whole latent tier. This deviation is *fairer* than the proposal. |
| proposal §2 / §6.1: "8 competitors", 10 methods | 16 method rows on disk. Family assignment is pinned in §4.1 above. |
| fairness §4: Ring B = 406 (incl. BraTS-PED 260) | Ring B = **146**. BraTS-PED was never inferred; backfill in progress. |
| proposal §4.4 Appendix A: train nnU-Net from scratch per cohort | **Decided 2026-07-16**: use a pretrained fixed segmenter, report Δ Dice. See `05_downstream_seg.md`. |

Open Phase-1 issues you inherit as **limitations to state, not to fix**:
- ⑤ VENA `_sample` used unseeded `torch.randn_like` → the predictions are not
  bit-reproducible and cross-NFE draws differ. Cannot be fixed without
  re-running inference. State it in `report.md`.
- ① C4–C7 condition on 2 modalities `{t1pre, flair}`; VENA/ResViT on 3
  `{t1pre, t2, flair}`; pGAN/SynDiff on 1. Disclose in every comparison.
- ② VENA-S1-v3b/v3b-rw receive the ground-truth WT mask as conditioning; no
  competitor does. `VENA-S1-v3a` (concat-only, no mask) is the no-oracle
  comparator — **report it next to the headline row every time.**

---

## 11. Traps that will silently produce wrong numbers

1. **Re-normalising.** §7. The single easiest way to be wrong.
2. **Scoring `t1c_synthetic_raw`.** It is method-native and unnormalised.
3. **Joining by row index** instead of `scan_id`. §6.3.
4. **Aggregating by `scan_id` instead of `patient_id`.** LUMIERE is 11/247 =
   4.5% of Ring-A *patients* but 72/321 = 22% of *scans*. Every paired test and
   every bootstrap is **per patient**: collapse scans → patient (mean) FIRST,
   then test. Getting this wrong is an anti-conservative bug that inflates
   significance.
5. **Split-brain imports** in a worktree. §2.
6. **`data_range` derived from the data** instead of fixed at 1.0.
7. **Region-restricted SSIM by mean-fill.** `ImageMetrics.ssim` fills
   out-of-region voxels with the in-region mean — a training-time proxy that
   `model-coding-standards.md` rule 14 explicitly says "degenerates on tiny
   regions". SSIM is a *local windowed* statistic; masking it is not
   well-defined. See `03_paired_fidelity.md` §3 for the required treatment.
8. **`dilate(WT, k=5)` ambiguity.** Pinned as `max_pool3d(kernel_size=5,
   stride=1, padding=2) > 0.5` (radius 2). Must be a YAML parameter so it is
   auditable.
9. **Mixed BraTS label conventions.** Corpus H5s self-declare a `label_system`
   root attr: `BraTS2021` → {1=necrosis, 2=edema, **4**=enhancing};
   `BraTS2023` → {1, 2, **3**}. **Branch on the attr. Never hard-code 4.**
10. **Writing volumes.** 134 G free locally. Scalars, CSVs, PNGs only.
11. **Silent truncation.** If you cap coverage (top-N patients, subsampled
    voxels, fewer bootstrap draws), it goes in `report.md` AND `decision.json`.
    An unlogged cap reads as "we covered everything".

---

## 12. Definition of done (every Phase-2 agent)

- [ ] Import-isolation self-check pasted into the closing report (§2).
- [ ] Library code under `src/vena/validation/<area>.py`; routine is a **thin
      engine** under `routines/validation/<name>/` per `preflight-pattern.md`
      (one positional YAML arg, frozen config with `from_yaml`,
      `Engine.run() -> Path`, no heavy work at import).
- [ ] **Ruff clean on YOUR OWN FILES ONLY** — see §14. The repo is *not* clean;
      do not "fix" it and do not reformat files you did not create.
- [ ] Unit tests under `tests/validation/`, **marked `validation`** (the marker
      is registered by the core task). Synthetic fixtures, no real data, no GPU.
- [ ] A **real-data smoke** that actually ran, on the subset in §13, with the
      produced artifact folder inspected and its correctness asserted — not
      just "it exited 0". Paste the tree and 3 real numbers.
- [ ] `decision.json` + `report.md` + `tables/` + `figures/` + `per_scan/`
      present and populated; `LATEST` symlink resolves.
- [ ] Type hints on every signature; Google/NumPy docstrings; no bare
      `except Exception`; no magic numbers (YAML/config only); no
      `module._private = ...` writes.
- [ ] **Do not edit**: `CLAUDE.md`, `.claude/rules/*`, `src/external/*`,
      `pyproject.toml` (the core task owns it — ask the orchestrator),
      another agent's lane, or `render_comparison_figure`.
- [ ] Report `STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED`. If a premise
      in your plan file is contradicted by the code or the data, **stop and
      report it with evidence — that is a successful outcome**, not a failure.

## 13. The smoke subset (use exactly this)

Three cohorts, chosen deliberately:

| Cohort | scans | patients | Why |
|---|---:|---:|---|
| `UCSF-PDGM` | 50 | 50 | the internal in-domain cohort |
| `LUMIERE` | 72 | **11** | **longitudinal — the only test of the scan→patient collapse** |
| `IvyGAP` | 5 | 5 | tiny, fast |

Methods for smoke: `VENA-S1-v3b-rw`, `C0-Identity`, `C5-T1C-RFlow` at their
`selection_nfe` (5, 1, 5). C0 is the sanity anchor — **every real method must
beat C0 inside the WT**; if it doesn't, your metric is wrong, not the model.

Smoke must exercise the LUMIERE patient-collapse path and assert
`n_patients(LUMIERE) == 11` while `n_scans(LUMIERE) == 72`.

---

## 14. Baseline state of the repo (measured 2026-07-16, before any Phase-2 work)

Recorded by the orchestrator so you can attribute a failure to yourself rather
than inherit blame for pre-existing state.

| Check | Baseline |
|---|---|
| `python -m pytest -m "not slow and not gpu" -q` | **1000 passed, 4 deselected** after T0 landed (was 943 + 57 new). Must stay green; **never lower the count**. |
| `python -m ruff check src/ routines/ tests/` | **475 pre-existing errors** |
| `python -m ruff format --check src/ routines/ tests/` | **70 files would be reformatted** |

`ruff` was **not installed** in the `vena` env; the orchestrator installed
`ruff 0.15.21` (`pip install --no-deps "ruff>=0.6"`) on 2026-07-16. It was
already declared in `pyproject.toml`'s `dev` extra and configured under
`[tool.ruff]`, so this is the env matching its own declaration, not a new
dependency. torch 2.12.0+cu130 / monai 1.5.2 confirmed unperturbed.

The 475 errors and 70 reformats are **pre-existing drift** — the repo was
linted with a much older ruff, and 0.15 added rules and changed formatting.

### 14.1 MANDATORY: always pass `--basetemp` — the suite writes 31 GB per run

**One `pytest` run of this suite writes ~31 GB into `tmp_path`.** pytest retains
the last 3 runs, so `/tmp/pytest-of-mpascual` reaches ~93 GB. `/tmp` lives on the
**137 GB root filesystem**, which this actually filled to **0 bytes free** on
2026-07-16 — killing pytest, the agent harness's own output capture, and all
work until it was cleared. `/home` has ~495 GB free; `/` does not.

**Therefore every pytest invocation you make MUST redirect basetemp to /home,
using a path UNIQUE TO YOU:**

```bash
# <SLUG> = your task slug: paired-fidelity | spatial-residual | downstream-seg
~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -q \
    --basetemp=/home/mpascual/.pytest-tmp-<SLUG>
```

Two rules, both load-bearing:

1. **Never let basetemp land on `/`.** If three agents run the suite against
   `/tmp`, `/` refills within minutes and every one of you stops dead.
2. **Never share a basetemp with another agent.** `--basetemp` **wipes the
   directory at start-up** — two agents pointing at the same path will delete
   each other's fixtures mid-run and produce baffling, non-reproducible
   failures. Use your own slug.

Clean up your own basetemp when you finish if you can; note that
`rm -rf $HOME*` is blocked by a global deny rule, so if you cannot remove it,
just say where you left it and move on. Do not fight the guardrail.

(The 31 GB/run is a pre-existing repo problem: `tests/competitors/*/test_multicohort.py`
writes ~1.4 GB per test across 6 competitor families. **Not yours to fix** —
just don't aim it at `/`.)

(The 31 GB/run is a pre-existing repo problem — some test writes multi-GB
volumes into `tmp_path`. **Not yours to fix**; just don't aim it at `/`.)

**Therefore the lint rule for you is:**

```bash
# Clean on YOUR files only — list them explicitly:
~/.conda/envs/vena/bin/python -m ruff check   src/vena/validation/<yours>.py routines/validation/<yours>/ tests/validation/<yours>.py
~/.conda/envs/vena/bin/python -m ruff format --check  <the same paths>
```

- **Your new files must be 100% clean** under `[tool.ruff]` (line-length 100,
  target py311, double quotes).
- **Do NOT** run `ruff check --fix` or `ruff format` across `src/`, `routines/`,
  or `tests/`. Reformatting 70 files you did not write would collide with every
  other agent's lane and make your diff unreviewable. It is an automatic reject.
- If a pre-existing error is *inside a file you legitimately own*, fix it and say
  so in your report.
