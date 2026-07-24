# 01 — SHARED CONTRACTS (fact sheet for every agent)

> Give this file to **every** agent alongside its task spec. These are the stable, verified facts (audited
> 2026-07-22). Line numbers are deliberately omitted — **read the named file and verify the current code**;
> `/orchestrate` §3 warns that transcribed stale numbers are the most dangerous kind. If a fact here is
> contradicted by the code, **stop and report it**.

> **🔴 ERRATUM (2026-07-22) — the served latent grid is `(48,56,48)`, NOT `(60,60,40)`.** Verified against Picasso
> disk, the producer `data/h5/latent_domain/manifest.py`, and the v3a warm-start config. Every mask/cache is
> `(2,48,56,48)`. Any `(60,60,40)` / `144000` in older revisions of this file or the task specs is stale; the code
> is correct. See §Geometry below.

> **🔴 ERRATUM (2026-07-22) — conditioning channel 0 = TUMOUR CORE (TC = NETC+ET), NOT whole-tumour (WT).** WT is
> ~81% non-enhancing edema (verified UCSF), which was diluting the enhancement signal. `TargetConfig.tumor_region`
> defaults to `"tc"` (`(label>0)&(label!=2)`; `"wt"` kept for the S7 ablation). Everywhere the specs say `[WT,NETC]`
> / `m_wt_soft`, read `[TC,NETC]` / `m_tc_soft` (channel 0 = tumour core, edema excluded); `TC−NETC = ET` = the true
> enhancing region. Segmenter target (Phase 2) is TC, so the G-SEG WT-Dice gate must be re-set to a TC-Dice gate.
> See `[[project_channel0_tumor_core_not_wt]]`.

> **✅ VERIFIED (2026-07-24) — the oracle soft-mask cache is CORRECT; do not re-litigate it.** Full invariant
> audit over **all 3,459 scans / 9 cohorts** (`scripts/mask_audit/`, Picasso array `1636104`):
> **`recompute_max_abs = 0.0` in every cohort** — re-deriving from GT and re-pooling through the canonical
> `apply_crop_pad` → `avg_pool3d(4)` path reproduces the cached `masks/tumor_latent_soft` **bit-exactly**.
> `median_dice_tc_img = median_volratio_tc = 1.000` in all 9; nesting `NETC ≤ TC` holds everywhere; **zero**
> violations of mass-conservation, continuity, or agreement with the independent `masks/tumor_latent` oracle.
> Verdict `0 error · 2 FAIL · 196 WARN · 3,245 OK`. Two caveats that ARE real and must be carried forward:
> **(a)** a TC below ~1-2 latent voxels (**≲130 image voxels**) cannot be represented on the `(48,56,48)`
> conditioning grid at all — it drops below 0.5 everywhere after 4x pooling; **(b)** **BraTS-PED latent
> fidelity is systematically degraded** (`median lat_iou_tc` **0.596** vs 0.82-0.86 for adult glioma
> cohorts, **19.6% TC-empty**), and both FAILs are **defective BraTS-PED GT labels**, not pipeline bugs
> (`dice_tc_img = 1.000`, `recompute = 0.0` on both). LUMIERE second-worst (0.735). **Report the
> oracle→predicted gap PER COHORT in S6** — BraTS-PED's oracle ceiling is already lower.
> See `[[project_mask_audit_2026_07_24]]` and `scripts/mask_audit/README.md`.

## Environment, paths, commands

| What | Value |
|---|---|
| Repo (local dev) | `/home/mpascual/research/code/VENA` |
| Python (local) | `~/.conda/envs/vena/bin/python` (conda env `vena`, ≥3.11) |
| Fast test suite | `~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -q --basetemp=/home/mpascual/.pytest-tmp-<slug>` |
| Lint / format | `~/.conda/envs/vena/bin/python -m ruff check <files>` / `ruff format <files>` |
| Picasso repo to RUN from | `fscratch/repos/VENA-validation` (real git repo) — **not** `fscratch/repos/VENA` |
| Picasso python | `fscratch/conda_envs/vena/bin/python` |
| MAISI VAE / trunk ckpts | see `src/external/LINKS.md` (immutable; never write there) |
| Design authority | `../segmenter_conditioning_design.md` (Part A = injection, Part B = segmenter, A.8/B.f = iter-8) |

**Import isolation (worktree agents MUST paste this proof):**
```bash
cd $WORKTREE && PYTHONPATH=$WORKTREE/src ~/.conda/envs/vena/bin/python -c "
import pathlib, vena, routines
wt = pathlib.Path('$WORKTREE').resolve()
for m in (vena, routines):
    p = pathlib.Path(m.__file__).resolve()
    assert p.is_relative_to(wt), f'LEAK: {m.__name__} -> {p}'
print('import isolation OK')"
```
Never `pip install -e .` from a worktree; the env is shared read-only.

## Project rules that bind every task

- **Routine pattern** (`.claude/rules/preflight-pattern.md`): `cli.py` takes **one positional YAML arg**; engine
  exports a **frozen Pydantic `<Name>RoutineConfig`** with `from_yaml(path)` + an `<Name>Engine.run() -> Path`;
  **no heavy work at import time**; register a console script `vena-<bucket>-<name>` in `pyproject.toml`; persist
  resolved YAML + ISO-8601 timestamp + git SHA + checkpoint paths into the artifact; validate deliverables before
  returning.
- **H5 principles** (`.claude/rules/h5-design-principles.md`): root attrs `schema_version`, `created_at`,
  `producer`, `config_json`, `git_sha`; **self-describing datasets** (`units`, `description`, `dtype`); paired
  `validate_<artifact>(path)->list[str]` + `assert_<artifact>_valid(path)`; gzip-4 + chunk `(1, …)` on bulky
  arrays; **validate-before-return**.
- **Coding standards** (`.claude/rules/coding-standards.md`, `model-coding-standards.md`): `from __future__ import
  annotations` everywhere; type hints on every signature; Google/NumPy docstrings; **no bare `except Exception`**
  (narrow or log-and-reraise); **no magic numbers** (everything via the Pydantic config); **custom exception per
  module**; **3D throughout** (no 2D ops outside clearly-labelled eval utils); logging via `logging`, never
  `print()` in library code; **MAISI primitives import only from `vena.common`**.
- **Tests** (`.claude/rules/model-coding-standards.md` §25-27): co-locate `tests/<area>/test_<name>.py`; **every
  test file declares a marker** (`pytestmark = pytest.mark.<marker>`); pure math/CSV/H5 tests need no checkpoint;
  mark GPU/checkpoint paths `gpu`/`slow`. **Add a `segmentation` marker** to `pyproject.toml` `[tool.pytest.ini_options].markers`.

## Geometry & normalization (LOAD-BEARING — audited)

- **Served latent grid = `(48, 56, 48)`**, 4 channels (MAISI 4× compression of the `(192,224,192)` crop of
  ~`(240,240,155)`; avg-pool stride 4). Every mask target, avg-pool output, and cached mask is **`(2, 48, 56, 48)`
  float32**. **[CORRECTED 2026-07-22: the earlier `(60, 60, 40)` was a transcription error. Verified against Picasso
  disk (`latents/* (N,4,48,56,48)`, `masks/tumor_latent (N,3,48,56,48)`), the producer
  `data/h5/latent_domain/manifest.py` (`LATENT_SPATIAL=(48,56,48)`, `LATENT_CROP_BOX=(192,224,192)`), and the v3a
  warm-start config. `(48,56,48)` is CORRECT; `(60,60,40)=144000` is the stale value.]**
- `rflow.base_img_size_numel = 129024 = 48×56×48` in the v3a config — this **matches** the served latent grid
  (CORRECT; there is **no** grid mismatch to reconcile — the earlier `(60,60,40)=144000` claim was the error). It
  only scales SD3 resolution-aware timestep weighting.
- **Intensity norm is 99.95 canonical** for the VAE/latent path (`percentile_normalise(lower=0, upper=99.95,
  foreground_only=True)` on skull-stripped brain foreground). The **segmenter** is a separate world: it works in
  **image space** with **z-score-on-brain** (nonzero, channel-wise; the `downstream_seg` convention) and never
  touches the VAE. Its soft output is avg-pooled to the latent grid afterward.

## Cohorts & splits

- **CV (trainable) cohorts:** UCSF-PDGM (202), BraTS-GLI (1133), IvyGAP (34), LUMIERE (91), REMBRANDT (63),
  UPENN-GBM (164). **Ring B `test_only` (OOD, never trained on):** BraTS-Africa (Glioma/Other), BraTS-PED (260).
- **Segmenter K-fold splits MUST be a subset of the FM train split** — the segmenter must be **out-of-fold** w.r.t.
  the FM val/test (leakage vector L2). Read the FM split source (`splits/{train,val,test}` in the latent/image H5
  and the corpus registry `routines/fm/train/configs/corpus/corpus_*.json`) before building folds. No independent
  segmenter partition.

## Latent H5 schema (what the generator DataModule consumes) — verify in `src/vena/data/h5/`

- Per-cohort latent H5 groups today: `latents/*` (per modality), `masks/tumor_latent` **`(N, 3, 48, 56, 48)`
  float32 = soft `[NETC, ED, ET]`**, `masks/brain_latent` `(N, 1, …)` int8 (when encoded). Root attrs incl.
  `vae_checkpoint_sha256`. The image-domain H5 carries `images/*`, `masks/{tumor,brain}`, `splits/*`, `metadata/*`.
- **`masks/tumor_latent_soft` `(N, 2, 48, 56, 48)` float32 = soft `[TC, NETC]` — CACHED AND AUDITED on all 9
  cohorts (3,459 scans), schema `2.1.0`, `tumor_region="tc"` attr, `mask_source="gt"`.** The oracle
  `masks/tumor_latent` is byte-untouched beside it. Validated by `assert_latent_soft_mask_group_valid`
  (9/9) plus the full invariant audit above. This is the group S2 serves via `data.mask_source: oracle_soft`.
- **New group to add (task 18):** `masks/tumor_latent_pred` **`(N, 2, 48, 56, 48)`** float32 = soft `[TC, NETC]`
  predicted (channel 0 = tumour core, edema excluded), + a `predicted_mask_seg_sha256` root attr, + a
  `schema_version` bump. Written **beside**
  `masks/tumor_latent` (oracle), never replacing it. Reuse the shared writer/validator under
  `src/vena/data/h5/shared/` and the augmented latent path `src/vena/data/h5/augmented/latent_domain.py`.

## DataModule batch keys (verify in `src/vena/model/fm/lightning/data.py`)

`patient_id`; `z_t1pre, z_t2, z_flair, z_t1c` `(4,48,56,48)`; `m_wt` `(1,…)` **binary** (0.5-threshold of the soft
union); `m_tumor` `(3,…)` soft `[NETC,ED,ET]`; `m_netc/m_ed/m_et` `(1,…)` views; `m_brain` `(1,…)` when present.
**`m_tc_soft` does NOT exist yet — task 20 adds it**, served from the cached `masks/tumor_latent_soft` channel 0 (=TC).
Fallback if the cache is absent = `clip(m_netc + m_et, 0, 1)` (tumour core = NETC+ET, **excluding** ED) — NOT the WT
union `clip(Σ m_tumor)`. (Channel 0 = TC per the erratum banner; `m_wt` binary is retained separately for back-compat.)

## ControlNet conditioning contract (verify in `src/vena/model/fm/controlnet/` + `…/maisi/`)

- **Spec string = `"<kind>:<key>:<downsampler>"`**, `kind ∈ {latent, mask, prior}`; `ConditioningSpec.batch_key()`
  → `z_<key>` / `m_<key>` / `prior_<key>`. Downsamplers: `identity, nearest, avg_pool, trilinear, zero_out`
  (stateless, `out_channels=None`) and `lift_to_4ch` (override `out_channels=4`, needs `in_channels=k`).
- `ConditioningAssembler.channels_per_spec` uses the **`mask_channels` constructor default (=1)**, NOT the runtime
  tensor shape. **⇒ a 2-ch `[TC,NETC]` mask MUST be two 1-ch specs** `mask:tc:identity` + `mask:netc:identity`
  (keys `m_tc_soft`, `m_netc`), else `total_channels` under-counts silently and the hint-net first conv is built wrong.
- **`MaisiControlNet`**: mask enters a **separate `controlnet_cond_embedding` hint net** (`[64]` = zero spatial
  downsampling), **added** to the CN `conv_in` output — **not** concatenated to the noisy latent. Residuals are
  emitted at conv_in + every down resblock + mid, added into the trunk **out-of-place** by `maisi/grad_safe.py`
  when the trunk is trainable, reaching the decoder via skips. `output_scale` is a **`persistent=False` buffer**
  multiplied into **every** residual; `OutputScaleRampCallback` fills it from `global_step`
  (`sigmoid(steepness·(step/ramp_steps−0.5))`). `init_from_trunk` copies the trunk down+mid into the CN encoder;
  the hint net + zero-init output convs stay fresh. The wrapper never passes MONAI `conditioning_scale` (stays 1.0).

## v3a checkpoint (the warm-start source) — verify `routines/fm/train/configs/runs/picasso_s1_v3a_concat_only_fft.yaml`

- v3a = **concat-only, NO ControlNet** (`controlnet.enabled: false`). Trunk `conv_in.in_channels = 16` (4 MAISI +
  3×4 zero-init concat channels for `[t1pre,t2,flair]` latents via `input_concat`). Trunk `trainable: true`,
  `regime: fft`. Loss = **L1 velocity CFM, reduction mean** (no region weights). `rflow.use_timestep_transform:
  true`, `base_img_size_numel: 129024`. EMA 0.9999. Run id `2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f`.
- **Canonical Picasso location — READ-ONLY, NEVER ALTER** (`external-deps.md`):
  `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/vena_project/2026-06-24_16-00-46_s1_v3a_concat_only_fft_ef000c9f/`
  holds `ema_best.ckpt` (all jobs) + `trunk_ema_snapshot.pt` (joint jobs). Every T-13/T-06 YAML sets
  `run.resume_from` to the **absolute `ema_best.ckpt` path**; WARM_START opens it read-only. **New runs write to
  `/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/<run_id>/`** — never the v3a source dir. Any tool that
  would write under the checkpoint path is a BLOCKED/PREMISE-FALSE report, not an action.
- **Warm-start:** `run.resume_from: <v3a_run_id>` → `WARM_START` (weights-only; optimiser/EMA/RNG fresh), handled
  by `_WarmStartCallback` (`routines/fm/train/engine.py`). **⚠ trainable-trunk warm-start is single-shot / not
  resume-safe** — `trunk_ema` is built in `setup()` after the ckpt load and needs a `trunk_ema_snapshot.pt`
  sibling. **`trunk.trainable: false` (freeze-trunk) sidesteps this** and is the recommended de-risking first run.
- **`decision.json` schema is `0.10.0`** (`routines.fm.train`), already carries `controlnet_enabled`,
  `controlnet_conditioning_inputs`, `input_concat`, `loss_cfm_*`, `region_weights`. Bump on any new field.

## Segmentation submodule layout (target — task 10 creates the skeleton)

```
src/vena/segmentation/
├── __init__.py        # public API + __all__:  SegmentationConfig, get_segmentation_model, ...
├── config.py          # Pydantic: SegmentationConfig ⊃ {Model,Data,Loss,Train,Derivation,Metrics}Config
├── exceptions.py      # SegmentationError, SegModelError, SegDataError, SegLossError, ...
├── models/            # 11  registry + bsf_swinunetr.py (Arm A/B) + segresnet.py (Arm C, fork downstream_seg)
├── targets/           # 12  sdt.py (per-component/geodesic) + soft_targets.py + harmonise.py
├── data/              # 14  kfold.py + dataset.py + augment.py
├── engine/            # 13 loss.py ; 17 train.py + predict.py
├── derivation/        # 16  temperature.py + pool.py + ensemble.py
└── metrics/           # 15  overlap.py + calibration.py + gate.py
```
Routines (task 18): `routines/segmentation/train/` (segmenter training) + `routines/segmentation/mask_predict/`
(T-04 latent-H5 write). Console scripts `vena-segmentation-train`, `vena-segmentation-mask-predict`.

## Model checkpoints for BrainSegFounder (LOCATED 2026-07-23 — pinned in `src/external/LINKS.md`)

BSF = **SwinUNETR `feature_size=48`**, **encoder-only** SSL (the decoder is trained from scratch). **Arm priority
RE-PRIORITIZED 2026-07-23: UKB-SSL is the leak-free HEADLINE/PRIMARY; BraTS-SSL is the comparator** (supersedes the
design's original "Arm A primary" — the generalization headline demands L3 purity).
- **Arm B `bsf_swinunetr_ukb` (PRIMARY, leak-free)** — `.../BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt`.
  Healthy UK Biobank, **no BraTS patients, no T1ce** → removes the patient-overlap representation leak (BraTS-SSL saw
  BraTS-GLI = a VENA CV cohort; **OOF cannot fix an SSL-stage leak**) and the contrast-exposure leak. Likely T1-focused →
  **the 3-ch {T1pre,T2,FLAIR} input stem may not transfer** (deep blocks do); `load_bsf_encoder` lists the stem in
  `skipped` if so. `strict=False`.
- **Arm A `bsf_swinunetr_brats` (COMPARATOR)** — `.../BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt` (5 folds,
  4-ch tumour-aware; drop the T1ce slice → feed `[FLAIR,T1pre,T2]`, `strict=False`). Leaks BraTS-GLI images + T1ce → not
  the headline; run it to quantify the leakage↔Dice trade-off.
- **NEVER `BrainSegFounder_finetuned_BraTS/*`** — a supervised BraTS segmenter = L1+L2+L3 leakage at once.
MONAI `SwinUNETR` + `SegResNet` are the library backbones (MONAI, Apache-2.0).

## Region semantics for the generator loss (task 21) — verify in `controlnet/losses/` + `lightning/module.py`

Regions for the region-weighted CFM: **`BG`** (outside brain, from `m_brain`), **`TC`** (tumour core, from the mask
channel 0 — edema EXCLUDED), **`Brain = NOT-BG ∩ NOT-TC`** (brain tissue minus tumour core; **edema now falls in the
Brain region** and is reconstructed from the inputs — this is intended). Iter-8 default weights `{brain: 1.0, tc: 1.0}` (BG excluded or
weight-0 per the existing `region_weights` semantics — **verify which**) → **numerically identical** to the current
unweighted L1 velocity loss. That equivalence is a required test (task 21).
