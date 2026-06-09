# Training Regime Overhaul — 2026-06-09

> **Author session timestamp:** 2026-06-09T10:57:00Z
> **Triggered by:** post-mortem of runs `2026-06-07_22-09-05_s1_238cc6ba` and `2026-06-07_22-54-10_s2_eb714abc` (see `.claude/notes/runs/`).
> **Target reader:** the agent that will implement these changes. Treat this file as the design contract; do not silently expand scope. Resolve every "OPEN" item before merging.

---

## TL;DR

Three coupled changes to the FM training pipeline:

1. **LR schedule fix** — add `cosine` branch to the LR lambda; default peak LR `1e-4` (was `5e-5`); set `total_steps` to a sane value so the scheduler actually engages (was `1e9`, effectively constant).
2. **`ContrastiveTumourLoss` reformulation** — replace the mask-sensitivity `|v_orig − v_perturb|^p` formulation with a region-weighted CFM residual `|v_pred − u_target|^p` over the **healthy-brain** region `brain ∩ ¬WT`. Requires propagating a brain mask (`masks/brain_latent`) into every latent H5.
3. **New S3 regime — `s3_cfg`** — keep S2's loss but add classifier-free-guidance-style conditioning dropout (`p_drop = 0.1`–`0.2`) on the WT mask channel.

S1 and S2 stage labels and YAML structure are preserved; only the S2 contrastive math changes. Verification = a 3-epoch S2-LoRA smoke on server-3 against a copy of `picasso_s2_1000ep_lora_r16.yaml`. The full Picasso queue (S1 + S2 + S3) does not run until the smoke passes.

---

## Why now (evidence from the 2026-06-07 runs)

| Finding | Source | Implication |
|---|---|---|
| LR is flat at `5e-5` for 117 728 steps; polynomial decay's `total_steps=1_000_000_000` makes it a no-op. | `.claude/notes/runs/2026-06-07_22-09-05_s1_238cc6ba.md` §(3a); `train_step.csv:lr` ends at `4.999e-5`. | Scheduler bug; cosine + finite horizon will fix it. |
| S2 PSNR at NFE=5 is **1.29 dB worse than S1** (25.95 vs 27.24 dB). | `.claude/notes/runs/2026-06-07_22-54-10_s2_eb714abc.md` §(5). | The current contrastive (mask-sensitivity) trades synthesis quality for attribution. Replacing it with a region-weighted synthesis-error penalty is justified. |
| `contrastive/roi_cap_hit_frac` saturates at 0.965; ROI term is a constant offset late in training. | `train_step.csv` columns. | The current contrastive provides no learning signal after early training — the cap design is broken-by-construction. |
| Trunk LoRA grad-norm grows 28 % over the run; CN grad-norm decays. | S2 note §(2). | LoRA is doing its job (per-parameter grad is 6× smaller than CN). Keep LoRA mechanism; just change what the loss is. |
| The current `m_bg = ¬dilate(m_wt)` over the full latent box includes air/skull. | `module.py:1061–1081`; `data.py:197–200`. | Wrong "background" by user's definition. New regime needs an actual brain mask. |

---

## Literature anchor

| Decision | Reference | Take |
|---|---|---|
| Drop polynomial → cosine + linear warmup | Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017 (arXiv:1608.03983). Goyal et al., *Accurate, Large Minibatch SGD*, 2017 (arXiv:1706.02677, §2.2 — linear warmup). HuggingFace `get_cosine_schedule_with_warmup` is the canonical reference impl. | Cosine is the default in modern diffusion / FM training (MAISI-V2 uses cosine with restarts; Stable Diffusion v2 uses cosine; TumorFlow uses cosine). |
| Peak LR `1e-4` | MAISI-V2 paper (AAAI 2026, arXiv:2508.05772) §4.1 trains joint trunk + ControlNet at `lr=1e-4` with cosine. TumorFlow (arXiv:2603.04058) `lr=1e-4`. | `5e-5` is below community norm; first 1000 steps of S1 hit `grad_clip_active=1.000` even at `5e-5`, but post-warmup the clip activity drops to 0.5 %. Doubling LR is safe with the existing `gradient_clip_val=1.0`. |
| Region-weighted CFM Lp on `brain ∩ ¬WT` | Foreground-weighted MSE is the *de facto* default for skull-stripped MRI synthesis (Chartsias et al., *Multimodal MR synthesis via modality-invariant latent representation*, IEEE TMI 2018). Mahapatra & Bozorgtabar, *Image super-resolution using progressive generative adversarial networks for medical image analysis*, MedIA 2020 — region-aware Lp for vessel emphasis. Konukoglu et al., *Unsupervised lesion detection via image restoration with a normative prior*, MedIA 2021 — ROI-restricted reconstruction. | Standard pattern. The novelty in our case is the latent-space restriction and the explicit `brain ∩ ¬WT` mask. |
| Classifier-free dropout for the conditioning | Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2022. Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), ICCV 2023 — 50 % drop on spatial conditioning. Brooks et al., *InstructPix2Pix*, CVPR 2023 — same recipe. | Canonical. Our `p_drop ∈ [0.1, 0.2]` is below ControlNet's 0.5 because the WT mask is a much stronger condition than a text prompt; over-dropping risks under-using a high-signal input. |
| Binary mask → downsample (not VAE-encode) | MAISI-V2 §3.3; binary masks lose nothing from VAE encoding only because the VAE is KL-regularised. TumorFlow encodes through the VAE because its inputs are *soft* (tumour concentration map, multi-class tissue seg) — not our case. | Keep current `mask:wt:identity` spec (`config.yaml:46`). Re-evaluate when SWAN-vessel soft masks land. |

---

## Decisions locked with the user (2026-06-09 session)

| # | Question | Choice |
|---|---|---|
| 1 | New contrastive math | **CFM residual `|v_pred − u_target|^p`**, regional. The `(v_orig − v_perturb)` machinery is removed from S2 (and re-purposed for S3 / CFG). |
| 2 | Brain mask source | **Encode `masks/brain_latent` into latent H5** via a one-pass routine. Max-pool 4×4×4 from image-domain `masks/brain` (already present in every cohort H5). |
| 3 | WT downweight in new contrastive | **Zero** — new contrastive operates *only* on healthy brain. CFM continues to cover the whole volume; tumour synthesis is trained via CFM. |
| 4 | p exponent | **p = 2** default; treat `p ∈ {1, 2, 3}` as a future ablation row. |
| 5 | Tumour-zeroing augmentation (zero modalities + re-encode) | **Defer to follow-up.** Marked as future work below. |
| 6 | Smoke duration on server-3 | **3 epochs**, S2-LoRA copy of `picasso_s2_1000ep_lora_r16.yaml`. |
| 7 | Where CFG dropout lives | **New stage `s3_cfg`** (own YAML), not a flag on S2. S1 / S2 YAMLs stay untouched apart from the LR-schedule keys. |

---

# CHANGE 1 — LR scheduler fix

## Why

`module.py:1040–1052` dispatches only `polynomial` and any unknown string falls through to `return 1.0` (constant LR after warmup). No `cosine` branch. The S1 run shows `lr` flat at `4.999e-5` from step 1000 to step 117 728 — `total_steps=1e9` makes polynomial's denominator (`max_steps − warmup_steps`) so large that one effective step of decay would take ~10 000 actual steps. The "polynomial scheduler" never decayed.

## What to change

### 1.1 — `src/vena/model/fm/lightning/module.py:1040–1052`

Replace the `lr_lambda` body with a dispatch over `cfg.optim.scheduler ∈ {"constant", "polynomial", "cosine"}`. Use the canonical recipe:

```python
def lr_lambda(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler_kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))   # 1.0 → 0.0 over decay window
    if scheduler_kind == "polynomial":
        return max(0.0, 1.0 - progress)                      # mathematically equivalent to the old form
    if scheduler_kind == "constant":
        return 1.0
    raise ValueError(f"unknown scheduler '{scheduler_kind}'")
```

The unknown-string silent fallthrough is the bug that hid the polynomial misconfiguration; replacing it with `raise` prevents recurrence.

### 1.2 — `routines/fm/train/engine.py:194–200` (`_OptimCfg`)

Add explicit validation that `scheduler ∈ {"constant", "polynomial", "cosine"}`. Default stays `"polynomial"` for backward compat of old YAMLs in flight, but the new run configs use `"cosine"`.

Also: bump `_OptimCfg.lr` default from `5e-5` to `1e-4` so a newly-created YAML matches the MAISI-V2 standard. Existing YAMLs override explicitly so the default change is non-breaking.

### 1.3 — `_TrainingCfg.total_steps` and `_estimated_total_steps()` coordination

`engine.py:215` (`_TrainingCfg.total_steps: int = 50_000`) feeds `optim_cfg["max_steps"]` at `engine.py:894`. The new constraint: **`total_steps` MUST equal `max_epochs × steps_per_epoch`** for cosine to decay over the actual run.

Two paths (pick A — simpler):

- **A. Manual:** require the YAML author to set `total_steps` correctly. Add a `model_validator` on `_TrainingCfg` that warns (not errors — Lightning's `estimated_stepping_batches` is the authoritative number once the DataModule is built) when `total_steps > 10 × max_epochs × 178` (a rough upper bound at our batch / step ratio).
- **B. Auto-derive:** in `FMTrainRoutineEngine.run()`, after the DataModule is built, overwrite `cfg.training.total_steps` with `trainer.estimated_stepping_batches`. Cleaner UX but requires un-freezing the Pydantic config (use `model_copy(update=...)`).

**Pick A** unless the implementing agent has time to refactor — A keeps the contract simple ("YAML is truth") and matches how `_estimated_total_steps()` already operates as a fallback. The `model_validator` is the safety net.

`_estimated_total_steps()` at `module.py:542–563` and `StepHalfWeight` at `schedule.py:60–64` already prefer Lightning's runtime estimate over the YAML value, so they remain correct under either choice.

### 1.4 — Configs that must update

All three production run YAMLs (`routines/fm/train/configs/runs/picasso_*.yaml`):

```yaml
optim:
  lr: 1.0e-04                 # was 5.0e-05
  scheduler: cosine           # was polynomial
  warmup_steps: 1000          # unchanged
training:
  total_steps: 178000         # was 1_000_000_000 — actual value: 1000 epochs × ~178 steps/epoch
```

The `~178 steps/epoch` is from `train_step.csv` step-count ÷ epoch-count in the finished runs (`117 728 / 566 ≈ 208`). The implementing agent must compute the exact value from a 1-epoch dry-run or from the registry (`routines/fm/train/configs/corpus/corpus_picasso.json` patient counts × variant_weights × batches/sample). Pin a comment in each YAML showing the math.

The smoke YAMLs under `routines/fm/train/configs/runs/smoke/` need the same fix.

### 1.5 — Tests to add

New file `tests/model/fm/test_lr_scheduler.py` (marker `unit`):

| Test | Assertion |
|---|---|
| `test_warmup_linear_ramp` | At step `warmup_steps // 2`, lambda equals `0.5`; at step `0`, lambda equals `0.0`; at step `warmup_steps`, lambda equals `1.0`. |
| `test_cosine_decay_endpoints` | After warmup, lambda goes from `1.0` (at `step=warmup_steps`) to `0.0` (at `step=max_steps`). |
| `test_cosine_midpoint` | At progress `0.5`, lambda equals `0.5` (cos(π/2)+1 / 2). |
| `test_polynomial_endpoints` | Backward-compat: poly lambda goes from `1.0` to `0.0` over the decay window. |
| `test_unknown_scheduler_raises` | `scheduler="bogus"` raises `ValueError`. |
| `test_constant_after_warmup` | `scheduler="constant"` returns 1.0 for all post-warmup steps. |

No GPU / Lightning needed — `lr_lambda` is a pure function; extract it into a module-level helper (or test through a stub `optim_cfg` dict) so unit tests run on CPU in < 1 s.

### 1.6 — Verification (end-to-end)

After server-3 smoke (Change 3 §3.6): inspect `metrics/train_step.csv`'s `lr` column. Expected at `batch_size=4`, `grad_accum=2`, 3 epochs, ≈ 178 steps/epoch:
- step 0: `~ 1e-7` (start of warmup)
- step 500: `~ 5e-5` (mid-warmup at lr=1e-4)
- step 1000: `1e-4` (warmup complete)
- step 534 (= 3 × 178): in early cosine decay — value approximately `1e-4 × 0.5 × (1 + cos(π × (534-1000)/(178000-1000)))` (if smoke YAML sets `total_steps=178000`). Sign check: lambda should still be very close to 1.0 (early in decay window).

For the smoke, override the YAML `training.total_steps` to `534` so cosine decay completes over the 3 epochs — that's the only way to *visually* confirm the cosine curve in a 3-epoch run.

---

# CHANGE 2 — `ContrastiveTumourLoss` reformulation

## Why

The current `ContrastiveTumourLoss` (`src/vena/model/fm/controlnet/losses/contrastive.py`) operates on `Δ = v_orig − v_perturb` — the velocity differential between the original forward pass and a second pass where the WT mask channel is zeroed in the conditioning. This is an **attribution loss** (does the model use the WT channel?) — useful in MAISI-V2 because they want controllable synthesis, but not aligned with our goal (high-quality T1Gd synthesis). The S1-vs-S2 1.29 dB regression is the empirical evidence that the trade-off is unfavourable.

The user's intent for "contrastive": penalise generation errors more heavily where it matters — in the **healthy brain region**, which contains the contrast-enhancing tissue (vessels) we care about. The cleanest formulation is a region-weighted CFM residual.

## New math

For each batch element, given the predicted velocity `v_pred` (the trunk + ControlNet output on the *original* conditioning) and the FM target `u_target = x_clean − noise`:

$$
\mathcal{L}_{\text{brain}}^{(p)} \;=\; \frac{1}{|m_{\text{healthy}}| \cdot C} \;\sum_{i \in m_{\text{healthy}}} \sum_{c=1}^{C} \big| v_{\text{pred}}^{(c)}(i) - u_{\text{target}}^{(c)}(i) \big|^{p}
$$

where
- $m_{\text{healthy}} = m_{\text{brain}} \cap \neg m_{\text{wt}}$ (binary, latent-space, shape `(B, 1, 60, 60, 40)`)
- $C = 4$ (latent channel count)
- $p = 2$ default; ablate {1, 2, 3} in a follow-up
- Empty-mask guard: `clamp_min(1.0)` on `|m_healthy| · C` so an all-zero brain mask returns 0, not NaN.

Per-sample (B,) tensor returned for the per-cohort breakdown (the per-cohort logging path is already wired in `module.py:438+` via the `per_sample()` accessor we added).

**No more `v_perturb`.** `requires_perturbed_pass` becomes `False` for S2. The perturb machinery (`conditioning.py:200–211`, `module.py:382`) stays in code for re-use by S3 (CFG).

## What to change

### 2.1 — New routine: `routines/encode/brain_to_latent/`

One-pass encoder that walks every latent H5 (base + offline-augmented variants v0–v4) and writes `masks/brain_latent` as `int8`, shape `(N, 1, 60, 60, 40)`, gzip level 4.

For each row:
1. Read `masks/brain` (int8, image shape) from the corresponding image-H5 (path via `corpus_*.json`).
2. If the latent row corresponds to an augmented variant, replay the same offline-augmentation geometric transform (translation / flip / rotation) on the brain mask. The augmentation manifest is under `src/vena/data/h5/latent_domain/manifest.py` — implementing agent should reuse `AugmentationTracker.replay_geometric()` (or add it if absent — check first).
3. Crop to the same brain-box as the latent (same `CropPadSpec`).
4. Max-pool 4×4×4 (stride 4) to `(60, 60, 40)`. Use `F.max_pool3d`. Threshold at `> 0` to keep binary.
5. Write `masks/brain_latent` to the latent H5.

CLI: `vena-encode-brain-latent <corpus_registry.yaml>`. Idempotent (skip rows where the dataset already exists, with `--overwrite` flag). Add a manifest update under `producer="routines.encode.brain_to_latent:0.1.0"`.

### 2.2 — Latent H5 schema bump

`src/vena/data/h5/latent_domain/manifest.py` (or wherever the latent H5 validator lives — confirm via `grep schema_version`):
- Bump `schema_version` to `"2.0"` (breaking add).
- Add `masks/brain_latent` to the validator's required-on-S2 list (S1 doesn't need it).
- Update `assert_<artifact>_valid()` to check shape `(N, 1, 60, 60, 40)`, dtype `int8`, units `"binary"`, description `"brain mask in latent space (max-pool 4 of masks/brain)"`.

### 2.3 — `LatentH5Dataset.__getitem__` — `src/vena/model/fm/lightning/data.py:184–209`

After the `m_wt` block, add `m_brain` read:

```python
m_brain_lat = h5["masks/brain_latent"][row]      # (1, 60, 60, 40) int8
item["m_brain"] = torch.from_numpy(m_brain_lat).bool()
```

Behaviour when `masks/brain_latent` is absent: emit a one-time WARNING and synthesise `m_brain = torch.ones_like(m_wt, dtype=torch.bool)` (degrades the new contrastive to "full image Lp" but does not crash). This protects S1 runs (which don't need it) and graceful-degrades when running against an old latent H5.

### 2.4 — `LossInputs` — `src/vena/model/fm/controlnet/losses/base.py:28–64`

Add field:

```python
m_brain: torch.Tensor | None = None        # binary (B, 1, h, w, d); brain foreground in latent space
```

Update the dataclass docstring (rule 17 — stale docstrings count as bugs).

### 2.5 — `LightningModule.training_step` — `src/vena/model/fm/lightning/module.py:392–408`

After existing `m_wt` and `m_bg` lines, add:

```python
m_brain = batch.get("m_brain")              # bool (B, 1, h, w, d) when latent H5 has the key
```

Pass `m_brain=m_brain` into `LossInputs(...)`.

The `_bg_from_wt` helper at `module.py:1061–1081` stays (it's still consumed by the optional S3 CFG path for symmetry with the perturbed conditioning). No call-site changes outside the LossInputs construction.

### 2.6 — `ContrastiveTumourLoss.forward()` — `src/vena/model/fm/controlnet/losses/contrastive.py`

Rewrite the entire body (keep the class name, signature, and `per_sample()` accessor we shipped on 2026-06-09; the per-cohort breakdown wiring at `module.py:438+` then continues to work without changes).

New constructor params (Pydantic-side: update `routines/fm/train/configs/runs/picasso_s2_*.yaml` `loss.contrastive` block):

```python
class ContrastiveTumourLoss(AbstractFMLoss):
    def __init__(self, p: float = 2.0) -> None:
        super().__init__()
        if p <= 0: raise ValueError(...)
        self.p = float(p)
```

Drop `lambda_roi`, `lambda_bg`, `delta`, `p_t`, `p_b` from the constructor; the outer weight is still applied by `CompositeLoss`. **This is a config-schema breaking change** — flag in the changelog and remove the obsolete keys from the production YAMLs in the same PR.

Body (mirror the per-sample / aux pattern of the old class):

```python
def forward(self, inputs: LossInputs) -> torch.Tensor:
    if inputs.v_orig is None:
        raise ValueError("ContrastiveTumourLoss requires v_orig (the predicted velocity).")
    if inputs.u_target is None:
        raise ValueError("ContrastiveTumourLoss requires u_target.")
    if inputs.m_wt is None or inputs.m_brain is None:
        raise ValueError(
            "New contrastive needs m_wt AND m_brain in LossInputs. Re-encode the "
            "latent H5 with `vena-encode-brain-latent` so `masks/brain_latent` is "
            "present, or use S1 (no contrastive)."
        )

    residual = (inputs.v_orig - inputs.u_target).abs()         # (B, C, h, w, d)
    m_brain = inputs.m_brain.to(residual.dtype)
    m_wt    = inputs.m_wt.to(residual.dtype)
    m_healthy = m_brain * (1.0 - m_wt)                          # binary, (B, 1, h, w, d)

    C = float(residual.shape[1])
    weighted = residual.pow(self.p) * m_healthy                 # broadcasts over C
    num = weighted.flatten(1).sum(dim=1)                        # (B,)
    den = (m_healthy.flatten(1).sum(dim=1) * C).clamp_min(1.0)  # (B,) — empty-brain guard
    per_sample = num / den                                       # (B,)

    self._per_sample = per_sample.detach()
    total = per_sample.mean()

    with torch.no_grad():
        # New aux diagnostics. Replace the old delta-based ones.
        healthy_voxel_frac = (m_healthy.flatten(1).sum(dim=1) /
                              (m_brain.flatten(1).sum(dim=1).clamp_min(1.0)))
        self._aux = {
            "residual_lp_mean_brain": per_sample.mean().detach(),
            "healthy_voxel_frac":     healthy_voxel_frac.mean().detach(),
            "residual_lp_mean_wt":    _masked_lp(residual, m_wt, C, self.p).detach(),
        }
    return total


def _masked_lp(residual, mask, C, p):
    """Helper: same Lp computation, different mask; diagnostic only."""
    w = residual.pow(p) * mask.to(residual.dtype)
    num = w.flatten(1).sum(dim=1)
    den = (mask.flatten(1).sum(dim=1) * C).clamp_min(1.0)
    return (num / den).mean()
```

Update the docstring to reflect the new semantics (rule 17). Add references to this design note and the user-decision row above.

### 2.7 — `_PerCohortContrastive` log key prefix

Already wired by the 2026-06-09 morning PR:
- `module.py:438+` reads `composite.terms["contrastive"].per_sample()` and logs `train/contrastive_cohort_<safe>`.
- `train_csv.py:_COHORT_PREFIXES` contains `"contrastive_cohort_"`.

**No change.** The per-cohort breakdown automatically reflects the new per-sample tensor.

### 2.8 — Downstream consumers

- `routines/fm/exhaustive_val/engine.py` — no change; this loss is training-only.
- `src/vena/model/fm/post_train/plot_loss_grad.py` — no change; auto-detects active losses.
- `src/vena/model/fm/post_train/loaders.py` — no change.

The change is **invisible to the post-train plotting layer**: the same `train/contrastive` key shows up, only the values change. The aux columns change names (`residual_lp_mean_brain` etc.); document the rename in the cohort runbook.

### 2.9 — Tests

#### Update existing
- `tests/model/fm/test_losses_contrastive.py` — every test currently constructs the old loss with `lambda_roi`/`lambda_bg`. These all break. Rewrite around the new signature. Keep the `test_per_sample_returns_b_shaped_tensor_after_forward` test (the new class still exposes `per_sample()`).

#### Add new
- `tests/model/fm/test_losses_contrastive_new.py` (marker `unit`):

| Test | Assertion |
|---|---|
| `test_zero_loss_when_brain_equals_wt` | If `m_brain == m_wt` (healthy region is empty), loss is exactly `0`, not NaN. |
| `test_loss_matches_manual_lp_on_synthetic_mask` | Hand-construct a `(2, 4, 4, 4, 4)` residual with known values, a mask where `m_healthy` covers exactly 8 voxels; assert the loss equals the analytical sum / (8 × 4). |
| `test_loss_isolates_to_brain_minus_wt` | Place a non-zero residual in three regions: (i) inside WT, (ii) inside healthy brain, (iii) outside brain. Vary only region (i) and (iii); loss must NOT change. Vary region (ii); loss MUST change. |
| `test_per_sample_shape` | Output of `per_sample()` is `(B,)` and detached. |
| `test_aux_keys` | `aux()` returns `{residual_lp_mean_brain, healthy_voxel_frac, residual_lp_mean_wt}`. |
| `test_missing_m_brain_raises` | Constructing with `m_brain=None` and calling `forward` raises a clear ValueError mentioning `vena-encode-brain-latent`. |

#### Mask-correctness suite
`tests/data/h5/test_brain_latent_encode.py` (marker `unit`):

| Test | Assertion |
|---|---|
| `test_brain_latent_shape_dtype` | Output H5 has `masks/brain_latent` with shape `(N, 1, 60, 60, 40)`, dtype `int8`, units `binary`. |
| `test_brain_latent_max_pool_matches_image_domain` | For a synthetic image-H5 with a known brain mask, the encoded `brain_latent` matches `F.max_pool3d(brain_image, kernel=4, stride=4).bool().to(int8)`. |
| `test_brain_latent_replays_augmentation` | With a flip-LR augmentation applied to the latent, the encoded brain mask is also flipped. |
| `test_brain_latent_idempotent` | Running the encoder twice produces byte-identical output (no `--overwrite`). |

### 2.10 — YAML migration (S2 configs)

In `routines/fm/train/configs/runs/picasso_s2_1000ep_lora_r16.yaml` and `picasso_s2_1000ep_fft.yaml`:

```yaml
loss:
  cfm:
    weight: 1.0
    reduction: mean
    norm: l2
  contrastive:
    weight: 0.1                   # was 0.01 — see decision row #5 in S2 run note; raise so signal is non-negligible
    p: 2.0                        # was: lambda_roi/lambda_bg/delta/p_t/p_b — drop those
model:
  controlnet:
    perturb_keys: []              # was [wt] — S2 no longer needs a perturbed pass
```

Keep `regions:` block as-is (used by exhaustive_val) but the implementing agent should sanity-check that `regions.brain.source` reflects whatever path the H5 takes.

---

# CHANGE 3 — New `s3_cfg` regime

## Why

Classifier-free dropout achieves the same mask-attribution that the OLD contrastive provided, at a fraction of the cost: no second forward pass per step (≈ 25 % wall-clock saving), and a CFG knob at inference. The S3 regime stacks this on top of S2's new contrastive — keeping high-quality healthy-brain synthesis AND mask-controllable inference.

## What to add

### 3.1 — `_TrainingCfg` extension

`routines/fm/train/engine.py:_TrainingCfg` — add:

```python
conditioning_dropout_p: float = 0.0           # 0.0 disables; S3 uses 0.1–0.2
conditioning_dropout_keys: tuple[str, ...] = ("wt",)
```

`0.0` default keeps S1/S2 byte-identical.

### 3.2 — `LightningModule.training_step` — `src/vena/model/fm/lightning/module.py:380–392`

Before the `cond_orig = self.conditioning(batch)` call, sample a Bernoulli `drop = (torch.rand((B,), device=...) < cfg.training.conditioning_dropout_p)`. For each sample where `drop[i] = True`, build conditioning with `perturb_keys=conditioning_dropout_keys` (re-using the existing `conditioning.py:200–211` zero-replace logic). For samples where `drop[i] = False`, use the unperturbed conditioning. Concat at the batch dimension.

Per-sample dropout is cleaner than per-batch (mixes conditioned and unconditioned in the same step → averaged gradient, identical to the canonical recipe in ControlNet appendix A.2).

Log `train/cfg_dropout_active_frac` per step (a Bernoulli observation; should hover near `p_drop`).

### 3.3 — Optional: inference-time CFG mixing

Out of scope for this PR. Track as future work (§"Out of scope"). The training-time change is sufficient to ship the regime; inference CFG can land before external-val.

### 3.4 — New stage label & config

`engine.py:_RunCfg.stage: Literal["s1", "s2", "s3_cfg"]` — extend the Literal.

`routines/fm/train/engine.py:_assert_preflight_gates(cfg)` — confirm S3 needs the same preflights as S2 (probably yes: cohort_dedup, latent_aug_equivariance). Update the gate logic to accept both.

New YAML: `routines/fm/train/configs/runs/picasso_s3_1000ep_cfg.yaml` — copy of `picasso_s2_1000ep_lora_r16.yaml` with:

```yaml
run:
  stage: s3_cfg
training:
  conditioning_dropout_p: 0.15
  conditioning_dropout_keys: [wt]
```

All other fields identical (same loss block as the new S2: CFM + new ContrastiveTumourLoss).

### 3.5 — Tests

`tests/model/fm/test_cfg_dropout.py` (marker `unit`):

| Test | Assertion |
|---|---|
| `test_zero_p_drop_no_op` | With `p_drop=0`, conditioning equals the unperturbed conditioning byte-identically over 100 random batches. |
| `test_p_drop_one_fully_zeros_wt_channel` | With `p_drop=1`, every sample's WT channel in the conditioning is exactly zero. |
| `test_p_drop_half_bernoulli` | With `p_drop=0.5`, the fraction of zeroed-WT samples over 1000 draws is within 3σ of 0.5. |
| `test_drop_only_affects_listed_keys` | With `conditioning_dropout_keys=[wt]`, latent channels (t1pre, t2, flair) are bit-identical between dropped and non-dropped samples. |

### 3.6 — Server-3 smoke run (acceptance criterion)

Copy `routines/fm/train/configs/runs/picasso_s2_1000ep_lora_r16.yaml` to `routines/fm/train/configs/runs/smoke/server3_s2_3ep_lora_r16.yaml`. Edits:

```yaml
run:
  stage: s2
training:
  max_epochs: 3
  total_steps: 534                # 3 × 178 — cosine decays over the smoke window
  checkpoint_every_epochs: 1
exhaustive_val:
  enabled: true
  every_epochs: 1                  # one validation per epoch on smoke
  n_patients: 6                    # cheap
  device: cuda:1                   # second 4090
output:
  experiments_root: /scratch/mpascual/vena/smoke
data:
  corpus_registry: routines/fm/train/configs/corpus/corpus_server3.json
  preflight_decision_path: /scratch/mpascual/vena/preflights/latent_aug_equivariance/LATEST/decision.json
  dedup_decisions_path:     /scratch/mpascual/vena/preflights/cohort_dedup/LATEST/decision.json
optim:
  lr: 1.0e-04
  scheduler: cosine
  warmup_steps: 1000
```

Exact server-3 paths (`/scratch/mpascual/...`) must be confirmed by the implementing agent against `corpus_server3.json` — do not invent paths. The `server3` skill knows how to dispatch the run.

Acceptance: smoke produces `<run_dir>/plots/` (post-train hook fires), `metrics/train_step.csv` shows the `lr` column cosine-decaying after warmup, `train_epoch.csv` shows `contrastive_mean` taking on negative-free positive values (Lp residual is non-negative), and `exhaustive_val/epoch_001/metrics.csv` exists.

---

# Configs to update — summary

| File | Change |
|---|---|
| `routines/fm/train/configs/runs/picasso_s1_1000ep.yaml` | `optim.lr: 1e-4`, `scheduler: cosine`, `training.total_steps: 178000` |
| `routines/fm/train/configs/runs/picasso_s2_1000ep_fft.yaml` | Same + new `loss.contrastive` block (drop lambda_roi/bg/delta/p_t/p_b; add `p: 2.0`, raise `weight` to `0.1`); `model.controlnet.perturb_keys: []` |
| `routines/fm/train/configs/runs/picasso_s2_1000ep_lora_r16.yaml` | Same as `picasso_s2_1000ep_fft.yaml` |
| `routines/fm/train/configs/runs/picasso_s3_1000ep_cfg.yaml` | **NEW.** Copy of `picasso_s2_1000ep_lora_r16.yaml` + `run.stage: s3_cfg` + `training.conditioning_dropout_p: 0.15` |
| `routines/fm/train/configs/runs/smoke/server3_s2_3ep_lora_r16.yaml` | **NEW.** Copy of `picasso_s2_1000ep_lora_r16.yaml` + server-3 paths + 3 epochs + `total_steps: 534` |
| Existing smoke YAMLs under `configs/runs/smoke/` | LR-scheduler fields updated to cosine + finite `total_steps` |

---

# Out of scope (track as future work)

1. **Tumour-zeroing offline variant** (`v5_tumour_zeroed`). The user wants this evaluated separately. Spec: zero T1pre + T1c + T2 + FLAIR + WT mask inside `m_wt`, re-encode the four modalities through MAISI-V2 VAE, store as a new variant alongside v0–v4 with `variant_weights.v5: 0.1`. **Risk:** zeroing the T1c target changes the synthesis goal in those samples — the model would learn "mask says nothing, generate nothing" instead of healthy tissue. Resolve before implementing.
2. **Inference-time CFG mixing.** Once `s3_cfg` is trained, the sampler needs a `guidance_scale` parameter: `v = v_uncond + w · (v_cond − v_uncond)`. Lives in `src/vena/model/fm/inference/euler.py`. Default `w = 1.0` (no guidance), ablate `w ∈ {1.0, 2.0, 5.0}` once S3 produces checkpoints.
3. **p-exponent ablation** for the new contrastive: `p ∈ {1, 2, 3}` × `weight ∈ {0.05, 0.1, 0.2}` matrix. Belongs to `routines/training/ablations`.
4. **Vessel-conspicuity loss (proposal §5.4).** Replaces or complements the healthy-brain regional Lp once SWAN-derived vessel masks land. Different code path entirely (operates on decoded image-space residuals). Stays parked until vessel pre-flight closes.
5. **`m_brain` for S3 + CFG-only training.** Strictly speaking S3 inherits S2's loss so it still needs `m_brain`. If a future S4 wants to train purely with CFG and no regional contrastive, it could skip the brain-mask encoding. Not relevant now.

---

# References

## Code locations (full list, file:line where useful)

| Symbol | Path |
|---|---|
| `configure_optimizers` | `src/vena/model/fm/lightning/module.py:1014–1053` |
| `lr_lambda` dispatch | `module.py:1040–1052` |
| `_OptimCfg` Pydantic | `routines/fm/train/engine.py:194–200` |
| `_TrainingCfg.total_steps` | `engine.py:215` |
| `optim_cfg.max_steps` plumbing | `engine.py:894` |
| `_estimated_total_steps()` | `module.py:542–563` |
| `StepHalfWeight` | `src/vena/model/fm/controlnet/losses/schedule.py:60–64` |
| `WT mask construction` | `src/vena/model/fm/lightning/data.py:197–200` |
| `m_wt, m_bg setup` | `module.py:392–396` |
| `_bg_from_wt` | `module.py:1061–1081` |
| `ContrastiveTumourLoss` | `src/vena/model/fm/controlnet/losses/contrastive.py` |
| `LossInputs` dataclass | `src/vena/model/fm/controlnet/losses/base.py:28–64` |
| `CompositeLoss.forward` | `base.py:141–159` |
| `REQUIRED_REGIONS` | `src/vena/model/fm/metrics/regions.py:69` |
| `Image-H5 `masks/brain` (UCSF-PDGM)` | `src/vena/data/h5/ucsf_pdgm/image_domain/convert.py:162–163` |
| `Image-H5 `masks/brain` (BraTS-GLI)` | `src/vena/data/h5/brats_gli/image_domain/convert.py:185` |
| `Image-H5 `masks/brain` (UPENN-GBM)` | `src/vena/data/h5/upenn_gbm/image_domain/convert.py:124` |
| `Latent H5 manifest` | `src/vena/data/h5/latent_domain/manifest.py` |
| `Encode pipeline (current)` | `routines/encode/maisi/` |
| `Conditioning perturb` | `src/vena/model/fm/controlnet/conditioning.py:200–211` |
| `perturb_keys default` | `module.py:147` |
| `cond_perturb call site` | `module.py:382` |
| `Per-cohort contrastive logging` | `module.py:438+` (added 2026-06-09 AM) |
| `_COHORT_PREFIXES` | `src/vena/model/fm/lightning/callbacks/train_csv.py` |
| `Old contrastive test file` | `tests/model/fm/test_losses_contrastive.py` (REWRITE) |
| `LR scheduler tests` | none — must be added |

## Project rules consulted

- `.claude/rules/model-coding-standards.md` — §6 (TrainMetricsCSV pattern), §11 (intensity-space parity), §16–18 (private-attr discipline)
- `.claude/rules/coding-standards.md` — §6 (libraries-first), §15 (exception narrowing), §17 (docstring drift)
- `.claude/rules/preflight-pattern.md` — `decision.json` v0.6.0 schema; new YAMLs must produce conformant decisions
- `.claude/rules/h5-design-principles.md` — schema_version bump policy

## Papers

| Decision | Paper |
|---|---|
| Cosine + linear warmup | Loshchilov & Hutter, ICLR 2017, arXiv:1608.03983; Goyal et al. 2017, arXiv:1706.02677 |
| Region-weighted MRI Lp | Chartsias et al., IEEE TMI 2018; Mahapatra & Bozorgtabar, MedIA 2020; Konukoglu et al., MedIA 2021 |
| Classifier-free guidance | Ho & Salimans, NeurIPS-W 2022, arXiv:2207.12598 |
| ControlNet conditioning dropout | Zhang et al., ICCV 2023, arXiv:2302.05543 §3.5 / Appendix A.2 |
| MAISI-V2 (baseline this work departs from) | AAAI 2026, arXiv:2508.05772 |
| TumorFlow (referenced for conditioning encoding) | arXiv:2603.04058 |

## Past session artefacts referenced

- `.claude/notes/runs/2026-06-07_22-09-05_s1_238cc6ba.md` — S1 post-mortem
- `.claude/notes/runs/2026-06-07_22-54-10_s2_eb714abc.md` — S2 post-mortem
- 2026-06-09 morning PR (already merged): per-cohort contrastive logging in `train_step.csv`/`train_epoch.csv`; healthy-brain (`nwt`) region in exhaustive-val `metrics.csv`; `ContrastiveTumourLoss.per_sample()` accessor.

---

# OPEN ITEMS (resolve before merging)

These are the few residual ambiguities the implementing agent should NOT silently decide:

1. **Exact `steps/epoch` for `total_steps`** in production YAMLs. The S1/S2 runs averaged 208 steps/epoch (`117 728 / 566`); the user-cited rough value was 178. Pin the exact number from a 1-epoch dry-run on Picasso or from `MultiCohortLatentDataset.__len__() / batch_size / grad_accum`. Confirm before the Picasso queue.
2. **`AugmentationTracker.replay_geometric()`** existence (Change 2.1 step 2). If absent, decide whether to add it or pre-store the per-row geometric transform parameters in the latent H5 manifest. Adding `replay_geometric` is cleaner; pre-storing per-row params is faster to implement.
3. **`contrastive.weight` for new S2** — proposed `0.1` based on the analysis in `.claude/notes/runs/2026-06-07_22-54-10_s2_eb714abc.md` §(5) ("outer weight 0.01 is too low"). Implementing agent: confirm `0.1` does not blow up training in the smoke. If it does, fall back to `0.05` and document the decision in the run dir's `decision.json`.
4. **AdamW parameter groups** — the current single-group setup mixes LoRA params with CN params at the same LR. With LoRA params getting `lr=1e-4` and `weight_decay=0.01`, the adapters may decay too fast. Standard LoRA recipes use `weight_decay=0.0` on the adapter weights. Implementing agent: decide whether to introduce per-group `weight_decay` (clean, requires param-group construction). My recommendation: yes, set `weight_decay=0.0` on LoRA params. Reference: HuggingFace PEFT recipes (`peft.utils.other.get_peft_model_state_dict`).

End of spec.
