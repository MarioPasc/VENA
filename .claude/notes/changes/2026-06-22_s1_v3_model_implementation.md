# S1 v3 — Model Implementation Spec: Channel-Concat + Per-Region ControlNet + Region-Weighted L1

*Mario Pascual González — VENA, IBIMA-BIONAND.*
*2026-06-22 — child of `.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md` (read **§2, §6.0, §6.1–§6.4, §6.9, §6.10** of that document for context). Sibling: `.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md` (data layer).*

---

## 0. Goal in one sentence

Implement two architectural variants of S1 v3 — **A: sequence-only**
(modality latents via channel-concat at trunk `conv_in`, ControlNet
dropped) and **B: sequence + per-sub-region masks** (modality latents
via channel-concat **plus** 3-channel `[NETC, ED, ET]` masks via
ControlNet) — together with region-weighted L1 velocity loss (with a
one-flag disable), per-region exhaustive-val metrics
(PSNR/MAE/MSE/SSIM for ET / NETC / ED / BRAIN_NOT_WT / whole), and the
schema bumps needed for round-tripping into `decision.json`.

The two variants share **the loss, the metrics, the data path, and the
YAML scaffolding** — only the architecture differs. Both ship as
independent Picasso configs (`s1_v3a_concat_only_fft.yaml`,
`s1_v3b_concat_plus_cn3ch_fft.yaml`) so they can train in parallel and
the A → B comparison answers "is per-sub-region ControlNet load-bearing
for ET-quality?".

This spec assumes the v3 latent corpus is already in place (sibling spec
§5). If the normalisation audit returns `winner: null` and the v3 latents
fall back to V0, the model-side spec is **unchanged** — the model
implementation is normalisation-independent.

---

## 1. Two variants — data-flow diagrams

### 1.1 Variant A — sequence-only (channel-concat at trunk, no ControlNet)

```
DataLoader batch
├── x_t           (B, 4,  h, w, d)   noisy target latent
├── z_t1pre       (B, 4,  h, w, d)   conditioning modality
├── z_t2          (B, 4,  h, w, d)   conditioning modality
├── z_flair       (B, 4,  h, w, d)   conditioning modality
├── m_brain       (B, 1,  h, w, d)   for the loss (BG/BRAIN regions)
└── m_tumor       (B, 3,  h, w, d)   for the loss (NETC/ED/ET regions)

trunk_input = torch.cat([x_t, z_t1pre, z_t2, z_flair], dim=1)   # (B, 16, h, w, d)

v_pred = trunk(
    x=trunk_input,
    timesteps=t,
    class_labels=class_labels,
    spacing_tensor=spacing,
    # NO ControlNet residuals
)
loss = region_weighted_L1(v_pred, u_target, m_brain, m_tumor, weights=cfg)
```

`m_brain` and `m_tumor` are **only used by the loss** (not by the
network). The network sees only the 16-channel concat at its input.

### 1.2 Variant B — sequence + per-sub-region masks (channel-concat + ControlNet)

```
DataLoader batch
├── x_t, z_t1pre, z_t2, z_flair, m_brain, m_tumor   as in Variant A

trunk_input  = torch.cat([x_t, z_t1pre, z_t2, z_flair], dim=1)   # (B, 16, h, w, d)
cn_cond      = m_tumor                                            # (B,  3, h, w, d)  — NETC, ED, ET

down_res, mid_res = controlnet(
    x=x_t,                              # (B, 4, h, w, d) — noisy latent only
    timesteps=t,
    controlnet_cond=cn_cond,            # (B, 3, h, w, d) — the mask
    class_labels=class_labels,
)
v_pred = trunk(
    x=trunk_input,                      # (B, 16, h, w, d)
    timesteps=t,
    class_labels=class_labels,
    spacing_tensor=spacing,
    down_block_additional_residuals=down_res,
    mid_block_additional_residual=mid_res,
)
loss = region_weighted_L1(v_pred, u_target, m_brain, m_tumor, weights=cfg)
```

The ControlNet's own `x` stays at 4 channels (the noisy latent, not the
expanded concat). The conditioning `cn_cond` is the 3-channel mask. The
existing `MaisiControlNet.forward` signature is unchanged — only its
`conv_in` shape (4 ch) and `cond_embedding` `in_channels` (3 ch) are
rebuilt at init time.

### 1.3 Why both variants share the channel-concat at trunk input

Because the modality latents (T1pre, T2, FLAIR) are **the carrier of
patient-specific information**. They must reach the trunk's first conv
directly, regardless of whether ControlNet is also active for masks.
Routing modality latents through ControlNet's residual-injection path is
exactly what S1 v2 did and what failed (father doc §2, §4 H6). Variant A
removes ControlNet entirely (the dose-response control); Variant B keeps
ControlNet on its designed-for use case (spatial-mask conditioning).

---

## 2. Architecture changes — module-by-module

### 2.1 Trunk `conv_in` expansion (both variants)

**New module**: `src/vena/model/fm/maisi/conv_in_expand.py`

**Public API**:

```python
def expand_conv_in(
    trunk: nn.Module,
    new_in_channels: int,
    *,
    zero_init_new: bool = True,
    keep_original_init: bool = True,
) -> nn.Module:
    """Replace trunk.conv_in with a (new_in_channels)-input convolution.

    The first `original_in_channels` input channels keep their original
    pretrained weights; the additional `new_in_channels − original_in_channels`
    are initialised to zero (if ``zero_init_new=True``) so that the
    expanded trunk is functionally identical to the original at step 0
    when the additional inputs are anything.

    The bias and all other conv parameters are preserved.
    """
```

Behaviour:

1. Read `trunk.conv_in.weight` (shape `(C_out, C_in_old=4, k_d, k_h, k_w)`).
2. Allocate `new_w` of shape `(C_out, C_in_new=16, k_d, k_h, k_w)`.
3. Copy `new_w[:, :C_in_old] ← trunk.conv_in.weight`.
4. If `zero_init_new`: zero-fill `new_w[:, C_in_old:]`; else use Kaiming uniform.
5. Replace `trunk.conv_in` with a fresh `nn.Conv3d(C_in_new, C_out, ...)` carrying the loaded weights.

**Optional warm-up ramp on the new channels' weights** (analogous to
`OutputScaleRampCallback`): a `ConvInRampCallback` multiplies a per-
new-channel scalar buffer `α(step) ∈ [0, 1]` (sigmoid over
`ramp_steps`) into the new channels of the conv at each forward.
Equivalent to having the new channels' inputs gated by α. **Implement as
optional** (`input_concat.ramp_steps: 0` disables, > 0 enables); rampless
behaviour is already correct (zero-init has the same step-0 effect).

**Unit tests** (`tests/model/fm/test_conv_in_expand.py`):

1. `test_expansion_preserves_step0_behaviour` — given a trunk and a
   batch `x_old (B, 4, *)`, the expanded trunk on
   `cat([x_old, zeros(B, 12, *)], dim=1)` must produce
   **bit-identical** output to the original trunk on `x_old`. ε = 1e-7.
2. `test_expansion_zero_init_default` — `zero_init_new=True` means the
   new channels' weight slice has `.abs().max() == 0`.
3. `test_expansion_preserves_bias` — bias is unchanged.
4. `test_ramp_callback_at_step_0_zeros_new_channels` — when the ramp
   callback is active and step=0, the new channels' contribution to
   conv_out is zero regardless of input.
5. `test_ramp_callback_saturates_after_ramp_steps` — at step >> ramp_steps,
   α = 1.0 exactly (or within 1e-4).

### 2.2 ControlNet refactor (Variant B only)

**Edit** `src/vena/model/fm/controlnet/maisi_controlnet.py`:

1. Make `init_from_trunk` **optional** via a constructor flag
   `init_from_trunk: bool = True` (default True for backwards compat;
   Variant B's YAML sets it to False).
2. Rebuild `controlnet_cond_embedding` first conv's `in_channels` to
   match the **shape of `controlnet_cond` after assembly** (i.e. 3 in
   Variant B, derived from `ConditioningAssembler.channels_per_spec`).
   Use random init (Kaiming) for it — there is no pretrained 3-channel-
   mask projection.
3. Keep `zero_init_output_projections()` (the existing zero-init of
   `controlnet_down_blocks.*` and `controlnet_mid_block.*` is what makes
   the ControlNet additive-from-zero, then scale-ramped up via
   `output_scale_ramp`).
4. **No change** to the residual-injection path
   (`down_block_res_samples`, `mid_block_res_sample`). Those still feed
   into the trunk's down-block skip connections and bottleneck.

The existing `MaisiControlNet.forward` signature is unchanged:

```python
def forward(self, x, timesteps, controlnet_cond, class_labels=None, **kwargs):
    ...
```

The `x` argument continues to be the 4-channel noisy latent. Only the
`controlnet_cond` shape changes (was 13, now 3).

**Unit tests** (`tests/model/fm/test_maisi_controlnet_v3.py`):

1. `test_controlnet_v3b_in_channels_match_cond_assembly` — given a 3-
   channel `tumor3` conditioning spec, the built ControlNet's
   `cond_embedding.0.in_channels == 3`.
2. `test_controlnet_v3b_skip_init_from_trunk` — `init_from_trunk=False`
   leaves the cond_embedding at Kaiming init, no trunk weights copied.
3. `test_controlnet_zero_init_output_still_applies` — even with
   `init_from_trunk=False`, output projections are zero-init.
4. `test_controlnet_forward_with_3channel_mask_shape_matches` —
   forward(...) returns down/mid residuals with shapes matching trunk
   expectations.

### 2.3 ConditioningAssembler — `mask:tumor3` spec

**Edit** `src/vena/model/fm/controlnet/conditioning.py`:

Add a new `ConditioningSpec.kind` value:

```python
KIND_VALUES = ("latent", "mask", "prior", "tumor3")   # was ("latent", "mask", "prior")
```

When `spec.kind == "tumor3"`, the assembler reads `batch["m_tumor"]`
(the 3-channel soft mask) and applies the configured downsampler. The
downsampler for `mask:tumor3:identity` is the existing `IdentityDownsampler`
(no resizing — `m_tumor` is already at latent resolution per the
encode-routine attr `mask_downsampler_attrs_json`).

`channels_per_spec` returns **3** for any `tumor3` spec (vs 1 for `mask`).

**Update the conditioning_inputs grammar** to allow `mask:tumor3:identity`:

```python
# in vena/model/fm/controlnet/conditioning.py::ConditioningSpec.from_string
"latent:t1pre"        → latent kind, key "t1pre"
"mask:wt:identity"    → mask kind, key "wt" (legacy, kept for backwards compat)
"mask:tumor3:identity" → tumor3 kind, key "tumor"   # NEW
"mask:netc:identity"  → mask kind, key "netc"       # NEW (alternative — single-channel each)
"mask:ed:identity"    → mask kind, key "ed"         # NEW
"mask:et:identity"    → mask kind, key "et"         # NEW
```

We support **both** forms (`tumor3` and three independent specs) — the
single `tumor3` spec is the preferred default because it's atomic;
three specs are useful for ablations that drop one sub-region.

**Unit tests** (`tests/model/fm/test_conditioning_tumor3.py`):

1. `test_tumor3_assembly_returns_3channel` — `mask:tumor3:identity` on a
   batch with `m_tumor (B, 3, h, w, d)` returns the same tensor with
   `channels_per_spec == [3]`.
2. `test_three_independent_mask_specs_equivalent_to_tumor3` — feeding
   `[mask:netc:identity, mask:ed:identity, mask:et:identity]` produces
   the same final tensor as `[mask:tumor3:identity]`.
3. `test_tumor3_with_unsupported_downsampler_raises` — only `identity`
   is supported; `mask:tumor3:nearest` raises an explicit error.

### 2.4 DataLoader — split tumor mask into the right batch keys

**Edit** `src/vena/model/fm/lightning/datamodule.py` (and the per-cohort
dataset classes under `src/vena/data/cohort/`):

Currently the dataset emits `batch["m_wt"]`. The v3 DataLoader emits:

| batch key | shape | dtype | source |
|---|---|---|---|
| `m_brain` | `(B, 1, h, w, d)` | int8 | `masks/brain_latent` (existing) |
| `m_tumor` | `(B, 3, h, w, d)` | float32 | `masks/tumor_latent` (existing — 3 channels NETC/ED/ET) |
| `m_netc` | `(B, 1, h, w, d)` | float32 | `m_tumor[:, 0:1]` (derived in `collate_fn`) |
| `m_ed`   | `(B, 1, h, w, d)` | float32 | `m_tumor[:, 1:2]` |
| `m_et`   | `(B, 1, h, w, d)` | float32 | `m_tumor[:, 2:3]` |
| `m_wt`   | `(B, 1, h, w, d)` | float32 | derived: `max(m_tumor, dim=1, keepdim=True)` (kept for back-compat with S1 v2 + the loss's WT region) |

The single-channel derivations live in the collate function so callers
of `batch[...]` get them for free. **Do NOT remove `m_wt` from the batch**
— the loss's WT region uses it for ablations that disable per-sub-region
weighting and revert to a single WT weight.

**Unit tests** (`tests/data/test_v3_batch_layout.py`):

1. `test_v3_batch_carries_all_mask_keys` — `m_brain`, `m_tumor`, `m_netc`,
   `m_ed`, `m_et`, `m_wt` all present.
2. `test_m_wt_equals_max_of_m_tumor` — `m_wt == m_tumor.max(dim=1)`.
3. `test_m_tumor_3_channels_in_NETC_ED_ET_order` — verify channel
   ordering by intersecting with the image-H5 BraTS labels on one patient.

---

## 3. Loss layer — region-weighted L1 with easy disable

The CFM velocity loss currently is:

```python
F.l1_loss(v_pred, u_target, reduction="mean")
```

The v3 loss is:

```python
loss_voxel = F.l1_loss(v_pred, u_target, reduction="none")   # (B, 4, h, w, d)
weight = build_region_weight_tensor(
    region_weights, m_brain, m_tumor, threshold=0.5,
    spatial_shape=loss_voxel.shape,
)                                                             # (B, 4, h, w, d), values in [0, ∞)
loss = (loss_voxel * weight).sum() / weight.sum().clamp_min(1e-12)
```

### 3.1 Region definitions

Region masks are built **inside the loss**, not by the dataset. The
inputs are `m_brain (B, 1, h, w, d)` and `m_tumor (B, 3, h, w, d)`;
all derived masks are computed in the loss layer.

Default region definitions (with a soft-mask threshold `τ = 0.5`):

| Region key | Definition | Voxel count (UCSF-PDGM avg) |
|---|---|---|
| `bg` | `m_brain == 0` | ~79 % |
| `brain_not_wt` (aka `bnwt`) | `m_brain == 1 AND m_tumor.max(dim=1) < τ` | ~20 % |
| `netc` | `m_tumor[:, 0] ≥ τ` | ~0.02 % |
| `ed` | `m_tumor[:, 1] ≥ τ` | ~0.05 % |
| `et` | `m_tumor[:, 2] ≥ τ` | ~0.02 % |

Note that NETC/ED/ET are **soft masks** in the source `m_tumor` — the
threshold τ binarises them for the loss. Regions are **disjoint by
construction** when applying `τ = 0.5` to per-class one-hot masks (the
encode routine guarantees one-hot at image resolution; the avg-pool
preserves disjointness when τ ≥ 0.5).

The brain mask is the union (m_brain == 1 covers all of brain_not_wt
∪ NETC ∪ ED ∪ ET). A voxel in tumor is **not** in `brain_not_wt` because
the `m_tumor.max < τ` clause excludes it. The five regions partition the
volume.

### 3.2 Weight construction

```python
def build_region_weight_tensor(
    region_weights: RegionWeights,
    m_brain: torch.Tensor,       # (B, 1, h, w, d), int8 or float
    m_tumor: torch.Tensor,       # (B, 3, h, w, d), float [0, 1]
    *,
    threshold: float = 0.5,
    spatial_shape: torch.Size,
) -> torch.Tensor:
    if not region_weights.enabled:
        return torch.ones(spatial_shape, dtype=torch.float32, device=m_brain.device)
    # ... build per-region disjoint masks, broadcast to 4 channels, multiply by weight ...
```

`RegionWeights` is a Pydantic model:

```python
class RegionWeights(BaseModel):
    enabled: bool = True
    bg: float = 1.0
    brain_not_wt: float = 1.0
    netc: float = 50.0
    ed: float = 50.0
    et: float = 300.0
    # Optional: aggregate "wt" weight, used only when per-sub-region weights are all equal.
    wt: float | None = None    # if non-null, overrides netc/ed/et with this single value
    threshold: float = 0.5
```

**Easy disable**: set `enabled: false` in YAML. The loss collapses to
the standard `F.l1_loss(reduction="mean")` (the weight tensor is all
ones; the normalising `sum() / sum()` reproduces the mean over voxels).

**Easy WT-only weighting**: set `wt: 200.0` and leave
`netc/ed/et` at their defaults. The loss layer detects `wt` is non-null
and uses it for all three sub-regions, recovering the "single WT weight"
ablation.

### 3.3 Reduction contract

`loss.cfm.reduction` is exposed in YAML as `none` | `mean` | `sum`.
- `none`: per-voxel loss tensor; weight applies as above. Final scalar
  is `sum(weight * loss) / sum(weight)`.
- `mean`: standard unweighted mean (only valid if `region_weights.enabled = false`).
- `sum`: unweighted sum (rarely useful).

`reduction: mean` together with `region_weights.enabled: true` is an
explicit config validation error — they conflict.

### 3.4 Backward compatibility

S1 v2's loss path used `reduction: mean` and no region weights. The v3
default `reduction: none + region_weights.enabled: true` reproduces
unweighted L1 when all weights are 1.0 (within ε). Existing S1/S2
checkpoints can be loaded into v3 without re-training (the loss change
is at training time only; checkpoint state_dict is unaffected).

### 3.5 Implementation files

- **Edit** `src/vena/model/fm/controlnet/losses/cfm.py` — extend
  `CFMLoss` to accept a `region_weights: RegionWeights` field; in
  `forward()`, branch on `region_weights.enabled`.
- **Edit** `src/vena/model/fm/controlnet/losses/base.py` — extend
  `LossInputs` to carry `m_brain`, `m_tumor` tensors (the loss layer
  builds the per-region masks from them).
- **Edit** `src/vena/model/fm/lightning/module.py` (training_step) — pass
  `m_brain`, `m_tumor` into `LossInputs`.
- **New** `src/vena/model/fm/controlnet/losses/region_weights.py` —
  `RegionWeights` model + `build_region_weight_tensor()` helper.

### 3.6 Unit tests

`tests/model/fm/test_cfm_region_weighted.py`:

1. `test_region_weights_enabled_false_matches_unweighted_l1` — with
   `enabled=false` (or all-ones weights), the v3 loss equals
   `F.l1_loss(v, u, reduction="mean")` within ε = 1e-6.
2. `test_region_weights_disjoint_partition` — for any (m_brain, m_tumor)
   input, the five region masks (BG, BNWT, NETC, ED, ET) partition the
   volume exactly (sum of masks == 1 everywhere).
3. `test_region_weights_apply_per_voxel` — with weights
   {bg=1, bnwt=1, netc=10, ed=10, et=100} and a controlled
   `(v_pred, u_target)` where the per-voxel error is 1.0 everywhere,
   the weighted loss equals `(1·n_bg + 1·n_bnwt + 10·n_netc + 10·n_ed + 100·n_et) / (n_bg + n_bnwt + 10·n_netc + 10·n_ed + 100·n_et)`.
4. `test_region_weights_wt_override` — setting `wt: 200` (and leaving
   netc/ed/et defaults) makes the three sub-region masks weighted at
   200 each.
5. `test_region_weights_threshold_respected` — soft masks with
   `m_tumor = 0.4` are excluded from NETC/ED/ET when threshold=0.5.

---

## 4. Metric layer — per-region PSNR/MAE/MSE/SSIM

### 4.1 New per-region metrics

**Edit** `src/vena/model/fm/metrics/regions.py` to expose per-region
masks for: `whole`, `bg`, `brain_not_wt`, `wt`, `netc`, `ed`, `et`.

**Edit** `src/vena/model/fm/metrics/image.py` to add per-region variants:

```python
def psnr_per_region(pred, real, m_brain, m_tumor, threshold=0.5,
                    data_range=1.0) -> dict[str, float]: ...
def mae_per_region(pred, real, m_brain, m_tumor, threshold=0.5) -> dict[str, float]: ...
def mse_per_region(pred, real, m_brain, m_tumor, threshold=0.5) -> dict[str, float]: ...
def ssim_per_region(pred, real, m_brain, m_tumor, threshold=0.5) -> dict[str, float]: ...
```

Each returns a dict with keys
`{whole, bg, brain_not_wt, wt, netc, ed, et}`.

`SSIM` per region uses a foreground-only window (the region mask multiplied
into the SSIM kernel) — implementation lives in `metrics/image.py` already;
extend it.

### 4.2 Exhaustive-val CSV columns (additive)

**Edit** `routines/fm/exhaustive_val/engine.py` to emit, per
(cohort, epoch, patient, nfe) row:

```
... (existing whole-volume + WT + BG + NWT columns) ...
psnr_db_et, psnr_db_netc, psnr_db_ed, psnr_db_bnwt,
mae_et, mae_netc, mae_ed, mae_bnwt, mae_whole,
mse_et, mse_netc, mse_ed, mse_bnwt, mse_whole,
ssim_et, ssim_netc, ssim_ed, ssim_bnwt,
n_voxels_et, n_voxels_netc, n_voxels_ed, n_voxels_bnwt, n_voxels_brain
```

The `n_voxels_*` columns let post-hoc analysis weight per-patient
contributions correctly when aggregating.

### 4.3 Per-cohort and per-epoch aggregation

Per cohort `C` and epoch `E`, emit a separate `<run>/exhaustive_val/aggregate.csv`
file with one row per (cohort, epoch, nfe, region):

```
cohort, epoch, nfe, region, n_patients, psnr_mean, psnr_std,
   mae_mean, mae_std, mse_mean, mse_std, ssim_mean
```

This is the file that the figure-rendering and convergence-plotting
scripts read from; producing it once at exhaustive-val time saves
re-computing aggregates downstream.

### 4.4 `best_metric_*` rewire

**Edit** `routines/fm/train/engine.py::_select_best_metric()`:

The current selection key is `f"{best_metric_name}_{best_metric_region}"`
(e.g. `mse_latent_bg`). Extend it to support the new region values
`{et, netc, ed, bnwt}`. For backwards compat keep the existing
`{bg, wt, whole}` paths intact.

When `best_metric_region == "et"` and exhaustive-val is the only source
of truth (validation.every_epochs == 0), the engine reads
`exhaustive_val/epoch_<N>/aggregate.csv` at each cadence epoch and
extracts `psnr_db_et @ nfe == best_metric_nfe` as the score. Higher is
better (mode="max"); the engine flips `ModelCheckpoint.mode` accordingly
(currently hardcoded to "min" for MSE-based metrics).

### 4.5 Unit tests

`tests/model/fm/test_per_region_metrics.py`:

1. `test_psnr_per_region_matches_manual_calc` — controlled prediction
   with known error per region; per-region PSNR matches the manual
   computation (within float-precision).
2. `test_per_region_empty_mask_returns_nan` — when a patient has no ET
   voxels, `psnr_per_region(...)['et'] == float('nan')` and downstream
   aggregation skips NaN.
3. `test_mae_mse_consistency` — `MAE² ≤ MSE`, both > 0.
4. `test_csv_columns_present` — after one exhaustive_val pass on a
   synthetic 2-patient cohort, the emitted metrics.csv has all expected
   columns.

---

## 5. YAML schema and `decision.json` bump

### 5.1 New / changed YAML keys

```yaml
data:
  conditioning_inputs: []    # used by ControlNet (Variant B) only; empty list for Variant A
  # all other data.* keys unchanged

model:
  trunk:
    input_concat:                  # NEW BLOCK
      enabled: bool                #   primary on/off switch
      cond_latents: list[str]      #   e.g. [t1pre, t2, flair]
      cond_masks: list[str]        #   typically []; reserved for "concat the mask too" ablation
      zero_init_new_channels: bool
      ramp_steps: int              #   0 disables ramp (zero-init alone suffices)
      ramp_steepness: float
    # all other trunk.* keys unchanged
  controlnet:
    enabled: bool                  # NEW — primary on/off (Variant A: false; Variant B: true)
    conditioning_inputs: list[str] # consumed only if enabled
    init_from_trunk: bool          # NEW — controls weight init source (Variant B: false)
    output_scale_ramp:             # retained unchanged
      enabled: bool
      ramp_steps: int
      steepness: float

loss:
  cfm:
    weight: float
    reduction: "none" | "mean" | "sum"   # NEW — "none" allowed
    norm: "l1" | "l2"
    region_weights:                # NEW BLOCK
      enabled: bool
      bg: float
      brain_not_wt: float
      netc: float
      ed: float
      et: float
      wt: float | null             # null = use per-sub-region weights; non-null = single WT weight
      threshold: float

training:
  best_metric_name: str            # extended values: psnr_db, mae, mse, ssim
  best_metric_region: str          # extended values: et, netc, ed, bnwt, wt, bg, whole
  best_metric_nfe: int             # selection NFE
  best_metric_mode: "min" | "max"  # NEW — auto-derived if absent (PSNR/SSIM → max; MAE/MSE → min)
  conditioning_dropout_p: float
  conditioning_dropout_keys: list[str]  # extended values: netc, ed, et, tumor3

exhaustive_val:
  emit_per_region_metrics: list[str]   # NEW — e.g. [et, netc, ed, bnwt, whole]
  # all other keys unchanged
```

### 5.2 `decision.json` schema 0.10.0

Bumps from 0.9.0 (S1 v2 schema). Additions to the schema:

```json
{
  "schema_version": "0.10.0",
  "input_concat": {
    "enabled": true,
    "cond_latents": ["t1pre", "t2", "flair"],
    "cond_masks": [],
    "trunk_in_channels_old": 4,
    "trunk_in_channels_new": 16,
    "zero_init_new_channels": true,
    "ramp_steps": 5000,
    "ramp_steepness": 10.0
  },
  "controlnet_enabled": false,    // Variant A
  "controlnet_init_from_trunk": null,
  "controlnet_conditioning_inputs": [],
  "region_weights": {
    "enabled": true,
    "bg": 1.0, "brain_not_wt": 1.0, "netc": 50.0, "ed": 50.0, "et": 300.0,
    "wt": null, "threshold": 0.5
  },
  "best_metric": {
    "name": "psnr_db",
    "region": "et",
    "nfe": 5,
    "mode": "max"
  },
  "exhaustive_val_per_region_metrics": ["et", "netc", "ed", "bnwt", "whole"],
  "normalization_audit_decision_path": "/abs/path/to/artifacts/preflights/normalization_audit/LATEST/decision.json",
  "normalization_variant_id": "V1"   // or "V0" if fallback
}
```

The normalization audit's `decision.json` is **read at training-engine
startup** and its `winner` field cross-checked against the latent H5's
`normalization_variant_id` attr. Mismatch is a hard fail.

### 5.3 Two production configs

Both diff against `picasso_s1_1000ep_fft.yaml` (S1 v2). Full YAMLs live
in `routines/fm/train/configs/runs/`. Their **shared block** (loss,
metrics, training, exhaustive_val) is identical between A and B; only
`model` and `data.conditioning_inputs` differ. The father doc §6.9
shows the diff explicitly — repeat is omitted here.

**Config files to create**:

- `routines/fm/train/configs/runs/picasso_s1_v3a_concat_only_fft.yaml` — Variant A
- `routines/fm/train/configs/runs/picasso_s1_v3b_concat_plus_cn3ch_fft.yaml` — Variant B
- `routines/fm/train/configs/smoke/server3_s1_v3a_concat_only_4ep.yaml` — Variant A smoke
- `routines/fm/train/configs/smoke/server3_s1_v3b_concat_plus_cn3ch_4ep.yaml` — Variant B smoke
- `routines/fm/train/configs/smoke/loginexa_s1_v3a_concat_only_2ep.yaml` — Variant A loginexa smoke
- `routines/fm/train/configs/smoke/loginexa_s1_v3b_concat_plus_cn3ch_2ep.yaml` — Variant B loginexa smoke

The smoke configs use `max_train_subjects: 50` and `max_epochs: 4`.

### 5.4 SLURM launchers

- `routines/fm/train/slurm/runs/launcher_picasso_s1_v3a_fft.sh`
- `routines/fm/train/slurm/runs/launcher_picasso_s1_v3b_fft.sh`

Both use the existing shared worker script
(`routines/fm/train/slurm/runs/worker_picasso_s1_v3.sh` — copy of the
S1 v2 worker with `train.cli` pointing at the new YAMLs). 4× A100,
7-day wallclock, patience=250 epochs.

---

## 6. File-by-file edit plan

### 6.1 Files to **create**

| Path | Purpose |
|---|---|
| `src/vena/model/fm/maisi/conv_in_expand.py` | trunk `conv_in` channel-expansion utility |
| `src/vena/model/fm/maisi/conv_in_ramp_callback.py` | (optional) ramp callback on new conv_in channels |
| `src/vena/model/fm/controlnet/losses/region_weights.py` | `RegionWeights` model + `build_region_weight_tensor` |
| `src/vena/model/fm/controlnet/downsample/tumor3.py` | identity downsampler for `mask:tumor3` (or extend existing identity) |
| `tests/model/fm/test_conv_in_expand.py` | conv_in expansion unit tests |
| `tests/model/fm/test_maisi_controlnet_v3.py` | 3-channel ControlNet unit tests |
| `tests/model/fm/test_conditioning_tumor3.py` | tumor3 assembler unit tests |
| `tests/data/test_v3_batch_layout.py` | DataLoader v3 batch layout tests |
| `tests/model/fm/test_cfm_region_weighted.py` | region-weighted L1 unit tests |
| `tests/model/fm/test_per_region_metrics.py` | per-region metrics unit tests |
| `routines/fm/train/configs/runs/picasso_s1_v3a_concat_only_fft.yaml` | Variant A production |
| `routines/fm/train/configs/runs/picasso_s1_v3b_concat_plus_cn3ch_fft.yaml` | Variant B production |
| `routines/fm/train/configs/smoke/*` | 4× smoke YAMLs (server3 + loginexa, per variant) |
| `routines/fm/train/slurm/runs/launcher_picasso_s1_v3a_fft.sh` | SLURM launcher A |
| `routines/fm/train/slurm/runs/launcher_picasso_s1_v3b_fft.sh` | SLURM launcher B |
| `routines/fm/train/slurm/runs/worker_picasso_s1_v3.sh` | Shared worker |

### 6.2 Files to **edit**

| Path | What to change |
|---|---|
| `src/vena/model/fm/controlnet/maisi_controlnet.py` | add `init_from_trunk: bool` flag (default True for back-compat); allow `cond_embedding` in_channels to be derived from assembler `channels_per_spec` |
| `src/vena/model/fm/controlnet/conditioning.py` | add `tumor3` `kind`; parser support for `mask:tumor3:identity`, `mask:netc:identity`, `mask:ed:identity`, `mask:et:identity`; `channels_per_spec` returns 3 for `tumor3` |
| `src/vena/model/fm/lightning/module.py` | (a) build `trunk_input` via `torch.cat` when `input_concat.enabled`; (b) skip ControlNet branch when `controlnet.enabled == False`; (c) pass `m_brain`, `m_tumor` to `LossInputs`; (d) handle warm-up ramp callback registration |
| `src/vena/model/fm/lightning/datamodule.py` | derive `m_netc`, `m_ed`, `m_et`, `m_wt` in `collate_fn`; emit all six mask keys |
| `src/vena/data/cohort/*` (each cohort dataset class) | ensure `masks/tumor_latent` is read into `batch["m_tumor"]` (verify; should already be the case for all v3 latent H5s) |
| `src/vena/model/fm/controlnet/losses/cfm.py` | add `region_weights` field; in `forward`, branch on `region_weights.enabled` and call `build_region_weight_tensor` |
| `src/vena/model/fm/controlnet/losses/base.py` | extend `LossInputs` with `m_brain`, `m_tumor` tensors |
| `src/vena/model/fm/metrics/regions.py` | expose `netc`, `ed`, `et`, `bnwt` region masks |
| `src/vena/model/fm/metrics/image.py` | add `psnr_per_region`, `mae_per_region`, `mse_per_region`, `ssim_per_region` |
| `routines/fm/exhaustive_val/engine.py` | emit new CSV columns; emit `aggregate.csv` |
| `routines/fm/train/engine.py` | (a) validate v3 corpus (`corpus_version == "v3"`); (b) cross-check normalization audit decision path; (c) bump `decision.json` schema to 0.10.0; (d) extend `_select_best_metric` to support ET; (e) flip `ModelCheckpoint.mode` based on `best_metric_mode` |
| `routines/fm/train/exceptions.py` | add `ConfigValidationError` for the `reduction: mean` + `region_weights.enabled: true` conflict |
| `vena.common` (if any new primitives need to be cross-cut) | re-exports as needed |

### 6.3 Files to **NOT touch**

- `vena.common.percentile_normalise` — the normalisation audit may
  extend this, but model implementation is normalisation-independent.
- `vena.model.fm.sampler.rflow` — the rectified-flow scheduler stays
  unchanged.
- `vena.model.fm.ema.*` — EMA logic is unchanged.
- All existing tests under `tests/model/fm/*` that are passing must
  continue to pass. New tests are additive.

---

## 7. Test plan

### 7.1 Local fast suite (must pass before any GPU work)

`~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -v`

All new unit tests from §2–§4 plus the existing fast suite. Estimated
wall-clock: 2 minutes. Must be green before any smoke run.

### 7.2 Server3 cuda:0 smoke runs (~30 min each, both variants)

Per the `server3` skill:

```bash
~/.claude/skills/server3/SKILL.md  # follow launch-mode workflow
```

1. Smoke A: `routines.fm.train.cli routines/fm/train/configs/smoke/server3_s1_v3a_concat_only_4ep.yaml`
2. Smoke B: `routines.fm.train.cli routines/fm/train/configs/smoke/server3_s1_v3b_concat_plus_cn3ch_4ep.yaml`

Acceptance per smoke:

- Run completes (4 epochs, ≤ 30 min wall-clock on cuda:0).
- `metrics/train_step.csv` populated (header + ≥ 100 rows; no header-only).
- `metrics/train_epoch.csv` has 4 rows + header.
- `decision.json` schema 0.10.0 with all v3 fields.
- `logs/train.log` ends with `"FM-train completed"` and contains no
  `Traceback` since the last `LOG opened` marker.
- `exhaustive_val/epoch_*` (whichever cadence emitted): `metrics.csv`
  has the new per-region columns populated.

### 7.3 Loginexa V100 smoke (~25 min each, OPTIONAL)

Per the `test-picasso-loginexa` skill. Validates that the v3 config
will execute on Picasso's loginexa node before committing to a full
queue. The 2-epoch smoke confirms (a) cuda init works, (b) MAISI VAE
loads under the `vena-v100` env, (c) the new module imports cleanly.

Optional — only run if a Picasso queue position is uncertain.

### 7.4 Full Picasso launch (variant A and B in parallel)

Each is a 7-day, 4× A100 SLURM job. Submit both at the same time; the
SLURM queue handles scheduling.

```bash
sbatch routines/fm/train/slurm/runs/launcher_picasso_s1_v3a_fft.sh
sbatch routines/fm/train/slurm/runs/launcher_picasso_s1_v3b_fft.sh
```

Each writes to a separate run directory under `experiments/`. Monitor
via `/loop monitor server3 run at <run_dir>` if mirrored to server3, or
via direct SSH polling of `metrics/train_epoch.csv`.

### 7.5 Acceptance criteria for v3 sign-off

Per the father doc §6.10 target metrics. Concretely:

| Metric (NFE=5, ep 200, UCSF-PDGM aggregate) | Variant A target | Variant B target |
|---|---:|---:|
| PSNR_whole | ≥ 29.0 | ≥ 29.0 |
| PSNR_BG    | ≥ 31.0 | ≥ 31.0 |
| **PSNR_ET (load-bearing)** | ≥ 18.0 | **≥ 20.0** |
| Visible enhancement in figure_best ≥ 5 / 5 patients | required | required |

The **B − A PSNR_ET delta** is the ablation result; it must be ≥ 2.0 dB
to justify retaining ControlNet for the S2 vessel-prior extension. If
B − A < 0.5 dB, the proposal's vessel-prior plan should be revisited
(maybe channel-concat the vessel prior too).

---

## 8. Smoke + production launch sequence

The recommended order, assuming the normalisation sibling spec has
landed (or fallen back to V0):

1. **Land the code changes** (§2–§5). Run the local fast suite. All
   green.
2. **Run server3 smoke A and smoke B** (§7.2). Both must complete with
   the acceptance criteria. Iterate on any bugs surfaced.
3. **(Optional) Run loginexa smoke A** (§7.3) to validate the Picasso
   environment. Skip if the team has high confidence in the Picasso
   path.
4. **Submit both Picasso jobs** (§7.4). They run in parallel for
   ~2 days each (target ~200 epochs).
5. **Read the first cadence's PSNR_ET** off the exhaustive-val output
   at epoch 25, 50, 75. If PSNR_ET is not trending upward by epoch 50
   (still ≤ 14 dB), STOP the run and investigate; do not let it burn
   the full patience window.
6. **At epoch 200** (or earlier convergence), generate the A vs B
   comparison report. The B − A PSNR_ET delta is the load-bearing
   number. The associated `figure_best` panels are the qualitative
   evidence.
7. **Promote the winner** to the S1 baseline; archive the loser as the
   ablation row. The S3 LPL programme proceeds from the winner.

---

## 9. Visualisations and tables required for v3 sign-off

Beyond the per-epoch `metrics.csv` and `aggregate.csv` automatically
emitted by the trainer, the following are required for the v3
go/no-go decision:

### 9.1 Convergence curves (per variant)

Plot per cohort, NFE=5:

- PSNR_whole, PSNR_BG, PSNR_ET, PSNR_NETC, PSNR_ED, PSNR_BNWT over
  epochs (one line per metric).
- Mark epoch 100, 200 (target check-in points).
- Horizontal target lines at PSNR_ET=18 (A) and 20 (B).

One figure per variant; vertical-stacked figures for direct comparison.

**Acceptance**: PSNR_ET trajectory is monotonically increasing (or at
least non-decreasing over a 50-epoch window) by epoch 50.

### 9.2 Side-by-side figure_best at epoch 200

For 5 representative val patients (one per cohort): a 3-row × 6-column
grid:

- Row 1: Real T1c (3 slices)
- Row 2: Variant A prediction at NFE=5 and NFE=200
- Row 3: Variant B prediction at NFE=5 and NFE=200

Visual inspection: does Variant B show clearer enhancement than Variant
A? If yes, supports B − A > 0; if visually similar, supports B − A ≈ 0.

### 9.3 A vs B head-to-head table

| Cohort | n | A PSNR_ET (NFE=5) | B PSNR_ET (NFE=5) | Δ | A PSNR_BG | B PSNR_BG | A vis. enhance ≥ 1 patient | B vis. enhance |
|---|---|---|---|---|---|---|---|---|
| UCSF-PDGM | 7 | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | |

**Acceptance**: at least one cohort shows B − A ≥ 2 dB on PSNR_ET to
declare B the winner; otherwise A wins (saves inference cost).

### 9.4 Per-region loss decomposition at epoch 50, 100, 200

For both variants, plot the **fraction of total loss** contributed by
each region (BG, BNWT, NETC, ED, ET) at three epoch checkpoints. The
expected pattern:

- Epoch 50: ET fraction ~10–15 % (the region weighting is working).
- Epoch 200: ET fraction drops as the model converges on it (becomes
  comparable in magnitude to BNWT). If ET stays > 15 % at epoch 200,
  the weighting may be too aggressive (model not converging on ET).
  If ET drops to < 2 % by epoch 50, the weighting may be too weak.

### 9.5 §6.5 healthy-control diagnostic (Variant B only)

Per proposal §6.5 / `preflights/shortcut_diag`: if the protocol is
feasible, run Variant B on healthy-control T1pre + T2 + FLAIR with all
three mask channels set to zero. **Acceptance**: predicted T1c shows
no enhancement (false-positive enhancement volume ≈ 0 mL).

If protocol_feasible is false (no control cohort yet), defer this to
post-acquisition.

---

## 10. Open questions and explicit non-goals

### 10.1 Open question

- The `input_concat.ramp_steps` value (default 5000) is heuristic from
  the prior `output_scale_ramp`. The zero-init alone makes step-0
  behaviour correct, so the ramp may be unnecessary. Recommendation:
  default `ramp_steps: 0` for the first round; if the trunk's
  early-training dynamics show instability (`grad_norm_trunk_preclip`
  spikes in the first 500 steps), enable the ramp at `5000`.

### 10.2 Explicit non-goals

- **No PEFT / LoRA changes.** Both variants are full-fine-tune (FFT).
  The LoRA path (`project_peft_lora_landed`) is orthogonal; LoRA-on-v3
  is a separate ablation.
- **No SWAN integration.** SWAN as a vessel-prior source is the S2
  programme's scope, not v3.
- **No LPL changes.** The S3 LPL programme (per `decoder_perceptual_loss_s3*`)
  starts from the v3 winner as its warm-start point. The LPL
  implementation is unchanged.
- **No augmentation changes.** v3 uses the same offline augmentation
  (v0–v4) as S1 v2; the augmented latents are re-encoded under the v3
  normalisation (sibling spec §5).

---

## 11. References

- Father: `.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md`
- Sibling: `.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md`
- Project rules:
  - `.claude/rules/coding-standards.md`
  - `.claude/rules/preflight-pattern.md`
  - `.claude/rules/extensibility.md`
  - `.claude/rules/model-coding-standards.md`
- Skills:
  - `~/.claude/skills/server3/SKILL.md`
  - `~/.claude/skills/test-picasso-loginexa/SKILL.md`
  - `~/.claude/skills/picasso-sbatch/SKILL.md`
- Memory:
  - `project_lp_contrastive_v04` — existing region-weighted infrastructure
  - `project_brain_latent_encoding` — `masks/brain_latent` and `masks/tumor_latent` structure
  - `reference_picasso_transfer_route` — server3 ↔ picasso transfer policy
  - `reference_icai_server` — server3 conventions

---

*End of model implementation spec. The two sibling specs (this one and
the normalization audit) together cover everything needed to land the
v3 recipe end-to-end.*
