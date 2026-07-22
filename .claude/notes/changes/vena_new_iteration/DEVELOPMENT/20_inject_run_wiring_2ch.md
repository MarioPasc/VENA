# 20 — Inject: serve cached soft masks + 2-ch [WT,NETC] wiring + v3a-resume T-13 oracle run

**Track/Wave/Deps.** INJECT · **Phase-1** · deps: 19 (the cached `masks/tumor_latent_soft`; may be developed in
parallel and integrated at the run step). Owns the DataModule mask-serving + a conditioning-spec wiring block + a
run/smoke YAML. **NO segmenter required** — the T-13 oracle reads the GT-derived soft cache.

## Objective
Make the v3a-warm-start **fresh `[WT,NETC]` ControlNet** run buildable and runnable on the **oracle** soft mask
(T-13): serve the cached soft `[WT,NETC]` selected by `data.mask_source`, wire the **two-spec** conditioning
correctly, and author the run + loginexa-smoke YAMLs. Design authority: Part A / **A.5, A.8-§1/§3/§4/§5/§6**.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (ControlNet contract; v3a config; grid; warm-start caveat).
- Task 19's cache group `masks/tumor_latent_soft (N,2,60,60,40)` (channel 0 = WT_soft, 1 = NETC_soft).
- `src/vena/model/fm/lightning/data.py` (mask serving), `controlnet/` (assembler/specs/downsamplers),
  `lightning/module.py` (`_trunk_forward`), `maisi/maisi_controlnet.py` (`conditioning_in_channels`,
  `controlnet_cond_embedding`).
- `routines/fm/train/configs/runs/picasso_s1_v3a_concat_only_fft.yaml` (base) + a v3b ControlNet YAML (block shape).

## Files to create / modify
```
MODIFY src/vena/model/fm/lightning/data.py     # serve m_wt_soft / m_netc_soft from masks/tumor_latent_soft (or _pred),
                                               #   selected by data.mask_source ∈ {oracle_soft, predicted}
MODIFY src/vena/model/fm/controlnet/...         # ensure the wt_soft/netc_soft keys resolve as specs (if needed)
CREATE routines/fm/train/configs/runs/picasso_ref_v1_v3a+cn[WT,NETC]_fft.yaml   # T-13 oracle run
CREATE routines/fm/train/configs/smoke/loginexa_v3a+cn[WT,NETC]_2ep.yaml        # smoke
```

## Interface & contract
- **Serving**: `data.mask_source ∈ {oracle_soft, predicted}` selects the H5 group (`masks/tumor_latent_soft` vs
  `masks/tumor_latent_pred`); the DataModule serves `batch["m_wt_soft"]` = group[:,0:1] and
  `batch["m_netc_soft"]` = group[:,1:2], each `(1,60,60,40)`. **Fallback** (if the soft cache is absent): derive
  `m_wt_soft = clip(Σ m_tumor,0,1)` on the fly and log a WARNING (so a run never silently trains on a missing
  mask). The oracle and predicted paths differ **only** by `mask_source` — the swap guarantee (task 19) end-to-end.
- **Conditioning**: `model.controlnet.conditioning_inputs: [mask:wt_soft:identity, mask:netc_soft:identity]` →
  assembler mask-part `total_channels == 2` (two 1-ch specs, NOT one 2-ch key — A.8-§4), hint-net
  `conditioning_in_channels == 2`. **MASK-ONLY** `controlnet_cond` (no latents → homogeneous [0,1], A.8-§3).
- **Run YAMLs** (T-13 oracle, **5-job matrix decided 2026-07-22**): all share `run.resume_from: <v3a run_id>`
  (WARM_START), `controlnet.enabled: true`, `init_from_trunk: true`, `output_scale_ramp: {enabled:true,
  ramp_steps:5000, steepness:10.0}`, `input_concat` from v3a (16-ch conv_in), `rflow.use_timestep_transform: true`,
  `data.mask_source: oracle_soft`, LR = **linear-warmup→cosine** (`warmup_steps:1000`, `scheduler:cosine`), and
  **EarlyStopping patience ≈ 400–500** (the harder objective transiently *raises* `train/total_epoch` — patience 250
  risks a premature stop; keep every epoch checkpoint). They differ only in trunk policy × `loss.region_weights`:
  - **J0** — `trunk.trainable: false` (freeze), `region_weights:{brain:1,wt:1}` — ControlNet-only floor; **no trunk
    EMA needed** (sidesteps A.8-§5). Author J0's YAML here + the loginexa smoke.
  - **J1–J4** — `trunk.trainable: true` (joint-low-LR + trunk-EMA), `region_weights:{brain:1,wt:1|5|10|20}` — the
    ceiling + WT-up-weight sweep; **need v3a's `trunk_ema_snapshot.pt`** on Picasso. Siblings of J0 differing by two
    keys; `region_weights` comes from task 21.
- **base_img_size_numel**: leave v3a's value; **add a YAML comment flagging the `(60,60,40)` vs `129024` mismatch
  (A.8-§6)** and raise it in the report — do NOT silently "fix" the timestep-transform reference.

## Acceptance criteria
1. With `mask_source: oracle_soft`, `batch["m_wt_soft"]`/`batch["m_netc_soft"]` present `(1,60,60,40)` ∈ [0,1] and
   read from `masks/tumor_latent_soft`; `NETC_soft ≤ WT_soft`.
2. Assembler from `[mask:wt_soft:identity, mask:netc_soft:identity]` → mask-part `total_channels == 2`; ControlNet
   `conditioning_in_channels == 2`.
3. The run YAML config-validates; a CPU/loginexa smoke **builds** the model and runs **2 optimiser steps** without
   shape errors.
4. **Step-0 identity (P1)**: with `output_scale=0`, ControlNet residuals are all-zero → trunk-with-CN output ==
   plain-trunk (v3a) output (assert_allclose).

## Tests (`tests/model/fm/test_soft_mask_serving_2ch.py`; `pytestmark = pytest.mark.fm`; CPU where possible, GPU build `gpu`)
- **serving by source**: a synthetic latent H5 with `masks/tumor_latent_soft` → `mask_source: oracle_soft` serves
  the soft channels; missing group → fallback clip-sum with a logged WARNING (not a silent zero).
- **two-spec channel count**: `[mask:wt_soft:identity, mask:netc_soft:identity]` → mask-part `total_channels == 2`;
  a single 2-ch key is asserted-against (A.8-§4).
- **step-0 identity**: `output_scale=0` → assembled residuals zero → CN output == no-CN output.
- **config validity**: `from_yaml(run_yaml)` succeeds; `mask_source` + `controlnet_conditioning_inputs` round-trip
  into `decision.json`.

## Do NOT touch
`src/vena/segmentation/*`; `maisi/grad_safe.py` numerics; the v3a config file (copy, don't edit).

## Report format
Readback run-YAML path, the served soft-mask range, mask-part `total_channels`, the step-0 identity residual (max
abs), the `base_img_size_numel` mismatch as an open QUESTION, import-isolation proof, ruff-clean, `STATUS`.
