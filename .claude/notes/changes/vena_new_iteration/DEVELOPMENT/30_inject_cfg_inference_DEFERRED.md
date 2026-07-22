# 30 — Inject: CFG-at-inference + noise-level output_scale  (DEFERRED — NOT this iteration)

**Track/Wave/Deps.** INJECT · **Wave 3 (DEFERRED)** · deps: 20. **Do NOT run this in the T-13/T-06 headline.**
Build only when the orchestrator explicitly schedules the guidance ablation; ship it OFF by default so the code
exists when gated in. Design authority: **A.8-§8** (both levers are FP-risk-gated ablations).

## Objective
Two inference-time levers, each behind a flag, each defaulting to a no-op: **(a) classifier-free guidance** on the
`[WT,NETC]` conditioning (currently *not implemented*), and **(b) noise-level-dependent `output_scale`** gating.
Both are gated on the §6.5 false-positive-enhancement rate (T-14), not on PSNR_ET alone — over-guidance
hallucinates enhancement, which is unsafe for a Gd replacement.

## Read and verify first
- `A.8-§8` in the design doc; the guidance literature: Kynkäänniemi 2024 (guidance-interval, arXiv:2404.07724),
  Sadat 2024 (APG anti-saturation, arXiv:2410.02416), Karras 2024 (autoguidance, arXiv:2406.02507).
- `src/vena/model/fm/inference/euler.py` (`EulerSampler`) + `routines/fm/exhaustive_val/engine.py` (sampler
  construction; the `input_img_size_numel` plumbing).
- Task 20's conditioning (`m_wt_soft`, `m_netc`); the training-time `conditioning_dropout_p` (the uncond branch is
  the zeroed mask — reuse it as the CFG null).

## Files to create / modify
```
MODIFY src/vena/model/fm/inference/euler.py       # optional guidance_scale + interval; noise-level output_scale gate
MODIFY routines/fm/exhaustive_val/engine.py       # plumb guidance params (OFF by default)
```

## Interface & contract
```python
# EulerSampler.sample(..., guidance_scale: float = 1.0, guidance_interval: tuple[float,float] | None = None,
#                      apg: bool = False, output_scale_schedule: Callable[[float], float] | None = None)
```
- **CFG**: `v = v_uncond + guidance_scale·(v_cond − v_uncond)`, uncond = zeroed conditioning (the training dropout
  null). `guidance_scale=1.0` ⇒ **exactly the current cond-only path** (no-op).
- **Guidance interval**: apply guidance only when α ∈ `guidance_interval` (else `guidance_scale=1`), per Kynkäänniemi.
- **APG**: decompose the CFG update into parallel/orthogonal components, down-weight the parallel (saturation)
  component (Sadat). Flag-gated.
- **Noise-level `output_scale`**: multiply the ControlNet `output_scale` at inference by
  `output_scale_schedule(α)` (default `None` ⇒ constant 1.0), a window peaking at α∈[0.3,0.7].

## Acceptance criteria
1. **All defaults are no-ops**: `guidance_scale=1.0`, `guidance_interval=None`, `apg=False`,
   `output_scale_schedule=None` → sampler output **identical** to the pre-change path (assert_allclose).
2. `guidance_scale>1` extrapolates cond vs uncond (a synthetic 2-model check reproduces the CFG formula exactly).
3. Guidance interval restricts to the α-band (outside the band the effective scale is 1.0).
4. Noise-level schedule multiplies `output_scale` by the expected factor at a given α.

## Tests (`tests/model/fm/test_cfg_inference.py`; `pytestmark = pytest.mark.fm`; CPU with stub velocity fields)
- **no-op defaults (load-bearing)**: stub cond/uncond velocity → default path == cond-only path exactly.
- **CFG formula**: `guidance_scale=g` → output == `v_uncond + g·(v_cond−v_uncond)` on synthetic vectors.
- **interval**: with an interval, α inside → guided, α outside → unguided (scale 1).
- **APG**: on a synthetic pair, APG reduces the parallel-component magnitude vs plain CFG at the same g.
- **output_scale schedule**: `schedule(α)` multiplies the buffer as expected at α∈{0.1,0.5,0.9}.

## Do NOT touch
`src/vena/segmentation/*`; the T-13/T-06 run YAMLs (this lever is not in the headline); `grad_safe.py` numerics.

## Report format
Report the no-op-default identity residual (max abs), the CFG-formula check, the interval behaviour, import-
isolation proof, ruff-clean, `STATUS`. State clearly in the report: **this is deferred infrastructure, not part of
the iter-8 headline.**
