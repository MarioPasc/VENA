# Patches applied to `src/external/lddpm_3d/upstream/`

Patches are applied in-place at vendoring time. Runtime monkey-patching is
forbidden (skill anti-pattern 2).

## None at vendoring time

The two vendored files (`train_ddpm.py`, `test_ddpm_t1_flair_final.py`) are
**reference-only**. The VENA wrapper at
`src/vena/competitors/lddpm_3d/` does not import either of them — it
rebuilds the model from MONAI primitives directly, mirroring the
T1C-RFlow wrapper policy (`src/external/t1c_rflow/PATCHES.md`). The
upstream files are kept verbatim so future readers can audit the paper's
own training and inference loops.

## What is NOT patched

- Model construction (`instantiate(cfg["diffusion_def"], cfg)` in
  `train_ddpm.py:109`) — unchanged. The wrapper does not call it.
- DDPM scheduler kwargs (`train_ddpm.py:113-119`) — unchanged. The wrapper
  mirrors them via MONAI's `DDPMScheduler` constructor with identical
  arguments.
- Loss form (`train_ddpm.py:170`, `F.mse_loss(noise_pred, noise)`) — unchanged.
  The wrapper mirrors it verbatim.

## Contingencies (not active by default)

### C1 — DDPM `beta_start` discrepancy at inference

`test_ddpm_t1_flair_final.py:125` sets `beta_start=0.0005` while
`train_ddpm.py:115` sets `beta_start=0.0015`. The two are inconsistent;
the wrapper inference path uses the **training** value (`0.0015`) since a
sampler that does not match the training schedule produces a different
noise profile than the model was trained under. If a reviewer requests
strict code reproduction at inference time, the YAML toggle
`hyperparams.ddpm_beta_start_inference: 0.0005` reverts to the upstream
test-time value as an ablation row.

## Verification

After vendoring (no patch applied):

1. Server-3 smoke (4 ep × 1 patient/cohort) completes in ≤ 10 min.
2. The wrapper builds the DDPM scheduler with the training-time betas
   (`beta_start=0.0015, beta_end=0.0195, schedule="scaled_linear_beta",
   clip_sample=False`).
3. The wrapper builds the U-Net at the **paper-faithful** channel set
   (`[128, 128, 256]`, 3 levels, 2 res-blocks, no attention) — same as
   the T1C-RFlow wrapper, isolating the scheduler/loss as the only
   delta vs. T1C-RFlow.
