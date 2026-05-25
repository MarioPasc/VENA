---
name: test
description: Run pytest tests for the VENA project
---

Run the following tests and report results:

`~/.conda/envs/vena/bin/python -m pytest $ARGUMENTS -v --tb=short`

If `$ARGUMENTS` is empty, run the safe default (excludes slow and GPU tests):

`~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -v --tb=short`

## Marker shortcuts

| Marker | Meaning |
|---|---|
| `unit` | Fast, no I/O, no GPU |
| `preflight_maisi` | MAISI-V2 VAE audit components (reconstruction, latent stats, modality-wise PSNR) |
| `preflight_vessel` | Vesselness QC (Frangi / Jerman / nnU-Net Dice + AHD vs hand-labels) |
| `preflight_aug` | Augmentation pipeline (latent composability, KS audit) |
| `fm` | Flow-matching training/inference components |
| `controlnet` | ControlNet conditioning branch (mask injection, skip connections) |
| `gpu` | Requires CUDA |
| `slow` | Wall-clock > 30 s |

Combine with: `-m "preflight_maisi and not slow"`, `-m "fm and gpu"`, `-m "unit and not gpu"`.

## Report format

Report only:
- Total tests collected
- Pass / fail / skip counts
- For each failure: test ID + first 5 lines of traceback + minimal repro hint
- Wall-clock time

Do not dump full pytest output unless explicitly requested.
