# Patches applied to `src/external/t1c_rflow/upstream/`

**None.** The upstream code targets torch ≥2.1 and modern MONAI (≥1.5) and runs
unmodified on VENA's `vena` conda env (torch 2.12+cu130 on A100, torch 2.7.1+cu126
on V100, torch 2.5.1+cu121 on server-3 RTX 4090).

The VENA wrapper does not import from `upstream/`. Patches would only become
necessary if the wrapper started invoking upstream files directly — at which point
this document gains content.

## What if a future patch IS needed?

Same convention as `src/external/pgan_cgan/PATCHES.md`:

1. Apply the patch in-place under `upstream/`; never monkey-patch at runtime.
2. Add a section here with a code-block diff and a one-line "why".
3. Re-run the server-3 smoke (4 ep, 1 patient/cohort) to verify no numerics drift.
4. If numerics drift, bisect the patch.

## What is NOT patched

- Model architecture, scheduler config, loss form, optimiser, training schedule —
  all preserved verbatim from upstream at SHA `fc8314f60d877f9ee55996f960f89b17b269200f`.
