# Patches applied to `src/external/dit_3d/upstream/`

Patches are applied in-place at vendoring time. Runtime monkey-patching is
forbidden (skill anti-pattern 2).

## P1 — Cosmetic: silence stdout `print` at model init

**File**: `upstream/dit3d.py`
**Why**: `dit3d.py:312` prints `grid_size: ...` from
`get_3d_sincos_pos_embed`, which is called inside `DiT.initialize_weights`
during every model instantiation. The stray line pollutes the rich-formatted
training log and reads as a warning in our smoke runs. Pure cosmetic; numerics
unchanged.

**Patch**:

```diff
 def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
     """
     grid_size: tuple of ints, (gx, gy, gz)
     """
-    print('grid_size:', grid_size)
+    # VENA patch: drop the debug `print('grid_size:', ...)` from upstream — it
+    # pollutes stdout at model init and produces spurious lines in our rich
+    # training logs. Cosmetic only; numerics unchanged.
     gx, gy, gz = grid_size
```

## What is NOT patched

- DiT architecture (`DiT`, `DiTBlock`, `PatchEmbed3D`, `FinalLayer`,
  `TimestepEmbedder`, `LabelEmbedder`) — preserved verbatim.
- Positional embedding helpers (`get_3d_sincos_pos_embed_from_grid`,
  `get_1d_sincos_pos_embed_from_grid`).
- DiT model-size aliases (`DiT_B_4`, `DiT_S_4`, etc.) at the bottom of
  `dit3d.py`.
- `dit3d_wrapper.py` — preserved verbatim.
- `test_dit.py` — preserved verbatim (reference-only; VENA invokes
  `vena.competitors.dit_3d.inference` instead).

## Contingencies (not active by default)

### C1 — timm < 0.9 compatibility (not currently triggered)

If a future env downgrade breaks `from timm.layers import to_2tuple`, the
import lives in `dit3d.py:16` but `to_2tuple` is **unused** inside the file
(`PatchEmbed3D` defines its own patch logic). Replace with `from timm.layers
import to_2tuple as _to_2tuple_unused` and document here, or drop the import
outright. Until the env breaks, do not touch — the principle is "don't
patch what isn't broken."

## Verification

After applying P1, the smoke run must:

1. Build the DiT-B/4 model without printing `grid_size:` to stdout.
2. Show the same train-loss trajectory as a control build that re-applies
   the print (run side-by-side once to confirm numerics drift = 0).
3. Server-3 smoke (4 ep × 1 patient/cohort) completes in ≤ 10 min.
