# 11 — Segmentation models (BSF-SwinUNETR Arm A/B + SegResNet Arm C)

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/models/` only.

> **🔴 TC, not WT (2026-07-22).** The 2-channel output is `[TC, NETC]` — channel 0 = **tumour core (TC = NETC+ET)**,
> edema EXCLUDED (`TargetConfig.tumor_region="tc"`); `TC−NETC = ET` = the enhancing region. The architecture is
> unchanged (2 soft channels), but the target/eval is TC not WT — and TC is harder to segment (see task 15's G-SEG note
> + `[[project_channel0_tumor_core_not_wt]]`). BSF-SwinUNETR's tumour-aware SSL (BraTS) suits the harder ET/TC boundary.

## Objective
Provide the three backbone arms as registry-registered builders returning a `torch.nn.Module` mapping
`(B, 3, H, W, D) → (B, 2, H, W, D)` logits `[TC, NETC]` (+ deep-supervision heads when configured). **BSF-SwinUNETR
is primary** (Arm A BraTS-SSL, Arm B UKB-SSL); **SegResNet is Arm C**, forked from the existing
`src/vena/validation/downstream_seg.py` 4-input model (drop the T1c input → 3-input). Design authority: Part B.a,
B.f-§1.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (BSF facts; checkpoints; `[verify]` the BSF ckpt paths — **if absent, report BLOCKED,
  do not guess a URL**).
- `src/vena/validation/downstream_seg.py` — the existing 4-input SegResNet loader (Arm C forks this; also confirm
  whether it already loads a usable BraTS segmenter you can reuse for wiring).
- MONAI `SwinUNETR` + `SegResNet` signatures in the installed monai version (`feature_size`, `in_channels`,
  `out_channels`, deep-supervision options).
- `models/registry.py` from task 10.

## Files to create
```
src/vena/segmentation/models/bsf_swinunetr.py   # Arm A + Arm B builders, BrainSegFounder encoder init
src/vena/segmentation/models/segresnet.py       # Arm C builder (fork downstream_seg 4→3 input)
```
(Register each via `@register_segmentation_model(...)` at import; ensure `models/__init__.py` imports them so
registration fires.)

## Interface & contract
```python
@register_segmentation_model("bsf_swinunetr_brats")   # Arm A
def build_bsf_swinunetr_brats(cfg: ModelConfig) -> nn.Module: ...
@register_segmentation_model("bsf_swinunetr_ukb")      # Arm B
def build_bsf_swinunetr_ukb(cfg: ModelConfig) -> nn.Module: ...
@register_segmentation_model("segresnet")              # Arm C
def build_segresnet(cfg: ModelConfig) -> nn.Module: ...
```
- SwinUNETR `feature_size=cfg.feature_size` (48), `in_channels=3`, `out_channels=2`, spatial_dims=3. Load the BSF
  **encoder-only** SSL state dict with `strict=False`; **return the count of matched vs total encoder tensors** via
  a helper `load_bsf_encoder(model, ckpt_path) -> LoadReport(matched:int, total:int, skipped:list[str])` so the
  test and `decision.json` can record load coverage.
- Arm A (BraTS-SSL, 4-ch SSL): the input stem is 4-ch; adapt to 3-ch by dropping the T1ce input channel of the
  first conv (`strict=False` skips the mismatched stem, or slice the pretrained stem weight `[:, [flair,t1pre,t2]]`
  — **implement the slice and document the channel order** rather than silently reinit the stem).
- Arm B (UKB-SSL, T1-only): 1-ch SSL encoder → 3-ch model; the stem cannot transfer (report it in `skipped`),
  deeper encoder blocks do.
- SegResNet Arm C: no pretraining; fork `downstream_seg`'s construction, `in_channels=3, out_channels=2`.
- Deep supervision: when `cfg.deep_supervision`, return a module whose forward yields the main logits plus the
  auxiliary lower-resolution logits (a tuple or a small wrapper) — document the exact return type; task 13/17 consume it.

## Implementation notes
- Frozen pretrained weights are **immutable** — load, never write back. Log the resolved ckpt path + SHA-256 at
  first load (`external-deps.md` rule 6).
- Keep each arm in its own file; `models/` stays tidy (one backbone family per file) so new backbones (MedNeXt,
  nnU-Net) drop in as new files + one decorator.

## Acceptance criteria
1. `get_segmentation_model("bsf_swinunetr_brats", cfg)` returns a module; a `(2,3,64,64,48)` input forward → logits
   `(2,2,64,64,48)` (or deep-sup tuple whose main head is that shape).
2. `load_bsf_encoder` returns `matched > 0` and `matched/total` printed; skipped tensors listed (stem for UKB).
3. Arm C builds without any checkpoint and forwards to `(B,2,…)`.
4. `strict_load=True` with a mismatched ckpt raises (surfaces a bad checkpoint instead of silently reinit).

## Tests (`tests/segmentation/models/test_backbones.py`; `pytestmark = pytest.mark.segmentation`; GPU-free, small tensors)
- **shape contract** (all 3 arms): synthetic `(2,3,32,32,24)` → main logits `(2,2,32,32,24)`; no NaN.
- **BSF load coverage** (mark `slow` if the ckpt must be read from disk; otherwise stub a tiny fake SSL state dict
  with a few matching keys): assert `LoadReport.matched > 0`, the stem appears in `skipped` for UKB, and the
  BraTS stem-slice keeps the `[flair,t1pre,t2]` channel order (assert the sliced weight equals the pretrained
  stem's corresponding 3 input channels).
- **determinism**: same seed → identical forward (guards against an uninitialised buffer).
- **registry wiring**: all three names resolve through `get_segmentation_model`.

## Do NOT touch
`src/vena/validation/downstream_seg.py` (read/fork by copy into `segresnet.py`, do not edit it); anything outside
`src/vena/segmentation/models/` + `tests/segmentation/models/`.

## Report format
Report the BSF ckpt paths used (read back, not reconstructed) + their SHA-256, the per-arm `matched/total` load
counts, the forward-shape numbers, import-isolation proof, ruff-clean, `STATUS`. If a BSF ckpt is missing →
`STATUS: BLOCKED` with the path you looked for.
