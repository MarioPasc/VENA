# 11 — Segmentation models (BSF-SwinUNETR Arm A/B + SegResNet Arm C)

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/models/` only.

> **🔴 TC, not WT (2026-07-22).** The 2-channel output is `[TC, NETC]` — channel 0 = **tumour core (TC = NETC+ET)**,
> edema EXCLUDED (`TargetConfig.tumor_region="tc"`); `TC−NETC = ET` = the enhancing region. The architecture is
> unchanged (2 soft channels), but the target/eval is TC not WT — and TC is harder to segment (see task 15's G-SEG note
> + `[[project_channel0_tumor_core_not_wt]]`). BSF-SwinUNETR's tumour-aware SSL (BraTS) suits the harder ET/TC boundary.

## 🔧 ITER-9 HARNESS ADDENDUM (2026-07-23)

**Parallel-launch.** SEG track — runnable NOW in parallel with the oracle (S1/S2/S3); no dependency on oracle results.
**✅ BSF checkpoints LOCATED + pinned in `src/external/LINKS.md` (2026-07-23)** — encoder-only SSL, SwinUNETR fs=48.
**Arm priority RE-PRIORITIZED (supersedes the design's original "Arm A primary"):**
- **Arm B UKB-SSL = the LEAK-FREE HEADLINE/PRIMARY** — `…/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt`
  (single ckpt). Healthy UK Biobank, **no BraTS patients, no T1ce** → removes the patient-overlap representation leak
  (BraTS-SSL saw BraTS-GLI = a VENA CV cohort; **OOF cannot fix an SSL-stage leak**) *and* the T1ce-exposure leak.
  **Caveat:** UKB transfers less (healthy, likely T1-focused) — the **input stem may not fully transfer** to the 3-ch
  {T1pre,T2,FLAIR} model; deep blocks do. `load_bsf_encoder` must **report the actual stem channel count and list the
  stem in `skipped` if it can't transfer** (don't force it). See design B.a + C.2-L3.
- **Arm A BraTS-SSL = domain-matched COMPARATOR (not headline)** — `…/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold{0..4}.pt`
  (5 per-fold ckpts; pick/document which fold). Higher tumour-domain match but leaks BraTS-GLI images + T1ce → run it
  to **quantify the leakage↔Dice trade-off**, cite as an upper bound.
- **NEVER use `BrainSegFounder_finetuned_BraTS/*`** — a supervised BraTS segmenter = L1+L2+L3 leakage at once.
- **Arm C (`segresnet`, no ckpt) = the from-scratch floor — build it FIRST (fastest to green).** Fork
  `src/vena/validation/downstream_seg.py` (543 lines) **by copy** into `segresnet.py` (do NOT edit the original), drop
  the T1c input → `in_channels=3, out_channels=2`.

**Reuse, don't rebuild:** MONAI `SwinUNETR`/`SegResNet` are the backbones; `models/registry.py` (task 10, merged) is
the decorator + lookup — import and register, don't re-create it.

**Sharper acceptance (additive; all must hold):**
1. **Arm C definition-of-done:** `segresnet` builds with NO checkpoint; `(2,3,32,32,24) → (2,2,32,32,24)` main logits,
   no NaN; registered name resolves through `get_segmentation_model`.
2. Arms A/B: arch builds (random init) + forwards to `(B,2,…)`; `load_bsf_encoder → LoadReport(matched,total,skipped)`;
   with real weights assert **Arm A `matched/total ≥ 0.80`** and **Arm B stem ∈ `skipped`** (1-ch→3-ch can't transfer).
   Mark these `slow` + **skip-if-ckpt-absent** (never fail the fast suite on a missing ckpt).
   > **✅ DONE + RESOLVED (S4/S5, 2026-07-23, commits `eab69b8`+`6f281db`):** all three arms green.
   > `load_bsf_encoder(model.backbone, …)` — the BraTS ckpt is a Swin-T-depth model (`depths=(2,2,6,2)`, 6 stage-3
   > blocks), so the standard `SwinUNETR(fs=48, depths=(2,2,2,2))` first gave only 126/198 = 0.636. Matching the depth
   > (per-arm constants `_BSF_BRATS_SWIN_KW`/`_BSF_UKB_SWIN_KW` in `bsf_swinunetr.py`) lifts **Arm A to 182/198 = 0.919**
   > (only 16 SSL pretext heads skipped; the ≥0.80 criterion now holds from the correct config). **Arm B UKB = 125/142
   > = 0.880** (16 heads + the 1-ch stem, `depths=(2,2,2,2)`). ckpt SHA-256: A `e46d80ce75f3…`, B `4be92492ae4f…`.
3. BraTS stem-slice preserves channel order `[flair,t1pre,t2]` — assert the sliced 3-ch stem equals the pretrained
   4-ch stem's corresponding channels (not a silent reinit).
4. Deep-supervision return type documented + the main-head shape asserted inside the tuple.
5. Determinism: same seed → identical forward (no uninitialised buffer).

**Definition of done:** Arm C green end-to-end + registered with no checkpoint; Arms A/B green (weights present) or a
clean `BLOCKED` report naming the path; fast suite green (BSF-ckpt tests skipped), ruff-clean on touched files.

## Objective
Provide the three backbone arms as registry-registered builders returning a `torch.nn.Module` mapping
`(B, 3, H, W, D) → (B, 2, H, W, D)` logits `[TC, NETC]` (+ deep-supervision heads when configured). **BSF-SwinUNETR
is primary** (Arm A BraTS-SSL, Arm B UKB-SSL); **SegResNet is Arm C**, forked from the existing
`src/vena/validation/downstream_seg.py` 4-input model (drop the T1c input → 3-input). Design authority: Part B.a,
B.f-§1.

## Read and verify first
- `01_SHARED_CONTRACTS.md` + `src/external/LINKS.md` (BSF facts + the **RESOLVED** ckpt paths — see the iter-9
  addendum above: UKB-SSL = primary/headline, BraTS-SSL = comparator, finetuned = NEVER; verify the file exists and
  **log its SHA-256 at load**, `external-deps.md` rule 6).
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
