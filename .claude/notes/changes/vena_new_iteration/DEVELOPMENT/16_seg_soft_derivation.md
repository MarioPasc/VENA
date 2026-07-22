# 16 — Soft-probability derivation (temperature, avg-pool→latent, ensemble)

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/derivation/` only.

## Objective
The post-prediction pipeline that turns per-model logits into the **soft `[WT,NETC]` conditioning map on the latent
grid `(2,60,60,40)`**: **per-class temperature scaling** → **avg-pool partial-volume** → **K-fold ensemble mean**
(+ optional variance channel). Design authority: Part B.c steps 3–4, **B.f-§2 (per-class T), B.f-§3/§5 (ensemble +
variance caveat)**.

## Read and verify first
- `01_SHARED_CONTRACTS.md` (grid `(60,60,40)`; VAE 4× → `avg_pool_stride=4`; but confirm the exact crop/pad so the
  pooled mask **spatially registers** with the served latents — see the encoder crop convention in
  `src/vena/common` / `vena.model.autoencoder.maisi` preprocessing).
- `DerivationConfig` from task 10; Guo 2017 temperature scaling; Buddenkotte 2023 (per-class + still-needed after
  ensemble).

## Files to create
```
src/vena/segmentation/derivation/temperature.py  # per-class T fit (NLL on OOF calib split) + apply
src/vena/segmentation/derivation/pool.py          # avg-pool to latent grid, partial-volume, spatial registration
src/vena/segmentation/derivation/ensemble.py      # K-fold mean (+ optional per-voxel std channel)
```

## Interface & contract
```python
@dataclass(frozen=True)
class ClassTemperatures: t_wt: float; t_netc: float
def fit_temperatures(logits: Tensor, target_hard: Tensor) -> ClassTemperatures:   # per-class, minimise NLL on calib split
def apply_temperature(logits: Tensor, temps: ClassTemperatures) -> Tensor:        # sigmoid(logit / T), per class
def pool_to_latent(prob_img: Tensor, cfg: DerivationConfig) -> Tensor:            # (2,H,W,D) img -> (2,60,60,40)
def ensemble_soft(maps: Sequence[Tensor], *, emit_variance: bool) -> Tensor:      # mean, or (mean ‖ std) if emit_variance
```
- **Temperature**: fit **two** scalars `T_WT`, `T_NETC` on the OOF calibration split by minimising per-class NLL;
  `apply_temperature` is **argmax-preserving** (T > 0). A single global T is not acceptable (B.f-§2).
- **Pooling**: `AvgPool3d(kernel=stride=cfg.avg_pool_stride)` = partial-volume integration → a boundary latent voxel
  gets a graded value in `(0,1)` by enclosed-lesion fraction. **Order is fixed: temperature/sigmoid first, THEN
  avg-pool** (never pool raw signed SDT). **Spatial registration**: the pooled `(2,60,60,40)` must align voxel-wise
  with the served latents — apply the **same crop/pad** the VAE encoder used, then pool (or pool then crop to match
  — **verify against the encoder convention and assert the output grid is exactly `(60,60,40)`**).
- **Ensemble**: mean over the K fold-models is the primary soft map. `emit_variance` appends the per-voxel std as
  an extra channel → `(3,60,60,40)`; **label it "k-fold disagreement", not epistemic uncertainty** (B.f-§3), and it
  is ablation-only.

## Implementation notes
- Keep the whole path differentiable and soft (SoftSeg).
- `fit_temperatures` must not touch train/test — only the held-out calibration slice per fold.

## Acceptance criteria
1. `fit_temperatures` returns positive `T_WT, T_NETC`; on synthetic overconfident logits the fitted T > 1 and
   applying it **reduces NLL/ECE**; `T_WT != T_NETC` when the two classes have different miscalibration.
2. `apply_temperature` is argmax-preserving (thresholded mask unchanged by any T > 0).
3. `pool_to_latent((2,240,240,160))` (or the real image grid, cropped) → **exactly `(2,60,60,40)`**, all values in
   `[0,1]`, boundary voxels graded (not 0/1).
4. `ensemble_soft` mean shape `(2,60,60,40)`; with `emit_variance` → `(3,60,60,40)`, std ≥ 0.

## Tests (`tests/segmentation/derivation/test_derivation.py`; `pytestmark = pytest.mark.segmentation`; pure-torch)
- **temperature calibration**: synthetic overconfident logits with known target → fitted T reduces NLL vs T=1;
  per-class T differs when WT/NETC miscalibration differs; argmax preserved.
- **partial-volume pooling**: a half-filled block at image res → the corresponding latent voxel ≈ 0.5 (the
  enclosed fraction); a fully-inside voxel ≈ 1.0, fully-outside ≈ 0.0.
- **grid + registration**: `pool_to_latent` output is exactly `(2,60,60,40)`; a mask with a known centroid at image
  res lands at the corresponding latent centroid (assert the argmax voxel maps correctly — catches a crop/stride
  misregistration).
- **order matters**: pooling a sigmoid-map vs pooling raw SDT-then-sigmoid differ; assert the code takes the
  sigmoid-first path (values stay in [0,1], no negatives).
- **ensemble**: 3 identical maps → mean equals them, std = 0; distinct maps → std > 0 at disagreement voxels;
  `emit_variance` channel count.

## Do NOT touch
Anything outside `src/vena/segmentation/derivation/` + `tests/segmentation/derivation/`.

## Report format
Report fitted `T_WT/T_NETC` + NLL-before/after, the partial-volume value at a half-filled voxel, the grid shape,
the registration-centroid check, import-isolation proof, ruff-clean, `STATUS`.
