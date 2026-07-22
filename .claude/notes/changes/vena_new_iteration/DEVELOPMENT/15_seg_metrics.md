# 15 — Segmentation metrics, G-SEG gate, dual selection

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/metrics/` only.

## Objective
Evaluation for the segmenter: overlap (Dice, AHD/HD95), **calibration (ECE, classwise-ECE, Brier)**, the **G-SEG
gate** (**TC Dice ≥ 0.75 (provisional)**, NETC Dice ≥ 0.50 per cohort incl. Ring B; healthy → ~empty), and the
**dual DSC+Brier** model-selection rule (the generator consumes soft probs, so calibration is load-bearing). Design
authority: Part B.b, **B.f-§2, B.f-§7**.

> **🔴 TC, not WT (2026-07-22).** The segmenter predicts the same `[TC, NETC]` soft targets used for conditioning
> (channel 0 = **tumour core TC = NETC+ET**, edema EXCLUDED — `TargetConfig.tumor_region="tc"`; `TC−NETC = ET`). The
> G-SEG gate is therefore a **TC-Dice** gate (`cfg.gseg_tc_dice`), NOT WT-Dice. **TC is harder to segment than WT**
> (BraTS TC Dice ~0.80-0.87 vs WT ~0.88-0.92), so the `0.75` default is **PROVISIONAL** — re-derive it from the
> measured per-cohort TC Dice in S5 before trusting the gate. See `[[project_channel0_tumor_core_not_wt]]`.

## Read and verify first
- `01_SHARED_CONTRACTS.md`; `MetricsConfig` from task 10.
- MONAI `DiceMetric`, `compute_hausdorff_distance` / `compute_average_surface_distance` — reuse; verify the surface
  metric handles empty masks (healthy controls) without crashing.

## Files to create
```
src/vena/segmentation/metrics/overlap.py      # dice, ahd/hd95 (threshold soft->hard at 0.5 for overlap only)
src/vena/segmentation/metrics/calibration.py  # ece, classwise_ece, brier (on the SOFT probs)
src/vena/segmentation/metrics/gate.py         # G-SEG check + dual DSC/Brier selection
```

## Interface & contract
```python
def dice(pred_soft: Tensor, target_hard: Tensor, *, threshold=0.5) -> float:   # per-class
def average_hausdorff(pred_soft, target_hard, *, threshold=0.5, percentile=95) -> float:
def expected_calibration_error(probs: Tensor, target_hard: Tensor, *, n_bins=15) -> float
def classwise_ece(probs, target_hard, *, n_bins=15) -> dict[str, float]         # {"tc":..,"netc":..}
def brier(probs, target_hard) -> dict[str, float]
@dataclass(frozen=True)
class GSegResult: passed: bool; per_cohort: dict[str, dict[str, float]]; failures: list[str]
def check_gseg(dice_by_cohort: Mapping[str, Mapping[str, float]], cfg: MetricsConfig) -> GSegResult
def select_ensemble(models: Sequence[ModelScore]) -> str    # dual: prefer better Brier within ~1% DSC
```
- **Overlap** thresholds the soft map at 0.5 (overlap is a hard-mask notion); **calibration** uses the **raw soft
  probs** (never thresholded — that is the whole point).
- **G-SEG**: TC Dice ≥ `cfg.gseg_tc_dice` **AND** NETC Dice ≥ `cfg.gseg_netc_dice` for **every** cohort incl. Ring
  B; a healthy-control case must yield a near-empty mask (TC volume ≈ 0). `failures` lists `(cohort, class, value)`.
- **Selection** (`selection_metric="dual"`): among candidate models, if Brier improves at the cost of `< 1%` DSC,
  prefer the better-Brier model (B.f-§7); `"dice"`/`"brier"` select on that metric alone.

## Implementation notes
- Empty-mask handling: Dice of two empty masks = 1.0; AHD undefined → return `nan` and exclude from aggregation
  (document it). Healthy-control gate checks TC **volume**, not Dice.
- Keep metrics on-device where cheap; no CPU loops over voxels.

## Acceptance criteria
1. `dice` on identical masks = 1.0; on disjoint = 0.0; on a known 50%-overlap synthetic = the hand value.
2. `expected_calibration_error` = 0 for perfectly-calibrated synthetic probs; > 0 for overconfident probs.
3. `brier` matches the mean-squared-error definition on a synthetic case.
4. `check_gseg` passes/fails exactly at the thresholds; a below-0.50 NETC cohort → `passed=False` with that cohort
   listed.
5. `select_ensemble` prefers the better-Brier model when DSC is within 1%, else the better-DSC model.

## Tests (`tests/segmentation/metrics/test_metrics.py`; `pytestmark = pytest.mark.segmentation`; pure-torch/numpy)
- **overlap**: identical / disjoint / 50%-overlap synthetic → 1.0 / 0.0 / hand value; AHD of identical = 0.
- **calibration**: construct perfectly-calibrated bins → ECE ≈ 0; skew to overconfident → ECE increases; Brier
  matches the closed-form on a 3-value example.
- **empty masks**: two empty → Dice 1.0, AHD nan (excluded); healthy-control (all-zero pred) → gate's WT-volume
  check flags "empty".
- **G-SEG threshold logic**: dict at exactly {TC:0.75, NETC:0.50} passes; {NETC:0.49} fails and lists the cohort.
- **dual selection**: two model scores (DSC 0.85/Brier 0.10 vs DSC 0.845/Brier 0.06) → picks the second (within 1%
  DSC, better Brier); widen the DSC gap to 3% → picks the first.

## Do NOT touch
Anything outside `src/vena/segmentation/metrics/` + `tests/segmentation/metrics/`.

## Report format
Report the overlap/ECE/Brier reference values, the G-SEG pass/fail example, the dual-selection decision, import-
isolation proof, ruff-clean, `STATUS`.
