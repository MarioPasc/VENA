# 17 — Segmentation engine: train (one model per invocation) + predict (ensemble)

**Track/Wave/Deps.** SEG · **Wave 2 (sequential)** · deps: 11, 12, 13, 14, 16 (all merged). Owns
`src/vena/segmentation/engine/{train.py,predict.py}` (NOT `loss.py` — that is task 13).

## Objective
Wire the leaves into a trainer and an inference path. **Train ONE model per invocation** (a given fold, or the
`all_train` model) so the K+1 models fan out as a SLURM array; **predict** assembles the K-fold **ensemble mean**
(the OOF soft map) + optional TTA. Design authority: Part B.b (K-fold OOF, the K models = the ensemble), B.d
(inference = ensemble mean; TTA optional on Málaga), B.f-§6.

## Read and verify first
- `01_SHARED_CONTRACTS.md`; tasks 10–16 interfaces (config, model registry, `make_soft_targets`, dataset+FoldPlan,
  loss, derivation).
- `src/vena/model/fm/lightning/` for the project's Lightning/engine idioms (EMA, grad-clip logging, CSV metrics,
  self-contained `logs/train.log`) — **reuse the conventions**, do not reinvent metric logging.

## Files to create
```
src/vena/segmentation/engine/train.py    # SegTrainer: trains ONE model (fold_index | "all_train")
src/vena/segmentation/engine/predict.py  # ensemble OOF prediction + optional TTA
```

## Interface & contract
```python
class SegTrainer:
    def __init__(self, cfg: SegmentationConfig, fold: int | Literal["all_train"]): ...
    def fit(self) -> Path:   # trains, early-stops on val Dice/loss, fits per-class temperature on the calib slice,
                             # writes ckpt + fold_plan.json + metrics CSVs; returns the run dir
def predict_oof(cfg: SegmentationConfig, ckpts: Mapping[int | str, Path], plan: FoldPlan,
                patient_ids: Sequence[str], *, tta: bool = False) -> dict[str, Tensor]:
    # for each patient, pick its OOF model (fold model, or all_train for FM-val/test), run derivation,
    # return {patient_id: soft [WT,NETC] at IMAGE res} (pooling to latent happens in the mask_predict routine)
```
- `fit` trains one model on `plan.folds` minus the held-out fold (or all FM-train for `all_train`), with deep
  supervision (task 13), the augmentation pipeline (task 14), AMP, cosine LR, early stopping; it fits `T_WT,T_NETC`
  on the held-out **calibration slice** (task 16) and stores them with the checkpoint.
- `predict_oof` routes each patient to the correct **out-of-fold** model (`oof_assignment`), applies temperature +
  ensemble-mean across the K fold-models where appropriate, optional TTA (flip + small rotation) — **Málaga only**.
- Metric logging + `logs/train.log` follow the FM-run conventions; no `print` in library code.

## Implementation notes
- Keep `predict.py` decoupled from H5/latent — it returns image-res soft maps; task 18's routine pools + writes.
- The `all_train` model is what predicts FM-val/test (their OOF); the K fold-models predict their held-out FM-train
  folds. Assert this routing (no in-fold prediction leaks).

## Acceptance criteria
1. `SegTrainer(cfg, fold=0).fit()` on a **synthetic 8-patient fixture** drives training loss **down** and returns a
   run dir with a checkpoint + `fold_plan.json` + metrics CSV.
2. On a tiny fixture the model can **overfit** (train Dice → high), proving the wiring is correct end-to-end.
3. `predict_oof` returns per-patient soft `[WT,NETC]` in `[0,1]`; **no patient is predicted by a model that trained
   on it** (OOF routing asserted).
4. TTA path runs and averages ≥2 augmentations without shape error.

## Tests (`tests/segmentation/engine/test_train_predict.py`; `pytestmark = pytest.mark.segmentation`; small, mark GPU parts `gpu`/`slow`)
- **overfit-tiny (integration)**: 8 synthetic volumes, 3 folds, a few epochs → train loss decreases monotonically
  (or final < initial by a margin); `fit` artifacts exist (readback).
- **OOF routing (leakage)**: build a `FoldPlan`, stub per-fold ckpts → `predict_oof` uses `all_train` for a
  val/test ID and the correct held-out model for a train ID; assert no self-prediction.
- **soft output**: predictions in `[0,1]`, shape `(2,H,W,D)`.
- **TTA**: enabling TTA changes the output (averaged) and stays in `[0,1]`.

## Do NOT touch
INJECT-track files; `engine/loss.py` (task 13); the routines dir (task 18).

## Report format
Report the readback run-dir path, initial-vs-final train loss, the overfit Dice, the OOF-routing assertion result,
import-isolation proof, ruff-clean, `STATUS`.
