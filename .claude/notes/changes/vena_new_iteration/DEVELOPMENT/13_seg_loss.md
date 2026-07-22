# 13 — Segmentation loss (DML + CE, proper on soft labels)

**Track/Wave/Deps.** SEG · **Wave 1 (parallel)** · deps: 10. Owns `src/vena/segmentation/engine/loss.py` only.

## Objective
Implement the composite segmentation loss: **DML (Dice Semimetric Loss, Wang MICCAI 2023) + CE**, the
soft-label-**proper** replacement for soft-Dice, with optional **focal-CE** and **Tversky / focal-Tversky** Dice
terms and deep-supervision aggregation. Design authority: Part B.a (the soft-Dice correction), B.f-§2.

## Read and verify first
- `01_SHARED_CONTRACTS.md`; `LossConfig` from task 10.
- Wang et al., *Dice Semimetric Losses*, MICCAI 2023 (arXiv:2303.16296): **DML equals soft-Dice on HARD labels**
  and is well-defined (symmetric, proper) on **soft** labels — this defines the two required property tests.
- MONAI `DiceCELoss`, `TverskyLoss`, `FocalLoss` — reuse where correct, but **verify MONAI's Dice is NOT the
  standard soft-Dice on soft targets** (it is improper on soft labels); implement DML explicitly if MONAI lacks it.

## Files to create
```
src/vena/segmentation/engine/loss.py   # SegmentationLoss(nn.Module) + component functions
```

## Interface & contract
```python
def dice_semimetric_loss(probs: Tensor, target: Tensor, *, reduction="mean", eps=1e-6) -> Tensor:  # DML
def ce_term(logits: Tensor, target: Tensor, *, focal_gamma: float | None) -> Tensor:               # CE or focal-CE
def tversky_term(probs, target, *, alpha, beta, focal_gamma: float | None) -> Tensor:              # (focal-)Tversky
class SegmentationLoss(nn.Module):
    def __init__(self, cfg: LossConfig): ...
    def forward(self, outputs, target) -> Tensor:   # outputs = main logits OR (main, *aux) for deep supervision
```
- `probs = sigmoid(logits)` per channel (independent WT/NETC sigmoids, region-based — **not** softmax; the two
  channels are nested, not mutually exclusive).
- DML operates on **soft** `probs` and **soft** `target ∈ [0,1]` (the SDT-sigmoid targets from task 12) and must
  reduce to standard soft-Dice when `target ∈ {0,1}`.
- Deep supervision: when `outputs` is a tuple, aggregate `Σ w_i · loss(head_i, downsampled target)` with
  `cfg.deep_supervision_weights`; downsample the target to each head's resolution with area/avg interpolation
  (keep it soft).
- Numerical guards: eps in every denominator; clamp probs to `[eps, 1-eps]` before log in CE.

## Implementation notes
- FN-costly NETC → Tversky/focal-Tversky is the FN-weighting lever (`alpha<beta`); default Dice term is DML but the
  config can select `tversky`/`focal_tversky` for the NETC channel.
- Keep component functions free-standing (module-level, testable without the `nn.Module`), per coding-standards
  rule 16.

## Acceptance criteria
1. On **hard** binary targets, `dice_semimetric_loss == 1 - soft_dice` (compute a reference soft-Dice inline) to
   `rtol=1e-5`.
2. On **soft** targets, DML is finite, non-negative, and **symmetric** under swapping identical-shape soft
   pred/target where the metric expects symmetry (state the exact DML symmetry property you test).
3. `SegmentationLoss.forward` returns a scalar with a finite gradient w.r.t. logits (`.backward()` populates grads,
   no NaN).
4. Deep-supervision path aggregates the configured weights (a 2-head stub → weighted sum matches a hand computation).

## Tests (`tests/segmentation/engine/test_loss.py`; `pytestmark = pytest.mark.segmentation`; pure-torch, CPU)
- **DML == soft-Dice on hard labels** (the defining property): random hard target + probs → assert equality to an
  independent soft-Dice implementation.
- **soft-label properness**: perfect match (probs == soft target) → DML ≈ its minimum; assert it does **not** NaN
  and is lower than any perturbed prediction (monotone toward the target).
- **gradient**: `loss.backward()` gives finite grads; grad is zero at a perfect prediction (within tol).
- **real-mask sanity**: build a soft target from task 12's `make_soft_targets` on a synthetic label (import it) and
  assert DML+CE decreases as probs move toward that target.
- **Tversky FN-weighting**: with `alpha<beta`, a false-negative-heavy prediction incurs a strictly larger Tversky
  loss than a false-positive-heavy one of equal count.

## Do NOT touch
Anything outside `src/vena/segmentation/engine/loss.py` + `tests/segmentation/engine/`. (Task 17 owns `engine/train.py`
+ `predict.py`; do not create them here.)

## Report format
Report the DML-vs-soft-Dice equality residual, the gradient-at-optimum value, the Tversky FN/FP asymmetry numbers,
import-isolation proof, ruff-clean, `STATUS`.
