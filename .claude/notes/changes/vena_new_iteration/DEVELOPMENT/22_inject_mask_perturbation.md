# 22 — Inject: mask-perturbation augmentation (for the deployable T-06 arm)

**Track/Wave/Deps.** INJECT · **Wave 1 (merge after 20)** · deps: 20 (conditioning keys). Owns a new augment
transform + a perturbation hook. **OFF by default** (T-13 oracle stays clean; this is for T-06).

## Objective
A latent-grid **mask-perturbation** transform applied to the `[WT,NETC]` conditioning **during T-06 training** to
close the train-on-GT → deploy-on-predicted gap and to enable optional CFG. Design authority: **A.8-§7** (Q_C:
clean T-13, perturb T-06); Ko ICML 2024 (arXiv:2402.16506), Ho NCA (arXiv:2106.15282).

## Read and verify first
- `01_SHARED_CONTRACTS.md`; `A.8-§7` in the design doc.
- `src/vena/data/augment/online/transforms/` — the existing online latent-transform pattern (match its interface so
  the new transform composes into the pipeline); how transforms are toggled per-config.
- Where conditioning masks enter the training batch (task 20's keys `m_wt_soft`, `m_netc`).

## Files to create / modify
```
CREATE src/vena/data/augment/online/transforms/mask_perturb.py   # the transform (match the existing transform ABC)
MODIFY the augmentation-pipeline registry/config to expose it (verify the existing wiring; do not hard-code)
```

## Interface & contract
```python
class MaskPerturbation(<existing transform base>):
    def __init__(self, *, enabled: bool = False, dilate_erode_vox: int = 3, prob_noise_std_max: float = 0.15,
                 dropout_p: float = 0.15, keys=("m_wt_soft","m_netc"), seed_key: str | None = None): ...
    def __call__(self, sample: dict) -> dict:   # perturbs the named soft masks in-sample; preserves NETC ⊆ WT
```
- **Operations** (each independently toggleable, applied on the latent-grid soft masks):
  1. random **dilation/erosion** by up to `±dilate_erode_vox` voxels (morphology on the thresholded support, then
     re-soften, or a soft morphological op — document which);
  2. additive **Gaussian noise** on the soft probabilities, `σ ∼ U[0, prob_noise_std_max]`, re-clip to `[0,1]`;
  3. whole-map **dropout** with prob `dropout_p` (zero the entire conditioning → dual-purpose: enables CFG training).
- **Invariant**: after perturbation, `NETC ≤ WT` still holds (re-clamp).
- **enabled=False → identity** (T-13 oracle path must be byte-unchanged).
- Determinism: honour the pipeline's seeding (the `AugmentationTracker` / seed-replay convention) so runs are
  reproducible.

## Implementation notes
- This is **conditioning-side only** — never perturb the target `z_t1c` or the trunk-concat latents.
- Keep it 3D and on-tensor; no per-voxel Python loops.
- The T-06 run YAML (predicted mask + this transform enabled) is assembled later in task 18's follow-up, **not
  here** — this task only ships the transform + its off-by-default wiring, validated in isolation.

## Acceptance criteria
1. `enabled=False` → output identical to input (assert_allclose, exact).
2. Dilation/erosion by `k` changes the WT support by ~the expected voxel band; erosion never grows, dilation never
   shrinks.
3. Noise keeps values in `[0,1]`; mean perturbation magnitude scales with `σ`.
4. Over N draws, whole-map dropout fires at ≈ `dropout_p`; a dropped sample's conditioning is all-zero.
5. `NETC ≤ WT` holds after every perturbation.

## Tests (`tests/data/augment/test_mask_perturb.py`; `pytestmark = pytest.mark.unit`; pure-torch)
- **identity when off**: `enabled=False` → exact passthrough.
- **morphology**: a soft ball → erosion shrinks support, dilation grows it, by ~`k` voxels; count the support delta.
- **noise bounds**: perturbed probs stay in `[0,1]`; std of the perturbation ≈ the sampled σ.
- **dropout rate**: N=2000 draws → dropout frequency within a tolerance of `dropout_p`; dropped map is all-zero.
- **nesting preserved**: random WT/NETC → after perturbation `NETC ≤ WT`.
- **determinism**: same seed → identical perturbation.

## Do NOT touch
`src/vena/segmentation/*`; the target/trunk-concat latents; the T-13 oracle YAML.

## Report format
Report the support-delta under ±k, the noise-range check, the measured dropout rate, the nesting-preserved check,
the identity-when-off proof, import-isolation proof, ruff-clean, `STATUS`.
