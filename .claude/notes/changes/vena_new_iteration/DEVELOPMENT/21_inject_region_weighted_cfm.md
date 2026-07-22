# 21 — Inject: region-weighted CFM with EQUAL-weight default (≡ L1)

**Track/Wave/Deps.** INJECT · **Wave 1 (merge after 20)** · deps: existing FM loss code (+ 20 for the run YAML).
Owns `controlnet/losses/` (the region-weighted CFM) + a loss-config block; may edit a loss-wiring block in
`lightning/module.py` (merge after 20 to avoid a module.py clash).

## Objective
Provide the region-weighted CFM velocity loss with regions **`{Brain = NOT-BG ∩ NOT-WT, WT}`** and a **default of
EQUAL weights `{brain:1.0, wt:1.0}`, numerically identical to the current unweighted L1 velocity loss**. The
mechanism is coded now; **WT up-weighting is a deferred, single-axis ablation** (Q_B). Design authority: **A.8-§2**,
Part A.4; motivation Ibarra MICCAI 2025 (arXiv:2508.13776).

## Read and verify first
- `01_SHARED_CONTRACTS.md` (region semantics).
- The **existing** `region_weights` implementation — grep `region_weights` across `src/vena/model/fm/` and
  `routines/fm/train/`; it shipped with the retired **v3b_rw** arm and appears in `decision.json` 0.10.0. **Confirm
  the current region set, how BG is handled, and how weights combine** before extending. If it already supports
  `{brain, wt}` with a BG rule, reuse it; only add the equal-weight default + the L1-equivalence guarantee.
- `src/vena/model/fm/controlnet/losses/` (CFM loss) + `lightning/module.py` (`loss.cfm`, region masks via
  `metrics/regions.py`, `m_brain`/`m_wt`).

## Files to create / modify
```
MODIFY/CREATE src/vena/model/fm/controlnet/losses/...   # region-weighted CFM (extend existing region_weights)
MODIFY src/vena/model/fm/lightning/module.py            # wire region_weights={brain:1,wt:1} default (loss block only)
```

## Interface & contract
```python
def region_weighted_cfm(v_pred, u_target, *, regions: Mapping[str, Tensor], weights: Mapping[str, float],
                        norm: Literal["l1","l2"]="l1") -> Tensor:
    # weighted mean of |v_pred-u_target|^p over the union of regions; EQUAL weights ⇒ == unweighted mean over that support
```
- **Regions**: `WT` from `m_wt` (dilated per `metrics/regions.py` if that is the existing convention — **verify**);
  `Brain = brain_foreground ∧ ¬WT` from `m_brain`; `BG` excluded (or weight-0) **matching the existing
  `region_weights` semantics** (verify; the test pins whichever is true).
- **Equivalence guarantee (load-bearing)**: with `weights={brain:1.0, wt:1.0}` (and BG handled identically to the
  current L1-mean support), `region_weighted_cfm == cfm_l1_mean` to floating tolerance. This is the whole point:
  ship the mechanism, change nothing numerically until a weight is deliberately raised.
- **Config**: `loss.region_weights: {brain: 1.0, wt: 1.0}` in the YAML; round-trip into `decision.json`.

## Implementation notes
- Region masks are derived **on-device** (no CPU loop) — reuse `metrics/regions.py` (`F.max_pool3d` dilation).
- Do not change the retired-arm behaviour for other weight values; only add the equal-weight default + equivalence.

## Acceptance criteria
1. `region_weighted_cfm(..., weights={brain:1,wt:1})` == unweighted L1-mean CFM over the same support
   (`torch.testing.assert_close`, tight tol).
2. Raising `wt` weight strictly increases the WT-voxel gradient contribution (grad-norm on WT voxels grows).
3. Region partition: `WT ∪ Brain` = brain foreground; `WT ∩ Brain` = ∅; BG handled per the verified existing rule.
4. The default flows through `module.py` → `decision.json.region_weights == {brain:1.0, wt:1.0}`.

## Tests (`tests/model/fm/test_region_weighted_cfm.py`; `pytestmark = pytest.mark.fm`; CPU)
- **L1-equivalence (the guarantee)**: random `v_pred,u_target`, synthetic `m_brain,m_wt` → equal-weight region CFM
  `assert_close` to the plain L1-mean over the same support.
- **up-weight effect**: `weights={brain:1,wt:10}` → `∂loss/∂v` summed over WT voxels is ~10× the equal-weight case
  (up to the region sizes); Brain-voxel grads unchanged.
- **partition**: constructed brain/WT masks → `Brain = brain ∧ ¬WT`, disjoint from WT, union = brain foreground; BG
  contribution matches the verified rule (0 or weight-0).
- **config round-trip**: YAML `region_weights` → `decision.json` value.

## Do NOT touch
`src/vena/segmentation/*`; the v3a config; any non-loss block of `module.py` (coordinate with task 20 on merge order).

## Report format
Report the L1-equivalence residual (max abs), the WT-grad ratio under up-weighting, the verified BG rule, the
`decision.json.region_weights`, import-isolation proof, ruff-clean, `STATUS`. If the existing `region_weights`
semantics differ from the assumed `{brain,wt,BG}` set → report `PREMISE-FALSE` with what you found.
