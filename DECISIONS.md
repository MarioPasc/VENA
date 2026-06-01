# DECISIONS

One entry per architectural decision: date, options considered, choice, rationale, reversibility. New entries go at the top.

---

## 2026-06-01 — S2 loss formulation (Lp-aware mask-perturbation contrastive, merged v0.3)

**Decision.** The S2 training regime adds a single composite contrastive term to
the CFM loss:

$$
\mathcal{L}_{\text{S2}} = \mathcal{L}_{\text{CFM}}
  + \lambda_{\text{contrast}}\bigl(\lambda_{\text{tum}}\,\mathcal{L}_{\text{roi}}^{(p_t)}
                                  + \lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}}^{(p_b)}\bigr)
$$

with $p_t = 1$ (MAE inside the dilated whole-tumour mask, aggregate cap, negated),
$p_b = 3$ (super-linear focus outside, per-voxel cap), $\delta = 2.0$,
$\lambda_{\text{tum}} = 0.3$, $\lambda_{\text{bg}} = 1.0$, and
$\lambda_{\text{contrast}}$ annealed step-half $0.01 \to 0.001$ via
`StepHalfWeight` (factor 0.1).

**Options considered.**

- *v0.2 factorised.* Pure $L^1$ contrastive (no exponent) plus a separate capped
  $L^p$ velocity-reconstruction term. Closer to MAISI-v2 upstream, but the two
  terms compete: the recon term tries to match $v$ to $u_{\text{target}}$ while
  the contrastive pushes $|v_\text{orig} - v_\text{perturb}|$ apart in the
  tumour. Pinned as an ablation row, not the headline.
- *v0.3 merged (this decision).* Drop the recon term; let the per-region
  $L^{p_t}$ / $L^{p_b}$ play both roles. Cleaner gradient signal; one fewer
  hyperparameter.
- *Pure MAISI-v2 port.* Single window-ReLU cap on the mean, single weight 0.01,
  no per-region $L^p$. Could not have been used: the contrast-enhancement
  region in T1c synthesis is harder than the rest of the brain, so we need
  region-specific exponents to bias the focus there.

**Rationale.** The 4-epoch smoke (`scratch/2026-06-01_s2_smoke_results.md`)
showed:

- *Ablation cleanliness.* S1 and S2 cfm trajectories are byte-equal to 4 sig figs
  at $\lambda_{\text{contrast}} = 0.01$. The contrastive does not poison the
  primary regression signal.
- *Contrastive bites.* $|\Delta_\theta|_{\text{WT}} / |\Delta_\theta|_{\text{BG}}$
  grows monotonically $3.7\times \to 15.5\times$ across the 4 epochs — the
  ControlNet learns to be sensitive to the WT mask inside the tumour and
  invariant to it outside, exactly as the proposal hypothesises.
- *Stability.* No NaN, no grad-norm spikes, both cap-hit fractions at 0% with
  $\delta = 2$.

The reduction-to-S1 contract is preserved: setting `loss.contrastive.weight: 0`
recovers $\mathcal{L}_{\text{S2}} = \mathcal{L}_{\text{CFM}}$ exactly (regression
test `test_lambda_contrast_zero_recovers_s1_total`).

**References.**

- Proposal: `.claude/notes/foundations/proposal_contrastive_loss.md` §3, §5, §8.
- Smoke results: `scratch/2026-06-01_s2_smoke_results.md`.
- Implementation: `src/vena/model/fm/controlnet/losses/{contrastive.py,schedule.py,builder.py,base.py}`.
- Tests: `tests/model/fm/test_losses_contrastive.py`, `tests/model/fm/test_loss_schedule.py`.

**Reversibility.** Fully reversible by config. Set `run.stage: s1` in any YAML
to drop S2 entirely. Set `loss.contrastive.weight: 0` to keep the S2 forward
machinery active (two ControlNet passes, $m_{\text{bg}}$ derivation, aux
diagnostics) but zero out the term's contribution to the loss. Switching from
v0.3-merged to v0.2-factorised requires re-enabling `S3` (CompositeLoss has
the stub) and writing the `CappedLpReconLoss` body — a separate, named
ablation row.

---

## 2026-06-01 — Trunk-EMA round-trip is resume-safe

**Decision.** Drop the "single-shot, not resume-safe" caveat for the
unfrozen-trunk path. SLURM job chains on Picasso may resume `last.ckpt`
without further code changes.

**Background.** `.claude/rules/model-coding-standards.md` flagged the unfrozen
trunk as not-resume-safe because `setup()` instantiates `trunk_ema` from the
*original* MAISI trunk. The concern was that Lightning's checkpoint restore
might not override the freshly built shadow.

**Verification.** `tests/model/fm/test_trunk_ema_resume.py` exercises three
contracts:

1. `WarmupEMA.load_state_dict(saved)` after a fresh `WarmupEMA(trunk)` build
   overrides the shadow with saved values (not the fresh init).
2. A `nn.Module` that registers a `WarmupEMA` submodule exposes the shadow
   parameters in `state_dict()` — so Lightning's checkpoint payload carries
   them.
3. End-to-end wrapper round-trip: save → fresh wrapper with different init →
   `load_state_dict` → shadow matches saved values.

All three pass. The "not resume-safe" wording was conservative.

**Reversibility.** N/A — purely a documentation correction. The underlying
behaviour was already correct.

---
