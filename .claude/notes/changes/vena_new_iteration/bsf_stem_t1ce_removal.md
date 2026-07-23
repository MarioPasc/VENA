# BSF Stem — T1ce Input Channel Removal

**Context.** VENA is gadolinium-free: T1ce (T1 post-contrast) is never available at training or inference. The BraTS-SSL checkpoint (Arm A) has a 4-channel stem `(48, 4, 2, 2, 2)` trained on BraTS input order `[FLAIR, T1pre, T1ce, T2]`. We need a 3-channel model for `{T1pre, T2, FLAIR}`. The UKB checkpoint (Arm B) has a 1-channel stem `(48, 1, 2, 2, 2)` trained on T1w alone; its stem cannot transfer at all under any strategy — this note applies to Arm A only.

---

## Strategies

### 1. Drop-slice (current — `[0, 1, 3]`)

Extract weight columns for FLAIR, T1pre, T2 by index; discard the T1ce column entirely.

```python
new_stem = pretrained_stem[:, [0, 1, 3], ...]   # (48, 3, 2, 2, 2)
```

**Pretrained signal preserved.** 3 of 4 filters survive intact. The T1ce column (index 2) is thrown away — its spatial patterns were partly driven by enhancement and are not portable, so discarding it is correct.

**Param cost.** Zero. No new parameters.

**Gadolinium-free.** Fully compatible: no T1ce dependence at any point.

**Interaction with deep-encoder transfer.** All subsequent encoder keys (stages 1–4) load unchanged. The stem initialisation affects only the input projection; the transformer blocks downstream see identical activations if the kept filters are representative.

**Risk.** The 3 kept filters were co-trained with the T1ce column. Their learned spatial frequencies (patch-level luminance/edge detectors at 2 mm isotropic) are not specific to T1ce so the co-training effect is small. The missing column leaves 1/4 of capacity unused at the input, but the network was not optimised to balance across columns and the dropped column carries enhancement signal that is precisely what we must not leak.

**Recommendation: adopt.** Already implemented in `_BRATS_STEM_CHANNEL_SLICE = [0, 1, 3]`.

---

### 2. Average / fold the T1ce filter

Sum (or mean) the T1ce column into one of the retained columns, e.g. fold into T1pre (indices differ in intensity range but both are T1-weighted).

```python
stem = pretrained_stem.clone()          # (48, 4, 2, 2, 2)
stem[:, 1, ...] += stem[:, 2, ...]      # add T1ce → T1pre
new_stem = stem[:, [0, 1, 3], ...]      # (48, 3, 2, 2, 2)
```

**Pretrained signal preserved.** All 4 column energies are retained, but conflated into 3 positions. The T1ce filter contains enhancement-specific patterns that, when folded into T1pre, corrupt the T1pre filter's semantics. The stem will activate spuriously on T1pre regions that resemble enhancing tissue.

**Param cost.** Zero.

**Gadolinium-free.** The folded filter encodes T1ce semantics applied to T1pre inputs at inference. This is a subtle enhancement shortcut risk.

**Recommendation: reject.** The folded T1ce signal is not representable from the gadolinium-free inputs and acts as a distorted prior. Drop-slice is strictly cleaner.

---

### 3. Re-initialise the whole stem (skip, learn from scratch)

Build a fresh `Conv3d(3, 48, kernel_size=(2,2,2))` with MONAI default init.

**Pretrained signal preserved.** None at the stem; all 4 existing columns are discarded.

**Param cost.** Zero (same param count, different init).

**Gadolinium-free.** Yes.

**Interaction with deep-encoder transfer.** The stage 1–4 transformer blocks still load from the checkpoint. However, a randomly-initialised stem produces activation distributions that are inconsistent with what the transformer blocks were calibrated against. Empirically this is a mild effect (the blocks adapt quickly), but it is strictly worse than drop-slice which preserves 3 of 4 stem columns.

**Recommendation: reject.** Drop-slice dominates: it preserves 3 columns at zero cost and does not corrupt the consistency between stem output and the downstream transformer input distributions.

---

### 4. Learned 3→4 adapter (`Conv3d(3, 4, 1)`) feeding the pretrained 4-channel stem

Insert a trainable 1×1×1 convolution that projects the 3-channel input to 4 channels, then feed the pretrained 4-channel stem unchanged.

```
Input (B,3,H,W,D) → Conv3d(3,4,1) [new, trained] → (B,4,H,W,D)
→ pretrained stem Conv3d(4,48,2,2,2) [frozen or fine-tuned]
```

**Pretrained signal preserved.** The entire 4-channel stem loads intact. The adapter learns a 3→4 projection that best approximates the 4-channel distribution the stem expects.

**Param cost.** 12 parameters (3×4 weight + 4 bias). Negligible.

**Gadolinium-free.** Yes. The adapter is trained on gadolinium-free inputs; the 4th virtual channel is a learned function of the 3 available, not a real T1ce estimate.

**Interaction with deep-encoder transfer.** The stem receives inputs from the adapter, whose distribution during training is determined by the data statistics. Because the adapter is random at init, stage 1 of the encoder is adapting to a different input than it saw during SSL, degrading the benefit of the pretrained stem somewhat. The benefit is recovered as the adapter trains.

**When to prefer.** If the T1ce stem filter carries critical spatial-frequency patterns not covered by FLAIR/T1pre/T2 — empirically unlikely for a patch-embed operating on 2 mm voxels where all channels share low-frequency anatomical contrast. For high-resolution edge-detector stems (< 1 mm, 3×3×3 kernels) the adapter strategy has stronger motivation.

**Recommendation: hold in reserve.** May be worth as an ablation if drop-slice shows degraded vessel conspicuity at the first encoder stage, but there is no a-priori reason to expect this for a 2³ voxel patch-embed.

---

### 5. Zero / synthetic T1ce channel feeding the 4-channel stem intact

Pad a zero (or noise) 4th channel to the 3-channel input, passing a `(B,4,H,W,D)` tensor to the unmodified pretrained stem.

```python
x_4ch = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)  # (B,4,H,W,D)
```

**Pretrained signal preserved.** The stem is fully intact. The zero channel activates exactly the T1ce filter's bias term but suppresses all data-driven contributions.

**Gadolinium-free.** Yes, but the model is architecturally 4-channel and must always receive 4 channels at inference.

**Interaction with deep-encoder transfer.** The 4-channel stem is consistent with checkpoint weights. The T1ce filter's contribution is set to its bias only, which is a structured but not informative signal.

**Risk.** At training time the model learns that channel 4 is always zero, which can cause the corresponding stem filter to atrophy (weight decay drives its effective capacity to zero). This is equivalent to drop-slice in the limit but with a more complex learning dynamic and an unnecessary 4th parameter column.

**Recommendation: reject.** Drop-slice achieves the same outcome more directly, with cleaner parameter utilisation and no architectural coupling to a zero channel.

---

## Summary table

| Strategy | Pretrained signal | Gd-free | New params | Recommended |
|---|---|---|---|---|
| 1. Drop-slice `[0,1,3]` | 3/4 columns intact | Yes | 0 | **Yes (current)** |
| 2. Average/fold T1ce | 4/4 (conflated) | Risk (T1ce shortcut) | 0 | No |
| 3. Re-init stem | 0/4 | Yes | 0 | No |
| 4. 3→4 adapter | 4/4 (adapter required) | Yes | 12 | Hold / ablation |
| 5. Zero 4th channel | 4/4 (bias only) | Yes | 0 | No |

**Chosen strategy: drop-slice `[0, 1, 3]` (Strategy 1).** It preserves the maximum valid pretrained signal (3 of 4 stem columns), adds zero parameters, is architecturally clean, and carries no enhancement-shortcut risk. The discarded T1ce column encodes contrast-agent-dependent spatial patterns that are not representable from the gadolinium-free input and should not be available to the model under any form.

---

## Note on Arm B (UKB)

The UKB checkpoint has a 1-channel stem `(48, 1, 2, 2, 2)` trained on T1w alone. No strategy transfers this stem to a 3-channel model without distortion: the single column was optimised for T1w intensity only and projecting it to 3 channels (by tiling, averaging, or adapting) is undefined given the per-channel semantic difference between T1pre, T2, and FLAIR. The stem is unconditionally placed in `LoadReport.skipped` and re-initialised from MONAI defaults. The remaining 125/142 (88.0%) of encoder keys (stages 1–4) transfer cleanly, so the majority of pretrained signal is retained despite the stem loss.
