# Study 7 — VAE reconstruction ceiling (latent floor)

**Paper §4.2 (floor reference) · Priority: support · Data status: 🔨 needs-gen
(cheap, ~1 GPU-h) · Scope: Ring A (extendable)**

References `00_HUB.md` §2. A small, decisive experiment that either **confirms or
kills** the paper's "latent methods pay a VAE compression tax" claim (Hub §0.2).

---

## 1. Question & claim

*What is the best any MAISI-latent method could possibly do?* Encode the **real**
$T_{1c}$ into the frozen MAISI latent and decode it straight back — no synthesis,
no conditioning. The reconstruction error is the **hard floor** that C4–C7 and
every VENA variant inherit, because they all generate in and decode from that
same latent (§2.1 note: even C5 uses the same `autoencoder_v2.pt`).

**Claim to test:** VENA-v3a's whole-brain MAE ≈ the VAE floor ⇒ the latent tax is
**VAE-bound** and VENA saturates what the latent allows (a *strong* positive
framing: "VENA is as good as the representation permits; closing the gap to the
image tier requires a better VAE, not a better generator"). If VENA ≫ floor ⇒
headroom remains and the tax claim is weaker — **drop the claim** rather than
overstate (HANDOFF P2 §10).

## 2. Protocol (generation task T7)

1. For each Ring-A patient, load the **real** $T_{1c}$ from the corpus image H5,
   apply the *same* `percentile_normalise(0, 99.5, foreground_only=True)` Phase-1
   used.
2. Encode → latent → decode via `vena.common` (`MaisiEncoder` / `decode_box`,
   **never** reach into `vena.model.autoencoder.maisi.*`; §rules extensibility).
3. Score decode vs real $T_{1c}$ with the **identical** metric stack as Study 1
   (MAE/RMSE/PSNR/SSIM/MS-SSIM over {brain, wt} — and {et} once Study 8 lands),
   same `data_range=1.0`, same region masks.
4. Emit a `vae_ceiling.csv` with one row per patient; add a `VAE-recon-floor` row
   to Study-1 Table 1 and a horizontal line to Figs 1(forest)/2(Pareto)/2(y-axis).

**Cost:** encode+decode of ~321 volumes on one A100 ≈ 1 GPU-h. **No new training.**
Reuse the exhaustive-val decode path (`decode_box`) so intensity-space parity is
guaranteed (§2.4).

## 3. Data

- **Input (exists):** real $T_{1c}$ in the corpus image H5 (per cohort). Paths in
  `src/external/LINKS.md` / corpus registry.
- **Model (exists, frozen):** `autoencoder_v2.pt` (SHA in every run's
  decision.json).
- **Output (to create):** `…/analyses/vae_ceiling/<UTC>/per_scan/vae_ceiling.csv`
  following the routine layout + a `decision.json` recording the VAE SHA.

## 4. Placement in the paper

Not a standalone section — a **reference floor** woven into §4.2:
- Table 1: a `VAE-recon-floor` row (italic, "no synthesis").
- Fig 2 (forest) + Study-2 Pareto: a dashed "latent ceiling" line.
- One sentence in the tier-gap discussion: "the latent tier's whole-brain deficit
  is at most X of the VAE floor; VENA reaches Y% of it."

## 5. Reviewer objections & pre-emptions

| Objection | Pre-emption |
|---|---|
| "Is the tax the VAE or the generator?" | This experiment *is* the answer — the floor separates the two. |
| "Ceiling only on T1c" | That is the correct target; optionally also encode T1pre/T2/FLAIR to show the VAE is worse on some modalities (proposal §3.4 audit) — supplementary. |
| "Same VAE for all latent methods?" | Yes — §2.1; state it, it makes the floor shared and the comparison fair. |

## 6. Task checklist

- [ ] **T7.1** Write a minimal `vae_ceiling` routine (or reuse encode + decode_box) over Ring A real T1c.
- [ ] **T7.2** Score with Study-1 metric stack; emit per_scan + decision.json (VAE SHA).
- [ ] **T7.3** Compare VENA-v3a MAE_brain to floor; decide keep/drop the tax claim.
- [ ] **T7.4** Add floor row/line to Study 1 & 2 artifacts.
- [ ] **T7.5 (opt.)** Extend to {T1pre,T2,FLAIR} for a per-modality VAE audit (supp).
