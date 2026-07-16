# TASK V3 — `downstream_seg`: §4.4 task-based downstream segmentation

**Read `01_SHARED_CONTRACTS.md` first, completely.** Then read
`.claude/notes/validation/validation_proposal.md` §4.4, Appendix A, and §6.

| | |
|---|---|
| **Model** | Opus 4.8, effort `max` |
| **Isolation** | **git worktree**, branch `task/validation-downstream-seg` |
| **Depends on** | T0 `validation-core` (merged into main before you start) |
| **Runs** | in parallel with V1 (`paired_fidelity`) and V2 (`spatial_residual`) |
| **Lane (you own)** | `src/vena/validation/downstream_seg.py`, `routines/validation/downstream_seg/**`, `tests/validation/test_downstream_seg.py`, `tests/validation/test_downstream_seg_engine.py` |
| **Do not touch** | T0's modules (read-only), `pyproject.toml`, the other agents' lanes, `CLAUDE.md`, `.claude/rules/**`, `src/external/**` |

Run the contracts §2 import-isolation self-check first and paste the output.

---

## 1. Why this exists, and the one thing you are measuring

§4.4 asks: **if you swap the real T1c for a synthetic one, how much does a
downstream tumour segmenter degrade?** (Preetha et al. 2021, *Lancet Digital
Health*, DOI 10.1016/S2589-7500(21)00205-3.)

**ET-Dice is the clinical-impact metric** — the only sub-label whose accuracy
depends predominantly on T1c contrast geometry (WT and TC also draw on
T2/FLAIR/T1pre). A model that wins on MAE but loses ET-Dice has produced
contrast that *looks* right voxel-wise but is **anatomically wrong**. That
finding, if it appears, is more valuable than a clean win — report it loudly.

**The primary statistic is the paired delta, not the absolute Dice:**
```
Δ_Dice(method, label) = Dice(real T1c) − Dice(synthetic T1c)
```
per patient, same segmenter, same inputs otherwise.

---

## 2. The segmenter — DECIDED, do not re-litigate

**Decision (2026-07-16, user-approved): a pretrained, fixed segmenter. Report
Dice_real, Dice_synth, and Δ.**

This **deviates from proposal Appendix A**, which says train nnU-Net from
scratch per Ring-A cohort. Record the deviation and this rationale in
`report.md` + `decision.json`:

> Appendix A's concern is that a BraTS-pretrained segmenter is *familiar with
> the real BraTS distribution*, so a high absolute Dice on synthetic T1c could
> reflect that familiarity rather than synthesis quality. That is a **level**
> confounder: it shifts `Dice_real` and `Dice_synth` together. It **cancels in
> the paired Δ**, which is the quantity §4.4 actually claims. The segmenter is
> a *fixed measuring instrument* applied identically to both arms; only the T1c
> channel changes. Absolute Dice is reported as context, never as the endpoint.

**`nnunetv2` is NOT installed in `vena`.** Do not install it. Do not add any
dependency (contracts §8) — if you think you must, **stop and report**.

**Preferred instrument:** the MONAI Model-Zoo BraTS bundle
(`brats_mri_segmentation`, SegResNet; 4-channel in, TC/WT/ET out).
`monai.bundle.download` is available (monai 1.5.2 confirmed).

**Step 1 of your task is to VERIFY the instrument exists and does what this
plan claims.** Do not build on my description. Check:
- the exact bundle name and that it downloads;
- its **channel order** (BraTS convention is usually `[FLAIR, T1ce, T1, T2]` —
  **getting this wrong silently destroys every number and still produces
  plausible Dice**; verify against the bundle's own `inference.json` /
  `metadata.json`, not against memory);
- its **preprocessing** (typically `NormalizeIntensityd(nonzero=True,
  channel_wise=True)` — i.e. z-scoring). Our volumes are percentile-normalised
  to [0,1]. **Apply the bundle's own transforms** — that is the instrument's
  contract, not a re-harmonisation, and it is applied identically to both arms
  (contracts §7 rule 1 exception);
- its **output convention** (usually 3 sigmoid channels = TC, WT, ET at
  threshold 0.5 — *not* softmax over labels);
- whether it expects the BraTS 240×240×155 1mm skull-stripped grid (ours is
  exactly that).

If the bundle is unavailable, or its contract doesn't match, **stop and report
with evidence (`STATUS: PREMISE-FALSE`)** rather than substituting something
silently. Fallbacks to propose in that report, in order: another public BraTS
checkpoint; a small MONAI `SegResNet` trained on the pooled Ring-A train split.

**Network access:** if Picasso compute nodes have no internet, download the
bundle on the login node (or locally) and pass a **local path** via YAML. Make
the bundle path a YAML parameter, never a hard-coded download at runtime — a
routine that reaches the network mid-sweep is not reproducible. Pin and log the
bundle's version + sha256 in `decision.json` (`external-deps.md` rule 6).

---

## 3. The data problem — read carefully, this is the hard part

The prediction/reference H5s carry **only a binary `masks/wt`** (contracts §6.1).
**TC and ET are not in them.** You must join back to the **corpus image H5s** for
the multi-label ground truth.

**MeningD2 is NOT mounted locally → the corpus H5s are only reachable on
Picasso.** This routine's real-data smoke therefore runs **on Picasso**, not
locally. Plan for that from the start (see §6).

### 3.1 Corpus H5 layout (VERIFIED on Picasso 2026-07-16)

`/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/vena/<cohort>/h5/<NAME>_image.h5`
(exact paths: `routines/fm/train/configs/corpus/corpus_picasso.json`)

- Root attrs: `schema_version="2.0.0"`, **`label_system`** (`"BraTS2021"` on
  UCSF-PDGM and IvyGAP; verify per cohort), `producer`
- `masks/tumor` — `(N,240,240,155)` int8. Verified `unique = [0 1 2 4]`.
  Its own attr: `"BraTS-style tumour labels {0=bg, 1=necrosis, 2=edema,
  4=enhancing}."`
- `masks/brain`, `images/{t1pre,t1c,t2,flair}` (raw — **use the harmonised
  versions from the inference reference H5, not these**)
- `ids` `(N,)` vlen str — **the join key, == `scan_id`**
- `patients/{keys,offsets}` — CSR grouping; `splits/test`, `splits/cv/fold_*`

### 3.2 Label conventions — trap #9

Cohorts declare their own convention via the **`label_system` root attr**:
- `BraTS2021` → `{1=NCR, 2=ED, 4=ET}`
- `BraTS2023` → `{1=NCR, 2=SNFH/ED, 3=ET}` (**no label 4**)

**Branch on the attr. NEVER hard-code 4.** Derive:
```
WT = label > 0
TC = label ∈ {1, 4}   (BraTS2021)   |   label ∈ {1, 3}   (BraTS2023)
ET = label == 4       (BraTS2021)   |   label == 3       (BraTS2023)
```
Raise on an unknown `label_system`. **Assert the derived WT agrees with the
inference H5's `masks/wt`** for the same `scan_id` — that cross-check is your
proof the join is right, and it is cheap. Report any disagreement; a mismatch
is a stop-the-line finding.

### 3.3 The join
`corpus.ids[i] == inference.metadata/scan_id[j]`. Join by **value**, never by
index — the corpus H5 holds the *whole* cohort (e.g. UCSF-PDGM 495 scans),
the inference H5 holds only the **test** rows (50). Build the map.

---

## 4. What to compute

Per (method, cohort, nfe, scan):

**Real arm** (once per scan, shared across methods — cache it):
`{t1pre_harmonised, t1c_real_harmonised, t2_harmonised, flair_harmonised}`
→ segmenter → Dice vs GT for WT / TC / ET.

**Synthetic arm** (per method):
`{t1pre_harmonised, t1c_synthetic_harmonised, t2_harmonised, flair_harmonised}`
→ same segmenter → Dice vs the **same** GT.

Then `Δ_Dice = Dice_real − Dice_synth` per patient per label.

**The real arm is method-invariant.** Compute it **once per scan** and reuse it
across all 16 methods — otherwise you do 16× the segmenter work for identical
results. This is the difference between a 3-hour job and a 2-day one.

Deterministic inference: `torch.no_grad()`, eval mode, fixed seed, no TTA (or
TTA fixed and identical across arms). Any nondeterminism appears directly in Δ
as noise. Log VRAM; free between volumes (`coding-standards.md` rule 13).

---

## 5. Statistics (§6.3)

Collapse scans → patients **first** (`stats.collapse_to_patient`; LUMIERE
72 → 11; contracts §11 trap 4).

- Family: **WT-Dice, TC-Dice, ET-Dice — 3 cells × 8 competitors, Holm-Bonferroni
  per cell** (§6.3).
- Paired Wilcoxon on per-patient `Δ_Dice`, VENA vs each competitor. Two-sided,
  α=0.05.
- Cliff's δ + 10,000-resample patient-stratified bootstrap CI.
- Ablation family (v3b, v3a, S3-LPL-b2c) separate.
- Ring A / Ring B separate. **Ring B has no in-domain segmenter guarantee** —
  the instrument is BraTS-pretrained and BraTS-Africa is exactly the
  distribution shift Adewole 2023 shows costs 5–15 Dice points. Report Ring B,
  but note in `report.md` that the *instrument itself* degrades there, so
  absolute Dice is uninterpretable and only Δ is meaningful.
- Cohorts with no tumour GT, or a scan with an empty ET label, → `NaN`, counted,
  reported. **An empty-GT ET is common in non-enhancing cases and Dice is
  undefined there** — do not score 0, that would be a systematic bias. State the
  convention explicitly in `report.md`.

---

## 6. Where this runs

- **Unit tests + development: locally**, against synthetic fixtures (reuse T0's
  `conftest.py`) plus a synthetic corpus-H5 fixture you add (with a
  `label_system` attr and `{0,1,2,4}` labels, and a `BraTS2023` `{0,1,2,3}`
  variant to prove the branch).
- **Real-data smoke: on Picasso** — the corpus H5s are not mountable here.
  `ssh picasso` works with key auth; the repo is at
  `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA` (commit `1ad2ba4`);
  env at `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena`;
  predictions at `/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference`.
  **Do not push to that repo or submit a full sweep** — rsync your worktree to a
  scratch path of your own, run the smoke on the contracts §13 subset (start
  with **IvyGAP, 5 scans**, then UCSF-PDGM), and report. The orchestrator owns
  the full submission.
  Use `--gres=gpu:1 --constraint=a100 --partition=gpu_partition` if you need a
  GPU (**never `--constraint=dgx`** — that also matches B200 nodes, which break
  the cu124 env; and `--gres=gpu:A100:1` matches nothing, the gres is untyped).
  A 5-scan smoke may fit in an interactive/login-node CPU run — prefer that if
  it works.

---

## 7. Outputs (contracts §9)

`routines/validation/downstream_seg/` → `<output_root>/downstream_seg/<UTC>/`

- **`per_scan/downstream_seg.csv`** — one row per (method, cohort, nfe, scan_id)
  with `patient_id`, `dice_wt_real`, `dice_tc_real`, `dice_et_real`,
  `dice_wt_synth`, `dice_tc_synth`, `dice_et_synth`, and the three `delta_*`.
  Frozen header. (The `*_real` columns repeat across methods by construction —
  that is correct and makes the CSV self-contained.)
- `tables/` — Δ-Dice per method × label, Ring A (headline) at `selection_nfe`,
  with Holm-adjusted p, Cliff's δ, bootstrap CI; the `Dice_real` reference row;
  per-cohort × method; Ring B separate.
- `figures/`
  - **ET-Dice Δ per method** — the clinical-impact figure. Patient-level
    distribution, **significance brackets vs VENA (Holm, per-cell family of 8)**,
    a reference line at Δ=0 (perfect substitutability), C0-Identity as the floor.
  - WT / TC / ET small-multiples.
  - Dice_real vs Dice_synth scatter per method (points = patients, y=x line) —
    shows whether degradation is uniform or concentrated in hard cases.
  - **Qualitative** (required): black background; for one patient, one row per
    method: the synthetic T1c with the **predicted segmentation overlaid**
    (semi-transparent, ET/TC/WT in distinct colours) next to the real-T1c
    prediction and the GT. This is where "contrast that looks right but is
    anatomically wrong" becomes visible. Per-slice `vmin/vmax` anchored to the
    real slice; reuse `select_content_slices` and T0's plotting helpers.
- `report.md`, `decision.json` — including the Appendix-A deviation + rationale,
  the bundle name/version/sha256/channel-order, the empty-ET convention.

---

## 8. Acceptance criteria

- [ ] Import-isolation self-check pasted.
- [ ] **Instrument verified from its own metadata** — bundle name, channel
      order, preprocessing, output convention pasted into your report. Not from
      this plan's description.
- [ ] `label_system` branch tested on **both** BraTS2021 `{1,2,4}` and BraTS2023
      `{1,2,3}` synthetic fixtures.
- [ ] **Derived WT from `masks/tumor` matches the inference H5's `masks/wt`** on
      real data. Report agreement (Dice ≈ 1.0). This is the join proof.
- [ ] **`Dice_real` is sane** (BraTS SegResNet on real 4-channel input should
      give roughly WT ≈ 0.85–0.92, TC ≈ 0.80–0.88, ET ≈ 0.70–0.85 on adult
      preoperative glioma). **If `Dice_real` is near zero, your channel order or
      preprocessing is wrong** — that is the designed canary. Do not proceed
      past it; report it.
- [ ] **`Δ_Dice(C0-Identity)` is the largest of any method** — C0 has no
      enhancement at all, so its ET-Dice must collapse. The designed floor.
- [ ] Real arm computed **once per scan**, not once per method. Prove it
      (timing or a call counter).
- [ ] LUMIERE collapses 72 → 11.
- [ ] Empty-ET convention implemented as NaN + counted, not 0.
- [ ] Real-data smoke ran **on Picasso**; artifact folder inspected; tree + real
      Dice numbers pasted.
- [ ] Ruff clean; unit tests marked `validation`, synthetic + no network.
- [ ] Smoke wall-clock reported and extrapolated to the full sweep
      (16 methods × 467 scans synthetic arms + 467 real arms, at
      `selection_nfe`). The orchestrator needs it to size the SLURM job.

## 9. Notes

- Scope: run the segmenter **only at each method's `selection_nfe`** by default
  (16 method-rows × 467 scans + 467 real = 7,939 segmenter passes). Sweeping all
  45 (method,nfe) pairs would be 21,015 passes for a metric whose purpose is the
  headline shift, not a cost curve. Make NFE a YAML filter; **log the choice**
  (contracts §11 trap 11).
- Engine must be shardable by (method, cohort) for the Picasso fan-out.
- If a cohort has no `masks/tumor`, skip it with a WARNING and record it in
  `decision.json` — do not crash the sweep.
