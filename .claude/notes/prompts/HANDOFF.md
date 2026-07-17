# VENA Phase-2 Validation — SESSION HANDOFF

*Rewritten 2026-07-17 by the orchestrator at the end of the second session.
The 2026-07-16 version is superseded; its §11 acceptance criteria are **all
met**. Read this top to bottom before touching anything.*

> Companion docs, still current: `01_SHARED_CONTRACTS.md` (verified facts +
> traps — **still wins over every other doc**), `00_ORCHESTRATION.md`,
> `02_validation_core.md`, `03_paired_fidelity.md`, `04_spatial_residual.md`,
> `05_downstream_seg.md`, `06_brats_ped_backfill.md`.

---

## 0. TL;DR — the five things that matter now

0. **Read `.claude/skills/orchestrate/SKILL.md` before spawning any agent.** It
   encodes this session's traps: the worktree-isolation rule that silently
   discards an agent's work, split-brain imports, the sbatch ANSI dependency
   trap, the verification order, and the measured sizing numbers.
1. **Phase 2 is built, merged, and the primary sweep is DONE.** All three
   routines on `main`, suite **1132 passed**, ruff clean on validation files.
   Results copied to
   `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/`
   (hash-verified; see its `README.md`).
2. **The paper's headline claim did not survive the full sweep.** VENA's
   tumour win is entirely attributable to an **oracle GT mask** no competitor
   receives. **§4 — read it, it is the most important thing in this file.**
3. **Region weighting (`-rw`, the pre-registered headline model) is a net
   negative.** It is beaten by its own ablation `VENA-S1-v3b`.
4. **Two sweeps remain**: `spatial_residual` (CPU, ~20 CPU-h) and
   `downstream_seg` (GPU). `paired_fidelity` is complete.
5. **Trust no artifact whose elapsed time you have not checked.** A merge that
   "COMPLETED" in 19 s published a plausible, wrong result (§6).

---

## 1. Current state — what is DONE

| | Status |
|---|---|
| `main` | suite **1132 passed**, 1 skipped, 4 deselected; ruff clean on `src/vena/validation/`, `routines/validation/`, `tests/validation/` |
| `paired_fidelity` (§4.2/§4.5/§4.7) | merged, smoked, **full sweep COMPLETE** |
| `spatial_residual` (§4.3) | merged, smoked; **sweep not run** |
| `downstream_seg` (§4.4) | merged, smoked; **sweep not run** |
| `preregister` | **frozen on Picasso** (§3) |
| BraTS-PED backfill | **landed and verified** — Ring B is now 406 |

**All four agent worktrees are merged.** They can be pruned
(`git worktree remove`), but check `git worktree list` first.

---

## 2. Verified facts — UPDATED, do NOT re-derive

**BraTS-PED landed 2026-07-17.** The grid is now **405 prediction files**
(45 (method,NFE) pairs × **9** cohorts), **727 scans / 653 patients**.

| Ring | cohorts | scans | patients |
|---|---|---:|---:|
| **A** | UCSF-PDGM 50, BraTS-GLI 127/114, UPENN-GBM 62, IvyGAP 5, **LUMIERE 72/11**, REMBRANDT 5 | **321** | **247** |
| **B** | BraTS-Africa-Glioma 95, BraTS-Africa-Other 51, **BraTS-PED 260** | **406** | **406** |
| **TOTAL** | | **727** | **653** |

Ring B = 406 is what `validation_fairness.md` §4 always claimed. That doc-drift
row in `01_SHARED_CONTRACTS.md` §10 is **closed by data**, not a doc edit.

`selection_nfe` is now **frozen in the pre-registration artifact** and mirrors
`vena.validation.registry.SELECTION_NFE`. Do not keep private copies (§6).

---

## 3. The pre-registration (the P3 integrity artifact) — FROZEN

```
picasso:~/execs/vena/inference/analyses/preregister/2026-07-17T09-33-09Z   (LATEST)
  git_sha                     7e60709
  ring_partitions.json sha256 0d888e9df9b1183b8ee732a6e1149572d4fda337cdfba80a4ba398ef55a60f72
  405 prediction files, smoke_loginexa skipped
  cross_check: PASSED — 9/9 cohorts checked against the corpus registry, 0 skipped
  selection_nfe frozen for all 16 methods
```
The cross-check only **runs** on Picasso (the corpus image H5s are not mounted
locally — it silently degrades to a WARNING here). That is why §11.6 insisted
on Picasso, and it passed.

---

## 4. ⚠ THE CRITICAL FINDING — VENA's win is an oracle-mask artefact

**`VENA-S1-v3b-rw` receives the ground-truth WT mask as ControlNet
conditioning. No competitor does** (`01_SHARED_CONTRACTS.md` §10, fairness
concern ②). The full sweep quantifies exactly what that is worth.

### 4a. The tumour win evaporates without the mask (Ring A, 247 patients)

| method | MAE_wt |
|---|---|
| VENA-S1-v3b-rw (**with GT mask**) | **0.0948** ← beats all 8 competitors, all Holm-significant |
| C2-ResViT | 0.1129 |
| C1-pGAN-t1pre | 0.1158 |
| C5-T1C-RFlow | 0.1234 |
| **VENA-S1-v3a (NO mask)** | **0.1283** ← **loses to all three** |

Mask conditioning buys **−0.0335 MAE_wt** and **+0.166 SSIM_wt**
(δ=+0.73), and **+0.0001 MAE_brain** — i.e. nothing whole-brain. Strip it and
VENA beats neither ResViT, pGAN, nor T1C-RFlow in the tumour.

### 4b. §4.4 confirms it independently — with a smoking gun

Paired 61-patient set, ΔDice_ET (lower = less degradation):

| method | ΔDice_ET | patients where **synth beats real** |
|---|---|---|
| C0-Identity | 0.578 | 10/61 |
| VENA-S1-v3a | 0.435 | 9/61 |
| VENA-S1-v3b-rw | **0.073** | **20/61 (33%)** |

**In a third of patients the synthetic image segments the tumour better than
the real T1c.** A synthetic image cannot genuinely beat ground truth at
locating ground truth. That is the conditioning mask leaking into the metric.
§4.4 leaks hardest because, unlike MAE, it is scored against the very object
used as conditioning.

**What this means for the paper.** v3b-rw is an **upper bound** ("given a
perfect tumour mask"); v3a is the **lower bound** (no mask). The deployable
number is between them — a mask from a segmenter run on *pre-contrast*
modalities. This is a legitimate and interesting framing, but it **must be
stated**. `VENA-S1-v3a` must appear beside every v3b-rw number.

**Do not try to "fix" the leak.** It is a property of the frozen Phase-1
predictions. Report it.

### 4c. Region weighting is a net negative

| | MAE_brain | SSIM_brain | MAE_wt | SSIM_wt |
|---|---|---|---|---|
| VENA-S1-v3b (no rw) | **0.0915** | **0.589** | 0.0965 | 0.570 |
| VENA-S1-v3b-rw (**the pre-registered headline**) | 0.0955 | 0.554 | 0.0948 | 0.574 |
| | v3b wins p=0.0022 | v3b wins p<1e-5 | **n.s. p=0.997** | **n.s. p=0.281** |

`-rw` costs whole-brain MAE and SSIM and buys **nothing measurable** in the
tumour — neither MAE_wt nor SSIM_wt separates. An honest negative ablation.
**The headline cannot be swapped to v3b** — the pre-registration is frozen and
picking the winner post hoc is the oracle selection §4.1 forbids.

On the primary endpoint the pre-registered arm ranks **7th of 16**, behind
ResViT, all three pGAN panels, and **its own ablations v3b (0.0915) and v3a
(0.0953)**.

### 4d. The whole-brain deficit is a VAE tax, not a conditioning failure

`mae_brain`: v3a 0.0953 vs v3b-rw 0.0950 → **p=0.97, identical.** Masking does
nothing whole-brain. The deficit tracks *tier*:

- **Image-domain**: C2-ResViT **0.0778**, C1-pGAN **0.0823** ← beat VENA
- **Latent-domain**: **VENA 0.0950** < C7 0.0994 < C5 0.1016 < C6 0.1112 < C4 0.1976

**Every latent method loses to every image method on whole-brain MAE; VENA is
the best of the latent tier.** 4× MAISI compression costs bulk-parenchyma
fidelity that image-space regression does not pay — and C1-pGAN achieves it
from **one** modality against VENA's three.

### 4e. LPL is inert — confirmed on 247 patients

`VENA-S3-LPL-b2c` ≈ v3b-rw everywhere: mae_brain **p=1.00**, mae_wt **p=0.997**,
ssim_wt **p=0.794**, all non-significant. The only survivor is ssim_brain
(p=0.023) at δ=0.02 — negligible. Independently confirms the stored decision to
stop the LPL programme. **Do not reopen it.**

*(The buggy NFE pass — see §6 — had manufactured significant LPL differences on
mae_wt (p=0.0125) and ssim_wt (p=3.4e-4). Both were artefacts. Correcting the
bug removed four spurious significances, every one involving a multi-NFE arm,
and made this conclusion cleaner.)*

### 4f. §4.3 — the C0-ceiling framing is dead; ρ_S is the statistic

C-noT, patient-collapsed:

| comparison | p_adj | Cliff's δ | |
|---|---|---|---|
| VENA vs C0 — **ρ_S** | 6.3e-07 | −0.54 | VENA's residual far less structure-correlated |
| VENA vs C5 — **ρ_S** | 1.5e-11 | −0.70 | same |
| VENA vs C5 — Conc(5%) | **0.86** | 0.03 | **does not discriminate** |

C0 is the ceiling on **nothing**. Conc(5%) is ≈3.0, not <1 — the pre-fix
"errors avoid bright voxels" result was a harmonisation artefact and has
evaporated. **ρ_S discriminates; Conc does not.** Proposal §4.3.3/§4.3.5 owe a
documented revision.

*(The old "VENA beats SOTA by 43%" came from the throwaway audit script, not
the routine. It does not reproduce. Delete that number from your mental model.)*

---

## 5. The completed sweep

```
picasso:~/execs/vena/paired_fidelity_sweep/analyses/paired_fidelity/2026-07-17T11-54-55Z  (LATEST)
  git_sha 285f203 · 405 shards · 32,715 scans · 653 patients · n_bootstrap 10,000
  skipped_smoke_shards: ["smoke_loginexa"]
  pred_mode: C0-Identity → harmonised 727/727; all 15 others → raw. Zero exceptions.
  Array 1604488: 405/405 COMPLETED, 0 failed.  Merge re-run 1607921 (NFE fix).
```
`report.md` now carries its own caveats **above** the methodology: the oracle
gap (computed, not asserted) and the full primary-endpoint ranking with the
VENA arm marked wherever it lands. Both derive from the run's own rows, so they
cannot drift out of agreement with the tables beside them. If a future run omits
`VENA-S1-v3a`, the section says so loudly rather than falling silent.

Earlier merges of the same shards are superseded: `10-27-27Z` (pre-NFE-fix
statistics) and `11-47-42Z` (pre-NFE-fix ranking). `VOID_2026-07-17T09-51-07Z`
is the 54-of-405 partial (§6). **Only `11-54-55Z` is valid.**
Counts reconcile exactly: C4/C5 727×6=4362, C6 727×4=2908, VENA 727×5=3635.

Cost, measured: **21.5 CPU-s/volume** (8 cores, TotalCPU 03:47:50 / 635 vols)
→ 32,715 volumes ≈ **195 CPU-h**. Wall ≈ 55 min at ~120 concurrent. The
cluster was 85% full (`Reason=(Priority)`), so the old "~10 min wall" estimate
assumed an empty cluster.

---

## 6. ⚠ Mistakes made THIS session — do not repeat

**The 19-second merge.** Merge job 1604489 "COMPLETED" while the array was
still running, merged **54 of 405** files, and published `LATEST`. Three
independent safety mechanisms failed at once:
1. **Picasso's `sbatch` wrapper injects ANSI colour codes into `--parsable`
   output.** Interpolated raw, the dependency became
   `afterok:<ESC>[31m<ESC>[0m1604488`. **sbatch ACCEPTED it** and recorded
   `Dependency=(null)`. *Never trust sbatch accepting a flag — read it back
   with `scontrol`.*
2. **`worker_paired_fidelity_merge.sh` auto-set `--allow-partial`** whenever
   shards were missing — a safety net wired to its own off switch.
3. `LATEST` published it anyway.

It was dangerous because it looked **perfect**: full tables, all figures,
correct `pred_mode_counts`, correct `skipped_smoke_shards`, real `git_sha` —
and **`n_patients: 393`**, which reconciles exactly against the *old* pre-PED
total in the previous handoff. Nothing invited suspicion. It was caught only by
checking **elapsed time** (19 s, TotalCPU 00:00:00) against the array state.
All three are now fixed and pinned by `tests/validation/test_sweep_launcher_guards.py`.
Evidence kept at `analyses/paired_fidelity/VOID_2026-07-17T09-51-07Z/`.

**The NFE asymmetry — every p-value in the primary result was wrong.**
`_statistical_pass` filtered each **competitor** to its pre-registered selection
NFE but averaged **VENA** over all five of its NFEs (1/2/5/10/20). So every
Holm-corrected p-value compared VENA-averaged-over-NFE against a competitor at a
single NFE — not the pre-registered comparison, and biased toward VENA on the
primary endpoint (0.0950 averaged vs 0.0955 at its own nfe=5; low-NFE samples
are smoother and score better on MAE). It removed four spurious significances
once fixed. **The only visible symptom was that `c0_sanity.csv` (which filtered
correctly) and `headline_table.csv` disagreed by 0.0005 about VENA's own
mae_brain** — two tables in one artifact disagreeing about the same quantity.
Nothing crashed. The lesson: *the asymmetry existed only because the two arms
were written as two code paths.* Both now route through
`_series_at_selection_nfe`, and the orchestrator reproduced the identical bug in
the report's ranking helper within the same hour by reaching for a plain
`groupby("method")`. **Any reduction of an arm to one value per patient goes
through that helper.** Guarded by `tests/validation/test_selection_nfe_symmetry.py`.

**Other traps closed:**
- **`git_sha: "unknown"`** was not a code bug: the Picasso checkouts were
  rsync'd *worktrees* whose `.git` file dangles back to the local box. Fixed by
  deploying a **real** repo at `fscratch/repos/VENA-validation` (§7). Run
  everything from there.
- **`selection_nfe: {}`** — the pre-registration froze an empty dict. Now
  mirrored from the registry, with stop-the-line assertions.
- **Duplicate `_SELECTION_NFE`** in `spatial_residual.py` — removed. Reference
  `registry.SELECTION_NFE` **via the module**; `load_partitions` rebinds it, so
  a from-import captures a stale binding.
- **The formatter strips a "currently unused" import** the moment you add it.
  Add the import and its first use in the same edit, or it vanishes.

**Agents':** V2 merged itself to `main` before its smoke verified (the
orchestrator owns merges). V3 reported an artifact path 26 s off from the real
one (`10-23-10Z` vs `10-23-36Z`) — **always read the path back from disk**.
Both then produced good work when sent back with specifics.

---

## 7. Where things are

| What | Path |
|---|---|
| **How to run a session like this one** | **`.claude/skills/orchestrate/SKILL.md`** — read it before spawning any agent. Encodes the worktree/split-brain traps, the sbatch ANSI trap, the verification order, and the sizing numbers. |
| Picasso repo to run from | `fscratch/repos/VENA-validation` — **a real git repo; `git rev-parse` resolves.** Keep it synced with `rsync -az --delete` from `main`. |
| Picasso shared repo | `fscratch/repos/VENA` @ `1ad2ba4` — **do not run validation from it**; `git_sha` would report `1ad2ba4`. It carries untracked `routines/validation/` + `src/vena/validation/` copies an agent left; **ask the user to delete them** (two sources of truth). |
| Predictions | `picasso:~/execs/vena/inference` (405+45 files, 9 cohorts) |
| Sweep output | `picasso:~/execs/vena/paired_fidelity_sweep/` |
| Analyses (Picasso) | `picasso:~/execs/vena/inference/analyses/{preregister,spatial_residual,downstream_seg}/` |
| **Results archive (local)** | `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/` — the four authoritative artifacts, copied 2026-07-17 and **verified byte-identical by content hash**. `README.md` there names what is authoritative, retains the VOID partial as marked evidence, and lists the six superseded runs deliberately left on Picasso (with why). Layout is flattened: the sweep's `paired_fidelity/` sits beside the other three. |

---

## 8. NEXT — the task list

Read `.claude/skills/orchestrate/SKILL.md` before spawning agents for any of it.

### P0 — blocks the paper's remaining results

1. **Run the `spatial_residual` sweep (§4.3).** CPU-only. Its smoke is
   single-threaded (TotalCPU≈Elapsed, ~3.45 CPU-s/volume → **~20 CPU-h**), so
   shard it one array task per prediction file (405) as `paired_fidelity` does,
   or it runs ~9 h serially. **Run from `VENA-validation`** — its own D4 is
   still unfixed and its smoke `decision.json` says `git_sha: "unknown"`.
   Reuse `launcher_paired_fidelity_sweep.sh`'s ID-sanitising + `scontrol`
   dependency check (§6) — do not re-introduce the ANSI bug.
2. **Run the `downstream_seg` sweep (§4.4).** Needs a **GPU** → queues behind
   `gres/gpu=32` (`--gres=gpu:1 --constraint=a100`). ~4 GPU-h with PED.
   **MUST include `VENA-S1-v3a`** — it is the honest §4.4 comparator (§4b).
   Its `report.md` must state the oracle confound and count negative deltas.
3. **Verify both sweeps** against the §11-style criteria: `pred_mode` (C0 →
   harmonised, rest → raw), `skipped_smoke_shards` contains `smoke_loginexa`,
   LUMIERE 72→11, real `git_sha`, counts reconciling to 405/32,715/653.
   **Check elapsed time against the work claimed** before believing any of it.
4. **Copy both to the local archive** and update its `README.md` index
   (`/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/`).
   Verify by content hash.

### P1 — integrity gaps that are not wrong today but could become wrong

5. **Wire `load_partitions`.** Exported but **never called**, so `COHORT_RING`
   stays `{}` and the drift check in `ring_of_cohort` ("Raises on disagreement
   so silent drift is caught immediately") can never fire — the H5 attr always
   wins by default rather than by verification, and the frozen
   `ring_partitions.json` is write-only. Needs a `preregistration_path` config
   param across the three validation engines. Not a correctness bug today:
   rings come from Phase-1's authoritative H5 attrs and preregister
   cross-checked 9/9.
6. **`spatial_residual`'s `git_sha: "unknown"`** — fixed for free by running
   from `VENA-validation` (item 1), but confirm it in the sweep artifact.
7. **Audit the other two routines for the NFE asymmetry** (§6). `paired_fidelity`
   is fixed and guarded. `spatial_residual` has its own `_filter_to_selection_nfe`
   (correct, and now sourcing `registry.SELECTION_NFE`); `downstream_seg` sets
   `selection_nfe_only: true`. **Verify both actually reduce every arm at its own
   selection NFE** — the bug is an asymmetry between arms, so check both sides.

### P2 — writing, and decisions that are the user's

8. **Reconcile the proposal** — §4.1 to the scoring-space rule (Phase-1
   double-harmonisation; under-saturation is reported, not absorbed; C4-3D-DiT
   reaches 38% of the reference's dynamic range → Table S1), and §4.3.3/§4.3.5
   to "**ρ_S is the discriminator, not Conc**". **Do not edit
   `/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md` without the user.**
9. **Frame the contribution** (§4). The pre-registered primary endpoint says
   VENA is *not* best — it ranks 7th of 16, behind ResViT, three pGAN panels,
   and its own ablations. What survives: the ablation-quantified mask effect,
   the ρ_S vessel-fidelity result, and "best latent-space method".
10. **Test the VAE-ceiling hypothesis, or drop it.** "Latent methods pay a
    compression tax" is NOT established — C3-SynDiff is image-domain and loses to
    three latent methods (§4d). The clean experiment: **the MAISI VAE's own
    reconstruction MAE on real T1c**, which is the floor no latent method can
    beat. If VENA ≈ that floor, the claim is made; if not, drop it.
11. **Ring B / OOD analysis.** All the §4 numbers above are **Ring A**. Ring B is
    now 406 patients (incl. BraTS-PED 260, pediatric) and is a separate endpoint
    family (§6.4). Nobody has looked at it yet — the sweep computed it.
12. **Matched-NFE sub-table and the cost-quality Pareto** (SHARED_CONTRACTS
    §4.2): NFE=5 across the few-step latent tier is the only apples-to-apples
    generative-formulation comparison; `cost_table.csv` + `figures/cost_quality_pareto.png`
    exist but nobody has read them.
13. **Limitations to state, not fix** (SHARED_CONTRACTS §10): VENA's `_sample`
    used unseeded `torch.randn_like` → predictions are not bit-reproducible;
    C4–C7 condition on 2 modalities, VENA/ResViT on 3, pGAN/SynDiff on 1.

### Housekeeping (`rm` is denied to the agent — ask the user)
```
rm -rf /home/mpascual/.pytest-tmp-*            # per-agent pytest temp (local)
ssh picasso 'rm -rf fscratch/repos/VENA/routines/validation fscratch/repos/VENA/src/vena/validation'
# The four merged agent worktrees can be pruned; check `git worktree list` first:
git worktree remove .claude/worktrees/agent-<hash>
```

---

## 9. The one-paragraph version

Phase 2 is built and merged (suite 1132, ruff clean), the pre-registration is
frozen on Picasso with a passing 9/9 cross-check, BraTS-PED landed so Ring B is
406, and the `paired_fidelity` sweep is complete over 405 files / 32,715 scans
/ 653 patients. The science did not go the way the last handoff predicted: the
"43% win" was an audit-script artefact, VENA is **not** best on the
pre-registered primary endpoint (C2-ResViT is — every latent method loses to
every image method, a VAE-compression tax), region weighting is a net negative
against its own ablation, and VENA's tumour win is **entirely** the
ground-truth mask it receives and no competitor does — quantified twice, most
damningly by 20/61 patients where the synthetic image segments the tumour
better than the real one. What survives is real and publishable: VENA is the
best latent-space method, its residual is essentially uncorrelated with anatomy
where C5's is not (ρ_S 0.017 vs 0.391), and the mask-conditioning effect is
cleanly isolated. Two sweeps remain. **Check elapsed time before believing any
artifact.**
