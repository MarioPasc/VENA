# TASK P1 — `brats-ped-backfill`: Phase-1 inference for the missing pediatric OOD cohort

**Read `01_SHARED_CONTRACTS.md` §1–§6 for the data layout.** You do **not** need
the Phase-2 metric sections — you are not writing analysis code.

| | |
|---|---|
| **Model** | Opus 4.8, effort `max` |
| **Isolation** | **git worktree**, branch `task/brats-ped-backfill` |
| **Depends on** | nothing — start immediately, run in parallel with all Phase-2 work |
| **Lane (you own)** | `routines/fm/inference/configs/picasso_ped_*.yaml`, `routines/fm/inference/slurm/runs/launcher_inference_picasso_ped.sh` (new files only) |
| **Do not touch** | `routines/fm/inference/engine.py`, `cli.py`, `configs/models/benchmark_full.yaml`, any existing config or worker, `src/vena/**`, `pyproject.toml`, `CLAUDE.md`, `.claude/rules/**`. **New files only.** If you believe you must modify existing inference code, **stop and report** — that is a scope change the orchestrator owns. |

---

## 1. Why this exists

Phase-1 inference ran 16 methods × 8 cohorts. **BraTS-PED (260 pediatric
patients) was never run**, though it is in both corpus registries as
`role=test_only`. It is the *pediatric* OOD ring the validation proposal §3
Ring-B calls for (Kazerooni et al. 2023, arXiv:2305.17033) — smaller heads,
different myelination contrast, a genuinely distinct distribution from the
adult-Western training pool.

Ring B on disk is currently only BraTS-Africa-Glioma (95) + BraTS-Africa-Other
(51) = 146 patients. With BraTS-PED it becomes 406, matching the plan.

Phase-2 analysis is being built in parallel and **discovers cohorts from disk**.
When your predictions land in the results tree, they flow into the analysis with
no code change. **You are not blocking anyone** — do this correctly, not fast.

---

## 2. Verified starting facts (checked 2026-07-16, do not re-derive)

- Picasso reachable: `ssh picasso` (key auth, `BatchMode=yes` works).
- Repo on Picasso: `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/VENA`,
  at commit **`1ad2ba4`** = current local `main`.
- Env: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/vena`.
- Output root of the existing run:
  `/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference` (289 G, 5 shards).
- **BraTS-PED inputs already exist — no encoding needed:**
  ```
  fscratch/datasets/vena/brats_ped/h5/BraTS_PED_image.h5     5.1G
  fscratch/datasets/vena/brats_ped/h5/BraTS_PED_latents.h5   1.9G
  ```
  Both are at exactly the paths `corpus_picasso.json` declares.
- Its registry entry: `role=test_only`, `n_patients=260`, `n_scans=260`,
  `longitudinal=false`, modalities `[t1pre, t1c, t2, flair]`,
  **`label_system: "BraTS2023"`** (labels {1=NCR, 2=ED, **3**=ET} — *not* the
  {1,2,4} of the BraTS2021 cohorts. Not your problem, but do not "fix" it.)
- Partition/gres: `--partition=gpu_partition --gres=gpu:1 --constraint=a100`.
  **Never `--constraint=dgx`** (also matches B200 nodes, which break the cu124
  env). **Never `--gres=gpu:A100:1`** (the gres is untyped; matches nothing).

---

## 3. What to do

### 3.1 Mirror the existing shard split

The 5 existing shards partition the 16 methods purely for wall-clock (C6 at
NFE=1000 dominates). Mirror that split, restricted to BraTS-PED:

| New `run_id_tag` | Methods | NFE |
|---|---|---|
| `picasso_ped_a_cheap` | C0-Identity, C1-pGAN-{t1pre,t2,flair}, C2-ResViT, C7-3D-Latent-Pix2Pix | [1] |
| `picasso_ped_b_vena` | VENA-S1-v3a, VENA-S1-v3b, VENA-S1-v3b-rw, VENA-S3-LPL-b2c | [1,2,5,10,20] |
| `picasso_ped_c_latent` | C4-3D-DiT, C5-T1C-RFlow | [1,2,5,10,20,100] |
| `picasso_ped_d_lddpm` | C6-3D-LDDPM | [5,10,20,1000] |
| `picasso_ped_e_syndiff` | C3-SynDiff-{t1pre,t2,flair} | [4] |

Expected output: **45 (method,NFE) pairs × 1 cohort = 45 prediction files**
+ 5 reference copies of `BraTS-PED.h5` (one per shard, as the existing tree does).

Copy the closest existing config (`configs/picasso_shard_*.yaml`) as the
skeleton. Keep `output_root` **identical** to the existing runs
(`/mnt/home/users/tic_163_uma/mpascual/execs/vena/inference`) so the new shard
dirs land beside the existing five and the Phase-2 glob
(`<root>/*/predictions/*/*/nfe_*.h5`) picks them up for free.
**New `run_id_tag`s ⇒ new, write-disjoint shard dirs. Never reuse an existing
tag — that would risk clobbering 289 GB of finished work.**

### 3.2 Cohort filter — verify the semantics, don't assume

`_CohortsFilter` is `{cv_test: list[str] | None, test_only: list[str] | None,
exclude: list[str]}`. **Read `routines/fm/inference/engine.py` and determine
what `None` vs `[]` actually mean** before writing the YAML — `None` plausibly
means "all", `[]` plausibly means "none", and getting it backwards either
re-runs all 8 cohorts (destroying 289 GB of work if tags collide, or burning
days of A100 if they don't) or runs nothing.

Target: **BraTS-PED only.** Prove the filter is right with a **smoke first**
(§3.4) — never by reading the code alone.

### 3.3 Disable the Phase-1 comparison figures

Set `figure.enabled: false`. Rationale: Phase 2 renders its own figures, and the
figure tensor cache is what OOM-killed `a_cheap` on host RAM (it scales with
method count — commit `1ad2ba4` bumped it to `--mem=300G`). Disabling removes
the OOM class entirely and saves wall-clock. If you keep figures for any shard,
size `--mem` from that shard's history, not from the default.

### 3.4 Smoke before the full run — non-negotiable

Use the engine's own `smoke` block (`{enabled: true, n_patients_per_cohort: 1,
use_selection_nfe_only: true}`) with a throwaway `run_id_tag`
(e.g. `smoke_ped_check`) into a **scratch `output_root` of your own**, not the
production one. Confirm:
- exactly BraTS-PED is selected, nothing else;
- one H5 per (method, cohort, selection_nfe) is produced and passes
  `assert_predictions_valid`;
- the reference file is written and `references_h5` resolves;
- root attr `ring == "B"`;
- `metadata/patient_id` and `metadata/scan_id` are populated (schema 1.1 fix);
- the harmonisation contract holds (inside brain ⊆ [0,1], exterior ≡ 0).

Then delete the scratch output and submit the real shards.

### 3.5 Get the code to Picasso

The Picasso repo is at `1ad2ba4`; your configs are new files. Do **not** push to
`main` and do **not** `git pull` on the cluster (that would move the repo under
any running job). Rsync your new config + launcher into the Picasso repo's
`routines/fm/inference/` tree, or into a scratch checkout of your own, and run
from there. State exactly what you did in your report so the orchestrator can
reproduce it. The config must also land in git on your branch, for the record.

### 3.6 Submit and monitor

Follow the existing launcher/worker convention (`slurm/runs/`): launcher on the
login node sets `REPO_DIR`, `CONDA_ENV_NAME`, `CONFIG_PATH` and
`sbatch --export=ALL,... worker.sh`; the worker activates conda, sets
`PYTHONPATH=$REPO_DIR/src:$REPO_DIR`, runs
`$PYTHON -m routines.fm.inference.cli $CONFIG_PATH`.

Reuse `worker_inference_picasso_full.sh` if it is parameterised enough;
otherwise add a **new** worker. Do not edit the existing one — other work
depends on it.

**Do not use `set -u` in any worker that activates the `vena` env** — it
contains `gxx_linux-64`, whose activation script dereferences an unbound
`SYS_SYSROOT` and kills the job. Use `set -eo pipefail`.

Submit the 5 shards as concurrent jobs, mirroring
`launcher_inference_picasso_shards.sh`. Sizing: the existing full-sweep worker
used `--time=4-00:00:00 --cpus-per-task=8 --mem=150G`. This run is **1 cohort of
260 patients instead of 8 cohorts of 467 scans**, so it is far smaller — size
`--time` from the existing shards' actual runtimes (check their
`logs/inference.log`) scaled by scan count, and add headroom. `d_lddpm`
(C6 @ NFE=1000) is the long pole.

Monitor with `squeue`. **Do not poll in a tight loop** — background a single
`until` loop or check on a sensible interval.

---

## 4. Acceptance criteria

- [ ] Smoke ran into a scratch output root, proved BraTS-PED-only selection, and
      was cleaned up.
- [ ] 5 shards submitted; job IDs reported.
- [ ] On completion: **45 prediction files** under
      `<output_root>/picasso_ped_*/predictions/`, covering all 16 methods with
      the NFE grids in §3.1, plus the BraTS-PED reference file per shard.
      Verify with a find/count, and report the count.
- [ ] Every produced H5 passes
      `vena.inference.h5_writer.assert_predictions_valid` / `assert_references_valid`.
- [ ] `metadata/patient_id` has 260 unique values; `metadata/scan_id` 260 unique
      (BraTS-PED is `longitudinal: false`, so scans == patients).
- [ ] Root attr `ring == "B"` on every file.
- [ ] The existing 5 shards are **untouched** — verify
      `du -sh <output_root>/picasso_shard_*` still totals ~289 G and the file
      count is still 360 + 40. **Report these numbers.** Damaging the existing
      tree is the one unrecoverable failure mode here.
- [ ] A `decision.json` per new shard, with the corpus/model sha256s.
- [ ] Configs + launcher committed to your branch. New files only —
      `git diff --stat main` must show no modifications to existing files.

## 5. Report back

`STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED` plus: job IDs, the file
count, the untouched-tree verification numbers, the wall-clock per shard, and
anything in this plan you found to be false.

**Transfer to local is NOT your job** — report that the predictions are ready on
Picasso and stop. The orchestrator decides when/whether to rsync 45 files
(~35 GB) down.
