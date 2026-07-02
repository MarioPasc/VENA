# Handoff prompt — LPL batch-2 monitoring agent

Copy the block below into a fresh Claude Code session (recommend `Opus 4.7`, effort `high`) after invoking `/loop Leaving an interval of 30-60 minutes, monitor the jobs and look for errors and performance.` — this file is the run book the agent should consult on wake-up.

Written 2026-07-02 for LPL batch 2 (Picasso jobs 1428339–1428342). If a later batch is launched, adapt the job IDs, run dirs, and expected-behaviour table below.

---

## Copy-paste prompt

```
You are the monitoring agent for VENA's LPL batch-2 tournament running on the Picasso HPC.
This is a long-running task: you check the four Picasso jobs at 30–60 min intervals via
/loop and update the batch-2 journal entry when the runs return. The user has kicked off
the loop with `/loop Leaving an interval of 30-60 minutes, monitor the jobs and look for
errors and performance.`

Your standing role until the batch completes:

1. Every wake-up, run the health check (Section B below).
2. If everything is healthy, briefly report and call ScheduleWakeup with delaySeconds
   in [1800, 3600] (30–60 min).
3. If you see an anomaly (see "Anomaly triggers" — Section D), STOP the loop by
   omitting ScheduleWakeup; call PushNotification with a one-line summary; wait for
   the user's direction. Do NOT try to fix the job yourself — the user makes the
   kill/continue call.
4. When ALL four jobs finish (Status other than R in squeue AND train.log ends
   with "FM-train completed"), pull the results from Picasso and update the
   batch-2 journal entry per Section E. Then STOP the loop.

BEFORE running any tool this session, READ these files in order — they are the ground
truth for what this batch is, what "healthy" looks like, and how to write the results:

  Repo root: /home/mpascual/research/code/VENA/

  Required reading (in order — do not skip):
    A0. CLAUDE.md                                          — project overview + conventions
    A1. .claude/rules/model-coding-standards.md            — FM training, EMA, LPL rules
    A2. .claude/rules/preflight-pattern.md                 — decision.json schema, gate pattern
    A3. .claude/skills/lpl-journal/SKILL.md                — the journal contract you'll enforce
    A4. .claude/notes/changes/lpl/2026-06-28_batch_1_default_recipe.md
                                                           — WHY batch 2 exists; how batch 1 failed
    A5. .claude/notes/changes/lpl/2026-07-02_batch_2_lambda_calibration.md
                                                           — THE living doc you'll update at end
    A6. .claude/notes/changes/decoder_perceptual_loss_s3.md
                                                           — LPL design doc (frozen)
    A7. .claude/notes/changes/decoder_perceptual_loss_s3_analysis_2026-06-20.md
                                                           — LPL 2026-06-20 P0 post-mortem

  Optional (only if you need it):
    - .claude/rules/coding-standards.md
    - .claude/rules/extensibility.md
    - .claude/notes/changes/s1_v3/2026-06-28_s1_v3_results_and_s3_plan.md
                                                           — parent S1 baseline design
    - CLAUDE.md > "Rectified-flow timestep convention" section — α, t_dn, LPL gate math
    - CLAUDE.md > "S1 v2 baseline recipe" — recipe deltas stack under LPL

------------------------------------------------------------------------------
SECTION B — health check (run this at every wake-up)
------------------------------------------------------------------------------

Batch-2 job IDs: 1428339, 1428340, 1428341, 1428342
Batch-2 run dirs on Picasso are under
  /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*

Run these Bash commands in parallel (all via `ssh picasso 'CMD'`). Do NOT paste large
outputs into your reply — summarise.

  # 1. queue status
  ssh picasso 'squeue -j 1428339,1428340,1428341,1428342 -o "%.10i %.35j %.2t %.10M"'

  # 2. latest 6 epochs per arm (extract the key fields only)
  ssh picasso 'for d in /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*; do
      name=$(basename $d | sed "s/.*s3_//;s/_[a-f0-9]*$//")
      echo "--- $name ---"
      grep -E "epoch [0-9]+ done" "$d/logs/train.log" 2>/dev/null | tail -6
    done'

  # 3. error scan (filter false-positive lightning noise)
  ssh picasso 'for j in 1428339 1428340 1428341 1428342; do
      f=$(ls /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s3lpl-b2-*_${j}.err 2>/dev/null | head -1)
      err=$(grep -iE "traceback|OOM|out of memory|assertion|nan loss|inf loss|cuda error|early stopping" "$f" 2>/dev/null | head -3)
      [ -n "$err" ] && echo "!!! $j: $err"
    done; echo done'

  # 4. exhaustive_val cadence completions
  ssh picasso 'for d in /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*; do
      name=$(basename $d | sed "s/.*s3_//;s/_[a-f0-9]*$//")
      echo "--- $name ---"
      ls -d "$d"/exhaustive_val/epoch_* 2>/dev/null | tail -3
    done'

------------------------------------------------------------------------------
SECTION C — what "healthy" looks like
------------------------------------------------------------------------------

Warm-start (epoch 0) magnitudes MUST be within 1% of these values:
  B2-A (λ_max=0.30, α=(1,1), A=[2,5]):     cfm ≈ 0.562   lpl ≈ 2.649
  B2-B (λ_max=0.10, α=(1,1), A=[2,5]):     cfm ≈ 0.562   lpl ≈ 2.650
  B2-C (λ_max=0.12, α=(2,3), A=[2,5]):     cfm ≈ 0.562   lpl ≈ 6.925
  B2-D (λ_max=0.30, α=(1,1), A=[2,3]):     cfm ≈ 0.562   lpl ≈ 1.266   (fast: ~25 min/ep, less GPU mem)

Warmup ramp (linear, warmup_epochs=30): `lam_active = (epoch / 30) × λ_max`
  ep 1 lam for B2-A = 0.010, B2-B = 0.003, B2-C = 0.004, B2-D = 0.010
  ep 30+ lam = λ_max (saturated)

Post-warmup expected trajectory:
  - cfm holds within ±0.02 of its ep-30 value for many epochs (LPL is not
    supposed to disturb CFM at these λs)
  - lpl slowly decreases (5–20 % over 50–100 epochs)
  - train/total_epoch monitor (cfm + λ_max·lpl) declines slowly for at least 50 epochs
    beyond warmup — patience=100 gives room

Throughput:
  ~35 min/epoch for A/B/C (2 A100 40GB, co-resident on same node OK)
  ~25 min/epoch for B2-D (max_block=3 saves ~30% decoder compute)
  Full run: 250 epochs × 35 min = ~146 h ≈ 6.1 days. Fits in 7-day walltime.

------------------------------------------------------------------------------
SECTION D — anomaly triggers (STOP the loop, PushNotification, wait for user)
------------------------------------------------------------------------------

STOP the loop if ANY of the following:

  D1. Job status changes from R to something other than R AND train.log does NOT end
      with "FM-train completed" — job crashed or was requeued. Get exit code from
      `ssh picasso 'sacct -j <JOBID> --format=JobID,State,ExitCode'`.
  D2. stderr contains "Traceback", "CUDA error", "out of memory", "OOM", "NaN loss",
      "inf loss", "assertion" — real crash.
  D3. cfm doubles from its ep-0 value within 20 epochs — LPL is destroying CFM
      convergence.
  D4. lpl explodes (>2× ep-0 value for standard, >2× for region) — feature
      standardization broken.
  D5. Early Stopping fires before epoch 100 (grep err for
      "Monitored metric ... did not improve in the last 100 records") — patience=100
      still not enough; something is very wrong.
  D6. No new epochs recorded in train.log for >2 hours — training hung.
  D7. Batch reads a warm-start value that differs from Section C's table by >5% —
      wrong checkpoint loaded.

Do NOT stop for:
  - Small cfm bounces (±0.05) during first 10 post-warmup epochs — normal transient
  - lpl fluctuations of ±20 % epoch-to-epoch — expected noise
  - One job finishing faster than others — B2-D is intentionally faster
  - Wall-clock mismatch between arms — co-residency effects are OK

------------------------------------------------------------------------------
SECTION E — batch complete: how to write the results
------------------------------------------------------------------------------

A job is DONE when it disappears from squeue AND
its `train.log` last line is `INFO ... FM-train completed`.

When ALL 4 jobs are done, invoke the `lpl-journal` skill (or read
`.claude/skills/lpl-journal/SKILL.md`) and update the batch-2 entry
(`.claude/notes/changes/lpl/2026-07-02_batch_2_lambda_calibration.md`) IN PLACE:

  1. Front-matter: set `status: analysed`, `result_date: <ISO date>`, populate
     `picasso_run_ids: [...]` (the run_id names from Picasso).

  2. Populate Section 6 (Results), Section 7 (Analysis — answer each H1..H4 with
     CONFIRMED / REFUTED / INCONCLUSIVE + evidence), Section 8 (Next-batch
     recommendation: CONTINUE / STOP / PIVOT per the STOP criterion in Section 8
     of the batch-2 entry).

  3. Pull data from Picasso:
       Per-arm training curves — the last ~50 rows of each arm's
       `metrics/train_epoch.csv` are sufficient. Use scp or rsync:
         rsync -av picasso:'/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*/metrics/train_epoch.csv' \
           /tmp/claude-scratch/b2_results/
       Per-arm exhaustive_val — the LAST completed cadence epoch per arm:
         rsync -av picasso:'/mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*/exhaustive_val/epoch_*/metrics.csv' \
           /tmp/claude-scratch/b2_results/

  4. If the local Sandisk is available, rsync the full run dirs to
     `/media/mpascual/Sandisk2TB/research/vena/results/fm/vena/lpl_module/`
     (only if the user asks or if the local dir has free space).

  5. Then STOP the loop (omit ScheduleWakeup). Send a PushNotification with a
     1-sentence outcome: "LPL batch 2 done: best ΔPSNR_ET_UCSF = +X.XX dB by
     <arm>. Journal updated. Recommendation: <CONTINUE/STOP/PIVOT>."

------------------------------------------------------------------------------
SECTION F — how the /loop mechanic works here
------------------------------------------------------------------------------

Dynamic mode: `/loop` fires with no interval → you self-pace.
  1. Do the health check (Section B).
  2. Report briefly in the assistant message (2–4 sentences + a compact table).
  3. Call ScheduleWakeup with:
       delaySeconds: 1800–3600 (30–60 min)
       prompt: the exact original /loop input
              ("/loop Leaving an interval of 30-60 minutes, monitor the jobs and look for errors and performance.")
       reason: one specific sentence, e.g.
              "60 min catches ~1.5 additional epochs on the standard arms"
  4. Return control (turn ends when ScheduleWakeup returns).

Cache-aware guidance (see the ScheduleWakeup tool description):
  - 30–60 min falls past the 5-minute prompt-cache window, so each wake-up pays
    one cache miss. That is acceptable at this cadence.
  - Do NOT go below 1800 s during warmup — the epoch cadence is 35 min so
    faster checks return the same numbers twice.

------------------------------------------------------------------------------
SECTION G — one-off tools you may need
------------------------------------------------------------------------------

  - Full sacct on all 4:
      ssh picasso 'sacct -j 1428339,1428340,1428341,1428342 --format=JobID,JobName%40,State,Elapsed,Timelimit,ExitCode'
  - Tail worker stdout (progress bar):
      ssh picasso 'tail -30 /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s3lpl-b2-a-lambda030-standard_1428339.out'
  - Show P0 EarlyStopping monitor state:
      ssh picasso 'grep -iE "best score|improved|stop" /mnt/home/users/tic_163_uma/mpascual/execs/vena/logs/vena-s3lpl-b2-a-lambda030-standard_1428339.err | tail -10'
  - Compact last N epochs across all arms:
      ssh picasso 'for d in /mnt/home/users/tic_163_uma/mpascual/execs/vena/experiments/2026-07-02_11-*_b2_*; do
          name=$(basename $d | sed "s/.*s3_//;s/_[a-f0-9]*$//")
          echo "--- $name ---"
          grep -E "epoch [0-9]+ done" "$d/logs/train.log" 2>/dev/null | tail -10
        done'

------------------------------------------------------------------------------
SECTION H — hard rules
------------------------------------------------------------------------------

  - NEVER call scancel on Picasso jobs unless the user explicitly asks.
  - NEVER modify the batch-2 YAML or launcher files while the jobs run — a
    Picasso resubmit could pick up the new YAML.
  - NEVER rsync from local to Picasso while jobs run (they may re-read config).
  - NEVER edit the LPL design docs (frozen historical record).
  - ALWAYS use ScheduleWakeup, not sleep loops.
  - If Bash output would exceed ~50 lines, filter with grep/tail/awk before
    running; keep the main context clean.
  - If the user asks a question mid-loop, answer it and continue the loop
    (do not stop unless they say so).

You are ready. Read files A0–A7 in order, then call SECTION B, then report
and schedule wakeup.
```

---

## How to use this handoff

1. Push the repo (this file included) to GitHub.
2. In a new Claude Code session on any machine that has:
   - the repo cloned at `/home/mpascual/research/code/VENA/`
   - SSH access to `picasso` (alias in `~/.ssh/config`)
3. Type `/loop Leaving an interval of 30-60 minutes, monitor the jobs and look for errors and performance.`
4. Then paste the copy-paste prompt above.
5. The agent will read Sections A0–A7 first, then start monitoring.

## Also written

- `.claude/notes/changes/lpl/2026-07-02_batch_2_lambda_calibration.md` — living doc (status: launched)
- `.claude/notes/changes/lpl/2026-06-28_batch_1_default_recipe.md` — batch-1 post-mortem
- `.claude/skills/lpl-journal/SKILL.md` — journal contract enforced across future batches
