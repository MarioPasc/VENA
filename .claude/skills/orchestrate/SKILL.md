---
name: orchestrate
description: Run a multi-agent VENA work session as the orchestrator — hold the plan, spawn Opus subagents to write the code, verify every number they report, and iterate until explicit acceptance criteria are met. Encodes the VENA-specific paths (Picasso repos, conda envs, inference tree, analyses roots), the worktree isolation rules for parallel agents, the SLURM traps that silently produce plausible wrong artifacts, and the verification discipline that caught every real defect on 2026-07-16/17. Invoke when a task is large enough to need 2+ agents, when spawning subagents to implement a spec, when the user says "orchestrate", "spawn agents", "run this with subagents", "parallelise this work", or at the start of any multi-hour VENA session with a written plan and acceptance criteria.
---

# Orchestrating a VENA multi-agent session

## Purpose and the one rule

**You hold the plan. Subagents write the code. You verify everything.**

This skill exists because of a measured fact from the 2026-07-16/17 Phase-2
sessions: **every real defect was found by a number that did not reconcile,
never by reading code.** Not one was caught by review, by a passing test suite,
or by an agent's closing report. Each looked correct and matched expectations:

- A merge that "COMPLETED" in 19 s over **54 of 405** shards, publishing full
  tables, all figures, correct provenance, and `n_patients: 393` — a number that
  reconciled *exactly* against the documented total. Caught only by comparing
  its **elapsed time** against the array still running.
- Every p-value in the primary result comparing one arm at the wrong NFE. Caught
  only because two tables in the same artifact disagreed by **0.0005**.
- An agent's `Dice_synth` silently void because its smoke used the one method
  where the corrupted volume happened to be correct.

> **The rule: a subagent's report is a hypothesis, not evidence.** Re-derive
> every load-bearing number yourself, from the artifact on disk. This costs
> minutes. Not doing it costs a paper.

---

## 1. Before spawning anything

1. **Write the acceptance criteria down first**, as a numbered list with
   *checkable* conditions ("`pred_mode`: C0 → harmonised, other 15 → raw"), not
   adjectives ("works correctly"). You stop when they are met — not before, not
   after. Without them you will iterate forever or stop early.
2. **Record the baseline** so a failure can be attributed rather than inherited:
   ```bash
   ~/.conda/envs/vena/bin/python -m pytest -m "not slow and not gpu" -q \
       --basetemp=/home/mpascual/.pytest-tmp-orchestrator | tail -2
   ~/.conda/envs/vena/bin/python -m ruff check src/ routines/ tests/ | tail -2
   df -h /            # MUST NOT be near 0 — see §6
   ```
   The repo is **not** ruff-clean globally (~475 pre-existing errors, ~70
   unformatted files). That is not yours and not the agents'.
3. **Check what is already running** before acting on anything:
   `ssh picasso 'sacct -j <ids> -X -o JobID,State,Elapsed'`. Acting one second
   after an agent complied created a duplicate job once.

---

## 2. Spawning agents

```
Agent(
  subagent_type: "general-purpose",
  model: "opus",
  isolation: "worktree",     # ONLY for fresh work — see the trap below
  run_in_background: true,   # default; several in ONE message run concurrently
)
```
Session effort is inherited — run the session at `xhigh`/`max` for this work.

**Give each agent exactly:** its task-spec path, `01_SHARED_CONTRACTS.md` (or
the project's equivalent fact sheet), its lane, and what it must not touch.
Nothing else — it starts cold with no memory of your session.

### ⚠ The worktree trap that will cost you an agent's work

`isolation: "worktree"` cuts a **fresh worktree from the SESSION BASE commit**,
not from current `main`, and **not** from an existing agent's branch.

- **Fresh, independent, parallel work** → `isolation: "worktree"`. Correct: each
  agent gets its own checkout and branch, so concurrent edits cannot collide.
- **Continuing an existing agent's branch** (its worktree survives on disk after
  the agent is gone) → **do NOT** pass `isolation`. Point the agent at the
  existing directory explicitly:
  ```
  WORKTREE=/home/mpascual/research/code/VENA/.claude/worktrees/agent-<hash>
  ```
  Passing `isolation: "worktree"` here silently discards everything that agent
  wrote.

After **any** merge to `main`, every existing worktree is stale. Tell the next
agent to `git merge main` **first, with proof**:
```bash
cd $WORKTREE && git merge main -m "merge main into <lane>"
git merge-base --is-ancestor <fix-sha> HEAD && echo "HAS FIX"
```

### ⚠ The split-brain import trap (worktrees only)

`vena` is an **editable install path-pinned to the main checkout**. From a
worktree, a naive `python -m pytest` loads `routines` from the worktree and
`vena` from main — half the agent's code, half someone else's, **no error**.
The only correct invocation, which every agent must paste back as proof:
```bash
cd $WORKTREE && PYTHONPATH=$WORKTREE/src ~/.conda/envs/vena/bin/python -c "
import pathlib, vena, routines
wt = pathlib.Path('$WORKTREE').resolve()
for m in (vena, routines):
    p = pathlib.Path(m.__file__).resolve()
    assert p.is_relative_to(wt), f'LEAK: {m.__name__} -> {p}'
print('import isolation OK')"
```
Never `pip install -e .` and never clone the conda env — it is shared read-only.

### The prompt that actually works

Agents skip work phrased as an instruction and comply with work phrased as a
**deliverable**. Two of three agents skipped a real-data smoke when told to "run
the smoke"; none skipped it when told to "report the artifact path and these
specific numbers, which I will check."

Demand, verbatim in the prompt:
- **the artifact path, read back from `readlink LATEST`** — not reconstructed
  (an agent once reported a path 26 s off from the real one; it did not exist)
- **specific real numbers** you have named in advance
- the branch SHA and the import-isolation proof
- a **live monitor** (§4) — agents that launch a job and stop are never woken

Close with: *"Report `STATUS: DONE | QUESTION | PREMISE-FALSE | BLOCKED`. If a
premise here is contradicted by the code or the data, stop and report it with
evidence — that is a successful outcome, not a failure."* Agents have found real
premise errors in orchestrator instructions this way.

**Two correction rounds max, then escalate to the user.**

---

## 3. Verifying what comes back — the part that matters

**Order matters. Check provenance before you read a single number.**

1. **Is the branch post-fix?** A real-but-stale number is more dangerous than a
   missing one, because it is persuasive.
   ```bash
   git -C $WORKTREE merge-base --is-ancestor <fix-sha> HEAD && echo OK
   ```
2. **Does the artifact exist where they said?** `readlink -f <root>/LATEST`.
3. **Does the elapsed time make sense for the work claimed?** This single check
   caught the 54-of-405 merge. `TotalCPU 00:00:00` on a job that "analysed 405
   files" is the tell.
   ```bash
   ssh picasso 'sacct -j <id> --format=JobID,State,Elapsed,TotalCPU,MaxRSS -P'
   ```
4. **Do the artifact's own tables agree with each other?** Two tables in one
   folder disagreeing about the same quantity is how the NFE bug surfaced. If
   two numbers *should* be equal, check that they are.
5. **Re-derive the headline numbers from the per-scan CSV yourself.** Never
   transcribe from an agent's report into a user-facing message.
6. **Do the counts reconcile exactly?** Not approximately. For a full VENA
   sweep: 405 files, 32,715 scans, 653 patients; C4/C5 = 727×6 = 4362 rows,
   C6 = 727×4 = 2908, VENA arms = 727×5 = 3635.

**Blame the environment before the agent.** An agent's "1008 passed" was honest;
318 errors elsewhere were `ENOSPC` from a full disk. Check `df -h /` first.

---

## 4. Compute — all of it on Picasso

Local foreground runs are reaped at the 10-minute tool timeout. The dev box is
an RTX 4060. **Everything real runs on Picasso via SLURM.**

### Canonical paths

| what | path |
|---|---|
| Repo (local) | `/home/mpascual/research/code/VENA` |
| Conda (local) | `~/.conda/envs/vena/bin/python` (3.11) |
| **Picasso repo to RUN FROM** | `fscratch/repos/VENA-validation` — **a real git repo; `git rev-parse` resolves** |
| Picasso shared repo | `fscratch/repos/VENA` — **do not run validation from it**; `git_sha` reports its stale HEAD |
| Picasso conda | `fscratch/conda_envs/vena/bin/python` |
| Predictions | `~/execs/vena/inference/` (405 files, 9 cohorts, 289 G) |
| Analyses out | `~/execs/vena/inference/analyses/` |
| Sweep out | `~/execs/vena/paired_fidelity_sweep/` |
| Logs | `~/execs/vena/logs/` |
| Corpus H5 (GT labels) | `fscratch/datasets/vena/<cohort>/h5/<NAME>_image.h5` — **Picasso only**; MeningD2 is unmounted locally |
| Results archive (local) | `/media/mpascual/Sandisk2TB/research/vena/results/fm/inference/analyses/` |

`ssh picasso` works with key auth. Sync with:
```bash
rsync -az --delete --exclude='.claude/worktrees/' --exclude='artifacts/' \
  --exclude='experiments/' --exclude='__pycache__/' --exclude='*.h5' \
  --exclude='*.pt' --exclude='*.ckpt' ./ picasso:fscratch/repos/VENA-validation/
```

### ⚠ `git_sha: "unknown"` is a deployment bug, not a code bug

An rsync'd **worktree** carries a `.git` *file* pointing back to the dev box, so
`git rev-parse` cannot resolve and every artifact records `"unknown"`. Sync a
**real repo** (include `.git/`) and it fixes every routine at once.

### ⚠ Picasso's `sbatch` wrapper emits ANSI colour codes

Even `--parsable` returns `$'\033[31m\033[0m1604488'`. Interpolated into
`--dependency`, sbatch **ACCEPTS it** and records `Dependency=(null)` — the
dependent job then runs immediately against a partial input. **sbatch accepting
a flag proves nothing.** Strip and verify:
```bash
_clean_job_id() { sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g' <<<"$1"; }
ID=$(_clean_job_id "$(sbatch --parsable ...)")
[[ "$ID" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id" >&2; exit 1; }
scontrol show job "$DEP" | grep -q 'Dependency=(null)' && { scancel "$DEP"; exit 1; }
```
Picasso's `squeue` wrapper also rejects `-h -o '%T'` and prints help instead —
use `/usr/bin/squeue` or `sacct` when scripting.

### Sizing (measured, not guessed)

- CPU analysis: **21.5 CPU-s/volume** (8 cores). Full sweep 32,715 volumes ≈
  **195 CPU-h**; shard one array task per prediction file (405 tasks).
- **CPU jobs bypass the `gres/gpu=32` group cap** that stalls GPU work. §4.2/§4.3
  need no GPU. GPU pinning: `--gres=gpu:1 --constraint=a100`.
- Wall time is set by cluster load, not your `%N`: at 85% full the array
  throttled to ~23 concurrent (`Reason=(Priority)`).

### Monitors — silence is not success

An agent that launches a job and stops is never woken. **You** keep the watch:
```
Monitor(command: "ssh picasso 'while true; do
  s=$(sacct -j <ID> -X -n -o State | head -1 | tr -d \" \")
  case \"$s\" in COMPLETED|FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL)
    echo \"terminal: $s\"; break ;; esac
  sleep 120
done'", persistent: true)
```
The filter **must** match every terminal state, not just success — a monitor
that greps only for the happy path is silent through a crash. Bash
`run_in_background` is capped at a 10-minute timeout; use `Monitor` for long
waits.

---

## 5. Merging

**Serially, re-running the full suite after each.** Branches were cut before
earlier merges, so expect conflicts and shape drift.

**You own merges.** An agent self-merged before its smoke had verified. Its code
was fine, but that call was not its to make: a merge asserts the acceptance
criteria are met, and only the orchestrator knows them.

Never lower the test count. Ruff must be clean **on the touched files only**.

---

## 6. Environment traps (all have bitten)

- **`--basetemp` is mandatory, unique per agent, and never on `/`:**
  `--basetemp=/home/mpascual/.pytest-tmp-<slug>`. The suite writes **31 GB/run**
  (`tests/competitors/*/test_multicohort.py`); pytest keeps 3 runs; `/tmp` is on
  the **137 GB root** and this filled it to **0 bytes**, wedging the machine.
  `--basetemp` **wipes its directory at startup**, so two agents sharing one
  delete each other's fixtures mid-run.
- pytest silently falls back to writing `pytest-of-*/` **under the CWD** when
  TMPDIR is full — 31 GB once landed inside the repo.
- `rm` is not allowlisted and `rm -rf $HOME*` is hard-denied. Use `mv` to
  quarantine; ask the user to delete.
- **The formatter hook strips a "currently unused" import the moment you add
  it.** Add the import and its first use in the *same* edit, or it vanishes and
  you get a `NameError` at runtime.

---

## 7. Design lessons worth more than the mechanics

- **Two arms, two code paths → the arms drift.** Every silent-wrongness bug this
  project has had is an asymmetry: one side filtered, the other not. Route both
  through **one** helper; that makes the bug structurally impossible rather than
  merely fixed.
- **A safety net wired to its own off switch is worse than no net.** A shell
  worker auto-set `--allow-partial` the instant the Python guard tripped.
  A guard that downgrades itself on failure is decoration.
- **Never derive a rule from a sample.** A tier-aware scoring rule inferred from
  3 methods was refuted by the 16-method population. Confirm across the whole
  population before designing around it.
- **Local-only verification cannot validate a cluster-side contract.** A shard
  glob written from the local tree was wrong on Picasso, where a stale smoke
  shard existed and nobody was looking.
- **Report what is computed, not what is tidy.** An "every image method beats
  every latent method" story died on the eighth row. If a mechanism is a
  hypothesis, label it one and keep it out of the artifact.
- **Artifacts must carry their own caveats**, generated by code from the run's
  own data — never hand-edited in, never hard-coded. A stated finding that
  drifts out of agreement with the table beside it is worse than no finding.
  When a caveat's input is absent, the artifact must **say so loudly**; silence
  reads as "no caveat".

---

## 8. Closing the session

1. **Re-run the suite and ruff** on the touched files; both must be clean.
2. **Update `.claude/notes/prompts/HANDOFF.md`** — the single entry point for
   the next session. It must carry: current state, the acceptance criteria and
   whether they are met, every finding the next agent must not re-derive (with
   real numbers), the mistakes made *with the tell that exposed each*, and an
   ordered next-steps list. Correct stale numbers in it when they change.
3. **Copy results** to the local archive and **verify by content hash**, not by
   file listing. Index them: name what is authoritative, and what was
   deliberately not copied and why.
4. **Write memories** for constraints that must survive `/clear` — scientific
   conclusions that would otherwise be re-litigated, and cluster traps. Fix a
   memory the moment it is wrong; a wrong memory is worse than none.
5. Leave the task list explicit about what remains.
