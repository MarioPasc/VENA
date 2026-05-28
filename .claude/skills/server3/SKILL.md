---
name: server3
description: Launch and monitor VENA training/eval trials on GPU server 3 (icai-server, 2x RTX 4090). Use whenever a smoke test or trial must run on a real GPU.
---

# Run a trial on GPU server 3

Server 3 (`icai-server`, host `icaigpuserver3`) has **2× RTX 4090 24 GB**. The
local workstation has the repo but **no `pytorch_lightning`**, so any run or
module-level test executes on the server. Use this skill for every GPU trial.

## Fixed facts

| Item | Value |
|---|---|
| SSH host | `icai-server` (alias in `~/.ssh/config`; `User mariopascual`, port 33430, `ControlMaster` on) |
| Repo on server | `/home/mariopascual/projects/VENA` |
| Python / env | `~/.conda/envs/vena/bin/python` (PL 2.6.5 + editable `vena`) |
| Latents H5 | `/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_latents.h5` |
| Image H5 (real T1c GT) | `/media/hddb/mario/data/GLIOMAS/UCSF_PDGM/h5/UCSFPDGM_image.h5` |
| MAISI checkpoints | `/media/hddb/mario/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/{diff_unet_3d_rflow-mr.pt,autoencoder_v2.pt}` |
| Experiment outputs | `/media/hddb/mario/experiments/<UTC>_s1_<hash>/` |
| Scratch logs | `/media/hddb/mario/smoke_logs/` (create with `mkdir -p` — does not persist) |

## Workflow

1. **Sync code with `rsync`, never `git push`/`pull`** (the FM tree is a working
   copy on the server; the user manages git separately). Verify with checksums:
   ```bash
   rsync -avR <changed files...> icai-server:/home/mariopascual/projects/VENA/
   # then: md5sum <files> locally vs `ssh icai-server "cd .../VENA && md5sum <files>"`
   ```
   For deletions, `rsync` without `--delete` will not remove server files — `ssh ... rm` them explicitly.

2. **Run the non-GPU tests first** (catches import/contract breaks fast):
   ```bash
   ssh icai-server 'cd /home/mariopascual/projects/VENA && ~/.conda/envs/vena/bin/python -m pytest tests/model/fm/ -q -m "not gpu and not slow"'
   ```
   (`pytest` may need a one-time `~/.conda/envs/vena/bin/python -m pip install -q pytest`.)

3. **Launch detached** so the trial survives the SSH session. Pick the GPU(s):
   - **Single-GPU trial** (most smokes): pin to one GPU.
     ```bash
     ssh icai-server 'cd /home/mariopascual/projects/VENA && mkdir -p /media/hddb/mario/smoke_logs && \
       LOG=/media/hddb/mario/smoke_logs/run_$(date +%Y%m%d_%H%M%S).log && \
       CUDA_VISIBLE_DEVICES=0 nohup ~/.conda/envs/vena/bin/python -m routines.fm.train.cli <config.yaml> > "$LOG" 2>&1 & disown; echo "$LOG"'
     ```
   - **Dual-GPU trial** (training + async exhaustive validation): launch with
     **both GPUs visible** — do NOT set `CUDA_VISIBLE_DEVICES`. Training uses
     `cuda:0`; the `ExhaustiveValLauncher` subprocess uses `cuda:1` (config
     `exhaustive_val.device`).

4. **Monitor without blocking.** Poll process exit, not `decision.json`
   (`decision.json` is a *startup* provenance stub, not a completion marker). Use
   a background until-loop:
   ```bash
   until ! ssh icai-server 'pgrep -f "routines.fm.train.cli" >/dev/null'; do sleep 15; done; echo DONE
   ```
   The true done signals are `ema_final.ckpt` + "FM-train completed" in the run's
   `logs/train.log`. Note: after `Trainer.fit`, the process may stay alive briefly
   while `on_fit_end` **joins** the last exhaustive-val subprocess — that is normal.

5. **Verify the run directory** (`/media/hddb/mario/experiments/<run>/`):
   - `logs/train.log` — full structured run log.
   - `metrics/train_step.csv`, `metrics/train_epoch.csv` — clean, fully populated.
   - `checkpoints/` — `ema_epoch_*`, `ema_best.ckpt`, `ema_final.ckpt`, `last.ckpt`.
   - `exhaustive_val/epoch_NNN/` (if enabled) — `metrics.csv`, `timing.csv`,
     `latent_preds.h5`, `figure_{best,worst}.png`, `subprocess.log`, `job.yaml`.
   - `exhaustive_val/gpu_usage.log` — per-device memory at each validation launch.

## Proving both GPUs are busy (dual-GPU runs)

GPU0 training is **bursty** (sub-second compute spikes), so instantaneous
`utilization.gpu` snapshots understate overlap. Use **memory co-residency** as
the robust signal: sample both GPUs and find timestamps where GPU0 holds the
training footprint (~several GB) while GPU1 holds the validation footprint. Run
a standalone sampler **as its own process** (a subshell with `kill -0 $PID`
loses scope over SSH — use a fixed-iteration loop):
```bash
ssh icai-server 'F=/media/hddb/mario/smoke_logs/gpu_ts.log; for i in $(seq 1 120); do \
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader >> "$F"; sleep 2; done'
```
Then count timestamps with both indices > ~1 GB. For a definitive snapshot,
`nvidia-smi --query-compute-apps=pid,used_memory` shows which PID is on which GPU.

## Config conventions

Smoke configs live in `routines/fm/train/configs/` (`smoke_short`, `smoke_full`,
`smoke_polish`, `smoke_exhaustive`). 4 train subjects, batch 1 ⇒ 4 micro-batches
per epoch; `max_steps` counts **optimiser** steps. Keep trials short but exercise
every path (checkpoint rotation, NFE sweep, exhaustive cadence).

$ARGUMENTS
