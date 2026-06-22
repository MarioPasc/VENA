# Model Coding Standards (FM generator)

Conventions for code under `src/vena/model/fm/` (the SWAN-conditioned latent
flow-matching generator) and its training/eval routines under `routines/fm/`.
These extend — not replace — `coding-standards.md`, `preflight-pattern.md` and
`external-deps.md`. Read those first. Where this file and the proposal
(`/media/mpascual/Sandisk2TB/research/vena/docs/proposal.md`) disagree, the
proposal wins.

## Library / routine / Lightning split

1. **Library in `src/vena/model/fm/<area>/`, importable and unit-testable.**
   Areas: `controlnet/` (conditioning assembler, downsamplers, losses, MAISI
   ControlNet), `ema/`, `inference/` (samplers + timing probe), `maisi/` (frozen
   trunk loader + config), `metrics/` (latent + image + region masks), `sampler/`
   (rectified-flow noising), `eval/` (image-space exhaustive validation helpers),
   `lightning/` (the LightningModule, DataModule, callbacks).
2. **`lightning/module.py` is the only `LightningModule`.** It wires trunk +
   ControlNet + RFlow + composite loss + EMA. The VAE is always frozen. The
   trunk is controlled by `trunk_config.trainable`:
   - **`trainable=True` (project default).** The trunk is unfrozen and
     fine-tuned jointly with the ControlNet (cf. TumorFlow). The optimiser is
     built over `self.controlnet.parameters()` **plus** `self.trunk.parameters()`
     in one group; a second `WarmupEMA` (`self.trunk_ema`) tracks the trunk and
     sampling uses the EMA trunk shadow. The fine-tuned trunk is registered as
     `self._trunk_module` so its weights round-trip through Lightning's native
     `state_dict` (PL 2.x restores model weights *after* `setup()`); the trunk
     EMA shadow is reloaded from a separate `trunk_ema_snapshot.pt` by the
     exhaustive job. This path is **single-shot — not resume-safe** (the trunk
     EMA is built in `setup()`, after Lightning's checkpoint load): do not rely
     on `resume_from` for unfrozen runs without first hardening trunk-EMA
     restore.
   - **`trainable=False`.** Canonical frozen-backbone recipe: optimiser over
     `self.controlnet.parameters()` only, no trunk EMA, trunk never written to
     checkpoints. Byte-identical to the original frozen path.
   MONAI's MAISI U-Net injects ControlNet residuals **in-place**, which breaks
   autograd once the trunk carries gradients; `maisi/grad_safe.py` rebinds those
   two adds out-of-place (numerics unchanged) and is applied only when trainable.
3. **Routines are thin engines** (`routines/fm/<name>/engine.py`) that wire
   library code to a YAML config and write the artifact. Follow
   `preflight-pattern.md` invariants (one positional YAML arg, frozen Pydantic
   config with `from_yaml`, `Engine.run() -> Path`, no heavy work at import).

## Training loop

4. **Training-only on the primary GPU.** In-process validation is **offloaded**
   to the async second-GPU job (see §"Exhaustive validation"). The training
   process builds the module with `region_resolver=None`, `vae_decoder=None`, and
   the Trainer runs with `limit_val_batches=0`. Do not re-enable an in-process
   validation loop without a clear reason.
5. **`ema_best` is selected on the epoch-aggregated training loss**
   (`train/total_epoch`, `mode="min"`, `save_on_train_epoch_end=True`) because
   validation is asynchronous. Caveat to honour: train loss ≠ synthesis quality,
   so always retain `last.ckpt` + every epoch checkpoint and treat the
   exhaustive-val PSNR/SSIM curves as the authoritative model-selection signal.
6. **EMA updates exactly once per optimiser step.** `on_train_batch_end` fires
   per *micro-batch*; gate the EMA update on `trainer.global_step` *advancing*
   (`if step <= self._last_ema_step: return`) so gradient accumulation does not
   over-update the shadow. The same gate (`global_step` strictly greater than the
   last-written step) drives the per-step train CSV.
7. **Gradient norms are logged in `configure_gradient_clipping`** (fires once per
   optimiser step, grad-accum-safe): pre-clip, post-clip, and a `grad_clip_active`
   flag. Do not claim "post-clip" from `on_before_optimizer_step` — that hook runs
   pre-clip.

## Metric / artifact logging (perfectly-logged, no white cells)

8. **We write our own CSVs; Lightning's `CSVLogger` is disabled** (`logger=False`)
   to avoid the sparse, val-duplicating wide `metrics.csv`. `callback_metrics` is
   still populated independent of the logger, so `ModelCheckpoint` and our
   callbacks work.
9. **Per-step metrics → `metrics/train_step.csv`; per-epoch aggregate →
   `metrics/train_epoch.csv`** (`TrainMetricsCSV`). The step CSV discovers its
   columns once (all `train/*` keys + `lr`) and *freezes the header*, so every row
   is fully populated. A value logged in the module's `on_train_batch_end` (e.g.
   `ema_decay`) is read live from the module by the callback, not from
   `callback_metrics` (the callback fires before the module hook).
10. **Callback ↔ module hook ordering.** Lightning fires
    `Callback.on_validation_epoch_end` / `on_train_epoch_end` *before* the
    `LightningModule` hook of the same name. A callback that needs a collapsed
    aggregate must call a pure `pl_module.collapse_*()` method (no mutation); the
    module clears the accumulator in its own (later) hook. Never read a raw
    accumulator from a callback expecting collapsed keys.
11. **Self-contained runs.** The engine attaches a plain `FileHandler` to the root
    logger writing `logs/train.log` so the run captures its own log regardless of
    stdout redirection. The run directory holds only what it produces:
    `checkpoints/`, `logs/`, `metrics/`, and (when enabled) `exhaustive_val/` —
    do not create empty placeholder dirs.

## Sampling, metrics, decoding

12. **One sampler, reused everywhere.** `inference/euler.py` (`EulerSampler`)
    drives the MONAI rectified-flow scheduler from `sampler/rflow.py`. Validation
    and the exhaustive job sample via the module's `_make_ema_call()` closure so
    EMA sampling is identical to training. NFE timing uses
    `inference/timing.py::NFETimingProbe` (CUDA-synced `section()` contexts);
    drop the first sampler step as warm-up when `nfe > 1`. **Load-bearing
    pitfall (2026-06-20):** when the scheduler runs with
    `use_timestep_transform=True`, MONAI's `set_timesteps` requires
    `input_img_size_numel` to be passed through or `timestep_transform` divides
    `None / int` and every per-patient exhaustive-val pass silently fails with
    a WARNING (the empty `metrics.csv` trap in
    `.claude/skills/server3/SKILL.md`). The patched
    `routines/fm/exhaustive_val/engine.py` sampler construction reads
    `cfg.rflow.base_img_size_numel` and falls back to `48*56*48` (VENA's
    brain-box latent). Any new sampler that wraps `RFlowScheduler` must plumb
    the same kwarg through.
13. **Region masks via GPU ops, never CPU loops.** `metrics/regions.py` derives
    `wt_dilated` with `F.max_pool3d` (stride 1, `padding=k//2`) — exact binary
    dilation for an all-ones element, no NumPy/CPU round-trip. New region
    derivations stay on-device.
14. **Latent metrics** (`metrics/latent.py`): masked MSE / L1 / cosine, one scalar
    per batch element, epsilon-clamped denominators. **Image metrics**
    (`metrics/image.py`): whole-volume PSNR/SSIM on `[0,1]` volumes, `data_range`
    must be `1.0`. The masked-SSIM mean-fill is a rough training-time proxy and
    degenerates on tiny regions — the exhaustive job uses whole-volume SSIM
    instead.
15. **Intensity-space parity is mandatory for image metrics.** A decoded
    prediction is in the VAE `[0,1]` space (MAISI decodes to the box
    `(B,1,*target_shape)`). To compare against a real image, apply the *same*
    `percentile_normalise(lower=0, upper=99.5, foreground_only=True)` over the
    skull-stripped brain foreground (nonzero voxels only) that the encoder applied
    to its input — multi-cohort training and encoding always use `foreground_only=True`
    because all stored volumes are skull-stripped. Never compare decoded `[0,1]`
    against raw intensities or use `foreground_only=False` for skull-stripped inputs.

## Exhaustive validation (async, second GPU)

16. **The expensive image-space validation runs as a detached subprocess on a
    second GPU** (`routines/fm/exhaustive_val/`, launched by
    `ExhaustiveValLauncher`). Training never blocks: at the cadence epoch the
    callback snapshots the EMA shadow `state_dict` to a small `.pt`, writes a
    self-contained job YAML, and `Popen`s the CLI on `device` (default `cuda:1`).
17. **Concurrency: at most one validation in flight (skip-if-busy), join at
    `on_fit_end`.** Do not block training to wait for a previous validation and do
    not launch concurrent validations onto one GPU.
18. **Artifacts per cadence epoch** live under `exhaustive_val/epoch_NNN/`:
    `metrics.csv` (per patient × NFE: PSNR, SSIM, latent MSE/L1/cosine, gen/decode
    seconds), `timing.csv` (per-NFE aggregates), `latent_preds.h5`
    (schema-versioned, per `h5-design-principles.md`), and `figure_{best,worst}.png`
    (best/worst patient by mean SSIM across NFE). A `gpu_usage.log` records both
    devices' memory at each launch (overlap evidence).
18a. **Comparison-figure conventions (2026-06-20 global overhaul, applies to
     every regime that uses `exhaustive_val`):** `render_comparison_figure` in
     `vena.model.fm.eval.exhaustive` produces a black-background panel with one
     row per NFE sorted by SSIM **descending** (best-quality synthesis
     immediately below the real T1c row). Each NFE row's ylabel carries
     `NFE=N (t=...s) / SSIM=... / PSNR=... dB` derived from a per-(patient, NFE)
     index built from `metric_rows` in the engine. The suptitle is bare
     `"<TAG> — <patient>"` — no aggregate SSIM. Each column anchors the synth
     `imshow(vmin, vmax)` to the **real slice's** per-slice `(min, max)` so
     pixels are visually comparable against the same reference window
     (constant-real-slice fallback `[0, 1]`). Don't reintroduce the `mean_ssim`
     kwarg — the unit test `test_render_figure_signature_dropped_mean_ssim`
     guards against it.

## Frozen weights & adapters

19. **Never edit `src/external/*` and never write to checkpoint paths**
    (`external-deps.md`). Adapters wrap the frozen MAISI VAE / trunk in
    `src/vena/model/fm/` (e.g. the ControlNet built around the frozen trunk via
    `init_from_trunk` + zero-init output projections, then scale-ramped to ~1
    over `output_scale_ramp.ramp_steps` via `OutputScaleRampCallback` — see
    `CLAUDE.md::S1 v2 baseline recipe`). Log the resolved checkpoint
    SHA-256 at first load.

## Cross-cutting helpers (mandatory imports)

20. **MAISI primitives come from `vena.common`**, not from
    `vena.model.autoencoder.maisi.*`. Decoding goes through
    `vena.common.decode.decode_box` (exhaustive val / external eval) or
    `vena.common.decode.decode_depth_identity` (in-process training proxy).
    Never inline `MaisiDecoder.decode` in a routine engine — see
    `.claude/rules/extensibility.md`.
21. **`MultiCohortLatentDataModule` is the only training DataModule.** The
    legacy single-cohort `LatentH5DataModule` was removed; a guard test
    (`tests/model/fm/test_legacy_dataset_removed.py`) fails if it returns.
    Engines build the DataModule from `cfg.data.corpus_registry` only.
22. **External code never writes to `LightningModule` private attrs.** The
    exhaustive-val engine calls `module.compute_val_conditioning(batch)`; do
    not regress to `module._val_cond = module.conditioning(batch)`.
23. **TF32 matmul precision is set once at engine entry**:
    `torch.set_float32_matmul_precision("high")` early in `Engine.run()`,
    before any model is built. ~10% step-time gain on A100/RTX-4090 at no
    measured cost to FM training numerics.
24. **Aggregation helpers** (`_finite_mean`, `_finite_std` in `module.py`) are
    module-level. Do not redefine inside `collapse_*` loops (per
    `coding-standards.md` rule 16).

## ControlNet scale-ramp + conditioning assembler

22a. **`MaisiControlNet.output_scale` is the canonical post-zero-init lever.**
     The buffer is non-persistent (`persistent=False`); the ramp formula is a
     pure function of `trainer.global_step`, so on resume the
     `OutputScaleRampCallback` recomputes the correct value at the first batch
     and the buffer state does not need to round-trip through `last.ckpt`. The
     scalar is multiplied into every element of `down_block_res_samples` and
     into `mid_block_res_sample` **inside `MaisiControlNet.forward`** —
     upstream of the `maisi/grad_safe.py` out-of-place residual-add patch — so
     gradient checkpointing and the trainable-trunk path are byte-identical
     when `output_scale == 1.0`.
22b. **Downsampler `out_channels` contract** (2026-06-20).
     `AbstractDownsampler.out_channels` returns `None` by default ("same as
     input — kind-based default applies"). Channel-lifting operators
     (currently `LiftTo4ChDownsampler`; future SPADE-style modulators)
     override the property with a concrete `int`. The assembler's
     `channels_per_spec` consults this and never assumes a fixed mask channel
     count. New downsamplers that lift channels MUST override `out_channels`
     or the assembler underestimates `total_channels` silently and the
     ControlNet's first conv is built with the wrong `in_channels`. The
     `mask:wt:zero_out` operator (used by S1 to drop mask conditioning while
     preserving the channel slot for S2/S3 warm-start) is stateless: it
     returns `out_channels=None` and the slot stays at `mask_channels=1`.

## Testing

25. Co-locate `tests/model/fm/test_<area>.py`. Pure helpers (metric math, slice
    selection, CSV writers, H5 round-trip, normalisation parity) must be tested
    without checkpoints; mark checkpoint/GPU paths `gpu`/`slow`. When a callback's
    contract changes (e.g. reads a `collapse_*()` method), update its test to the
    new contract in the same change.
26. **Every test file declares its marker explicitly.** Either
    `pytestmark = pytest.mark.unit` at module level or `@pytest.mark.<marker>`
    per test. Tests with no marker are invisible to the marker-filtered fast
    suite (`-m "not slow and not gpu"`) — that has burned us before.
27. **Removal of a public symbol gets a guard test.** When deleting a class or
    function that lived in `__all__` (e.g. `LatentH5DataModule`), add a test
    that asserts the name is no longer importable. The guard makes a
    re-introduction a one-line CI failure.
