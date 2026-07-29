"""Flow-matching training routine engine (v2).

Implements the full training_routine.md spec:

* Run directory layout (`experiments/{run_id}/` per §2.2) — atomic, self-contained.
* Validation cadence (per-epoch single-NFE + sweep every K epochs).
* Qualitative-latent dumps, NFE timing, per-region metrics, EMA, RNG-state-in-checkpoint
  resume, SIGTERM-aware checkpointing.
* YAML schema mirroring §2.3 with a top-level ``regions:`` block declaring per-region
  source so the audit trail makes the measurement scope explicit.

The engine is a thin orchestrator: it wires Lightning's Trainer to the
:class:`vena.model.fm.lightning.module.FMLightningModule`, the
:class:`vena.model.fm.lightning.data.MultiCohortLatentDataModule`, the VAE
decoder, and the suite of custom callbacks in
:mod:`vena.model.fm.lightning.callbacks`.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vena.data.augment import AugmentationTracker, VariantTracker, build_pipeline_from_yaml
from vena.data.h5.shared import now_iso_utc, resolve_git_dirty, resolve_git_sha, sha256_file
from vena.data.registry import load_registry
from vena.model.fm.lightning import FMLightningModule, MultiCohortLatentDataModule
from vena.model.fm.lightning.callbacks import (
    TRUNK_EMA_SNAPSHOT_FILENAME,
    BestCheckpointCallback,
    ExhaustiveValLauncher,
    GradClipValidityCallback,
    SigtermHandler,
    TrainMetricsCSV,
    TrunkEMASnapshotCallback,
    VENACheckpointCallback,
)
from vena.model.fm.maisi.config import TrunkConfig
from vena.model.fm.metrics import RegionSpec
from vena.preflight.cohort_dedup import (
    DedupDecisionSchemaError,
    assert_dedup_decision_valid,
    build_allowlists,
)

from .exceptions import InvalidResumeFromError, PreflightGateError
from .runner import generate_run_id, normalise_tag, write_provenance

logger = logging.getLogger(__name__)


# =============================================================================
# Resume semantics (see ``.claude/rules/preflight-pattern.md`` §"resume_from").
#
# The YAML field ``run.resume_from`` carries three distinct intents that the
# old "scan ``experiments_root`` newest-first" heuristic conflated. The
# 2026-06-10 Picasso incident showed why that's dangerous: an s2 job that
# meant to start fresh from the MAISI FM base trunk silently latched onto a
# sibling s1 run's ``last.ckpt`` because both runs shared the workspace.
#
# We now classify the YAML value explicitly:
#
# * ``baseline`` / null   → BASELINE   : new dir, no checkpoint load (default).
# * ``latest`` / ``best`` → CONTINUE   : continue THIS recipe in place — glob
#                                       scoped to ``*_{stage}_{tag}_*/`` so a
#                                       different recipe's checkpoint is
#                                       invisible. Same dir, full state.
# * ``<run_id>``          → WARM_START : new dir; load weights from the named
#                                       prior run, leave optimiser/EMA/sched
#                                       untouched. The s1→s2 experiment path.
# * absolute ``.ckpt``    → WARM_START : same as above for external checkpoints.
# * anything else         → raise InvalidResumeFromError (no silent fallback).
#
# WARM_START → CONTINUE auto-promotion: if the YAML carries a WARM_START
# ``resume_from`` and a recipe-matching sibling dir (``*_{stage}_{tag}_*/``)
# already exists with ``checkpoints/last.ckpt``, the resolver promotes to
# CONTINUE on that sibling. This is the Picasso walltime-resubmit case — the
# first launch creates the warm-started dir; subsequent launches of the same
# YAML continue in place rather than re-warm-starting from the external
# source. ``decision.json``'s ``resume_source`` still records the original
# YAML value so the audit trail of intent is preserved.
# =============================================================================


# 4-field run_id: <UTC>_<stage>_<tag>_<sha>; tag/stage are ``[a-z0-9_]+``.
_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-z0-9_]+_[a-z0-9_]+_[0-9a-f]{6,}$")


class ResumeMode(str, Enum):  # noqa: UP042 — keep str+Enum for PL ≤2.x compat
    """Classification of ``run.resume_from``; see comment block above."""

    BASELINE = "baseline"
    CONTINUE = "continue"
    WARM_START = "warm_start"


def _classify_resume_from(rf: str | None) -> ResumeMode:
    """Classify ``run.resume_from`` (string) into a :class:`ResumeMode`.

    Path-existence is intentionally NOT checked here — that's the resolver's
    job. We only need to know which branch the engine should take.
    """
    if rf is None or rf == "" or rf == "baseline":
        return ResumeMode.BASELINE
    if rf in ("latest", "best"):
        return ResumeMode.CONTINUE
    if _RUN_ID_RE.match(rf):
        return ResumeMode.WARM_START
    if Path(rf).is_absolute():
        return ResumeMode.WARM_START
    raise InvalidResumeFromError(
        f"run.resume_from={rf!r} is unrecognised. Use one of: 'baseline' (or null) "
        "for a fresh run from MAISI; 'latest'/'best' to continue the same recipe in "
        "place; a run_id like '2026-06-10_10-24-10_s1_fft_cfm_9441bf91' to warm-start "
        "a new run from a prior run; or an absolute path to a .ckpt file."
    )


# =============================================================================
# Pydantic schema (training_routine.md §2.3 + ``regions:`` block)
# =============================================================================


class _RunCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = "s1"
    # Recipe identifier within a stage (``fft_cfm``, ``lora_r16_contrastive``,
    # ``lora_r16_contrastive_cfg``, …). Embedded in the run_id so that
    # ``resume_from: latest`` globs ``*_{stage}_{tag}_*/`` and never picks up
    # a sibling job of a different recipe. Required — there is no sensible
    # default once the experiments_root is shared across recipes.
    tag: str
    resume_from: str | None = None
    seed: int = 1337
    device: str = "cuda"
    precision: str = "bf16-mixed"
    full_determinism: bool = False

    @field_validator("tag")
    @classmethod
    def _normalise_tag(cls, v: str) -> str:
        # Single source of truth lives in routines.fm.train.runner.run_id —
        # this validator delegates so the format is enforced identically at
        # config-load time and at run_id generation time.
        return normalise_tag(v)


class _DataCfg(BaseModel):
    """Training data configuration.

    The legacy ``latents_h5`` single-cohort key was removed in the pre-long-run
    hardening pass; every run flows through ``corpus_registry``. To run a
    single-cohort experiment, write a registry JSON listing only that cohort
    (see ``routines/fm/train/configs/corpus/``).
    """

    model_config = ConfigDict(extra="forbid")
    corpus_registry: Path
    tau: float = 0.5
    max_train_patients_per_cohort: int | None = None
    fold: int = 0
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    max_train_subjects: int | None = None
    max_val_subjects: int | None = None
    # Latent-space augmentation. ``augmentation_config_path`` points at a
    # YAML built per ``vena.data.augment.config.SCHEMA_VERSION``. When set,
    # the engine builds an ``AugmentationPipeline`` and passes it as the
    # train transform; the ``AugmentationTracker`` callback writes
    # ``metrics/augmentations_per_epoch.csv``. When
    # ``preflight_decision_path`` is also set, the loader gates the pipeline
    # by the preflight's ``latent_safe_augmentations`` allowlist and
    # fast-fails if any requested augmentation is not safe.
    augmentation_config_path: Path | None = None
    preflight_decision_path: Path | None = None
    # Cohort-deduplication gate. Points at the ``decision.json`` produced by
    # ``routines.preflights.cohort_dedup`` (schema v1.0). When set, the
    # gate ``_assert_preflight_gates`` validates the file, checks the corpus
    # registry SHA-256 matches, and the engine passes per-cohort allow-lists
    # into ``MultiCohortLatentDataModule`` so train/val/test scan IDs are
    # filtered before sampling. Mandatory when supplied.
    dedup_decisions_path: Path | None = None
    # S3 stage gate. Points at the ``decision.json`` v1.0 produced by
    # ``routines.preflights.decoder_lpl_profile``. Mandatory when
    # ``run.stage == 's3'``; the gate validates the schema and checks
    # that every active variant is in ``allowed_variants``.
    decoder_lpl_decision_path: Path | None = None
    # Offline image-domain augmentation bank (per
    # ``routines.offline_aug.maisi``). When True, every cv cohort in the
    # registry must carry ``latent_aug_h5``; the DataModule wraps the train
    # cohort dataset in ``OfflineAugmentedLatentH5Dataset`` and draws a
    # variant ∈ {v0..vK} per ``__getitem__`` with ``variant_weights``.
    # Val/test never see augmented data.
    use_offline_augmented_data: bool = False
    variant_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "v0": 0.2,
            "v1": 0.2,
            "v2": 0.2,
            "v3": 0.2,
            "v4": 0.2,
        }
    )
    # Soft 2-channel mask [TC, NETC] serving policy (task-20 / S2 T-13).
    # "none" (default) → m_tc_soft / m_netc_soft absent; every existing run
    #   YAML is back-compat with no change required.
    # "oracle_soft" → reads masks/tumor_latent_soft (schema 2.1.0, cached).
    # "predicted"   → reads masks/tumor_latent_pred (written by task-18).
    # "derived"     → clip(NETC+ET, 0, 1) from masks/tumor_latent (aug-safe).
    # Absent group with oracle_soft/predicted → hard raise, never a warning.
    mask_source: Literal["none", "oracle_soft", "predicted", "derived"] = "none"
    # HDF5 key for the brain foreground mask used in brain-masked metrics
    # (psnr_db_brain, ssim_brain, etc.). The only valid value is
    # "masks/brain_latent"; this field exists so ``_assert_run_invariants``
    # can catch any attempt to switch to the real_box>0 fallback (which IS
    # derived from the real T1c and leaks target information into metrics).
    brain_mask_key: str = "masks/brain_latent"

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_latents_h5(cls, data: Any) -> Any:
        if isinstance(data, dict) and "latents_h5" in data:
            raise ValueError(
                "data.latents_h5 was removed in the pre-long-run hardening pass. "
                "Use data.corpus_registry pointing at a registry JSON with the "
                "single cohort entry instead. See routines/fm/train/configs/corpus/."
            )
        return data


class _OutputScaleRampCfg(BaseModel):
    """Scale-ramped zero-init on the ControlNet output projections.

    Modulates :attr:`MaisiControlNet.output_scale` from ~0 to ~1 over
    ``ramp_steps`` optimisation steps via a sigmoid (see
    :class:`OutputScaleRampCallback`). Disabled by default for byte-identical
    backward compatibility with the retired S1 recipe.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    ramp_steps: int = 5000
    steepness: float = 10.0


class _InputConcatCfg(BaseModel):
    """S1 v3 (2026-06-22) trunk channel-concat conditioning.

    When ``enabled``, the listed ``cond_latents`` (e.g. ``[t1pre, t2, flair]``)
    are channel-concatenated to the noisy T1c latent at the trunk's first
    convolution. The trunk's ``conv_in`` ``in_channels`` is widened from 4
    to ``4 + 4 * len(cond_latents) + len(cond_masks)`` via
    :func:`vena.model.fm.maisi.conv_in_expand.expand_conv_in`, with the
    additional channels zero-initialised so the trunk's step-0 behaviour is
    bit-identical to the pretrained MAISI baseline.

    Disabled (default ``enabled=false``) is byte-identical to the S1 v2 path.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    cond_latents: list[str] = Field(default_factory=list)
    cond_masks: list[str] = Field(default_factory=list)  # reserved for ablations
    zero_init_new_channels: bool = True
    # Optional sigmoid ramp on the new channels' weights. ``ramp_steps=0``
    # disables the ramp (zero-init alone is enough for step-0 correctness).
    ramp_steps: int = 0
    ramp_steepness: float = 10.0


class _ControlNetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # S1 v3 (2026-06-22): primary on/off switch. ``False`` = Variant A (no
    # ControlNet at all); module sets ``self.controlnet = None`` and the
    # trunk is the sole prediction path. ``True`` = legacy / Variant B
    # (3-channel mask conditioning) — the canonical S1 v2 path defaults here.
    enabled: bool = True
    # S1 v3: gate :meth:`MaisiControlNet.init_from_trunk`. Variant B sets
    # ``False`` because the 3-channel-mask cond_embedding has no useful
    # warm-start from the trunk's 4-channel-input encoder. Default ``True``
    # preserves S1 v2 behaviour.
    init_from_trunk: bool = True
    # ``conditioning_inputs`` is OPTIONAL in v3 (Variant A leaves it empty).
    conditioning_inputs: list[str] = Field(default_factory=list)
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    perturb_keys: list[str] = Field(default_factory=lambda: ["wt"])
    # Scale-ramped zero-init (2026-06-20 analysis §4a). When ``enabled``, an
    # :class:`OutputScaleRampCallback` is attached to the trainer and the
    # ramp value is written into ``MaisiControlNet.output_scale`` every step.
    # ``None`` is functionally identical to ``enabled=false``.
    output_scale_ramp: _OutputScaleRampCfg | None = None

    @model_validator(mode="after")
    def _validate_enabled_requires_inputs(self) -> _ControlNetCfg:
        if self.enabled and not self.conditioning_inputs:
            raise ValueError(
                "controlnet.enabled=true requires a non-empty conditioning_inputs list. "
                "(For S1 v3 Variant A, set controlnet.enabled=false.)"
            )
        if not self.enabled and self.conditioning_inputs:
            raise ValueError(
                "controlnet.enabled=false rejects conditioning_inputs — Variant A has no "
                "ControlNet to consume them. Move modality latents to "
                "model.trunk.input_concat.cond_latents instead."
            )
        return self


class _TrunkCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Project default: fine-tune the trunk jointly with the ControlNet. Set
    # ``false`` for the frozen-backbone baseline arm of the A/B.
    trainable: bool = True
    # How a trainable trunk is parameterised. ``'fft'`` (default) updates every
    # trunk tensor; ``'peft'`` routes through ``vena.model.fm.maisi.peft`` to
    # inject adapter tensors (LoRA / IA3 / DoRA / ...) on top of the frozen
    # backbone. The ``peft`` block then selects variant + params.
    regime: Literal["fft", "peft"] = "fft"
    peft: dict[str, Any] | None = None
    # S1 v3: channel-concat conditioning at the trunk's first convolution
    # (both Variant A and Variant B). Disabled by default ⇒ S1 v2 behaviour.
    input_concat: _InputConcatCfg = Field(default_factory=_InputConcatCfg)

    @model_validator(mode="after")
    def _validate_regime_peft(self) -> _TrunkCfg:
        if self.regime == "peft":
            if not self.trainable:
                raise ValueError("trunk.regime='peft' requires trunk.trainable=true")
            if not self.peft or "variant" not in self.peft:
                raise ValueError(
                    "trunk.regime='peft' requires a peft block of the form "
                    "{variant: <name>, params: {...}}"
                )
        elif self.peft is not None:
            raise ValueError(
                "trunk.peft must be null when trunk.regime='fft' (got "
                f"{self.peft!r}); set regime='peft' to enable adapter training"
            )
        return self


class _ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trunk: _TrunkCfg
    controlnet: _ControlNetCfg
    vae_checkpoint: Path | None = None  # only needed when val image metrics are on


class _RFlowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_train_timesteps: int = 1000
    use_discrete_timesteps: bool = True
    sample_method: str = "uniform"
    # SD3-style resolution-aware timestep weighting (Esser et al. 2024,
    # arXiv:2403.03206) — biases sampling toward intermediate α where
    # semantic structure forms. Complementary to the LPL high-SNR gate
    # (2026-06-20 analysis §4b). Off by default to keep the retired S1
    # recipe reproducible.
    use_timestep_transform: bool = False
    base_img_size_numel: int | None = None


class _OptimCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Default bumped from 5e-5 (2026-06-07 runs) to 1e-4 to match the MAISI-V2
    # joint trunk+ControlNet recipe (arXiv:2508.05772 §4.1) and TumorFlow
    # (arXiv:2603.04058). The 2026-06-09 overhaul note documents the change.
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    # Restricted to the three values handled by ``_lr_lambda`` in
    # ``vena.model.fm.lightning.module``. Anything else raises at module init,
    # which is the bug-prevention the 2026-06-09 overhaul installed (the old
    # silent fallthrough to constant LR hid the polynomial misconfiguration).
    scheduler: Literal["constant", "polynomial", "cosine"] = "cosine"


class _EMACfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1
    inv_gamma: float = 10.0
    power: float = 1.0
    min_value: float = 0.0


class _TrainingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_steps: int = 50_000
    # Optional epoch cap. When set, training stops at whichever of
    # ``max_epochs`` / ``total_steps`` is reached first (Lightning semantics).
    # Used by short diagnostic runs that want an exact epoch count regardless of
    # dataset size; leave ``null`` for step-governed production runs.
    max_epochs: int | None = None
    batch_size: int = 4
    grad_accum: int = 1
    checkpoint_every_epochs: int = 5
    log_train_every_steps: int = 100
    gradient_clip_val: float = 5.0
    # Epochs of plateau on ``train/total_epoch`` (mode=min) before Lightning
    # halts training. ``None`` disables EarlyStopping. Set to e.g. 100 for the
    # 1000-epoch Picasso runs so a converged + plateaued run releases the
    # allocation early. Monitor key is the epoch-aggregated training loss
    # because exhaustive-val PSNR/SSIM never enter ``trainer.callback_metrics``
    # (the launcher writes them to CSV from a subprocess).
    patience: int | None = None
    # Classifier-free-guidance training-time dropout. Per-sample Bernoulli flip
    # — when True for sample ``i``, the listed conditioning channels are zeroed
    # for that sample in the trunk forward (Ho & Salimans 2022; ControlNet
    # §3.5). ``0.0`` disables the path (byte-identical to a run without CFG).
    # The Picasso S2-LoRA+CFG run uses 0.15.
    conditioning_dropout_p: float = 0.0
    conditioning_dropout_keys: tuple[str, ...] = ("wt",)


class _ValidationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every_epochs: int = 1
    per_epoch_nfe: int = 5
    full_sweep_every_epochs: int = 5
    sweep_nfes: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 50])
    qualitative_every_epochs: int = 10
    image_metrics: bool = True  # master switch for image-space PSNR/SSIM
    # Image-space metrics are expensive (one VAE decode/patient) and only
    # meaningful at the canonical per_epoch_nfe, so they run on a slow cadence
    # rather than every epoch. 0 disables them entirely.
    image_metrics_every_epochs: int = 20


class _ExhaustiveValCfg(BaseModel):
    """Asynchronous image-space validation offloaded to a second GPU.

    On a slow cadence the trainer snapshots the EMA weights and launches a
    standalone subprocess (``routines.fm.exhaustive_val``) on ``device`` while
    training continues uninterrupted on the primary GPU. The subprocess samples
    each validation patient at every ``nfe_levels`` entry, decodes to image
    space, compares against the real T1c (percentile-normalised exactly as the
    encoder's input), and writes metrics/timing/figures + ``latent_preds.h5``.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    every_epochs: int = 20
    n_patients: int = 20
    nfe_levels: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 20])
    integrator: str = (
        "euler"  # ODE integrator for sampling (registry: vena...inference.get_sampler)
    )
    corpus_registry: Path | None = None
    device: str = "cuda:1"
    # Python used to launch the subprocess; defaults to the running interpreter.
    python_executable: str | None = None
    # Join each validation before training continues (one completed exhaustive
    # pass per cadence epoch). Default False = production async/skip-if-busy.
    block_until_complete: bool = False
    # How many top-best / top-worst patients to render as qualitative panels per
    # epoch (``figure_best_{1..k}.png`` + ``figure_worst_{1..k}.png``). Clamped
    # at job runtime to ``len(scored_patients) // 2`` so the lists never overlap.
    figure_top_k: int = 3
    # S3 — emit per-block real-vs-synth decoder-feature panels for the top-K
    # best/worst patients. Active only on K=2 production runs (the deeper K=5
    # readout makes the activation footprint risky on the val GPU). False by
    # default; the K=2 YAMLs flip it on.
    export_per_block_figures: bool = False
    # Prune ``ema_snapshot.pt`` / ``trunk_ema_snapshot.pt`` from epoch dirs older
    # than ``prune_snapshots_keep`` cadence epochs. ``latent_preds.h5`` and
    # ``metrics.csv`` are NEVER pruned — they are the long-run diagnostic record.
    # 0 disables pruning.
    prune_snapshots_keep: int = 2
    # Write ``latent_preds.h5`` only every N cadence passes (1-indexed: passes
    # 1, N+1, 2N+1, …). 0 or 1 writes on every cadence epoch (default 4 →
    # one H5 per ~80 epochs at every_epochs=20). The pass counter is tracked
    # by ``ExhaustiveValLauncher`` and injected into the job YAML as
    # ``latent_preds_pass_count``; the subprocess gates the write via
    # ``_should_write_latent_preds(pass_count, every_n)``.
    latent_preds_every_n: int = 4

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_image_h5(cls, data: Any) -> Any:
        if isinstance(data, dict) and "image_h5" in data:
            raise ValueError(
                "exhaustive_val.image_h5 was removed in the pre-long-run hardening pass. "
                "Use exhaustive_val.corpus_registry instead (same registry as "
                "data.corpus_registry)."
            )
        return data


class _OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiments_root: Path
    retention_n_checkpoints: int = 40
    tensorboard: bool = False
    wandb: bool = False


class _PostTrainCfg(BaseModel):
    """Optional post-training plotting hook.

    When enabled (default), :meth:`FMTrainRoutineEngine.run` calls the
    post-training plotting routine after ``trainer.fit()`` returns; failures
    are logged at WARNING and never fail the training run.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    formats: tuple[str, ...] = ("png",)


class FMTrainRoutineConfig(BaseModel):
    """Pydantic root config for ``vena-fm-train`` v2.

    Loaded via :meth:`from_yaml`; the original YAML is round-tripped into
    ``experiments/{run_id}/config.original.yaml``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: _RunCfg
    data: _DataCfg
    model: _ModelCfg
    loss: dict[str, Any] = Field(default_factory=dict)
    rflow: _RFlowCfg = Field(default_factory=_RFlowCfg)
    optim: _OptimCfg = Field(default_factory=_OptimCfg)
    ema: _EMACfg = Field(default_factory=_EMACfg)
    training: _TrainingCfg = Field(default_factory=_TrainingCfg)
    validation: _ValidationCfg = Field(default_factory=_ValidationCfg)
    exhaustive_val: _ExhaustiveValCfg = Field(default_factory=_ExhaustiveValCfg)
    output: _OutputCfg
    post_train: _PostTrainCfg = Field(default_factory=_PostTrainCfg)
    # Region specs are no longer consumed in-process (validation is offloaded),
    # but the field is kept (optional) for backward compatibility with configs
    # that still declare it.
    regions: dict[str, RegionSpec] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> FMTrainRoutineConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _validate_v3_consistency(self) -> FMTrainRoutineConfig:
        """S1 v3 cross-block consistency checks.

        * ``loss.cfm.region_weights.enabled=true`` requires
          ``loss.cfm.reduction='none'``.
        * ``model.trunk.input_concat.enabled=true`` requires
          ``model.trunk.arch_overrides.in_channels`` to match the implied
          width (4 base + 4 * len(cond_latents)). If unset, this is auto-
          filled — Pydantic configs are frozen so we raise instead with a
          clear remediation message.
        """
        cfm = (self.loss or {}).get("cfm") or {}
        rw = cfm.get("region_weights") or {}
        rw_enabled = rw.get("enabled", True if ("brain" in rw or "tc" in rw) else False)
        if rw_enabled and cfm.get("reduction", "mean") != "none":
            raise ValueError(
                "loss.cfm.region_weights enabled=true requires loss.cfm.reduction='none' "
                f"(got reduction={cfm.get('reduction', 'mean')!r}). Set "
                "loss.cfm.reduction: none in the YAML."
            )
        ic = self.model.trunk.input_concat
        if ic.enabled:
            expected_in = 4 + 4 * len(ic.cond_latents) + len(ic.cond_masks)
            declared = int(self.model.trunk.arch_overrides.get("in_channels", 4))
            if declared != expected_in:
                raise ValueError(
                    "model.trunk.input_concat.enabled=true with "
                    f"cond_latents={ic.cond_latents}, cond_masks={ic.cond_masks} "
                    f"implies trunk in_channels={expected_in}, but "
                    f"model.trunk.arch_overrides.in_channels={declared}. Set "
                    f"model.trunk.arch_overrides: {{in_channels: {expected_in}}}."
                )
        return self


# =============================================================================
# Pre-flight gates
# =============================================================================


def _safe_sha256(path: Path | None) -> str | None:
    """Best-effort SHA-256 of ``path``; ``None`` if the file is missing.

    Logged-and-skipped rather than raised so a misconfigured experiments-root
    does not block training on its own — the SHAs are provenance, not gating.
    """
    if path is None:
        return None
    try:
        return sha256_file(Path(path))
    except FileNotFoundError:
        logger.warning("checkpoint not found at %s; provenance SHA will be null", path)
        return None


def _load_preflight_decision(path: Path) -> dict[str, Any]:
    """Load a pre-flight ``decision.json``; raise ``PreflightGateError`` on miss."""
    if not path.exists():
        raise PreflightGateError(
            f"Pre-flight decision missing at {path}. "
            f"Run the corresponding pre-flight routine first and "
            f"point data.preflight_decision_path at its artifact."
        )
    return json.loads(path.read_text())


def _assert_preflight_gates(cfg: FMTrainRoutineConfig) -> None:
    """Validate that every pre-flight required by ``cfg`` is present.

    Gates enforced:

    * Augmentation: when ``data.augmentation_config_path`` is set, the
      referenced pre-flight (``data.preflight_decision_path``) must exist and
      must declare every augmentation in the runtime YAML inside its
      ``latent_safe_augmentations`` allowlist. The pipeline builder
      (``vena.data.augment.build_pipeline_from_yaml``) re-checks this, but
      catching it up-front yields a clearer message and avoids partial setup.
    * Cohort deduplication: when ``data.dedup_decisions_path`` is set, the
      referenced ``decision.json`` (schema v1.0) must validate; the corpus
      registry SHA-256 inside the decision must match the SHA-256 of the
      currently-loaded ``data.corpus_registry``; every cv cohort listed in
      the registry must have an entry in ``decision["cohorts"]``; and the
      decision must not carry unresolvable overlaps (these would mean some
      cross-cohort duplicates are still in the training corpus).

    Raises
    ------
    PreflightGateError
        If any gate fails.
    """
    if cfg.data.augmentation_config_path is not None:
        if cfg.data.preflight_decision_path is None:
            raise PreflightGateError(
                "data.augmentation_config_path is set but data.preflight_decision_path "
                "is None. Augmentations must be gated by the latent_aug_equivariance "
                "pre-flight — see .claude/rules/preflight-pattern.md."
            )

        decision = _load_preflight_decision(cfg.data.preflight_decision_path)
        allowlist = set(decision.get("latent_safe_augmentations") or [])
        aug_yaml = yaml.safe_load(Path(cfg.data.augmentation_config_path).read_text()) or {}
        requested = {entry["name"] for entry in (aug_yaml.get("augmentations") or [])}
        forbidden = requested - allowlist
        if forbidden:
            raise PreflightGateError(
                f"Augmentations {sorted(forbidden)} requested in "
                f"{cfg.data.augmentation_config_path} are not on the pre-flight "
                f"allowlist {sorted(allowlist)} from {cfg.data.preflight_decision_path}. "
                f"Either drop the augmentation or rerun the equivariance pre-flight."
            )

    if cfg.data.dedup_decisions_path is not None:
        _assert_dedup_gate(cfg)

    if cfg.data.use_offline_augmented_data:
        _assert_offline_aug_gate(cfg)

    if cfg.run.stage.lower() == "s3":
        _assert_decoder_lpl_gate(cfg)


def _build_lpl_config(cfg: FMTrainRoutineConfig) -> Any:
    """Build :class:`vena.model.fm.lpl.LplConfig` from the decision.json (S3 only).

    Returns ``None`` for non-S3 stages or when no decision path is set
    (the LightningModule then skips its S3 branch entirely). The decision
    contract owns ``A`` / ``w_l`` / ``t_min`` / ``outlier_k`` / region
    recipe; the ``loss.lpl`` YAML block supplies ``lambda_img`` /
    ``schedule`` / ``grad_checkpoint_segments`` / ``soft_region`` plus
    optional ``*_override`` knobs for non-preflight arms (K=5 canonical,
    Standard-LPL α=(1,1)).
    """
    if cfg.run.stage.lower() != "s3":
        return None
    path = getattr(cfg.data, "decoder_lpl_decision_path", None)
    if path is None:
        return None
    from vena.model.fm.lpl import LambdaImgSchedule, LplConfig

    lpl_block = (cfg.loss or {}).get("lpl") if isinstance(cfg.loss, dict) else None
    lpl_block = lpl_block or {}

    schedule_raw = lpl_block.get("schedule")
    schedule = LambdaImgSchedule.model_validate(schedule_raw) if schedule_raw is not None else None

    def _coerce_int_keys(d: Any) -> dict[int, float] | None:
        if d is None:
            return None
        return {int(k): float(v) for k, v in d.items()}

    return LplConfig.from_decision(
        Path(path),
        lambda_img=float(lpl_block.get("lambda_img", 0.1)),
        soft_region=lpl_block.get("soft_region"),
        grad_checkpoint_segments=lpl_block.get("grad_checkpoint_segments"),
        schedule=schedule,
        A_override=lpl_block.get("A_override"),
        w_l_override=_coerce_int_keys(lpl_block.get("w_l_override")),
        outlier_k_override=_coerce_int_keys(lpl_block.get("outlier_k_override")),
        alpha_override=lpl_block.get("alpha_override"),
    )


def _assert_decoder_lpl_gate(cfg: FMTrainRoutineConfig) -> None:
    """S3 requires ``data.decoder_lpl_decision_path`` → a v1.0 decision.json.

    Reads the decision via the canonical validator, asserts the schema, and
    re-checks that every variant referenced in ``data.variant_weights`` is
    in the decision's ``allowed_variants`` list — otherwise S3 would train
    against an augmentation the LPL preflight explicitly rejected.
    """
    from vena.preflight.decoder_lpl_profile.decision import (
        assert_decoder_lpl_decision_valid,
    )

    path = getattr(cfg.data, "decoder_lpl_decision_path", None)
    if path is None:
        raise PreflightGateError(
            "S3 stage requires data.decoder_lpl_decision_path → the v1.0"
            " decision.json emitted by routines/preflights/decoder_lpl_profile."
            " Point at artifacts/preflights/decoder_lpl_profile/LATEST/decision.json."
        )
    p = Path(path)
    if not p.is_file():
        raise PreflightGateError(
            f"data.decoder_lpl_decision_path={p} does not exist or is not a file."
        )
    try:
        decision = assert_decoder_lpl_decision_valid(p)
    except Exception as exc:  # pydantic ValidationError + others
        raise PreflightGateError(
            f"decoder_lpl_profile decision at {p} failed validation: {exc}"
        ) from exc

    # Variant intersection check: every variant in variant_weights with
    # nonzero weight must be in allowed_variants.
    weights = cfg.data.variant_weights or {}
    active = {v for v, w in weights.items() if float(w) > 0.0}
    not_allowed = active - set(decision.allowed_variants)
    if not_allowed:
        raise PreflightGateError(
            f"variant_weights uses {sorted(not_allowed)} but the decoder_lpl"
            f" preflight only allows {decision.allowed_variants}."
            f" Either zero the weight or rerun the preflight."
        )
    logger.info(
        "decoder_lpl gate passed: A=%s w_l=%s t_min=%.3f allowed=%s",
        decision.A_recommended,
        decision.w_l,
        decision.t_min,
        decision.allowed_variants,
    )


def _assert_offline_aug_gate(cfg: FMTrainRoutineConfig) -> None:
    """Validate that every cv cohort carries ``latent_aug_h5`` when the flag is on.

    Variant-weight sanity is also enforced here: weights must be non-empty,
    non-negative, and contain at least one non-``v0`` entry whose
    corresponding aug-H5 row will be served.
    """
    from vena.data.registry import load_registry

    registry = load_registry(cfg.data.corpus_registry)
    cv_cohorts = registry.cv_cohorts()
    missing = [c.name for c in cv_cohorts if c.latent_aug_h5 is None]
    if missing:
        raise PreflightGateError(
            "data.use_offline_augmented_data=True but the following cv cohorts "
            f"have no latent_aug_h5 in the registry: {missing}. Either "
            "build their banks via `vena-offline-aug-maisi` and update the "
            "registry, or remove them from the registry."
        )
    bad_paths = [
        (c.name, c.latent_aug_h5) for c in cv_cohorts if not Path(c.latent_aug_h5).is_file()
    ]
    if bad_paths:
        raise PreflightGateError(
            "data.use_offline_augmented_data=True but these latent_aug_h5 "
            f"paths do not exist on disk: {bad_paths}."
        )
    weights = cfg.data.variant_weights
    if not weights:
        raise PreflightGateError("data.variant_weights is empty")
    if any(v < 0 for v in weights.values()):
        raise PreflightGateError(f"data.variant_weights has negative entries: {weights}")
    if sum(weights.values()) <= 0:
        raise PreflightGateError(f"data.variant_weights sum to zero: {weights}")
    if all(k == "v0" for k in weights if weights[k] > 0):
        raise PreflightGateError(
            "data.variant_weights only assigns probability to v0 — the offline "
            "bank will never be sampled; either drop use_offline_augmented_data "
            "or give v1..vN a non-zero weight."
        )


def _assert_dedup_gate(cfg: FMTrainRoutineConfig) -> None:
    """Validate the cohort-dedup ``decision.json`` and cross-check against cfg."""
    path = Path(cfg.data.dedup_decisions_path)
    if not path.exists():
        raise PreflightGateError(
            f"data.dedup_decisions_path={path} does not exist. Run "
            f"`vena-preflight-cohort-dedup` first."
        )
    try:
        decision = assert_dedup_decision_valid(path)
    except DedupDecisionSchemaError as exc:
        raise PreflightGateError(str(exc)) from exc

    # Cross-check: the corpus registry the decision was built against must
    # match the one this run uses (SHA-256 over the JSON file bytes).
    current_sha = sha256_file(Path(cfg.data.corpus_registry))
    if decision["corpus_registry_sha256"] != current_sha:
        raise PreflightGateError(
            f"dedup decision was built against corpus registry SHA-256 "
            f"{decision['corpus_registry_sha256']} "
            f"({decision['corpus_registry_path']}), but cfg.data.corpus_registry "
            f"({cfg.data.corpus_registry}) currently hashes to {current_sha}. "
            f"Re-run `vena-preflight-cohort-dedup` to refresh the decision."
        )

    # Every cv cohort must have an allow-list entry — a partial decision is a bug.
    registry = load_registry(cfg.data.corpus_registry, require_latents=False)
    missing = [c.name for c in registry.cv_cohorts() if c.name not in decision["cohorts"]]
    if missing:
        raise PreflightGateError(
            f"dedup decision {path} is missing cohorts {missing} that appear in "
            f"the corpus registry. Re-run the preflight against the current registry."
        )

    # Unresolvable overlaps are a policy decision made at preflight time
    # (`on_unresolvable: warn` keeps both cohorts whole). The gate surfaces
    # them as a WARNING so the run log carries the residual-risk note;
    # `on_unresolvable: error` would have already prevented the preflight
    # from emitting a decision at all.
    if decision.get("unresolvable_overlaps"):
        logger.warning(
            "dedup decision %s carries %d unresolvable overlap(s) "
            "(accepted at preflight time). Residual cross-cohort duplicates "
            "may remain. See decision.json.unresolvable_overlaps for details.",
            path,
            len(decision["unresolvable_overlaps"]),
        )


def _assert_module_paths_in_repo_root() -> None:
    """B11 — verify ``routines`` and ``vena`` are imported from inside this repo.

    A stale editable install pointing at a different checkout or a wrong
    PYTHONPATH silently trains against different code.  This check is
    CWD-independent: it resolves the imported module ``__file__`` and
    asserts it falls under the same repository root as this engine.
    """
    import routines
    import vena

    repo_root = Path(__file__).resolve().parents[3]
    for mod_name, mod in [("routines", routines), ("vena", vena)]:
        mod_path = Path(mod.__file__).resolve()
        try:
            mod_path.relative_to(repo_root)
        except ValueError as exc:
            raise AssertionError(
                f"Module '{mod_name}' is imported from {mod_path}, which is "
                f"outside the expected repo root {repo_root}. Check PYTHONPATH "
                f"(needs '<repo>/src:<repo>' — see MEMORY: PYTHONPATH routines leak)."
            ) from exc


def _assert_run_invariants(cfg: FMTrainRoutineConfig) -> None:
    """Hard guards that have each already cost this project a run.

    Called at the very top of :meth:`FMTrainRoutineEngine.run` before any
    side effect.  Every check raises (not warns) — these are all survivable-
    looking conditions at WARNING level, which is exactly why they went
    undetected until a full run surfaced them.
    """
    # N2 provenance — encoder percentile must be 99.95 (MEMORY entry
    # "Encoder percentile 99.95"); mismatched percentile biases all
    # intensity metrics (worst on ET).
    from vena.common import ENCODER_PERCENTILE_UPPER

    if ENCODER_PERCENTILE_UPPER != 99.95:
        raise AssertionError(
            f"ENCODER_PERCENTILE_UPPER={ENCODER_PERCENTILE_UPPER} != 99.95. "
            "The 99.95 percentile normalisation is load-bearing for intensity "
            "metrics — decoded predictions and real T1c must be normalised "
            "identically. Rebuild the vena package from the current source."
        )

    # §15 — brain mask source must not be the real_box>0 fallback, which is
    # derived from the real T1c and leaks target information into metrics.
    brain_key = getattr(cfg.data, "brain_mask_key", None)
    if brain_key != "masks/brain_latent":
        raise AssertionError(
            f"data.brain_mask_key={brain_key!r}; must be 'masks/brain_latent'. "
            "The real_box>0 fallback is derived from the real T1c and leaks "
            "target information into brain-masked PSNR/SSIM metrics."
        )

    # §17 / B2 — no test_only cohort must appear in the cv monitor set.
    # A test_only cohort in aggregate_cv.csv would leak held-out data into
    # early-stopping and checkpoint selection.
    _reg = load_registry(cfg.data.corpus_registry, require_latents=False)
    cv_names = {c.name for c in _reg.cv_cohorts()}
    test_only_names = {c.name for c in _reg.test_cohorts()}
    overlap = cv_names & test_only_names
    if overlap:
        raise AssertionError(
            f"test_only cohorts found in cv monitor set: {sorted(overlap)}. "
            "This leaks held-out test data into early stopping and checkpoint "
            "selection. Fix the corpus registry role assignments."
        )

    # §9 / B11 — resolved module paths (CWD-independent).
    _assert_module_paths_in_repo_root()


def _run_post_train(run_dir: Path, *, formats: tuple[str, ...]) -> None:
    """Render the post-training plot bundle for ``run_dir``.

    Failures are caught and logged at WARNING so the training run is still
    considered successful even if matplotlib is unavailable or a CSV column
    is malformed. The hook is import-deferred so a missing matplotlib does
    not crash module import.
    """
    try:
        from routines.fm.post_train.engine import render_for_run_dir

        render_for_run_dir(run_dir, formats=formats)
    except Exception as exc:
        logger.warning(
            "post-train plotting failed (training run remains successful): %s",
            exc,
            exc_info=True,
        )


def _assert_grad_clip_validity(run_dir: Path, cfg: FMTrainRoutineConfig) -> None:
    """§18 validity criterion: grad_clip_active mean must be < 5 % past step 5 000.

    An arm that clips more than 5 % of its optimiser steps past the warm-up
    window is invalid for the §18 norm comparison: the gradient-clip confound
    is not controlled and the loss-norm effect is uninterpretable.  This fires
    *post-training* so the full evidence is in hand; it does not abort a run
    mid-flight if clipping is heavy in the first 5 000 steps.

    ``train/grad_clip_active`` is 1.0 when the gradient norm exceeds
    ``gradient_clip_val``, else 0.0.  It is logged every optimiser step by
    :meth:`FMLightningModule.configure_gradient_clipping`.
    """
    import csv as _csv

    step_csv = run_dir / "metrics" / "train_step.csv"
    if not step_csv.exists():
        logger.warning("§18 grad_clip validity: %s not found — check skipped", step_csv)
        return

    clip_vals: list[float] = []
    try:
        with step_csv.open(newline="") as f:
            for row in _csv.DictReader(f):
                step_raw = row.get("global_step", "")
                clip_raw = row.get("train/grad_clip_active", "")
                if step_raw in ("", None) or clip_raw in ("", None):
                    continue
                try:
                    if int(float(step_raw)) <= 5000:
                        continue
                    clip_vals.append(float(clip_raw))
                except ValueError:
                    continue
    except OSError as exc:
        logger.warning("§18 grad_clip validity: cannot read %s: %s", step_csv, exc)
        return

    if not clip_vals:
        logger.warning(
            "§18 grad_clip validity: no step-CSV rows past step 5 000 in %s "
            "— check skipped (run too short?)",
            step_csv,
        )
        return

    mean_clip = sum(clip_vals) / len(clip_vals)
    logger.info(
        "§18 grad_clip_active: mean=%.4f over %d steps past step 5 000 "
        "(threshold < 0.05; gradient_clip_val=%s)",
        mean_clip,
        len(clip_vals),
        cfg.training.gradient_clip_val,
    )
    if mean_clip >= 0.05:
        raise AssertionError(
            f"§18 validity criterion FAILED: grad_clip_active mean={mean_clip:.4f} "
            f">= 0.05 over {len(clip_vals)} steps past step 5 000. "
            f"This arm clips too frequently for the loss-norm comparison to be "
            f"interpretable. Current gradient_clip_val={cfg.training.gradient_clip_val}. "
            f"Investigate the norm distribution before reporting this arm's results. "
            f"Do not raise gradient_clip_val beyond 10.0 without re-running the "
            f"entire ablation at the new value."
        )


def _record_termination_reason(
    run_dir: Path,
    trainer: pl.Trainer,
    cfg: FMTrainRoutineConfig,
    decision_path: Path,
) -> None:
    """B17 (2026-07-29): Append termination metadata to decision.json (schema 0.12.0).

    Resolves the termination reason from live trainer state, not from config,
    so post-mortem audits are unambiguous.  Priority order:

    1. ``early_stopping`` — EarlyStopping callback's ``stopped_epoch > 0``.
       When patience fires it is a **divergence / plateau signal**, not
       convergence; log at WARNING and flag for investigation.
    2. ``total_steps`` — ``trainer.global_step >= cfg.training.total_steps``.
       This is the canonical termination reason for all v3a arms.
    3. ``max_epochs`` — ``trainer.current_epoch >= cfg.training.max_epochs``.
       Triggers only if total_steps is higher than the epoch budget allows.
    4. ``unknown`` — should not occur in a normal run.

    This function permanently closes the class of audit error that produced
    the wrong §3 claim in v3a_retraining.md (EarlyStopping vs. total_steps).
    """
    final_step = int(trainer.global_step)
    final_epoch = int(trainer.current_epoch)

    # Detect EarlyStopping callback; ``stopped_epoch > 0`` means it fired.
    stopped_epoch: int | None = None
    for cb in trainer.callbacks:
        if hasattr(cb, "stopped_epoch"):
            se = int(getattr(cb, "stopped_epoch", 0))
            if se > 0:
                stopped_epoch = se
            break

    if stopped_epoch is not None:
        reason = "early_stopping"
    elif cfg.training.total_steps is not None and final_step >= cfg.training.total_steps:
        reason = "total_steps"
    elif cfg.training.max_epochs is not None and final_epoch >= cfg.training.max_epochs - 1:
        reason = "max_epochs"
    else:
        reason = "unknown"

    logger.info(
        "B17 termination: reason=%s  global_step=%d  epoch=%d",
        reason,
        final_step,
        final_epoch,
    )
    if reason == "early_stopping":
        logger.warning(
            "B17: EarlyStopping (divergence guard) fired at epoch=%d — "
            "this arm diverged or plateaued pathologically; investigate "
            "before accepting its checkpoints. This does NOT mean converged.",
            stopped_epoch,
        )
    if reason == "unknown":
        logger.warning(
            "B17: termination reason is 'unknown' (step=%d epoch=%d "
            "total_steps=%s max_epochs=%s) — unexpected; investigate.",
            final_step,
            final_epoch,
            cfg.training.total_steps,
            cfg.training.max_epochs,
        )

    # Patch decision.json in-place: read existing payload → add termination
    # fields → write back.  Bumps schema_version 0.12.0 → 0.13.0.
    try:
        payload = json.loads(decision_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("B17: cannot read %s to patch termination reason: %s", decision_path, exc)
        return
    payload["schema_version"] = "0.13.0"
    payload["termination_reason"] = reason
    payload["final_global_step"] = final_step
    payload["final_epoch"] = final_epoch
    if stopped_epoch is not None:
        payload["early_stopping_stopped_epoch"] = stopped_epoch
    decision_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("B17: decision.json updated (schema 0.13.0) at %s", decision_path)


class _WarmStartCallback(pl.Callback):
    """One-shot ``on_fit_start`` weights-only load for ``ResumeMode.WARM_START``.

    The callback fires *after* ``LightningModule.setup()`` (when the trunk has
    been built and the module's full ``state_dict`` is assembled) but *before*
    the training loop, so :meth:`FMLightningModule.load_warm_start` sees every
    key it needs to overlap against. Optimiser / EMA / scheduler / RNG state
    stay fresh — that's what distinguishes WARM_START from CONTINUE.

    The ``_applied`` flag makes the callback idempotent: ``on_fit_start`` only
    runs once per ``trainer.fit`` invocation, but defending against re-fits is
    cheap and surfaces accidental misuse loudly (the duplicate-load attempt is
    a single no-op).

    The :meth:`setup` hook fires *before* ``LightningModule.setup`` (PL 2.x
    lifecycle), which is exactly when the R6 trunk-EMA path needs the
    snapshot path published on the module so ``setup`` can reload the saved
    shadow into the freshly-built ``trunk_ema`` (model-coding-standards.md
    §4.5). The path is the sibling ``trunk_ema_snapshot.pt`` written by
    :class:`TrunkEMASnapshotCallback` during the source run; a missing
    snapshot is a non-fatal warning (covers pre-R6 S1 checkpoints).
    """

    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self._applied = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        snapshot = Path(self.ckpt_path).parent / TRUNK_EMA_SNAPSHOT_FILENAME
        # The setter is no-op-safe: pl_module.setup() decides whether to
        # actually load (skips silently when trunk_ema is None on a frozen
        # trunk, warns when the file is missing on a trainable trunk).
        pl_module.set_pending_trunk_ema_snapshot(  # type: ignore[attr-defined]
            snapshot if snapshot.is_file() else None
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._applied:
            return
        # The module is FMLightningModule — typed loosely for Lightning's hook.
        pl_module.load_warm_start(self.ckpt_path)  # type: ignore[attr-defined]
        self._applied = True


# =============================================================================
# Engine
# =============================================================================


class FMTrainRoutineEngine:
    """End-to-end training engine following training_routine.md."""

    def __init__(self, cfg: FMTrainRoutineConfig, config_yaml_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path

    def _make_run_dir(self) -> tuple[str, Path]:
        run_id = generate_run_id(self.cfg.run.stage, self.cfg.run.tag)
        run_dir = Path(self.cfg.output.experiments_root) / run_id
        # ``qualitative`` and ``performance`` are no longer produced in-process —
        # their content (qualitative figures + latent preds, per-NFE timing) now
        # lives under ``exhaustive_val/epoch_NNN/`` (created by the launcher).
        for sub in ("checkpoints", "logs", "metrics"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _resolve_run_dir(self, resume_ckpt: str | None, mode: ResumeMode) -> tuple[str, Path, bool]:
        """Choose the run directory based on the resume mode.

        * CONTINUE — reuse the dir of the prior run we resolved a checkpoint
          from; ``trainer.fit(ckpt_path=...)`` will then restore the optimiser
          / EMA / scheduler / RNG state and Lightning's ``ModelCheckpoint``
          will append to the same ``dirpath``. This is the SIGTERM auto-resubmit
          path and the only one that keeps a single contiguous artifact across
          Picasso walltime kills.
        * BASELINE / WARM_START — always mint a fresh timestamped dir; the
          warm-start mode pre-loads weights but the new run starts with a
          fresh optimiser, scheduler, EMA, and RNG state.

        Returns
        -------
        tuple[str, Path, bool]
            ``(run_id, run_dir, resuming_in_place)``.
        """
        if mode is ResumeMode.CONTINUE and resume_ckpt is not None:
            p = Path(resume_ckpt).resolve()
            root = Path(self.cfg.output.experiments_root).resolve()
            if root in p.parents:
                run_dir = p.parents[1]  # <root>/<run>/checkpoints/<file> -> <root>/<run>
                for sub in ("checkpoints", "logs", "metrics"):
                    (run_dir / sub).mkdir(parents=True, exist_ok=True)
                return run_dir.name, run_dir, True
        run_id, run_dir = self._make_run_dir()
        return run_id, run_dir, False

    def _attach_file_log(self, run_dir: Path) -> logging.Handler:
        """Tee log records to ``logs/train.log`` so the run is self-contained.

        The CLI configures a console (rich) handler; here we add a plain
        file handler on the root logger so the run directory captures its own
        training log regardless of how stdout is redirected.
        """
        handler = logging.FileHandler(run_dir / "logs" / "train.log", mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        return handler

    def _write_static_provenance(self, run_dir: Path, resuming_in_place: bool = False) -> None:
        merged = self.cfg.model_dump(mode="json")
        if resuming_in_place:
            # Preserve the original run's provenance; record the resume invocation
            # under a timestamped name so the audit trail shows every restart.
            ts = now_iso_utc().replace(":", "-")
            (run_dir / f"config.resume_{ts}.yaml").write_text(
                yaml.safe_dump(merged, sort_keys=False)
            )
            return
        (run_dir / "config.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
        if self.config_yaml_path is not None:
            shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        else:
            (run_dir / "config.original.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
        write_provenance(run_dir, repo=Path(__file__).resolve().parents[3])

    def _build_decision_payload(
        self,
        run_id: str,
        run_dir: Path,
        *,
        resume_mode: ResumeMode,
        resume_source: str | None,
        resume_source_run_id: str | None,
    ) -> dict[str, Any]:
        """Schema-0.11.0 decision JSON written once at run creation.

        Carries enough provenance for a downstream consumer to reproduce the
        run end-to-end: data registry, trunk + VAE SHA-256, loss stage,
        optimiser/EMA hyperparameters, augmentation gate path, the list of
        cohort names actually wired in, the cohort deduplication decision
        file + SHA-256 used to filter the corpus, the offline-augmentation
        bank toggle + per-cohort latent_aug_h5 paths + variant_weights, the
        trunk regime + PEFT variant + params (schema 0.6.0), the CFG
        conditioning dropout (schema 0.7.0), and — from schema 0.8.0 — the
        recipe ``tag`` plus the classified resume mode and its source so an
        auditor can tell at a glance whether a given run was a fresh
        baseline, a SIGTERM-resume continuation, or a warm-start from a
        prior run.

        Schema changelog:
        - 0.6.0: ``trunk_regime`` / ``trunk_peft_variant`` / ``trunk_peft_params``
        - 0.7.0: ``conditioning_dropout_p`` / ``conditioning_dropout_keys``
        - 0.8.0: ``tag`` / ``resume_mode`` / ``resume_source`` / ``resume_source_run_id``
        - 0.9.0: ``decoder_lpl_decision_path`` / ``decoder_lpl_decision_sha256``
        - 0.10.0: ``controlnet_enabled`` / ``controlnet_conditioning_inputs`` / ``input_concat`` / ``loss_cfm_*`` / ``region_weights`` / ``mask_source``
        - 0.11.0: ``loss_cfm_delta`` / ``gradient_clip_val`` / ``latent_preds_every_n`` / ``exhaustive_val_aggregation``
        """
        cfg = self.cfg
        registry = load_registry(cfg.data.corpus_registry)
        aug_image_paths: dict[str, str] = {}
        aug_latent_paths: dict[str, str] = {}
        if cfg.data.use_offline_augmented_data:
            for c in registry.cv_cohorts_with_aug():
                aug_image_paths[c.name] = str(c.image_aug_h5)
                aug_latent_paths[c.name] = str(c.latent_aug_h5)
        trunk_peft_variant: str | None = None
        trunk_peft_params: dict[str, Any] | None = None
        if cfg.model.trunk.regime == "peft" and cfg.model.trunk.peft is not None:
            trunk_peft_variant = cfg.model.trunk.peft.get("variant")
            trunk_peft_params = dict(cfg.model.trunk.peft.get("params", {}))
        cfm_block = (cfg.loss or {}).get("cfm") or {}
        rw_block = cfm_block.get("region_weights")
        return {
            "schema_version": "0.12.0",
            "produced_at": now_iso_utc(),
            "producer": "routines.fm.train:0.12.0",
            # B19 (2026-07-29): code provenance in the machine-readable contract.
            # git_commit.txt carries the same SHA as a human-readable record.
            "git_sha": resolve_git_sha() or "unknown",
            "git_dirty": resolve_git_dirty() or False,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stage": cfg.run.stage,
            "tag": cfg.run.tag,
            "seed": cfg.run.seed,
            # Resume audit trail (schema 0.8.0).
            "resume_mode": resume_mode.value,
            "resume_source": resume_source,
            "resume_source_run_id": resume_source_run_id,
            "corpus_registry": str(cfg.data.corpus_registry),
            "cohorts_used": [c.name for c in registry.cohorts],
            "trunk_checkpoint": str(cfg.model.trunk.checkpoint),
            "trunk_checkpoint_sha256": _safe_sha256(cfg.model.trunk.checkpoint),
            "trunk_trainable": cfg.model.trunk.trainable,
            "trunk_regime": cfg.model.trunk.regime,
            "trunk_peft_variant": trunk_peft_variant,
            "trunk_peft_params": trunk_peft_params,
            "vae_checkpoint": (str(cfg.model.vae_checkpoint) if cfg.model.vae_checkpoint else None),
            "vae_checkpoint_sha256": _safe_sha256(cfg.model.vae_checkpoint),
            "loss_stage": cfg.run.stage,
            "ema_decay": cfg.ema.decay,
            "augmentation_config_path": (
                str(cfg.data.augmentation_config_path)
                if cfg.data.augmentation_config_path
                else None
            ),
            "augmentation_preflight_path": (
                str(cfg.data.preflight_decision_path) if cfg.data.preflight_decision_path else None
            ),
            "dedup_decision_path": (
                str(cfg.data.dedup_decisions_path) if cfg.data.dedup_decisions_path else None
            ),
            "dedup_decision_sha256": _safe_sha256(cfg.data.dedup_decisions_path),
            "exhaustive_val_enabled": cfg.exhaustive_val.enabled,
            "use_offline_augmented_data": cfg.data.use_offline_augmented_data,
            "variant_weights": dict(cfg.data.variant_weights)
            if cfg.data.use_offline_augmented_data
            else None,
            "aug_image_h5_paths": aug_image_paths or None,
            "aug_latent_h5_paths": aug_latent_paths or None,
            # Schema 0.7.0 — CFG training-time dropout (additive, defaults to 0).
            "conditioning_dropout_p": cfg.training.conditioning_dropout_p,
            "conditioning_dropout_keys": list(cfg.training.conditioning_dropout_keys),
            # Schema 0.9.0 — decoder-LPL preflight gate (S3 stage).
            "decoder_lpl_decision_path": (
                str(cfg.data.decoder_lpl_decision_path)
                if cfg.data.decoder_lpl_decision_path
                else None
            ),
            "decoder_lpl_decision_sha256": _safe_sha256(cfg.data.decoder_lpl_decision_path),
            # Schema 0.10.0 — S1 v3 architecture deltas (channel-concat at
            # trunk + optional ControlNet for masks + region-weighted L1).
            "controlnet_enabled": cfg.model.controlnet.enabled,
            "controlnet_init_from_trunk_enabled": (
                cfg.model.controlnet.init_from_trunk if cfg.model.controlnet.enabled else None
            ),
            "controlnet_conditioning_inputs": list(cfg.model.controlnet.conditioning_inputs),
            "input_concat": {
                "enabled": cfg.model.trunk.input_concat.enabled,
                "cond_latents": list(cfg.model.trunk.input_concat.cond_latents),
                "cond_masks": list(cfg.model.trunk.input_concat.cond_masks),
                "trunk_in_channels_old": 4,
                "trunk_in_channels_new": int(cfg.model.trunk.arch_overrides.get("in_channels", 4)),
                "zero_init_new_channels": cfg.model.trunk.input_concat.zero_init_new_channels,
                "ramp_steps": cfg.model.trunk.input_concat.ramp_steps,
                "ramp_steepness": cfg.model.trunk.input_concat.ramp_steepness,
            },
            "loss_cfm_reduction": cfm_block.get("reduction", "mean"),
            "loss_cfm_norm": cfm_block.get("norm", "l2"),
            "region_weights": (dict(rw_block) if rw_block is not None else None),
            # Schema 0.11.0 — soft 2-channel mask [TC, NETC] serving policy.
            "mask_source": cfg.data.mask_source,
            # Reserved for the (deferred) normalisation-audit sibling spec;
            # null until that preflight lands and v3 latents carry a
            # ``normalization_variant_id`` attr cross-checked at engine init.
            "normalization_audit_decision_path": None,
            "normalization_variant_id": "V0",
            # Schema 0.11.0 — §18 ablation arm tracking + aggregation contract.
            "loss_cfm_delta": float(cfm_block.get("delta", 0.90)),
            "gradient_clip_val": cfg.training.gradient_clip_val,
            "latent_preds_every_n": cfg.exhaustive_val.latent_preds_every_n,
            "exhaustive_val_aggregation": "patient_mean_then_cohort_mean",
        }

    def _build_exhaustive_job_base(self, cfg: FMTrainRoutineConfig) -> dict[str, Any]:
        """Static fields for the exhaustive-val job YAML (epoch/snapshot added later).

        All paths are stringified so the launcher can ``yaml.safe_dump`` them.
        Multi-cohort only — single-cohort ``image_h5`` was removed in the
        pre-long-run hardening pass.
        """
        ev = cfg.exhaustive_val
        if cfg.model.vae_checkpoint is None:
            raise ValueError("exhaustive_val.enabled is true but model.vae_checkpoint is null")

        if ev.corpus_registry is None:
            raise ValueError(
                "exhaustive_val.enabled is true but exhaustive_val.corpus_registry is not set"
            )

        job: dict[str, Any] = {
            "stage": cfg.run.stage,
            "seed": cfg.run.seed,
            "trunk": {
                "checkpoint": str(cfg.model.trunk.checkpoint),
                "arch_json": str(cfg.model.trunk.arch_json) if cfg.model.trunk.arch_json else None,
                "arch_overrides": dict(cfg.model.trunk.arch_overrides),
                "class_token": cfg.model.trunk.class_token,
                "spacing_mm": list(cfg.model.trunk.spacing_mm),
                "trainable": cfg.model.trunk.trainable,
                "regime": cfg.model.trunk.regime,
                "peft": (dict(cfg.model.trunk.peft) if cfg.model.trunk.peft is not None else None),
                # S1 v3 — propagate input-concat so the sub-process rebuilds
                # the trunk with the same widened conv_in shape.
                "input_concat": cfg.model.trunk.input_concat.model_dump(),
            },
            "controlnet": {
                # S1 v3 — propagate the enable + init flags so Variant A
                # sub-jobs skip the ControlNet entirely.
                "enabled": cfg.model.controlnet.enabled,
                "init_from_trunk": cfg.model.controlnet.init_from_trunk,
                "conditioning_inputs": list(cfg.model.controlnet.conditioning_inputs),
                "arch_overrides": dict(cfg.model.controlnet.arch_overrides),
            },
            "vae_checkpoint": str(cfg.model.vae_checkpoint),
            "rflow": cfg.rflow.model_dump(),
            "ema": cfg.ema.model_dump(),
            "fold": cfg.data.fold,
            # S2 T-13 (task-20): mirror the training data path's mask_source so
            # the exhaustive-val subprocess builds its LatentH5Dataset with the
            # same mask-serving policy that the ConditioningAssembler expects.
            "mask_source": cfg.data.mask_source,
            "nfe_levels": list(ev.nfe_levels),
            "integrator": ev.integrator,
            "n_patients": ev.n_patients,
            "figure_top_k": ev.figure_top_k,
            # B12 — H5 write gate; pass_count is injected per-launch by
            # ExhaustiveValLauncher._launch (dynamic counter).
            "latent_preds_every_n": ev.latent_preds_every_n,
        }
        # S3 — per-block real-vs-synth feature-map render. Active only when the
        # YAML sets ``exhaustive_val.export_per_block_figures=true`` and the
        # trainer has an LPL config (so ``lpl_A`` is populated from the live
        # readout depth, not the YAML's overrideable knobs).
        export_per_block = bool(getattr(ev, "export_per_block_figures", False))
        job["export_per_block_figures"] = export_per_block
        if export_per_block:
            lpl_cfg = _build_lpl_config(cfg)
            job["lpl_A"] = list(lpl_cfg.A) if lpl_cfg is not None else []
            job["vae_checkpoint"] = str(cfg.model.vae_checkpoint)
        else:
            job["lpl_A"] = []
        job["corpus_registry"] = str(ev.corpus_registry)
        return job

    def _resolve_resume_ckpt(
        self, exclude_dir: Path | None = None
    ) -> tuple[str | None, ResumeMode]:
        """Resolve ``run.resume_from`` to ``(checkpoint_path, mode)``.

        * BASELINE  → ``(None, BASELINE)``.
        * CONTINUE  → newest sibling under ``experiments_root`` whose dir name
          matches ``*_{stage}_{tag}_*`` and contains the target checkpoint
          (``last.ckpt`` for ``latest``; ``ema_best.ckpt`` for ``best``). The
          scan is scoped to the same recipe so a sibling job of a different
          recipe is invisible — fixing the 2026-06-10 Picasso bug where an s2
          job inherited an s1 ``last.ckpt``.
        * WARM_START → either ``experiments_root/<run_id>/checkpoints/last.ckpt``
          (for a literal run_id) or the explicit absolute ``.ckpt`` path. The
          source run may live under any recipe — that's the point. Before
          resolving the source, the WARM_START branch checks for a sibling
          dir under the *current* recipe; if one exists with ``last.ckpt``
          the resolver returns ``(<sibling>/last.ckpt, CONTINUE)`` so the
          walltime-resubmit case continues in place instead of repeatedly
          warm-starting from the external source.
        * Unrecognised → ``InvalidResumeFromError`` (no silent fallback).

        ``exclude_dir`` skips a just-created run dir whose ``checkpoints/`` is
        empty (CONTINUE only). Unused in BASELINE / WARM_START.
        """
        cfg = self.cfg
        rf = cfg.run.resume_from
        mode = _classify_resume_from(rf)

        if mode is ResumeMode.BASELINE:
            return None, mode

        root = Path(cfg.output.experiments_root)

        if mode is ResumeMode.CONTINUE:
            assert rf is not None  # narrowed by _classify_resume_from
            target = "last.ckpt" if rf == "latest" else "ema_best.ckpt"
            glob_pat = f"*_{cfg.run.stage}_{cfg.run.tag}_*/"
            skip = exclude_dir.resolve() if exclude_dir is not None else None
            dirs = sorted(
                (d for d in root.glob(glob_pat) if d.is_dir() and d.resolve() != skip),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for d in dirs:
                cand = d / "checkpoints" / target
                if cand.is_file():
                    logger.info("CONTINUE (%s): resuming from %s", rf, cand)
                    return str(cand), mode
            logger.info(
                "CONTINUE (%s): no checkpoint under %s matching recipe %s/%s; "
                "treating as BASELINE (fresh run from MAISI trunk).",
                rf,
                root,
                cfg.run.stage,
                cfg.run.tag,
            )
            return None, ResumeMode.BASELINE

        # WARM_START
        assert mode is ResumeMode.WARM_START and rf is not None

        # Auto-promotion: a WARM_START YAML resubmitted after the first launch
        # (Picasso walltime kill is the canonical case) should CONTINUE the
        # already-created warm-started run rather than re-warm-start from the
        # original external source. If a recipe-matching sibling dir
        # ``*_{stage}_{tag}_*/`` already carries ``checkpoints/last.ckpt``,
        # promote to CONTINUE on that dir; ``resume_source`` in decision.json
        # still records the original YAML value so the audit trail of "what
        # the user asked for" survives. Skip the just-minted run dir so a
        # CONTINUE promotion never targets an empty sibling.
        glob_pat = f"*_{cfg.run.stage}_{cfg.run.tag}_*/"
        skip = exclude_dir.resolve() if exclude_dir is not None else None
        sibling_dirs = sorted(
            (d for d in root.glob(glob_pat) if d.is_dir() and d.resolve() != skip),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in sibling_dirs:
            cand = d / "checkpoints" / "last.ckpt"
            if cand.is_file():
                logger.info(
                    "WARM_START -> CONTINUE auto-promotion: sibling %s has last.ckpt; "
                    "original resume_from=%s preserved in decision.json's resume_source.",
                    d.name,
                    rf,
                )
                return str(cand), ResumeMode.CONTINUE

        if _RUN_ID_RE.match(rf):
            cand = root / rf / "checkpoints" / "last.ckpt"
            if not cand.is_file():
                raise InvalidResumeFromError(
                    f"WARM_START: run_id {rf!r} resolves to {cand} which does not exist."
                )
            logger.info("WARM_START: weights from %s (new run dir will be created)", cand)
            return str(cand), mode

        p = Path(rf)
        if not p.is_file():
            raise InvalidResumeFromError(f"WARM_START: explicit path {rf!r} does not exist.")
        logger.info("WARM_START: weights from explicit path %s", p)
        return str(p), mode

    def run(self) -> Path:
        cfg = self.cfg
        pl.seed_everything(cfg.run.seed, workers=True)

        # Pre-flight gate. Raises ``PreflightGateError`` before anything else
        # has been done so the failure message names exactly which artifact is
        # missing or non-conformant. See ``.claude/rules/preflight-pattern.md``.
        _assert_preflight_gates(cfg)
        # Hard invariants — each has already cost this project a run.  Raises
        # ``AssertionError`` (not warns) so misconfiguration is un-swallowable.
        _assert_run_invariants(cfg)

        # TF32 matmul: ~10% speed-up on A100/RTX-4090 at no measured cost to
        # FM training numerics. Set before any model is built; ignored on CPU
        # and on GPUs that do not advertise TF32 capability.
        torch.set_float32_matmul_precision("high")

        # Resolve the resume checkpoint *before* choosing the run dir so we can
        # continue in place (CONTINUE — same dir) instead of forking a new
        # empty run. BASELINE and WARM_START always mint a fresh dir.
        resume_ckpt, resume_mode = self._resolve_resume_ckpt()
        run_id, run_dir, resuming_in_place = self._resolve_run_dir(resume_ckpt, resume_mode)
        self._attach_file_log(run_dir)
        self._write_static_provenance(run_dir, resuming_in_place=resuming_in_place)
        logger.info(
            "FM-train run_id=%s dir=%s resume_mode=%s%s",
            run_id,
            run_dir,
            resume_mode.value,
            " (RESUMING IN PLACE)" if resuming_in_place else "",
        )

        # Decision JSON for downstream consumers — written once at run creation;
        # left intact on in-place resume. ``resume_source_run_id`` is only set
        # in WARM_START mode (when ``resume_from`` is a literal run_id).
        decision_path = run_dir / "decision.json"
        if not decision_path.exists():
            resume_source = cfg.run.resume_from
            resume_source_run_id = (
                resume_source
                if (
                    resume_mode is ResumeMode.WARM_START
                    and resume_source is not None
                    and _RUN_ID_RE.match(resume_source)
                )
                else None
            )
            decision_path.write_text(
                json.dumps(
                    self._build_decision_payload(
                        run_id,
                        run_dir,
                        resume_mode=resume_mode,
                        resume_source=resume_source,
                        resume_source_run_id=resume_source_run_id,
                    ),
                    indent=2,
                )
            )

        # Augmentation pipeline (optional). Built once on the main process
        # before fork so every DataLoader worker inherits the same operator
        # objects.  The pipeline manages its own per-worker RNG.
        train_transform = None
        if cfg.data.augmentation_config_path is not None:
            train_transform = build_pipeline_from_yaml(
                cfg.data.augmentation_config_path,
                preflight_decision_path=cfg.data.preflight_decision_path,
            )
            logger.info(
                "augmentation pipeline ENABLED from %s (gate=%s) — augmentations: %s",
                cfg.data.augmentation_config_path,
                cfg.data.preflight_decision_path,
                list(train_transform.names()),
            )

        # Data. In-process validation is offloaded to the async second-GPU job
        # (see ExhaustiveValLauncher), so the training process runs *only*
        # training on the primary GPU; ``region_resolver``/``vae_decoder`` are
        # not needed here.
        registry = load_registry(cfg.data.corpus_registry)

        # Load the cohort-dedup decision (already validated by the gate above)
        # and turn it into a per-cohort allow-list — pure in-memory sets, no
        # I/O during training.
        dedup_allowlists: dict[str, set[str]] | None = None
        if cfg.data.dedup_decisions_path is not None:
            dedup_payload = assert_dedup_decision_valid(cfg.data.dedup_decisions_path)
            dedup_allowlists = build_allowlists(dedup_payload)
            logger.info(
                "cohort_dedup ENABLED from %s — per-cohort kept: %s",
                cfg.data.dedup_decisions_path,
                {k: len(v) for k, v in dedup_allowlists.items()},
            )

        dm = MultiCohortLatentDataModule(
            registry=registry,
            fold=cfg.data.fold,
            batch_size=cfg.data.batch_size,
            tau=cfg.data.tau,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            seed=cfg.run.seed,
            max_train_patients_per_cohort=cfg.data.max_train_patients_per_cohort,
            train_transform=train_transform,
            dedup_allowlists=dedup_allowlists,
            use_offline_augmented_data=cfg.data.use_offline_augmented_data,
            variant_weights=cfg.data.variant_weights,
            mask_source=cfg.data.mask_source,
        )
        if cfg.data.use_offline_augmented_data:
            logger.info(
                "offline augmentation ENABLED — variant_weights: %s; per-cohort aug paths: %s",
                cfg.data.variant_weights,
                {c.name: str(c.latent_aug_h5) for c in registry.cv_cohorts_with_aug()},
            )
        logger.info(
            "Using MultiCohortLatentDataModule (registry=%s, tau=%.2f)",
            cfg.data.corpus_registry,
            cfg.data.tau,
        )

        # LightningModule (training-only: region_resolver/vae_decoder = None).
        trunk_cfg = TrunkConfig(
            checkpoint=cfg.model.trunk.checkpoint,
            arch_json=cfg.model.trunk.arch_json,
            arch_overrides=cfg.model.trunk.arch_overrides,
            class_token=cfg.model.trunk.class_token,
            spacing_mm=cfg.model.trunk.spacing_mm,
            trainable=cfg.model.trunk.trainable,
            regime=cfg.model.trunk.regime,
            peft=cfg.model.trunk.peft,
        )
        optim_cfg = {
            "lr": cfg.optim.lr,
            "betas": list(cfg.optim.betas),
            "weight_decay": cfg.optim.weight_decay,
            "warmup_steps": cfg.optim.warmup_steps,
            "scheduler": cfg.optim.scheduler,
            "max_steps": cfg.training.total_steps,
        }
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(cfg.model.controlnet.conditioning_inputs),
            stage=cfg.run.stage.upper() if cfg.run.stage.startswith("s") else cfg.run.stage,
            loss_cfg=cfg.loss,
            perturb_keys=set(cfg.model.controlnet.perturb_keys),
            controlnet_arch_overrides=cfg.model.controlnet.arch_overrides,
            optim_cfg=optim_cfg,
            rflow_cfg=cfg.rflow.model_dump(),
            ema_cfg=cfg.ema.model_dump(),
            region_resolver=None,
            vae_decoder=None,
            conditioning_dropout_p=cfg.training.conditioning_dropout_p,
            conditioning_dropout_keys=cfg.training.conditioning_dropout_keys,
            lpl_config=_build_lpl_config(cfg),
            lpl_vae_checkpoint=(
                cfg.model.vae_checkpoint
                if cfg.run.stage.lower() == "s3" and cfg.model.vae_checkpoint is not None
                else None
            ),
            # S1 v3 wiring (defaults preserve S1 v2 behaviour byte-for-byte).
            controlnet_enabled=cfg.model.controlnet.enabled,
            controlnet_init_from_trunk_enabled=cfg.model.controlnet.init_from_trunk,
            input_concat_cfg=cfg.model.trunk.input_concat.model_dump(),
        )

        # Checkpoint selection (ema_best) is on the epoch-aggregated training
        # loss, since validation is offloaded and runs asynchronously.
        ckpt_monitor = "train/total_epoch"
        callbacks: list[pl.Callback] = [
            VENACheckpointCallback(
                dirpath=run_dir / "checkpoints",
                retention_n_checkpoints=cfg.output.retention_n_checkpoints,
                every_n_epochs=cfg.training.checkpoint_every_epochs,
                monitor_key=ckpt_monitor,
                best_mode="min",
                save_on_train_epoch_end=True,
            ),
            BestCheckpointCallback(
                dirpath=run_dir / "checkpoints",
                monitor_key=ckpt_monitor,
                best_mode="min",
                save_on_train_epoch_end=True,
            ),
            # R6 (model-coding-standards.md §4.5): mirror the trunk-EMA shadow
            # next to the Lightning checkpoint on every save so a future
            # WARM_START (S1→S3) can restore the exact fine-tuned EMA shadow.
            # No-op for frozen-trunk runs (callback returns when
            # ``pl_module.trunk_ema is None``); safe to attach unconditionally.
            TrunkEMASnapshotCallback(dirpath=run_dir / "checkpoints"),
            TrainMetricsCSV(out_dir=run_dir / "metrics"),
            # §18 early-abort: raises AssertionError at step 10 000 if
            # mean(grad_clip_active) ≥ 5 % over steps >5 000. Fires ~1 % into
            # the 800 000-step budget — saves ~5 A100-days on an invalid arm.
            # _assert_grad_clip_validity (post-fit) is kept as belt-and-braces.
            GradClipValidityCallback(tag=cfg.run.tag),
            SigtermHandler(ckpt_dir=run_dir / "checkpoints", filename="ema_final.ckpt"),
        ]
        if train_transform is not None:
            callbacks.append(AugmentationTracker(out_dir=run_dir / "metrics"))
        if cfg.data.use_offline_augmented_data:
            callbacks.append(VariantTracker(out_dir=run_dir / "metrics"))
        ramp_cfg = cfg.model.controlnet.output_scale_ramp
        if ramp_cfg is not None and ramp_cfg.enabled:
            from vena.model.fm.lightning.callbacks import OutputScaleRampCallback

            callbacks.append(
                OutputScaleRampCallback(
                    ramp_steps=int(ramp_cfg.ramp_steps),
                    steepness=float(ramp_cfg.steepness),
                )
            )
            logger.info(
                "OutputScaleRampCallback ENABLED: ramp_steps=%d steepness=%.1f",
                int(ramp_cfg.ramp_steps),
                float(ramp_cfg.steepness),
            )
        if cfg.exhaustive_val.enabled:
            callbacks.append(
                ExhaustiveValLauncher(
                    run_dir=run_dir,
                    run_id=run_id,
                    job_base=self._build_exhaustive_job_base(cfg),
                    every_epochs=cfg.exhaustive_val.every_epochs,
                    device=cfg.exhaustive_val.device,
                    cwd=Path(__file__).resolve().parents[3],
                    python_executable=cfg.exhaustive_val.python_executable,
                    block_until_complete=cfg.exhaustive_val.block_until_complete,
                    prune_snapshots_keep=cfg.exhaustive_val.prune_snapshots_keep,
                    latent_preds_every_n=cfg.exhaustive_val.latent_preds_every_n,
                )
            )
        if cfg.training.patience is not None:
            from pytorch_lightning.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor=ckpt_monitor,
                    mode="min",
                    patience=int(cfg.training.patience),
                    check_on_train_epoch_end=True,
                    verbose=True,
                    strict=False,
                )
            )
            logger.info(
                "EarlyStopping DIVERGENCE GUARD ONLY: monitor=%s patience=%d epochs. "
                "Under normal monotone-loss convergence this will never fire. "
                "If it fires, the arm diverged or plateaued pathologically — "
                "investigate before treating it as converged. "
                "Checkpoint selection is always post-hoc via select_checkpoint.py / ssim_brain.",
                ckpt_monitor,
                int(cfg.training.patience),
            )
        else:
            logger.info(
                "EarlyStopping DISABLED (patience=null): training runs until "
                "total_steps=%s or max_epochs=%s; post-hoc selection via select_checkpoint.py.",
                cfg.training.total_steps,
                cfg.training.max_epochs,
            )

        # Trainer. We write our own clean metric CSVs, so Lightning's logger is
        # disabled. Validation is fully offloaded -> ``limit_val_batches=0``.
        trainer = pl.Trainer(
            max_steps=cfg.training.total_steps,
            max_epochs=cfg.training.max_epochs,
            precision=cfg.run.precision,
            devices=1 if cfg.run.device.startswith("cuda") else "auto",
            accelerator="gpu" if cfg.run.device.startswith("cuda") else "auto",
            log_every_n_steps=cfg.training.log_train_every_steps,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.grad_accum,
            deterministic=cfg.run.full_determinism,
            default_root_dir=str(run_dir),
            logger=False,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )

        # Dispatch on resume mode for the fit call:
        # * CONTINUE: hand ``ckpt_path`` to Lightning — it restores weights +
        #   optimiser + scheduler + EMA + RNG state and resumes the same
        #   epoch counter (same dir, contiguous artifact).
        # * WARM_START: the trunk does not exist until ``module.setup()`` runs
        #   (it's built lazily per ``model-coding-standards.md`` rule #2), so
        #   we cannot call ``load_warm_start`` here — the full state_dict is
        #   not assembled yet. We inject a one-shot callback whose
        #   ``on_fit_start`` fires after ``setup()`` and loads weights then.
        # * BASELINE: no checkpoint touched; trunk is loaded from
        #   ``model.trunk.checkpoint`` inside ``module.setup()`` as usual.
        if resume_mode is ResumeMode.WARM_START and resume_ckpt is not None:
            trainer.callbacks.append(_WarmStartCallback(resume_ckpt))
            fit_ckpt: str | None = None
        elif resume_mode is ResumeMode.CONTINUE:
            fit_ckpt = resume_ckpt
        else:
            fit_ckpt = None
        trainer.fit(model=module, datamodule=dm, ckpt_path=fit_ckpt)

        # No explicit final dump on graceful exit: ``last.ckpt`` (ModelCheckpoint
        # ``save_last``) already holds the final weights + optimiser + loop state
        # and is the resume anchor, so a separate ``ema_final.ckpt`` would be
        # redundant. ``ema_final.ckpt`` is reserved for the SigtermHandler's
        # preemption save (captures mid-epoch state at signal time).

        # B17 (2026-07-29): Record why training stopped so post-mortem audits
        # never have to guess. Resolves from live trainer state, not from config.
        # This permanently closes the class of error that produced the wrong §3
        # claim in v3a_retraining.md (EarlyStopping vs. total_steps confusion).
        _record_termination_reason(
            run_dir=run_dir,
            trainer=trainer,
            cfg=cfg,
            decision_path=decision_path,
        )

        # §18 validity criterion (BLOCKER 2 Assertion 2): grad_clip_active mean
        # must be < 5 % over all steps past step 5 000.  Hard raise — an arm that
        # violates this is invalid for the loss-norm comparison and must NOT be
        # reported without investigation.
        _assert_grad_clip_validity(run_dir=run_dir, cfg=cfg)

        if cfg.post_train.enabled:
            _run_post_train(run_dir, formats=cfg.post_train.formats)
        logger.info("FM-train completed; artifact dir: %s", run_dir)
        return run_dir
