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
import shutil
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vena.data.augment import AugmentationTracker, VariantTracker, build_pipeline_from_yaml
from vena.data.h5.shared import now_iso_utc, sha256_file
from vena.data.registry import load_registry
from vena.model.fm.lightning import FMLightningModule, MultiCohortLatentDataModule
from vena.model.fm.lightning.callbacks import (
    BestCheckpointCallback,
    ExhaustiveValLauncher,
    SigtermHandler,
    TrainMetricsCSV,
    VENACheckpointCallback,
)
from vena.model.fm.lightning.data import MultiCohortLatentDataModule
from vena.model.fm.maisi.config import TrunkConfig
from vena.model.fm.metrics import RegionSpec
from vena.preflight.cohort_dedup import (
    DedupDecisionSchemaError,
    assert_dedup_decision_valid,
    build_allowlists,
)

from .exceptions import PreflightGateError
from .runner import generate_run_id, write_provenance

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic schema (training_routine.md §2.3 + ``regions:`` block)
# =============================================================================


class _RunCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = "s1"
    resume_from: str | None = None
    seed: int = 1337
    device: str = "cuda"
    precision: str = "bf16-mixed"
    full_determinism: bool = False


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


class _ControlNetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning_inputs: list[str]
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    perturb_keys: list[str] = Field(default_factory=lambda: ["wt"])


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
    best_metric_name: str = "mse_latent"
    best_metric_region: str = "bg"
    best_metric_nfe: int = 5
    gradient_clip_val: float = 1.0
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
    # Prune ``ema_snapshot.pt`` / ``trunk_ema_snapshot.pt`` from epoch dirs older
    # than ``prune_snapshots_keep`` cadence epochs. ``latent_preds.h5`` and
    # ``metrics.csv`` are NEVER pruned — they are the long-run diagnostic record.
    # 0 disables pruning.
    prune_snapshots_keep: int = 2

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
    retention_n_checkpoints: int = 3
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


# =============================================================================
# Engine
# =============================================================================


class FMTrainRoutineEngine:
    """End-to-end training engine following training_routine.md."""

    def __init__(self, cfg: FMTrainRoutineConfig, config_yaml_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path

    def _make_run_dir(self) -> tuple[str, Path]:
        run_id = generate_run_id(self.cfg.run.stage)
        run_dir = Path(self.cfg.output.experiments_root) / run_id
        # ``qualitative`` and ``performance`` are no longer produced in-process —
        # their content (qualitative figures + latent preds, per-NFE timing) now
        # lives under ``exhaustive_val/epoch_NNN/`` (created by the launcher).
        for sub in ("checkpoints", "logs", "metrics"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _resolve_run_dir(self, resume_ckpt: str | None) -> tuple[str, Path, bool]:
        """Choose the run directory, continuing in place when resuming.

        If ``resume_ckpt`` points inside ``experiments_root`` (a prior run of
        this project), reuse that run's directory so the resumed training appends
        to the same metrics/checkpoints rather than forking a new empty run —
        essential for long, preemptible Picasso runs to stay one contiguous
        artifact, and it lets Lightning's ``ModelCheckpoint`` reload its full loop
        state (same ``dirpath``). An explicit external checkpoint, or no resume,
        creates a fresh timestamped directory.

        Returns
        -------
        tuple[str, Path, bool]
            ``(run_id, run_dir, resuming_in_place)``.
        """
        if resume_ckpt is not None:
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

    def _build_decision_payload(self, run_id: str, run_dir: Path) -> dict[str, Any]:
        """Schema-0.6.0 decision JSON written once at run creation.

        Carries enough provenance for a downstream consumer to reproduce the
        run end-to-end: data registry, trunk + VAE SHA-256, loss stage,
        optimiser/EMA hyperparameters, augmentation gate path, the list of
        cohort names actually wired in, the cohort deduplication decision
        file + SHA-256 used to filter the corpus, the offline-augmentation
        bank toggle + per-cohort latent_aug_h5 paths + variant_weights, and
        (from schema 0.6.0) the trunk regime + PEFT variant + params.
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
        return {
            "schema_version": "0.7.0",
            "produced_at": now_iso_utc(),
            "producer": "routines.fm.train:0.7.0",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stage": cfg.run.stage,
            "seed": cfg.run.seed,
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
            },
            "controlnet": {
                "conditioning_inputs": list(cfg.model.controlnet.conditioning_inputs),
                "arch_overrides": dict(cfg.model.controlnet.arch_overrides),
            },
            "vae_checkpoint": str(cfg.model.vae_checkpoint),
            "rflow": cfg.rflow.model_dump(),
            "ema": cfg.ema.model_dump(),
            "fold": cfg.data.fold,
            "nfe_levels": list(ev.nfe_levels),
            "integrator": ev.integrator,
            "n_patients": ev.n_patients,
            "figure_top_k": ev.figure_top_k,
        }
        job["corpus_registry"] = str(ev.corpus_registry)
        return job

    def _resolve_resume_ckpt(self, exclude_dir: Path | None = None) -> str | None:
        """Resolve ``run.resume_from`` to a checkpoint path.

        ``latest``/``best`` scan ``experiments_root`` newest-first and return the
        first run directory that actually contains the target checkpoint, skipping
        ``exclude_dir`` (the run we just created, whose ``checkpoints/`` is still
        empty). An explicit path is returned verbatim if it exists.
        """
        rf = self.cfg.run.resume_from
        if not rf:
            return None

        if rf in ("latest", "best"):
            root = Path(self.cfg.output.experiments_root)
            skip = exclude_dir.resolve() if exclude_dir is not None else None
            dirs = sorted(
                (d for d in root.glob("*/") if d.is_dir() and d.resolve() != skip),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            target = "last.ckpt" if rf == "latest" else "ema_best.ckpt"
            for d in dirs:
                cand = d / "checkpoints" / target
                if cand.is_file():
                    logger.info("Resuming (%s) from %s", rf, cand)
                    return str(cand)
                if rf == "latest":
                    ema = sorted((d / "checkpoints").glob("ema_epoch_*.ckpt"))
                    if ema:
                        logger.info("Resuming (latest) from %s", ema[-1])
                        return str(ema[-1])
            logger.warning(
                "resume_from=%r: no matching checkpoint under %s; starting fresh.", rf, root
            )
            return None

        p = Path(rf)
        if p.is_file():
            logger.info("Resuming from explicit path %s", p)
            return str(p)
        logger.warning("resume_from=%r could not be resolved; starting fresh.", rf)
        return None

    def run(self) -> Path:
        cfg = self.cfg
        pl.seed_everything(cfg.run.seed, workers=True)

        # Pre-flight gate. Raises ``PreflightGateError`` before anything else
        # has been done so the failure message names exactly which artifact is
        # missing or non-conformant. See ``.claude/rules/preflight-pattern.md``.
        _assert_preflight_gates(cfg)

        # TF32 matmul: ~10% speed-up on A100/RTX-4090 at no measured cost to
        # FM training numerics. Set before any model is built; ignored on CPU
        # and on GPUs that do not advertise TF32 capability.
        torch.set_float32_matmul_precision("high")

        # Resolve the resume checkpoint *before* choosing the run dir so we can
        # continue in place (same dir) instead of forking a new empty run.
        resume_ckpt = self._resolve_resume_ckpt()
        run_id, run_dir, resuming_in_place = self._resolve_run_dir(resume_ckpt)
        self._attach_file_log(run_dir)
        self._write_static_provenance(run_dir, resuming_in_place=resuming_in_place)
        logger.info(
            "FM-train run_id=%s dir=%s%s",
            run_id,
            run_dir,
            " (RESUMING IN PLACE)" if resuming_in_place else "",
        )

        # Decision JSON for downstream consumers — written once at run creation;
        # left intact on in-place resume.
        decision_path = run_dir / "decision.json"
        if not decision_path.exists():
            decision_path.write_text(
                json.dumps(self._build_decision_payload(run_id, run_dir), indent=2)
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
            TrainMetricsCSV(out_dir=run_dir / "metrics"),
            SigtermHandler(ckpt_dir=run_dir / "checkpoints", filename="ema_final.ckpt"),
        ]
        if train_transform is not None:
            callbacks.append(AugmentationTracker(out_dir=run_dir / "metrics"))
        if cfg.data.use_offline_augmented_data:
            callbacks.append(VariantTracker(out_dir=run_dir / "metrics"))
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
                "EarlyStopping ENABLED: monitor=%s mode=min patience=%d epochs",
                ckpt_monitor,
                int(cfg.training.patience),
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

        trainer.fit(model=module, datamodule=dm, ckpt_path=resume_ckpt)

        # No explicit final dump on graceful exit: ``last.ckpt`` (ModelCheckpoint
        # ``save_last``) already holds the final weights + optimiser + loop state
        # and is the resume anchor, so a separate ``ema_final.ckpt`` would be
        # redundant. ``ema_final.ckpt`` is reserved for the SigtermHandler's
        # preemption save (captures mid-epoch state at signal time).
        if cfg.post_train.enabled:
            _run_post_train(run_dir, formats=cfg.post_train.formats)
        logger.info("FM-train completed; artifact dir: %s", run_dir)
        return run_dir
