"""LightningModule wrapping trunk + ControlNet + RFlow + composite loss.

The VAE is always frozen. The trunk is controlled by ``trunk_config.trainable``
combined with ``trunk_config.regime``:

* ``trainable=False`` (canonical frozen-backbone recipe): the optimiser is
  constructed over ``self.controlnet.parameters()`` only and the trunk is held
  as an unregistered property — trunk weights are not written into
  checkpoints and the EMA shadow only tracks the ControlNet. ``regime`` must
  be ``'fft'``.

* ``trainable=True`` + ``regime='fft'`` (joint full fine-tune, TumorFlow-style):
  the trunk is unfrozen and joins the same optimiser group as the ControlNet.
  ``self._trunk_module`` is registered as a Lightning submodule so the
  fine-tuned trunk weights round-trip through ``state_dict`` natively (PL 2.x
  restores model weights *after* ``setup()``).

* ``trainable=True`` + ``regime='peft'`` (parameter-efficient adapter, LoRA &
  variants): the trunk's matched ``nn.Linear`` modules are replaced in place
  by adapter-bearing subclasses via
  :func:`vena.model.fm.maisi.peft.build_peft` and
  :meth:`vena.model.fm.maisi.peft.BasePEFT.apply`. The optimiser sees only the
  adapter tensors (filtered by ``requires_grad``); base trunk weights stay
  frozen. ``self._trunk_module`` registration covers both base and adapter
  tensors so the small adapter delta round-trips through ``state_dict``.
  Joint training with ControlNet is stable from step 0 because both
  components are identity-at-init (ControlNet zero-conv + LoRA zero-init B).

A second EMA — ``self.trunk_ema`` — is built in ``setup()`` (after PEFT
injection so it shadows adapter tensors too) and updated in lockstep with
the ControlNet EMA so sampling uses an EMA-smoothed trunk. Caveat:
``trunk_ema`` is created in ``setup()`` (after Lightning's checkpoint
restore), so the trainable-trunk paths are **single-shot, not resume-safe**
as written. Do not rely on ``run.resume_from`` for trainable runs without
first hardening the trunk-EMA restore path.

The training step follows MAISI-v2's ControlNet recipe (see
``.claude/rules/model-coding-standards.md``):

    down_residuals, mid_residual = controlnet(x_t, t, c_orig, class_labels)
    v = trunk(
        x_t, t,
        class_labels=class_labels,
        spacing_tensor=spacing,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
    )
    loss = composite(LossInputs(..., v_orig=v, v_perturb=optional))

The validation step:

* runs the EMA shadow model on a sampler (default Euler, NFE in {per-epoch,
  sweep_nfes}),
* computes region-masked latent metrics (always) and image metrics (decoded
  through the frozen VAE on the per-epoch NFE and on every sweep NFE),
* populates buffers consumed by callbacks (`val_csv`, `qualitative`,
  `nfe_timing`).
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# PyTorch 2.6+ enforces ``weights_only=True`` on ``torch.load`` by default,
# which rejects unknown picklable globals. Lightning's checkpoint round-trip
# stores our :meth:`on_save_checkpoint` payload including the NumPy RNG state
# tuple (``np.random.get_state()``), whose internals reference
# ``numpy._core.multiarray._reconstruct`` and an ``ndarray`` dtype. We
# allowlist the safe globals once at module import so resume just works.
def _register_safe_globals() -> None:
    try:
        from numpy._core.multiarray import _reconstruct

        torch.serialization.add_safe_globals(
            [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        )
    except Exception:
        pass


_register_safe_globals()

from ..controlnet.base import AbstractControlNet
from ..controlnet.conditioning import ConditioningAssembler, ConditioningSpec
from ..controlnet.losses import CompositeLoss, LossInputs, build_loss
from ..controlnet.maisi_controlnet import MaisiControlNet
from ..ema import WarmupEMA
from ..inference import NFETimingProbe, get_sampler
from ..maisi.config import TrunkConfig
from ..maisi.peft import BasePEFT, build_peft
from ..maisi.trunk import TrunkHandle, load_trunk
from ..metrics import ImageMetrics, LatentMetrics, RegionMasks, RegionResolver
from ..sampler.rflow import RFlowEngine

logger = logging.getLogger(__name__)


REGION_NAMES: tuple[str, ...] = ("full", "wt", "bg", "vessel")
REGION_TO_RESOLVER_KEY: dict[str, str] = {
    "full": "brain",  # full-brain mask
    "wt": "wt",
    "bg": "bg",
    "vessel": "vessel",
}


class FMLightningModule(pl.LightningModule):
    """End-to-end FM training step (ControlNet only)."""

    def __init__(
        self,
        trunk_config: TrunkConfig,
        conditioning_specs: list[str | ConditioningSpec],
        stage: str = "S1",
        loss_cfg: dict[str, Any] | None = None,
        perturb_keys: Iterable[str] | None = None,
        controlnet_arch_overrides: dict[str, Any] | None = None,
        optim_cfg: dict[str, Any] | None = None,
        rflow_cfg: dict[str, Any] | None = None,
        ema_cfg: dict[str, Any] | None = None,
        region_resolver: RegionResolver | None = None,
        validation_cfg: dict[str, Any] | None = None,
        vae_decoder: Any | None = None,
        nan_tolerance: dict[str, int] | None = None,
        conditioning_dropout_p: float = 0.0,
        conditioning_dropout_keys: Iterable[str] | None = None,
        lpl_config: Any = None,  # vena.model.fm.lpl.LplConfig (lazy import)
        lpl_vae_checkpoint: Path | None = None,
        # ---- S1 v3 (2026-06-22) ----
        controlnet_enabled: bool = True,
        controlnet_init_from_trunk_enabled: bool = True,
        input_concat_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # Lightning saves these into checkpoint hparams. We exclude unpicklables.
        self.save_hyperparameters(
            ignore=["trunk_config", "region_resolver", "vae_decoder", "lpl_config"]
        )

        self.trunk_config = trunk_config
        self.stage = stage
        self.perturb_keys: set[str] = set(perturb_keys or ()) if perturb_keys else {"wt"}
        # Classifier-free-guidance training-time dropout (per-sample Bernoulli
        # on the listed conditioning keys). ``p == 0`` disables the path
        # entirely → byte-identical to a run without CFG. Keys default to
        # ``{wt}`` matching the doc's CHANGE 3.
        if conditioning_dropout_p < 0.0 or conditioning_dropout_p > 1.0:
            raise ValueError(
                f"conditioning_dropout_p must be in [0, 1]; got {conditioning_dropout_p}"
            )
        self.conditioning_dropout_p: float = float(conditioning_dropout_p)
        self.conditioning_dropout_keys: set[str] = (
            set(conditioning_dropout_keys) if conditioning_dropout_keys else {"wt"}
        )

        # PEFT adapter handler. Built eagerly from ``trunk_config.peft`` so a
        # malformed config fails at __init__ time, not deep inside setup() under
        # DDP. Remains None for the FFT regime.
        self.peft_handler: BasePEFT | None = None
        if trunk_config.regime == "peft":
            assert trunk_config.peft is not None  # enforced by TrunkConfig validator
            self.peft_handler = build_peft(
                variant=trunk_config.peft["variant"],
                params=trunk_config.peft.get("params"),
            )
            logger.info(
                "PEFT handler ready: variant=%s params=%s",
                trunk_config.peft["variant"],
                trunk_config.peft.get("params"),
            )

        self._trunk_handle: TrunkHandle | None = None
        # Registered alias for the live trunk, set in setup() only when the trunk
        # is trainable. Registration puts the fine-tuned trunk weights into the
        # Lightning state_dict so they are saved and restored natively (PL 2.x
        # restores model weights *after* setup()). Frozen trunk stays unregistered
        # so frozen checkpoints are not bloated with 72 M immutable params.
        self._trunk_module: torch.nn.Module | None = None

        # ---- S1 v3 (2026-06-22) — controlnet/input_concat gating ----
        # Variant A: ``controlnet_enabled=False`` ⇒ NO ConditioningAssembler,
        # NO ControlNet, NO ControlNet EMA. Modality latents reach the trunk
        # exclusively via the channel-concat at ``conv_in`` (see
        # ``input_concat_cfg`` below). Variant B: ``controlnet_enabled=True``
        # with a 3-channel mask cond_embedding (``init_from_trunk_enabled=False``)
        # and modality latents still channel-concat into the trunk.
        self.controlnet_enabled: bool = bool(controlnet_enabled)
        self.input_concat_cfg: dict[str, Any] = dict(input_concat_cfg or {})
        self._input_concat_enabled: bool = bool(self.input_concat_cfg.get("enabled", False))
        self._input_concat_cond_latents: list[str] = list(
            self.input_concat_cfg.get("cond_latents") or []
        )

        if self.controlnet_enabled:
            self.conditioning = ConditioningAssembler(conditioning_specs)
            cond_in = self.conditioning.total_channels
            logger.info("FMLightningModule: conditioning_total_channels=%d", cond_in)
            self.controlnet: AbstractControlNet | None = MaisiControlNet(
                conditioning_in_channels=cond_in,
                arch_overrides=controlnet_arch_overrides or {},
                init_from_trunk_enabled=bool(controlnet_init_from_trunk_enabled),
            )
        else:
            self.conditioning = None  # type: ignore[assignment]
            self.controlnet = None  # type: ignore[assignment]
            logger.info(
                "FMLightningModule: ControlNet DISABLED (Variant A); "
                "input_concat_enabled=%s cond_latents=%s",
                self._input_concat_enabled,
                self._input_concat_cond_latents,
            )

        self.composite: CompositeLoss = build_loss(stage, loss_cfg or {})
        self.rflow = RFlowEngine(**(rflow_cfg or {}))
        self.optim_cfg: dict[str, Any] = optim_cfg or {}
        self.ema_cfg: dict[str, Any] = ema_cfg or {}

        # EMA must be built in __init__ so its parameters exist by the time
        # Lightning's checkpoint load_state_dict runs (it loads *before*
        # setup()). Variant A (controlnet_enabled=False) has no ControlNet
        # to shadow, so ``self.ema`` stays ``None``; sampling reads the
        # trunk EMA shadow only.
        self.ema: WarmupEMA | None = (
            WarmupEMA(self.controlnet, **self.ema_cfg) if self.controlnet is not None else None
        )
        # Trunk EMA only exists in the unfrozen-trunk ablation; it is built in
        # setup() once the trunk is loaded (the trunk does not exist in
        # __init__). Resume-safety is delivered by the R6 path: the engine
        # sets ``_pending_trunk_ema_snapshot`` via
        # :meth:`set_pending_trunk_ema_snapshot` *before* ``trainer.fit``, and
        # :meth:`setup` reloads the saved shadow into the freshly-built
        # ``trunk_ema`` so a S1→S3 warm-start continues the same EMA average.
        self.trunk_ema: WarmupEMA | None = None
        # Path to a ``trunk_ema_shadow_state_dict`` saved by
        # :class:`TrunkEMASnapshotCallback`. ``None`` (default) → leave
        # ``trunk_ema`` at its fresh init (legacy behaviour). Set publicly via
        # :meth:`set_pending_trunk_ema_snapshot` from the engine when a
        # warm-start is resolved.
        self._pending_trunk_ema_snapshot: Path | None = None
        # ----- S3 LPL artefacts (per ``model-coding-standards.md`` §4.5) -----
        # Built only when the stage is S3 AND the engine passed an
        # ``LplConfig``. The VAE handle is loaded lazily in :meth:`setup`
        # to keep the build cheap when LPL is off.
        self.lpl_config = lpl_config
        self.lpl_vae_checkpoint: Path | None = (
            Path(lpl_vae_checkpoint) if lpl_vae_checkpoint else None
        )
        self.feature_stats: Any = None  # vena.model.fm.lpl.FeatureStatsEMA
        self.lpl_loss: Any = None  # vena.model.fm.lpl.LplLoss
        self._lpl_vae_handle: Any = None  # AutoencoderHandle, loaded in setup
        if self.lpl_config is not None and stage.upper() == "S3":
            from vena.common import LATENT_CHANNELS
            from vena.model.fm.lpl import FeatureStatsEMA, LplLoss

            # Per-block channels — discovered statically from the design
            # doc's §3.5 geometry table (block→channels):
            #   blocks 0,1,2 = 256 (latent res)
            #   block 3 = 256 (level-0→1 Upsample preserves channels)
            #   blocks 4, 5 = 128 (level-1 ResBlocks, channel halving)
            # The VAE is frozen so the channel counts cannot drift between
            # init and setup.
            _CH = {0: 256, 1: 256, 2: 256, 3: 256, 4: 128, 5: 128}
            stats_channels = {blk: _CH[blk] for blk in self.lpl_config.A}
            self.feature_stats = FeatureStatsEMA(channels=stats_channels)
            self.lpl_loss = LplLoss(self.lpl_config, self.feature_stats)
            logger.info(
                "S3 LPL artefacts built: A=%s w_l=%s t_min=%.3f lambda_img=%.3f",
                self.lpl_config.A,
                self.lpl_config.w_l,
                self.lpl_config.t_min,
                self.lpl_config.lambda_img,
            )
            _ = LATENT_CHANNELS  # silence the unused-import noise on lint

        # Validation/region wiring.
        self.region_resolver = region_resolver
        self.validation_cfg: dict[str, Any] = validation_cfg or {}
        self.vae_decoder = vae_decoder
        self.latent_metrics = LatentMetrics()
        self.image_metrics: ImageMetrics | None = None  # built lazily if vae_decoder set

        # Buffers consumed by callbacks.
        self._val_accumulator: dict[tuple[int, str], dict[str, Any]] = {}
        self._qualitative_buffer: dict[tuple[str, int], torch.Tensor] = {}
        # Per-epoch NFE timing accumulator, keyed by nfe. Each value collects
        # per-batch lists so the callback can emit one aggregated row per
        # (epoch, nfe) instead of one row per validation batch.
        self._nfe_timing_accum: dict[int, dict[str, Any]] = {}

        # NaN guard counters.
        nt = nan_tolerance or {}
        self._nan_max_in_window = int(nt.get("max_in_window", 10))
        self._nan_window = int(nt.get("window_steps", 1000))
        self._nan_history: list[int] = []  # step indices where NaN occurred

        # Step-time tracking (logged per training step).
        self._step_t0: float | None = None
        # Last optimiser step at which the EMA updated (grad-accum guard).
        # 0 = "no optimiser step yet"; ``global_step`` advances past it only
        # once a real optimiser step completes.
        self._last_ema_step: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self._trunk_handle is None:
            self._setup_trunk_and_controlnet()
        # Move EMA shadow to the same device as the live model.
        if self.ema is not None:
            self.ema = self.ema.to(self.device)
        # Unfrozen-trunk ablation: a second EMA over the trunk so that sampling
        # (validation + exhaustive job) uses EMA-smoothed trunk weights exactly
        # as it uses the EMA ControlNet. Built here because the trunk does not
        # exist until ``_setup_trunk_and_controlnet`` has run.
        if self.trunk_config.trainable and self.trunk_ema is None:
            self.trunk_ema = WarmupEMA(self.trunk, **self.ema_cfg).to(self.device)
            logger.info("Trunk EMA shadow created (unfrozen-trunk ablation).")
            self._maybe_load_trunk_ema_snapshot()
        if self.image_metrics is None and self.vae_decoder is not None:
            self.image_metrics = ImageMetrics()
        # Lazy VAE handle load for the S3 LPL gated branch.
        if (
            self.lpl_loss is not None
            and self._lpl_vae_handle is None
            and self.lpl_vae_checkpoint is not None
        ):
            from vena.common import load_autoencoder

            self._lpl_vae_handle = load_autoencoder(
                self.lpl_vae_checkpoint, device=str(self.device)
            )
            logger.info("S3 LPL VAE handle loaded for in-process decoder-feature extraction.")

    def set_pending_trunk_ema_snapshot(self, path: Path | None) -> None:
        """Public setter for the warm-start trunk-EMA snapshot path (R6).

        Called by the engine *before* ``trainer.fit`` when a WARM_START
        resume mode is resolved. ``setup`` reads this attribute and reloads
        the saved shadow state_dict into ``self.trunk_ema.ema_model``,
        keeping the EMA average continuous across the S1→S3 boundary.

        A value of ``None`` (default) means "no snapshot — leave the
        freshly-built trunk_ema at its current state". This is the legacy
        path; callers must not raise just because the snapshot is missing.
        """
        self._pending_trunk_ema_snapshot = path

    def _maybe_load_trunk_ema_snapshot(self) -> None:
        """Restore the trunk-EMA shadow from the R6 snapshot when set.

        No-op when the pending path is ``None``. Warns (does not raise)
        when ``trunk_config.trainable`` is on but the file is missing —
        that's the documented "pre-R6 S1 checkpoint" path which should
        still run (with the caveat that the trunk-EMA starts fresh).
        """
        snapshot = self._pending_trunk_ema_snapshot
        if snapshot is None:
            return
        if not snapshot.exists():
            logger.warning(
                "Trunk EMA snapshot expected at %s but not found; trunk_ema"
                " starts fresh (legacy pre-R6 warm-start path).",
                snapshot,
            )
            return
        assert self.trunk_ema is not None, "snapshot load needs trunk_ema built"
        shadow_sd = torch.load(snapshot, map_location=self.device, weights_only=True)
        missing, unexpected = self.trunk_ema.ema_model.load_state_dict(shadow_sd, strict=False)
        if unexpected:
            logger.warning(
                "Trunk EMA snapshot %s contained %d unexpected keys: %s",
                snapshot,
                len(unexpected),
                list(unexpected)[:8],
            )
        if missing:
            logger.warning(
                "Trunk EMA snapshot %s missing %d keys vs live shadow: %s",
                snapshot,
                len(missing),
                list(missing)[:8],
            )
        logger.info(
            "Trunk EMA shadow restored from %s (%d tensors loaded).",
            snapshot,
            len(shadow_sd) - len(missing),
        )

    def _setup_trunk_and_controlnet(self) -> None:
        ckpt = Path(self.trunk_config.checkpoint)
        arch_json = Path(self.trunk_config.arch_json) if self.trunk_config.arch_json else None
        self._trunk_handle = load_trunk(
            checkpoint_path=ckpt,
            device=self.device,
            arch_config=arch_json,
            arch_overrides=self.trunk_config.arch_overrides or None,
            trainable=self.trunk_config.trainable,
        )
        # IMPORTANT: ``init_from_trunk`` must consume the trunk state_dict with
        # its pretrained MAISI key paths (e.g. ``to_q.weight``). PEFT injection
        # replaces those Linears with ``LoraLayer`` wrappers whose state_dict
        # keys would be ``to_q.base_layer.weight`` + ``to_q.lora_A.default.weight``
        # — so we MUST apply PEFT AFTER ``init_from_trunk``.
        trunk_sd = self._trunk_handle.model.state_dict()
        if self.controlnet is not None:
            self.controlnet.init_from_trunk(trunk_sd)
            self.controlnet.zero_init_output_projections()
        if self.peft_handler is not None:
            self.peft_handler.apply(self._trunk_handle.model)
            adapter_params = self.peft_handler.trainable_parameters(self._trunk_handle.model)
            n_trainable = sum(p.numel() for p in adapter_params)
            logger.info(
                "PEFT injected on trunk (%s): %d adapter tensors, %d trainable params",
                self.trunk_config.peft.get("variant") if self.trunk_config.peft else "?",
                len(adapter_params),
                n_trainable,
            )
        if self.trunk_config.trainable:
            # Register the trunk so its weights (base + any PEFT adapter slots)
            # are checkpointed and restored natively. On resume, setup() reloads
            # the *original* MAISI trunk here (harmless), then Lightning's
            # post-setup state_dict restore overwrites both base and adapter
            # tensors (and trunk_ema + optimiser state) with the fine-tuned
            # values.
            self._trunk_module = self._trunk_handle.model
        if self.controlnet is not None:
            self.controlnet = self.controlnet.to(self.device)
        logger.info(
            "FMLightningModule.setup: trunk on %s (sha=%s) controlnet=%s regime=%s",
            self._trunk_handle.device,
            self._trunk_handle.checkpoint_sha256[:12],
            "ON" if self.controlnet is not None else "DISABLED (Variant A)",
            self.trunk_config.regime,
        )

    @property
    def trunk(self) -> torch.nn.Module:
        if self._trunk_handle is None:
            raise RuntimeError("trunk not loaded — call setup() first")
        return self._trunk_handle.model

    # ------------------------------------------------------------------
    # Padding helpers (trunk requires dims divisible by 8).
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        sizes = x.shape[-3:]
        pad_h = (multiple - sizes[0] % multiple) % multiple
        pad_w = (multiple - sizes[1] % multiple) % multiple
        pad_d = (multiple - sizes[2] % multiple) % multiple
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x, (0, 0, 0)
        padded = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        return padded, (pad_h, pad_w, pad_d)

    @staticmethod
    def _unpad(x: torch.Tensor, pad: tuple[int, int, int]) -> torch.Tensor:
        pad_h, pad_w, pad_d = pad
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x
        return x[..., : x.shape[-3] - pad_h, : x.shape[-2] - pad_w, : x.shape[-1] - pad_d]

    def _trunk_forward(
        self,
        controlnet: AbstractControlNet | None,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor | None,
        class_labels: torch.Tensor,
        spacing: torch.Tensor,
        probe: NFETimingProbe | None = None,
        trunk: torch.nn.Module | None = None,
        latents_concat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the (optionally-ControlNet-augmented) trunk forward.

        S1 v3 (2026-06-22) adds two optional behaviours, both back-compat
        with the S1 v2 caller:

        * ``controlnet=None`` ⇒ Variant A path: skip the ControlNet forward
          entirely; pass ``down_res=None, mid_res=None`` to the trunk. ``cond``
          is ignored.
        * ``latents_concat`` not None ⇒ build the trunk's ``x`` argument as
          ``torch.cat([x_t_p, latents_concat_p], dim=1)``. The trunk's
          ``conv_in`` must have been expanded (via
          :func:`vena.model.fm.maisi.conv_in_expand.expand_conv_in`) to accept
          the wider input. The ControlNet (if present) still receives the
          plain ``x_t_p`` — its conditioning is the mask only in v3.
        """
        # ``trunk`` defaults to the live trunk (training path). The EMA-call
        # closure passes the EMA trunk shadow so sampling uses smoothed weights.
        trunk_model = trunk if trunk is not None else self.trunk
        x_t_p, pad = self._pad_to_multiple(x_t, multiple=8)

        # ControlNet branch (Variant B) — optional.
        down_res = None
        mid_res = None
        if controlnet is not None and cond is not None:
            cond_p, _ = self._pad_to_multiple(cond, multiple=8)
            cn_ctx = probe.section("controlnet") if probe is not None else nullcontext()
            with cn_ctx:
                down_res, mid_res = controlnet(
                    x=x_t_p,
                    timesteps=timesteps,
                    controlnet_cond=cond_p,
                    class_labels=class_labels,
                )

        # Build the trunk's input. If ``latents_concat`` is supplied
        # (S1 v3 Variants A + B), channel-concat after padding so spatial
        # shapes align.
        if latents_concat is not None:
            latents_concat_p, _ = self._pad_to_multiple(latents_concat, multiple=8)
            trunk_input = torch.cat([x_t_p, latents_concat_p], dim=1)
        else:
            trunk_input = x_t_p

        trunk_ctx = probe.section("trunk") if probe is not None else nullcontext()
        with trunk_ctx:
            v_p = trunk_model(
                x=trunk_input,
                timesteps=timesteps,
                class_labels=class_labels,
                spacing_tensor=spacing,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
        return self._unpad(v_p, pad)

    def _build_trunk_input_latents_concat(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Assemble the channel-concat conditioning tensor for the trunk input.

        Returns ``None`` when input-concat is disabled (S1 v2 behaviour). When
        enabled, concatenates ``batch["z_<name>"]`` for each name in
        ``input_concat_cfg.cond_latents`` along the channel axis. The result
        has shape ``(B, sum_channels, h, w, d)`` and is concatenated to ``x_t``
        in :meth:`_trunk_forward` to form the trunk's first-conv input.
        """
        if not self._input_concat_enabled:
            return None
        if not self._input_concat_cond_latents:
            return None
        pieces: list[torch.Tensor] = []
        for name in self._input_concat_cond_latents:
            key = f"z_{name}"
            if key not in batch:
                raise KeyError(
                    f"input_concat_cfg requested latent '{name}' but batch is missing key '{key}'"
                )
            pieces.append(batch[key])
        return torch.cat(pieces, dim=1)

    # ------------------------------------------------------------------
    # Classifier-free-guidance dropout helper.
    # ------------------------------------------------------------------

    def _build_conditioning_with_cfg_dropout(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the trunk conditioning, mixing kept + dropped channels per sample.

        Implements the training-time classifier-free-guidance recipe (Ho &
        Salimans 2022; ControlNet §3.5 / appendix A.2): for each sample,
        independently flip a Bernoulli(``p``) coin. If True, replace the
        listed conditioning keys with zeros (the ``perturb_keys`` path of the
        :class:`ConditioningAssembler`). If False, keep the original
        conditioning.

        ``conditioning_dropout_p == 0`` returns the unperturbed conditioning
        directly — byte-identical to the legacy path. Logs
        ``train/cfg_dropout_active_frac`` as a per-step Bernoulli observation.
        """
        # Variant A: no ConditioningAssembler — return None and let the
        # caller pass that through to _trunk_forward, which skips CN.
        if self.conditioning is None:
            return None  # type: ignore[return-value]
        if self.conditioning_dropout_p <= 0.0:
            return self.conditioning(batch)

        cond_keep = self.conditioning(batch)
        cond_drop = self.conditioning(batch, perturb_keys=self.conditioning_dropout_keys)
        drop = torch.rand(batch_size, device=device) < self.conditioning_dropout_p
        # Broadcast (B,) -> (B, 1, 1, 1, 1) so torch.where picks the whole
        # spatial+channel tensor per sample.
        mask = drop.view(-1, 1, 1, 1, 1)
        cond = torch.where(mask, cond_drop, cond_keep)
        # Per-step diagnostic: empirical Bernoulli rate. Stays near p_drop.
        self.log(
            "train/cfg_dropout_active_frac",
            drop.float().mean(),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        return cond

    # ------------------------------------------------------------------
    # Training step.
    # ------------------------------------------------------------------

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._step_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        x1 = batch["z_t1c"]
        B = x1.shape[0]
        device = x1.device

        x0 = torch.randn_like(x1)
        timesteps = self.rflow.sample_timesteps(x1).to(device)
        x_t = self.rflow.add_noise(x1, x0, timesteps)
        u_target = self.rflow.target_velocity(x1, x0)

        class_labels = self.trunk_config.make_class_labels(B, device)
        spacing = self.trunk_config.make_spacing_tensor(B, device)

        # Classifier-free-guidance dropout (per-sample Bernoulli on the listed
        # conditioning keys). ``conditioning_dropout_p == 0`` is the legacy path
        # — a single ``self.conditioning(batch)`` call. See doc CHANGE 3.
        cond_orig = self._build_conditioning_with_cfg_dropout(batch, B, device)
        # S1 v3: when input_concat is enabled, build the channel-concat tensor
        # once and reuse for both the v_orig and v_perturb passes (its content
        # is identical across CFG-dropout choices).
        latents_concat = self._build_trunk_input_latents_concat(batch)
        v_orig = self._trunk_forward(
            self.controlnet,
            x_t,
            timesteps,
            cond_orig,
            class_labels,
            spacing,
            latents_concat=latents_concat,
        )

        v_perturb: torch.Tensor | None = None
        if self.composite.requires_perturbed_pass and self.conditioning is not None:
            cond_perturb = self.conditioning(batch, perturb_keys=self.perturb_keys)
            v_perturb = self._trunk_forward(
                self.controlnet,
                x_t,
                timesteps,
                cond_perturb,
                class_labels,
                spacing,
                latents_concat=latents_concat,
            )

        m_wt = batch.get("m_wt")
        m_bg = (
            _bg_from_wt(m_wt)
            if (m_wt is not None and self.composite.requires_perturbed_pass)
            else None
        )
        m_brain = batch.get("m_brain")
        m_tumor = batch.get("m_tumor")
        inputs = LossInputs(
            x_clean=x1,
            noise=x0,
            x_t=x_t,
            timesteps=timesteps,
            u_target=u_target,
            v_orig=v_orig,
            v_perturb=v_perturb,
            m_wt=m_wt,
            m_bg=m_bg,
            m_brain=m_brain,
            m_tumor=m_tumor,
        )
        total_steps = self._estimated_total_steps()
        total, per_term = self.composite(
            inputs,
            global_step=int(self.global_step),
            total_steps=total_steps,
        )
        # ----- S3 LPL gated branch -----
        # Computes the decoder-feature perceptual term per
        # ``.claude/notes/changes/decoder_perceptual_loss_s3.md``. The gate
        # (``t > t_min``) is inside :meth:`LplLoss.forward`; this block only
        # decides whether to assemble the feature dicts. Skipped silently
        # when LPL is off (every stage except S3) or the VAE handle is not
        # loaded yet (smoke before setup).
        if (
            self.lpl_loss is not None
            and self._lpl_vae_handle is not None
            and m_wt is not None
            and m_brain is not None
        ):
            from vena.model.fm.lpl import compute_lambda_img, decoder_feature_extractor

            # `timesteps` is the integer code in [0, num_train_timesteps) that
            # MONAI's RFlowScheduler returns (use_discrete_timesteps=True). The
            # MAISI/MONAI noise schedule is
            #     x_t = (1 - α) x_clean + α x_noise     with α = timesteps / T
            #     u   = x_clean - x_noise               (RFlowEngine.target_velocity)
            #     → x_1 = x_t + α · u  ⇒  x̂_1 = x_t + α · v_orig
            # The design-note convention (decoder_perceptual_loss_s3.md §0)
            # uses t_dn = 1 - α (1 = data, 0 = noise) and the high-SNR gate
            # is t_dn > t_min — so we hand t_dn to LplLoss for its own gate
            # while building x̂_1 from α here.
            T = float(self.rflow.num_train_timesteps)
            alpha = timesteps.float() / T  # noise fraction ∈ [0, 1]
            t_dn = 1.0 - alpha  # data fraction; design-note "t"
            alpha_view = alpha.view(-1, *([1] * (x_t.ndim - 1)))
            x1_hat = x_t + alpha_view * v_orig
            blocks = frozenset(int(b) for b in self.lpl_config.A)
            max_block = int(max(self.lpl_config.A))
            grad_ckpt = bool(self.lpl_config.grad_checkpoint_segments >= 2)
            # Schedule-driven lambda_img — falls back to the static field when
            # no schedule is set (legacy / unit-test path).
            if self.lpl_config.schedule is not None:
                lam_active = compute_lambda_img(self.lpl_config.schedule, int(self.current_epoch))
            else:
                lam_active = float(self.lpl_config.lambda_img)
            # Pre-populate every LPL diagnostic key with a NaN placeholder so
            # the train_step.csv header includes the LPL columns even when the
            # very first batch OOMs (otherwise the header is auto-frozen from
            # the surviving keys and the CSV becomes blind to LPL outcomes).
            # ``lambda_img_active`` is logged unconditionally (not NaN) — it is
            # a property of the schedule, independent of whether the step ran.
            # ``lpl_skipped`` is the 0/1 visibility column so the CSV reader
            # can count silent skips at a glance.
            nan_dev = total.device
            nan_t = torch.full((), float("nan"), device=nan_dev)
            per_term["lpl"] = nan_t.clone()
            per_term["lambda_img_active"] = torch.as_tensor(lam_active, device=nan_dev)
            per_term["hi_frac"] = nan_t.clone()
            per_term["lpl_skipped"] = torch.zeros((), device=nan_dev)
            for blk in self.lpl_config.A:
                per_term[f"lpl_b{int(blk)}"] = nan_t.clone()
            for region in self.lpl_config.region_set:
                per_term[f"lpl_{region}"] = nan_t.clone()
            try:
                with decoder_feature_extractor(
                    self._lpl_vae_handle,
                    blocks=blocks,
                    max_block=max_block,
                    grad_checkpoint=grad_ckpt,
                ) as extract:
                    phi_pred = extract(x1_hat)
                    with torch.no_grad():
                        phi_tgt = extract(x1.detach())
                self.feature_stats.update(phi_pred)
                # LplLoss applies the `t_dn > t_min` high-SNR gate internally;
                # pass the normalised data-fraction (design-note "t"), not the
                # raw integer timesteps that MONAI returns.
                lpl_scalar, lpl_break = self.lpl_loss(phi_pred, phi_tgt, m_wt, m_brain, t_dn)
                total = total + lam_active * lpl_scalar
                per_term["lpl"] = lpl_scalar.detach()
                # CSV-monitor parity (LPL FIX 2026-06-20):
                # ``CompositeLoss.forward`` writes ``per_term["total"]`` at
                # ``losses/base.py:167`` *before* the LPL term is added to the
                # local ``total`` above, so the per-step CSV column ``total``
                # equals cfm even when LPL is active. Refresh it here so the
                # CSV matches the value that goes to ``loss.backward()``.
                per_term["total"] = total.detach()
                # Overwrite the per-term breakdown with real values now that we
                # made it past the partial decode.
                for k, v in lpl_break.items():
                    per_term[k] = (
                        torch.as_tensor(v, device=nan_dev) if not torch.is_tensor(v) else v
                    )
            except RuntimeError as exc:
                # OOM, shape mismatch, or other CUDA-side error. Log and keep
                # the NaN placeholders so the per-step CSV records the skip.
                logger.warning("S3 LPL step skipped: %s", exc)
                per_term["lpl_skipped"] = torch.ones((), device=nan_dev)
                # Release the cached blocks of memory that the OOM left
                # half-allocated; otherwise consecutive batches keep failing
                # at the same allocation boundary.
                torch.cuda.empty_cache()
        # Per-cohort CFM breakdown (P1.2). The multi-cohort dataset attaches a
        # ``cohort`` string per sample; the DataLoader collates strings into a
        # list. When all samples are from one cohort the .mean() across that
        # cohort equals the global cfm; otherwise the per-cohort values diverge
        # and reveal cohort-imbalanced drift.
        cohort_tags = batch.get("cohort")
        if cohort_tags is not None and v_orig.shape[0] > 1:
            # CFM is MSE between v_orig and u_target; recompute per-sample then
            # group by cohort (cheap: B≤8, mean is sub-microsecond).
            per_sample = (v_orig.detach() - u_target).pow(2).flatten(1).mean(dim=1)
            cohort_groups: dict[str, list[float]] = {}
            for i, tag in enumerate(cohort_tags):
                cohort_groups.setdefault(str(tag), []).append(float(per_sample[i].item()))
            for tag, vals in cohort_groups.items():
                # Sanitise cohort name for CSV column compatibility.
                safe = tag.replace("/", "_").replace(" ", "_")
                self.log(
                    f"train/cfm_cohort_{safe}",
                    sum(vals) / len(vals),
                    on_step=True,
                    on_epoch=False,
                    batch_size=len(vals),
                )

            # Per-cohort contrastive breakdown. Only fires when the composite
            # included a contrastive term (S2/S3); ``per_sample()`` returns the
            # cached (B,) tensor the term stored during its forward call.
            # ``ModuleDict`` does not implement ``.get``; check membership manually.
            contrastive_term = (
                self.composite.terms["contrastive"]
                if "contrastive" in self.composite.terms
                else None
            )
            contrastive_per_sample = (
                contrastive_term.per_sample() if contrastive_term is not None else None
            )
            if contrastive_per_sample is not None:
                cohort_groups_contrastive: dict[str, list[float]] = {}
                for i, tag in enumerate(cohort_tags):
                    cohort_groups_contrastive.setdefault(str(tag), []).append(
                        float(contrastive_per_sample[i].item())
                    )
                for tag, vals in cohort_groups_contrastive.items():
                    safe = tag.replace("/", "_").replace(" ", "_")
                    self.log(
                        f"train/contrastive_cohort_{safe}",
                        sum(vals) / len(vals),
                        on_step=True,
                        on_epoch=False,
                        batch_size=len(vals),
                    )

        # NaN guard.
        if not torch.isfinite(total):
            self._nan_history.append(int(self.global_step))
            self._nan_history = [
                s for s in self._nan_history if self.global_step - s <= self._nan_window
            ]
            logger.error(
                "NaN/Inf loss at step %d (%d in last %d steps)",
                self.global_step,
                len(self._nan_history),
                self._nan_window,
            )
            if len(self._nan_history) >= self._nan_max_in_window:
                raise RuntimeError(
                    f"Training diverged: {len(self._nan_history)} NaN losses "
                    f"in the last {self._nan_window} steps."
                )
            return None  # skip this step

        for name, value in per_term.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "total"),
                batch_size=B,
            )
        # Epoch-aggregated training loss under a distinct key — the
        # checkpoint monitor (ema_best) selects on this when in-process
        # validation is offloaded to the async second-GPU job. Distinct name so
        # the per-step ``train/total`` key the train CSV reads is not renamed.
        #
        # LPL FIX 2026-06-20: during S3 warmup the schedule ramps
        # ``lam_active`` from 0 to ``lambda_max`` over ``warmup_epochs``,
        # which makes the *live* ``total`` grow monotonically purely from
        # the weight schedule — not from loss-quality degradation. With
        # ``EarlyStopping(monitor="train/total_epoch", mode="min")`` and
        # ``patience == warmup_epochs`` this kills training at exactly the
        # post-warmup epoch where LPL would start to teach. Project the
        # monitor onto the steady-state lambda_max so EarlyStopping sees
        # the objective the run is actually minimising. S1/S2 (no LPL) is
        # untouched: ``lpl`` is absent from ``per_term`` and the original
        # ``total`` is logged. ``cfm_epoch`` is exported as a stable
        # secondary signal for diagnostics and post-hoc analysis.
        monitor_total = total
        lpl_val = per_term.get("lpl")
        cfm_val = per_term.get("cfm")
        if (
            self.lpl_loss is not None
            and self.lpl_config is not None
            and isinstance(lpl_val, torch.Tensor)
            and torch.is_tensor(cfm_val)
            and torch.isfinite(lpl_val).all()
        ):
            schedule = getattr(self.lpl_config, "schedule", None)
            lam_max = (
                float(getattr(schedule, "lambda_max", lam_active))
                if schedule is not None
                else float(lam_active)
            )
            monitor_total = cfm_val + lam_max * lpl_val
        self.log("train/total_epoch", monitor_total, on_step=False, on_epoch=True, batch_size=B)
        if torch.is_tensor(cfm_val):
            self.log("train/cfm_epoch", cfm_val, on_step=False, on_epoch=True, batch_size=B)
        # Sanity on the timestep sampler (should hover near T/2 for uniform).
        self.log("train/t_mean", timesteps.float().mean(), on_step=True, on_epoch=False)
        if self._step_t0 is not None:
            step_time = time.perf_counter() - self._step_t0
            self.log("train/step_time_sec", step_time, on_step=True, on_epoch=False)
            self.log(
                "train/samples_per_sec",
                float(B) / max(step_time, 1e-9),
                on_step=True,
                on_epoch=False,
            )
        if torch.cuda.is_available():
            # Peak (not current) allocation gives OOM headroom; reset in
            # ``on_train_batch_start`` so this reflects the step just executed.
            self.log(
                "train/gpu_mem_peak_mb",
                float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                on_step=True,
                on_epoch=False,
            )
        return total

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema is None:
            return
        # ``on_train_batch_end`` fires once per *micro-batch*. With gradient
        # accumulation (``accumulate_grad_batches > 1``) the optimizer steps
        # only every N micro-batches, so the EMA must update once per optimiser
        # step — not per micro-batch — or the shadow decays N× too fast.
        # ``trainer.global_step`` increments only on optimiser steps, so we gate
        # on it *advancing* — this also skips the pre-first-step accumulation
        # micro-batches where ``global_step`` is still 0.
        step = int(self.trainer.global_step)
        if step <= self._last_ema_step:
            return
        self._last_ema_step = step
        if self.ema is not None:
            self.ema.update()
        if self.trunk_ema is not None:
            # Same once-per-optimiser-step gate as the ControlNet EMA above.
            self.trunk_ema.update()
        # Log the EMA decay used this step. Prefer the ControlNet EMA's value
        # (S1 v2 default); fall back to the trunk EMA when no CN EMA exists
        # (S1 v3 Variant A). When neither EMA exists, the log is skipped.
        active_ema = self.ema if self.ema is not None else self.trunk_ema
        if active_ema is not None:
            self.log(
                "train/ema_decay",
                active_ema.get_current_decay(),
                on_step=True,
                on_epoch=False,
            )

    def _estimated_total_steps(self) -> int | None:
        """Total optimiser-step budget for this run, used by weight schedules.

        Prefers Lightning's ``trainer.estimated_stepping_batches`` (available
        once ``trainer.fit`` has computed the dataloader size), falling back to
        ``optim_cfg["max_steps"]`` (read from the YAML's ``training.total_steps``
        in the engine) when the trainer is not attached yet. Returns ``None``
        when neither is known so schedules can no-op.
        """
        if getattr(self, "trainer", None) is not None:
            try:
                est = int(self.trainer.estimated_stepping_batches)
                if est > 0:
                    return est
            except (AttributeError, ValueError, TypeError):
                pass
        ms = self.optim_cfg.get("max_steps")
        try:
            ms_int = int(ms) if ms is not None else None
            return ms_int if ms_int and ms_int > 0 else None
        except (TypeError, ValueError):
            return None

    def _trainable_grad_norm(self) -> torch.Tensor:
        """Global L2 norm over all optimised parameters.

        ControlNet (when enabled) + the trunk when ``trunk_config.trainable``
        (the unfrozen-trunk ablation). Variant A (no ControlNet) returns the
        trunk grad norm alone — matches the optimiser's param set.
        """
        sq_sum = torch.zeros((), device=self.device)
        params: list[torch.nn.Parameter] = []
        if self.controlnet is not None:
            params.extend(self.controlnet.parameters())
        if self.trunk_config.trainable:
            params.extend(self.trunk.parameters())
        for p in params:
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().float().pow(2).sum()
        return sq_sum.sqrt()

    def _trunk_grad_norm(self) -> torch.Tensor:
        """Global L2 norm over the trunk parameters only (unfrozen-trunk run).

        Returned independently so the train CSV can monitor the trunk's
        gradient magnitude alongside the combined-norm and detect the case where
        the trunk explodes while the ControlNet stays quiet (or vice versa).
        Callers gate on ``self.trunk_config.trainable``.
        """
        sq_sum = torch.zeros((), device=self.device)
        for p in self.trunk.parameters():
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().float().pow(2).sum()
        return sq_sum.sqrt()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip ControlNet gradients and log pre/post-clip norms.

        Lightning calls this once per optimiser step (grad-accum-safe), after
        ``backward`` and before ``optimizer.step``. We measure the norm before
        and after the clip so the logs show both the raw gradient magnitude
        (stability signal) and the effective post-clip norm, plus whether the
        clip was active this step.
        """
        pre = self._trainable_grad_norm()
        trunk_pre: torch.Tensor | None = (
            self._trunk_grad_norm() if self.trunk_config.trainable else None
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        post = self._trainable_grad_norm()
        trunk_post: torch.Tensor | None = (
            self._trunk_grad_norm() if self.trunk_config.trainable else None
        )
        # ``grad_norm_cn_*`` is misnamed historically — it is the combined
        # ControlNet + (unfrozen) trunk norm. Kept as-is for CSV back-compat with
        # earlier runs. The trunk-only keys below are the actual decomposition.
        self.log("train/grad_norm_cn_preclip", pre, on_step=True, on_epoch=False)
        self.log("train/grad_norm_cn_postclip", post, on_step=True, on_epoch=False)
        if trunk_pre is not None and trunk_post is not None:
            self.log("train/grad_norm_trunk_preclip", trunk_pre, on_step=True, on_epoch=False)
            self.log("train/grad_norm_trunk_postclip", trunk_post, on_step=True, on_epoch=False)
        if gradient_clip_val:
            self.log(
                "train/grad_clip_active",
                (pre > float(gradient_clip_val)).float(),
                on_step=True,
                on_epoch=False,
            )

    # ------------------------------------------------------------------
    # Validation step.
    # ------------------------------------------------------------------

    def _make_ema_call(self, probe: NFETimingProbe | None = None) -> Any:
        """Build a model_call(x_t, timestep) closure that runs the EMA shadow.

        When ``probe`` is given, the controlnet and trunk forwards inside each
        sampler step are wrapped in CUDA-synchronised timing sections so the
        per-component NFE timing can be reported.

        Variant A (``self.controlnet is None``) has no ControlNet EMA; the
        closure passes ``controlnet=None`` to :meth:`_trunk_forward` so the
        sampler runs the trunk alone. ``self._val_cond`` is also None on
        that path and ignored by ``_trunk_forward``.
        """
        if self.controlnet is not None:
            ema_cn: torch.nn.Module | None = (
                self.ema.ema_model if self.ema is not None else self.controlnet
            )
            ema_cn.eval()  # type: ignore[union-attr]
        else:
            ema_cn = None
        ema_trunk = self._ema_trunk()
        # S1 v3: rebuild the channel-concat tensor for the trunk input
        # closure-side. Validation conditioning is set by
        # ``compute_val_conditioning``; for the latents-concat path we read
        # the saved cache populated in the same call.
        latents_concat = getattr(self, "_val_latents_concat", None)

        def model_call(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            B = x_t.shape[0]
            device = x_t.device
            class_labels = self.trunk_config.make_class_labels(B, device)
            spacing = self.trunk_config.make_spacing_tensor(B, device)
            cond = self._val_cond  # set by validation_step before calling sampler
            return self._trunk_forward(
                ema_cn,
                x_t,
                timesteps,
                cond,
                class_labels,
                spacing,
                probe=probe,
                trunk=ema_trunk,
                latents_concat=latents_concat,
            )

        return model_call

    def _ema_trunk(self) -> torch.nn.Module:
        """Trunk to sample with: EMA shadow when fine-tuning, else the live trunk.

        In the frozen-trunk default this returns ``self.trunk`` unchanged, so the
        frozen sampling path is identical to before.
        """
        if self.trunk_config.trainable and self.trunk_ema is not None:
            shadow = self.trunk_ema.ema_model
            shadow.eval()
            return shadow
        return self.trunk

    def compute_val_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Build the validation conditioning tensor for ``batch`` and stash it.

        Called from ``validation_step`` and from the external exhaustive-val
        engine. The result is also written to ``self._val_cond`` so the closure
        returned by :meth:`_make_ema_call` reads the same conditioning across
        every NFE in the sweep.

        S1 v3 also caches the channel-concat latents tensor in
        ``self._val_latents_concat`` so the EMA-call closure can reproduce the
        same trunk input at every NFE.

        Variant A (``self.conditioning is None``) returns ``None`` for the
        ControlNet conditioning and only populates the latents-concat cache.
        """
        if self.conditioning is not None:
            self._val_cond = self.conditioning(batch)
        else:
            self._val_cond = None  # type: ignore[assignment]
        self._val_latents_concat = self._build_trunk_input_latents_concat(batch)
        return self._val_cond

    def _which_nfes(self, epoch: int) -> list[int]:
        vcfg = self.validation_cfg
        do_sweep = int(vcfg.get("full_sweep_every_epochs", 5)) > 0 and (
            epoch % int(vcfg.get("full_sweep_every_epochs", 5)) == 0
        )
        if do_sweep:
            return [int(n) for n in vcfg.get("sweep_nfes", [1, 2, 5, 10, 50])]
        return [int(vcfg.get("per_epoch_nfe", 5))]

    def _do_image_metrics(self, epoch: int) -> bool:
        """Whether to decode to image space and compute PSNR/SSIM this epoch.

        Image-space metrics are expensive (one VAE decode per patient, ~2.5 s)
        and the small-region SSIM is noisy, so they run on a slow cadence
        (``validation.image_metrics_every_epochs``) rather than every epoch, and
        only at the canonical ``per_epoch_nfe``. ``0`` disables them entirely.
        """
        if not self.validation_cfg.get("image_metrics", True):
            return False
        if self.vae_decoder is None or self.image_metrics is None:
            return False
        every = int(self.validation_cfg.get("image_metrics_every_epochs", 0))
        return every > 0 and (epoch % every == 0)

    @torch.inference_mode()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.region_resolver is None:
            return  # validation disabled
        masks: RegionMasks = self.region_resolver.resolve(batch)
        z_target = batch["z_t1c"]
        patient_ids = batch.get("patient_id")
        if isinstance(patient_ids, str):
            patient_ids = [patient_ids]

        epoch = int(self.current_epoch)
        nfes = self._which_nfes(epoch)
        per_epoch_nfe = int(self.validation_cfg.get("per_epoch_nfe", 5))
        qual_every = int(self.validation_cfg.get("qualitative_every_epochs", 10))
        do_qual = qual_every > 0 and (epoch % qual_every == 0)
        do_image_epoch = self._do_image_metrics(epoch)

        sampler = get_sampler(self.validation_cfg.get("integrator", "euler"))(
            scheduler=self.rflow.scheduler
        )
        # Materialise the conditioning once per batch; the EMA closure
        # constructed by ``_make_ema_call`` reads ``self._val_cond`` so the
        # same conditioning is reused across NFE values.
        self._val_cond = self.compute_val_conditioning(batch)
        B = int(z_target.shape[0])

        for nfe in nfes:
            probe = NFETimingProbe()
            model_call = self._make_ema_call(probe=probe)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                x0 = torch.randn_like(z_target)
                t_start = time.perf_counter()
                z_pred = sampler.sample(model_call, x0, num_inference_steps=int(nfe))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sample = time.perf_counter() - t_start
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at NFE=%d epoch=%d — skipping this NFE.", nfe, epoch)
                torch.cuda.empty_cache()
                continue

            decode_sec = self._update_val_accumulator(
                masks=masks,
                z_pred=z_pred,
                z_target=z_target,
                nfe=nfe,
                do_image=(do_image_epoch and int(nfe) == per_epoch_nfe),
            )

            # Per-component timings: drop the first sampler step (CUDA warm-up).
            comp = probe.aggregate(drop_first=int(nfe) > 1)
            self._accumulate_nfe_timing(
                nfe=int(nfe),
                t_total_per_patient=t_sample / max(1, B),
                t_trunk=comp.get("trunk", {}).get("mean", float("nan")),
                t_controlnet=comp.get("controlnet", {}).get("mean", float("nan")),
                t_decode_per_patient=(decode_sec / max(1, B)) if decode_sec is not None else None,
                gpu_mem_peak_mb=(
                    float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                    if torch.cuda.is_available()
                    else 0.0
                ),
                n_patients=B,
            )

            if do_qual and patient_ids is not None:
                for b, pid in enumerate(patient_ids):
                    self._qualitative_buffer[(str(pid), int(nfe))] = z_pred[b].detach().cpu().half()

    def _update_val_accumulator(
        self,
        masks: RegionMasks,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        nfe: int,
        do_image: bool,
    ) -> float | None:
        """Update region metrics for one (nfe, batch).

        Returns
        -------
        float | None
            Wall-clock seconds to decode the predicted volume(s) through the
            VAE (one decode for the whole batch), or ``None`` when image
            metrics are disabled / no decoder is available. Decoding happens
            once here and the resulting images are reused across regions.
        """
        B = z_pred.shape[0]
        decode_sec: float | None = None
        img_pred: torch.Tensor | None = None
        img_target: torch.Tensor | None = None
        if do_image and self.vae_decoder is not None and self.image_metrics is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            img_pred, img_target = self._decode_pair(z_pred, z_target)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_sec = time.perf_counter() - t0

        for region in REGION_NAMES:
            mask = masks.get(REGION_TO_RESOLVER_KEY[region])
            key = (int(nfe), region)
            agg = self._val_accumulator.setdefault(key, _new_agg())
            if mask is None:
                agg["n_patients"] = 0
                continue
            mse = self.latent_metrics.mse(z_pred, z_target, mask)
            l1 = self.latent_metrics.l1(z_pred, z_target, mask)
            cos = self.latent_metrics.cosine(z_pred, z_target, mask)
            agg["mse"].extend(mse.detach().cpu().tolist())
            agg["l1"].extend(l1.detach().cpu().tolist())
            agg["cosine"].extend(cos.detach().cpu().tolist())
            agg["n_patients"] = len(agg["mse"])

            # Log per-batch — Lightning aggregates across the validation set
            # into ``trainer.callback_metrics`` which the ModelCheckpoint
            # reads when picking the best epoch.
            self.log(
                f"val/mse_latent_{region}_nfe{nfe}",
                mse.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                f"val/l1_latent_{region}_nfe{nfe}",
                l1.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                f"val/cosine_latent_{region}_nfe{nfe}",
                cos.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )

            if img_pred is not None and img_target is not None:
                img_mask = F.interpolate(
                    mask.float(), size=img_pred.shape[-3:], mode="nearest"
                ).bool()
                psnr = self.image_metrics.psnr(img_pred, img_target, img_mask)
                ssim = self.image_metrics.ssim(img_pred, img_target, img_mask)
                agg["psnr"].extend(_safe_tolist(psnr))
                agg["ssim"].extend(_safe_tolist(ssim))
                agg["n_image_patients"] = len(agg["psnr"])
                # Replace NaN entries (empty region) with 0 weight when logging.
                psnr_clean = psnr[torch.isfinite(psnr)] if psnr.numel() else psnr
                ssim_clean = ssim[torch.isfinite(ssim)] if ssim.numel() else ssim
                if psnr_clean.numel() > 0:
                    self.log(
                        f"val/psnr_image_{region}_nfe{nfe}",
                        psnr_clean.mean(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=B,
                    )
                if ssim_clean.numel() > 0:
                    self.log(
                        f"val/ssim_image_{region}_nfe{nfe}",
                        ssim_clean.mean(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=B,
                    )

        return decode_sec

    def _decode_pair(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Best-effort decode — exceptions bubble up (validation OOM is caught upstream).
        from vena.common.decode import decode_depth_identity

        out_pred = decode_depth_identity(self.vae_decoder, z_pred)
        out_target = decode_depth_identity(self.vae_decoder, z_target)
        return out_pred.image, out_target.image

    # ------------------------------------------------------------------
    # NFE timing accumulation (per-component, aggregated per epoch).
    # ------------------------------------------------------------------

    def _accumulate_nfe_timing(
        self,
        nfe: int,
        t_total_per_patient: float,
        t_trunk: float,
        t_controlnet: float,
        t_decode_per_patient: float | None,
        gpu_mem_peak_mb: float,
        n_patients: int,
    ) -> None:
        acc = self._nfe_timing_accum.setdefault(nfe, _new_timing_agg())
        acc["t_total"].append(float(t_total_per_patient))
        acc["t_trunk"].append(float(t_trunk))
        acc["t_controlnet"].append(float(t_controlnet))
        if t_decode_per_patient is not None:
            acc["t_decode"].append(float(t_decode_per_patient))
        acc["gpu_mem_peak_mb"] = max(acc["gpu_mem_peak_mb"], float(gpu_mem_peak_mb))
        acc["n_patients"] += int(n_patients)

    def collapse_nfe_timing(self) -> list[dict[str, Any]]:
        """Aggregate the per-epoch NFE timing accumulator to one row per nfe.

        Pure read (no mutation), mirroring :meth:`collapse_val_metrics`. The
        ``NFETimingCSV`` callback consumes this on ``on_validation_epoch_end``;
        the accumulator itself is cleared in this module's later-firing
        ``on_validation_epoch_end``.

        Returns
        -------
        list[dict[str, Any]]
            One dict per nfe with the columns the CSV expects.
        """
        rows: list[dict[str, Any]] = []
        for nfe in sorted(self._nfe_timing_accum):
            acc = self._nfe_timing_accum[nfe]
            rows.append(
                {
                    "nfe": int(nfe),
                    "t_trunk_mean_sec": _finite_mean(acc["t_trunk"]),
                    "t_controlnet_mean_sec": _finite_mean(acc["t_controlnet"]),
                    "t_decode_sec": _finite_mean(acc["t_decode"]),
                    "t_total_mean_sec": _finite_mean(acc["t_total"]),
                    "t_total_std_sec": _finite_std(acc["t_total"]),
                    "gpu_mem_peak_mb": acc["gpu_mem_peak_mb"],
                    "n_patients_measured": int(acc["n_patients"]),
                }
            )
        return rows

    def collapse_val_metrics(self) -> dict[tuple[int, str], dict[str, Any]]:
        """Collapse the raw per-region accumulator to mean/std stats.

        Pure read: does not mutate ``self._val_accumulator``. The
        ``ValMetricsCSV`` callback calls this on ``on_validation_epoch_end``.
        It must be a separate method (not done in-place in this module's own
        ``on_validation_epoch_end``) because Lightning fires
        ``Callback.on_validation_epoch_end`` *before*
        ``LightningModule.on_validation_epoch_end`` — so an in-place collapse
        here would run too late and the callback would read raw lists.

        Returns
        -------
        dict[tuple[int, str], dict[str, Any]]
            Mapping ``(nfe, region)`` to the collapsed stat dict produced by
            :func:`_agg_to_stats` (``*_mean`` / ``*_std`` / ``n_patients``).
        """
        return {
            (nfe, region): _agg_to_stats(agg)
            for (nfe, region), agg in self._val_accumulator.items()
        }

    def on_validation_epoch_end(self) -> None:
        # Fires *after* every callback's ``on_validation_epoch_end`` (Lightning
        # calls callback hooks before the module hook). By this point the
        # ValMetricsCSV callback has already consumed the accumulator via
        # ``collapse_val_metrics``; clear it so the next epoch starts fresh.
        # Per-batch ``self.log`` calls in ``_update_val_accumulator`` already
        # populated ``trainer.callback_metrics`` for ModelCheckpoint.
        self._val_accumulator.clear()
        self._nfe_timing_accum.clear()

    # ------------------------------------------------------------------
    # Checkpoint pathway: RNG state + best metric.
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        # When the trunk is trainable it is a registered submodule
        # (``_trunk_module``) and its EMA (``trunk_ema``) is registered too, so
        # both are already in ``checkpoint["state_dict"]`` and restored natively
        # on resume — no custom payload needed.

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        rng = checkpoint.get("rng_state")
        if not rng:
            return
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            logger.info("RNG state restored from checkpoint.")
        except Exception as exc:
            logger.warning("RNG restore failed: %s", exc)

    # ------------------------------------------------------------------
    # Warm-start (weights-only) load.
    # ------------------------------------------------------------------

    def load_warm_start(self, ckpt_path: str | Path) -> dict[str, int]:
        """Load module weights from a Lightning checkpoint, weights-only.

        The engine calls this in ``ResumeMode.WARM_START`` *before*
        ``trainer.fit``. Optimiser state, scheduler state, EMA state, RNG
        state are intentionally *not* restored — the new run starts a fresh
        training schedule on top of the loaded weights. This is the s1→s2
        warm-start path and the way to bring in an external checkpoint.

        Behaviour:

        * Loads ``checkpoint["state_dict"]`` with ``strict=False`` so a
          partial overlap is fine — e.g. an s1 ControlNet warm-starting an
          s2 module that adds extra heads, or an FFT trunk warm-starting a
          LoRA recipe where the LoRA adapter params get a fresh init.
        * Logs an INFO-level summary ``loaded=N missing=M unexpected=K`` so
          the audit trail records exactly what was transferred (per
          ``.claude/rules/coding-standards.md`` rule #15: no silent swallow).
        * Returns the same counts as a dict so callers can persist them in
          ``decision.json`` if they want.

        Parameters
        ----------
        ckpt_path : str | Path
            Absolute path to a Lightning ``.ckpt`` file.

        Returns
        -------
        dict[str, int]
            ``{"loaded": ..., "missing": ..., "unexpected": ...}``.

        Raises
        ------
        FileNotFoundError
            If ``ckpt_path`` does not exist.
        KeyError
            If the file does not carry a ``state_dict`` key (not a Lightning
            checkpoint).
        """
        p = Path(ckpt_path)
        if not p.is_file():
            raise FileNotFoundError(f"warm-start checkpoint not found: {p}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt:
            raise KeyError(f"warm-start: {p} is not a Lightning checkpoint (no 'state_dict' key).")
        src_state = ckpt["state_dict"]
        own_state = self.state_dict()
        # Filter to shape-compatible overlapping keys; everything else is
        # surfaced in the missing/unexpected log line.
        loadable = {
            k: v for k, v in src_state.items() if k in own_state and own_state[k].shape == v.shape
        }
        result = self.load_state_dict(loadable, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys) + [k for k in src_state if k not in loadable]
        logger.info(
            "load_warm_start: src=%s loaded=%d missing=%d unexpected=%d",
            p,
            len(loadable),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.info("  first missing (cn-only / new heads): %s", missing[:5])
        if unexpected:
            logger.info("  first unexpected (dropped from src): %s", unexpected[:5])
        return {"loaded": len(loadable), "missing": len(missing), "unexpected": len(unexpected)}

    # ------------------------------------------------------------------
    # Optimiser.
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        lr = float(self.optim_cfg.get("lr", 1e-4))
        betas = tuple(self.optim_cfg.get("betas", (0.9, 0.95)))
        weight_decay = float(self.optim_cfg.get("weight_decay", 1e-2))
        warmup_steps = int(self.optim_cfg.get("warmup_steps", 100))
        max_steps = int(self.optim_cfg.get("max_steps", 50_000))
        scheduler_kind = str(self.optim_cfg.get("scheduler", "cosine")).lower()

        # Param groups (rule: LoRA adapter weights get weight_decay=0.0 — the
        # standard HuggingFace PEFT recipe; otherwise the adapters decay too
        # fast and cancel the gains from joint trunk fine-tuning. ControlNet
        # + non-LoRA trainable params keep the standard weight_decay.).
        # Variant A (controlnet disabled): no CN params; the trunk
        # parameters carry the optimisation budget alone.
        cn_params: list[torch.nn.Parameter] = (
            [p for p in self.controlnet.parameters() if p.requires_grad]
            if self.controlnet is not None
            else []
        )
        trunk_params: list[torch.nn.Parameter] = []
        if self.trunk_config.trainable:
            trunk_params = [p for p in self.trunk.parameters() if p.requires_grad]

        is_peft = self.trunk_config.regime == "peft" and bool(trunk_params)
        if is_peft:
            param_groups = [
                {"params": cn_params, "weight_decay": weight_decay},
                {"params": trunk_params, "weight_decay": 0.0},
            ]
            logger.info(
                "configure_optimizers: PEFT param groups — ControlNet=%d (wd=%g) + LoRA adapters=%d (wd=0.0)",
                len(cn_params),
                weight_decay,
                len(trunk_params),
            )
        else:
            param_groups = [{"params": cn_params + trunk_params, "weight_decay": weight_decay}]
            logger.info(
                "configure_optimizers: %s param group — ControlNet=%d + trunk=%d (wd=%g)",
                "FFT" if trunk_params else "frozen-trunk",
                len(cn_params),
                len(trunk_params),
                weight_decay,
            )
        opt = AdamW(param_groups, lr=lr, betas=betas)

        def lr_lambda(step: int) -> float:
            return _lr_lambda(scheduler_kind, step, warmup_steps, max_steps)

        sched = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ----------------------------------------------------------------------
# Aggregator helpers — kept module-level to be picklable for DataLoader workers.
# ----------------------------------------------------------------------


def _lr_lambda(
    scheduler: str,
    step: int,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Linear-warmup → {cosine, polynomial, constant} decay LR multiplier.

    Pure function so the unit tests cover every branch without spinning up
    Lightning. The old in-method ``lr_lambda`` returned ``1.0`` for any
    unknown ``scheduler`` string, which is the bug that hid the polynomial
    misconfiguration in the 2026-06-07 runs (``total_steps=1e9`` made one
    decay step span ~10 000 actual steps). Unknown values now raise.

    Parameters
    ----------
    scheduler : str
        One of ``"constant"``, ``"polynomial"``, ``"cosine"``.
    step : int
        Current optimiser step (``0``-indexed).
    warmup_steps : int
        Steps over which lr ramps linearly from 0 to peak.
    max_steps : int
        Total optimiser steps; cosine and polynomial decay reach 0 at
        ``step == max_steps``.

    Returns
    -------
    float
        Multiplier in ``[0, 1]``; multiply by the optimiser's base lr to get
        the live lr.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler == "polynomial":
        return max(0.0, 1.0 - progress)
    if scheduler == "constant":
        return 1.0
    raise ValueError(
        f"unknown LR scheduler '{scheduler}'; choose from cosine | polynomial | constant"
    )


def _bg_from_wt(m_wt: torch.Tensor) -> torch.Tensor:
    """Dilated-complement background mask in latent space.

    ``m_bg = 1 - dilate3(m_wt)`` where ``dilate3`` is a 3×3×3 max-pool — the
    same primitive used by ``RegionMasks._dilate_wt`` for the validation-time WT
    dilation. The complement spans every voxel that is *not* within one latent
    voxel of the tumour, matching proposal §5.3 step 1.

    Parameters
    ----------
    m_wt : Tensor
        Binary whole-tumour mask in latent space, shape ``(B, 1, h, w, d)``.

    Returns
    -------
    Tensor
        Background mask of the same shape, valued in ``{0, 1}``.
    """
    m = m_wt.to(dtype=torch.float32)
    dilated = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return (1.0 - dilated).clamp_(0.0, 1.0)


def _new_agg() -> dict[str, Any]:
    return {
        "mse": [],
        "l1": [],
        "cosine": [],
        "psnr": [],
        "ssim": [],
        "n_patients": 0,
        "n_image_patients": 0,
    }


def _new_timing_agg() -> dict[str, Any]:
    return {
        "t_total": [],
        "t_trunk": [],
        "t_controlnet": [],
        "t_decode": [],
        "gpu_mem_peak_mb": 0.0,
        "n_patients": 0,
    }


def _finite_mean(xs: list[float]) -> float | None:
    """Mean over the finite (non-NaN) entries; ``None`` if no finite samples."""
    finite = [x for x in xs if not math.isnan(x)]
    return sum(finite) / len(finite) if finite else None


def _finite_std(xs: list[float]) -> float | None:
    """Sample stddev (Bessel) over the finite entries.

    Returns ``0.0`` for a single finite value and ``None`` when no finite
    samples exist. Matches the legacy nested ``_std`` semantics.
    """
    finite = [x for x in xs if not math.isnan(x)]
    if len(finite) < 2:
        return 0.0 if finite else None
    m = sum(finite) / len(finite)
    return math.sqrt(sum((x - m) ** 2 for x in finite) / (len(finite) - 1))


def _agg_to_stats(agg: dict[str, Any]) -> dict[str, Any]:
    def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(v)

    mse_m, mse_s = mean_std(agg["mse"])
    l1_m, l1_s = mean_std(agg["l1"])
    cos_m, _ = mean_std(agg["cosine"])
    psnr_m, psnr_s = mean_std([x for x in agg["psnr"] if x is not None and not math.isnan(x)])
    ssim_m, ssim_s = mean_std([x for x in agg["ssim"] if x is not None and not math.isnan(x)])
    return {
        "mse_latent_mean": mse_m,
        "mse_latent_std": mse_s,
        "l1_latent_mean": l1_m,
        "l1_latent_std": l1_s,
        "cosine_latent_mean": cos_m,
        "psnr_image_mean": psnr_m,
        "psnr_image_std": psnr_s,
        "ssim_image_mean": ssim_m,
        "ssim_image_std": ssim_s,
        "n_patients": int(agg["n_patients"]),
        "n_image_patients": int(agg.get("n_image_patients", 0)),
    }


def _safe_tolist(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().tolist()]
