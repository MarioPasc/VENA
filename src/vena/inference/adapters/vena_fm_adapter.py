"""A1-VENA-S1 + VENA-S2 adapter (latent FM + ControlNet + EMA sampling).

Reconstructs the training-time ``FMLightningModule`` from
``<run_dir>/config.yaml`` and restores the Lightning checkpoint state
(``checkpoints/ema_best.ckpt`` by default). The state_dict contains the
``_trunk_module`` (fine-tuned trunk when ``trunk.trainable=true``),
``controlnet`` (live ControlNet), ``ema`` (the EMA ControlNet shadow)
and ``trunk_ema`` (the EMA trunk shadow when applicable) — so a single
``load_state_dict`` restores everything needed for EMA-based sampling.

Inference per patient × per NFE:

1. Read the conditioning latents (``z_t1pre, z_t2, z_flair``), tumour-
   derived WT mask, and optionally the brain mask, from the cohort's
   ``latent_h5`` via :class:`LatentH5Dataset` — the same path
   exhaustive validation uses.
2. Stash the conditioning via ``module.compute_val_conditioning(batch)``
   (public method on the module).
3. Sample with the :class:`EulerSampler` over ``module.rflow.scheduler``
   for ``nfe`` steps using ``module._make_ema_call()``.
4. Decode the predicted latent through
   :func:`vena.common.decode.decode_box` for intensity-space parity
   with the §4.1 contract.
5. Apply :func:`apply_harmonisation` on top of the decoded volume so
   the H5 row matches the harmonisation recipe verbatim.

Every VENA row uses this single class — rows differ only in ``run_dir``
(which stage/recipe was trained) and ``name`` (the YAML registry tag).
The sampling architecture is byte-identical across stages because the
only S1 ↔ S2 ↔ S3 delta is the *loss* configuration, which is
irrelevant at inference time.

That last point has a concrete consequence for state restoration. An S3
(decoder-perceptual-loss) run checkpoints two training-only submodules —
``lpl_loss`` and ``feature_stats`` (the LPL feature-statistics EMA of
Berrada 2025). :meth:`_build_module` rebuilds the module for *sampling*
only and therefore passes no ``lpl_config``, so neither submodule exists
on the inference module and their keys arrive as ``unexpected``. They are
dropped in :meth:`_load_state_dict` (see ``_TRAIN_ONLY_PREFIXES``): they
parameterise the loss, never the velocity field. Any *other* unexpected
key still raises — that guard is what catches a genuinely mismatched
checkpoint.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from vena.common import MaisiDecoder, invert_crop_pad, load_autoencoder
from vena.common.decode import decode_box
from vena.inference.base import (
    InferenceModel,
    InferenceModelError,
    InferenceResult,
    resolve_device,
    resolve_path,
)
from vena.inference.harmonisation import apply_harmonisation
from vena.inference.registry import register_inference_model
from vena.model.fm.eval.exhaustive import build_crop_spec_from_h5

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry

logger = logging.getLogger(__name__)


class VenaFMAdapterError(InferenceModelError):
    """Raised on VENA-FM checkpoint or config-recovery failures."""


#: State-dict prefixes that exist only while training and have no counterpart on
#: the sampling module. ``lpl_loss`` / ``feature_stats`` are the S3 decoder-
#: perceptual-loss submodules (Berrada 2025); they shape the *loss*, never the
#: velocity field, so dropping them is a no-op for sampling. Keys outside this
#: set are still treated as a hard error.
_TRAIN_ONLY_PREFIXES: tuple[str, ...] = ("lpl_loss.", "feature_stats.")

#: Fallback for `rflow.base_img_size_numel` — VENA's brain-box latent (48x56x48).
#: Only used if a run's config omits the key while enabling the timestep transform.
_BRAIN_BOX_LATENT_NUMEL: int = 48 * 56 * 48


def resolve_timestep_transform_kwargs(rflow_cfg: dict[str, Any] | None) -> dict[str, int]:
    """Sampler kwargs implied by a run's ``rflow`` config.

    When a run trained with MONAI's SD3-style timestep transform
    (``use_timestep_transform: true`` — true for every S1 v3 run), the sampler's
    ``set_timesteps`` requires ``input_img_size_numel``. Omit it and MONAI's
    ``timestep_transform`` evaluates ``None / int``, which fails every patient
    behind a WARNING instead of an exception — the failure mode is a silently
    empty output tree, not a crash.

    Parameters
    ----------
    rflow_cfg
        The run's ``rflow`` block from ``config.yaml`` (``None`` if absent).

    Returns
    -------
    dict[str, int]
        ``{"input_img_size_numel": N}`` when the transform is on, else ``{}``.
        The key must be omitted entirely when the transform is off — a scheduler
        built without it does not accept the kwarg.
    """
    cfg = rflow_cfg or {}
    if not cfg.get("use_timestep_transform"):
        return {}
    numel = cfg.get("base_img_size_numel") or _BRAIN_BOX_LATENT_NUMEL
    return {"input_img_size_numel": int(numel)}


@register_inference_model("vena_fm")
class VenaFMAdapter(InferenceModel):
    """Latent FM + ControlNet adapter shared by A1-VENA-S1 and VENA-S2."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        checkpoint: str = "ema_best.ckpt",
        vae_checkpoint: str | Path,
        device: str | torch.device = "cuda:0",
        nfe_list: tuple[int, ...] = (1, 2, 5, 10, 20),
        selection_nfe: int = 5,
        config_filename: str = "config.yaml",
        integrator: str = "euler",
    ) -> None:
        super().__init__(
            name=name,
            device=resolve_device(device),
            nfe_list=nfe_list,
            selection_nfe=selection_nfe,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.checkpoint_path = self.run_dir / "checkpoints" / checkpoint
        self.vae_checkpoint = resolve_path(vae_checkpoint)
        if self.vae_checkpoint is None:
            raise VenaFMAdapterError(f"{name}: vae_checkpoint is required")
        self.config_path = self.run_dir / config_filename
        self.integrator = integrator

        self._module: Any = None
        self._sampler: Any = None
        self._vae: MaisiDecoder | None = None
        self._train_cfg: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        if self._is_setup:
            return
        if not self.checkpoint_path.is_file():
            raise VenaFMAdapterError(f"{self.name}: checkpoint missing at {self.checkpoint_path}")
        if not self.config_path.is_file():
            raise VenaFMAdapterError(f"{self.name}: config.yaml missing at {self.config_path}")

        with self.config_path.open("r") as f:
            self._train_cfg = yaml.safe_load(f)

        module = self._build_module()
        self._load_state_dict(module)
        module.eval()
        self._module = module

        # Sampler over the rebuilt RFlow scheduler. The timestep-transform kwarg
        # is load-bearing — see resolve_timestep_transform_kwargs.
        from vena.model.fm.inference import get_sampler

        ts_kwargs = resolve_timestep_transform_kwargs(self._train_cfg.get("rflow"))
        if ts_kwargs:
            logger.info(
                "%s: timestep transform on — input_img_size_numel=%d",
                self.name,
                ts_kwargs["input_img_size_numel"],
            )
        self._sampler = get_sampler(self.integrator)(
            scheduler=module.rflow.scheduler, **ts_kwargs
        )

        # VAE for image-space decode.
        self._vae = MaisiDecoder(
            handle=load_autoencoder(self.vae_checkpoint, device=str(self.device))
        )

        super().setup()

    def _build_module(self) -> Any:
        """Rebuild the training-time ``FMLightningModule`` from ``config.yaml``."""
        from vena.model.fm.lightning import FMLightningModule
        from vena.model.fm.maisi.config import TrunkConfig

        cfg = self._train_cfg
        trunk_yaml = cfg["model"]["trunk"]
        controlnet_yaml = cfg["model"]["controlnet"]
        rflow_yaml = cfg.get("rflow", {})
        ema_yaml = cfg.get("ema", {})

        trunk_cfg = TrunkConfig(
            checkpoint=Path(trunk_yaml["checkpoint"]),
            arch_json=Path(trunk_yaml["arch_json"]) if trunk_yaml.get("arch_json") else None,
            arch_overrides=trunk_yaml.get("arch_overrides", {}),
            class_token=trunk_yaml.get("class_token", 9),
            spacing_mm=tuple(trunk_yaml.get("spacing_mm", (1.0, 1.0, 1.0))),
            trainable=bool(trunk_yaml.get("trainable", True)),
            regime=trunk_yaml.get("regime", "fft"),
            peft=trunk_yaml.get("peft"),
        )

        stage = cfg["run"]["stage"]
        stage = stage.upper() if str(stage).startswith("s") else stage

        # The three S1-v3 (2026-06-22) knobs. Forwarding them is mandatory —
        # omitting any one silently rebuilds a DIFFERENT architecture than the
        # checkpoint was trained with:
        #
        #  * `controlnet.enabled: false` (v3a, the concat-only ablation) means
        #    Variant A: no ControlNet and no ConditioningAssembler at all. Passing
        #    the default `True` with an empty `conditioning_inputs` made the
        #    assembler raise "requires at least one spec" and the row was skipped.
        #
        #  * `trunk.input_concat` (every v3b* run) channel-concatenates the
        #    conditioning latents onto x_t, which is why those runs carry
        #    `arch_overrides.in_channels: 16` (4 noisy + 3x4 cond). Omitting it
        #    built a 16-channel conv_in and then fed it 4 channels — every patient
        #    died on "expected input to have 16 channels, but got 4".
        #
        # The module handles both paths internally: compute_val_conditioning()
        # caches the concat tensor and _make_ema_call() reads it, so nothing else
        # in the sampling path needs to change.
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(controlnet_yaml.get("conditioning_inputs") or []),
            stage=stage,
            controlnet_arch_overrides=controlnet_yaml.get("arch_overrides", {}),
            rflow_cfg=dict(rflow_yaml) if isinstance(rflow_yaml, dict) else {},
            ema_cfg=dict(ema_yaml) if isinstance(ema_yaml, dict) else {},
            region_resolver=None,
            vae_decoder=None,
            controlnet_enabled=bool(controlnet_yaml.get("enabled", True)),
            controlnet_init_from_trunk_enabled=bool(controlnet_yaml.get("init_from_trunk", True)),
            input_concat_cfg=trunk_yaml.get("input_concat"),
        )
        module = module.to(self.device)
        module.setup()
        return module

    def _load_state_dict(self, module: Any) -> None:
        """Load the Lightning checkpoint's ``state_dict`` into ``module``.

        The Lightning ckpt and the freshly-built module must align on every
        parameter that participates in sampling — trunk-EMA + controlnet-EMA +
        live trunk + live controlnet. Training-only submodules
        (``_TRAIN_ONLY_PREFIXES``) are stripped first: an S3 run checkpoints its
        LPL loss and feature-statistics EMA, which the sampling module never
        builds. Any unexpected key that survives the strip is a genuine
        checkpoint/architecture mismatch and raises.
        """
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        dropped = [k for k in state if k.startswith(_TRAIN_ONLY_PREFIXES)]
        if dropped:
            state = {k: v for k, v in state.items() if not k.startswith(_TRAIN_ONLY_PREFIXES)}
            logger.info(
                "%s: dropped %d training-only key(s) absent from the sampling module "
                "(LPL loss / feature-stats EMA); first=%s",
                self.name,
                len(dropped),
                dropped[0],
            )

        missing, unexpected = module.load_state_dict(state, strict=False)
        if unexpected:
            raise VenaFMAdapterError(
                f"{self.name}: unexpected keys in checkpoint {self.checkpoint_path}: "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        # The trunk_ema buffer may be absent for FFT runs that did not enable
        # the unfrozen-trunk EMA; the module's _make_ema_call falls back to
        # the live trunk in that case. We log but do not fail on `missing`.
        if missing:
            logger.info(
                "%s: load_state_dict missing=%d (first=%s)",
                self.name,
                len(missing),
                missing[0] if missing else "",
            )

    # ------------------------------------------------------------------ predict

    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        self._require_setup()
        assert self._module is not None and self._sampler is not None and self._vae is not None

        self._reset_peak_vram(self.device)
        self._sync(self.device)
        t0 = time.perf_counter()

        batch = self._build_patient_batch(cohort.latent_h5, patient_id)
        crop_spec = build_crop_spec_from_h5(cohort.image_h5, patient_id)

        self._module.compute_val_conditioning(batch)
        z_target = batch["z_t1c"]
        z_pred = self._sample(z_target, int(nfe))
        pred_box = decode_box(self._vae, z_pred, crop_spec)  # (Hbox, Wbox, Dbox) in [0,1]
        # Map box → native so the prediction aligns with masks/brain.
        pred_native = invert_crop_pad(pred_box[None, None], crop_spec)[0, 0]

        from vena.inference.image_dataset import load_image_modalities

        mods = load_image_modalities(cohort.image_h5, patient_id, ())
        brain_native = torch.from_numpy(mods["brain"]).to(torch.float32)

        harmonised = apply_harmonisation(pred_native.cpu(), brain_mask=brain_native)
        raw = pred_native.detach().cpu().contiguous()

        self._sync(self.device)
        seconds = time.perf_counter() - t0
        return InferenceResult(
            t1c_synthetic_harmonised=harmonised,
            t1c_synthetic_raw=raw,
            inference_seconds=float(seconds),
            peak_vram_mb=self._peak_vram_mb(self.device),
        )

    # ------------------------------------------------------------------ helpers

    def _build_patient_batch(self, latent_h5: Path, patient_id: str) -> dict[str, Any]:
        """Load one patient's latent batch using ``LatentH5Dataset``."""
        from vena.model.fm.lightning import LatentH5Dataset

        dataset = LatentH5Dataset(Path(latent_h5), [patient_id])
        item = dataset[0]
        return {
            k: (v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in item.items()
        }

    def _sample(self, z_target: torch.Tensor, nfe: int) -> torch.Tensor:
        model_call = self._module._make_ema_call()
        x0 = torch.randn_like(z_target)
        with torch.inference_mode():
            return self._sampler.sample(model_call, x0, num_inference_steps=int(nfe))

    @staticmethod
    def _brain_mask_in_box(brain_native: torch.Tensor, crop_spec: Any) -> torch.Tensor:
        """Crop/pad the native brain mask to the decoded box shape."""
        from vena.common import apply_crop_pad

        box = apply_crop_pad(brain_native[None, None].float(), crop_spec)[0, 0]
        # decode_box's brain mask matches the *spatial* shape of the decoded
        # volume; cast to float so it can multiply the harmonised volume in
        # apply_harmonisation.
        return (box > 0.5).to(torch.float32)

    # ------------------------------------------------------------------ teardown

    def teardown(self) -> None:
        if self._module is not None:
            del self._module
            self._module = None
        if self._sampler is not None:
            del self._sampler
            self._sampler = None
        if self._vae is not None:
            del self._vae
            self._vae = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False
