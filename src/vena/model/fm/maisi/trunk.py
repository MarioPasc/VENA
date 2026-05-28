"""Load the frozen MAISI-V2 rectified-flow U-Net trunk.

The NV-Generate-MR checkpoint at the path declared in ``src/external/LINKS.md``
stores its weights under ``unet_state_dict`` and a per-volume latent
``scale_factor`` (a scalar ``MetaTensor``) alongside the optimiser/scheduler
state. This module instantiates ``DiffusionModelUNetMaisi`` with the kwargs in
``configs/diff_unet_rflow.json``, loads weights, sets train/eval mode and
``requires_grad`` according to ``trainable``, and returns a
provenance-carrying :class:`TrunkHandle`.

By default the trunk is **frozen throughout VENA training** (proposal §4.2) and
only the ControlNet branch is trained. Passing ``trainable=True`` unfreezes the
trunk for joint fine-tuning with the ControlNet (the unfrozen-trunk ablation,
cf. TumorFlow).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vena.data.h5.shared import sha256_file

from .exceptions import TrunkLoadError

logger = logging.getLogger(__name__)

_DEFAULT_ARCH_CONFIG = Path(__file__).parent / "configs" / "diff_unet_rflow.json"


@dataclass(frozen=True)
class TrunkHandle:
    """Frozen MAISI rectified-flow trunk plus provenance.

    Attributes
    ----------
    model : nn.Module
        ``DiffusionModelUNetMaisi`` instance in ``eval`` mode with all
        parameters ``requires_grad=False``.
    device : torch.device
        Where ``model`` resides.
    checkpoint_path : Path
        Resolved path of the loaded ``.pt`` file.
    checkpoint_sha256 : str
        SHA-256 of the checkpoint file.
    arch_kwargs : dict
        Exact kwargs passed to ``DiffusionModelUNetMaisi.__init__``.
    latent_scale_factor : float | None
        Scalar ``scale_factor`` recovered from the checkpoint, if present.
        Some MAISI training scripts multiply VAE latents by this factor before
        feeding them to the trunk; we expose it so downstream code can apply
        it consistently.
    num_train_timesteps : int
        ``num_train_timesteps`` stored in the checkpoint (default 1000).
    """

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    arch_kwargs: dict[str, Any]
    latent_scale_factor: float | None
    num_train_timesteps: int


def _load_arch_kwargs(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrunkLoadError(f"failed to read arch config {path!r}: {exc}") from exc
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _instantiate_trunk(arch_kwargs: dict[str, Any]) -> nn.Module:
    try:
        from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
            DiffusionModelUNetMaisi,
        )
    except ImportError as exc:
        raise TrunkLoadError(
            "monai.apps.generation.maisi is not installed; ensure monai>=1.4 "
            "with the [einops] extra is available in this env"
        ) from exc
    return DiffusionModelUNetMaisi(**arch_kwargs)


def load_trunk(
    checkpoint_path: Path | str,
    device: torch.device | str = "cuda",
    arch_config: Path | str | None = None,
    arch_overrides: dict[str, Any] | None = None,
    trainable: bool = False,
) -> TrunkHandle:
    """Instantiate and load the frozen MAISI rectified-flow trunk.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to ``diff_unet_3d_rflow-mr.pt``.
    device : torch.device | str
        Destination device.
    arch_config : Path | str | None
        Optional override for the architecture-kwargs JSON.
    arch_overrides : dict | None
        Optional per-call kwargs overrides.
    trainable : bool
        If ``True``, unfreeze the trunk (``train()`` mode, ``requires_grad=True``)
        for joint fine-tuning. If ``False`` (default), freeze it (``eval()`` mode,
        ``requires_grad=False``).

    Returns
    -------
    TrunkHandle
        Frozen ``eval``-mode trunk with provenance.

    Raises
    ------
    TrunkLoadError
        If the checkpoint cannot be found, parsed, or loaded.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise TrunkLoadError(f"checkpoint not found: {ckpt}")

    arch_path = Path(arch_config) if arch_config is not None else _DEFAULT_ARCH_CONFIG
    arch_kwargs = _load_arch_kwargs(arch_path)
    if arch_overrides:
        arch_kwargs.update(arch_overrides)
    logger.info(
        "Loading MAISI FM trunk from %s (config=%s overrides=%s)",
        ckpt,
        arch_path,
        arch_overrides or {},
    )

    model = _instantiate_trunk(arch_kwargs)
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrunkLoadError(f"torch.load failed for {ckpt}: {exc}") from exc

    if not isinstance(blob, dict) or "unet_state_dict" not in blob:
        raise TrunkLoadError(
            f"unexpected checkpoint structure: expected dict with 'unet_state_dict' key, "
            f"got {type(blob).__name__}"
        )

    state_dict = blob["unet_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("trunk state_dict missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning(
            "trunk state_dict has %d unexpected keys (first: %s)", len(unexpected), unexpected[:3]
        )

    # Recover the latent scale factor (MetaTensor scalar in MAISI checkpoints).
    sf_raw = blob.get("scale_factor", None)
    if sf_raw is None:
        scale_factor: float | None = None
    else:
        try:
            scale_factor = float(sf_raw)
        except (TypeError, ValueError):
            scale_factor = None
            logger.warning("trunk scale_factor present but non-scalar: %r", type(sf_raw).__name__)

    num_train_timesteps = int(blob.get("num_train_timesteps", 1000))

    dev = torch.device(device)
    # ``train(trainable)`` sets eval() when frozen and train() when fine-tuning.
    # The trunk is held as an unregistered property on the LightningModule, so
    # Lightning's own train()/eval() toggles do not reach it — its mode is owned
    # here and must match ``trainable``.
    model = model.to(dev).train(trainable)
    for p in model.parameters():
        p.requires_grad_(trainable)
    if trainable:
        # MONAI's MAISI U-Net injects ControlNet residuals in-place, which breaks
        # autograd once the trunk carries gradients. Rebind those two adds
        # out-of-place (numerics unchanged). See ``grad_safe`` for the rationale.
        from .grad_safe import make_trunk_grad_safe

        make_trunk_grad_safe(model)

    sha = sha256_file(ckpt)
    logger.info(
        "MAISI FM trunk loaded: sha256=%s device=%s scale_factor=%s T=%d trainable=%s",
        sha[:12],
        dev,
        scale_factor,
        num_train_timesteps,
        trainable,
    )
    return TrunkHandle(
        model=model,
        device=dev,
        checkpoint_path=ckpt,
        checkpoint_sha256=sha,
        arch_kwargs=arch_kwargs,
        latent_scale_factor=scale_factor,
        num_train_timesteps=num_train_timesteps,
    )
