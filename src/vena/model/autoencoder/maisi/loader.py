"""Load the frozen MAISI-V2 VAE-GAN checkpoint into a ready-to-use module.

The checkpoint at the canonical path declared in ``src/external/LINKS.md``
ships as a dict with three top-level keys (``epoch``, ``unet_state_dict``,
``epoch_finished``); the actual weights live under ``unet_state_dict``. The
architecture kwargs come from
``src/vena/model/autoencoder/maisi/configs/autoencoder_v2.json``, themselves
mirrored from MAISI's own ``config_network_rflow.json``.

This module never *trains* the autoencoder — it instantiates, loads weights,
``eval()``s, and freezes parameters. Downstream code consumes the returned
handle through :class:`vena.model.autoencoder.maisi.encode.engine.MaisiEncoder`
and :class:`vena.model.autoencoder.maisi.decode.engine.MaisiDecoder`.
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

from .exceptions import CheckpointLoadError

logger = logging.getLogger(__name__)

_DEFAULT_ARCH_CONFIG = Path(__file__).parent / "configs" / "autoencoder_v2.json"


@dataclass(frozen=True)
class AutoencoderHandle:
    """Frozen MAISI VAE-GAN module + provenance.

    Attributes
    ----------
    model : nn.Module
        ``AutoencoderKlMaisi`` instance with ``encode_stage_2_inputs`` and
        ``decode_stage_2_outputs`` callable; parameters are frozen and the
        module is in ``eval`` mode.
    device : torch.device
        Where ``model`` resides.
    checkpoint_path : Path
        Resolved path of the loaded ``.pt`` file.
    checkpoint_sha256 : str
        SHA-256 of the file at ``checkpoint_path``; round-tripped into every
        artifact produced by routines that consume this handle.
    arch_kwargs : dict[str, Any]
        Exact kwargs passed to ``AutoencoderKlMaisi.__init__``; serialised
        into the latent H5's ``autoencoder_arch_config_json`` root attr.
    """

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    arch_kwargs: dict[str, Any]


def _load_arch_kwargs(path: Path) -> dict[str, Any]:
    """Read the architecture-kwargs JSON, stripping comment keys."""
    try:
        with path.open("r") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointLoadError(f"failed to read arch config {path!r}: {exc}") from exc
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _instantiate_autoencoder(arch_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate ``AutoencoderKlMaisi`` directly from MONAI.

    We bypass MONAI's ``ConfigParser`` / ``define_instance`` indirection used
    by the MAISI CTMR scripts. Direct instantiation gives us a stable import
    surface and avoids dragging in MAISI's bundle machinery.
    """
    try:
        from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
            AutoencoderKlMaisi,
        )
    except ImportError as exc:
        raise CheckpointLoadError(
            "monai.apps.generation.maisi is not installed; "
            "ensure monai>=1.4 with the [einops] extra is available in this env"
        ) from exc
    return AutoencoderKlMaisi(**arch_kwargs)


def load_autoencoder(
    checkpoint_path: Path | str,
    device: torch.device | str = "cuda",
    arch_config: Path | str | None = None,
    arch_overrides: dict[str, Any] | None = None,
) -> AutoencoderHandle:
    """Instantiate and load the MAISI-V2 VAE-GAN encoder/decoder.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to ``autoencoder_v2.pt``.
    device : torch.device | str
        Destination device. ``"cuda"`` is the typical choice; ``"cpu"`` is
        supported for shape-only smoke tests.
    arch_config : Path | str | None
        Optional override for the architecture-kwargs JSON. Defaults to the
        bundled ``configs/autoencoder_v2.json``.
    arch_overrides : dict[str, Any] | None
        Optional per-call overrides applied on top of the JSON config — e.g.
        ``{"norm_float16": False}`` to force fp32 GroupNorm during inference
        without forking a separate config file. Overrides do *not* change
        the on-disk weights; they only affect the runtime construction.

    Returns
    -------
    AutoencoderHandle
        Frozen, eval-mode module plus provenance.

    Raises
    ------
    CheckpointLoadError
        If the checkpoint cannot be found, parsed, or loaded.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise CheckpointLoadError(f"checkpoint not found: {ckpt}")

    arch_path = Path(arch_config) if arch_config is not None else _DEFAULT_ARCH_CONFIG
    arch_kwargs = _load_arch_kwargs(arch_path)
    if arch_overrides:
        arch_kwargs.update(arch_overrides)
    logger.info(
        "Loading MAISI autoencoder from %s (config=%s overrides=%s)",
        ckpt,
        arch_path,
        arch_overrides or {},
    )

    model = _instantiate_autoencoder(arch_kwargs)
    try:
        # weights_only=False because the upstream checkpoint stores Python
        # objects (epoch counter); we trust the path declared in LINKS.md.
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointLoadError(f"torch.load failed for {ckpt}: {exc}") from exc

    state_dict = blob.get("unet_state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(
            f"unexpected checkpoint structure: {type(blob).__name__} (no state_dict found)"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("autoencoder state_dict missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning(
            "autoencoder state_dict has %d unexpected keys (first: %s)", len(unexpected), unexpected[:3]
        )

    dev = torch.device(device)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sha = sha256_file(ckpt)
    logger.info("MAISI autoencoder loaded: sha256=%s device=%s", sha[:12], dev)
    return AutoencoderHandle(
        model=model,
        device=dev,
        checkpoint_path=ckpt,
        checkpoint_sha256=sha,
        arch_kwargs=arch_kwargs,
    )
