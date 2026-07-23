"""SegResNet segmentation backbone (Arm C — no SSL pretraining, baseline).

Architecture spec (fixed, matches MONAI SegResNet conventions):
    init_filters  = 16
    blocks_down   = (1, 2, 2, 4)   → 4 encoder levels
    blocks_up     = (1, 1, 1)      → 3 decoder levels
    dropout_prob  = 0.2
    in_channels   = cfg.in_channels  (default 3)
    out_channels  = cfg.out_channels (default 2 = [TC, NETC])

Deep supervision (hook-based):
    ``register_forward_hook`` on ``backbone.up_layers[0]`` and
    ``backbone.up_layers[1]``.  Hooks capture intermediate outputs into a
    list cleared at the start of each forward call.

Deep-supervision channel derivation:
    Bottleneck filters = init_filters × 2^(n_down-1) = 16 × 8 = 128
    After up_samples[0] (128 → 64): up_layers[0] outputs 64ch at H/4 spatial
    After up_samples[1] (64  → 32): up_layers[1] outputs 32ch at H/2 spatial

Aux head channels (matching the above):
    _aux_head0 = Conv3d(64, out_channels, kernel_size=1)  applied to H/4 feat
    _aux_head1 = Conv3d(32, out_channels, kernel_size=1)  applied to H/2 feat

Return convention (``deep_supervision=True``):
    ``(logits, aux_H2, aux_H4)``
        [0] logits  : (B, out_channels, H, W, D)    full resolution
        [1] aux_H2  : (B, out_channels, H/2, W/2, D/2)
        [2] aux_H4  : (B, out_channels, H/4, W/4, D/4)

Spatial divisibility:
    SegResNet (3 stride-2 downsamples) requires H, W, D divisible by 2^3 = 8.
    Inputs of shape (B, 3, 32, 32, 24) are valid; (B, 3, 32, 32, 32) also.
    There is NO 32-divisibility constraint (unlike SwinUNETR).

No pretrained checkpoint is used — Arm C is a scratch baseline.
``cfg.checkpoint`` is ignored with a DEBUG-level notice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from vena.segmentation.config import ModelConfig

from vena.segmentation.models.registry import register_segmentation_model

logger = logging.getLogger(__name__)

# Fixed architecture constants (must stay in sync with builder kwargs).
_INIT_FILTERS: int = 16
_BLOCKS_DOWN: tuple[int, ...] = (1, 2, 2, 4)
_BLOCKS_UP: tuple[int, ...] = (1, 1, 1)
_N_DOWN: int = len(_BLOCKS_DOWN)

# Analytically derived channel counts for the two aux heads.
# up_layers[0]: init_filters × 2^(n_down-2) = 64ch at H/4 spatial resolution.
# up_layers[1]: init_filters × 2^(n_down-3) = 32ch at H/2 spatial resolution.
_AUX_CH0: int = _INIT_FILTERS * (2 ** (_N_DOWN - 2))  # 64
_AUX_CH1: int = _INIT_FILTERS * (2 ** (_N_DOWN - 3))  # 32


class _VenaSegResNet(nn.Module):
    """MONAI SegResNet with optional hook-based deep supervision.

    Hooks are registered on ``backbone.up_layers[0]`` and
    ``backbone.up_layers[1]`` in ``__init__`` and removed in
    :meth:`remove_hooks` (called automatically in ``__del__``).  Always call
    ``remove_hooks()`` before discarding an instance to avoid dangling hooks.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        from monai.networks.nets import SegResNet

        super().__init__()
        self._backbone = SegResNet(
            spatial_dims=3,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            init_filters=_INIT_FILTERS,
            blocks_down=_BLOCKS_DOWN,
            blocks_up=_BLOCKS_UP,
            dropout_prob=0.2,
        )
        self._ds = cfg.deep_supervision

        # Storage for hook-captured intermediate decoder outputs.
        # Cleared at the start of every forward call.
        self._ds_outputs: list[Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        if cfg.deep_supervision:
            self._aux_head0 = nn.Conv3d(_AUX_CH0, cfg.out_channels, kernel_size=1)
            self._aux_head1 = nn.Conv3d(_AUX_CH1, cfg.out_channels, kernel_size=1)

            # Hook registration order mirrors decoder execution order:
            # up_layers[0] fires first (H/4), up_layers[1] second (H/2).
            self._handles = [
                self._backbone.up_layers[0].register_forward_hook(self._capture_hook),
                self._backbone.up_layers[1].register_forward_hook(self._capture_hook),
            ]

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _capture_hook(
        self,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Append the decoder layer output to the capture buffer."""
        self._ds_outputs.append(output)

    def remove_hooks(self) -> None:
        """Detach all registered forward hooks from the backbone.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        for h in self._handles:
            h.remove()
        self._handles = []

    def __del__(self) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Run the segmentation model.

        Parameters
        ----------
        x:
            Input tensor ``(B, in_channels, H, W, D)`` with H, W, D each
            divisible by 8.

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            ``deep_supervision=False``: single ``(B, out_channels, H, W, D)``
            ``deep_supervision=True`` : ``(logits, aux_H2, aux_H4)`` where:
                ``aux_H2`` has shape ``(B, out_channels, H/2, W/2, D/2)``
                ``aux_H4`` has shape ``(B, out_channels, H/4, W/4, D/4)``
        """
        if not self._ds:
            return self._backbone(x)

        # Clear the capture buffer before each forward to prevent accumulation.
        self._ds_outputs = []
        logits = self._backbone(x)

        if len(self._ds_outputs) != 2:
            raise RuntimeError(
                f"Expected 2 deep-supervision outputs from hooks, got "
                f"{len(self._ds_outputs)}.  Backbone up_layers structure may "
                f"have changed in this MONAI version."
            )

        # _ds_outputs[0] = up_layers[0] output: H/4 spatial, _AUX_CH0 = 64ch
        # _ds_outputs[1] = up_layers[1] output: H/2 spatial, _AUX_CH1 = 32ch
        aux_h4 = self._aux_head0(self._ds_outputs[0])
        aux_h2 = self._aux_head1(self._ds_outputs[1])

        return logits, aux_h2, aux_h4


# ---------------------------------------------------------------------------
# Builder (registered with the model registry)
# ---------------------------------------------------------------------------


@register_segmentation_model("segresnet")
def build_segresnet(cfg: ModelConfig) -> nn.Module:  # type: ignore[misc]
    """Build the SegResNet baseline backbone (Arm C, no SSL pretraining).

    This arm uses no pre-trained weights.  ``cfg.checkpoint`` is accepted but
    logged and ignored (Arm C is trained from random MONAI weight init).

    No spatial divisibility constraint beyond 2^3 = 8 (vs 32 for SwinUNETR).
    """
    if cfg.checkpoint is not None:
        logger.debug(
            "build_segresnet: cfg.checkpoint=%s is set but Arm C uses no "
            "pretrained weights — checkpoint ignored.",
            cfg.checkpoint,
        )

    return _VenaSegResNet(cfg)


__all__: list[str] = []
