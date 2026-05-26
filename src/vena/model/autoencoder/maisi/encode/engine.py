"""Image → latent encoding through the MAISI VAE-GAN.

:class:`MaisiEncoder` wraps the frozen :class:`AutoencoderKlMaisi` with three
concerns the bare module does not handle:

1. **Preprocessing** — percentile rescale into ``[0, 1]`` and depth-padding
   so the spatial axes are divisible by the compression factor.
2. **Inference strategy** — try full-volume first; on
   ``torch.cuda.OutOfMemoryError`` clear the cache and retry through MONAI's
   :class:`SlidingWindowInferer` with the MAISI-MR default ROI
   ``(80, 80, 32)`` / overlap ``0.4``. If sliding-window also OOMs, raise
   :class:`EncodeOOMError` so the routine reports a clear cause.
3. **Provenance** — record which inference path was used and the pad info,
   so the latent H5 root attrs and the per-row dataset attrs round-trip the
   information needed for a clean decode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..exceptions import EncodeOOMError, ShapeContractError
from ..loader import AutoencoderHandle
from ..preprocessing import DepthPad, pad_depth_to_multiple_of, percentile_normalise

logger = logging.getLogger(__name__)

InferenceMode = Literal["full", "sliding"]


@dataclass(frozen=True)
class EncodeResult:
    """One forward pass through the MAISI encoder.

    Attributes
    ----------
    latent : torch.Tensor
        Shape ``(B, latent_channels, H/4, W/4, D'/4)`` where ``D'`` is the
        depth-padded value.
    pad : DepthPad
        Depth-pad metadata; required by :class:`MaisiDecoder` to restore the
        original ``D``.
    inference_mode : str
        ``"full"`` or ``"sliding"``; round-tripped into H5 attrs.
    """

    latent: torch.Tensor
    pad: DepthPad
    inference_mode: InferenceMode


class MaisiEncoder:
    """Forward-only adapter around :class:`AutoencoderKlMaisi.encode_stage_2_inputs`."""

    def __init__(
        self,
        handle: AutoencoderHandle,
        sliding_roi: tuple[int, int, int] = (80, 80, 32),
        sliding_overlap: float = 0.4,
        sliding_mode: str = "gaussian",
        depth_pad_base: int = 8,
        percentile_lower: float = 0.0,
        percentile_upper: float = 99.5,
        percentile_foreground_only: bool = False,
        precision_mode: str = "autocast",
    ) -> None:
        self.handle = handle
        self.sliding_roi = tuple(sliding_roi)
        self.sliding_overlap = float(sliding_overlap)
        self.sliding_mode = sliding_mode
        self.depth_pad_base = depth_pad_base
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.percentile_foreground_only = percentile_foreground_only
        if precision_mode not in {"autocast", "fp32"}:
            raise ValueError(
                f"precision_mode must be 'autocast' or 'fp32'; got {precision_mode!r}"
            )
        self.precision_mode = precision_mode

    # ------------------------------------------------------------------ API

    @torch.inference_mode()
    def encode(
        self,
        x: torch.Tensor,
        mode: Literal["auto", "full", "sliding"] = "auto",
        normalise: bool = True,
    ) -> EncodeResult:
        """Encode an image batch to the MAISI latent.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, H, W, D)``. On CPU or GPU; will be moved to the
            handle's device.
        mode : {"auto", "full", "sliding"}
            ``"auto"`` (default) tries full-volume and falls back to sliding
            window on OOM. ``"full"`` and ``"sliding"`` force a path; no
            fallback is attempted.
        normalise : bool
            Apply :func:`percentile_normalise` first. Disable only if the
            caller already normalised (e.g. when re-encoding from a cached
            ``[0,1]`` tensor).
        """
        if x.ndim != 5 or x.shape[1] != 1:
            raise ShapeContractError(
                f"encode expects (B,1,H,W,D); got {tuple(x.shape)}"
            )
        x = x.to(self.handle.device, dtype=torch.float32, non_blocking=True)
        if normalise:
            x = percentile_normalise(
                x,
                lower=self.percentile_lower,
                upper=self.percentile_upper,
                foreground_only=self.percentile_foreground_only,
            )
        x, pad = pad_depth_to_multiple_of(x, base=self.depth_pad_base)

        if mode == "full":
            return EncodeResult(self._full(x), pad, "full")
        if mode == "sliding":
            return EncodeResult(self._sliding(x), pad, "sliding")

        # auto: try full first
        try:
            z = self._full(x)
            return EncodeResult(z, pad, "full")
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "MAISI encode OOM on full-volume input %s; retrying via sliding-window roi=%s",
                tuple(x.shape),
                self.sliding_roi,
            )
            torch.cuda.empty_cache()
        try:
            z = self._sliding(x)
            return EncodeResult(z, pad, "sliding")
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EncodeOOMError(
                f"sliding-window encode OOM at roi={self.sliding_roi}; "
                f"reduce roi_size or lower the input spatial shape"
            ) from exc

    # ------------------------------------------------------------------ impl
    #
    # MAISI's AutoencoderKlMaisi is built with ``norm_float16=True`` so its
    # GroupNorm modules cast activations to ``torch.float16`` while the conv
    # weights stay ``float32``. We must enter ``torch.amp.autocast`` so the
    # convs also run in fp16 on CUDA and dtypes align. On CPU we run the
    # bare module (autocast on CPU dtype=float16 is not supported by MAISI's
    # custom conv-split layer).

    def _autocast(self) -> Any:
        from contextlib import nullcontext

        if self.precision_mode == "fp32":
            return nullcontext()
        if self.handle.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _full(self, x: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            return self.handle.model.encode_stage_2_inputs(x).to(torch.float32)

    def _sliding(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from monai.inferers import SlidingWindowInferer
        except ImportError as exc:  # pragma: no cover — monai is a hard dep
            raise EncodeOOMError(
                "monai.inferers.SlidingWindowInferer unavailable; install monai"
            ) from exc
        inferer = SlidingWindowInferer(
            roi_size=self.sliding_roi,
            sw_batch_size=1,
            overlap=self.sliding_overlap,
            mode=self.sliding_mode,
            progress=False,
        )

        def _predictor(window: torch.Tensor) -> torch.Tensor:
            with self._autocast():
                return self.handle.model.encode_stage_2_inputs(window).to(torch.float32)

        return inferer(inputs=x, network=_predictor)

    # ------------------------------------------------------------------ attrs

    def to_attrs(self) -> dict[str, Any]:
        return {
            "sliding_window_roi": list(self.sliding_roi),
            "sliding_window_overlap": self.sliding_overlap,
            "sliding_window_mode": self.sliding_mode,
            "depth_pad_base": self.depth_pad_base,
            "percentile_lower": self.percentile_lower,
            "percentile_upper": self.percentile_upper,
            "percentile_foreground_only": self.percentile_foreground_only,
            "precision_mode": self.precision_mode,
        }
