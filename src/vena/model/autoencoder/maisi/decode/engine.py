"""MAISI latent → image decoding with depth crop-back.

:class:`MaisiDecoder` mirrors :class:`MaisiEncoder`: full-volume first,
sliding-window on ``OutOfMemoryError``, then :func:`crop_to_original` to
restore the pre-pad depth so the decoded image lives on the same grid as
the H5 source.

This module is *not* responsible for renormalising decoded intensities into
the original raw-MR range. MAISI returns roughly ``[0, 1]`` after decode for
MR inputs; callers comparing against the original H5 must apply the same
percentile rescale they used at encode time. The QC routine handles this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..exceptions import EncodeOOMError, ShapeContractError
from ..loader import AutoencoderHandle
from ..preprocessing import CropPadSpec, DepthPad, crop_to_original

logger = logging.getLogger(__name__)

DecodeMode = Literal["full", "sliding"]


@dataclass(frozen=True)
class DecodeResult:
    """One forward pass through the MAISI decoder.

    Attributes
    ----------
    image : torch.Tensor
        Decoded image in normalised ``[0, 1]`` space, shape
        ``(B, 1, H, W, D_original)``.
    inference_mode : str
        ``"full"`` or ``"sliding"``.
    """

    image: torch.Tensor
    inference_mode: DecodeMode


class MaisiDecoder:
    """Forward-only adapter around :class:`AutoencoderKlMaisi.decode_stage_2_outputs`."""

    def __init__(
        self,
        handle: AutoencoderHandle,
        sliding_roi: tuple[int, int, int] = (20, 20, 8),
        sliding_overlap: float = 0.4,
        sliding_mode: str = "gaussian",
        precision_mode: str = "autocast",
    ) -> None:
        # Decoder sliding window operates in *latent* coordinates, so the
        # ROI is 4× smaller than the encoder's. (80,80,32) → (20,20,8).
        self.handle = handle
        self.sliding_roi = tuple(sliding_roi)
        self.sliding_overlap = float(sliding_overlap)
        self.sliding_mode = sliding_mode
        if precision_mode not in {"autocast", "fp32"}:
            raise ValueError(
                f"precision_mode must be 'autocast' or 'fp32'; got {precision_mode!r}"
            )
        self.precision_mode = precision_mode

    # ------------------------------------------------------------------ API

    @torch.inference_mode()
    def decode(
        self,
        z: torch.Tensor,
        pad: DepthPad | None = None,
        crop_spec: CropPadSpec | None = None,
        mode: Literal["auto", "full", "sliding"] = "auto",
    ) -> DecodeResult:
        """Decode a MAISI latent batch to image space.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``(B, C, h, w, d)`` in MAISI latent space.
        pad : DepthPad | None
            Legacy depth-pad metadata. When provided (and ``crop_spec`` is
            ``None``), :func:`crop_to_original` restores the original depth.
        crop_spec : CropPadSpec | None
            When provided, decode to box space ``(B, 1, *target_shape)`` and
            return the decoded box AS-IS (no inversion to native). QC and
            exhaustive-val compare in box space. Mutually exclusive with
            ``pad``; if both are provided ``crop_spec`` takes precedence.
        mode : {"auto", "full", "sliding"}
            Inference strategy; same semantics as the encoder.
        """
        if z.ndim != 5:
            raise ShapeContractError(f"decode expects (B,C,h,w,d); got {tuple(z.shape)}")
        z = z.to(self.handle.device, dtype=torch.float32, non_blocking=True)

        if crop_spec is not None:
            # Box path: decode straight to box volume; no inversion.
            if mode == "full":
                return DecodeResult(self._full(z), "full")
            if mode == "sliding":
                return DecodeResult(self._sliding(z), "sliding")
            try:
                return DecodeResult(self._full(z), "full")
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "MAISI decode OOM on full-volume latent %s (box path); retrying via sliding-window roi=%s",
                    tuple(z.shape),
                    self.sliding_roi,
                )
                torch.cuda.empty_cache()
            try:
                return DecodeResult(self._sliding(z), "sliding")
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise EncodeOOMError(
                    f"sliding-window decode OOM at roi={self.sliding_roi}; "
                    f"reduce roi_size or downsample the latent batch"
                ) from exc

        # Legacy depth-pad path. Require pad to be provided.
        if pad is None:
            raise ValueError(
                "decode: either crop_spec or pad must be provided; both are None"
            )

        if mode == "full":
            x = self._full(z)
            return DecodeResult(crop_to_original(x, pad), "full")
        if mode == "sliding":
            x = self._sliding(z)
            return DecodeResult(crop_to_original(x, pad), "sliding")

        try:
            x = self._full(z)
            return DecodeResult(crop_to_original(x, pad), "full")
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "MAISI decode OOM on full-volume latent %s; retrying via sliding-window roi=%s",
                tuple(z.shape),
                self.sliding_roi,
            )
            torch.cuda.empty_cache()
        try:
            x = self._sliding(z)
            return DecodeResult(crop_to_original(x, pad), "sliding")
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EncodeOOMError(
                f"sliding-window decode OOM at roi={self.sliding_roi}; "
                f"reduce roi_size or downsample the latent batch"
            ) from exc

    # ------------------------------------------------------------------ impl

    def _autocast(self) -> Any:
        from contextlib import nullcontext

        if self.precision_mode == "fp32":
            return nullcontext()
        if self.handle.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _full(self, z: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            return self.handle.model.decode_stage_2_outputs(z).to(torch.float32)

    def _sliding(self, z: torch.Tensor) -> torch.Tensor:
        try:
            from monai.inferers import SlidingWindowInferer
        except ImportError as exc:  # pragma: no cover
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
                return self.handle.model.decode_stage_2_outputs(window).to(torch.float32)

        return inferer(inputs=z, network=_predictor)
