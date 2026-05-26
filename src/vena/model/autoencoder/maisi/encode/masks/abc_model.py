"""Abstract contract for mask downsamplers from image space to MAISI space.

Each concrete model under ``encode/masks/models/`` implements one strategy
for compressing a discrete-label mask from the native MRI grid (typically
``(240, 240, 155)`` for UCSF-PDGM) to the MAISI latent grid (``(60, 60, 40)``
after depth padding). The choice is a research lever: nearest-neighbor
preserves discrete labels, per-class avg-pool yields soft conditioning maps,
max-pool keeps only "presence" information.

Models are registered by name in :mod:`encode.masks.models.__init__` and
selected from a YAML config string. Adding a new method is a self-contained
change: drop a file under ``models/``, register it in
``encode/masks/models/__init__._REGISTRY``, list it in this docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class AbstractMaskDownsampler(ABC):
    """Abstract base for image-space → MAISI-space mask downsamplers.

    Subclasses declare four class-level attributes that document the output
    contract *before* a tensor is even allocated:

    * ``output_channels`` — number of channels in the downsampled mask.
    * ``output_dtype`` — torch dtype of the output (``float32`` for soft maps,
      ``int8`` for discrete labels).
    * ``channel_names`` — semantic names for each channel; round-tripped into
      the H5 ``masks/tumor_latent`` dataset attrs so consumers can interpret
      a 3-channel tensor as ``(NETC, ED, ET)`` without guessing.
    * ``name`` — lowercase snake_case registry key.

    The instance method :meth:`downsample` takes a 5-D mask in image space
    and returns a 5-D tensor in MAISI space. Per-batch operation; the
    leading axis is preserved.
    """

    # Class-level contract attrs. Concrete subclasses must override all
    # four; abstract intermediaries may leave them as ``None``.
    output_channels: int = 0
    output_dtype: torch.dtype = torch.float32
    channel_names: tuple[str, ...] = ()
    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ``__abstractmethods__`` is populated by ABCMeta *after*
        # ``__init_subclass__`` runs, so we cannot rely on it. Walk the
        # standard attribute lookup chain for the two abstract methods we
        # require — a concrete override anywhere in the MRO masks the
        # abstract base. If either method is still abstract for ``cls``,
        # treat it as an intermediate (skip the attribute check).
        for name in ("downsample", "to_attrs"):
            method = getattr(cls, name, None)
            if method is None or getattr(method, "__isabstractmethod__", False):
                return
        for required, default in (
            ("output_channels", 0),
            ("output_dtype", None),
            ("channel_names", ()),
            ("name", ""),
        ):
            val = getattr(cls, required)
            if val == default or val is None:
                raise TypeError(
                    f"{cls.__name__} is a concrete AbstractMaskDownsampler subclass "
                    f"but does not override class attribute {required!r}."
                )
        if len(cls.channel_names) != cls.output_channels:
            raise TypeError(
                f"{cls.__name__}: len(channel_names)={len(cls.channel_names)} "
                f"does not match output_channels={cls.output_channels}"
            )

    @abstractmethod
    def downsample(
        self,
        mask: torch.Tensor,
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Downsample ``mask`` to the MAISI grid.

        Parameters
        ----------
        mask : torch.Tensor
            Image-space mask, shape ``(B, 1, H, W, D)``, integral or float dtype.
        target_shape : tuple[int, int, int]
            Spatial shape of the latent grid, e.g. ``(60, 60, 40)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, output_channels, *target_shape)``, dtype
            ``output_dtype``.

        Raises
        ------
        vena.model.autoencoder.maisi.exceptions.ShapeContractError
            If ``mask`` or the output does not satisfy the shape contract.
        vena.model.autoencoder.maisi.encode.masks.shared.exceptions.LabelCodeError
            If ``mask`` contains unexpected label codes.
        """

    @abstractmethod
    def to_attrs(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict describing this downsampler.

        Written verbatim into the H5 ``masks/tumor_latent`` dataset attrs so
        a downstream consumer can re-derive the channel ordering, label
        codes, and any tunable hyperparameters without re-instantiating the
        Python object.
        """
