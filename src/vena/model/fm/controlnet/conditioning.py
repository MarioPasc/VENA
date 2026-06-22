"""Configurable conditioning-tensor assembler.

The assembler maps a batch dict produced by the DataModule into the spatial
conditioning tensor expected by the ControlNet's
``conditioning_embedding_in_channels``. Channel ordering is controlled by a
list of :class:`ConditioningSpec` entries — the **single point of variation**
for ablations.

Example
-------
>>> specs = [
...     ConditioningSpec(kind="latent", key="t1pre"),
...     ConditioningSpec(kind="latent", key="t2"),
...     ConditioningSpec(kind="latent", key="flair"),
...     ConditioningSpec(kind="mask", key="wt", downsampler="identity"),
... ]
>>> asm = ConditioningAssembler(specs, latent_channels=4)
>>> asm.total_channels  # 4 + 4 + 4 + 1
13
>>> c = asm.forward(batch)  # (B, 13, 60, 60, 40)

For S2 perturbation (proposal §5.3 step 2), the same batch can be re-assembled
with the WT mask zeroed:

>>> c_perturb = asm.forward(batch, perturb_keys={"wt"})

A spec string of the form ``"latent:t1pre"`` or
``"mask:wt:identity"`` is parsed by :meth:`ConditioningSpec.from_string`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn

from .downsample import get_downsampler

logger = logging.getLogger(__name__)


SpecKind = Literal["latent", "mask", "prior"]


class ConditioningSpec(BaseModel):
    """One channel-group in the conditioning tensor.

    Attributes
    ----------
    kind : {"latent", "mask", "prior"}
        Determines the batch-dict key prefix:
            * ``latent`` → ``batch["z_<key>"]`` (e.g. ``z_t1pre``).
            * ``mask``   → ``batch["m_<key>"]`` (e.g. ``m_wt``).
            * ``prior``  → ``batch["prior_<key>"]`` (e.g. ``prior_vessel``).
    key : str
        Modality / mask / prior name.
    downsampler : str
        Registry name from
        :func:`vena.model.fm.controlnet.downsample.get_downsampler`.
    downsampler_kwargs : dict
        Forwarded to the constructor of the chosen downsampler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SpecKind
    key: str
    downsampler: str = "identity"
    downsampler_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def _no_separator(cls, v: str) -> str:
        if ":" in v:
            raise ValueError("spec key must not contain ':' (reserved separator)")
        return v

    @classmethod
    def from_string(cls, s: str) -> ConditioningSpec:
        """Parse a YAML-friendly ``kind:key[:downsampler[:k=v,...]]`` string."""
        parts = s.split(":")
        if len(parts) < 2:
            raise ValueError(f"conditioning spec '{s}' must be 'kind:key[:downsampler[:kv,...]]'")
        kind, key, *rest = parts
        downsampler = rest[0] if rest else "identity"
        kwargs: dict[str, Any] = {}
        for tok in rest[1:]:
            if "=" not in tok:
                raise ValueError(f"downsampler kwarg '{tok}' must be 'k=v'")
            k, v = tok.split("=", 1)
            kwargs[k.strip()] = _coerce(v.strip())
        return cls(kind=kind, key=key, downsampler=downsampler, downsampler_kwargs=kwargs)

    def batch_key(self) -> str:
        if self.kind == "latent":
            return f"z_{self.key}"
        if self.kind == "mask":
            return f"m_{self.key}"
        return f"prior_{self.key}"


def _coerce(v: str) -> Any:
    """Best-effort parse of a downsampler kwarg value from string."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


class ConditioningAssembler(nn.Module):
    """Assemble the ControlNet's spatial conditioning tensor.

    Parameters
    ----------
    specs : list[ConditioningSpec | str]
        Ordered channel-group descriptors. Strings are parsed via
        :meth:`ConditioningSpec.from_string`.
    latent_channels : int
        Channels per latent modality (4 for MAISI-V2).
    mask_channels : int
        Channels per mask spec. The H5 stores a 3-channel BraTS-soft tumour
        mask; for the WT union we collapse to 1 at the DataModule level, so
        the assembler sees a 1-channel input. Configurable here to support
        the multi-channel ablation directly later.
    prior_channels : int
        Channels per prior spec. Default 1 (single soft-prior map).
    """

    def __init__(
        self,
        specs: list[ConditioningSpec | str],
        latent_channels: int = 4,
        mask_channels: int = 1,
        prior_channels: int = 1,
    ) -> None:
        super().__init__()
        parsed: list[ConditioningSpec] = [
            s if isinstance(s, ConditioningSpec) else ConditioningSpec.from_string(s) for s in specs
        ]
        if not parsed:
            raise ValueError("ConditioningAssembler requires at least one spec")
        self.specs: list[ConditioningSpec] = parsed
        self.latent_channels = int(latent_channels)
        self.mask_channels = int(mask_channels)
        self.prior_channels = int(prior_channels)
        self.downsamplers = nn.ModuleList(
            [get_downsampler(spec.downsampler, **spec.downsampler_kwargs) for spec in parsed]
        )

    @property
    def channels_per_spec(self) -> list[int]:
        """Per-spec output channel count, downsampler-aware.

        When a downsampler exposes an integer ``out_channels`` property
        (i.e. it lifts the channel dim, e.g. :class:`LiftTo4ChDownsampler`),
        the assembler uses that value instead of the kind-based default.
        Stateless operators return ``None`` for ``out_channels`` and the
        kind default applies.
        """
        out: list[int] = []
        for spec, ds in zip(self.specs, self.downsamplers, strict=True):
            if spec.kind == "latent":
                default_n = self.latent_channels
            elif spec.kind == "mask":
                default_n = self.mask_channels
            else:
                default_n = self.prior_channels
            override = getattr(ds, "out_channels", None)
            out.append(int(override) if override is not None else default_n)
        return out

    @property
    def total_channels(self) -> int:
        return sum(self.channels_per_spec)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        perturb_keys: set[str] | None = None,
    ) -> torch.Tensor:
        """Assemble the conditioning tensor.

        Parameters
        ----------
        batch : dict[str, Tensor]
            DataModule batch; see :meth:`ConditioningSpec.batch_key` for the
            key naming convention.
        perturb_keys : set[str] | None
            Set of spec ``key``s whose channel-group should be **zeroed** in
            the output. Used by S2 to materialise ``c_perturb`` (proposal
            §5.3). Only ``kind="mask"`` and ``kind="prior"`` entries are
            sensitive to perturbation; latents are passed through regardless.

        Returns
        -------
        Tensor of shape ``(B, total_channels, h, w, d)``.
        """
        pieces: list[torch.Tensor] = []
        perturb = perturb_keys or set()
        for spec, downsampler in zip(self.specs, self.downsamplers, strict=True):
            key = spec.batch_key()
            if key not in batch:
                raise KeyError(
                    f"ConditioningAssembler: batch missing required key '{key}' "
                    f"(spec={spec.kind}:{spec.key})"
                )
            x = batch[key]
            x = downsampler(x)
            if spec.key in perturb and spec.kind in ("mask", "prior"):
                x = torch.zeros_like(x)
            pieces.append(x)
        return torch.cat(pieces, dim=1)
