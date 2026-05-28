"""Region-mask resolver — declares per-region source and produces per-batch masks.

Per the user requirement, the YAML config carries an explicit ``regions:`` block
naming the source of every region (``latents_h5``, ``derived_*``,
``fallback_all_ones``, ``skipped``). The :class:`RegionResolver` constructs
:class:`RegionMasks` from each training batch following those declarations and
logs a one-shot summary at startup so the audit trail is unambiguous.

Source values
-------------
``latents_h5``
    Read the mask directly from the batch (the DataModule has copied it from
    the H5). Requires the H5 key in ``h5_key``.

``derived_from_tumor_latent``
    Compute ``(masks/tumor_latent.sum(0) >= threshold)``. Used for the WT mask
    while ``masks/wt_latent`` is missing.

``derived_via_scipy_binary_dilation``
    Compute ``scipy.ndimage.binary_dilation(wt, structure=...)``. Used for
    ``wt_dilated`` while the H5 lacks it.

``derived``
    Generic "computed from the others" — used for ``bg = brain & ~wt_dilated``.

``fallback_all_ones``
    Constant-True mask of the latent shape. Used for ``brain`` while the H5
    lacks an explicit brain mask; warns once at resolver construction.

``skipped``
    No mask. Metrics for this region are emitted as NaN with ``n_patients=0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


SourceKind = Literal[
    "latents_h5",
    "derived_from_tumor_latent",
    "derived_via_scipy_binary_dilation",
    "derived",
    "fallback_all_ones",
    "skipped",
]


class RegionSpec(BaseModel):
    """One row of the YAML ``regions`` block."""

    model_config = ConfigDict(extra="forbid")

    source: SourceKind
    h5_key: str | None = None
    threshold: float = 0.5
    structure: str = "ones_3x3x3"  # interpretation handled in resolver
    extra: dict[str, Any] = Field(default_factory=dict)


REQUIRED_REGIONS: tuple[str, ...] = ("brain", "wt", "wt_dilated", "bg", "vessel")


@dataclass
class RegionMasks:
    """Per-batch boolean masks at latent resolution. ``None`` if skipped."""

    brain: torch.Tensor | None
    wt: torch.Tensor | None
    wt_dilated: torch.Tensor | None
    bg: torch.Tensor | None
    vessel: torch.Tensor | None

    def get(self, name: str) -> torch.Tensor | None:
        return getattr(self, name)


def _structure_kernel_size(name: str) -> int:
    """Map a structuring-element name to the equivalent max-pool kernel size."""
    if name == "ones_3x3x3":
        return 3
    raise ValueError(f"unknown structure '{name}'; supported: {{ones_3x3x3}}")


class RegionResolver:
    """Resolves per-batch region masks from the spec block."""

    def __init__(self, specs: dict[str, RegionSpec]) -> None:
        missing = [r for r in REQUIRED_REGIONS if r not in specs]
        if missing:
            raise ValueError(
                f"regions config missing required keys {missing}; "
                f"must list every region in {REQUIRED_REGIONS}"
            )
        self.specs: dict[str, RegionSpec] = dict(specs)
        self._summarise()

    def _summarise(self) -> None:
        for name in REQUIRED_REGIONS:
            spec = self.specs[name]
            logger.info(
                "region '%s': source=%s%s",
                name,
                spec.source,
                f" h5_key={spec.h5_key}" if spec.h5_key else "",
            )
            if spec.source == "skipped":
                logger.warning(
                    "region '%s' is skipped — metrics for this region will be NaN.",
                    name,
                )
            if spec.source == "fallback_all_ones":
                logger.warning(
                    "region '%s' falls back to all-ones — metrics include non-brain voxels.",
                    name,
                )

    def resolve(self, batch: dict[str, Any]) -> RegionMasks:
        """Build a :class:`RegionMasks` from a batch dict.

        The batch is expected to carry tensors keyed:

        * ``m_wt``         — binary WT mask, shape ``(B, 1, h, w, d)``.
        * optionally ``m_brain``, ``m_vessel``, ``m_wt_dilated`` if the
          corresponding source is ``latents_h5``.
        """
        wt = self._resolve_wt(batch)
        wt_dilated = self._resolve_wt_dilated(batch, wt)
        brain = self._resolve_brain(batch, reference=wt)
        bg = self._resolve_bg(batch, brain=brain, wt_dilated=wt_dilated)
        vessel = self._resolve_vessel(batch)
        return RegionMasks(brain=brain, wt=wt, wt_dilated=wt_dilated, bg=bg, vessel=vessel)

    # ------------------------------------------------------------------

    def _resolve_wt(self, batch: dict[str, Any]) -> torch.Tensor | None:
        spec = self.specs["wt"]
        if spec.source == "skipped":
            return None
        if spec.source == "latents_h5":
            return batch[spec.h5_key or "m_wt"].bool()
        if spec.source == "derived_from_tumor_latent":
            # DataModule already produced `m_wt` from tumor_latent ≥ threshold.
            return batch["m_wt"].bool()
        raise ValueError(f"unsupported source for wt: {spec.source}")

    def _resolve_wt_dilated(
        self, batch: dict[str, Any], wt: torch.Tensor | None
    ) -> torch.Tensor | None:
        spec = self.specs["wt_dilated"]
        if spec.source == "skipped":
            return None
        if spec.source == "latents_h5":
            return batch[spec.h5_key or "m_wt_dilated"].bool()
        if spec.source == "derived_via_scipy_binary_dilation":
            if wt is None:
                return None
            # Binary dilation with an all-ones (k,k,k) structuring element is
            # exactly stride-1 max-pooling of the binary mask with padding
            # ``k//2`` (which preserves the spatial shape for odd ``k``). Doing
            # it on-device avoids the per-batch CPU/NumPy round-trip the old
            # ``scipy.ndimage.binary_dilation`` loop incurred — same result.
            k = _structure_kernel_size(spec.structure)
            dilated = F.max_pool3d(wt.float(), kernel_size=k, stride=1, padding=k // 2)
            return dilated > 0.5
        raise ValueError(f"unsupported source for wt_dilated: {spec.source}")

    def _resolve_brain(
        self, batch: dict[str, Any], reference: torch.Tensor | None
    ) -> torch.Tensor | None:
        spec = self.specs["brain"]
        if spec.source == "skipped":
            return None
        if spec.source == "latents_h5":
            return batch[spec.h5_key or "m_brain"].bool()
        if spec.source == "fallback_all_ones":
            if reference is None:
                return None
            return torch.ones_like(reference, dtype=torch.bool)
        raise ValueError(f"unsupported source for brain: {spec.source}")

    def _resolve_bg(
        self,
        batch: dict[str, Any],
        brain: torch.Tensor | None,
        wt_dilated: torch.Tensor | None,
    ) -> torch.Tensor | None:
        spec = self.specs["bg"]
        if spec.source == "skipped":
            return None
        if spec.source == "latents_h5":
            return batch[spec.h5_key or "m_bg"].bool()
        if spec.source == "derived":
            if brain is None or wt_dilated is None:
                return None
            return brain & ~wt_dilated
        raise ValueError(f"unsupported source for bg: {spec.source}")

    def _resolve_vessel(self, batch: dict[str, Any]) -> torch.Tensor | None:
        spec = self.specs["vessel"]
        if spec.source == "skipped":
            return None
        if spec.source == "latents_h5":
            return batch[spec.h5_key or "m_vessel"].bool()
        raise ValueError(f"unsupported source for vessel: {spec.source}")
