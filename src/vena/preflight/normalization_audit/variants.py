"""Variant registry for the V3 normalisation audit.

Each variant is a frozen :class:`NormalizationVariant` whose ``apply``
takes a per-modality image dict + an optional brain-mask and returns the
normalised modality dict. Variants register themselves into a module-level
dict via ``@register_variant(id)`` at import time.

Variants V0..V8 mapping (see spec §1):

* V0 — production baseline. Per-modality 99.5%ile, fg=True, clip=True.
* V1 — V0 with clip=False. Minimal mechanical fix.
* V2 — per-modality 99.9%ile clip. Preserves top 0.1 %.
* V3 — per-modality 99.99%ile clip. Preserves top 0.01 %.
* V4 — joint-modality 99.5%ile, fg=True (mask), clip=True. The only
        variant that preserves inter-modality scale; see :mod:`.joint`.
* V7 — whole-volume 99.5%ile (fg=False), clip=False. T1C-RFlow-style.
* V8 — asymmetric per-modality: T1c at 99.9%ile, others at 99.5%ile.

V5 (z-score + tanh softclip) and V6 (WhiteStripe) are deferred and not
implemented in this pass — see plan §2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import torch

from vena.model.autoencoder.maisi.preprocessing import percentile_normalise

from .joint import joint_modality_percentile_normalise

ApplyFn = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor] | None],
    dict[str, torch.Tensor],
]

VARIANT_REGISTRY_VERSION: Final[str] = "0.1.0"


@dataclass(frozen=True)
class NormalizationVariant:
    """A single normalisation variant in the V3 audit registry.

    Attributes
    ----------
    id : str
        Stable identifier (V0..V8). Round-trips into ``decision.json``.
    variant_version : str
        Variant-specific version pin. Bumped on any behavioural change.
    description : str
        One-line human-readable rationale.
    params : dict[str, float | bool | str]
        Parameters that control the variant (e.g. ``upper=99.5``,
        ``clip=False``). Serialised into ``decision.json``.
    apply : Callable
        ``apply(images, masks) -> images_normalised``. ``masks`` may be
        ``None`` (foreground heuristic falls back to ``x > 0``) or a dict
        with key ``brain`` mapping to the brain mask tensor. Per-patient
        batches; shape ``(B, 1, H, W, D)``.
    """

    id: str
    variant_version: str
    description: str
    apply: ApplyFn = field(repr=False)
    params: dict[str, float | bool | str] = field(default_factory=dict)


_REGISTRY: dict[str, NormalizationVariant] = {}


def register_variant(variant: NormalizationVariant) -> NormalizationVariant:
    """Add ``variant`` to the module-level registry; idempotent on re-import."""
    if variant.id in _REGISTRY and _REGISTRY[variant.id] is not variant:
        existing = _REGISTRY[variant.id]
        if existing.variant_version != variant.variant_version:
            raise ValueError(
                f"register_variant: id '{variant.id}' already registered with "
                f"version '{existing.variant_version}'; refusing to overwrite "
                f"with '{variant.variant_version}'."
            )
    _REGISTRY[variant.id] = variant
    return variant


def get_variant_registry() -> dict[str, NormalizationVariant]:
    """Return a shallow copy of the registry (callers may not mutate)."""
    return dict(_REGISTRY)


def _get_brain_mask(
    masks: dict[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Resolve the brain mask from the masks dict (key 'brain'), else None."""
    if masks is None:
        return None
    return masks.get("brain")


# ---------------------------------------------------------------------------
# V0 — production baseline. Per-modality 99.5%ile, fg=True (or mask), clip=True.
# ---------------------------------------------------------------------------
def _v0_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    return {
        k: percentile_normalise(
            x,
            lower=0.0,
            upper=99.5,
            foreground_only=True,
            mask=m,
            clip=True,
        )
        for k, x in images.items()
    }


# ---------------------------------------------------------------------------
# V1 — V0 with clip=False. Lets enhancement keep its super-percentile values.
# ---------------------------------------------------------------------------
def _v1_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    return {
        k: percentile_normalise(
            x,
            lower=0.0,
            upper=99.5,
            foreground_only=True,
            mask=m,
            clip=False,
        )
        for k, x in images.items()
    }


# ---------------------------------------------------------------------------
# V2 — per-modality 99.9%ile clip. Preserves top 0.1 %.
# ---------------------------------------------------------------------------
def _v2_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    return {
        k: percentile_normalise(
            x,
            lower=0.0,
            upper=99.9,
            foreground_only=True,
            mask=m,
            clip=True,
        )
        for k, x in images.items()
    }


# ---------------------------------------------------------------------------
# V3 — per-modality 99.99%ile clip. Preserves essentially the entire tail.
# ---------------------------------------------------------------------------
def _v3_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    return {
        k: percentile_normalise(
            x,
            lower=0.0,
            upper=99.99,
            foreground_only=True,
            mask=m,
            clip=True,
        )
        for k, x in images.items()
    }


# ---------------------------------------------------------------------------
# V4 — joint-modality 99.5%ile (mask-driven), clip=True. The only variant
#      that preserves the inter-modality scale. T1c stays brighter than
#      T1pre at the enhancing voxel in the normalised space.
# ---------------------------------------------------------------------------
def _v4_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    return joint_modality_percentile_normalise(
        images,
        lower=0.0,
        upper=99.5,
        clip=True,
        mask=m,
    )


# ---------------------------------------------------------------------------
# V7 — whole-volume 99.5%ile (fg=False), clip=False. T1C-RFlow-style.
# ---------------------------------------------------------------------------
def _v7_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    return {
        k: percentile_normalise(
            x,
            lower=0.0,
            upper=99.5,
            foreground_only=False,
            mask=None,
            clip=False,
        )
        for k, x in images.items()
    }


# ---------------------------------------------------------------------------
# V8 — asymmetric per-modality. T1c at (0, 99.9), others at (0, 99.5).
#      All foreground-only with brain mask, clip=True. Cheapest targeted fix.
# ---------------------------------------------------------------------------
_V8_UPPER_PER_MODALITY: Final[dict[str, float]] = {
    "t1c": 99.9,
    "t1pre": 99.5,
    "t2": 99.5,
    "flair": 99.5,
}


def _v8_apply(
    images: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    m = _get_brain_mask(masks)
    out: dict[str, torch.Tensor] = {}
    for k, x in images.items():
        upper = _V8_UPPER_PER_MODALITY.get(k, 99.5)
        out[k] = percentile_normalise(
            x,
            lower=0.0,
            upper=upper,
            foreground_only=True,
            mask=m,
            clip=True,
        )
    return out


# ---------------------------------------------------------------------------
# Registration.
# ---------------------------------------------------------------------------
register_variant(
    NormalizationVariant(
        id="V0",
        variant_version="0.1.0",
        description="Production baseline — per-modality 99.5%ile, fg=True (mask), clip=True.",
        params={"lower": 0.0, "upper": 99.5, "foreground_only": True, "clip": True},
        apply=_v0_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V1",
        variant_version="0.1.0",
        description="V0 with clip=False — preserve the super-percentile T1c bright tail.",
        params={"lower": 0.0, "upper": 99.5, "foreground_only": True, "clip": False},
        apply=_v1_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V2",
        variant_version="0.1.0",
        description="Per-modality 99.9%ile clip — preserves top 0.1 %.",
        params={"lower": 0.0, "upper": 99.9, "foreground_only": True, "clip": True},
        apply=_v2_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V3",
        variant_version="0.1.0",
        description="Per-modality 99.99%ile clip — preserves top 0.01 %.",
        params={"lower": 0.0, "upper": 99.99, "foreground_only": True, "clip": True},
        apply=_v3_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V4",
        variant_version="0.1.0",
        description=(
            "Joint-modality 99.5%ile (one (lo, hi) per patient over the union of "
            "foreground voxels across all modalities), clip=True. Preserves "
            "inter-modality intensity scale — T1c stays brighter than T1pre at "
            "the enhancing voxel."
        ),
        params={
            "lower": 0.0,
            "upper": 99.5,
            "joint_modality": True,
            "clip": True,
        },
        apply=_v4_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V7",
        variant_version="0.1.0",
        description=(
            "Whole-volume 99.5%ile (fg=False), clip=False — T1C-RFlow-style. "
            "Background dilutes the percentile so the foreground gets a wider "
            "headroom; the no-clip flag preserves the bright tail."
        ),
        params={"lower": 0.0, "upper": 99.5, "foreground_only": False, "clip": False},
        apply=_v7_apply,
    )
)
register_variant(
    NormalizationVariant(
        id="V8",
        variant_version="0.1.0",
        description=(
            "Asymmetric per-modality: T1c at (0, 99.9), T1pre/T2/FLAIR at "
            "(0, 99.5). Cheapest targeted fix — zero KL risk on T2/FLAIR "
            "(their distribution is unchanged)."
        ),
        params={
            "lower": 0.0,
            "upper_t1c": 99.9,
            "upper_other": 99.5,
            "foreground_only": True,
            "clip": True,
        },
        apply=_v8_apply,
    )
)


__all__ = [
    "VARIANT_REGISTRY_VERSION",
    "NormalizationVariant",
    "get_variant_registry",
    "register_variant",
]
