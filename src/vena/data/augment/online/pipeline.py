"""Compose a list of :class:`LatentAugmentation` into a runtime pipeline.

The pipeline is dataset-local — instantiated once per ``LatentH5Dataset`` and
invoked from :meth:`_read_one` so it runs on the DataLoader worker process.
A worker-private :class:`random.Random` is built lazily on first use (seeded
from ``torch.initial_seed()`` so workers diverge) and reused for the lifetime
of the worker.

The result of a call is a tuple ``(sample, combo_tag)`` where ``combo_tag``
is the sorted-and-joined string of every applied augmentation's
:meth:`LatentAugmentation.param_tag`, or :data:`NO_AUG_TAG` when nothing fired.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch

from vena.data.augment.online.base import LatentAugmentation, LatentAugmentationError

logger = logging.getLogger(__name__)

# Canonical tag for the "no augmentation applied" case. Hard-coded so the
# per-epoch CSV is self-describing across configs.
NO_AUG_TAG: str = "none"

# Separator between per-augmentation tags inside a combination string. Kept
# distinct from any character used in the individual param tags.
COMBO_SEP: str = "+"


class AugmentationPipeline:
    """Apply a list of :class:`LatentAugmentation` independently per sample.

    Parameters
    ----------
    augmentations : list[LatentAugmentation]
        The pipeline applies them in list order, each gated independently by
        its own probability ``aug.p``.
    seed : int | None
        Base seed for the worker-private RNG. The actual per-worker seed is
        derived as ``seed ^ torch.initial_seed()`` so distinct workers in the
        same epoch see distinct streams without losing reproducibility.
    """

    def __init__(
        self,
        augmentations: list[LatentAugmentation],
        seed: int | None = 0,
    ) -> None:
        if not augmentations:
            raise LatentAugmentationError("AugmentationPipeline requires at least one augmentation")
        # Names must be unique within a pipeline. Two augmentations sharing a
        # tag would collide in the combination string.
        seen: set[str] = set()
        for aug in augmentations:
            if aug.name in seen:
                raise LatentAugmentationError(
                    f"duplicate augmentation name in pipeline: {aug.name!r}"
                )
            seen.add(aug.name)
        self.augmentations = list(augmentations)
        self.seed = int(seed) if seed is not None else 0
        self._rng: random.Random | None = None

    # ------------------------------------------------------------------

    def _ensure_rng(self) -> random.Random:
        if self._rng is None:
            # Mixing in ``torch.initial_seed()`` (which Lightning seeds
            # deterministically per worker) makes the pipeline reproducible
            # under ``pl.seed_everything(workers=True)`` while still giving
            # distinct workers distinct sequences.
            try:
                worker_seed = int(torch.initial_seed())
            except RuntimeError:
                worker_seed = 0  # outside a DataLoader worker
            self._rng = random.Random(self.seed ^ (worker_seed & 0xFFFF_FFFF))
        return self._rng

    # ------------------------------------------------------------------

    def __call__(self, sample: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Apply each augmentation independently; return the augmented sample.

        The Bernoulli gate fires per augmentation, not jointly: with three
        independent augmentations of probability 0.5 the combination space
        has eight elements (2³). The combination tag is the sorted-joined
        union of every fired ``param_tag``, or :data:`NO_AUG_TAG`.
        """
        rng = self._ensure_rng()
        applied_tags: list[str] = []
        for aug in self.augmentations:
            sample, tag = aug.maybe_apply(sample, rng)
            if tag is not None:
                applied_tags.append(tag)
        combo = COMBO_SEP.join(sorted(applied_tags)) if applied_tags else NO_AUG_TAG
        sample["_aug_combo"] = combo
        return sample, combo

    # ------------------------------------------------------------------

    def names(self) -> tuple[str, ...]:
        """Return the constituent augmentation names in pipeline order."""
        return tuple(a.name for a in self.augmentations)

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        names = ", ".join(self.names())
        return f"AugmentationPipeline([{names}])"
