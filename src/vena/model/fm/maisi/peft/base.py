"""Abstract base class for PEFT adapters on the MAISI rectified-flow trunk.

A concrete subclass implements one parameter-efficient fine-tuning recipe
(LoRA, IA3, DoRA, LoHa, ...) on the frozen :class:`DiffusionModelUNetMaisi`
backbone. The contract is intentionally narrow:

* The adapter mutates the trunk **in place** in :meth:`apply` so the rest of
  the FM stack — :class:`vena.model.fm.lightning.module.FMLightningModule`,
  the ControlNet's ``init_from_trunk``, ``grad_safe.py``, and the exhaustive
  validation subprocess — does not need to know which variant is active.
* The adapter is **identity-at-init**: after :meth:`apply`, the trunk's
  forward output is unchanged versus the pretrained backbone at step 0. This
  matches ControlNet's zero-convolution discipline (Zhang & Agrawala 2023)
  and LoRA's zero-init of the down-projection matrix (Hu et al. 2022) and is
  what makes joint ControlNet + PEFT training stable from step 0.
* The adapter knows how to (a) report its trainable parameters for the
  optimiser, (b) extract a tiny state-dict of only its adapter tensors for
  snapshotting to disk (so the exhaustive-val subprocess does not have to
  reload the full ~1 GB trunk EMA shadow), and (c) reload that snapshot
  into a freshly-wrapped trunk.

Variants register themselves via :func:`vena.model.fm.maisi.peft.registry.register_peft`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import nn


class BasePEFT(ABC):
    """Variant-agnostic PEFT adapter for the MAISI trunk."""

    #: Registry key. Subclasses set this as a class attribute and call
    #: :func:`register_peft` to advertise themselves.
    variant: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_dict(cls, params: dict[str, Any]) -> BasePEFT:
        """Build an instance from the YAML ``peft.params`` block.

        Parameters
        ----------
        params : dict
            Variant-specific parameters; the schema is owned by the subclass.
            Unknown keys must raise :class:`PeftConfigError`.

        Returns
        -------
        BasePEFT
            A configured adapter ready for :meth:`apply`.
        """

    @abstractmethod
    def apply(self, trunk: nn.Module) -> nn.Module:
        """Mutate the trunk in place and return it.

        Implementations must guarantee that the wrapped trunk's forward
        output equals the unwrapped trunk's forward output at step 0
        (identity-at-init). Base trunk parameters are set to
        ``requires_grad=False``; adapter parameters keep
        ``requires_grad=True``.

        Returning the same module instance (not a wrapper) is required so
        the ``grad_safe.py`` instance-monkeypatch on
        ``forward`` / ``_apply_down_blocks`` survives and so the
        :class:`FMLightningModule._trunk_module` registration continues to
        point at the live trunk.
        """

    @abstractmethod
    def trainable_parameters(self, trunk: nn.Module) -> list[nn.Parameter]:
        """Return only the adapter tensors for the optimiser.

        The base optimiser-construction code in
        :meth:`FMLightningModule.configure_optimizers` filters by
        ``requires_grad`` and so naturally picks up adapter-only tensors;
        this helper exists for explicit logging and tests.
        """

    @abstractmethod
    def extract_state(self, trunk: nn.Module) -> dict[str, torch.Tensor]:
        """Return a tiny state-dict containing only the adapter tensors.

        Used by callers (the exhaustive-val launcher, model-shipping
        scripts) that want to persist or transmit just the adapter delta
        rather than the full trunk + EMA shadow.
        """

    @abstractmethod
    def load_state(self, trunk: nn.Module, state: dict[str, torch.Tensor]) -> None:
        """Reload an adapter state-dict into a freshly-wrapped trunk.

        The trunk must already have been passed through :meth:`apply` so the
        adapter parameter slots exist. Implementations should raise
        :class:`PeftError` on missing or shape-mismatched keys.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Round-trip the resolved configuration into ``decision.json``.

        Returns a plain JSON-serialisable dict suitable for direct inclusion
        in the training run's ``decision.json`` v0.6.0 payload.
        """
