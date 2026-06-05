"""LoRA adapter for the MAISI rectified-flow trunk.

Implements :class:`vena.model.fm.maisi.peft.base.BasePEFT` using the
HuggingFace ``peft`` library's :func:`peft.inject_adapter_in_model` entry
point. ``inject_adapter_in_model`` mutates the trunk in place: every matched
``nn.Linear`` is replaced by a :class:`peft.tuners.lora.LoraLayer` subclass
with the base weight frozen and a low-rank update
``\\Delta W = (\\alpha / r) \\cdot B A`` where ``A \\in \\mathbb{R}^{r \\times d}``
is Gaussian-initialised and ``B \\in \\mathbb{R}^{d \\times r}`` is zero-init,
giving identity-at-init (Hu et al. 2022, *LoRA: Low-Rank Adaptation of Large
Language Models*, arXiv:2106.09685).

By default the target modules are ``["to_q", "to_k", "to_v", "out_proj"]``
— the four projection sub-modules exposed by MONAI's ``SABlock`` (the
self-attention block ``attn1``) and ``CrossAttentionBlock`` (the cross-attn
block ``attn2``) inside every ``DiffusionUNetTransformerBlock`` of
``DiffusionModelUNetMaisi``. ``peft`` matches by suffix on the module's
qualified name, so this single ``target_modules`` list catches both the
self- and cross-attention paths.

Use ``get_peft_model`` was deliberately not chosen: that helper wraps the
backbone in a :class:`peft.PeftModel`, which is convenient for HF Trainer
flows but for our diffusion-U-Net pipeline it (a) introduces an extra
``forward`` indirection that interacts subtly with the
:mod:`vena.model.fm.maisi.grad_safe` instance-monkeypatch on
``forward`` / ``_apply_down_blocks``, and (b) ships a heavier state-dict.
``inject_adapter_in_model`` does the minimum: same trunk instance, extra
adapter slots inside each matched ``nn.Linear``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .base import BasePEFT
from .exceptions import PeftConfigError, PeftError
from .registry import register_peft

_DEFAULT_TARGETS = ["to_q", "to_k", "to_v", "out_proj"]
_ALLOWED_KEYS: set[str] = {
    "r",
    "alpha",
    "dropout",
    "target_modules",
    "bias",
    "init_lora_weights",
    "use_rslora",
    "use_dora",
}


@dataclass(frozen=True)
class _LoRACfg:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = tuple(_DEFAULT_TARGETS)
    bias: str = "none"
    init_lora_weights: str | bool = "gaussian"
    use_rslora: bool = False
    use_dora: bool = False


@register_peft("lora")
class LoRA(BasePEFT):
    """LoRA on the MAISI trunk's attention projections."""

    variant = "lora"

    def __init__(self, cfg: _LoRACfg) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> LoRA:
        unknown = set(params) - _ALLOWED_KEYS
        if unknown:
            raise PeftConfigError(
                f"LoRA: unknown params {sorted(unknown)}; allowed: {sorted(_ALLOWED_KEYS)}"
            )
        targets = params.get("target_modules", _DEFAULT_TARGETS)
        if not isinstance(targets, (list, tuple)) or not all(isinstance(t, str) for t in targets):
            raise PeftConfigError(
                f"LoRA: target_modules must be a list of strings; got {targets!r}"
            )
        if not targets:
            raise PeftConfigError("LoRA: target_modules must be non-empty")
        cfg = _LoRACfg(
            r=int(params.get("r", 16)),
            alpha=int(params.get("alpha", params.get("r", 16))),
            dropout=float(params.get("dropout", 0.0)),
            target_modules=tuple(targets),
            bias=str(params.get("bias", "none")),
            init_lora_weights=params.get("init_lora_weights", "gaussian"),
            use_rslora=bool(params.get("use_rslora", False)),
            use_dora=bool(params.get("use_dora", False)),
        )
        if cfg.r <= 0:
            raise PeftConfigError(f"LoRA: r must be > 0, got {cfg.r}")
        if cfg.alpha <= 0:
            raise PeftConfigError(f"LoRA: alpha must be > 0, got {cfg.alpha}")
        if cfg.bias not in {"none", "all", "lora_only"}:
            raise PeftConfigError(
                f"LoRA: bias must be one of 'none'|'all'|'lora_only', got {cfg.bias!r}"
            )
        return cls(cfg)

    # ------------------------------------------------------------------
    # Apply / state.
    # ------------------------------------------------------------------

    def apply(self, trunk: nn.Module) -> nn.Module:
        from peft import LoraConfig, inject_adapter_in_model

        peft_cfg = LoraConfig(
            r=self.cfg.r,
            lora_alpha=self.cfg.alpha,
            lora_dropout=self.cfg.dropout,
            target_modules=list(self.cfg.target_modules),
            bias=self.cfg.bias,
            init_lora_weights=self.cfg.init_lora_weights,
            use_rslora=self.cfg.use_rslora,
            use_dora=self.cfg.use_dora,
            task_type=None,
        )
        inject_adapter_in_model(peft_cfg, trunk, adapter_name="default")
        # ``inject_adapter_in_model`` sets ``requires_grad=True`` on the
        # adapter tensors but does NOT freeze the rest of the trunk; do it
        # explicitly here so the optimiser's ``requires_grad`` filter picks
        # up only the LoRA params.
        for name, p in trunk.named_parameters():
            if "lora_" not in name:
                p.requires_grad_(False)
        return trunk

    def trainable_parameters(self, trunk: nn.Module) -> list[nn.Parameter]:
        return [p for n, p in trunk.named_parameters() if "lora_" in n and p.requires_grad]

    def extract_state(self, trunk: nn.Module) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in trunk.named_parameters() if "lora_" in n}

    def load_state(self, trunk: nn.Module, state: dict[str, torch.Tensor]) -> None:
        existing = {n for n, _ in trunk.named_parameters() if "lora_" in n}
        missing = existing - set(state)
        extra = set(state) - existing
        if missing or extra:
            raise PeftError(
                f"LoRA.load_state: state mismatch (missing={sorted(missing)[:3]}, "
                f"extra={sorted(extra)[:3]})"
            )
        param_by_name = dict(trunk.named_parameters())
        with torch.no_grad():
            for name, tensor in state.items():
                target = param_by_name[name]
                if target.shape != tensor.shape:
                    raise PeftError(
                        f"LoRA.load_state: shape mismatch at {name}: "
                        f"target={tuple(target.shape)} state={tuple(tensor.shape)}"
                    )
                target.copy_(tensor.to(target.device, dtype=target.dtype))

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "params": {
                "r": self.cfg.r,
                "alpha": self.cfg.alpha,
                "dropout": self.cfg.dropout,
                "target_modules": list(self.cfg.target_modules),
                "bias": self.cfg.bias,
                "init_lora_weights": self.cfg.init_lora_weights,
                "use_rslora": self.cfg.use_rslora,
                "use_dora": self.cfg.use_dora,
            },
        }
