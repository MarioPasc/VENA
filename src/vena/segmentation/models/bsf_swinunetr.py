"""BrainSegFounder (BSF) SwinUNETR backbones for vena.segmentation.

Three public builders are registered here:

Arm A — ``bsf_swinunetr_brats``  (BraTS SSL, comparator)
    Checkpoint: BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold0.pt
    SSL task trained on BraTS data with 4-channel input (FLAIR, T1pre, T1ce, T2).
    Stem: shape (48, 4, 2, 2, 2) → sliced to (48, 3, 2, 2, 2) via indices
    ``_BRATS_STEM_CHANNEL_SLICE = [0, 1, 3]`` (FLAIR=0, T1pre=1, T2=3, dropping T1ce=2).
    Channel order follows BraTS convention ``[FLAIR, T1pre, T1ce, T2]``.
    NEVER load from ``BrainSegFounder_finetuned_BraTS/*`` — those are BraTS fine-tuned
    models with L1+L2+L3 data leakage (BraTS train split overlaps our OOF folds).

Arm B — ``bsf_swinunetr_ukb``  (UKB SSL, primary / headline)
    Checkpoint: BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt
    SSL task trained on UK Biobank with 1-channel input (T1w).
    Stem: shape (48, 1, 2, 2, 2) — incompatible with 3-ch target → goes to
    ``LoadReport.skipped``; all other 125 encoder keys transfer.
    This is the leak-free headline arm (UKB has no glioma patients, no
    BraTS-style labelling — OOF-safe by cohort independence).

Architecture constants ``_BSF_BRATS_SWIN_KW`` / ``_BSF_UKB_SWIN_KW``
    Empirically inferred from the checkpoint key structure (2026-07-23):

    Arm A (BraTS) has ``depths=(2, 2, 6, 2)`` — stage 3 contains 6 transformer
    blocks (indices 0–5) vs the MONAI default of 2.  Without this correction,
    56 stage-3 encoder keys are absent from a default-architecture build,
    yielding 126/198 (63.6%) coverage.  With ``depths=(2, 2, 6, 2)`` all 182
    encoder keys transfer; the 16 remaining skipped keys are exclusively the
    SSL task heads (rotation_head, contrastive_head, conv.*).

    Arm B (UKB) has ``depths=(2, 2, 2, 2)`` — the MONAI default.  No change
    required; 125/142 coverage (88.0%); the single skipped swinViT key is the
    1-ch stem which cannot transfer to a 3-ch model.

    These constants are NOT user hyperparameters.  They are immutable facts
    about the frozen checkpoints and must not be overridden via ``ModelConfig``.

Verified checkpoint SHA-256 (logged at load time via :func:`load_bsf_encoder`):
    Arm A: e46d80ce75f3828222cdfd3f4891c753e9cd0356719e7d21803c89e1b8797077
    Arm B: 4be92492ae4f55e934e278700a13a6c18ef8406daba2177c5f9f157dcff5f341

Real load statistics (validated on local workstation, 2026-07-23):
    Arm A: total=198 ckpt keys, matched=182 (91.9%), skipped=16
        — 16 skipped = SSL task heads (rotation_head, contrastive_head, conv.*).
        — Zero shape mismatches; all 182 encoder keys transfer cleanly.
        — Requires ``depths=(2, 2, 6, 2)``; the MONAI default depths=(2,2,2,2)
          yields only 126/198 (63.6%) because 56 stage-3 blocks are absent.
    Arm B: total=142 ckpt keys, matched=125 (88.0%), skipped=17
        — 1 skipped = stem shape mismatch (48,1,2,2,2) ≠ (48,3,2,2,2);
          16 skipped = SSL task heads (same as Arm A).

SwinUNETR spatial divisibility (load-bearing):
    MONAI SwinUNETR._check_input_size requires each spatial dim to be
    divisible by patch_size ** num_stages = 2 ** 5 = 32.
    Use (B, 3, 32, 32, 32) or (B, 3, 64, 64, 64) for Arms A/B.
    (B, 3, 32, 32, 24) raises ValueError because 24 % 32 ≠ 0.
    SegResNet (Arm C) has no such constraint beyond 8-divisibility.

Deep supervision (hook-based):
    ``register_forward_hook`` on ``backbone.decoder3`` (fires at H/4) and
    ``backbone.decoder2`` (fires at H/2).  Execution order in SwinUNETR
    forward: decoder5 → decoder4 → decoder3 → decoder2 → decoder1 → out.

    Captured outputs:
        _ds_outputs[0] = decoder3 output: (B, 2*feature_size=96, H/4, W/4, D/4)
        _ds_outputs[1] = decoder2 output: (B, feature_size=48,   H/2, W/2, D/2)

    Aux head channels:
        aux_head0 = Conv3d(feature_size,    out_channels, 1)  → applied to dec2 (H/2)
        aux_head1 = Conv3d(2*feature_size, out_channels, 1)  → applied to dec3 (H/4)

    Return tuple (deep_supervision=True):
        (logits, aux_head0(ds[1]), aux_head1(ds[0]))
        [0]: (B, out_channels, H, W, D)       full resolution
        [1]: (B, out_channels, H/2, W/2, D/2) from decoder2
        [2]: (B, out_channels, H/4, W/4, D/4) from decoder3

    The hook structure is identical for Arms A and B regardless of ``depths``,
    because ``decoder3`` and ``decoder2`` are MONAI decoder blocks (not swinViT
    encoder blocks) and are unaffected by the deeper stage-3.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from vena.segmentation.config import ModelConfig

from vena.segmentation.exceptions import SegModelError
from vena.segmentation.models.registry import register_segmentation_model

logger = logging.getLogger(__name__)

# BraTS-SSL stem slice: [FLAIR=0, T1pre=1, T2=3], dropping T1ce=2.
# BraTS channel order: [FLAIR, T1pre, T1ce, T2] (0-indexed).
_BRATS_STEM_CHANNEL_SLICE: list[int] = [0, 1, 3]
_STEM_KEY: str = "swinViT.patch_embed.proj.weight"

# Per-arm SwinViT architecture constants.
# Inferred empirically from checkpoint key structure (2026-07-23):
#   BraTS SSL: stage 3 contains 6 transformer blocks (block indices 0–5).
#     depth=6 at stage 3 vs the MONAI default of 2.  Missing this yields 56
#     absent stage-3 keys and 63.6% coverage; with the correct value, 91.9%.
#   UKB SSL: depths=(2,2,2,2) — MONAI default, no correction needed.
# These dicts are passed to SwinUNETR(**_BSF_*_SWIN_KW) in each builder.
# They are NOT user-configurable and must not be added to ModelConfig.
_BSF_BRATS_SWIN_KW: dict[str, object] = {
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
}
_BSF_UKB_SWIN_KW: dict[str, object] = {
    "depths": (2, 2, 2, 2),
    "num_heads": (3, 6, 12, 24),
}

# Default checkpoint paths — overridden by cfg.checkpoint when set.
_BSF_BRATS_CKPT = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models"
    "/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold0.pt"
)
_BSF_UKB_CKPT = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models"
    "/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt"
)


# ---------------------------------------------------------------------------
# LoadReport
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LoadReport:
    """Summary of a BSF checkpoint load operation.

    Attributes
    ----------
    matched:
        Number of checkpoint keys that transferred successfully (name present
        in target model AND shapes identical after any slicing).
    total:
        Total number of checkpoint keys attempted (after stripping the
        DataParallel ``module.`` prefix).  ``matched + len(skipped) == total``.
    skipped:
        Checkpoint keys (stripped) that were NOT loaded: either the name is
        absent from the target model's state_dict (SSL task heads, extra
        architecture blocks) or the shape did not match after optional slicing.
    """

    matched: int
    total: int
    skipped: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_bsf_encoder(
    model: nn.Module,
    ckpt_path: Path,
    *,
    brats_channel_slice: list[int] | None = None,
) -> LoadReport:
    """Load BSF SSL encoder weights into a MONAI SwinUNETR model.

    All checkpoint keys carry a DataParallel ``module.`` prefix, which is
    stripped before matching against the target model's state_dict.  The load
    is always non-strict: keys present in the checkpoint but absent from the
    target (or with mismatching shapes) are collected in
    :class:`LoadReport`.skipped.

    Parameters
    ----------
    model:
        Instantiated MONAI SwinUNETR (or any ``nn.Module`` with a
        ``state_dict()`` / ``load_state_dict()`` API).
    ckpt_path:
        Absolute path to the BSF ``.pt`` checkpoint file.
    brats_channel_slice:
        If provided (e.g. ``[0, 1, 3]``), the stem weight at
        ``swinViT.patch_embed.proj.weight`` is sliced along dim 1 before
        shape comparison.  Use for the BraTS-SSL 4-ch checkpoint (Arm A).
        Pass ``None`` for UKB (Arm B) — the 1-ch stem will mismatch and land
        in ``LoadReport.skipped``.

    Returns
    -------
    LoadReport
        ``matched`` = keys transferred;
        ``total``   = total checkpoint keys after prefix strip;
        ``skipped`` = stripped key names that were not loaded.
        Invariant: ``matched + len(skipped) == total``.

    Raises
    ------
    SegModelError
        If ``ckpt_path`` does not exist.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SegModelError(f"BSF checkpoint not found: {ckpt_path}")

    sha = _sha256_file(ckpt_path)
    logger.info("Loading BSF checkpoint: %s  sha256=%s", ckpt_path, sha)

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    src_sd: dict[str, Tensor] = raw["state_dict"]

    # Strip DataParallel 'module.' prefix from every key.
    stripped: dict[str, Tensor] = {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in src_sd.items()
    }

    # Optional 4ch → 3ch stem slice (Arm A only).
    if brats_channel_slice is not None and _STEM_KEY in stripped:
        w = stripped[_STEM_KEY]
        stripped[_STEM_KEY] = w[:, brats_channel_slice, ...]
        logger.debug(
            "BraTS stem sliced: %s → %s via indices %s",
            list(w.shape),
            list(stripped[_STEM_KEY].shape),
            brats_channel_slice,
        )

    # Match against target model by name AND shape.
    target_sd = model.state_dict()
    to_load: dict[str, Tensor] = {}
    skipped: list[str] = []

    for k, v in stripped.items():
        if k in target_sd and v.shape == target_sd[k].shape:
            to_load[k] = v
        else:
            skipped.append(k)
            if k in target_sd:
                logger.debug(
                    "Shape mismatch '%s': ckpt %s ≠ model %s — skipping",
                    k,
                    list(v.shape),
                    list(target_sd[k].shape),
                )
            else:
                logger.debug("Key '%s' absent from target model — skipping", k)

    model.load_state_dict(to_load, strict=False)
    matched = len(to_load)
    total = len(stripped)

    logger.info(
        "BSF load complete: matched %d/%d (%.1f%%), skipped %d",
        matched,
        total,
        100.0 * matched / total if total else 0.0,
        len(skipped),
    )
    return LoadReport(matched=matched, total=total, skipped=skipped)


# ---------------------------------------------------------------------------
# _VenaSwinUNETR wrapper
# ---------------------------------------------------------------------------


class _VenaSwinUNETR(nn.Module):
    """MONAI SwinUNETR with optional hook-based deep supervision.

    Hooks are registered on ``backbone.decoder3`` (fires at H/4) and
    ``backbone.decoder2`` (fires at H/2) and removed in :meth:`remove_hooks`
    (called automatically in ``__del__``).  Always call ``remove_hooks()``
    before discarding the instance to avoid dangling references.

    The :attr:`backbone` property gives direct access to the underlying MONAI
    SwinUNETR for checkpoint loading and inspection.

    Parameters
    ----------
    cfg:
        Segmentation model config (``in_channels``, ``out_channels``,
        ``feature_size``, ``deep_supervision``).
    swinunetr_kwargs:
        Extra keyword arguments forwarded verbatim to ``SwinUNETR.__init__``
        (e.g. ``depths``, ``num_heads``).  Use the per-arm constants
        ``_BSF_BRATS_SWIN_KW`` / ``_BSF_UKB_SWIN_KW`` — do NOT expose these
        through ``ModelConfig``; they are immutable facts about the checkpoints.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        swinunetr_kwargs: dict[str, object] | None = None,
    ) -> None:
        from monai.networks.nets import SwinUNETR

        super().__init__()
        extra: dict[str, object] = swinunetr_kwargs or {}
        self._backbone = SwinUNETR(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            feature_size=cfg.feature_size,
            spatial_dims=3,
            **extra,
        )
        self._ds = cfg.deep_supervision
        self._ds_outputs: list[Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        if cfg.deep_supervision:
            fs = cfg.feature_size
            # decoder2 outputs feature_size ch at H/2 spatial.
            # decoder3 outputs 2*feature_size ch at H/4 spatial.
            self._aux_head0 = nn.Conv3d(fs, cfg.out_channels, kernel_size=1)
            self._aux_head1 = nn.Conv3d(2 * fs, cfg.out_channels, kernel_size=1)

            # Registration order mirrors SwinUNETR forward execution:
            # decoder3 fires first (H/4), decoder2 fires second (H/2).
            self._handles = [
                self._backbone.decoder3.register_forward_hook(self._capture_hook),
                self._backbone.decoder2.register_forward_hook(self._capture_hook),
            ]

    @property
    def backbone(self) -> nn.Module:
        """The underlying MONAI SwinUNETR module (for weight loading)."""
        return self._backbone

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _capture_hook(
        self,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        self._ds_outputs.append(output)

    def remove_hooks(self) -> None:
        """Detach all registered forward hooks.  Idempotent."""
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
            divisible by 32 (SwinUNETR spatial constraint; 24 % 32 ≠ 0 raises
            ValueError inside MONAI's ``_check_input_size``).

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            ``deep_supervision=False``: single ``(B, out_channels, H, W, D)``
            ``deep_supervision=True``  : ``(logits, aux_H2, aux_H4)`` where:
                ``aux_H2`` has shape ``(B, out_channels, H/2, W/2, D/2)``
                ``aux_H4`` has shape ``(B, out_channels, H/4, W/4, D/4)``
        """
        if not self._ds:
            return self._backbone(x)

        self._ds_outputs = []
        logits = self._backbone(x)

        if len(self._ds_outputs) != 2:
            raise RuntimeError(
                f"Expected 2 DS outputs from hooks, got {len(self._ds_outputs)}.  "
                "backbone.decoder3/decoder2 structure may have changed."
            )

        # _ds_outputs[0] = decoder3 output (H/4, 2*feature_size ch)
        # _ds_outputs[1] = decoder2 output (H/2, feature_size ch)
        aux_h4 = self._aux_head1(self._ds_outputs[0])
        aux_h2 = self._aux_head0(self._ds_outputs[1])

        return logits, aux_h2, aux_h4


# ---------------------------------------------------------------------------
# Builders (registered with the model registry)
# ---------------------------------------------------------------------------


@register_segmentation_model("bsf_swinunetr_brats")
def build_bsf_swinunetr_brats(cfg: ModelConfig) -> nn.Module:  # type: ignore[misc]
    """Build Arm A: BraTS-SSL SwinUNETR (comparator).

    Loads SSL pre-training weights from the BraTS fold-0 checkpoint.
    The 4-ch stem is sliced to 3-ch via ``_BRATS_STEM_CHANNEL_SLICE``.
    NEVER loads from ``BrainSegFounder_finetuned_BraTS/*`` (data leakage).

    Uses ``_BSF_BRATS_SWIN_KW`` (``depths=(2,2,6,2)``) to match the BraTS
    checkpoint's deeper stage-3 architecture.  The MONAI default
    ``depths=(2,2,2,2)`` misses 56 stage-3 encoder blocks (63.6% coverage);
    the correct architecture achieves 182/198 (91.9%).

    If ``cfg.checkpoint`` is set, uses that path instead of the default.
    If the checkpoint is absent, the model is returned with MONAI random init
    and a WARNING is emitted (allows smoke runs without the checkpoint).

    Spatial divisibility: H, W, D must each be divisible by 32.
    """
    model = _VenaSwinUNETR(cfg, swinunetr_kwargs=_BSF_BRATS_SWIN_KW)
    ckpt = Path(cfg.checkpoint) if cfg.checkpoint is not None else _BSF_BRATS_CKPT

    if not ckpt.exists():
        logger.warning(
            "BSF BraTS checkpoint not found at %s — using MONAI random init.",
            ckpt,
        )
        return model

    load_bsf_encoder(model.backbone, ckpt, brats_channel_slice=_BRATS_STEM_CHANNEL_SLICE)
    return model


@register_segmentation_model("bsf_swinunetr_ukb")
def build_bsf_swinunetr_ukb(cfg: ModelConfig) -> nn.Module:  # type: ignore[misc]
    """Build Arm B: UKB-SSL SwinUNETR (primary / headline).

    Loads SSL pre-training weights from the UKB checkpoint.  The 1-ch stem
    is incompatible with the 3-ch target and goes to ``LoadReport.skipped``;
    all other 125 encoder keys transfer cleanly (88.0% matched/total).

    Uses ``_BSF_UKB_SWIN_KW`` (``depths=(2,2,2,2)``) — the MONAI default —
    which matches the UKB checkpoint's architecture exactly.

    This is the leak-free headline arm — UK Biobank contains no glioma
    patients and no BraTS-style labels (OOF-safe by cohort independence).

    If ``cfg.checkpoint`` is set, uses that path instead of the default.
    If the checkpoint is absent, the model is returned with MONAI random init.

    Spatial divisibility: H, W, D must each be divisible by 32.
    """
    model = _VenaSwinUNETR(cfg, swinunetr_kwargs=_BSF_UKB_SWIN_KW)
    ckpt = Path(cfg.checkpoint) if cfg.checkpoint is not None else _BSF_UKB_CKPT

    if not ckpt.exists():
        logger.warning(
            "BSF UKB checkpoint not found at %s — using MONAI random init.",
            ckpt,
        )
        return model

    # No channel slice: 1-ch stem goes to LoadReport.skipped as expected.
    load_bsf_encoder(model.backbone, ckpt, brats_channel_slice=None)
    return model


__all__: list[str] = []
