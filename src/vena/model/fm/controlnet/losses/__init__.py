"""Curriculum-aware FM losses (S1 / S2 / S3).

Public surface:

* :class:`CompositeLoss` — applies a stage-defined combination of terms and
  returns ``(total_scalar, per_term_dict)``.
* :func:`build_loss` — factory that maps a stage name + config to a
  :class:`CompositeLoss`.

S1 implements :class:`CFMLoss` only. S2 / skipS1 add the region-weighted
:class:`ContrastiveTumourLoss`. S3 returns a CFM-only composite and the
decoder-feature :class:`vena.model.fm.lpl.LplLoss` is added by
:class:`FMLightningModule.training_step` as
``lambda_img(epoch) * lpl_scalar`` — never through :class:`CompositeLoss`.
The legacy ``CappedLpReconLoss`` stub is no longer wired by any stage.
"""

from .base import AbstractFMLoss, CompositeLoss, LossInputs
from .builder import build_loss
from .cfm import CFMLoss
from .contrastive import ContrastiveTumourLoss, RegionTerm

__all__ = [
    "AbstractFMLoss",
    "CFMLoss",
    "CompositeLoss",
    "ContrastiveTumourLoss",
    "LossInputs",
    "RegionTerm",
    "build_loss",
]
