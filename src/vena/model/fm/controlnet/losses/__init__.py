"""Curriculum-aware FM losses (S1 / S2 / S3).

Public surface:

* :class:`CompositeLoss` — applies a stage-defined combination of terms and
  returns ``(total_scalar, per_term_dict)``.
* :func:`build_loss` — factory that maps a stage name + config to a
  :class:`CompositeLoss`.

S1 implements :class:`CFMLoss` only. S2 and S3 ship as stubs that raise
``NotImplementedError`` when the builder is asked for them; this keeps the
curriculum's wiring visible in the code while constraining the present change
to the proposal's acceptance criterion (S1 smoke).
"""

from .base import AbstractFMLoss, CompositeLoss, LossInputs
from .builder import build_loss
from .cfm import CFMLoss

__all__ = [
    "AbstractFMLoss",
    "CFMLoss",
    "CompositeLoss",
    "LossInputs",
    "build_loss",
]
