"""Label harmonisation for BraTS-style segmentation masks.

Converts multi-class integer labels to boolean WT and NETC masks using
code-agnostic rules that cover both BraTS-2021 and BraTS-2023 label
conventions without branching on the cohort name.

BraTS-2021 (UCSF-PDGM, UPENN-GBM, IvyGAP, REMBRANDT):
    0 = background, 1 = NETC, 2 = ED, 4 = ET.

BraTS-2023 (BraTS-GLI/PED/Africa, LUMIERE):
    0 = background, 1 = NETC, 2 = ED, 3 = ET.

Derived regions (code-agnostic across both conventions):
    WT   = label > 0   (whole-tumour: any non-background label)
    NETC = label == 1  (necrotic core: label 1 in both conventions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vena.segmentation.exceptions import SegTargetError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["harmonise_labels"]


def harmonise_labels(label: NDArray) -> dict[str, NDArray]:
    """Harmonise an integer BraTS label map to boolean region masks.

    Parameters
    ----------
    label : NDArray
        Integer label array of arbitrary shape.  Dtype must be integer-
        compatible (int8, int16, int32, uint8, etc.).  Expected values are a
        subset of ``{0, 1, 2, 3, 4}``; unexpected values are accepted without
        error (they contribute to WT if non-zero).

    Returns
    -------
    dict[str, NDArray]
        ``{"wt": ..., "netc": ...}`` where each value is a *bool* array of
        the same shape as *label*.

    Raises
    ------
    SegTargetError
        If *label* is empty (zero-size).
    """
    if label.size == 0:
        raise SegTargetError("label array is empty — cannot harmonise zero-size input")

    # WT: any non-background label — code-agnostic across BraTS-2021 and 2023
    wt: NDArray = (label > 0).astype(bool)
    # NETC: label == 1 in both conventions (necrotic core / non-enhancing tumour core)
    netc: NDArray = (label == 1).astype(bool)

    return {"wt": wt, "netc": netc}
