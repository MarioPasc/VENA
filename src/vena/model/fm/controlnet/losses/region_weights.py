"""Per-region weight tensor for the S1 v3 region-weighted L1 velocity loss.

Two modes
---------

**Sub-region mode** (legacy S1 v3, :class:`RegionWeights`)
    Five disjoint regions: BG, Brain-not-WT, NETC, ED, ET.  Used by the
    retired v3b_rw arm.  ``RegionWeights(enabled=False)`` collapses to
    standard ``F.l1_loss(reduction="mean")``.

**Brain/TC mode** (:class:`BrainTCWeights`)
    Three strict-partition regions: BG (``~in_brain``), Brain
    (``in_brain & ~TC``), TC (tumour core, ``in_brain & m_tc_hard``).
    Edema falls in Brain — reconstructed from inputs, intended.  Default
    ``{brain: 1.0, tc: 1.0}`` is *numerically identical* to unweighted
    mean L1.  Select by putting ``brain`` / ``tc`` keys in the
    ``loss.cfm.region_weights`` YAML block.

Why this exists
---------------
S1 v2's mean-reduction L1 over the 4-channel velocity field gave WT-region
voxels **0.095 %** of the total loss magnitude (n=30 UCSF-PDGM fold-0 val,
audit table in
``.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md``
§3.1). The optimiser favoured non-WT correctness by ~1047:1. With WT
upweighted by ``α_wt = 200`` the WT loss contribution becomes ~17 % of total
— comparable to the brain non-WT contribution and large enough to actually
steer the trunk's predictions inside the tumour.

The five regions are constructed **disjoint by construction** at threshold
``τ = 0.5``: a tumour voxel is excluded from ``brain_not_wt`` via the
``~wt_hard`` clause, so the per-voxel weight is exactly ``weight[region]``
(not a sum across regions).

Easy disable
------------
* ``RegionWeights(enabled=False)`` → the caller falls back to standard
  ``F.l1_loss(reduction="mean")``. Byte-identical to the legacy S1 v2 path.
* ``RegionWeights(wt=200.0, netc=..., ed=..., et=...)`` → setting ``wt`` to
  a non-null value overrides netc/ed/et with the single weight, recovering
  the "single WT weight" ablation.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field


class RegionWeights(BaseModel):
    """Per-region L1 weighting configuration.

    Attributes
    ----------
    enabled : bool
        Master switch. ``False`` (or all weights equal to 1.0) collapses the
        weighted loss to standard mean L1 within float-precision.
    bg : float
        Background (``brain == 0``) voxel weight. Default 1.0.
    brain_not_wt : float
        Brain-but-not-tumour voxel weight. Default 1.0.
    netc : float
        Necrosis-and-non-enhancing-tumour-core voxel weight (BraTS label 1).
        Defaults to 50.0; together with ``ed`` and ``et`` they push the
        WT-region loss contribution from 0.095 % to ~17 %.
    ed : float
        Peritumoural edema voxel weight (BraTS label 2). Default 50.0.
    et : float
        Enhancing tumour voxel weight (BraTS label 4). Default 300.0 —
        higher than NETC/ED because PSNR_ET is the load-bearing clinical
        signal (father doc §1.4 / P6 metric-trap) and ET is the smallest
        sub-region by voxel count.
    wt : float | None
        If non-null, overrides ``netc/ed/et`` with this single weight (the
        "single WT weight" ablation). ``None`` means use the per-sub-region
        weights.
    threshold : float
        Soft-mask threshold τ. Voxels with channel value ``≥ τ`` count as
        belonging to that sub-region. Default 0.5 (matches the existing
        ``derived_from_tumor_latent`` region spec).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    bg: float = Field(default=1.0, ge=0.0)
    brain_not_wt: float = Field(default=1.0, ge=0.0)
    netc: float = Field(default=50.0, ge=0.0)
    ed: float = Field(default=50.0, ge=0.0)
    et: float = Field(default=300.0, ge=0.0)
    wt: float | None = Field(default=None, ge=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def build_region_weight_tensor(
    rw: RegionWeights,
    m_brain: torch.Tensor | None,
    m_tumor: torch.Tensor | None,
    *,
    channels: int = 4,
) -> torch.Tensor | None:
    """Build a per-voxel weight tensor for region-weighted L1.

    Parameters
    ----------
    rw : RegionWeights
        Configuration. ``rw.enabled == False`` returns ``None`` so the caller
        can fall back to the standard mean-L1 path.
    m_brain : Tensor | None
        Brain mask, shape ``(B, 1, h, w, d)``. Voxels with value > 0.5
        count as brain.
    m_tumor : Tensor | None
        Soft per-class tumour mask, shape ``(B, 3, h, w, d)``; channels
        ``[NETC, ED, ET]``. Voxels with channel value ≥ ``rw.threshold``
        count as belonging to that sub-region.
    channels : int
        Velocity-field channel count (4 for MAISI latent). The returned
        tensor is broadcast to ``(B, channels, h, w, d)`` so it multiplies
        elementwise with ``F.l1_loss(v_pred, u_target, reduction="none")``.

    Returns
    -------
    Tensor of shape ``(B, channels, h, w, d)`` (broadcast view) or ``None``
    when ``rw.enabled == False``.

    Raises
    ------
    ValueError
        When ``rw.enabled`` but either mask is missing — region weighting
        cannot be computed without them; the caller should config-validate.
    """
    if not rw.enabled:
        return None
    if m_brain is None or m_tumor is None:
        raise ValueError(
            "build_region_weight_tensor: enabled=True requires both m_brain "
            "and m_tumor; got m_brain=%r m_tumor=%r" % (m_brain, m_tumor)
        )

    τ = rw.threshold
    in_brain = m_brain > 0.5  # (B, 1, h, w, d) bool
    m_t_hard = m_tumor >= τ  # (B, 3, h, w, d) bool
    m_wt_hard = m_t_hard.any(dim=1, keepdim=True)  # (B, 1, ...) bool

    # Disjoint partition. A tumour voxel is excluded from brain_not_wt via
    # the ``~m_wt_hard`` clause, so the per-voxel weight is exactly one term.
    region_bg = ~in_brain
    region_bnwt = in_brain & ~m_wt_hard
    region_netc = m_t_hard[:, 0:1] & in_brain
    region_ed = m_t_hard[:, 1:2] & in_brain
    region_et = m_t_hard[:, 2:3] & in_brain

    w_netc = rw.wt if rw.wt is not None else rw.netc
    w_ed = rw.wt if rw.wt is not None else rw.ed
    w_et = rw.wt if rw.wt is not None else rw.et

    w = (
        region_bg.float() * rw.bg
        + region_bnwt.float() * rw.brain_not_wt
        + region_netc.float() * w_netc
        + region_ed.float() * w_ed
        + region_et.float() * w_et
    )  # (B, 1, h, w, d)

    # Broadcast across the velocity-field channels.
    return w.expand(-1, channels, -1, -1, -1)


# ---------------------------------------------------------------------------
# Brain / TC mode (S2 v3a+)
# ---------------------------------------------------------------------------


class BrainTCWeights(BaseModel):
    """Per-region weight config for the three-partition Brain/TC loss mode.

    Regions (strict partition — every voxel gets exactly one weight):

    * **BG** (``~in_brain``): weight ``brain``.
    * **Brain** (``in_brain & ~m_tc_hard``): weight ``brain``. Edema falls here
      and is reconstructed from the input modalities — this is intended.
    * **TC** (tumour core, ``in_brain & m_tc_hard``): weight ``tc``.

    With ``{brain: 1.0, tc: 1.0}`` every voxel gets weight 1.0 and the
    weighted loss is numerically identical to ``F.l1_loss(reduction="mean")``.
    This is the load-bearing guarantee: ship the mechanism, change nothing
    numerically until ``tc`` is deliberately raised.

    Attributes
    ----------
    enabled : bool
        Master switch. ``False`` falls back to standard mean L1.
    brain : float
        Weight applied to BG and Brain-not-TC voxels. Default 1.0.
    tc : float
        Weight applied to tumour-core (TC = NETC + ET) voxels. Default 1.0.
    threshold : float
        Soft-mask threshold τ. Voxels with ``m_tc_soft >= τ`` belong to TC.
        Default 0.5.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    brain: float = Field(default=1.0, ge=0.0)
    tc: float = Field(default=1.0, ge=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BrainTCMissingMaskError(ValueError):
    """Raised when TC-soft mask is absent but brain/tc mode is active.

    A silent fallback would silently change the training objective for GPU-days.
    """


def build_brain_tc_weight_tensor(
    rw: BrainTCWeights,
    m_brain: torch.Tensor | None,
    m_tc_soft: torch.Tensor | None,
    *,
    channels: int = 4,
) -> torch.Tensor | None:
    """Build a per-voxel weight tensor for the brain/TC three-region loss.

    Parameters
    ----------
    rw : BrainTCWeights
        Configuration.  ``rw.enabled == False`` returns ``None`` so the
        caller can fall back to standard mean-L1.
    m_brain : Tensor | None
        Brain mask, shape ``(B, 1, h, w, d)``.  Voxels ``> 0.5`` are brain.
    m_tc_soft : Tensor | None
        Soft tumour-core probability map, shape ``(B, 1, h, w, d)``, in
        ``[0, 1]``.  Channel 0 of ``masks/tumor_latent_soft`` (TC = NETC+ET;
        edema excluded).  **Must be non-None when ``rw.enabled``** — absence
        raises :class:`BrainTCMissingMaskError`.
    channels : int
        Velocity-field channel count (4 for MAISI latent). The returned
        tensor broadcasts to ``(B, channels, h, w, d)``.

    Returns
    -------
    Tensor of shape ``(B, channels, h, w, d)`` or ``None`` when disabled.

    Raises
    ------
    BrainTCMissingMaskError
        When ``rw.enabled`` but ``m_tc_soft`` is ``None``.
    ValueError
        When ``rw.enabled`` but ``m_brain`` is ``None``.
    """
    if not rw.enabled:
        return None
    if m_tc_soft is None:
        raise BrainTCMissingMaskError(
            "build_brain_tc_weight_tensor: brain/tc mode enabled but "
            "m_tc_soft is None. Task 20 must have landed (batch['m_tc_soft'] "
            "served from masks/tumor_latent_soft channel 0). Never fall back "
            "to m_tumor — a silent fallback trains the wrong objective."
        )
    if m_brain is None:
        raise ValueError(
            "build_brain_tc_weight_tensor: brain/tc mode enabled but "
            "m_brain is None; cannot partition BG from Brain without it."
        )

    τ = rw.threshold
    in_brain = m_brain > 0.5  # (B, 1, h, w, d) bool
    m_tc_hard = m_tc_soft >= τ  # (B, 1, h, w, d) bool — TC = NETC + ET

    # Strict partition: every voxel is assigned to exactly one region.
    # BG and Brain-not-TC both receive weight ``brain``; only TC gets ``tc``.
    # Implementation: start from a uniform ``brain`` tensor and overlay ``tc``
    # on TC voxels — exactly one assignment per voxel, no double-counting.
    region_tc = m_tc_hard & in_brain  # (B, 1, h, w, d)

    w = torch.where(
        region_tc,
        torch.tensor(rw.tc, dtype=m_brain.dtype, device=m_brain.device),
        torch.tensor(rw.brain, dtype=m_brain.dtype, device=m_brain.device),
    )  # (B, 1, h, w, d)

    # Broadcast across velocity-field channels.
    return w.expand(-1, channels, -1, -1, -1)
