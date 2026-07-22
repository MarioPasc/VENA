"""Per-class temperature calibration for segmentation soft outputs.

Temperature scaling converts overconfident logits to calibrated probabilities
by dividing each logit by a scalar T > 0 before the sigmoid:
``p = sigmoid(logit / T)``.

Two separate scalars T_WT and T_NETC are fitted independently on the
held-out OOF calibration split by minimising binary NLL via L-BFGS.
A single global T is NOT acceptable (see design note B.f-§2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from vena.segmentation.exceptions import SegDerivationError

# L-BFGS solver settings for temperature fitting.
_LBFGS_LR: float = 0.1
_LBFGS_MAX_ITER: int = 200

# Guard against degenerate T ≤ 0.
_MIN_TEMPERATURE: float = 1e-4


@dataclass(frozen=True)
class ClassTemperatures:
    """Per-class temperature scalars for WT and NETC channels.

    Attributes
    ----------
    t_wt : float
        Temperature for the WT (whole-tumour) channel.  Always > 0.
    t_netc : float
        Temperature for the NETC (necrotic core) channel.  Always > 0.
    """

    t_wt: float
    t_netc: float


def _fit_single_temperature(logit: Tensor, target: Tensor) -> float:
    """Fit one temperature scalar minimising binary NLL via L-BFGS.

    Uses the ``exp(log_T)`` parameterisation to guarantee T > 0 without
    clamping, which provides unbounded gradient flow.

    Parameters
    ----------
    logit : Tensor
        Pre-sigmoid logits, arbitrary shape (flattened internally).
    target : Tensor
        Binary hard targets in {0, 1}, same shape as ``logit``.

    Returns
    -------
    float
        Optimal T > 0.
    """
    # Detach and move to CPU to avoid GPU memory pressure on large volumes.
    logit_cpu = logit.detach().float().cpu()
    target_cpu = target.detach().float().cpu()

    # Parameterise as log(T) so T = exp(log_T) is always positive.
    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_t], lr=_LBFGS_LR, max_iter=_LBFGS_MAX_ITER)

    def closure() -> Tensor:
        optimizer.zero_grad()
        t = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(logit_cpu / t, target_cpu)
        loss.backward()
        return loss

    optimizer.step(closure)

    t_fitted = float(torch.exp(log_t).item())
    return max(t_fitted, _MIN_TEMPERATURE)


def fit_temperatures(logits: Tensor, target_hard: Tensor) -> ClassTemperatures:
    """Fit per-class temperature scalars on calibration data.

    Minimises binary NLL independently for each channel (WT, NETC) via
    L-BFGS.  Must be called only on the held-out OOF calibration split;
    never on train or test data.

    Parameters
    ----------
    logits : Tensor
        Pre-sigmoid logits, shape ``(2, *spatial)``.
        Channel 0 = WT, channel 1 = NETC.
    target_hard : Tensor
        Binary hard targets in {0, 1}, shape ``(2, *spatial)``.
        Must match ``logits.shape``.

    Returns
    -------
    ClassTemperatures
        Fitted T_WT and T_NETC (both > 0).

    Raises
    ------
    SegDerivationError
        If shapes are incompatible or the first dimension is not 2.
    """
    if logits.shape != target_hard.shape:
        raise SegDerivationError(
            f"logits and target_hard shapes must match; "
            f"got {tuple(logits.shape)} vs {tuple(target_hard.shape)}"
        )
    if logits.ndim < 1 or logits.shape[0] != 2:
        raise SegDerivationError(f"expected 2 channels (WT, NETC); got shape {tuple(logits.shape)}")

    t_wt = _fit_single_temperature(logits[0], target_hard[0])
    t_netc = _fit_single_temperature(logits[1], target_hard[1])
    return ClassTemperatures(t_wt=t_wt, t_netc=t_netc)


def apply_temperature(logits: Tensor, temps: ClassTemperatures) -> Tensor:
    """Convert per-class logits to soft probabilities via temperature scaling.

    Computes ``sigmoid(logit / T)`` per channel.  The operation is
    argmax-preserving: since T > 0, ``sign(logit) == sign(logit / T)`` and
    the decision threshold at 0.5 is unchanged.

    Parameters
    ----------
    logits : Tensor
        Pre-sigmoid logits, shape ``(2, *spatial)``.
        Channel 0 = WT, channel 1 = NETC.
    temps : ClassTemperatures
        Per-class temperatures.  Both must be > 0.

    Returns
    -------
    Tensor
        Soft probability map in ``[0, 1]``, same shape as ``logits``.

    Raises
    ------
    SegDerivationError
        If ``logits`` does not have 2 channels or temperatures are non-positive.
    """
    if logits.ndim < 1 or logits.shape[0] != 2:
        raise SegDerivationError(f"expected 2 channels (WT, NETC); got shape {tuple(logits.shape)}")
    if temps.t_wt <= 0.0 or temps.t_netc <= 0.0:
        raise SegDerivationError(
            f"temperatures must be > 0; got T_WT={temps.t_wt}, T_NETC={temps.t_netc}"
        )

    # Build a broadcastable (2, 1, 1, ...) temperature tensor.
    t = logits.new_tensor([temps.t_wt, temps.t_netc])
    for _ in range(logits.ndim - 1):
        t = t.unsqueeze(-1)

    return torch.sigmoid(logits / t)


__all__ = [
    "ClassTemperatures",
    "apply_temperature",
    "fit_temperatures",
]
