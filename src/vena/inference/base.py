"""Inference ABC and immutable result container.

Every benchmarked method exposes the same lifecycle:

1. :meth:`InferenceModel.setup` — load checkpoint(s), build the model on
   the configured device. Idempotent (a re-call is a no-op).
2. :meth:`InferenceModel.predict` — produce one harmonised T1c volume
   per patient × NFE. Timing and peak VRAM are captured CUDA-synced
   inside the call.
3. :meth:`InferenceModel.teardown` — free the GPU memory before the
   next method runs (the engine runs methods sequentially on one GPU).

The :class:`InferenceResult` is a frozen dataclass — the engine bundles
many of these into the per-cohort H5.

Conventions
-----------
* ``t1c_synthetic_harmonised`` is float32 in ``[0, 1]`` over the brain
  mask (exterior forced to 0) — the volume that lands in
  ``/predictions/t1c_synthetic_harmonised`` per validation §5.3.
* ``t1c_synthetic_raw`` is the method-native output before
  harmonisation. For the latent-tier (C4..C7, A1, VENA) this is already
  the decoded ``[0, 1]`` volume from ``vena.common.decode.decode_box``
  (so for those methods raw == harmonised); for the 2D image-tier
  (C1..C3) it is the slice-stacked output before percentile
  normalisation; for C0 it is the raw ``images/t1pre`` volume.
* ``inference_seconds`` is CUDA-synced wall-clock for the *whole*
  ``predict()`` body — encode, sample, decode, harmonise — per
  validation §5.2.
* ``peak_vram_mb`` is ``torch.cuda.max_memory_allocated(device) / 1024**2``
  read at the end of ``predict()``. The engine resets it before each
  call so the number is per-prediction, not cumulative.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vena.data.registry import CohortEntry


@dataclass(frozen=True)
class InferenceResult:
    """One predicted T1c volume + timing + VRAM."""

    t1c_synthetic_harmonised: torch.Tensor
    t1c_synthetic_raw: torch.Tensor
    inference_seconds: float
    peak_vram_mb: float


class InferenceModelError(Exception):
    """Raised on adapter setup / predict failures.

    Adapters subclass this if they want a per-family error type
    (``Pix2PixSetupError`` etc.); the engine catches the base class.
    """


class InferenceModel(ABC):
    """Abstract base class every benchmark method satisfies.

    Concrete adapters declare a class attribute ``model_type`` (the
    string used as the registry key) and override ``setup``,
    ``predict``, and ``teardown``. The constructor stores ``name``
    (the registry entry's friendly name), ``device``, ``nfe_list``,
    and ``selection_nfe``; subclasses extend the kwargs as needed.

    Parameters
    ----------
    name
        Friendly tag from the YAML registry (e.g. ``"VENA-S2-FFT"``).
        Used in log lines and as the H5 sub-directory name.
    device
        The CUDA device to load the model onto. The engine guarantees
        that only one adapter occupies the device at a time.
    nfe_list
        Forward NFEs to compute. Always non-empty; ``(1,)`` for
        non-iterative methods (C0, C1, C2, C7).
    selection_nfe
        The single NFE used by the cross-method comparison figure
        (validation §5.1 selection-NFE per family).
    """

    model_type: str = "abstract"

    def __init__(
        self,
        *,
        name: str,
        device: torch.device | str,
        nfe_list: tuple[int, ...] = (1,),
        selection_nfe: int = 1,
    ) -> None:
        if not nfe_list:
            raise InferenceModelError(f"adapter '{name}': nfe_list must be non-empty")
        if selection_nfe not in nfe_list:
            raise InferenceModelError(
                f"adapter '{name}': selection_nfe={selection_nfe} not in nfe_list={nfe_list}"
            )
        self.name = name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.nfe_list = tuple(int(n) for n in nfe_list)
        self.selection_nfe = int(selection_nfe)
        self._is_setup = False

    # ------------------------------------------------------------------ lifecycle

    @abstractmethod
    def setup(self) -> None:
        """Load checkpoints and instantiate the model on ``self.device``.

        Adapters must call ``super().setup()`` *after* their own setup
        finishes, which sets ``self._is_setup = True``.
        """
        self._is_setup = True

    @abstractmethod
    def predict(
        self,
        cohort: CohortEntry,
        patient_id: str,
        nfe: int,
    ) -> InferenceResult:
        """Generate one harmonised T1c volume for one patient at one NFE.

        Parameters
        ----------
        cohort
            The cohort entry from the corpus registry; carries the
            ``image_h5`` and ``latent_h5`` paths the adapter needs.
        patient_id
            ID matching the cohort's ``/ids`` dataset (image H5) or
            ``patients/keys`` (latent H5) — equality is enforced by the
            converter pipeline.
        nfe
            Number of function evaluations. For non-iterative methods
            the adapter receives ``1`` and ignores the value.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Free GPU memory and detach checkpoints.

        Adapters must set ``self._is_setup = False`` at the end.
        """

    # ------------------------------------------------------------------ helpers

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise InferenceModelError(f"adapter '{self.name}': predict() called before setup()")

    @staticmethod
    def _sync(device: torch.device) -> None:
        """Synchronise the CUDA stream for accurate wall-clock timing."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _peak_vram_mb(device: torch.device) -> float:
        """Return ``torch.cuda.max_memory_allocated`` for ``device`` in MB."""
        if device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)

    @staticmethod
    def _reset_peak_vram(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, device={self.device}, "
            f"nfe_list={self.nfe_list}, selection_nfe={self.selection_nfe})"
        )


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a user-facing device string against the visible CUDA devices.

    Replicates the small helper in
    ``routines.fm.exhaustive_val.engine.ExhaustiveValEngine._resolve_device``
    so adapters and the engine share the same fallback policy.
    """
    if isinstance(device, torch.device):
        return device
    if not torch.cuda.is_available():
        return torch.device("cpu")
    idx = int(device.split(":")[1]) if ":" in device else 0
    if idx >= torch.cuda.device_count():
        idx = 0
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def resolve_path(maybe_str: str | Path | None) -> Path | None:
    """Coerce a YAML-side path (possibly ``None``) to an absolute ``Path``."""
    if maybe_str is None:
        return None
    return Path(maybe_str).expanduser().resolve()
