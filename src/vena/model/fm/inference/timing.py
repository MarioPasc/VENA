"""CUDA-synchronised timing probe for NFE benchmarking (training_routine §5).

Usage::

    probe = NFETimingProbe()
    with probe.section("trunk"):
        ... trunk forward ...
    with probe.section("controlnet"):
        ... controlnet forward ...
    timings = probe.aggregate()  # dict[str, float] seconds per section, mean

The first patient's measurements should be discarded by the caller — the
probe itself does not implement the warm-up policy, only the timing pathway.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class NFETimingProbe:
    """Tracks wall-clock per labelled section, CUDA-synchronised when available."""

    use_cuda_sync: bool = True
    _t: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def section(self, name: str):
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._t.setdefault(name, []).append(dt)

    def aggregate(self, drop_first: bool = True) -> dict[str, dict[str, float]]:
        """Return per-section aggregates after dropping the warm-up entry.

        Parameters
        ----------
        drop_first : bool
            If True, discard the first timing per section (CUDA kernel warm-up).

        Returns
        -------
        dict
            Mapping ``section -> {"mean": float, "std": float, "n": int}``
            (seconds).
        """
        out: dict[str, dict[str, float]] = {}
        for name, entries in self._t.items():
            xs = entries[1:] if drop_first and len(entries) > 1 else entries
            if not xs:
                out[name] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            n = len(xs)
            mean = sum(xs) / n
            var = sum((x - mean) ** 2 for x in xs) / max(1, n - 1)
            out[name] = {"mean": mean, "std": var**0.5, "n": n}
        return out

    def reset(self) -> None:
        self._t.clear()
