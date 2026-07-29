"""Unit tests for the latent_preds.h5 every-N write gate (B12).

The ``_should_write_latent_preds(pass_count, every_n)`` helper gates writes
to once every N cadence passes (1-indexed: passes 1, N+1, 2N+1, …).
"""

from __future__ import annotations

import pytest
from routines.fm.exhaustive_val.engine import _should_write_latent_preds

pytestmark = pytest.mark.unit


class TestShouldWriteLatentPreds:
    """Tests for the every-N gate helper."""

    def test_every_n_zero_always_writes(self) -> None:
        for pass_count in range(1, 10):
            assert _should_write_latent_preds(pass_count, every_n=0)

    def test_every_n_one_always_writes(self) -> None:
        for pass_count in range(1, 10):
            assert _should_write_latent_preds(pass_count, every_n=1)

    def test_every_4_fires_at_1_5_9(self) -> None:
        """With N=4, write at passes 1, 5, 9, … (1-indexed)."""
        write_passes = {p for p in range(1, 14) if _should_write_latent_preds(p, every_n=4)}
        assert write_passes == {1, 5, 9, 13}

    def test_every_4_skips_intermediate_passes(self) -> None:
        skip_passes = {2, 3, 4, 6, 7, 8, 10, 11, 12}
        for p in skip_passes:
            assert not _should_write_latent_preds(p, every_n=4), f"pass {p} should be skipped"

    def test_every_n_equals_pass_count(self) -> None:
        """Pass N with every_n=N should NOT fire (only pass 1 fires in a window starting at 1)."""
        # N=5, pass_count=5: (5-1)%5 = 4 ≠ 0 → no write
        assert not _should_write_latent_preds(5, every_n=5)
        # N=5, pass_count=6: (6-1)%5 = 0 → write
        assert _should_write_latent_preds(6, every_n=5)

    def test_pass_count_1_always_writes_for_any_n(self) -> None:
        """First pass always writes (backward-compat with old job YAMLs)."""
        for n in [1, 2, 4, 5, 10, 20]:
            assert _should_write_latent_preds(1, every_n=n), f"pass 1 should write for every_n={n}"
