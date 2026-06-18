"""Unit tests for :class:`vena.model.fm.lpl.feature_stats.FeatureStatsEMA`.

Covers:

* First-batch bootstrap (decay=0 for the first ``update``).
* EMA convergence on a stationary stream (mean, var → true).
* ``standardise`` output has mean ≈ 0, var ≈ 1 after warmup.
* state_dict / load_state_dict round-trip preserves the running stats.
* ``is_warmed_up`` reflects ``n_updates``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vena.model.fm.lpl import FeatureStatsEMA

pytestmark = pytest.mark.unit


def test_first_update_bootstraps_from_batch() -> None:
    """First call uses decay=0 — buffers equal the batch statistics, NOT a
    mix with the zero-init.
    """
    stats = FeatureStatsEMA(channels={2: 4}, decay=0.9)
    # Distinct per-channel means so the first-batch fingerprint is clear.
    t = torch.tensor(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
            ]
        ]
    ).unsqueeze(-1)  # (1, 4, 1, 2, 1)
    stats.update({2: t})
    assert torch.allclose(stats.mean(2), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # First batch has 2 spatial samples per channel and identical values →
    # var should be 0.
    assert torch.allclose(stats.var(2), torch.zeros(4))
    assert int(stats.n_updates.item()) == 1


def test_subsequent_updates_blend_with_decay() -> None:
    """Second update applies the configured decay."""
    stats = FeatureStatsEMA(channels={0: 1}, decay=0.5)
    one = torch.ones(1, 1, 1, 1, 1)
    five = torch.full((1, 1, 1, 1, 1), 5.0)
    stats.update({0: one})  # bootstrap → mean=1
    stats.update({0: five})  # 0.5 * 1 + 0.5 * 5 = 3
    assert stats.mean(0).item() == pytest.approx(3.0)


def test_standardise_zero_mean_unit_var_after_warmup() -> None:
    """After running 32 updates over a stationary stream, ``standardise``
    output has mean ≈ 0 and var ≈ 1 within numerical tolerance.
    """
    torch.manual_seed(0)
    stats = FeatureStatsEMA(channels={5: 8}, decay=0.9)
    true_mean = torch.linspace(-2.0, 2.0, 8)
    true_std = torch.linspace(0.5, 2.0, 8)
    for _ in range(64):
        z = torch.randn(2, 8, 4, 4, 4)
        feat = z * true_std.view(1, 8, 1, 1, 1) + true_mean.view(1, 8, 1, 1, 1)
        stats.update({5: feat})

    # One last fresh sample — standardise it.
    z = torch.randn(2, 8, 4, 4, 4)
    feat = z * true_std.view(1, 8, 1, 1, 1) + true_mean.view(1, 8, 1, 1, 1)
    std_feat = stats.standardise(feat, block_idx=5)
    # Per-channel.
    chan_axis = 1
    means = std_feat.movedim(chan_axis, -1).reshape(-1, 8).mean(dim=0)
    vars_ = std_feat.movedim(chan_axis, -1).reshape(-1, 8).var(dim=0, unbiased=False)
    assert torch.allclose(means, torch.zeros(8), atol=0.4)
    assert torch.allclose(vars_, torch.ones(8), atol=0.4)


def test_state_dict_round_trip() -> None:
    """Buffers survive ``state_dict`` / ``load_state_dict``."""
    stats_src = FeatureStatsEMA(channels={2: 4, 5: 8}, decay=0.9)
    for _ in range(8):
        stats_src.update({2: torch.randn(1, 4, 2, 2, 2), 5: torch.randn(1, 8, 2, 2, 2)})
    saved = {k: v.clone() for k, v in stats_src.state_dict().items()}

    stats_dst = FeatureStatsEMA(channels={2: 4, 5: 8}, decay=0.9)
    stats_dst.load_state_dict(saved)
    assert torch.equal(stats_dst.mean(2), stats_src.mean(2))
    assert torch.equal(stats_dst.mean(5), stats_src.mean(5))
    assert torch.equal(stats_dst.var(2), stats_src.var(2))
    assert torch.equal(stats_dst.var(5), stats_src.var(5))
    assert int(stats_dst.n_updates.item()) == int(stats_src.n_updates.item())


def test_is_warmed_up() -> None:
    stats = FeatureStatsEMA(channels={0: 1})
    assert not stats.is_warmed_up(min_samples=1)
    for _ in range(3):
        stats.update({0: torch.randn(1, 1, 2, 2, 2)})
    assert stats.is_warmed_up(min_samples=3)
    assert not stats.is_warmed_up(min_samples=99)


def test_buffers_round_trip_through_nn_module_wrapper() -> None:
    """When held inside a wrapper :class:`nn.Module`, the EMA buffers
    appear under a ``stats.*`` prefix in the wrapper's state_dict and
    round-trip through ``Wrapper.load_state_dict``.
    """

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stats = FeatureStatsEMA(channels={2: 3})

    w_src = _Wrap()
    w_src.stats.update(
        {2: torch.tensor([[1.0, 2.0, 3.0]]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)}
    )
    saved = w_src.state_dict()
    assert any(k.startswith("stats.") for k in saved)

    w_dst = _Wrap()
    w_dst.load_state_dict(saved)
    assert torch.equal(w_dst.stats.mean(2), w_src.stats.mean(2))


def test_unknown_block_raises_on_standardise() -> None:
    stats = FeatureStatsEMA(channels={2: 4})
    with pytest.raises(KeyError):
        stats.standardise(torch.randn(1, 4, 2, 2, 2), block_idx=99)


def test_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay"):
        FeatureStatsEMA(channels={0: 1}, decay=1.0)


def test_empty_channels_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FeatureStatsEMA(channels={})


def test_nonpositive_channel_raises() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        FeatureStatsEMA(channels={0: 0})
