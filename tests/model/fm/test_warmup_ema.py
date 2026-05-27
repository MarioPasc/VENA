"""Unit tests for the WarmupEMA wrapper."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vena.model.fm.ema import WarmupEMA


@pytest.mark.unit
def test_warmup_ema_constructs_and_decay_curves_upward() -> None:
    model = nn.Linear(4, 4)
    ema = WarmupEMA(model, decay=0.9999, inv_gamma=10.0, power=1.0)
    early = ema.get_current_decay()
    for _ in range(50):
        ema.update()
    later = ema.get_current_decay()
    assert later >= early
    assert later <= 0.9999


@pytest.mark.unit
def test_warmup_ema_state_dict_roundtrip() -> None:
    model_a = nn.Linear(4, 4)
    model_b = nn.Linear(4, 4)
    ema_a = WarmupEMA(model_a, decay=0.999)
    for _ in range(5):
        ema_a.update()
    sd = ema_a.state_dict()
    ema_b = WarmupEMA(model_b, decay=0.999)
    ema_b.load_state_dict(sd)
    # After load, ema_b's shadow should equal ema_a's shadow.
    for (na, pa), (nb, pb) in zip(
        ema_a.ema_model.named_parameters(), ema_b.ema_model.named_parameters()
    ):
        assert na == nb
        assert torch.allclose(pa, pb)


@pytest.mark.unit
def test_warmup_ema_shadow_tracks_model_over_time() -> None:
    model = nn.Linear(8, 8)
    ema = WarmupEMA(model, decay=0.5, inv_gamma=0.001, power=1.0)
    # Saturate the decay schedule quickly so updates are meaningful.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
    ema.update()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    for _ in range(200):
        ema.update()
    # Shadow should be pulled toward 1.0.
    p_shadow = next(ema.ema_model.parameters())
    assert p_shadow.mean().item() > 0.5
