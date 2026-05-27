"""Unit test for GradNormLogger."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vena.model.fm.lightning.callbacks.grad_norm import GradNormLogger


@pytest.mark.unit
def test_grad_norm_logs_post_clip_norm_of_controlnet_only() -> None:
    cn = nn.Linear(4, 4)
    other = nn.Linear(4, 4)
    # Populate gradients.
    for p in cn.parameters():
        p.grad = torch.ones_like(p)
    for p in other.parameters():
        p.grad = torch.full_like(p, 100.0)
    cb = GradNormLogger(attr_name="controlnet")
    module = MagicMock()
    module.controlnet = cn
    module.other = other
    module.device = torch.device("cpu")
    cb.on_before_optimizer_step(MagicMock(), module, MagicMock())
    # The logger must have been called with the cn-only norm.
    assert module.log.called
    args, kwargs = module.log.call_args
    assert args[0] == "train/grad_norm_cn"
    # cn has 4*4 + 4 = 20 params, all-ones grad → norm = sqrt(20)
    assert abs(args[1].item() - (20 ** 0.5)) < 1e-5
