"""Round-trip test for the unfrozen-trunk EMA shadow on resume (P3.1).

The project rule in ``.claude/rules/model-coding-standards.md`` flagged the
unfrozen-trunk run as *not* resume-safe: ``trunk_ema`` is built in ``setup()``
from the *original* MAISI trunk, then Lightning's post-setup
``load_state_dict`` is expected to overwrite the shadow with the saved values.
This test verifies that contract.

We don't need to invoke a full Lightning ``Trainer`` — the failure mode would
manifest as the EMA shadow keeping its setup()-time init instead of being
overwritten by the checkpoint payload. That is testable directly on the
``WarmupEMA`` round-trip.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vena.model.fm.ema import WarmupEMA

pytestmark = pytest.mark.unit


class _ToyTrunk(nn.Module):
    """Same shape as the trunk would expose to ``WarmupEMA`` — a few tensors."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.layer = nn.Linear(8, 8)
        self.layer2 = nn.Linear(8, 4)
        for p in self.parameters():
            p.data.fill_(init_value)


def _step_until_warmed(ema: WarmupEMA, n: int = 32) -> None:
    """Bring the EMA past ``update_after_step``."""
    for _ in range(n):
        ema.update()


def test_trunk_ema_load_state_dict_overrides_fresh_init() -> None:
    """A fresh ``WarmupEMA(trunk)`` followed by ``load_state_dict(saved)`` must
    overwrite the shadow with the SAVED weights, not retain the init from the
    fresh trunk. This is the exact contract Lightning relies on when the
    trainer restores model weights after ``setup()``.
    """
    # Train-time state: trunk initialised to 1.0, EMA shadow drifted to known
    # values by mutating the trunk and stepping the EMA.
    train_trunk = _ToyTrunk(init_value=1.0)
    train_ema = WarmupEMA(train_trunk, decay=0.5, update_after_step=0, update_every=1)
    # Mutate the trunk and step the EMA so the shadow has non-trivial values.
    with torch.no_grad():
        for p in train_trunk.parameters():
            p.data.fill_(7.0)
    _step_until_warmed(train_ema, n=8)
    saved_state = {k: v.clone() for k, v in train_ema.state_dict().items()}
    saved_shadow_value = train_ema.ema_model.layer.weight.detach().clone()

    # Resume-time: setup() reconstructs the EMA from the ORIGINAL trunk (init
    # value 99.0 here, distinct from both 1.0 and the post-mutation 7.0). The
    # naive failure mode would be the resumed shadow == 99.0; the correct
    # behaviour is the resumed shadow == saved_shadow_value (≈ a moving
    # average toward 7.0).
    resumed_trunk = _ToyTrunk(init_value=99.0)
    resumed_ema = WarmupEMA(resumed_trunk, decay=0.5, update_after_step=0, update_every=1)
    pre_load = resumed_ema.ema_model.layer.weight.detach().clone()
    resumed_ema.load_state_dict(saved_state)

    # The shadow now matches the saved snapshot, NOT the fresh 99.0 init.
    post_load = resumed_ema.ema_model.layer.weight.detach()
    assert not torch.allclose(pre_load, post_load), "load_state_dict did not change the shadow"
    assert torch.allclose(post_load, saved_shadow_value), (
        "trunk_ema shadow did not match saved values after load_state_dict"
    )


def test_trunk_ema_is_lightning_state_dict_visible() -> None:
    """An ``nn.Module`` that owns a ``WarmupEMA`` submodule must expose the EMA
    shadow's parameters in its ``state_dict()`` so Lightning's checkpoint
    payload carries them. Failure here = ``trunk_ema`` is not registered and
    won't round-trip via ``Trainer.save_checkpoint``.
    """

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = _ToyTrunk(init_value=1.0)
            self.trunk_ema = WarmupEMA(self.trunk, decay=0.5, update_after_step=0, update_every=1)

    w = _Wrapper()
    sd = w.state_dict()
    ema_keys = [k for k in sd if k.startswith("trunk_ema.")]
    assert ema_keys, "trunk_ema submodule did not register any params in state_dict"
    # Sanity: the shadow weights are present (not just buffers).
    assert any("layer.weight" in k for k in ema_keys)


def test_trunk_ema_resume_through_wrapper_state_dict() -> None:
    """End-to-end: a Lightning-style wrapper saves+loads the full ``state_dict``,
    and the trunk_ema shadow round-trips with the rest of the model. This is
    what Lightning actually does post-``setup()``.
    """

    class _Wrapper(nn.Module):
        def __init__(self, init_value: float) -> None:
            super().__init__()
            self.trunk = _ToyTrunk(init_value=init_value)
            self.trunk_ema = WarmupEMA(self.trunk, decay=0.5, update_after_step=0, update_every=1)

    # Train-side: mutate trunk + drive the EMA.
    train_w = _Wrapper(init_value=1.0)
    with torch.no_grad():
        for p in train_w.trunk.parameters():
            p.data.fill_(11.0)
    for _ in range(16):
        train_w.trunk_ema.update()
    saved = {k: v.detach().clone() for k, v in train_w.state_dict().items()}
    saved_ema_shadow = train_w.trunk_ema.ema_model.layer.weight.detach().clone()

    # Resume-side: fresh wrapper with a different init; load saved state.
    resumed_w = _Wrapper(init_value=42.0)
    # Mirror Lightning's flow: setup() rebuilds trunk_ema (already done by
    # __init__ here), THEN load_state_dict is called.
    missing, unexpected = resumed_w.load_state_dict(saved, strict=False)
    assert not unexpected, f"unexpected keys in load: {unexpected}"
    # All trunk_ema params should be in `saved` (no missing trunk_ema.* keys).
    missing_ema = [k for k in missing if k.startswith("trunk_ema.")]
    assert not missing_ema, f"trunk_ema keys missing from saved payload: {missing_ema}"

    post = resumed_w.trunk_ema.ema_model.layer.weight.detach()
    assert torch.allclose(post, saved_ema_shadow), (
        "trunk_ema shadow did not survive the wrapper.load_state_dict round-trip"
    )
