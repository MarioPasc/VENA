"""S1 v3 ControlNet regression tests — ``init_from_trunk_enabled`` flag.

Variant B builds the ControlNet with ``conditioning_in_channels=3`` (the
three NETC/ED/ET soft masks) and ``init_from_trunk_enabled=False``: the
encoder is no longer warm-started from the trunk because the cond_embedding's
input distribution is now masks, not modality latents.

These tests verify the surgical contract:

1. ``cond_embedding`` first conv has the requested input-channel count.
2. ``init_from_trunk_enabled=False`` ⇒ a subsequent
   :meth:`MaisiControlNet.init_from_trunk` call is a no-op and the encoder
   weights are unchanged.
3. ``zero_init_output_projections`` still applies (independent of the flag).
4. ``init_from_trunk_enabled=True`` ⇒ existing behaviour: matching keys are
   copied from the supplied trunk state_dict.

CPU-only; no MAISI checkpoint needed.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from vena.model.fm.controlnet.maisi_controlnet import MaisiControlNet


def test_cond_embedding_in_channels_matches_constructor_arg() -> None:
    """Conditioning_in_channels=3 ⇒ cond_embedding.0 has in_channels=3."""
    cn = MaisiControlNet(conditioning_in_channels=3)
    first_conv = cn.net.controlnet_cond_embedding.conv_in
    assert first_conv.in_channels == 3, f"expected in_channels=3, got {first_conv.in_channels}"


def test_init_from_trunk_disabled_is_noop() -> None:
    """With the flag off, init_from_trunk must not change encoder weights."""
    cn = MaisiControlNet(conditioning_in_channels=3, init_from_trunk_enabled=False)
    # Snapshot encoder params before the (no-op) call.
    snap = {n: p.detach().clone() for n, p in cn.net.named_parameters()}
    # A non-empty fake trunk state_dict with one shape-matching key. MONAI
    # wraps Conv3d inside ``Convolution``; the parameter key path is
    # ``conv_in.conv.weight``.
    own_sd = cn.net.state_dict()
    fake_trunk_sd = {"conv_in.conv.weight": torch.ones_like(own_sd["conv_in.conv.weight"])}
    cn.init_from_trunk(fake_trunk_sd)
    for name, p in cn.net.named_parameters():
        torch.testing.assert_close(p, snap[name])


def test_init_from_trunk_enabled_copies_matching_keys() -> None:
    """Default flag ⇒ shape-matching keys get copied verbatim."""
    cn = MaisiControlNet(conditioning_in_channels=3, init_from_trunk_enabled=True)
    own_sd = cn.net.state_dict()
    # Forge a trunk state_dict containing a recognisable conv_in pattern.
    sentinel = torch.full_like(own_sd["conv_in.conv.weight"], 7.0)
    fake_trunk_sd = {"conv_in.conv.weight": sentinel}
    cn.init_from_trunk(fake_trunk_sd)
    torch.testing.assert_close(cn.net.state_dict()["conv_in.conv.weight"], sentinel)


def test_zero_init_output_projections_applies_under_both_flags() -> None:
    """zero_init_output_projections must zero the controlnet_* projections."""
    for flag in (True, False):
        cn = MaisiControlNet(conditioning_in_channels=3, init_from_trunk_enabled=flag)
        # Perturb to ensure zero_init_output_projections does the work.
        for _, p in cn.net.named_parameters():
            with torch.no_grad():
                p.add_(0.5)
        cn.zero_init_output_projections()
        for name, p in cn.net.named_parameters():
            if name.startswith("controlnet_down_blocks.") or name.startswith(
                "controlnet_mid_block."
            ):
                assert torch.all(p == 0.0), f"flag={flag}: non-zero output projection {name}"


def test_init_from_trunk_default_flag_is_true() -> None:
    """Back-compat: existing callers omit the flag and get the legacy path."""
    cn = MaisiControlNet(conditioning_in_channels=13)
    assert cn.init_from_trunk_enabled is True


def test_variant_b_three_channel_mask_forward_shape() -> None:
    """Forward pass with the v3 Variant B layout — 3-channel mask conditioning.

    Builds a 3-channel cond tensor, runs forward on a 4-channel noisy latent
    (the latent input ``x`` to the ControlNet is always 4-channel, regardless
    of the trunk's expanded conv_in). Verifies the returned residuals are
    non-empty tensors and the mid-block residual has the expected rank.
    """
    cn = MaisiControlNet(conditioning_in_channels=3, init_from_trunk_enabled=False)
    cn.eval()
    B = 1
    h, w, d = 8, 8, 8  # multiple of 8 — trunk requires
    x = torch.zeros(B, 4, h, w, d)
    cond = torch.zeros(B, 3, h, w, d)
    timesteps = torch.tensor([100], dtype=torch.long)
    class_labels = torch.tensor([9], dtype=torch.long)
    with torch.no_grad():
        down_res, mid_res = cn(
            x=x, timesteps=timesteps, controlnet_cond=cond, class_labels=class_labels
        )
    assert isinstance(down_res, list) and len(down_res) > 0
    assert mid_res.ndim == 5
    # With zero-init outputs and zero input, residuals are exactly zero.
    for t in down_res:
        assert torch.all(t == 0.0)
    assert torch.all(mid_res == 0.0)
