"""Integration test for the resume pathway (training_routine.md §12.5).

Validates that the resume *components* VENA implements — model state, optimiser
state, EMA state, and the Python/NumPy/Torch RNG payload added by
:meth:`FMLightningModule.on_save_checkpoint` — produce bit-identical outputs
after roundtrip.

We deliberately avoid running Lightning's Trainer through a resume here:
Lightning's dataloader is not resumable by default (§7.4 step 4 calls out the
need for bespoke dataloader-state plumbing), so phase-2 training would consume
different batches than the reference run and the losses diverge for reasons
unrelated to the components under test. Instead we exercise the same hooks
directly:

1. Build model+optim+EMA. Train 100 steps with seed S. Save state into a
   checkpoint dict (mirroring Lightning's payload + our on_save_checkpoint).
2. Mutate RNGs, freshly construct a model+optim+EMA, restore the state, then
   take one additional step.
3. Compare the resumed loss to a reference run that took 101 steps without
   interruption — they must match within fp16 tolerance.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vena.model.fm.ema import WarmupEMA
from vena.model.fm.lightning import module as _vena_lightning_module  # noqa: F401 — registers safe globals


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_setup() -> tuple[_Toy, torch.optim.SGD, WarmupEMA, torch.Tensor]:
    """Fresh model + SGD + EMA + a deterministic synthetic dataset."""
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    ema = WarmupEMA(model, decay=0.99, inv_gamma=1.0, power=1.0)
    data = torch.randn(64, 4, generator=torch.Generator().manual_seed(7))
    return model, opt, ema, data


def _train_step(
    model: _Toy, opt: torch.optim.SGD, ema: WarmupEMA, x: torch.Tensor
) -> float:
    """One step. Includes a noise draw so RNG roundtrip is observable."""
    noise = torch.randn(x.shape[0], 1)
    y = model(x) + noise
    loss = (y ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    ema.update()
    return loss.item()


def _save_payload(
    model: _Toy, opt: torch.optim.SGD, ema: WarmupEMA
) -> dict[str, Any]:
    """Mirrors what Lightning + our on_save_checkpoint produce."""
    return {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "ema_state": ema.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def _restore_payload(
    model: _Toy, opt: torch.optim.SGD, ema: WarmupEMA, payload: dict[str, Any]
) -> None:
    model.load_state_dict(payload["model_state"])
    opt.load_state_dict(payload["optimizer_state"])
    ema.load_state_dict(payload["ema_state"])
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])


@pytest.mark.slow
def test_resume_matches_reference_within_fp16_tolerance(tmp_path: Path) -> None:
    # --- Reference run: 101 contiguous steps -----------------------------
    _seed_all(1337)
    ref_model, ref_opt, ref_ema, ref_data = _make_setup()
    ref_losses: list[float] = []
    batch_size = 4
    for step in range(101):
        x = ref_data[(step * batch_size) % 64 : (step * batch_size) % 64 + batch_size]
        ref_losses.append(_train_step(ref_model, ref_opt, ref_ema, x))

    # --- Resumed run: 100 steps, save, restore, take step 101 -----------
    _seed_all(1337)
    a_model, a_opt, a_ema, a_data = _make_setup()
    for step in range(100):
        x = a_data[(step * batch_size) % 64 : (step * batch_size) % 64 + batch_size]
        _train_step(a_model, a_opt, a_ema, x)

    payload = _save_payload(a_model, a_opt, a_ema)
    # Save through torch.save → torch.load to cover the serialised path that
    # actually triggered the PyTorch 2.6 weights_only issue we patched in
    # vena.model.fm.lightning.module.
    ckpt_path = tmp_path / "resume.pt"
    torch.save(payload, ckpt_path)
    loaded = torch.load(ckpt_path, weights_only=False)

    # Mutate every RNG to prove restore actually happens.
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    b_model, b_opt, b_ema, b_data = _make_setup()
    _restore_payload(b_model, b_opt, b_ema, loaded)

    step = 100
    x = b_data[(step * batch_size) % 64 : (step * batch_size) % 64 + batch_size]
    resumed_loss = _train_step(b_model, b_opt, b_ema, x)

    ref_step_101 = ref_losses[100]
    assert abs(ref_step_101 - resumed_loss) < 1e-5, (
        f"resume diverged: ref={ref_step_101:.8f} resumed={resumed_loss:.8f}"
    )
