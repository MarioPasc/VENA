"""Decoder feature-extractor context manager.

Wraps the two-step :func:`vena.common.partial_decode` invocation
(``handle.model.post_quant_conv`` then ``decoder.blocks[0..K]`` with hooks)
into a single context manager that consumers can use without reaching into
the MAISI internals. The context manager is the only place in the LPL
machinery that touches ``handle.model.decoder.blocks`` directly — every
other module in :mod:`vena.model.fm.lpl` goes through :func:`partial_decode`
via this wrapper.

Per ``.claude/rules/extensibility.md`` (``vena.common`` is the canonical
adapter surface), cross-module access to the MAISI decoder must route
through :mod:`vena.common`. The single allowed reach into
``handle.model.decoder.blocks`` is the forward-hook registration inside
:func:`vena.common.partial_decode`; the assembly of the
``post_quant_conv → blocks`` chain lives here, in :mod:`vena.model.fm.lpl`,
because the post_quant_conv → partial-decode sequence is FM-specific
plumbing rather than a primitive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextlib import nullcontext as _nullcontext

import torch

from vena.common import AutoencoderHandle, partial_decode


@contextmanager
def decoder_feature_extractor(
    handle: AutoencoderHandle,
    *,
    blocks: set[int] | frozenset[int],
    max_block: int,
    grad_checkpoint: bool = False,
) -> Iterator[Callable[[torch.Tensor], dict[int, torch.Tensor]]]:
    """Yield an ``extract(latent) -> {block_idx: features}`` closure.

    Parameters
    ----------
    handle : AutoencoderHandle
        The frozen-VAE handle returned by ``load_autoencoder``. The
        decoder accessed via ``handle.model.decoder``.
    blocks : set[int]
        Indices to capture (must be ⊆ ``[0, max_block]``).
    max_block : int
        Last decoder block to run.
    grad_checkpoint : bool, default False
        Forward-time activation memory mitigation. Forwarded to
        :func:`vena.common.partial_decode`.

    Yields
    ------
    Callable[[torch.Tensor], dict[int, torch.Tensor]]
        A closure that takes a *raw* latent of shape ``(B, C_latent,
        h, w, d)`` (the dataloader's standard format) and returns the
        captured features. Each call runs ``post_quant_conv`` then the
        truncated decoder block sequence; the dict is fresh per call.

    Notes
    -----
    The context manager does not currently hold any reusable resources
    (the hooks are registered + removed inside ``partial_decode``), so
    ``__exit__`` is a pure no-op. The interface is kept context-manager
    shaped so a future grad-checkpoint warm-up cache or a cuda-graph
    pinning layer can slot in without touching call sites.
    """
    decoder = handle.model.decoder
    post_quant = handle.model.post_quant_conv
    blocks_frozen = frozenset(blocks)

    def _extract(latent: torch.Tensor) -> dict[int, torch.Tensor]:
        # The MAISI VAE-GAN was trained with norm_float16=True; the
        # internal convs accept either fp16 or fp32 input under
        # ``torch.autocast(device_type="cuda", dtype=float16)``. Without
        # autocast a float32 latent vs a half-buffer Conv hits
        # "Input type (Half) and bias type (float) should be the same".
        use_autocast = latent.is_cuda
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_autocast
            else _nullcontext()
        )
        with ctx:
            z_post = post_quant(latent)
            return partial_decode(
                decoder,
                z_post,
                blocks=blocks_frozen,
                max_block=max_block,
                grad_checkpoint=grad_checkpoint,
            )

    try:
        yield _extract
    finally:
        # No persistent state to release — hooks are per-call inside
        # ``partial_decode``. The empty ``finally`` is intentional and
        # keeps the contract symmetric for future resource attach.
        pass
