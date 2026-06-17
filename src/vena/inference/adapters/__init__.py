"""Adapter implementations for every benchmarked method.

Adapter modules are **lazy-loaded**: they are imported on demand by
:func:`vena.inference.registry.get_inference_factory` when the
corresponding ``model_type`` is requested. A process that only runs one
competitor family (e.g. SynDiff in the ``vena-syndiff`` env) therefore
never imports the latent-tier (``vena_fm``, ``t1c_rflow``, ``dit_3d``,
``lddpm_3d``, ``lpix2pix_3d``) adapters and never pulls their
transitive deps (MAISI, Lightning, sklearn, pandas, ...).

Each adapter module declares its registry key with
``@register_inference_model("...")`` at module scope; the decorator
fires when ``importlib.import_module(<adapter_module>)`` runs inside
the factory, populating the live registry on first use.

* ``identity_adapter`` — C0 baseline (T1pre passes through).
* ``pgan_adapter`` — C1 (2D axial GAN).
* ``resvit_adapter`` — C2 (2D axial residual ViT).
* ``syndiff_adapter`` — C3 (2D axial DDPM with adversarial bridge).
* ``dit3d_adapter`` — C4 (3D latent DiT).
* ``t1c_rflow_adapter`` — C5 (3D latent RFlow).
* ``lddpm3d_adapter`` — C6 (3D latent DDPM).
* ``lpix2pix3d_adapter`` — C7 (3D latent Pix2Pix).
* ``vena_fm_adapter`` — A1 (VENA-S1 CFM-only) + VENA (S2 + L^p contrastive).
"""

from __future__ import annotations

# NOTE: do NOT eager-import the adapter modules here. See the docstring above.
