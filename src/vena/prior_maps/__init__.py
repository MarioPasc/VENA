"""Conditioning-channel extraction subsystems for the VENA latent flow trunk.

Each submodule produces a small set of spatial conditioning channels from a
single non-contrast sequence (or a derived quantitative map):

* :mod:`vena.prior_maps.vessel_priors` — SWI/SWAN → vessel soft mask
  (retained but not currently incorporated into the FM trunk).
* :mod:`vena.prior_maps.perfusion_priors` — ASL → ``cbf_rel, cbf`` (Term I).
* :mod:`vena.prior_maps.cellularity_priors` — ADC → ``adc_rel, cell`` (Term II
  intra-tumour proxy).
* :mod:`vena.prior_maps.susceptibility_priors` — SWAN magnitude →
  ``sus, itss`` (Term I slow-flow venous + Term II proxy; sub-option A).

The design rationale and channel definitions live in
``/media/mpascual/Sandisk2TB/research/vena/docs/soft_priors_sources.md``.
"""
