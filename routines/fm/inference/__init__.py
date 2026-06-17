"""Unified validation-inference routine (see plan §3 / approved plan).

Drives every method in the models-YAML registry against every test
patient of every Ring A + Ring B cohort, writes per
(method × cohort × NFE) prediction H5s + per-cohort comparison PNGs,
and emits a self-describing ``decision.json`` summarising provenance.
"""
