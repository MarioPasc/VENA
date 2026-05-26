"""Regression test: ASL was added to the UCSF-PDGM modality registry."""

from __future__ import annotations

import typing

from vena.data.niigz.ucsf_pdgm import _MODALITY_SUFFIX, Modality


def test_asl_in_modality_literal() -> None:
    assert "ASL" in typing.get_args(Modality)


def test_asl_in_suffix_dict() -> None:
    assert _MODALITY_SUFFIX["ASL"] == "ASL"
