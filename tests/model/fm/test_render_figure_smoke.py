"""Smoke test for the 2026-06-20 ``render_comparison_figure`` overhaul.

The figure refactor changes the public signature (drops ``mean_ssim``,
adds ``psnr_ssim_by_nfe``) and applies several visual conventions (black
background, SSIM-sorted NFE rows, per-NFE annotation, per-slice intensity
match). These tests verify the file is produced and the signature change
is honoured; visual correctness is checked manually on the smoke run.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.eval.exhaustive import render_comparison_figure


@pytest.mark.unit
def test_render_figure_produces_file(tmp_path) -> None:
    real = torch.rand(20, 20, 12)
    synth_by_nfe = {1: torch.rand(20, 20, 12), 5: torch.rand(20, 20, 12)}
    time_by_nfe = {1: 0.1, 5: 0.5}
    psnr_ssim_by_nfe = {1: (20.0, 0.5), 5: (25.0, 0.7)}
    slices = [3, 6, 9]

    out_path = tmp_path / "figure_best_1.png"
    result = render_comparison_figure(
        real,
        synth_by_nfe,
        time_by_nfe,
        slices,
        patient_id="UCSF-PDGM-0001",
        title_tag="best_1",
        out_path=out_path,
        psnr_ssim_by_nfe=psnr_ssim_by_nfe,
    )
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.unit
def test_render_figure_empty_psnr_ssim_does_not_crash(tmp_path) -> None:
    """Degenerate-cohort path: every metric is nan, but the figure renders."""
    real = torch.rand(16, 16, 8)
    synth_by_nfe = {1: torch.rand(16, 16, 8)}
    out_path = tmp_path / "figure_worst_1.png"
    render_comparison_figure(
        real,
        synth_by_nfe,
        time_by_nfe={1: 0.2},
        slice_indices=[2, 4],
        patient_id="X",
        title_tag="worst_1",
        out_path=out_path,
        psnr_ssim_by_nfe={},
    )
    assert out_path.exists()


@pytest.mark.unit
def test_render_figure_signature_dropped_mean_ssim() -> None:
    """Guard against accidental reintroduction of the ``mean_ssim`` kwarg."""
    import inspect

    sig = inspect.signature(render_comparison_figure)
    assert "mean_ssim" not in sig.parameters
    assert "psnr_ssim_by_nfe" in sig.parameters
