"""End-to-end smoke test for the MAISI encode routine.

Runs the smoke YAML against the production image H5 and asserts that:

* the latent H5 is produced and validates,
* both QC figures exist,
* per-modality mean MAE on the roundtrip rows is below a loose ceiling
  (this is a smoke test on 1 patient per WHO grade; the bound is generous
  to keep CI noise low while still catching catastrophic decoder breakage).

Marked ``slow`` + ``gpu`` and skipped when either the image H5 or the
autoencoder checkpoint is missing — so the test is a no-op on CI without
GPU + datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_IMAGE_H5 = Path("/media/mpascual/MeningD2/GLIOMA/UCSF_PDGM/h5/UCSFPDGM_image.h5")
_CKPT = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/MAISI_V2_RM/NV-Generate-MR/models/autoencoder_v2.pt"
)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.preflight_maisi
@pytest.mark.skipif(not _IMAGE_H5.exists(), reason="UCSF-PDGM image H5 not on disk")
@pytest.mark.skipif(not _CKPT.exists(), reason="MAISI checkpoint not on disk")
def test_encode_routine_smoke(tmp_path: Path) -> None:
    import yaml

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from routines.encode.maisi.engine import (
        EncodeMaisiRoutineConfig,
        EncodeMaisiRoutineEngine,
    )

    cfg_dict = {
        "source_image_h5": str(_IMAGE_H5),
        "autoencoder_checkpoint": str(_CKPT),
        "output_dir": str(tmp_path / "encode_maisi"),
        "modalities": ["t1pre", "t1c"],
        "device": "cuda",
        "inference_mode": "sliding",
        "depth_pad_base": 8,
        "limit": 2,
        "overwrite": True,
        "roundtrip": {"enabled": True, "modalities": ["t1c"]},
        "pca": {"enabled": True},
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = EncodeMaisiRoutineConfig.from_yaml(cfg_path)
    engine = EncodeMaisiRoutineEngine(cfg)
    latent_h5 = engine.run()

    assert latent_h5.is_file()
    decision = json.loads((engine.run_dir / "decision.json").read_text())
    assert decision["n_patients_encoded"] == 2
    assert decision["modalities_encoded"] == ["t1pre", "t1c"]

    # Roundtrip and PCA figures present.
    figures = engine.run_dir / "figures"
    assert (figures / "roundtrip.png").is_file()
    assert (figures / "pca.png").is_file()

    rt = decision["roundtrip"]
    assert "per_modality_mean" in rt
    mae = rt["per_modality_mean"]["t1c"]["mae"]
    assert mae < 0.15, f"unexpectedly high t1c MAE on smoke run: {mae}"
