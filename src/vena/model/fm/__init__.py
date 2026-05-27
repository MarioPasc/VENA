"""Flow-matching model package for VENA.

Submodules:
    maisi       — frozen MAISI-V2 rectified-flow trunk (warm-started U-Net).
    controlnet  — trainable ControlNet branch + conditioning + downsamplers + losses.
    sampler     — rectified-flow scheduler primitives (noising, target velocity).
    lightning   — PyTorch Lightning glue (DataModule + LightningModule).
"""
