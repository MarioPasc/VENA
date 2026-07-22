"""Frozen Pydantic v2 configuration for the vena.segmentation submodule.

All hyperparameters that cross routine boundaries are declared here.  Nothing
in this file does I/O, builds models, or touches CUDA — it is safe to import
at any time without side effects.

Usage::

    cfg = SegmentationConfig.from_yaml("routines/segmentation/train/configs/default.yaml")
    # cfg is frozen; assignments raise pydantic.ValidationError
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Architecture and checkpoint settings for the segmentation backbone.

    Attributes
    ----------
    name:
        Registered model key.  Must match a name in the model registry.
    feature_size:
        SwinUNETR feature embedding width (powers of 2; default 48 matches
        BrainSegFounder SSL pre-training).
    in_channels:
        Number of input image channels (z-scored; default 3 = T1pre, T2, FLAIR).
    out_channels:
        Number of segmentation output channels.  Default 2 = [WT, NETC] as
        per the iter-8 two-channel soft-mask contract.
    checkpoint:
        Optional path to a pre-trained encoder checkpoint (SSL or fine-tuned).
        ``None`` means train from scratch or rely on MONAI weight init.
    strict_load:
        Whether to load the checkpoint strictly.  ``False`` allows partial
        loading (e.g. encoder-only SSL weights missing decoder parameters).
    deep_supervision:
        Enable auxiliary losses on intermediate decoder heads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["bsf_swinunetr_brats", "bsf_swinunetr_ukb", "segresnet"]
    feature_size: int = 48
    in_channels: int = 3
    out_channels: int = 2
    checkpoint: Path | None = None
    strict_load: bool = False
    deep_supervision: bool = True


class DataConfig(BaseModel):
    """Data-loading and K-fold split configuration.

    Attributes
    ----------
    corpus_registry:
        Path to the corpus registry JSON (same format as the FM DataModule).
        Segmenter folds are a subset of the FM train split — no independent
        partition (leakage vector L2).
    image_h5_root:
        Directory containing per-cohort image-domain H5 files.
    modalities:
        Input modalities consumed in order.  Default = (t1pre, t2, flair)
        matching ``in_channels=3``.
    k_folds:
        Number of cross-validation folds.
    fold_seed:
        RNG seed for the stratified K-fold split.
    patch_size:
        3D spatial patch size fed to the model during training (voxels).
    cache_rate:
        Fraction of the training set to pin in CPU RAM via MONAI
        ``CacheDataset``.  Set to 0.0 to disable caching.
    num_workers:
        DataLoader worker count.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_registry: Path
    image_h5_root: Path
    modalities: tuple[str, ...] = ("t1pre", "t2", "flair")
    k_folds: int = 5
    fold_seed: int = 1337
    patch_size: tuple[int, int, int]
    cache_rate: float
    num_workers: int


class TargetConfig(BaseModel):
    """Soft target generation settings (SDT + per-component operators).

    Attributes
    ----------
    soft:
        Use soft (probability) targets instead of hard binary masks.
    sdt_sigma_vox:
        Gaussian sigma for signed-distance transform softening (voxels).
    netc_operator:
        Distance operator for the NETC component.
        ``"euclidean_percomponent"`` applies per-channel Euclidean SDT.
        ``"geodesic"`` applies geodesic distance (slower, path-aware).
    clip_vox:
        Clip absolute SDT values to this radius (voxels) before softening.
        Prevents extreme gradient magnitudes from large background regions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    soft: bool = True
    sdt_sigma_vox: float = 3.0
    netc_operator: Literal["euclidean_percomponent", "geodesic"] = "euclidean_percomponent"
    clip_vox: float = 10.0


class LossConfig(BaseModel):
    """Loss function configuration.

    Attributes
    ----------
    dice_variant:
        Dice-family loss variant.  ``"dml"`` = deep metric learning dice
        (Eidex 2025); ``"soft_dice"`` = standard soft Dice.
    ce_variant:
        Cross-entropy variant.
    dice_weight:
        Weight for the Dice loss term.
    ce_weight:
        Weight for the cross-entropy loss term.
    tversky_alpha:
        False-negative weight for Tversky variants.
    tversky_beta:
        False-positive weight for Tversky variants.
    deep_supervision_weights:
        Per-scale weights for auxiliary decoder head losses.  Length must
        match the number of auxiliary outputs emitted by the backbone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dice_variant: Literal["dml", "soft_dice", "tversky", "focal_tversky"] = "dml"
    ce_variant: Literal["ce", "focal_ce"] = "ce"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    deep_supervision_weights: tuple[float, ...] = (1.0, 0.5, 0.25)


class TrainConfig(BaseModel):
    """Training-loop hyperparameters.

    Attributes
    ----------
    max_epochs:
        Maximum training epochs per fold.
    lr:
        Peak learning rate for AdamW.
    batch_size:
        Per-GPU batch size.
    optimizer:
        Optimiser name (currently only ``"adamw"`` is supported).
    scheduler:
        LR scheduler (currently only ``"cosine"`` is supported).
    amp:
        Enable PyTorch AMP (bfloat16 on A100, float16 elsewhere).
    val_every_epochs:
        Run validation every N training epochs.
    early_stop_patience:
        Stop early if the selection metric does not improve for this many
        validation events.
    calibration_split_frac:
        Fraction of the training fold held out for post-hoc temperature
        calibration.  Not used for model selection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_epochs: int
    lr: float
    batch_size: int
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["cosine"] = "cosine"
    amp: bool = True
    val_every_epochs: int
    early_stop_patience: int
    calibration_split_frac: float = 0.1


class DerivationConfig(BaseModel):
    """Latent-space mask derivation settings.

    Attributes
    ----------
    temperature:
        Calibration mode applied to the segmenter soft output before
        avg-pooling to the latent grid.
        ``"per_class"`` scales each channel separately;
        ``"global"`` uses a single scalar;
        ``"none"`` skips calibration.
    avg_pool_stride:
        Spatial stride for the average-pool that maps image-space soft masks
        to the latent grid.  The default 4 corresponds to the MAISI VAE 4×
        spatial compression (240→60, 240→60, 155→~40).
    latent_grid:
        Expected spatial dimensions of the output latent mask tensor
        ``(H, W, D)``.  MUST be ``(60, 60, 40)`` — the MAISI-V2 4×
        compression of the ~(240, 240, 155) brain box.  ``(48, 56, 48)`` is
        STALE and must never appear here.
    emit_variance:
        If True, also write the per-voxel predictive variance alongside the
        mean soft mask (requires MC-dropout or ensemble).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: Literal["per_class", "global", "none"] = "per_class"
    avg_pool_stride: int = 4
    latent_grid: tuple[int, int, int] = (60, 60, 40)
    emit_variance: bool = False


class MetricsConfig(BaseModel):
    """Evaluation metric thresholds and selection criterion.

    Attributes
    ----------
    gseg_wt_dice:
        Gate threshold: mean Dice on WT must exceed this value across folds
        for the model to be accepted.
    gseg_netc_dice:
        Gate threshold: mean Dice on NETC must exceed this value.
    selection_metric:
        Criterion used for best-checkpoint selection and early stopping.
        ``"dice"`` uses mean Dice; ``"brier"`` uses Brier score (lower=better);
        ``"dual"`` combines both via a harmonic mean (default).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gseg_wt_dice: float = 0.80
    gseg_netc_dice: float = 0.50
    selection_metric: Literal["dice", "brier", "dual"] = "dual"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class SegmentationConfig(BaseModel):
    """Top-level frozen configuration for the segmentation submodule.

    All fields map one-to-one to top-level YAML keys.  Load via
    :meth:`from_yaml`; the returned instance is frozen (immutable).

    Attributes
    ----------
    model:
        Architecture and checkpoint settings.
    data:
        Data-loading and fold-split settings.
    targets:
        Soft target generation settings.
    loss:
        Loss function configuration.
    train:
        Training-loop hyperparameters.
    derivation:
        Latent-space mask derivation settings.
    metrics:
        Evaluation metric thresholds and selection criterion.
    seed:
        Global RNG seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelConfig
    data: DataConfig
    targets: TargetConfig = TargetConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig
    derivation: DerivationConfig = DerivationConfig()
    metrics: MetricsConfig = MetricsConfig()
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str | Path) -> SegmentationConfig:
        """Load and validate a YAML config file.

        Parameters
        ----------
        path:
            Path to a YAML file whose top-level keys match
            :class:`SegmentationConfig` fields.

        Returns
        -------
        SegmentationConfig
            A frozen, fully-validated configuration instance.

        Raises
        ------
        pydantic.ValidationError
            If any required field is missing, has the wrong type, or an
            unknown key is present (``extra="forbid"``).
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        with path.open("r") as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)


__all__ = [
    "DataConfig",
    "DerivationConfig",
    "LossConfig",
    "MetricsConfig",
    "ModelConfig",
    "SegmentationConfig",
    "TargetConfig",
    "TrainConfig",
]
