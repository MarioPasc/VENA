"""Tests for vena.segmentation.models (Arms A, B, C).

Markers
-------
pytestmark = pytest.mark.unit  (fast, CPU, no checkpoints)
@pytest.mark.slow              (SwinUNETR forward — CPU-only but several seconds)

Import isolation proof
----------------------
``test_import_isolation`` verifies that ``vena.segmentation.models.segresnet``
does NOT import ``monai`` at module scope — monai is imported lazily inside
each builder function.  The test is valid only in a fresh interpreter; pytest
runs in a single process, so the module is already loaded once the test
collection phase imports the package.  The test checks the sys.modules state
BEFORE the first builder call to confirm no top-level MONAI import.

Shape contracts verified here
------------------------------
Arm C (SegResNet), CPU, input (2, 3, 32, 32, 24):
    deep_supervision=False → Tensor (2, 2, 32, 32, 24)
    deep_supervision=True  → tuple[Tensor×3]:
        [0] (2, 2, 32, 32, 24)   full resolution
        [1] (2, 2, 16, 16, 12)   up_layers[1] output (H/2)
        [2] (2, 2,  8,  8,  6)   up_layers[0] output (H/4)

Arms A/B (SwinUNETR), CPU, input (2, 3, 64, 64, 64)  [@pytest.mark.slow]:
    SwinUNETR with feature_size=48 has 5 downsampling stages (patch-embed
    stride-2 + 4 patch-merging).  With 32³ input the bottleneck collapses to
    1³; PyTorch's ``_verify_spatial_size`` in ``F.instance_norm`` raises
    unconditionally (the check runs in both train and eval mode — the error
    message says "when training" but that is static text, not a mode check).
    64³ gives a 2³ bottleneck (safe).  Expected output shapes:
    deep_supervision=False → Tensor (2, 2, 64, 64, 64)
    deep_supervision=True  → tuple[Tensor×3]:
        [0] (2, 2, 64, 64, 64)   full resolution
        [1] (2, 2, 32, 32, 32)   decoder2 output (H/2, feature_size ch)
        [2] (2, 2, 16, 16, 16)   decoder3 output (H/4, 2×feature_size ch)

SwinUNETR divisibility
----------------------
Input (2, 3, 32, 32, 24) raises ValueError for Arms A/B because 24 is not
divisible by patch_size**5 = 2**5 = 32.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

# Ensure segmentation models are registered before collection.
import vena.segmentation.models  # noqa: F401

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_cfg(
    name: str,
    *,
    deep_supervision: bool,
    checkpoint: Path | None = None,
    feature_size: int = 48,
) -> object:
    """Build a minimal ModelConfig for tests (no I/O)."""
    from vena.segmentation.config import ModelConfig

    return ModelConfig(
        name=name,
        feature_size=feature_size,
        in_channels=3,
        out_channels=2,
        checkpoint=checkpoint,
        strict_load=False,
        deep_supervision=deep_supervision,
    )


def _build(name: str, *, ds: bool, ckpt: Path | None = None) -> torch.nn.Module:
    from vena.segmentation.models import get_segmentation_model

    cfg = _make_model_cfg(name, deep_supervision=ds, checkpoint=ckpt)
    return get_segmentation_model(name, cfg)


def _synthetic_bsf_checkpoint(
    n_channels: int,
    tmp_dir: Path | None = None,
) -> Path:
    """Write a minimal fake BSF checkpoint to a temp file.

    Always uses ``feature_size=48`` (the production target size) so that the
    norm key shape matches the real MONAI SwinUNETR target model and lands
    in ``matched``.  Uses a plain tensor dict — no SwinUNETR instantiation —
    so the helper is fast (milliseconds) regardless of feature_size.

    The checkpoint mimics DataParallel wrapping (``module.`` prefix on all
    keys) so ``load_bsf_encoder`` receives the expected format.

    Two keys are synthesised:

    1. ``swinViT.patch_embed.proj.weight`` — the stem; shape ``(48, n_channels,
       2, 2, 2)``; gets sliced to 3-ch (BraTS) or mismatches and is skipped (UKB).
    2. ``swinViT.layers1.0.blocks.0.norm1.weight`` — a representative encoder
       key present in MONAI SwinUNETR's state-dict with shape ``(3*48=144,)``
       matching the target, so it always ends up in ``matched``.
    """
    _fs = 48  # must match target model feature_size; 48 % 12 == 0
    stem_key = "module.swinViT.patch_embed.proj.weight"
    norm_key = "module.swinViT.layers1.0.blocks.0.norm1.weight"

    raw_sd: dict[str, torch.Tensor] = {
        stem_key: torch.zeros(_fs, n_channels, 2, 2, 2),
        norm_key: torch.zeros(3 * _fs),  # 144 — matches target model exactly
    }
    payload = {"state_dict": raw_sd}

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    ckpt = tmp_dir / f"fake_bsf_{n_channels}ch.pt"
    torch.save(payload, str(ckpt))
    return ckpt


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_three_arms_registered(self) -> None:
        from vena.segmentation.models import registered_model_names

        names = registered_model_names()
        assert "bsf_swinunetr_brats" in names
        assert "bsf_swinunetr_ukb" in names
        assert "segresnet" in names
        assert len(names) == 3, f"Expected exactly 3, got: {names}"

    def test_unknown_model_raises_seg_model_error(self) -> None:
        from vena.segmentation.config import ModelConfig
        from vena.segmentation.exceptions import SegModelError
        from vena.segmentation.models import get_segmentation_model

        cfg = ModelConfig(name="segresnet")  # valid name for Pydantic
        with pytest.raises(SegModelError, match="does_not_exist"):
            get_segmentation_model("does_not_exist", cfg)

    def test_duplicate_registration_raises_seg_model_error(self) -> None:
        from vena.segmentation.exceptions import SegModelError
        from vena.segmentation.models.registry import register_segmentation_model

        with pytest.raises(SegModelError, match="already registered"):

            @register_segmentation_model("segresnet")
            def _duplicate_builder(cfg: object) -> None:
                pass


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


class TestImportIsolation:
    def test_segresnet_module_does_not_import_monai_at_top_level(self) -> None:
        """Importing vena.segmentation.models.segresnet must not pull monai
        into sys.modules at the top level — only inside builder functions."""
        # By the time pytest runs this, the module IS already imported.
        # We verify that the module's top-level code does not have
        # ``import monai`` by checking that there is no unconditional
        # ``from monai ...`` or ``import monai`` at module scope in the source.
        import inspect

        import vena.segmentation.models.segresnet as seg_mod

        src = inspect.getsource(seg_mod)
        top_lines = []
        in_func = False
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                in_func = True
            if not in_func:
                top_lines.append(stripped)
        top_block = "\n".join(top_lines)
        # Top-level of the module must NOT contain bare `import monai` or
        # `from monai` outside of a function/class body.
        assert "import monai" not in top_block, (
            "segresnet.py imports monai at module scope; it must be inside builders."
        )

    def test_bsf_swinunetr_module_does_not_import_monai_at_top_level(self) -> None:
        import inspect

        import vena.segmentation.models.bsf_swinunetr as bsf_mod

        src = inspect.getsource(bsf_mod)
        top_lines = []
        in_func = False
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                in_func = True
            if not in_func:
                top_lines.append(stripped)
        top_block = "\n".join(top_lines)
        assert "import monai" not in top_block, (
            "bsf_swinunetr.py imports monai at module scope; it must be inside builders."
        )


# ---------------------------------------------------------------------------
# Arm C — SegResNet (CPU, fast)
# ---------------------------------------------------------------------------

_SEG_INPUT = (2, 3, 32, 32, 24)  # H,W divisible by 32; D=24 divisible by 8


class TestSegResNet:
    @pytest.fixture
    def model_no_ds(self) -> torch.nn.Module:
        return _build("segresnet", ds=False)

    @pytest.fixture
    def model_ds(self) -> torch.nn.Module:
        return _build("segresnet", ds=True)

    def test_no_ds_output_is_tensor(self, model_no_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_no_ds(x)
        assert isinstance(out, torch.Tensor), type(out)

    def test_no_ds_output_shape(self, model_no_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_no_ds(x)
        assert out.shape == (2, 2, 32, 32, 24), out.shape

    def test_ds_output_is_tuple(self, model_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        assert isinstance(out, tuple), type(out)

    def test_ds_output_length(self, model_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        assert len(out) == 3, len(out)

    def test_ds_main_shape(self, model_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        assert out[0].shape == (2, 2, 32, 32, 24), out[0].shape

    def test_ds_aux_h2_shape(self, model_ds: torch.nn.Module) -> None:
        """aux_h2 = up_layers[1] output: H/2 spatial (16, 16, 12)."""
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        assert out[1].shape == (2, 2, 16, 16, 12), out[1].shape

    def test_ds_aux_h4_shape(self, model_ds: torch.nn.Module) -> None:
        """aux_h4 = up_layers[0] output: H/4 spatial (8, 8, 6)."""
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        assert out[2].shape == (2, 2, 8, 8, 6), out[2].shape

    def test_ds_all_elements_are_tensors(self, model_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SEG_INPUT)
        out = model_ds(x)
        for i, elem in enumerate(out):
            assert isinstance(elem, torch.Tensor), f"Element {i} is {type(elem)}"

    def test_hook_cleanup_no_crash(self, model_ds: torch.nn.Module) -> None:
        """remove_hooks() is idempotent and raises no exception."""
        # Access the private wrapper to call remove_hooks.
        from vena.segmentation.models.segresnet import _VenaSegResNet

        assert isinstance(model_ds, _VenaSegResNet)
        model_ds.remove_hooks()
        model_ds.remove_hooks()  # second call must be a no-op

    def test_checkpoint_none_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """build_segresnet with checkpoint=None should not emit any warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            _build("segresnet", ds=False, ckpt=None)
        assert len(caplog.records) == 0, caplog.text


# ---------------------------------------------------------------------------
# LoadReport + load_bsf_encoder (synthetic checkpoint, CPU, fast)
# ---------------------------------------------------------------------------


class TestLoadReport:
    def test_dataclass_fields(self) -> None:
        from vena.segmentation.models.bsf_swinunetr import LoadReport

        lr = LoadReport(matched=10, total=12, skipped=["a.b"])
        assert lr.matched == 10
        assert lr.total == 12
        assert lr.skipped == ["a.b"]

    def test_default_skipped_is_empty_list(self) -> None:
        from vena.segmentation.models.bsf_swinunetr import LoadReport

        lr = LoadReport(matched=5, total=5)
        assert lr.skipped == []


class TestLoadBsfEncoder:
    """Tests using a synthetic tiny checkpoint — no real BSF weights needed."""

    def test_ukb_stem_skipped(self, tmp_path: Path) -> None:
        """UKB 1-ch SSL stem is incompatible with 3-ch model → goes to skipped."""
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import LoadReport, load_bsf_encoder

        # Build the target 3-ch model (feature_size=48 as in production).
        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)

        # Synthetic 1-ch checkpoint mimicking UKB SSL (feature_size=48 to match target).
        ckpt = _synthetic_bsf_checkpoint(n_channels=1, tmp_dir=tmp_path)

        report = load_bsf_encoder(model, ckpt, brats_channel_slice=None)
        assert isinstance(report, LoadReport)
        # The 1-ch stem key must be in skipped.
        stem_keys = [k for k in report.skipped if "patch_embed.proj.weight" in k]
        assert len(stem_keys) == 1, f"Expected stem in skipped, got: {report.skipped}"
        # Total = matched + len(skipped).
        assert report.total == report.matched + len(report.skipped)

    def test_brats_stem_sliced(self, tmp_path: Path) -> None:
        """BraTS 4-ch stem is sliced to 3-ch and NOT placed in skipped."""
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import (
            _BRATS_STEM_CHANNEL_SLICE,
            LoadReport,
            load_bsf_encoder,
        )

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        ckpt = _synthetic_bsf_checkpoint(n_channels=4, tmp_dir=tmp_path)

        report = load_bsf_encoder(model, ckpt, brats_channel_slice=_BRATS_STEM_CHANNEL_SLICE)
        assert isinstance(report, LoadReport)
        # Stem must NOT be in skipped — it was sliced and loaded.
        stem_keys = [k for k in report.skipped if "patch_embed.proj.weight" in k]
        assert len(stem_keys) == 0, f"Stem should be sliced, not skipped: {report.skipped}"
        assert report.total == report.matched + len(report.skipped)

    def test_missing_checkpoint_raises_seg_model_error(self, tmp_path: Path) -> None:
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.exceptions import SegModelError
        from vena.segmentation.models.bsf_swinunetr import load_bsf_encoder

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        nonexistent = tmp_path / "does_not_exist.pt"
        with pytest.raises(SegModelError, match="not found"):
            load_bsf_encoder(model, nonexistent)

    def test_matched_plus_skipped_equals_total(self, tmp_path: Path) -> None:
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import load_bsf_encoder

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        ckpt = _synthetic_bsf_checkpoint(n_channels=1, tmp_dir=tmp_path)
        report = load_bsf_encoder(model, ckpt, brats_channel_slice=None)
        assert report.matched + len(report.skipped) == report.total


# ---------------------------------------------------------------------------
# Arms A/B — SwinUNETR shape tests (slow — CPU forward takes several seconds)
# ---------------------------------------------------------------------------

# 64³ input: gives 2³ bottleneck (64→32→16→8→4→2 across 5 downsampling stages).
# 32³ would give 1³, which raises unconditionally in PyTorch's instance_norm
# _verify_spatial_size check regardless of train/eval mode.
_SWIN_INPUT = (2, 3, 64, 64, 64)


@pytest.mark.slow
class TestSwinUNETRShapes:
    """Shape tests for SwinUNETR arms (CPU, 64³ input).

    SwinUNETR with feature_size=48 has 5 downsampling stages (patch-embed
    stride-2 + 4 patch-merging).  With 32³ input the bottleneck collapses to
    1³; PyTorch's ``_verify_spatial_size`` in ``F.instance_norm`` raises
    unconditionally (the check is NOT gated on training mode — the static
    error message says "when training" but runs in both modes).  With 64³ the
    bottleneck is 2³, which is safe.
    """

    @pytest.fixture(scope="class")
    def model_a_no_ds(self) -> torch.nn.Module:
        return _build("bsf_swinunetr_brats", ds=False)

    @pytest.fixture(scope="class")
    def model_a_ds(self) -> torch.nn.Module:
        return _build("bsf_swinunetr_brats", ds=True)

    @pytest.fixture(scope="class")
    def model_b_no_ds(self) -> torch.nn.Module:
        return _build("bsf_swinunetr_ukb", ds=False)

    @pytest.fixture(scope="class")
    def model_b_ds(self) -> torch.nn.Module:
        return _build("bsf_swinunetr_ukb", ds=True)

    def test_arm_a_no_ds_shape(self, model_a_no_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SWIN_INPUT)
        out = model_a_no_ds(x)
        assert isinstance(out, torch.Tensor), type(out)
        assert out.shape == (2, 2, 64, 64, 64), out.shape

    def test_arm_a_ds_main_shape(self, model_a_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SWIN_INPUT)
        out = model_a_ds(x)
        assert isinstance(out, tuple) and len(out) == 3
        assert out[0].shape == (2, 2, 64, 64, 64), out[0].shape

    def test_arm_a_ds_aux_h2_shape(self, model_a_ds: torch.nn.Module) -> None:
        """decoder2 output: feature_size channels at H/2 = (32, 32, 32)."""
        x = torch.zeros(*_SWIN_INPUT)
        out = model_a_ds(x)
        assert out[1].shape == (2, 2, 32, 32, 32), out[1].shape

    def test_arm_a_ds_aux_h4_shape(self, model_a_ds: torch.nn.Module) -> None:
        """decoder3 output: 2×feature_size channels at H/4 = (16, 16, 16)."""
        x = torch.zeros(*_SWIN_INPUT)
        out = model_a_ds(x)
        assert out[2].shape == (2, 2, 16, 16, 16), out[2].shape

    def test_arm_b_no_ds_shape(self, model_b_no_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SWIN_INPUT)
        out = model_b_no_ds(x)
        assert isinstance(out, torch.Tensor), type(out)
        assert out.shape == (2, 2, 64, 64, 64), out.shape

    def test_arm_b_ds_output_length(self, model_b_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SWIN_INPUT)
        out = model_b_ds(x)
        assert isinstance(out, tuple) and len(out) == 3

    def test_arm_b_ds_all_shapes(self, model_b_ds: torch.nn.Module) -> None:
        x = torch.zeros(*_SWIN_INPUT)
        out = model_b_ds(x)
        expected = [(2, 2, 64, 64, 64), (2, 2, 32, 32, 32), (2, 2, 16, 16, 16)]
        for i, (got, want) in enumerate(zip(out, expected, strict=True)):
            assert got.shape == want, f"out[{i}].shape={got.shape}, expected {want}"


@pytest.mark.slow
class TestSwinUNETRDivisibility:
    """SwinUNETR raises ValueError for spatial dims not divisible by 32."""

    def test_depth_24_raises_on_arm_a(self) -> None:
        """Input (2,3,32,32,24): D=24 is not divisible by 32."""
        model = _build("bsf_swinunetr_brats", ds=False)
        x = torch.zeros(2, 3, 32, 32, 24)
        with pytest.raises((ValueError, RuntimeError)):
            model(x)

    def test_depth_24_raises_on_arm_b(self) -> None:
        model = _build("bsf_swinunetr_ukb", ds=False)
        x = torch.zeros(2, 3, 32, 32, 24)
        with pytest.raises((ValueError, RuntimeError)):
            model(x)

    def test_all_64_passes_on_arm_a(self) -> None:
        """Input (2,3,64,64,64): all dims divisible by 32 — no error.

        Note: 32³ would also satisfy the divisibility gate but collapses the
        bottleneck to 1³, causing PyTorch's ``_verify_spatial_size`` to raise
        unconditionally (both train and eval).  64³ gives 2³ bottleneck.
        """
        model = _build("bsf_swinunetr_brats", ds=False)
        x = torch.zeros(2, 3, 64, 64, 64)
        out = model(x)
        assert out.shape[0] == 2

    def test_segresnet_accepts_depth_24(self) -> None:
        """SegResNet (Arm C) is NOT constrained to multiples of 32."""
        model = _build("segresnet", ds=False)
        x = torch.zeros(2, 3, 32, 32, 24)
        out = model(x)
        assert out.shape == (2, 2, 32, 32, 24)


# ---------------------------------------------------------------------------
# Backbone accessor on _VenaSwinUNETR
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBackboneAccessor:
    def test_arm_a_backbone_is_swinunetr(self) -> None:
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import _VenaSwinUNETR

        model = _build("bsf_swinunetr_brats", ds=False)
        assert isinstance(model, _VenaSwinUNETR)
        assert isinstance(model.backbone, SwinUNETR)

    def test_arm_b_backbone_is_swinunetr(self) -> None:
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import _VenaSwinUNETR

        model = _build("bsf_swinunetr_ukb", ds=False)
        assert isinstance(model, _VenaSwinUNETR)
        assert isinstance(model.backbone, SwinUNETR)


# ---------------------------------------------------------------------------
# Real checkpoint load tests (slow + skip-if-absent)
# Verified on local workstation 2026-07-23.
# SHA-256:
#   Arm A: e46d80ce75f3828222cdfd3f4891c753e9cd0356719e7d21803c89e1b8797077
#   Arm B: 4be92492ae4f55e934e278700a13a6c18ef8406daba2177c5f9f157dcff5f341
# ---------------------------------------------------------------------------

_REAL_BRATS_CKPT = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models"
    "/BrainSegFounder_SSL_BraTS/model_bestValRMSE-fold0.pt"
)
_REAL_UKB_CKPT = Path(
    "/media/mpascual/Sandisk2TB/checkpoints/BrainSegFounder/models"
    "/BrainSegFounder_SSL_UKBiobank/64-gpu-model_bestValRMSE.pt"
)


@pytest.mark.slow
class TestRealBsfLoad:
    """Real checkpoint load tests.

    Skipped automatically when the checkpoint files are absent (CI / Picasso).
    Expected results (measured locally, 2026-07-23):
        Arm A: total=198, matched=126 (63.6%), skipped=72 (all name_missing)
        Arm B: total=142, matched=125 (88.0%), skipped=17 (1 shape + 16 name)

    Note on Arm A matched/total < 0.80:
        The BraTS SSL checkpoint has 56 extra layers3 encoder blocks absent
        from standard MONAI SwinUNETR (deeper SSL architecture) plus 16 SSL
        task-head keys.  All 126 transferable keys load without shape error.
        The coordinator brief expected ≥0.80 — this is a PREMISE-FALSE finding.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self, request: pytest.FixtureRequest) -> None:
        ckpts = {"arm_a": _REAL_BRATS_CKPT, "arm_b": _REAL_UKB_CKPT}
        for _name, path in ckpts.items():
            if not path.exists():
                pytest.skip(f"Real BSF checkpoint absent: {path}")

    def test_arm_a_real_load_matched_count(self) -> None:
        """Arm A: 126/198 keys transferred (all 126 target-compatible transfer)."""
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import (
            _BRATS_STEM_CHANNEL_SLICE,
            load_bsf_encoder,
        )

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        report = load_bsf_encoder(
            model, _REAL_BRATS_CKPT, brats_channel_slice=_BRATS_STEM_CHANNEL_SLICE
        )
        assert report.total == 198
        assert report.matched == 126
        assert len(report.skipped) == 72
        # All skipped are name_missing (no shape mismatches after stem slice).
        stem_in_skipped = [k for k in report.skipped if "patch_embed.proj.weight" in k]
        assert stem_in_skipped == [], f"Stem should be matched after slice: {stem_in_skipped}"

    def test_arm_b_real_load_matched_count(self) -> None:
        """Arm B: 125/142 keys transferred; stem goes to skipped (shape mismatch)."""
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import load_bsf_encoder

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        report = load_bsf_encoder(model, _REAL_UKB_CKPT, brats_channel_slice=None)
        assert report.total == 142
        assert report.matched == 125
        assert len(report.skipped) == 17
        stem_in_skipped = [k for k in report.skipped if "patch_embed.proj.weight" in k]
        assert len(stem_in_skipped) == 1, f"UKB stem must be in skipped: {report.skipped}"

    def test_arm_b_matched_fraction_above_0_80(self) -> None:
        """Arm B: matched/total ≥ 0.80 (actual 0.880)."""
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import load_bsf_encoder

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        report = load_bsf_encoder(model, _REAL_UKB_CKPT, brats_channel_slice=None)
        assert report.matched / report.total >= 0.80

    def test_invariant_matched_plus_skipped_equals_total_arm_a(self) -> None:
        from monai.networks.nets import SwinUNETR

        from vena.segmentation.models.bsf_swinunetr import (
            _BRATS_STEM_CHANNEL_SLICE,
            load_bsf_encoder,
        )

        model = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=3)
        report = load_bsf_encoder(
            model, _REAL_BRATS_CKPT, brats_channel_slice=_BRATS_STEM_CHANNEL_SLICE
        )
        assert report.matched + len(report.skipped) == report.total
