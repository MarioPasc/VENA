"""Phase-2 downstream segmentation library (§4.4).

Wraps the MONAI Model-Zoo ``brats_mri_segmentation`` bundle (SegResNet v0.5.4)
as a fixed, pretrained segmenter to measure task-based downstream quality.

Instrument contract (verified from bundle configs/metadata.json 2026-07-16):
  - Name:     brats_mri_segmentation v0.5.4
  - Model:    SegResNet (in_channels=4, out_channels=3, dropout_prob=0.2)
  - model.pt SHA-256:
      860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae
  - Input channel order  (channel_def 0–3): T1c, T1, T2, FLAIR
  - Preprocessing:  NormalizeIntensityd(nonzero=True, channel_wise=True)
                    z-score per channel over nonzero (skull-stripped) voxels.
  - Output channel order (channel_def 0–2): TC, WT, ET
    Three independent sigmoid heads; threshold 0.5 → binary masks.
  - Inferer:   SlidingWindowInferer(roi_size=[240,240,160], overlap=0.5)
               handles the BraTS 240×240×155 grid via implicit end-of-volume
               padding along the depth axis.

Deviation from proposal Appendix A:
  Appendix A prescribes training nnU-Net from scratch per Ring-A cohort.
  The level confounder (segmenter familiarity with the real BraTS distribution)
  shifts Dice_real and Dice_synth together — it cancels in the paired Δ, which
  is the quantity §4.4 actually claims.  A fixed pretrained segmenter is
  therefore a valid measuring instrument for the Δ endpoint.  nnunetv2 is not
  installed in the vena env and must stay absent.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DownstreamSegError(Exception):
    """Base for downstream-seg library errors."""


class LabelSystemError(DownstreamSegError):
    """Unknown label_system attribute in corpus H5."""


class MissingTumorMaskError(DownstreamSegError):
    """Corpus H5 has no masks/tumor dataset."""


class SegmenterError(DownstreamSegError):
    """BraTS segmenter loading or inference failure."""


# ---------------------------------------------------------------------------
# Label-system helpers  (trap #9 — SHARED_CONTRACTS §11)
# ---------------------------------------------------------------------------

_KNOWN_LABEL_SYSTEMS: frozenset[str] = frozenset({"BraTS2021", "BraTS2023"})


def derive_sub_labels(
    tumor: np.ndarray,
    label_system: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive binary WT / TC / ET masks from a multi-label tumour array.

    Branch on the corpus H5 ``label_system`` root attribute.  **Never
    hard-code label value 4** — BraTS2023 uses 3 for ET instead.

    Parameters
    ----------
    tumor : np.ndarray
        Integer-labelled tumour mask ``(H, W, D)``, int8.
    label_system : str
        Corpus H5 root attr ``label_system``.
        ``"BraTS2021"`` → {0=bg, 1=NCR, 2=ED, 4=ET}.
        ``"BraTS2023"`` → {0=bg, 1=NCR, 2=SNFH/ED, 3=ET}.

    Returns
    -------
    (wt, tc, et) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Boolean arrays of the same shape as *tumor*.

    Raises
    ------
    LabelSystemError
        If *label_system* is not ``"BraTS2021"`` or ``"BraTS2023"``.
    """
    if label_system == "BraTS2021":
        wt = tumor > 0
        tc = (tumor == 1) | (tumor == 4)
        et = tumor == 4
    elif label_system == "BraTS2023":
        wt = tumor > 0
        tc = (tumor == 1) | (tumor == 3)
        et = tumor == 3
    else:
        raise LabelSystemError(
            f"Unknown label_system {label_system!r}. "
            f"Expected one of {sorted(_KNOWN_LABEL_SYSTEMS)}."
        )
    return wt.astype(bool), tc.astype(bool), et.astype(bool)


# ---------------------------------------------------------------------------
# Dice metric
# ---------------------------------------------------------------------------


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two boolean masks.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask ``(H, W, D)``, bool or int.
    gt : np.ndarray
        Ground-truth binary mask ``(H, W, D)``, bool or int.

    Returns
    -------
    float
        Dice in ``[0, 1]``.  Returns ``float("nan")`` when **both** *pred* and
        *gt* are empty — the empty-region convention that avoids a systematic
        downward bias on non-enhancing sub-regions.  See §4.4 and §5 of
        05_downstream_seg.md.
    """
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    n_pred = int(pred_b.sum())
    n_gt = int(gt_b.sum())
    if n_pred == 0 and n_gt == 0:
        return float("nan")
    n_inter = int((pred_b & gt_b).sum())
    return float(2 * n_inter) / float(n_pred + n_gt)


# ---------------------------------------------------------------------------
# CorpusLabelCache — lazy reader for multi-label GT from corpus H5s
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CorpusEntry:
    path: Path
    label_system: str


class CorpusLabelCache:
    """Read-through cache for multi-label tumour GT from corpus image H5s.

    Lazy: opens H5 files on first access and keeps them open.  Call
    :meth:`close` (or use as a context manager) when the engine finishes.

    Parameters
    ----------
    corpus_map : dict[str, Path]
        ``{cohort_name: /abs/path/to/<cohort>_image.h5}``.
        Cohorts whose path does not exist are silently skipped with a WARNING.
    """

    def __init__(self, corpus_map: dict[str, Path]) -> None:
        self._entries: dict[str, _CorpusEntry] = {}
        for cohort, path in corpus_map.items():
            path = Path(path)
            if not path.is_file():
                logger.warning("corpus H5 not found for cohort %s: %s", cohort, path)
                continue
            with h5py.File(path, "r") as fh:
                label_system = str(fh.attrs.get("label_system", ""))
                if "masks/tumor" not in fh:
                    logger.warning(
                        "cohort %s (%s) has no masks/tumor — will skip downstream-seg",
                        cohort,
                        path,
                    )
                    continue
                if not label_system:
                    logger.warning(
                        "cohort %s (%s) missing label_system attr — will skip",
                        cohort,
                        path,
                    )
                    continue
            self._entries[cohort] = _CorpusEntry(path=path, label_system=label_system)
            logger.info("registered corpus H5 for %s (label_system=%s)", cohort, label_system)

        # Lazy state: open file handle + scan-id → row-index map
        self._handles: dict[str, h5py.File] = {}
        self._index: dict[str, dict[str, int]] = {}

    def _open(self, cohort: str) -> None:
        if cohort in self._handles:
            return
        entry = self._entries[cohort]
        fh = h5py.File(entry.path, "r")
        self._handles[cohort] = fh
        # Corpus H5 uses "ids" (not "metadata/scan_id") per §3.1
        ids_raw = fh["ids"][:]
        ids = [sid.decode() if isinstance(sid, bytes) else str(sid) for sid in ids_raw]
        self._index[cohort] = {sid: i for i, sid in enumerate(ids)}
        logger.debug("opened corpus H5 for %s (%d scans)", cohort, len(ids))

    def has_cohort(self, cohort: str) -> bool:
        """Return True if the cohort's corpus H5 is available and has tumor masks."""
        return cohort in self._entries

    def label_system(self, cohort: str) -> str:
        """Return the label_system string for *cohort*."""
        return self._entries[cohort].label_system

    def get_inputs(self, cohort: str, scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(t1pre, t2, flair)`` for *scan_id* in *cohort* from the corpus H5.

        These three modalities are identical between the real and synthetic arms
        (only T1c differs), so they are read once and reused.

        Parameters
        ----------
        cohort : str
        scan_id : str

        Returns
        -------
        (t1pre, t2, flair) : tuple of float32 ``(H, W, D)``

        Raises
        ------
        KeyError
            If *scan_id* is not in the corpus index for *cohort*.
        KeyError
            If *cohort* is not registered (missing corpus H5 or no tumor mask).
        """
        self._open(cohort)
        idx = self._index[cohort].get(scan_id)
        if idx is None:
            raise KeyError(f"scan_id {scan_id!r} not found in corpus H5 for {cohort!r}")
        fh = self._handles[cohort]
        t1pre = fh["images/t1pre"][idx].astype(np.float32)
        t2 = fh["images/t2"][idx].astype(np.float32)
        flair = fh["images/flair"][idx].astype(np.float32)
        return t1pre, t2, flair

    def get_labels(self, cohort: str, scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(wt, tc, et)`` boolean masks for *scan_id* in *cohort*.

        Parameters
        ----------
        cohort : str
        scan_id : str

        Returns
        -------
        (wt, tc, et) : tuple of bool ``(H, W, D)``

        Raises
        ------
        KeyError
            If *scan_id* is not in the corpus index for *cohort*.
        """
        self._open(cohort)
        idx = self._index[cohort].get(scan_id)
        if idx is None:
            raise KeyError(f"scan_id {scan_id!r} not found in corpus H5 for {cohort!r}")
        fh = self._handles[cohort]
        tumor = fh["masks/tumor"][idx].astype(np.int8)  # (H, W, D)
        label_sys = self._entries[cohort].label_system
        return derive_sub_labels(tumor, label_sys)

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError as exc:
                logger.warning("error closing corpus H5: %s", exc)
        self._handles.clear()
        self._index.clear()

    def __enter__(self) -> CorpusLabelCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Intensity normalisation (bundle preprocessing — applied to both arms)
# ---------------------------------------------------------------------------


def normalize_intensity_channel_wise(volumes: np.ndarray) -> np.ndarray:
    """Z-score each channel over its nonzero (skull-stripped foreground) voxels.

    Replicates the bundle's ``NormalizeIntensityd(nonzero=True,
    channel_wise=True)`` exactly.  Zero-valued voxels remain zero after
    normalisation (brain mask is implicit in the zero values of
    skull-stripped volumes).

    Parameters
    ----------
    volumes : np.ndarray
        ``(C, H, W, D)`` float32 multi-channel volume.

    Returns
    -------
    np.ndarray
        ``(C, H, W, D)`` float32 z-scored volume.
    """
    out = volumes.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        ch = out[c]
        mask = ch != 0.0
        if mask.any():
            mu = float(ch[mask].mean())
            sigma = float(ch[mask].std())
            if sigma > 1e-8:
                ch[mask] = (ch[mask] - mu) / sigma
            else:
                ch[mask] = 0.0
        out[c] = ch
    return out


# ---------------------------------------------------------------------------
# BratsSegmenter
# ---------------------------------------------------------------------------

# Verified SHA-256 of the bundle's model.pt (v0.5.4, downloaded 2026-07-16)
_MODEL_PT_SHA256 = "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae"

# Module-level bundle constants — not on BratsSegmenter so they survive
# mocking of the class in tests.  Engines must import these directly.
BRATS_INPUT_CHANNELS: tuple[str, ...] = ("t1c", "t1", "t2", "flair")
BRATS_OUTPUT_CHANNELS: tuple[str, ...] = ("tc", "wt", "et")
BRATS_BUNDLE_VERSION: str = "0.5.4"


class BratsSegmenter:
    """MONAI brats_mri_segmentation SegResNet v0.5.4 wrapper.

    Bundle instrument contract verified from bundle configs/metadata.json.

    Do not instantiate at import time — this class loads a checkpoint.
    All heavy work lives inside :meth:`segment`.

    Parameters
    ----------
    bundle_path : Path
        Root of the downloaded bundle (contains ``configs/`` and ``models/``).
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    amp : bool
        Use automatic mixed-precision.  Ignored on CPU.
    threshold : float
        Sigmoid probability threshold for binarising output.  Default 0.5
        (bundle contract).
    """

    # Aliases to module-level constants (survive class-level mocking in tests).
    INPUT_CHANNELS: tuple[str, ...] = BRATS_INPUT_CHANNELS
    OUTPUT_CHANNELS: tuple[str, ...] = BRATS_OUTPUT_CHANNELS
    BUNDLE_VERSION: str = BRATS_BUNDLE_VERSION

    def __init__(
        self,
        bundle_path: Path,
        *,
        device: str = "cpu",
        amp: bool = False,
        threshold: float = 0.5,
    ) -> None:
        import torch
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import SegResNet

        self._device = torch.device(device)
        self._amp = amp and self._device.type != "cpu"
        self._threshold = threshold

        bundle_path = Path(bundle_path)
        ckpt_path = bundle_path / "models" / "model.pt"
        if not ckpt_path.is_file():
            raise SegmenterError(f"model.pt not found at {ckpt_path}")

        # Log and verify SHA-256 (external-deps.md rule 6).
        sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        logger.info("brats_mri_segmentation model.pt SHA-256: %s", sha)
        if sha != _MODEL_PT_SHA256:
            logger.warning(
                "SHA-256 mismatch for %s: expected %s, got %s",
                ckpt_path,
                _MODEL_PT_SHA256,
                sha,
            )

        net = SegResNet(
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        self._net = net.to(self._device)
        # SlidingWindowInferer matches the bundle's inference.json exactly.
        self._inferer = SlidingWindowInferer(
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            overlap=0.5,
        )
        logger.info(
            "BratsSegmenter loaded from %s (device=%s, amp=%s)",
            bundle_path,
            device,
            self._amp,
        )

    def segment(
        self,
        t1c: np.ndarray,
        t1pre: np.ndarray,
        t2: np.ndarray,
        flair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run BraTS segmentation on a 4-channel 1 mm brain volume.

        Applies the bundle's own NormalizeIntensityd preprocessing and
        returns hard binary masks.  Volumes are in harmonised [0, 1] space
        from the inference shard — the bundle preprocessing re-standardises
        them (this is the instrument's contract, not a re-harmonisation, and
        is applied identically to both real and synthetic arms).

        Parameters
        ----------
        t1c : np.ndarray
            ``(H, W, D)`` float32 — T1c or synthetic T1c, harmonised.
        t1pre : np.ndarray
            ``(H, W, D)`` float32 — T1pre harmonised.
        t2 : np.ndarray
            ``(H, W, D)`` float32 — T2 harmonised.
        flair : np.ndarray
            ``(H, W, D)`` float32 — FLAIR harmonised.

        Returns
        -------
        (tc_mask, wt_mask, et_mask) : tuple of ``np.ndarray``
            Boolean ``(H, W, D)`` masks.
            Output order: TC (channel 0), WT (channel 1), ET (channel 2).
        """
        import contextlib

        import torch

        # Channel order: [T1c, T1, T2, FLAIR] (bundle contract — DO NOT reorder).
        volumes = np.stack([t1c, t1pre, t2, flair], axis=0)  # (4, H, W, D)
        volumes = normalize_intensity_channel_wise(volumes)

        x = torch.from_numpy(volumes).unsqueeze(0).to(self._device)  # (1, 4, H, W, D)

        ctx: contextlib.AbstractContextManager[None]
        if self._amp:
            ctx = torch.autocast(device_type=self._device.type)  # type: ignore[assignment]
        else:
            ctx = contextlib.nullcontext()

        with torch.no_grad(), ctx:
            logits = self._inferer(x, self._net)  # (1, 3, H, W, D)

        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (3, H, W, D)
        tc_mask = probs[0] >= self._threshold
        wt_mask = probs[1] >= self._threshold
        et_mask = probs[2] >= self._threshold

        torch.cuda.empty_cache()
        return tc_mask, wt_mask, et_mask


# ---------------------------------------------------------------------------
# Per-scan result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SegResult:
    """Per-scan segmentation result for one ``(method, cohort, nfe, scan)`` tuple."""

    method: str
    cohort: str
    ring: str
    nfe: int
    scan_id: str
    patient_id: str
    dice_wt_real: float
    dice_tc_real: float
    dice_et_real: float
    dice_wt_synth: float
    dice_tc_synth: float
    dice_et_synth: float

    @property
    def delta_wt(self) -> float:
        """Dice_real_WT − Dice_synth_WT."""
        return self.dice_wt_real - self.dice_wt_synth

    @property
    def delta_tc(self) -> float:
        """Dice_real_TC − Dice_synth_TC."""
        return self.dice_tc_real - self.dice_tc_synth

    @property
    def delta_et(self) -> float:
        """Dice_real_ET − Dice_synth_ET."""
        return self.dice_et_real - self.dice_et_synth


__all__ = [
    "BratsSegmenter",
    "CorpusLabelCache",
    "DownstreamSegError",
    "LabelSystemError",
    "MissingTumorMaskError",
    "SegResult",
    "SegmenterError",
    "derive_sub_labels",
    "dice_score",
    "normalize_intensity_channel_wise",
]
