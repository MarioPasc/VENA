"""Add ``masks/brain_latent`` to an existing latent H5 (base + optional aug).

The latent H5 is produced by :class:`LatentH5Converter` (or
``build_aug_latent_manifest`` for the offline aug bank). Both schemas omit a
latent-space brain mask. The 2026-06-09 training-regime overhaul
(CHANGE 2 of ``.claude/notes/changes/2026-06-09_training-regime-overhaul.md``)
introduced the v0.4 contrastive loss, which restricts its Lp residual to a
configurable list of latent-space regions — ``healthy`` ≡ ``brain ∩ ¬wt`` is
the doc default. The brain mask is required for that to be well-defined.

The brain mask source is the image-domain ``masks/brain`` (binary, ``int8``,
present in every cohort's image H5 — see :doc:`/.claude/rules/h5-design-principles`).
The encoder:

1. Walks every row of the target latent H5.
2. Looks up the corresponding patient ID in the source image H5 (via
   ``ids``) and reads its ``masks/brain``.
3. Applies the same brain-centred crop box used at latent-encode time
   (``192 × 224 × 192`` per :data:`LATENT_CROP_BOX`).
4. Max-pools 4×4×4 → ``(48, 56, 48)``, binarises.
5. Writes ``masks/brain_latent`` (``int8``).

Augmented latent H5s (one cohort's offline aug bank) follow the same recipe
for ``v0–v3`` rows — those are intensity-only augmentations, so the brain
mask is invariant. ``v4`` rows applied an elastic+affine deformation, and
replaying that deterministically from ``aug_params_json`` is out of scope
for this PR. Those rows are written as ``ones`` with a ``v4_brain_synthesised_ones``
attribute set on the dataset; downstream the contrastive degrades to a
"full-volume Lp" for those rows but does not corrupt training for the v0–v3
majority.

The routine emits a ``decision.json`` per invocation under
``artifacts/encode/brain_to_latent/<UTC-timestamp>/``:

.. code-block:: json

    {
      "schema_version": "0.1.0",
      "target_h5": "/abs/path/.../<cohort>_latents.h5",
      "n_rows_written": 501,
      "n_v4_synthesised_ones": 0,
      "source_image_h5_sha256_first_8": "ab12cd34"
    }
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict

from vena.common import CropPadSpec, apply_crop_pad
from vena.data.h5.latent_domain.manifest import LATENT_SPATIAL

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"routines.encode.brain_to_latent:{_PRODUCER_VERSION}"
_DECISION_SCHEMA = "0.1.0"


class BrainToLatentRoutineConfig(BaseModel):
    """Pydantic root config for ``vena-encode-brain-to-latent``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The latent H5 whose rows will gain ``masks/brain_latent``.
    target_h5: Path
    # Source image H5 from which ``masks/brain`` is read. The patient IDs in
    # this file must form a superset of ``target_h5["ids"]``.
    source_image_h5: Path
    # Optional offline-augmented latent H5 (one row per (patient, variant)).
    # When set, the encoder writes brain masks for its rows too; v4 rows
    # become all-ones with a per-dataset attribute flag.
    target_aug_h5: Path | None = None
    # Overwrite an existing ``masks/brain_latent`` dataset; default skips.
    overwrite: bool = False
    # Directory under which a timestamped artifact dir is created carrying
    # the run's ``decision.json``.
    artifacts_root: Path = Path("artifacts/encode/brain_to_latent")
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path | str) -> BrainToLatentRoutineConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------


def _read_crop_box(image_h5_path: Path) -> tuple[int, int, int]:
    """Read the canonical crop box from the image H5's root attrs.

    The image H5 always carries either ``crop_box`` (vlen-str JSON list, the
    canonical schema-v2 attr) or, very rarely, an older legacy form. We parse
    leniently and require the result is a 3-tuple of ints.
    """
    with h5py.File(image_h5_path, "r") as f:
        raw = f.attrs.get("crop_box")
        if raw is None:
            raise KeyError(f"image H5 {image_h5_path} has no `crop_box` root attr")
        # Could be a string (JSON list) or already a tuple/list/ndarray.
        if isinstance(raw, (bytes, str)):
            parsed = json.loads(raw if isinstance(raw, str) else raw.decode())
        else:
            parsed = list(raw)
    if len(parsed) != 3:
        raise ValueError(f"crop_box must be length-3; got {parsed}")
    return tuple(int(x) for x in parsed)  # type: ignore[return-value]


def _encode_brain_mask(
    brain_image: np.ndarray,
    crop_origin: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Image-domain ``masks/brain`` → latent-grid binary mask.

    Pipeline: float-cast → ``apply_crop_pad`` (brain-centred crop box at
    per-scan ``crop_origin``) → max-pool 4 → binarise. Output shape
    ``(1, 48, 56, 48)``, ``int8``.
    """
    if brain_image.ndim != 3:
        raise ValueError(f"masks/brain must be 3-D (H, W, D); got shape {brain_image.shape}")
    spec = CropPadSpec(
        crop_origin=crop_origin,
        native_shape=tuple(brain_image.shape),  # type: ignore[arg-type]
        target_shape=target_shape,
    )
    t = torch.from_numpy(brain_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    cropped = apply_crop_pad(t, spec)  # (1, 1, *target_shape)
    pooled = F.max_pool3d(cropped, kernel_size=4, stride=4)
    pooled_np = (pooled > 0).to(torch.int8).numpy()[0]  # (1, 48, 56, 48)
    expected = (1, *LATENT_SPATIAL)
    if pooled_np.shape != expected:
        raise RuntimeError(
            f"brain-latent shape mismatch: got {pooled_np.shape}; expected {expected}"
        )
    return pooled_np


class BrainToLatentRoutineEngine:
    """Orchestrate the brain-mask latent encoding for one cohort."""

    def __init__(self, cfg: BrainToLatentRoutineConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the routine; return the artifact directory path."""
        if not self.cfg.target_h5.exists():
            raise FileNotFoundError(f"target H5 not found: {self.cfg.target_h5}")
        if not self.cfg.source_image_h5.exists():
            raise FileNotFoundError(f"source image H5 not found: {self.cfg.source_image_h5}")

        crop_box = _read_crop_box(self.cfg.source_image_h5)
        logger.info("Source image H5 crop_box=%s", crop_box)

        # Cache brain masks (image-domain) by patient_id so we read each source
        # row at most once even when an aug-H5 has multiple rows per patient.
        brain_cache: dict[str, np.ndarray] = self._load_brain_cache(crop_box)
        logger.info(
            "Encoded %d unique brain masks from %s",
            len(brain_cache),
            self.cfg.source_image_h5,
        )

        n_base = self._write_target(
            target_h5=self.cfg.target_h5,
            brain_cache=brain_cache,
            is_aug=False,
        )
        n_aug = 0
        n_v4_ones = 0
        if self.cfg.target_aug_h5 is not None:
            if not self.cfg.target_aug_h5.exists():
                raise FileNotFoundError(
                    f"aug-latent H5 declared but missing: {self.cfg.target_aug_h5}"
                )
            n_aug, n_v4_ones = self._write_aug_target(
                target_h5=self.cfg.target_aug_h5,
                brain_cache=brain_cache,
            )

        artifact_dir = self._write_decision(
            n_base=n_base,
            n_aug=n_aug,
            n_v4_ones=n_v4_ones,
        )
        logger.info("Decision written to %s", artifact_dir)
        return artifact_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_brain_cache(self, crop_box: tuple[int, int, int]) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        with h5py.File(self.cfg.source_image_h5, "r") as f:
            ids = self._read_id_array(f)
            origins = f["crop/origin"]
            for row, pid in enumerate(ids):
                brain = np.asarray(f["masks/brain"][row])  # (H, W, D), int8
                crop_origin = tuple(int(v) for v in origins[row])
                cache[pid] = _encode_brain_mask(
                    brain,
                    crop_origin=crop_origin,  # type: ignore[arg-type]
                    target_shape=crop_box,
                )
        return cache

    @staticmethod
    def _read_id_array(f: h5py.File) -> list[str]:
        raw = f["ids"][:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

    def _ensure_writable_dataset(
        self,
        h5: h5py.File,
        n_rows: int,
        path: str = "masks/brain_latent",
    ) -> h5py.Dataset:
        shape = (n_rows, 1, *LATENT_SPATIAL)
        if path in h5:
            if not self.cfg.overwrite:
                logger.info("`%s` already exists in target; will skip filled rows", path)
                return h5[path]
            del h5[path]
        ds = h5.create_dataset(
            path,
            shape=shape,
            dtype="int8",
            chunks=(1, 1, *LATENT_SPATIAL),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["units"] = "binary"
        ds.attrs["description"] = (
            "brain mask in latent space (max-pool 4 of masks/brain); "
            "written by vena-encode-brain-to-latent"
        )
        ds.attrs["producer"] = _PRODUCER
        return ds

    def _write_target(
        self,
        target_h5: Path,
        brain_cache: dict[str, np.ndarray],
        is_aug: bool,
    ) -> int:
        n_written = 0
        with h5py.File(target_h5, "a") as f:
            ids = self._read_id_array(f)
            n = len(ids)
            ds = self._ensure_writable_dataset(f, n_rows=n)
            for row, pid in enumerate(ids):
                if pid not in brain_cache:
                    raise KeyError(
                        f"patient_id {pid!r} (target row {row}) absent from "
                        f"source image H5 {self.cfg.source_image_h5}"
                    )
                if not self.cfg.overwrite and bool((ds[row] != 0).any()):
                    continue  # already populated; idempotent skip
                ds[row] = brain_cache[pid]
                n_written += 1
        return n_written

    def _write_aug_target(
        self,
        target_h5: Path,
        brain_cache: dict[str, np.ndarray],
    ) -> tuple[int, int]:
        n_written = 0
        n_v4_ones = 0
        ones = np.ones((1, *LATENT_SPATIAL), dtype=np.int8)
        with h5py.File(target_h5, "a") as f:
            ids = self._read_id_array(f)
            variants = [x.decode() if isinstance(x, bytes) else str(x) for x in f["variants"][:]]
            n = len(ids)
            ds = self._ensure_writable_dataset(f, n_rows=n)
            ds.attrs["v4_brain_synthesised_ones"] = True  # one-time flag
            for row, (pid, variant) in enumerate(zip(ids, variants, strict=True)):
                if not self.cfg.overwrite and bool((ds[row] != 0).any()):
                    continue
                if variant == "v4":
                    ds[row] = ones
                    n_v4_ones += 1
                else:
                    if pid not in brain_cache:
                        raise KeyError(
                            f"aug patient_id {pid!r} (row {row}, variant {variant}) "
                            f"absent from source image H5"
                        )
                    ds[row] = brain_cache[pid]
                n_written += 1
        if n_v4_ones:
            logger.warning(
                "%s: %d v4 rows written with synthesised-ones brain mask. "
                "Replay of the elastic+affine deformation from aug_params_json "
                "is a follow-up; the contrastive degrades to a full-volume Lp "
                "on those rows.",
                target_h5,
                n_v4_ones,
            )
        return n_written, n_v4_ones

    def _write_decision(
        self,
        n_base: int,
        n_aug: int,
        n_v4_ones: int,
    ) -> Path:
        utc = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        artifact_dir = self.cfg.artifacts_root / utc
        artifact_dir.mkdir(parents=True, exist_ok=True)
        decision = {
            "schema_version": _DECISION_SCHEMA,
            "produced_at": datetime.now(UTC).isoformat(),
            "producer": _PRODUCER,
            "target_h5": str(self.cfg.target_h5),
            "target_aug_h5": (str(self.cfg.target_aug_h5) if self.cfg.target_aug_h5 else None),
            "source_image_h5": str(self.cfg.source_image_h5),
            "n_rows_written_base": int(n_base),
            "n_rows_written_aug": int(n_aug),
            "n_v4_synthesised_ones": int(n_v4_ones),
            "overwrite": bool(self.cfg.overwrite),
        }
        (artifact_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        # Maintain a LATEST symlink for the routine.
        latest = self.cfg.artifacts_root / "LATEST"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(artifact_dir.name)
        except OSError as exc:
            logger.warning("could not write LATEST symlink (%s); continuing", exc)
        return artifact_dir


def _to_dict_for_yaml(cfg: Any) -> dict[str, Any]:
    """Helper for routine-config round-trip (used by tests)."""
    return json.loads(cfg.model_dump_json())
