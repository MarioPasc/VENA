"""Image-domain bank builder for the offline-augmentation routine.

Produces ``<COHORT>_image_aug.h5`` for one cohort: per (scan, variant) row
with the augmented modalities, the warped tumour mask (v4 only — copy for
v1/v2/v3), and the per-row sampled hyperparameters. The encode step is the
caller's responsibility — it pipes this H5 through
:class:`vena.data.h5.latent_domain.LatentH5Converter` in ``aug_mode=True``.

Invariants:

* Every row is stored at the common brain-centred crop box
  :data:`vena.data.h5.augmented.AUG_IMAGE_CROP_BOX`. The bank-builder
  crops/pads the source row once before augmenting. ``crop/origin`` is
  ``(0, 0, 0)`` for every row.
* Per-cohort dedup allowlist (from
  :mod:`vena.preflight.cohort_dedup`) is intersected with the source scan
  IDs before any row is written. ``splits/test`` patients are excluded
  by construction.
* Two-GPU sharding by scan: ``rank in {0, 1}, world_size = 2`` writes the
  even/odd source rows respectively. The routine engine merges the two
  ranks' H5s after both finish.
* Reproducibility: the worker-level RNG is seeded as
  ``seed XOR (rank << 16) XOR source_row_index XOR hash(variant)`` so a
  re-run produces byte-identical augmented volumes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torchio as tio

from vena.data.augment.offline.variants import (
    VARIANT_NAMES,
    make_variant,
)
from vena.data.h5.augmented import (
    AUG_IMAGE_CROP_BOX,
    AUG_IMAGE_SCHEMA_VERSION,
    assert_aug_image_h5_valid,
    build_aug_image_manifest,
)
from vena.data.h5.shared import (
    H5Writer,
    assign_row,
    now_iso_utc,
    resolve_git_sha,
    sha256_file,
)
from vena.model.autoencoder.maisi.preprocessing import CropPadSpec, apply_crop_pad

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "0.1.0"
_PRODUCER = f"vena.data.augment.offline.bank_builder:{_PRODUCER_VERSION}"

_INPUT_KEYS: tuple[str, ...] = ("t1pre", "t2", "flair")
_TARGET_KEY: str = "t1c"


def _variant_seed(base_seed: int, rank: int, source_row_index: int, variant: str) -> int:
    """Stable per-row, per-variant seed; reproducible across reruns."""
    variant_hash = int.from_bytes(hashlib.blake2b(variant.encode(), digest_size=4).digest(), "big")
    return int(np.uint32(base_seed ^ (rank << 16) ^ source_row_index ^ variant_hash))


def _box_native(
    arr: np.ndarray,
    crop_origin: tuple[int, int, int],
    target_box: tuple[int, int, int] = AUG_IMAGE_CROP_BOX,
) -> np.ndarray:
    """Crop/pad a single ``(H, W, D)`` native volume onto ``target_box``.

    Wraps :func:`apply_crop_pad`, which is the same primitive the encoder
    uses. The aug-image H5 stores everything at ``target_box`` so the
    downstream latent converter sees ``crop_origin=(0, 0, 0)`` and its
    crop+pad becomes the identity.
    """
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    spec = CropPadSpec(
        crop_origin=tuple(int(v) for v in crop_origin),  # type: ignore[arg-type]
        native_shape=tuple(int(v) for v in arr.shape),  # type: ignore[arg-type]
        target_shape=target_box,
    )
    boxed = apply_crop_pad(t, spec)
    return boxed[0, 0].numpy()


class OfflineAugBankBuilder:
    """One cohort × K variants × N train scans → ``<COHORT>_image_aug.h5``.

    Parameters
    ----------
    source_image_h5 : Path
        Clean image H5 (e.g. ``UCSFPDGM_image.h5``). The bank-builder reads
        rows by integer index; the ``ids`` array supplies the scan keys.
    output_path : Path
        Destination ``<COHORT>_image_aug_rank{rank}.h5`` (or
        ``<COHORT>_image_aug.h5`` when ``world_size=1``).
    cohort : str
        Cohort tag, must match the source H5's ``cohort`` root attr.
    modalities : list[str]
        Modalities to write into ``images/<m>``. Defaults to the canonical
        ``["t1pre", "t1c", "t2", "flair"]``.
    variants : list[str]
        Subset of :data:`VARIANT_NAMES` to write (default: all four).
    variant_hyperparams : dict[str, dict[str, Any]]
        Per-variant overrides passed to :func:`make_variant`.
    aug_config_json : str
        JSON dump of the aug-pipeline YAML used for this build. Round-trips
        verbatim into the root attr.
    aug_config_sha256 : str
        SHA-256 of the aug-pipeline YAML.
    dedup_allowlist : set[str] | None
        Patient-ID allowlist from
        :func:`vena.preflight.cohort_dedup.build_allowlists`. When ``None``,
        all non-test scans are eligible.
    world_size, rank : int
        Scan-level sharding (``i % world_size == rank``).
    seed : int
        Base RNG seed.
    overwrite : bool
        Whether to unlink an existing output file.
    """

    def __init__(
        self,
        *,
        source_image_h5: Path,
        output_path: Path,
        cohort: str,
        modalities: list[str] | None = None,
        variants: list[str] | None = None,
        variant_hyperparams: dict[str, dict[str, Any]] | None = None,
        aug_config_json: str,
        aug_config_sha256: str,
        dedup_allowlist: set[str] | None = None,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        overwrite: bool = False,
        limit_source_rows: int | None = None,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}); got {rank}")
        self.source_image_h5 = Path(source_image_h5)
        self.output_path = Path(output_path)
        self.cohort = cohort
        self.modalities = list(modalities or ["t1pre", "t1c", "t2", "flair"])
        self.variants = list(variants or VARIANT_NAMES)
        unknown = [v for v in self.variants if v not in VARIANT_NAMES]
        if unknown:
            raise ValueError(f"unknown variants {unknown!r}; allowed: {VARIANT_NAMES}")
        self.variant_hyperparams = dict(variant_hyperparams or {})
        self.aug_config_json = aug_config_json
        self.aug_config_sha256 = aug_config_sha256
        self.dedup_allowlist = dedup_allowlist
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.overwrite = bool(overwrite)
        self.limit_source_rows = None if limit_source_rows is None else int(limit_source_rows)

    # ------------------------------------------------------------------ public

    def build(self) -> Path:
        """Build the aug-image H5 for this rank's shard."""
        if not self.source_image_h5.is_file():
            raise FileNotFoundError(f"source image H5 not found: {self.source_image_h5}")

        rows = self._resolve_rows()
        if not rows:
            raise RuntimeError(
                f"no augmentable rows for cohort {self.cohort!r} "
                f"(rank {self.rank}/{self.world_size}); check splits + dedup allowlist"
            )
        n_rows = len(rows) * len(self.variants)

        manifest = build_aug_image_manifest(self.cohort, self.modalities)
        with h5py.File(self.source_image_h5, "r") as src:
            extra_root_attrs = self._collect_extra_root_attrs(src)

        timestamp = now_iso_utc()
        git_sha = resolve_git_sha()

        with H5Writer(
            self.output_path,
            manifest=manifest,
            config_json=self._config_json(),
            producer=_PRODUCER,
            created_at=timestamp,
            git_sha=git_sha,
            overwrite=self.overwrite,
            extra_root_attrs=extra_root_attrs,
        ) as w:
            self._stamp_aug_root_attrs(w)

            # ids, source_row_index, variants, aug_params_json: 1-D per row.
            ids_dset = w.create_1d(manifest.get("ids"), n=n_rows)
            srci_dset = w.create_1d(manifest.get("source_row_index"), n=n_rows)
            variants_dset = w.create_1d(manifest.get("variants"), n=n_rows)
            params_dset = w.create_1d(manifest.get("aug_params_json"), n=n_rows)

            # Stacked image/mask datasets at the unified crop box.
            image_dsets = {
                slug: w.create_stacked(
                    manifest.get(f"images/{slug}"),
                    n=n_rows,
                    spatial_shape=AUG_IMAGE_CROP_BOX,
                )
                for slug in self.modalities
            }
            mask_dset = w.create_stacked(
                manifest.get("masks/tumor"),
                n=n_rows,
                spatial_shape=AUG_IMAGE_CROP_BOX,
            )

            # crop/origin is required by LatentH5Converter._assert_source_compatibility
            # but is the all-zeros vector for every row here.
            crop_origin_dset = w.file.create_dataset(
                "crop/origin",
                shape=(n_rows, 3),
                dtype=np.int32,
            )
            crop_origin_dset[:] = 0
            crop_origin_dset.attrs["units"] = "voxels"
            crop_origin_dset.attrs["description"] = (
                "Per-row crop origin; identically zero because the aug-image H5 IS the crop box."
            )
            crop_origin_dset.attrs["dtype"] = "int32"

            # Per-row loop.
            with h5py.File(self.source_image_h5, "r") as src:
                self._encode_rows(
                    src=src,
                    rows=rows,
                    ids_dset=ids_dset,
                    srci_dset=srci_dset,
                    variants_dset=variants_dset,
                    params_dset=params_dset,
                    image_dsets=image_dsets,
                    mask_dset=mask_dset,
                )

        assert_aug_image_h5_valid(self.output_path, self.cohort, self.modalities)
        logger.info(
            "wrote aug-image H5: %s (cohort=%s n_rows=%d rank=%d/%d)",
            self.output_path,
            self.cohort,
            n_rows,
            self.rank,
            self.world_size,
        )
        return self.output_path

    # ------------------------------------------------------------------ private

    def _config_json(self) -> str:
        return json.dumps(
            {
                "source_image_h5": str(self.source_image_h5),
                "output_path": str(self.output_path),
                "cohort": self.cohort,
                "modalities": self.modalities,
                "variants": self.variants,
                "world_size": self.world_size,
                "rank": self.rank,
                "seed": self.seed,
                "aug_config_sha256": self.aug_config_sha256,
                "schema_version": AUG_IMAGE_SCHEMA_VERSION,
            },
            sort_keys=True,
        )

    def _stamp_aug_root_attrs(self, w: H5Writer) -> None:
        f = w.file
        f.attrs["source_image_h5_path"] = str(self.source_image_h5)
        f.attrs["source_image_h5_sha256"] = sha256_file(self.source_image_h5)
        f.attrs["aug_config_json"] = self.aug_config_json
        f.attrs["aug_config_sha256"] = self.aug_config_sha256
        f.attrs["variants_json"] = json.dumps(self.variants)
        f.attrs["seed"] = self.seed
        f.attrs["world_size"] = self.world_size
        f.attrs["rank"] = self.rank

    @staticmethod
    def _collect_extra_root_attrs(src: h5py.File) -> dict[str, Any]:
        """Copy the schema-v2 source attrs that ``LatentH5Converter`` reads."""
        keys = ("split_role", "longitudinal", "label_system", "crop_box", "orientation")
        out: dict[str, Any] = {}
        for k in keys:
            if k in src.attrs:
                out[k] = src.attrs[k]
        return out

    def _resolve_rows(self) -> list[int]:
        """Return the source-row indices this rank should augment.

        Logic:
        1. Read ``ids`` + ``splits/test`` from the source image H5.
        2. Build the train+val pool = all_ids \\ test_patients.
        3. Intersect with ``dedup_allowlist`` when set.
        4. Apply scan-level shard: ``i for i in pool if i % world_size == rank``.
        """
        with h5py.File(self.source_image_h5, "r") as src:
            ids_raw = src["ids"][:]
            ids = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in ids_raw]
            test_set: set[str] = set()
            if "splits/test" in src:
                raw = src["splits/test"][:]
                test_set = {
                    v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw
                }
            # patient/scan mapping for longitudinal cohorts: row → patient key
            row_to_patient: list[str] = list(ids)
            if "patients/offsets" in src and "patients/keys" in src:
                offsets = np.asarray(src["patients/offsets"][:], dtype=np.int64)
                keys_raw = src["patients/keys"][:]
                keys = [
                    v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in keys_raw
                ]
                row_to_patient = [""] * len(ids)
                for k_idx, key in enumerate(keys):
                    for r in range(int(offsets[k_idx]), int(offsets[k_idx + 1])):
                        row_to_patient[r] = key

        pool: list[int] = []
        for row_idx, scan_id in enumerate(ids):
            patient = row_to_patient[row_idx] if row_to_patient[row_idx] else scan_id
            # Exclude test patients: splits/test may store patient or scan ID
            # depending on cohort; both forms are checked.
            if scan_id in test_set or patient in test_set:
                continue
            if self.dedup_allowlist is not None and patient not in self.dedup_allowlist:
                continue
            pool.append(row_idx)

        # Scan-level shard
        sharded = [r for r in pool if r % self.world_size == self.rank]
        if self.limit_source_rows is not None:
            sharded = sharded[: self.limit_source_rows]
        return sharded

    def _encode_rows(
        self,
        *,
        src: h5py.File,
        rows: list[int],
        ids_dset: h5py.Dataset,
        srci_dset: h5py.Dataset,
        variants_dset: h5py.Dataset,
        params_dset: h5py.Dataset,
        image_dsets: dict[str, h5py.Dataset],
        mask_dset: h5py.Dataset,
    ) -> None:
        ids_raw = src["ids"][:]
        ids_all = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in ids_raw]
        out_row = 0
        n_total = len(rows) * len(self.variants)
        for src_idx in rows:
            scan_id = ids_all[src_idx]
            crop_origin = tuple(int(v) for v in src["crop/origin"][src_idx])
            volumes_boxed: dict[str, np.ndarray] = {}
            for slug in self.modalities:
                arr = np.asarray(src[f"images/{slug}"][src_idx], dtype=np.float32)
                volumes_boxed[slug] = _box_native(arr, crop_origin)
            mask_arr = np.asarray(src["masks/tumor"][src_idx], dtype=np.int8)
            mask_boxed = _box_native(mask_arr, crop_origin).astype(np.int8)

            for variant in self.variants:
                seed = _variant_seed(self.seed, self.rank, src_idx, variant)
                torch.manual_seed(seed)
                np.random.seed(seed)
                # TorchIO uses `random.random()` for some transforms
                # (e.g. RandomMotion, RandomElasticDeformation); seed it too
                # or those variants are not byte-reproducible across re-runs.
                random.seed(seed)
                subject = self._build_subject(volumes_boxed, mask_boxed)
                transform = make_variant(variant, self.variant_hyperparams.get(variant))
                augmented = transform(subject)

                ids_dset[out_row] = scan_id
                srci_dset[out_row] = src_idx
                variants_dset[out_row] = variant
                params_dset[out_row] = json.dumps(self._snapshot_applied_params(augmented, variant))
                for slug in self.modalities:
                    assign_row(
                        image_dsets[slug],
                        out_row,
                        augmented[slug].data[0].numpy().astype(np.float32, copy=False),
                    )
                assign_row(
                    mask_dset,
                    out_row,
                    augmented["tumor"].data[0].numpy().astype(np.int8, copy=False),
                )
                out_row += 1
                if out_row % 32 == 0 or out_row == n_total:
                    logger.info(
                        "bank-build %s (rank %d/%d): %d/%d rows",
                        self.cohort,
                        self.rank,
                        self.world_size,
                        out_row,
                        n_total,
                    )

    def _build_subject(
        self,
        volumes_boxed: dict[str, np.ndarray],
        mask_boxed: np.ndarray,
    ) -> tio.Subject:
        members: dict[str, Any] = {}
        for slug in self.modalities:
            arr = volumes_boxed[slug]
            members[slug] = tio.ScalarImage(tensor=torch.from_numpy(arr).unsqueeze(0))
        members["tumor"] = tio.LabelMap(tensor=torch.from_numpy(mask_boxed).unsqueeze(0).long())
        return tio.Subject(**members)

    @staticmethod
    def _snapshot_applied_params(subject: tio.Subject, variant: str) -> dict[str, Any]:
        """Read the history of TorchIO transforms that fired for this Subject.

        TorchIO appends a record per applied transform to
        ``subject.history``; each record carries the random parameters
        sampled at apply time. We dump the public attribute set so the
        per-row provenance JSON captures exactly the augmentation that
        was applied.
        """
        snapshot: dict[str, Any] = {"variant": variant, "transforms": []}
        for record in getattr(subject, "history", []):
            params = {
                k: _coerce_json(v)
                for k, v in record.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            params["name"] = type(record).__name__
            snapshot["transforms"].append(params)
        return snapshot


def _coerce_json(value: Any) -> Any:
    """Best-effort conversion of TorchIO record values into JSON-serialisable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return {"_ndarray_shape": list(value.shape), "_dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return _coerce_json(value.detach().cpu().numpy())
    return repr(value)


def merge_aug_image_h5_shards(
    shards: Iterable[Path],
    merged_path: Path,
    cohort: str,
    modalities: list[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Merge per-rank aug-image H5 shards into a single cohort H5.

    Same-cohort shards are validated to share schema_version, modality list,
    aug_config_sha256, and source_image_h5_sha256; rows are concatenated in
    rank order. Used by the routine engine after the two-GPU run finishes.
    """
    shards = [Path(p) for p in shards]
    if not shards:
        raise ValueError("merge_aug_image_h5_shards needs at least one shard")
    merged_path = Path(merged_path)
    if merged_path.exists() and not overwrite:
        raise FileExistsError(f"merged output already exists: {merged_path}. Pass overwrite=True.")
    if merged_path.exists():
        merged_path.unlink()

    # Sanity: all shards must share the same schema/cohort/aug-config.
    with h5py.File(shards[0], "r") as f0:
        ref_schema = str(f0.attrs["schema_version"])
        ref_cohort = str(f0.attrs["cohort"])
        ref_aug_sha = str(f0.attrs["aug_config_sha256"])
        ref_src_sha = str(f0.attrs["source_image_h5_sha256"])
        ref_src_path = str(f0.attrs["source_image_h5_path"])
    for s in shards[1:]:
        with h5py.File(s, "r") as fs:
            for attr, ref in (
                ("schema_version", ref_schema),
                ("cohort", ref_cohort),
                ("aug_config_sha256", ref_aug_sha),
                ("source_image_h5_sha256", ref_src_sha),
            ):
                got = str(fs.attrs[attr])
                if got != ref:
                    raise ValueError(f"shard {s} disagrees on {attr}: {got!r} != {ref!r}")

    total_rows = 0
    for s in shards:
        with h5py.File(s, "r") as fs:
            total_rows += int(fs["ids"].shape[0])
    logger.info(
        "merging %d shards into %s (total rows=%d)",
        len(shards),
        merged_path,
        total_rows,
    )

    # Replay the writer machinery over the merged file. Reuse the manifest
    # from the first shard.
    manifest = build_aug_image_manifest(cohort, modalities)
    with h5py.File(shards[0], "r") as fs0:
        extra_root_attrs = {
            k: fs0.attrs[k]
            for k in ("split_role", "longitudinal", "label_system", "crop_box", "orientation")
            if k in fs0.attrs
        }
        config_json = str(fs0.attrs["config_json"])
        aug_config_json = str(fs0.attrs["aug_config_json"])
        seed = int(fs0.attrs["seed"])
        variants_json = str(fs0.attrs["variants_json"])
    timestamp = now_iso_utc()
    git_sha = resolve_git_sha()
    with H5Writer(
        merged_path,
        manifest=manifest,
        config_json=config_json,
        producer=_PRODUCER,
        created_at=timestamp,
        git_sha=git_sha,
        overwrite=overwrite,
        extra_root_attrs=extra_root_attrs,
    ) as w:
        f = w.file
        f.attrs["aug_config_json"] = aug_config_json
        f.attrs["aug_config_sha256"] = ref_aug_sha
        f.attrs["variants_json"] = variants_json
        f.attrs["source_image_h5_path"] = ref_src_path
        f.attrs["source_image_h5_sha256"] = ref_src_sha
        f.attrs["seed"] = seed
        f.attrs["world_size"] = 1  # merged; sharding is past tense
        f.attrs["rank"] = 0
        f.attrs["merged_from"] = json.dumps([str(s) for s in shards])

        # Allocate the merged datasets.
        ids_dset = w.create_1d(manifest.get("ids"), n=total_rows)
        srci_dset = w.create_1d(manifest.get("source_row_index"), n=total_rows)
        variants_dset = w.create_1d(manifest.get("variants"), n=total_rows)
        params_dset = w.create_1d(manifest.get("aug_params_json"), n=total_rows)
        image_dsets = {
            slug: w.create_stacked(
                manifest.get(f"images/{slug}"),
                n=total_rows,
                spatial_shape=AUG_IMAGE_CROP_BOX,
            )
            for slug in modalities
        }
        mask_dset = w.create_stacked(
            manifest.get("masks/tumor"),
            n=total_rows,
            spatial_shape=AUG_IMAGE_CROP_BOX,
        )
        crop_origin_dset = f.create_dataset(
            "crop/origin",
            shape=(total_rows, 3),
            dtype=np.int32,
        )
        crop_origin_dset[:] = 0
        crop_origin_dset.attrs["units"] = "voxels"
        crop_origin_dset.attrs["description"] = "Per-row crop origin (all zeros)."
        crop_origin_dset.attrs["dtype"] = "int32"

        # Copy rows from each shard in order.
        out_row = 0
        for s in shards:
            with h5py.File(s, "r") as fs:
                n = int(fs["ids"].shape[0])
                if n == 0:
                    continue
                end = out_row + n
                ids_dset[out_row:end] = np.asarray(fs["ids"][:], dtype=object)
                srci_dset[out_row:end] = fs["source_row_index"][:]
                variants_dset[out_row:end] = np.asarray(fs["variants"][:], dtype=object)
                params_dset[out_row:end] = np.asarray(fs["aug_params_json"][:], dtype=object)
                for slug in modalities:
                    image_dsets[slug][out_row:end] = fs[f"images/{slug}"][:]
                mask_dset[out_row:end] = fs["masks/tumor"][:]
                out_row = end

    assert_aug_image_h5_valid(merged_path, cohort, modalities)
    logger.info("merge complete: %s", merged_path)
    return merged_path
