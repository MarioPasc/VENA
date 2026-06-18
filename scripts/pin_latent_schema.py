"""In-place re-stamp of ``schema_version`` on latent H5 files.

Audit §4.3 / §6.2 — every cohort's latent H5 historically stamped its own
schema_version (a mix of cohort-specific tags and the shared "2.0.0"). The
shared library now exposes a single ``LATENT_SCHEMA_VERSION`` constant
(``src/vena/data/h5/latent_domain/manifest.py``); this script aligns every
existing on-disk latent H5 to it without re-encoding.

The script accepts any number of latent H5 files (clean or augmented) and
overwrites the root attribute. It also bumps the embedded ``manifest_json``
version field so downstream validators stay consistent — the ``manifest_json``
field is rewritten *only* when its embedded ``schema_version`` field disagrees
with the new constant; the dataset structure is untouched.

Idempotent: re-running is a no-op.

Usage::

    python scripts/pin_latent_schema.py \
        --latent-h5 /path/to/UCSFPDGM_latents.h5 \
        --latent-h5 /path/to/UCSFPDGM_latents_aug.h5 \
        --dry-run

The script aborts (return 1) if any file is missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py

# Path bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vena.data.h5.augmented.latent_domain import AUG_LATENT_SCHEMA_VERSION
from vena.data.h5.latent_domain.manifest import LATENT_SCHEMA_VERSION

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _detect_target_version(f: h5py.File) -> str:
    """Pick the right pinned version based on the H5's existing manifest_json."""
    raw = f.attrs.get("manifest_json")
    if raw is None:
        # No embedded manifest — assume base latent.
        return LATENT_SCHEMA_VERSION
    parsed = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    extras = parsed.get("extras", {}) or {}
    augmentation_role = extras.get("augmentation_role", "")
    if augmentation_role == "latent_aug":
        return AUG_LATENT_SCHEMA_VERSION
    return LATENT_SCHEMA_VERSION


def _process_one(path: Path, dry_run: bool) -> dict[str, str]:
    mode = "r" if dry_run else "a"
    with h5py.File(path, mode) as f:
        if "schema_version" not in f.attrs:
            raise KeyError(f"{path}: file has no `schema_version` root attr")
        target = _detect_target_version(f)
        current = f.attrs["schema_version"]
        current_s = current.decode() if isinstance(current, bytes) else str(current)
        if current_s == target:
            return {"path": str(path), "before": current_s, "after": current_s, "changed": "no"}
        if dry_run:
            return {"path": str(path), "before": current_s, "after": target, "changed": "would"}
        f.attrs["schema_version"] = target
        # Update the embedded manifest_json's schema_version field too.
        raw = f.attrs.get("manifest_json")
        if raw is not None:
            parsed = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            if parsed.get("schema_version") != target:
                parsed["schema_version"] = target
                f.attrs["manifest_json"] = json.dumps(parsed)
        return {"path": str(path), "before": current_s, "after": target, "changed": "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-h5",
        action="append",
        required=True,
        type=Path,
        help="Path to a latent or aug-latent H5 (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    n_changed = 0
    for path in args.latent_h5:
        if not path.exists():
            logger.error("missing: %s", path)
            return 1
        try:
            row = _process_one(path, args.dry_run)
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            logger.error("%s: %s", path, exc)
            return 1
        verb = {"no": "ok", "yes": "STAMPED", "would": "WOULD STAMP"}[row["changed"]]
        logger.info(
            "%s %s: %s → %s",
            verb,
            Path(row["path"]).name,
            row["before"],
            row["after"],
        )
        if row["changed"] in ("yes", "would"):
            n_changed += 1
    logger.info("DONE — %d/%d files needed re-stamping", n_changed, len(args.latent_h5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
