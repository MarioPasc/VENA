"""Drop legacy split-tree aliases on cohort H5s.

Audit §4.3 / §6.2 — some cohorts ship both flat ``splits/{train,val,test}``
and the CV-style ``splits/cv/fold_N/{train,val}``; test-only cohorts ship
``splits/test`` plus ``splits/cv/fold_0/{train,val}`` aliases where val ≡
test and train is empty. This script reduces every H5 to the canonical
layout via :func:`vena.data.h5.shared.splits.normalize_splits`:

* ``role="cv"`` → keep only ``splits/test`` + ``splits/cv/fold_N/*``.
* ``role="test_only"`` → keep only ``splits/test``.

The role is inferred from the corpus JSON (``role`` field per cohort) when
``--corpus`` is supplied; otherwise must be passed via ``--role``.

Idempotent. Writes a per-file delta CSV at
``<basename>.splits_normalize.csv`` listing dropped + kept H5 paths.

Usage::

    python scripts/normalize_splits_inplace.py \
        --corpus routines/fm/train/configs/corpus/corpus_picasso.json \
        --dry-run

    python scripts/normalize_splits_inplace.py \
        --h5 /path/to/X_image.h5 \
        --role cv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Literal

# Path bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vena.data.h5.shared.splits import normalize_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

Role = Literal["cv", "test_only"]


def _enumerate_from_corpus(corpus_path: Path) -> list[tuple[Path, Role]]:
    """Walk a corpus JSON; emit (h5_path, role) for every image + latent H5."""
    with corpus_path.open("r") as f:
        corpus = json.load(f)
    targets: list[tuple[Path, Role]] = []
    cohorts = corpus.get("cohorts") or corpus  # tolerate top-level list / dict
    if isinstance(cohorts, dict):
        items = cohorts.items()
    else:
        items = [(c.get("name", ""), c) for c in cohorts]
    for name, entry in items:
        role: Role = entry.get("role", "cv")
        for key in ("image_h5", "latent_h5", "latent_aug_h5"):
            v = entry.get(key)
            if not v:
                continue
            p = Path(v)
            if p.exists():
                targets.append((p, role))
            else:
                logger.warning("corpus %s: missing %s for %s", corpus_path.name, key, name)
    return targets


def _process_one(path: Path, role: Role, dry_run: bool) -> dict[str, list[str]]:
    result = normalize_splits(path, role, dry_run=dry_run)
    csv_path = path.with_suffix(path.suffix + ".splits_normalize.csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["status", "h5_node"])
        w.writeheader()
        for node in result["removed"]:
            w.writerow({"status": "removed", "h5_node": node})
        for node in result["kept"]:
            w.writerow({"status": "kept", "h5_node": node})
    verb = "WOULD DROP" if dry_run else "dropped"
    logger.info(
        "%s %s (role=%s): %s %d nodes (%s); kept %d",
        path.parent.name,
        path.name,
        role,
        verb,
        len(result["removed"]),
        ",".join(result["removed"]) or "—",
        len(result["kept"]),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, help="Corpus JSON; enumerates every cohort H5.")
    parser.add_argument("--h5", action="append", type=Path, help="Explicit H5 path (repeatable).")
    parser.add_argument("--role", choices=("cv", "test_only"), help="Role for --h5 paths.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    targets: list[tuple[Path, Role]] = []
    if args.corpus:
        targets.extend(_enumerate_from_corpus(args.corpus))
    if args.h5:
        if not args.role:
            parser.error("--role is required when using --h5")
        for p in args.h5:
            targets.append((p, args.role))  # type: ignore[arg-type]
    if not targets:
        parser.error("pass --corpus or --h5")

    n_changed = 0
    for path, role in targets:
        try:
            result = _process_one(path, role, args.dry_run)
        except (OSError, KeyError) as exc:
            logger.error("%s: %s", path, exc)
            return 1
        if result["removed"]:
            n_changed += 1
    logger.info("DONE — %d/%d files had legacy split nodes", n_changed, len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
