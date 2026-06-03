"""One-off ETL: build a UPENN-GBM ↔ BraTS-2021 lookup CSV.

The UPENN-GBM cohort does not ship a clinical CSV with a ``BraTS21 ID``
column (the analogue of UCSF-PDGM's ``UCSF-PDGM-metadata_v5.csv``).
The cross-cohort deduplication preflight (``routines/preflights/cohort_dedup``)
needs per-patient BraTS-2021 IDs stored in ``metadata/brats21_id`` of the
image-H5; the converter joins that field through ``metadata_csv``.

This script produces ``UPENN-GBM_brats21_lookup_v1.csv`` by filtering
``BraTS2021_MappingToTCIA.xlsx`` to the rows whose ``Data Collection`` is
``"UPENN-GBM"`` or ``"UPENN-GBM_Additional"`` and the portal ID looks like
``UPENN-GBM-NNNNN_NN`` (matches the on-disk patient directory naming under
``images_structural/``).

Columns: ``patient_id, brats21_id, brats21_data_collection``.

The CSV is treated as a versioned source artefact (``_v1`` in the filename).
Re-run only when the upstream xlsx is re-released.

Usage
-----
    ~/.conda/envs/vena/bin/python scripts/preprocess/build_upenn_gbm_brats21_lookup.py \
        --xlsx /media/mpascual/MeningD2/GLIOMA/BraTS2021_MappingToTCIA.xlsx \
        --out  /media/mpascual/MeningD2/GLIOMA/UPENN_GBM/metadata/UPENN-GBM_brats21_lookup_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

from vena.preflight.cohort_dedup.xlsx import parse_brats2021_mapping

logger = logging.getLogger(__name__)

_UPENN_COLLECTIONS: tuple[str, ...] = ("UPENN-GBM", "UPENN-GBM_Additional")
_PORTAL_RE = re.compile(r"^UPENN-GBM-\d{5}_\d{2}$")


def build_lookup(xlsx_path: Path) -> list[dict[str, str]]:
    """Filter the BraTS-21 mapping to UPenn rows.

    Returns one row per matched portal-id; emits a warning for any UPenn
    row whose ``portal_id`` does not match the on-disk pattern.
    """
    mapping = parse_brats2021_mapping(xlsx_path)
    out: list[dict[str, str]] = []
    n_skipped_format = 0
    for row in mapping.rows:
        if row.data_collection not in _UPENN_COLLECTIONS:
            continue
        portal = (row.portal_id or "").strip()
        if not _PORTAL_RE.match(portal):
            n_skipped_format += 1
            logger.warning(
                "skip BraTS-21 row %s (collection=%s) — portal_id %r does not match UPENN-GBM-NNNNN_NN",
                row.brats21_id,
                row.data_collection,
                portal,
            )
            continue
        out.append(
            {
                "patient_id": portal,
                "brats21_id": row.brats21_id,
                "brats21_data_collection": row.data_collection,
            }
        )
    logger.info(
        "matched %d UPenn rows (skipped %d for portal_id format)",
        len(out),
        n_skipped_format,
    )
    return out


def write_lookup(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "brats21_id", "brats21_data_collection"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), out_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="build_upenn_gbm_brats21_lookup")
    p.add_argument(
        "--xlsx",
        type=Path,
        default=Path("/media/mpascual/MeningD2/GLIOMA/BraTS2021_MappingToTCIA.xlsx"),
        help="BraTS-2021 ↔ TCIA mapping xlsx",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/media/mpascual/MeningD2/GLIOMA/UPENN_GBM/metadata/UPENN-GBM_brats21_lookup_v1.csv"
        ),
        help="Destination CSV path",
    )
    args = p.parse_args(argv)

    if not args.xlsx.is_file():
        logger.error("xlsx not found: %s", args.xlsx)
        return 2

    rows = build_lookup(args.xlsx)
    # The xlsx has ~562 rows for Data Collection ∈ {UPENN-GBM, UPENN-GBM_Additional};
    # ~115 of those (the "Additional" set added post-TCIA) carry the placeholder
    # ``portal_id = "new-not-previously-in-TCIA"`` and are unmatchable to on-disk
    # patient IDs. The realistic floor is ~440 ID-based matches.
    if len(rows) < 400:
        logger.error("lookup has only %d rows; expected ≥400 from xlsx", len(rows))
        return 3

    write_lookup(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
