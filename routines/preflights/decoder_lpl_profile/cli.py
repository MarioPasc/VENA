"""CLI entry point for the ``decoder_lpl_profile`` preflight."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from routines.preflights.decoder_lpl_profile.engine import (
    DecoderLplProfileRoutineEngine,
)
from vena.preflight.decoder_lpl_profile import (
    DecoderLplProfileConfig,
    aggregate,
    update_latest_symlink,
)


def _run_subcommand(args: argparse.Namespace) -> int:
    cfg = DecoderLplProfileConfig.from_yaml(args.config)
    # CLI overrides for the 4-way shard fan-out — saves writing 4 YAMLs.
    if args.shard is not None or args.n_shards is not None:
        update = {}
        if args.shard is not None:
            update["shard_id"] = args.shard
        if args.n_shards is not None:
            update["n_shards"] = args.n_shards
        cfg = cfg.model_copy(update=update)
    engine = DecoderLplProfileRoutineEngine(cfg=cfg, config_yaml_path=args.config)
    out_dir = engine.run()
    logging.getLogger(__name__).info("done — artefacts at %s", out_dir)
    return 0


def _aggregate_subcommand(args: argparse.Namespace) -> int:
    out_dir = Path(args.artifact_dir)
    cohorts = list(args.cohorts) if args.cohorts else []
    decision = aggregate(out_dir, cohorts=cohorts)
    update_latest_symlink(out_dir)
    logging.getLogger(__name__).info(
        "aggregate done: allowed=%s A=%s t_min=%.3f",
        decision.allowed_variants,
        decision.A_recommended,
        decision.t_min,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vena-preflight-decoder-lpl-profile",
        description=(
            "Decoder-LPL profiling preflight (§4.1 + §4.2 + §4.7b)."
            " Single-stream and 4-way sharded runs supported."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the per-shard sweep (default).")
    run_p.add_argument("config", type=Path, help="Path to the YAML config.")
    run_p.add_argument(
        "--shard", type=int, default=None, help="0-indexed shard id (overrides cfg.shard_id)."
    )
    run_p.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Total number of shards (overrides cfg.n_shards).",
    )
    run_p.set_defaults(func=_run_subcommand)

    agg_p = sub.add_parser(
        "aggregate",
        help="Aggregate shard CSVs → decision.json + report.md + figures.",
    )
    agg_p.add_argument(
        "artifact_dir", type=Path, help="Timestamped artefact dir under output_root."
    )
    agg_p.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help="Cohort names that should appear in the decision report.",
    )
    agg_p.set_defaults(func=_aggregate_subcommand)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
