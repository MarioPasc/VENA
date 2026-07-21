"""CLI entrypoint for the ρ_S normalisation audit preflight.

Usage
-----
    vena-preflight-rho-s-norm-audit <config.yaml>
    python -m routines.preflights.rho_s_norm_audit.cli <config.yaml>
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse one positional YAML argument and run the engine."""
    parser = argparse.ArgumentParser(
        description="ρ_S normalisation audit — re-run spatial-residual ρ_S with "
        "forced-percentile normalisation to detect the 99.5 vs 99.95 confound."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file.")
    args = parser.parse_args(argv)

    # Heavy imports deferred to avoid slowing `--help`.
    from vena.preflight.rho_s_norm_audit import RhoSNormAuditConfig, RhoSNormAuditEngine

    cfg = RhoSNormAuditConfig.from_yaml(args.config)
    artifact_dir = RhoSNormAuditEngine(cfg).run()
    print(f"Artifact: {artifact_dir}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
