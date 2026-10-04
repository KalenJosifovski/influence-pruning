"""Command-line interface for the influence-guided pruning study."""

import argparse
import sys

from influence_pruning.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ipp",
        description=(
            "Conditional boostin pruning: does a full-pooled attribution ranking predict the "
            "effect of removing batches from that same pool?"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("audit", "standardisation, contracts, and partitions; no model fits"),
        ("dry-run", "capped end-to-end run exercising the complete artifact contract"),
        ("benchmark", "one scoring/full/pruned fit and projected run budget"),
        ("boostin-prune", "run the conditional BoostIn pruning episode"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--config", required=True, help="YAML run configuration")
    verify_parser = subparsers.add_parser("verify", help="read-only integrity check of a run")
    verify_parser.add_argument("--run", required=True, help="run directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    :param argv: argument list; defaults to ``sys.argv[1:]``.
    :returns: process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        from influence_pruning.run import verify

        facts = verify(args.run)
        print(f"{args.run}: ok (kind={facts['kind']}, hash={facts['config_hash']})")
        return 0

    config = load_config(args.config)
    if args.command == "audit":
        from influence_pruning.run import run_audit

        path = run_audit(config)
    elif args.command == "dry-run":
        from influence_pruning.run import run_dry_run

        path = run_dry_run(config)
    elif args.command == "benchmark":
        from influence_pruning.run import run_benchmark

        path = run_benchmark(config)
    else:
        from influence_pruning.run import run_prune

        path = run_prune(config)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
