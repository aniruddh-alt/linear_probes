"""Thin CLI wrapper over the experiment runner API."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sonde",
        description="sonde — a mech interp toolkit for activation extraction and linear probing.",
        usage="sonde <config.yaml> [-o key=val ...]\n       sonde run -c <config.yaml> [-o key=val ...]",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Dot-notation override (e.g. probe.learning_rate=0.01)",
    )
    return parser


def _parse_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in raw_overrides:
        if "=" not in item:
            print(
                f"Warning: ignoring malformed override '{item}' (missing '=')",
                file=sys.stderr,
            )
            continue
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from sonde.runners.experiment_runner import run_experiment

    overrides = _parse_overrides(args.override)
    result = run_experiment(config_path=args.config, overrides=overrides or None)
    print(json.dumps(result.summary, indent=2))
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
