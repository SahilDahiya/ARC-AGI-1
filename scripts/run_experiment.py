#!/usr/bin/env python3
"""Run an experiment with canonical local registry logging."""

from __future__ import annotations

import argparse
from pathlib import Path

from arc_agi_1.experiments import run_logged_command


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="Experiment family, for example task_conditioned_baseline")
    parser.add_argument("--label", required=True, help="Human-meaningful short run label")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory that stores the append-only registry and run directories.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--', for example -- .venv/bin/python scripts/train_task_conditioned_baseline.py ...",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.command:
        raise ValueError("Missing command. Use '--' before the experiment command.")

    command = list(args.command)
    if command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing command after '--'.")

    project_root = _project_root()
    results_root = (project_root / args.results_root).resolve()

    record = run_logged_command(
        command=command,
        family=args.family,
        label=args.label,
        results_root=results_root,
        project_root=project_root,
        prepare_known_scripts=True,
    )

    print(f"Run ID: {record['run_id']}")
    print(f"Status: {record['status']}")
    print(f"Run dir: {record['run_dir']}")
    return int(record["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())
