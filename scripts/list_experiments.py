#!/usr/bin/env python3
"""List locally recorded experiments from the append-only registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from arc_agi_1.experiments import format_registry_rows, load_registry


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing registry.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Show at most the most recent N runs.")
    parser.add_argument("--family", default=None, help="Optional family filter.")
    parser.add_argument(
        "--status",
        choices=["running", "completed", "failed"],
        default=None,
        help="Optional status filter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = (_project_root() / args.results_root / "registry.jsonl").resolve()
    rows = load_registry(registry_path)
    if args.family is not None:
        rows = [row for row in rows if row.get("family") == args.family]
    if args.status is not None:
        rows = [row for row in rows if row.get("status") == args.status]

    if not rows:
        print("No experiment records found.")
        return 0

    print(format_registry_rows(rows, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
