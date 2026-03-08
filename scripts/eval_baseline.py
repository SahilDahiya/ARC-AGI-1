#!/usr/bin/env python3
"""Run baseline exact-match evaluation on ARC-AGI-1."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from arc_agi_1.baselines import BASELINES
from arc_agi_1.data import load_split
from arc_agi_1.scoring import SplitResult, evaluate_split, limit_tasks

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Path to ARC data directory containing training/ and evaluation/",
    )
    parser.add_argument(
        "--split",
        choices=["training", "evaluation", "all"],
        default="all",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES.keys()),
        default="copy_input",
        help="Baseline predictor to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N tasks per split (for smoke testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def git_commit() -> str:
    """Best-effort current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()
    predictor = BASELINES[args.baseline]

    splits = ["training", "evaluation"] if args.split == "all" else [args.split]
    split_results: list[SplitResult] = []
    for split in splits:
        tasks = load_split(args.data_root, split)
        tasks = limit_tasks(tasks, args.limit)
        result = evaluate_split(tasks, predictor, split)
        split_results.append(result)

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "baseline": args.baseline,
        "limit": args.limit,
        "commit": git_commit(),
        "results": [
            {
                **asdict(row),
                "solve_rate": round(row.solve_rate, 6),
            }
            for row in split_results
        ],
    }

    for row in split_results:
        solve_rate_pct = 100.0 * row.solve_rate
        print(
            f"{row.split}: solved {row.solved_tasks}/{row.total_tasks} "
            f"({solve_rate_pct:.2f}%)"
        )

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
        print(f"Wrote results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
