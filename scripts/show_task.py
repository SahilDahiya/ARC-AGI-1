#!/usr/bin/env python3
"""Render an ARC task in the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from arckit import vis

from arc_agi_1.data import Task, load_task_file

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        help="Task id like '007bbfb7' or a path to a task JSON file.",
    )
    parser.add_argument(
        "--split",
        choices=["training", "evaluation"],
        default="training",
        help="Split to search when task is given as an id.",
    )
    parser.add_argument(
        "--digits",
        action="store_true",
        help="Render digits instead of ANSI colors.",
    )
    return parser.parse_args()


def resolve_task_path(task_arg: str, split: str) -> Path:
    candidate = Path(task_arg)
    if candidate.is_file():
        return candidate.resolve()

    task_path = DATA_ROOT / split / f"{task_arg}.json"
    if task_path.is_file():
        return task_path

    raise FileNotFoundError(f"Task not found: {task_arg}")


def render_grid(grid: list[list[int]], *, digits: bool) -> str:
    if not digits:
        raise ValueError("render_grid() is only used for --digits mode.")

    lines: list[str] = []
    for row in grid:
        lines.append(" ".join(str(cell) for cell in row))
    return "\n".join(lines)


def print_pair(title: str, pair: dict[str, list[list[int]]], *, digits: bool) -> None:
    input_grid = pair["input"]
    output_grid = pair["output"]
    print(title)
    print(f"input  ({len(input_grid)}x{len(input_grid[0])})")
    if digits:
        print(render_grid(input_grid, digits=True))
    else:
        vis.print_grid(np.array(input_grid, dtype=int))
    print()
    print(f"output ({len(output_grid)}x{len(output_grid[0])})")
    if digits:
        print(render_grid(output_grid, digits=True))
    else:
        vis.print_grid(np.array(output_grid, dtype=int))
    print()


def print_task(task_id: str, task: Task, *, digits: bool) -> None:
    print(f"task: {task_id}")
    print()

    for index, pair in enumerate(task["train"], start=1):
        print_pair(f"train[{index}]", pair, digits=digits)

    for index, pair in enumerate(task["test"], start=1):
        print_pair(f"test[{index}]", pair, digits=digits)


def main() -> int:
    args = parse_args()
    task_path = resolve_task_path(args.task, args.split)
    task = load_task_file(task_path)
    print_task(task_path.stem, task, digits=args.digits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
