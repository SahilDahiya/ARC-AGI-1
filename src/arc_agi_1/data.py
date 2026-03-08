"""Data loading helpers for ARC-AGI-1 tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

Grid: TypeAlias = list[list[int]]
Pair: TypeAlias = dict[str, Grid]
Task: TypeAlias = dict[str, list[Pair]]
TaskMap: TypeAlias = dict[str, Task]


def load_task_file(path: Path) -> Task:
    """Load one ARC task file."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    train_pairs = raw.get("train")
    test_pairs = raw.get("test")
    if not isinstance(train_pairs, list) or not isinstance(test_pairs, list):
        raise ValueError(f"Invalid task format: {path}")

    return {"train": train_pairs, "test": test_pairs}


def load_split(data_root: Path, split: str) -> TaskMap:
    """Load all task files in a split directory."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    tasks: TaskMap = {}
    for task_path in sorted(split_dir.glob("*.json")):
        tasks[task_path.stem] = load_task_file(task_path)
    return tasks
