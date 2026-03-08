"""Scoring helpers for ARC exact-match evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from arc_agi_1.baselines import Predictor
from arc_agi_1.data import Grid, Task, TaskMap


@dataclass(slots=True)
class SplitResult:
    split: str
    total_tasks: int
    solved_tasks: int
    solve_rate: float


def exact_grid_match(predicted: Grid, target: Grid) -> bool:
    """Return True only for strict grid equality."""
    return predicted == target


def task_is_solved(task: Task, predictor: Predictor) -> bool:
    """
    A task is solved only if all test pairs are predicted exactly.
    """
    train_pairs = task["train"]
    for pair in task["test"]:
        test_input = pair["input"]
        expected_output = pair["output"]
        predicted_output = predictor(train_pairs, test_input)
        if not exact_grid_match(predicted_output, expected_output):
            return False
    return True


def evaluate_split(
    tasks: Mapping[str, Task], predictor: Predictor, split: str
) -> SplitResult:
    """Evaluate solved-task accuracy for a split."""
    total = len(tasks)
    solved = sum(1 for task in tasks.values() if task_is_solved(task, predictor))
    solve_rate = (solved / total) if total else 0.0
    return SplitResult(
        split=split, total_tasks=total, solved_tasks=solved, solve_rate=solve_rate
    )


def limit_tasks(tasks: TaskMap, limit: int | None) -> TaskMap:
    """Return first N tasks for quick smoke runs."""
    if limit is None:
        return tasks

    task_items = list(tasks.items())[:limit]
    return dict(task_items)
