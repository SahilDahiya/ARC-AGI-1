"""Simple baseline predictors for ARC-AGI tasks."""

from __future__ import annotations

from collections import Counter
from typing import Callable

from arc_agi_1.data import Grid, Pair

Predictor = Callable[[list[Pair], Grid], Grid]


def copy_input_predictor(_train_pairs: list[Pair], test_input: Grid) -> Grid:
    """Predict output as an exact copy of input."""
    return [row[:] for row in test_input]


def zeros_like_input_predictor(_train_pairs: list[Pair], test_input: Grid) -> Grid:
    """Predict an all-zero grid with same dimensions as input."""
    return [[0 for _ in row] for row in test_input]


def mode_color_fill_predictor(train_pairs: list[Pair], test_input: Grid) -> Grid:
    """Fill output with dominant color from training outputs."""
    color_counter: Counter[int] = Counter()
    for pair in train_pairs:
        output_grid = pair.get("output", [])
        for row in output_grid:
            color_counter.update(row)

    mode_color = color_counter.most_common(1)[0][0] if color_counter else 0
    return [[mode_color for _ in row] for row in test_input]


BASELINES: dict[str, Predictor] = {
    "copy_input": copy_input_predictor,
    "zeros": zeros_like_input_predictor,
    "mode_color": mode_color_fill_predictor,
}
