"""Dataset and tensorization helpers for neural ARC baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from arc_agi_1.data import Grid, Pair, TaskMap, load_split


@dataclass(slots=True)
class PairSample:
    task_id: str
    split: str
    source: str
    pair_index: int
    input_grid: Grid
    output_grid: Grid


@dataclass(slots=True)
class TaskConditionedSample:
    task_id: str
    split: str
    query_source: str
    query_index: int
    demo_pairs: list[Pair]
    query_pair: Pair


class PairBatchRow(TypedDict):
    input_grid: Tensor
    input_mask: Tensor
    output_grid: Tensor
    output_mask: Tensor
    output_h_idx: int
    output_w_idx: int
    task_id: str
    source: str
    pair_index: int


class TaskBatchRow(TypedDict):
    demo_input_grids: Tensor
    demo_input_masks: Tensor
    demo_output_grids: Tensor
    demo_output_masks: Tensor
    demo_mask: Tensor
    demo_count: int
    query_input_grid: Tensor
    query_input_mask: Tensor
    output_grid: Tensor
    output_mask: Tensor
    output_h_idx: int
    output_w_idx: int
    task_id: str
    query_source: str
    query_index: int


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (height, width) for a rectangular grid."""
    height = len(grid)
    if height == 0:
        return 0, 0

    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Grid is not rectangular.")
    return height, width


def encode_grid(grid: Grid, *, max_grid: int, pad_color: int) -> tuple[Tensor, Tensor]:
    """
    Encode a variable-size ARC grid to fixed-size tensors.

    Returns:
    - encoded grid tensor [max_grid, max_grid]
    - valid-cell mask tensor [max_grid, max_grid]
    """
    height, width = grid_shape(grid)
    if height > max_grid or width > max_grid:
        raise ValueError(f"Grid shape {(height, width)} exceeds max_grid={max_grid}.")

    encoded = torch.full((max_grid, max_grid), pad_color, dtype=torch.long)
    mask = torch.zeros((max_grid, max_grid), dtype=torch.bool)

    if height > 0 and width > 0:
        encoded[:height, :width] = torch.tensor(grid, dtype=torch.long)
        mask[:height, :width] = True

    return encoded, mask


def decode_grid(cells: Tensor, *, height: int, width: int) -> Grid:
    """Decode a fixed-size tensor back into a variable-size grid."""
    return cells[:height, :width].tolist()


def build_pair_samples(tasks: TaskMap, *, split: str, pair_sets: Sequence[str]) -> list[PairSample]:
    """Expand task map into supervised (input, output) samples."""
    if not pair_sets:
        raise ValueError("pair_sets must not be empty.")

    valid_sets = {"train", "test"}
    unknown = set(pair_sets) - valid_sets
    if unknown:
        raise ValueError(f"Invalid pair sets: {sorted(unknown)}")

    samples: list[PairSample] = []
    for task_id, task in tasks.items():
        for source in pair_sets:
            for pair_index, pair in enumerate(task[source]):
                samples.append(
                    PairSample(
                        task_id=task_id,
                        split=split,
                        source=source,
                        pair_index=pair_index,
                        input_grid=pair["input"],
                        output_grid=pair["output"],
                    )
                )
    return samples


def load_pair_samples(data_root: Path, *, split: str, pair_sets: Sequence[str]) -> list[PairSample]:
    """Load task files for a split and convert them to pair samples."""
    tasks = load_split(data_root, split)
    return build_pair_samples(tasks, split=split, pair_sets=pair_sets)


def build_task_conditioned_samples(
    tasks: TaskMap,
    *,
    split: str,
    query_sets: Sequence[str],
) -> list[TaskConditionedSample]:
    """Build samples where a query pair is conditioned on task train demonstrations."""
    if not query_sets:
        raise ValueError("query_sets must not be empty.")

    valid_sets = {"train", "test"}
    unknown = set(query_sets) - valid_sets
    if unknown:
        raise ValueError(f"Invalid query sets: {sorted(unknown)}")

    samples: list[TaskConditionedSample] = []
    for task_id, task in tasks.items():
        train_pairs = task["train"]
        for source in query_sets:
            for query_index, query_pair in enumerate(task[source]):
                if source == "train":
                    demo_pairs = [pair for idx, pair in enumerate(train_pairs) if idx != query_index]
                else:
                    demo_pairs = list(train_pairs)

                if not demo_pairs:
                    raise ValueError(f"Task {task_id} has no demonstrations for {source}[{query_index}].")

                samples.append(
                    TaskConditionedSample(
                        task_id=task_id,
                        split=split,
                        query_source=source,
                        query_index=query_index,
                        demo_pairs=demo_pairs,
                        query_pair=query_pair,
                    )
                )
    return samples


def load_task_conditioned_samples(
    data_root: Path,
    *,
    split: str,
    query_sets: Sequence[str],
) -> list[TaskConditionedSample]:
    """Load task-conditioned samples from one split."""
    tasks = load_split(data_root, split)
    return build_task_conditioned_samples(tasks, split=split, query_sets=query_sets)


class ArcPairDataset(Dataset[PairBatchRow]):
    """Torch dataset producing fixed-size tensors for ARC pair supervision."""

    def __init__(self, samples: Sequence[PairSample], *, max_grid: int = 30, pad_color: int = 10) -> None:
        self.samples = list(samples)
        self.max_grid = max_grid
        self.pad_color = pad_color

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PairBatchRow:
        sample = self.samples[index]

        input_grid, input_mask = encode_grid(
            sample.input_grid,
            max_grid=self.max_grid,
            pad_color=self.pad_color,
        )
        output_grid, output_mask = encode_grid(
            sample.output_grid,
            max_grid=self.max_grid,
            pad_color=0,
        )
        out_height, out_width = grid_shape(sample.output_grid)

        return {
            "input_grid": input_grid,
            "input_mask": input_mask,
            "output_grid": output_grid,
            "output_mask": output_mask,
            "output_h_idx": out_height - 1,
            "output_w_idx": out_width - 1,
            "task_id": sample.task_id,
            "source": sample.source,
            "pair_index": sample.pair_index,
        }


class ArcTaskDataset(Dataset[TaskBatchRow]):
    """Torch dataset for task-conditioned ARC supervision."""

    def __init__(
        self,
        samples: Sequence[TaskConditionedSample],
        *,
        max_grid: int = 30,
        max_demos: int = 4,
        pad_color: int = 10,
    ) -> None:
        self.samples = list(samples)
        self.max_grid = max_grid
        self.max_demos = max_demos
        self.pad_color = pad_color

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TaskBatchRow:
        sample = self.samples[index]
        if len(sample.demo_pairs) > self.max_demos:
            raise ValueError(
                f"Task {sample.task_id} needs {len(sample.demo_pairs)} demos, exceeds max_demos={self.max_demos}."
            )

        demo_input_grids = torch.full(
            (self.max_demos, self.max_grid, self.max_grid),
            self.pad_color,
            dtype=torch.long,
        )
        demo_input_masks = torch.zeros((self.max_demos, self.max_grid, self.max_grid), dtype=torch.bool)
        demo_output_grids = torch.zeros((self.max_demos, self.max_grid, self.max_grid), dtype=torch.long)
        demo_output_masks = torch.zeros((self.max_demos, self.max_grid, self.max_grid), dtype=torch.bool)
        demo_mask = torch.zeros((self.max_demos,), dtype=torch.bool)

        for demo_index, pair in enumerate(sample.demo_pairs):
            encoded_input, input_mask = encode_grid(
                pair["input"],
                max_grid=self.max_grid,
                pad_color=self.pad_color,
            )
            encoded_output, output_mask = encode_grid(
                pair["output"],
                max_grid=self.max_grid,
                pad_color=0,
            )
            demo_input_grids[demo_index] = encoded_input
            demo_input_masks[demo_index] = input_mask
            demo_output_grids[demo_index] = encoded_output
            demo_output_masks[demo_index] = output_mask
            demo_mask[demo_index] = True

        query_input_grid, query_input_mask = encode_grid(
            sample.query_pair["input"],
            max_grid=self.max_grid,
            pad_color=self.pad_color,
        )
        output_grid, output_mask = encode_grid(
            sample.query_pair["output"],
            max_grid=self.max_grid,
            pad_color=0,
        )
        out_height, out_width = grid_shape(sample.query_pair["output"])

        return {
            "demo_input_grids": demo_input_grids,
            "demo_input_masks": demo_input_masks,
            "demo_output_grids": demo_output_grids,
            "demo_output_masks": demo_output_masks,
            "demo_mask": demo_mask,
            "demo_count": len(sample.demo_pairs),
            "query_input_grid": query_input_grid,
            "query_input_mask": query_input_mask,
            "output_grid": output_grid,
            "output_mask": output_mask,
            "output_h_idx": out_height - 1,
            "output_w_idx": out_width - 1,
            "task_id": sample.task_id,
            "query_source": sample.query_source,
            "query_index": sample.query_index,
        }


def collate_pair_batch(batch: list[PairBatchRow]) -> dict[str, object]:
    """Collate function for ArcPairDataset."""
    return {
        "input_grid": torch.stack([row["input_grid"] for row in batch]),
        "input_mask": torch.stack([row["input_mask"] for row in batch]),
        "output_grid": torch.stack([row["output_grid"] for row in batch]),
        "output_mask": torch.stack([row["output_mask"] for row in batch]),
        "output_h_idx": torch.tensor([row["output_h_idx"] for row in batch], dtype=torch.long),
        "output_w_idx": torch.tensor([row["output_w_idx"] for row in batch], dtype=torch.long),
        "task_ids": [row["task_id"] for row in batch],
        "sources": [row["source"] for row in batch],
        "pair_indices": [row["pair_index"] for row in batch],
    }


def collate_task_batch(batch: list[TaskBatchRow]) -> dict[str, object]:
    """Collate function for ArcTaskDataset."""
    return {
        "demo_input_grids": torch.stack([row["demo_input_grids"] for row in batch]),
        "demo_input_masks": torch.stack([row["demo_input_masks"] for row in batch]),
        "demo_output_grids": torch.stack([row["demo_output_grids"] for row in batch]),
        "demo_output_masks": torch.stack([row["demo_output_masks"] for row in batch]),
        "demo_mask": torch.stack([row["demo_mask"] for row in batch]),
        "demo_count": torch.tensor([row["demo_count"] for row in batch], dtype=torch.long),
        "query_input_grid": torch.stack([row["query_input_grid"] for row in batch]),
        "query_input_mask": torch.stack([row["query_input_mask"] for row in batch]),
        "output_grid": torch.stack([row["output_grid"] for row in batch]),
        "output_mask": torch.stack([row["output_mask"] for row in batch]),
        "output_h_idx": torch.tensor([row["output_h_idx"] for row in batch], dtype=torch.long),
        "output_w_idx": torch.tensor([row["output_w_idx"] for row in batch], dtype=torch.long),
        "task_ids": [row["task_id"] for row in batch],
        "query_sources": [row["query_source"] for row in batch],
        "query_indices": [row["query_index"] for row in batch],
    }
