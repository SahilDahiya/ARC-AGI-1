from __future__ import annotations

import unittest

import torch

from arc_agi_1.model import ArcTaskConditionedModel
from arc_agi_1.training import evaluate_task_conditioned_solve_rate


class _CopyQueryModel(torch.nn.Module):
    def __init__(self, *, max_grid: int, num_colors: int) -> None:
        super().__init__()
        self.max_grid = max_grid
        self.num_colors = num_colors

    def eval(self) -> "_CopyQueryModel":
        return self

    def forward(
        self,
        demo_input_grids: torch.Tensor,
        demo_input_masks: torch.Tensor,
        demo_output_grids: torch.Tensor,
        demo_output_masks: torch.Tensor,
        demo_mask: torch.Tensor,
        query_input_grid: torch.Tensor,
        query_input_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del demo_input_grids
        del demo_input_masks
        del demo_output_grids
        del demo_output_masks
        del demo_mask

        batch_size, max_grid, _ = query_input_grid.shape
        height_indices = query_input_mask.any(dim=2).sum(dim=1) - 1
        width_indices = query_input_mask.any(dim=1).sum(dim=1) - 1

        height_logits = torch.full((batch_size, max_grid), -1e9, dtype=torch.float32)
        width_logits = torch.full((batch_size, max_grid), -1e9, dtype=torch.float32)
        height_logits.scatter_(1, height_indices.unsqueeze(1), 1e9)
        width_logits.scatter_(1, width_indices.unsqueeze(1), 1e9)

        cell_logits = torch.full(
            (batch_size, max_grid, max_grid, self.num_colors),
            -1e9,
            dtype=torch.float32,
        )
        safe_query_grid = query_input_grid.masked_fill(~query_input_mask, 0)
        cell_logits.scatter_(3, safe_query_grid.unsqueeze(-1), 1e9)

        return {
            "height_logits": height_logits,
            "width_logits": width_logits,
            "cell_logits": cell_logits,
        }


class ModelTests(unittest.TestCase):
    def test_task_conditioned_model_forward_shapes(self) -> None:
        model = ArcTaskConditionedModel(
            max_grid=4,
            max_demos=2,
            num_colors=10,
            pad_color=10,
            d_model=32,
            n_heads=4,
            n_layers=2,
        )

        out = model(
            demo_input_grids=torch.zeros((3, 2, 4, 4), dtype=torch.long),
            demo_input_masks=torch.zeros((3, 2, 4, 4), dtype=torch.bool),
            demo_output_grids=torch.zeros((3, 2, 4, 4), dtype=torch.long),
            demo_output_masks=torch.zeros((3, 2, 4, 4), dtype=torch.bool),
            demo_mask=torch.tensor(
                [
                    [True, False],
                    [True, True],
                    [False, False],
                ],
                dtype=torch.bool,
            ),
            query_input_grid=torch.zeros((3, 4, 4), dtype=torch.long),
            query_input_mask=torch.ones((3, 4, 4), dtype=torch.bool),
        )

        self.assertEqual(out["height_logits"].shape, (3, 4))
        self.assertEqual(out["width_logits"].shape, (3, 4))
        self.assertEqual(out["cell_logits"].shape, (3, 4, 4, 10))

    def test_evaluate_task_conditioned_solve_rate_counts_tasks(self) -> None:
        tasks = {
            "copy_ok": {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[2, 2]], "output": [[2, 2]]}],
            },
            "copy_bad": {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[3]], "output": [[0]]}],
            },
        }

        model = _CopyQueryModel(max_grid=4, num_colors=10)
        metrics = evaluate_task_conditioned_solve_rate(
            model,
            tasks,
            split="training",
            device=torch.device("cpu"),
            max_grid=4,
            max_demos=3,
            pad_color=10,
        )

        self.assertEqual(metrics.total_tasks, 2)
        self.assertEqual(metrics.solved_tasks, 1)
        self.assertAlmostEqual(metrics.solve_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
