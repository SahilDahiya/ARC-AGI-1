from __future__ import annotations

import unittest

from arc_agi_1.baselines import copy_input_predictor, mode_color_fill_predictor
from arc_agi_1.scoring import evaluate_split, exact_grid_match, task_is_solved


class ScoringTests(unittest.TestCase):
    def test_exact_grid_match(self) -> None:
        self.assertTrue(exact_grid_match([[1, 2]], [[1, 2]]))
        self.assertFalse(exact_grid_match([[1, 2]], [[2, 1]]))

    def test_task_is_solved_requires_all_test_pairs(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [
                {"input": [[2]], "output": [[2]]},
                {"input": [[3]], "output": [[9]]},
            ],
        }
        self.assertFalse(task_is_solved(task, copy_input_predictor))

    def test_evaluate_split_counts_solved_tasks(self) -> None:
        tasks = {
            "task_ok": {
                "train": [],
                "test": [{"input": [[5]], "output": [[5]]}],
            },
            "task_bad": {
                "train": [],
                "test": [{"input": [[0]], "output": [[1]]}],
            },
        }
        result = evaluate_split(tasks, copy_input_predictor, "training")
        self.assertEqual(result.total_tasks, 2)
        self.assertEqual(result.solved_tasks, 1)
        self.assertAlmostEqual(result.solve_rate, 0.5)

    def test_mode_color_uses_train_outputs(self) -> None:
        train_pairs = [
            {"input": [[0, 0]], "output": [[3, 3], [3, 1]]},
            {"input": [[1]], "output": [[3]]},
        ]
        pred = mode_color_fill_predictor(train_pairs, [[7, 7, 7]])
        self.assertEqual(pred, [[3, 3, 3]])


if __name__ == "__main__":
    unittest.main()
