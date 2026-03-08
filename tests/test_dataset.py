from __future__ import annotations

import unittest

from arc_agi_1.dataset import (
    ArcTaskDataset,
    build_pair_samples,
    build_task_conditioned_samples,
    collate_task_batch,
    decode_grid,
    encode_grid,
)


class DatasetTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self) -> None:
        grid = [[1, 2], [3, 4]]
        encoded, mask = encode_grid(grid, max_grid=5, pad_color=10)

        self.assertEqual(encoded.shape, (5, 5))
        self.assertTrue(mask[0, 0].item())
        self.assertTrue(mask[1, 1].item())
        self.assertFalse(mask[2, 2].item())

        decoded = decode_grid(encoded, height=2, width=2)
        self.assertEqual(decoded, grid)

    def test_build_pair_samples_expands_pairs(self) -> None:
        tasks = {
            "t1": {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[3]]}],
            }
        }
        samples = build_pair_samples(tasks, split="training", pair_sets=["train", "test"])
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].task_id, "t1")
        self.assertEqual(samples[1].source, "test")

    def test_build_task_conditioned_samples_uses_other_train_pairs_as_context(self) -> None:
        tasks = {
            "t1": {
                "train": [
                    {"input": [[0]], "output": [[1]]},
                    {"input": [[2]], "output": [[3]]},
                ],
                "test": [{"input": [[4]], "output": [[5]]}],
            }
        }

        samples = build_task_conditioned_samples(tasks, split="training", query_sets=["train", "test"])

        self.assertEqual(len(samples), 3)

        first_train = samples[0]
        self.assertEqual(first_train.query_source, "train")
        self.assertEqual(first_train.query_pair["input"], [[0]])
        self.assertEqual(len(first_train.demo_pairs), 1)
        self.assertEqual(first_train.demo_pairs[0]["input"], [[2]])

        test_query = samples[2]
        self.assertEqual(test_query.query_source, "test")
        self.assertEqual(len(test_query.demo_pairs), 2)

    def test_arc_task_dataset_encodes_context_and_query(self) -> None:
        tasks = {
            "t1": {
                "train": [
                    {"input": [[0, 1]], "output": [[1, 0]]},
                    {"input": [[2, 2]], "output": [[2, 0]]},
                ],
                "test": [{"input": [[3, 3]], "output": [[3, 0]]}],
            }
        }

        samples = build_task_conditioned_samples(tasks, split="training", query_sets=["test"])
        dataset = ArcTaskDataset(samples, max_grid=4, max_demos=3, pad_color=10)
        row = dataset[0]

        self.assertEqual(row["demo_input_grids"].shape, (3, 4, 4))
        self.assertEqual(row["demo_output_grids"].shape, (3, 4, 4))
        self.assertEqual(row["demo_mask"].shape, (3,))
        self.assertEqual(row["query_input_grid"].shape, (4, 4))
        self.assertEqual(row["output_grid"].shape, (4, 4))
        self.assertEqual(row["demo_count"], 2)
        self.assertTrue(row["demo_mask"][0].item())
        self.assertTrue(row["demo_mask"][1].item())
        self.assertFalse(row["demo_mask"][2].item())

        batch = collate_task_batch([row])
        self.assertEqual(batch["demo_input_grids"].shape, (1, 3, 4, 4))
        self.assertEqual(batch["query_input_grid"].shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
