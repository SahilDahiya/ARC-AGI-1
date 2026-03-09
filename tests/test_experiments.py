from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from arc_agi_1.experiments import (
    prepare_experiment_command,
    resolve_omegaconf_config,
    run_logged_command,
    summarize_run_artifacts,
)


class ExperimentTests(unittest.TestCase):
    def test_resolve_omegaconf_config_merges_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  d_model: 64\n"
                "optim:\n"
                "  epochs: 1\n",
                encoding="utf-8",
            )

            resolved = resolve_omegaconf_config(
                config_path,
                overrides=["optim.epochs=4", "model.d_model=96"],
            )

            self.assertEqual(resolved["optim"]["epochs"], 4)
            self.assertEqual(resolved["model"]["d_model"], 96)

    def test_prepare_experiment_command_injects_task_conditioned_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            prepared = prepare_experiment_command(
                [
                    sys.executable,
                    "scripts/train_task_conditioned_baseline.py",
                    "--config",
                    "conf/train_task_conditioned.yaml",
                    "optim.epochs=4",
                ],
                run_dir=run_dir,
            )

            self.assertEqual(prepared.config_path, Path("conf/train_task_conditioned.yaml"))
            self.assertIn("optim.epochs=4", prepared.overrides)
            self.assertIn(f"output.checkpoint_path={run_dir / 'checkpoint.pt'}", prepared.command)
            self.assertIn(f"output.metrics_path={run_dir / 'metrics.json'}", prepared.command)
            self.assertIn(f"output.eval_path={run_dir / 'final_eval.json'}", prepared.command)

    def test_summarize_run_artifacts_reads_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "final": {
                            "train": {
                                "solve_rate": 0.25,
                                "pair_accuracy": 0.5,
                                "solved_tasks": 1,
                                "total_tasks": 4,
                            },
                            "evaluation": {
                                "solve_rate": 0.1,
                                "pair_accuracy": 0.2,
                                "solved_tasks": 2,
                                "total_tasks": 20,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_run_artifacts(run_dir)

            self.assertEqual(summary["train_solve_rate"], 0.25)
            self.assertEqual(summary["eval_solve_rate"], 0.1)
            self.assertEqual(summary["train_pair_accuracy"], 0.5)
            self.assertEqual(summary["eval_pair_accuracy"], 0.2)

    def test_run_logged_command_records_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir) / "results"
            record = run_logged_command(
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import json, os, pathlib; "
                        "run_dir = pathlib.Path(os.environ['ARC_EXPERIMENT_DIR']); "
                        "(run_dir / 'eval.json').write_text(json.dumps({'splits':[{'split':'training','solve_rate':0.5,'solved_tasks':1,'total_tasks':2}]}), encoding='utf-8'); "
                        "print('ok')"
                    ),
                ],
                family="tests",
                label="success",
                results_root=results_root,
                project_root=Path.cwd(),
            )

            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["summary"]["train_solve_rate"], 0.5)
            self.assertTrue(Path(record["run_dir"]).exists())
            self.assertTrue(Path(record["stdout_path"]).exists())

            registry_path = results_root / "registry.jsonl"
            rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")

    def test_run_logged_command_records_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir) / "results"
            record = run_logged_command(
                command=[sys.executable, "-c", "raise SystemExit(3)"],
                family="tests",
                label="failure",
                results_root=results_root,
                project_root=Path.cwd(),
            )

            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["return_code"], 3)


if __name__ == "__main__":
    unittest.main()
