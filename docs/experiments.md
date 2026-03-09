# Experiment Log

Use this log to make results reproducible and comparable.

## Fields

- Date (YYYY-MM-DD)
- Run ID
- Commit
- Config summary
- Dataset split
- Metric(s)
- Baseline delta
- Notes / next action

## Entries

| Date | Run ID | Commit | Config summary | Dataset split | Metric(s) | Baseline delta | Notes / next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-08 | init-docs | n/a | Project docs initialized | n/a | n/a | n/a | Start implementation of baseline scoring harness |
| 2026-03-08 | baseline-copy-input-v1 | 6379a8595507422b6cf28df1ae2a8bab677f1c1c | `scripts/eval_baseline.py --baseline copy_input --split all` | training + evaluation | training: 0/400, evaluation: 0/400 | n/a | Saved to `results/baseline_copy_input.json`; proceed to TTT-capable model baseline |
| 2026-03-08 | baseline-zeros-v1 | 6379a8595507422b6cf28df1ae2a8bab677f1c1c | `scripts/eval_baseline.py --baseline zeros --split all` | training + evaluation | training: 0/400, evaluation: 0/400 | 0.0 vs copy_input | Confirms trivial heuristic floor |
| 2026-03-08 | baseline-mode-color-v1 | 6379a8595507422b6cf28df1ae2a8bab677f1c1c | `scripts/eval_baseline.py --baseline mode_color --split all` | training + evaluation | training: 0/400, evaluation: 0/400 | 0.0 vs copy_input | Confirms non-structured color prior is insufficient |
| 2026-03-08 | neural-baseline-smoke-v1 | 6379a8595507422b6cf28df1ae2a8bab677f1c1c | `scripts/train_neural_baseline.py --config conf/train_baseline.yaml optim.epochs=1 model.d_model=96 model.n_heads=4 model.n_layers=2 optim.batch_size=32` | training + evaluation | train solved: 1/400, eval solved: 0/400; train pair-acc: 1/416, eval pair-acc: 0/419 | +0.25pp training vs heuristics, 0.0pp evaluation vs heuristics | Artifacts: `results/neural_baseline/smoke_*`; next run should increase epochs and model capacity before TTT |
| 2026-03-08 | neural-baseline-smoke-v2 | 6379a8595507422b6cf28df1ae2a8bab677f1c1c | `scripts/train_neural_baseline.py --config conf/train_baseline.yaml optim.epochs=1 model.d_model=64 model.n_heads=4 model.n_layers=1 optim.batch_size=64` | training + evaluation | train solved: 0/400, eval solved: 0/400; train pair-acc: 0/416, eval pair-acc: 0/419 | 0.0pp vs heuristics | Confirms 1-epoch low-capacity setting is too weak; use deeper model and multi-epoch schedule for baseline before TTT |
| 2026-03-08 | task-conditioned-smoke-v1 | 66fe46a1d9eafbb83be313358923917b9a082b0d | `scripts/train_task_conditioned_baseline.py --config conf/train_task_conditioned.yaml optim.epochs=1 model.d_model=64 model.n_heads=4 model.n_layers=1 optim.batch_size=16` | training + evaluation | train solved: 0/400, eval solved: 0/400; train pair-acc: 1/416, eval pair-acc: 0/419 | 0.0pp solved-task delta vs heuristics; equal pair-acc to neural-baseline-smoke-v1 on training | End-to-end task-conditioned path now exists; next run should increase capacity and epochs before judging architecture |
