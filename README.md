# ARC-AGI-1 TTT Project (Living Document)

Last updated: 2026-03-08

## Goal

Build a reproducible Test-Time Training (TTT) system for ARC-AGI-1 that improves exact-match task solve rate over a non-TTT baseline while respecting ARC evaluation rules.

## Success Criteria

1. A full pipeline exists and is reproducible end-to-end:
   - load ARC tasks
   - run base model inference
   - run per-task TTT adaptation
   - score exact task solves
2. TTT demonstrates measurable gain over the same base model without TTT.
3. Runs are traceable (config, commit, metrics, notes) so results can be reproduced.

## Scope

- In scope:
  - ARC-AGI-1 `data/training` and `data/evaluation`
  - Neural baseline plus TTT adaptation loop
  - Ablations for adaptation steps, learning rate, and parameter-efficient tuning
- Out of scope (for now):
  - ARC-AGI-2
  - Hand-coded task-specific solvers
  - Ensembling multiple unrelated systems

## Constraints

- Hardware: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)
- Keep first implementation small enough for iterative experimentation on this GPU.
- Avoid data leakage from evaluation set into training decisions.

## Current Plan

1. Done: establish a no-TTT baseline scoring harness.
2. Done: add first neural no-TTT baseline training/evaluation pipeline.
3. Next: harden the neural baseline into a credible non-TTT reference.
4. In progress: replace weak pooled demo-context conditioning with richer task-conditioned interaction between demonstrations and query.
5. Then: add TTT loop with parameter-efficient updates (for example LoRA/adapters/BitFit).
6. Run controlled ablations and track deltas vs baseline.
7. Harden reproducibility and prepare a clean report of best configuration.

## Current Status

- Baseline evaluation harness implemented.
- Terminal task visualization available via `arckit` in `scripts/show_task.py`.
- Model strategy: train a small ARC-native model from scratch rather than download a large pretrained foundation model.
- Baselines available:
  - `copy_input`
  - `zeros`
  - `mode_color`
- Initial exact-match results (full 400/400 per split):
  - `copy_input`: training 0/400, evaluation 0/400
  - `zeros`: training 0/400, evaluation 0/400
  - `mode_color`: training 0/400, evaluation 0/400
- Neural baseline pipeline implemented:
  - pair dataset/tokenization (`src/arc_agi_1/dataset.py`)
  - 2D-aware transformer baseline (`src/arc_agi_1/model.py`)
  - train/eval utilities (`src/arc_agi_1/training.py`)
  - train and checkpoint eval CLIs
- Phase 1 data pipeline started:
  - task-conditioned samples can now bundle train demonstrations with a query pair
  - task-conditioned dataset tensors are available in `src/arc_agi_1/dataset.py`
- First task-conditioned model path implemented:
  - `ArcTaskConditionedModel` added in `src/arc_agi_1/model.py`
  - task-conditioned training/evaluation helpers added in `src/arc_agi_1/training.py`
  - task-conditioned trainer added in `scripts/train_task_conditioned_baseline.py`
- Task-conditioned smoke run (1 epoch, smaller model):
  - training solved tasks: 0/400
  - evaluation solved tasks: 0/400
  - training pair accuracy: 1/416
  - artifacts in `results/task_conditioned_baseline/`
- Stronger task-conditioned run (4 epochs, `d_model=96`, 2 layers):
  - training solved tasks: 0/400
  - evaluation solved tasks: 0/400
  - training pair accuracy: 1/416
  - evaluation pair accuracy: 0/419
  - loss fell from `2.79` to `1.56`, but exact-match metrics did not move
  - implication: the current pooled demo-context formulation is not enough; next work should improve demo-query interaction and decoding, not just train longer
- Neural smoke run (1 epoch, smaller model):
  - training solved tasks: 1/400 (0.25%)
  - evaluation solved tasks: 0/400 (0.00%)
  - artifacts in `results/neural_baseline/`
- Phase-based improvement roadmap documented in `docs/model-improvement-phases.md`
- Local experiment registry implemented:
  - append-only registry at `results/registry.jsonl`
  - one immutable run directory per experiment under `results/<family>/<run_id>/`
  - canonical launcher in `scripts/run_experiment.py`
  - inspection CLI in `scripts/list_experiments.py`

## Experiment Tracking

Every experiment run must be recorded locally, even if it fails or gets interrupted.

Canonical process:
1. Launch runs through `scripts/run_experiment.py`.
2. Let the launcher create a unique run directory under `results/<family>/<run_id>/`.
3. Keep all run-local artifacts in that run directory only.
4. Use `scripts/list_experiments.py` to inspect recent history before starting new work.
5. Update `docs/experiments.md` only for runs that materially affect project direction.

Registry contract:
- `results/registry.jsonl` is append-only.
- Do not overwrite prior run directories.
- Record failed runs as failed runs; do not delete them.
- The launcher snapshots command metadata, git state, stdout, and resolved OmegaConf config when available.

## Runbook

Canonical experiment launcher:

```bash
.venv/bin/python scripts/run_experiment.py \
  --family task_conditioned_baseline \
  --label poolctx_e04_d96_l2 \
  -- \
  .venv/bin/python scripts/train_task_conditioned_baseline.py \
  --config conf/train_task_conditioned.yaml \
  optim.epochs=4 \
  model.d_model=96 \
  model.n_heads=4 \
  model.n_layers=2 \
  optim.batch_size=16 \
  runtime.num_workers=0
```

List recent experiment records:

```bash
.venv/bin/python scripts/list_experiments.py --limit 10
```

Heuristic baseline evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_baseline.py \
  --split all \
  --baseline copy_input \
  --output results/baseline_copy_input.json
```

Neural baseline training (smoke run example):

```bash
PYTHONPATH=src .venv/bin/python scripts/train_neural_baseline.py \
  --config conf/train_baseline.yaml \
  optim.epochs=1 \
  model.d_model=96 \
  model.n_heads=4 \
  model.n_layers=2 \
  optim.batch_size=32 \
  runtime.num_workers=0 \
  output.checkpoint_path=results/neural_baseline/smoke_model.pt \
  output.metrics_path=results/neural_baseline/smoke_train_metrics.json \
  output.eval_path=results/neural_baseline/smoke_eval_metrics.json
```

Neural checkpoint evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_neural_baseline.py \
  --checkpoint results/neural_baseline/smoke_model.pt \
  --split all \
  --output results/neural_baseline/smoke_eval_from_ckpt.json
```

Task-conditioned baseline training (smoke run example):

```bash
PYTHONPATH=src .venv/bin/python scripts/train_task_conditioned_baseline.py \
  --config conf/train_task_conditioned.yaml \
  optim.epochs=1 \
  model.d_model=64 \
  model.n_heads=4 \
  model.n_layers=1 \
  optim.batch_size=16 \
  runtime.num_workers=0 \
  output.checkpoint_path=results/task_conditioned_baseline/smoke_model.pt \
  output.metrics_path=results/task_conditioned_baseline/smoke_train_metrics.json \
  output.eval_path=results/task_conditioned_baseline/smoke_eval_metrics.json
```

Show a task in the terminal:

```bash
PYTHONPATH=src .venv/bin/python scripts/show_task.py 007bbfb7
```

Show digits instead of colored cells:

```bash
PYTHONPATH=src .venv/bin/python scripts/show_task.py 007bbfb7 --digits
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Repository Layout

- `data/`: ARC-AGI-1 task files (`training`, `evaluation`)
- `conf/`: baseline training config (`train_baseline.yaml`)
- `conf/train_task_conditioned.yaml`: task-conditioned baseline config
- `src/arc_agi_1/`: loaders, baselines, dataset, model, training utilities
- `src/arc_agi_1/experiments.py`: local experiment registry and launcher helpers
- `scripts/eval_baseline.py`: heuristic baseline evaluator
- `scripts/train_neural_baseline.py`: neural baseline trainer
- `scripts/eval_neural_baseline.py`: checkpoint evaluator
- `scripts/train_task_conditioned_baseline.py`: task-conditioned baseline trainer
- `scripts/run_experiment.py`: canonical experiment launcher
- `scripts/list_experiments.py`: registry inspection CLI
- `scripts/show_task.py`: terminal ARC task viewer using `arckit`
- `tests/`: scorer/baseline behavior tests
- `results/`: local experiment registry and run artifacts (git-ignored)
- `docs/`: experiment and decision logs
- `guides/experiment-workflow.md`: experiment process for future sessions

## Documentation Policy (Must Evolve With Project)

This repository uses living docs. Any meaningful project change must update docs in the same change set.

Update triggers:
- goal/scope changes
- model or training objective changes
- evaluation method changes
- dependency/framework changes
- new experimental findings that affect direction

Required updates:
1. Update this README when goals, scope, plan, or constraints change.
2. Register every run locally in `results/registry.jsonl` via `scripts/run_experiment.py`.
3. Append every meaningful run to `docs/experiments.md`.
4. Record important architectural/process decisions in `docs/decisions.md`.

Definition of "done" for any milestone:
- code change merged
- experiment logged
- relevant doc sections updated

## Project Docs

- [Experiment Log](docs/experiments.md)
- [Decision Log](docs/decisions.md)
- [Model Improvement Phases](docs/model-improvement-phases.md)
- [ARC Data Model Guide](guides/arc-data-model.md)
- [Experiment Workflow Guide](guides/experiment-workflow.md)
