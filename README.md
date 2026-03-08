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
4. Next: move to a task-conditioned model that can use train demonstrations.
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
- Neural smoke run (1 epoch, smaller model):
  - training solved tasks: 1/400 (0.25%)
  - evaluation solved tasks: 0/400 (0.00%)
  - artifacts in `results/neural_baseline/`
- Phase-based improvement roadmap documented in `docs/model-improvement-phases.md`

## Runbook

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
- `src/arc_agi_1/`: loaders, baselines, dataset, model, training utilities
- `scripts/eval_baseline.py`: heuristic baseline evaluator
- `scripts/train_neural_baseline.py`: neural baseline trainer
- `scripts/eval_neural_baseline.py`: checkpoint evaluator
- `scripts/show_task.py`: terminal ARC task viewer using `arckit`
- `tests/`: scorer/baseline behavior tests
- `results/`: JSON run artifacts
- `docs/`: experiment and decision logs

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
2. Append every meaningful run to `docs/experiments.md`.
3. Record important architectural/process decisions in `docs/decisions.md`.

Definition of "done" for any milestone:
- code change merged
- experiment logged
- relevant doc sections updated

## Project Docs

- [Experiment Log](docs/experiments.md)
- [Decision Log](docs/decisions.md)
- [Model Improvement Phases](docs/model-improvement-phases.md)
