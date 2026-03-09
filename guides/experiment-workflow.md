# Experiment Workflow

This guide defines the canonical local experiment process for this repo.

## Goal

Never lose track of:
- what command ran
- what code version ran
- what config ran
- where artifacts were written
- whether the run completed, failed, or was interrupted

## Canonical Rules

1. Launch experiments through `scripts/run_experiment.py`.
2. Give every run a meaningful `--family` and `--label`.
3. Never overwrite old run directories.
4. Record failed runs as failed runs.
5. Use `docs/experiments.md` only for runs that materially affect project direction.

## Storage Layout

Every run gets a unique directory:

```text
results/
  registry.jsonl
  <family>/
    <run_id>/
      metadata.json
      stdout.log
      config.resolved.yaml
      checkpoint.pt
      metrics.json
      final_eval.json
      eval.json
```

Not every run will have every artifact file. The launcher records what happened even when the child command fails.

## Required Metadata

Each registry row must capture:
- `run_id`
- `family`
- `label`
- `status`
- `started_at_utc`
- `finished_at_utc`
- `return_code`
- `git_commit`
- `git_dirty`
- `command`
- `run_dir`
- `stdout_path`
- `resolved_config_path` when available
- `summary` when metrics can be extracted

## How To Run

Task-conditioned training example:

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

Baseline evaluation example:

```bash
.venv/bin/python scripts/run_experiment.py \
  --family heuristics \
  --label copy_input_all \
  -- \
  .venv/bin/python scripts/eval_baseline.py \
  --baseline copy_input \
  --split all
```

## How To Inspect

Show recent runs:

```bash
.venv/bin/python scripts/list_experiments.py --limit 10
```

Filter by family:

```bash
.venv/bin/python scripts/list_experiments.py --family task_conditioned_baseline
```

Inspect one run directory directly:

```bash
ls results/task_conditioned_baseline/<run_id>/
cat results/task_conditioned_baseline/<run_id>/metadata.json
```

## When To Update Markdown Logs

Update `docs/experiments.md` when a run:
- changes model direction
- establishes or replaces a baseline
- reveals a blocker
- invalidates a prior assumption

Do not spam `docs/experiments.md` with every smoke run. The local registry already covers complete run history.

## Agent Session Rule

Before starting a new experiment:
1. check `scripts/list_experiments.py`
2. read the most relevant prior run metadata
3. choose a new label that encodes the main changed variables

After finishing a meaningful run:
1. inspect `metadata.json` and artifact files
2. update `docs/experiments.md` if the result changed direction
3. update `docs/decisions.md` if the result changed project policy or architecture
