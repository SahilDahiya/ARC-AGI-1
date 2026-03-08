# Model Improvement Phases

Last updated: 2026-03-08

## Purpose

This document breaks the real model-improvement path into forward-only phases from the current neural baseline to a TTT-capable system. Each phase has a concrete objective, expected code changes, evaluation criteria, and exit condition.

## Current Starting Point

- Exact-match heuristic baselines exist and score `0/400` on both splits.
- A first neural baseline exists:
  - input-only supervision
  - predicts output size plus output canvas
  - current smoke runs are weak and not yet a meaningful reference for TTT
- Default model strategy is from-scratch training of a small ARC-native architecture.
- Task-conditioned data plumbing has started:
  - train demonstrations can be packaged with a query pair
  - future model work should build on this path rather than the input-only dataset

## Phase 0: Baseline Hardening

Objective:
- Turn the current neural baseline into a credible non-TTT reference.

Work:
- Increase training duration and model capacity within 11 GB VRAM.
- Add deterministic train/eval configs for short, medium, and full runs.
- Improve artifact structure for checkpoints, metrics, and config snapshots.
- Add stricter tests around dataset encoding, decode paths, and eval metrics.

Metrics:
- Stable train/eval runs without runtime warnings or import hacks.
- A reproducible no-TTT baseline better than current smoke results.

Exit condition:
- One reproducible no-TTT run is selected as the official reference baseline for later TTT comparisons.

## Phase 1: Task-Conditioned Modeling

Objective:
- Stop treating each pair independently and condition predictions on the in-task train demonstrations.

Work:
- Redesign the dataset and model input format so a test input is encoded together with the task's train examples.
- Add explicit encoding for:
  - train input grids
  - train output grids
  - current test input grid
  - segment/type markers
- Decide whether to use:
  - packed token sequence with 2D tags
  - multi-grid encoder with pooling/cross-attention

Metrics:
- Task-conditioned model outperforms the Phase 0 input-only baseline on held-out evaluation.

Exit condition:
- Official baseline is replaced by a task-conditioned no-TTT model.

## Phase 2: Stronger Output Parameterization

Objective:
- Improve output generation quality beyond a single fixed-canvas prediction head.

Work:
- Evaluate alternatives:
  - fixed-canvas classification head with better masking
  - autoregressive grid decoder
  - iterative refinement decoder
- Compare explicit size prediction vs learned stop tokens.
- Add exact-match and intermediate diagnostics for dimension errors vs cell-value errors.

Metrics:
- Reduction in dimension mistakes.
- Better pair accuracy and solved-task rate than Phase 1.

Exit condition:
- One output parameterization is chosen as the base architecture for TTT.

## Phase 3: Parameter-Efficient TTT

Objective:
- Add per-task adaptation without full model retraining.

Work:
- Introduce adapters / LoRA-style low-rank updates or similarly scoped task-time trainable parameters.
- Reset adaptation state between tasks.
- Restrict TTT updates to task-available information only.
- Add per-task adaptation loop:
  - load task
  - optimize on train pairs
  - infer test pairs
  - clear adapted state

Metrics:
- Exact solved-task delta vs the same model without TTT.
- Per-task adaptation runtime and memory cost.

Exit condition:
- TTT produces a measurable gain over the Phase 2 no-TTT reference.

## Phase 4: TTT Objective Design

Objective:
- Improve what the model learns during adaptation.

Work:
- Compare adaptation objectives:
  - direct supervised reconstruction on task train pairs
  - auxiliary consistency losses
  - masked-cell reconstruction
  - self-distillation or confidence-based objectives if justified
- Tune:
  - adaptation steps
  - learning rate
  - trainable parameter subset
  - regularization

Metrics:
- Better TTT delta than Phase 3 under similar runtime budget.

Exit condition:
- One TTT objective/config is selected as the default.

## Phase 5: Ablations And Failure Analysis

Objective:
- Understand what actually drives gains and where the system fails.

Work:
- Run targeted ablations on:
  - model size
  - context format
  - decoder choice
  - adapter rank / parameter subset
  - TTT step count
- Cluster failure types:
  - wrong dimensions
  - wrong colors
  - wrong object placement
  - partial rule capture

Metrics:
- Decision-quality evidence for architecture choices.

Exit condition:
- Roadmap decisions are backed by ablation data rather than intuition.

## Phase 6: Finalize Training And Evaluation Protocol

Objective:
- Lock a clean, reproducible protocol for serious comparison runs.

Work:
- Freeze default configs.
- Standardize artifact naming and result summaries.
- Add one-command train/eval pipelines for official runs.
- Document exact comparison methodology for no-TTT vs TTT.

Metrics:
- Repeated runs are explainable and comparable.

Exit condition:
- Project has a stable protocol for future iteration and reporting.

## Prioritization

Immediate next steps:
1. Finish Phase 0 and define the official no-TTT reference baseline.
2. Finish the rest of Phase 1 so the model can actually use train demonstrations.
3. Do not start TTT before a task-conditioned baseline exists.

## Non-Goals For Now

- No compatibility-preserving architecture layers.
- No hidden fallback paths.
- No premature ensembling.
- No ARC-AGI-2 work until ARC-AGI-1 pipeline is credible.
