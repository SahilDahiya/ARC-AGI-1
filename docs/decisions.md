# Decision Log

Record high-impact decisions and why they were made.

## Template

### [ID] Title
- Date:
- Status: proposed | accepted | superseded
- Context:
- Decision:
- Consequences:
- Supersedes:

## Decisions

### [D-001] Use living documentation as a project requirement
- Date: 2026-03-08
- Status: accepted
- Context: Project direction will change quickly while iterating on TTT methods.
- Decision: Keep `README.md`, `docs/experiments.md`, and `docs/decisions.md` as mandatory, continuously updated project docs.
- Consequences: Slight overhead on each change, but better reproducibility and fewer ambiguous decisions.
- Supersedes: none

### [D-002] Start with exact-match solved-task harness and simple heuristic baselines
- Date: 2026-03-08
- Status: accepted
- Context: TTT progress must be measured against a reproducible non-TTT reference under ARC solved-task criteria.
- Decision: Implement a minimal harness that scores strict exact-match on all test pairs per task and include simple baselines (`copy_input`, `zeros`, `mode_color`) for sanity checks.
- Consequences: Provides immediate reproducible floor and validates scoring path before model/TTT complexity.
- Supersedes: none

### [D-003] Use OmegaConf CLI overrides instead of Hydra runtime wrapper
- Date: 2026-03-08
- Status: accepted
- Context: `hydra-core` CLI wrapper raised an `argparse` compatibility error on Python 3.14 in this environment.
- Decision: Keep YAML-based hierarchical configs and dotlist overrides via `OmegaConf` directly in training scripts.
- Consequences: Retains reproducible config workflow while removing runtime fragility from current Hydra CLI behavior.
- Supersedes: none

### [D-004] First neural baseline predicts output size + full output canvas from input-only supervision
- Date: 2026-03-08
- Status: accepted
- Context: We need a trainable no-TTT baseline quickly to establish a meaningful floor before TTT integration.
- Decision: Use a 2D-aware transformer encoder that predicts output height/width and per-cell colors on a fixed 30x30 canvas, trained on ARC pair supervision.
- Consequences: Provides immediate train/eval path and checkpoint artifacts; does not yet condition on in-task train demonstrations, so expected performance is limited.
- Supersedes: none

### [D-005] Use `arckit` for terminal task visualization
- Date: 2026-03-08
- Status: accepted
- Context: We need a low-friction way to inspect ARC tasks in the terminal while debugging.
- Decision: Add `arckit` as a dependency and route `scripts/show_task.py` through `arckit.vis.print_grid`, keeping `--digits` as a plain-text fallback.
- Consequences: Better terminal visualization with less maintenance than a custom renderer.
- Supersedes: none

### [D-006] Use a phased roadmap for real model improvement
- Date: 2026-03-08
- Status: accepted
- Context: The next work is no longer a single implementation step; we need a disciplined path from weak baseline to task-conditioned TTT system.
- Decision: Track model work through explicit phases in `docs/model-improvement-phases.md`, with exit criteria before moving to the next phase.
- Consequences: Better sequencing, less architectural thrash, and clearer success criteria for each stage.
- Supersedes: none

### [D-007] Default to from-scratch ARC-native models
- Date: 2026-03-08
- Status: accepted
- Context: ARC-AGI grids and task-conditioning needs do not align well with dropping in a large downloaded pretrained model as the primary path, and test-time adaptation must remain cheap on 11 GB VRAM.
- Decision: Use small from-scratch ARC-specific architectures as the default path; downloaded pretrained models, if any, are comparison branches rather than the main plan.
- Consequences: More architecture work up front, but tighter control over data format, task conditioning, and TTT cost.
- Supersedes: none

### [D-008] First task-conditioned model pools demonstration context into query prediction
- Date: 2026-03-08
- Status: accepted
- Context: We need a first task-conditioned baseline quickly, without jumping straight to a more complex cross-attention or autoregressive architecture.
- Decision: Encode demo inputs, demo outputs, and the query input with a shared grid encoder; pool demo representations into a single context vector; add that context to query predictions for output size and output cells.
- Consequences: Provides a minimal end-to-end task-conditioned baseline now, while leaving room for a richer architecture later.
- Supersedes: none

### [D-009] Use a local append-only experiment registry as the canonical run record
- Date: 2026-03-08
- Status: accepted
- Context: Metrics JSON files and Markdown notes alone are too easy to lose track of once runs multiply, especially across multiple coding-agent sessions.
- Decision: Record every experiment through `scripts/run_experiment.py`, writing one immutable run directory under `results/<family>/<run_id>/` plus an append-only row in `results/registry.jsonl`. Reserve `docs/experiments.md` for the subset of runs that materially change project direction.
- Consequences: Slight workflow overhead, but stronger reproducibility, clearer failure accounting, and a single canonical place to inspect recent local runs.
- Supersedes: none
