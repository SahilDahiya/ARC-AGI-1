## Engineering Direction

- No fallback behavior, ever.
- Fail hard on failure conditions.
- No backward compatibility guarantees.
- Forward-only development: do not carry legacy baggage.
- Default to a strict TDD workflow for durable feature development; exploratory research spikes may be exempt until promoted into maintained code.
- Use full red-to-green cycles: write a failing test first, implement the minimal change to pass, then refactor safely.

## Hard-Cut Product Policy

- Optimize for one canonical current-state implementation, not compatibility with historical local states.
- Do not preserve or introduce compatibility bridges, migration shims, fallback paths, compact adapters, or dual behavior for old local states unless the user explicitly asks for that support.
- Prefer one canonical current-state codepath.
- Prefer fail-fast diagnostics.
- Prefer explicit recovery steps.
- Do not add automatic migration.
- Do not add compatibility glue.
- Do not add silent fallbacks.
- Do not add "temporary" second paths without an explicit removal plan.
- If temporary migration or compatibility code is introduced for debugging or a narrowly scoped transition, it must be called out in the same diff with:
- why it exists
- why the canonical path is insufficient
- exact deletion criteria
- the ADR/task that tracks its removal
- Default stance across the app: delete old-state compatibility code rather than carrying it forward.

## Experiment Tracking

- Use `scripts/run_experiment.py` as the default entrypoint for local experiments.
- Every run must produce one immutable run directory under `results/<family>/<run_id>/`.
- Every run must append one row to `results/registry.jsonl`, including failed runs.
- Do not overwrite prior run artifacts or reuse run ids.
- Use `scripts/list_experiments.py` to inspect recent run history before starting new experiments.
- Update `docs/experiments.md` only for runs that materially change project direction.
