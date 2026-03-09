"""Experiment tracking utilities for local, append-only ARC runs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class ExperimentPaths:
    results_root: Path
    run_dir: Path
    registry_path: Path
    metadata_path: Path
    stdout_path: Path
    resolved_config_path: Path


@dataclass(slots=True)
class PreparedCommand:
    command: list[str]
    config_path: Path | None
    overrides: list[str]


def _sanitize_token(raw: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return token or "run"


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def build_run_id(*, family: str, label: str, started_at: datetime | None = None) -> str:
    timestamp = (started_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{_sanitize_token(family)}_{_sanitize_token(label)}"


def create_experiment_paths(results_root: Path, *, family: str, run_id: str) -> ExperimentPaths:
    family_dir = results_root / _sanitize_token(family)
    run_dir = family_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        results_root=results_root,
        run_dir=run_dir,
        registry_path=results_root / "registry.jsonl",
        metadata_path=run_dir / "metadata.json",
        stdout_path=run_dir / "stdout.log",
        resolved_config_path=run_dir / "config.resolved.yaml",
    )


def _script_basename(command: Sequence[str]) -> str | None:
    for arg in command:
        if arg.endswith(".py"):
            return Path(arg).name
    return None


def _parse_omegaconf_train_command(command: Sequence[str]) -> tuple[Path, list[str]]:
    args = list(command)
    script_index = -1
    for index, arg in enumerate(args):
        if arg.endswith(".py"):
            script_index = index
            break
    if script_index < 0:
        raise ValueError("Expected a Python script path in command.")

    script_name = Path(args[script_index]).name
    if script_name == "train_neural_baseline.py":
        config_path = Path("conf/train_baseline.yaml")
    elif script_name == "train_task_conditioned_baseline.py":
        config_path = Path("conf/train_task_conditioned.yaml")
    else:
        raise ValueError(f"Unsupported OmegaConf training script: {script_name}")

    overrides: list[str] = []
    index = script_index + 1
    while index < len(args):
        token = args[index]
        if token == "--config":
            if index + 1 >= len(args):
                raise ValueError("--config requires a value.")
            config_path = Path(args[index + 1])
            index += 2
            continue
        overrides.append(token)
        index += 1
    return config_path, overrides


def prepare_experiment_command(command: Sequence[str], *, run_dir: Path) -> PreparedCommand:
    if not command:
        raise ValueError("Command must not be empty.")

    prepared = list(command)
    script_name = _script_basename(prepared)
    if script_name in {"train_neural_baseline.py", "train_task_conditioned_baseline.py"}:
        config_path, overrides = _parse_omegaconf_train_command(prepared)
        if any(item.startswith("output.") for item in overrides):
            raise ValueError("Do not pass output.* overrides when using run_experiment.py.")

        output_overrides = [
            f"output.checkpoint_path={run_dir / 'checkpoint.pt'}",
            f"output.metrics_path={run_dir / 'metrics.json'}",
            f"output.eval_path={run_dir / 'final_eval.json'}",
        ]
        prepared.extend(output_overrides)
        return PreparedCommand(
            command=prepared,
            config_path=config_path,
            overrides=overrides + output_overrides,
        )

    if script_name in {"eval_baseline.py", "eval_neural_baseline.py"}:
        if "--output" in prepared:
            raise ValueError("Do not pass --output when using run_experiment.py.")
        prepared.extend(["--output", str(run_dir / "eval.json")])
        return PreparedCommand(command=prepared, config_path=None, overrides=[])

    return PreparedCommand(command=prepared, config_path=None, overrides=[])


def resolve_omegaconf_config(config_path: Path, *, overrides: Sequence[str]) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected DictConfig from OmegaConf.")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Expected resolved OmegaConf container to be a dict.")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_registry_entry(registry_path: Path, payload: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_registry(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise TypeError("Registry row must be a JSON object.")
        rows.append(parsed)
    return rows


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _git_is_dirty(project_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _summary_from_split_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        split_value = row.get("split")
        if not isinstance(split_value, str):
            continue
        prefix = "train" if split_value == "training" else "eval" if split_value == "evaluation" else None
        if prefix is None:
            continue
        for key in ("solve_rate", "solved_tasks", "total_tasks", "pair_accuracy", "exact_pair_matches", "total_pairs"):
            if key in row:
                summary[f"{prefix}_{key}"] = row[key]
    return summary


def summarize_run_artifacts(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            final = payload.get("final")
            if isinstance(final, dict):
                summary: dict[str, Any] = {}
                train = final.get("train")
                if isinstance(train, dict):
                    summary.update(_summary_from_split_rows([{"split": "training", **train}]))
                evaluation = final.get("evaluation")
                if isinstance(evaluation, dict):
                    summary.update(_summary_from_split_rows([{"split": "evaluation", **evaluation}]))
                if summary:
                    return summary

    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                rows = [row for row in payload["results"] if isinstance(row, dict)]
                summary = _summary_from_split_rows(rows)
                if summary:
                    return summary
            if isinstance(payload.get("splits"), list):
                rows = [row for row in payload["splits"] if isinstance(row, dict)]
                summary = _summary_from_split_rows(rows)
                if summary:
                    return summary
            final = payload.get("final")
            if isinstance(final, dict):
                summary = {}
                train = final.get("train")
                if isinstance(train, dict):
                    summary.update(_summary_from_split_rows([{"split": "training", **train}]))
                evaluation = final.get("evaluation")
                if isinstance(evaluation, dict):
                    summary.update(_summary_from_split_rows([{"split": "evaluation", **evaluation}]))
                if summary:
                    return summary

    return {}


def run_logged_command(
    *,
    command: Sequence[str],
    family: str,
    label: str,
    results_root: Path,
    project_root: Path,
    config_path: Path | None = None,
    config_overrides: Sequence[str] = (),
    prepare_known_scripts: bool = False,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    run_id = build_run_id(family=family, label=label, started_at=started_at)
    paths = create_experiment_paths(results_root, family=family, run_id=run_id)

    if prepare_known_scripts:
        prepared = prepare_experiment_command(command, run_dir=paths.run_dir)
        command = prepared.command
        config_path = prepared.config_path
        config_overrides = prepared.overrides

    resolved_config_written = False
    if config_path is not None:
        resolved_config = resolve_omegaconf_config(project_root / config_path, overrides=config_overrides)
        paths.resolved_config_path.write_text(OmegaConf.to_yaml(resolved_config), encoding="utf-8")
        resolved_config_written = True

    base_record: dict[str, Any] = {
        "run_id": run_id,
        "family": _sanitize_token(family),
        "label": label,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "return_code": None,
        "git_commit": _git_commit(project_root),
        "git_dirty": _git_is_dirty(project_root),
        "project_root": str(project_root),
        "run_dir": str(paths.run_dir),
        "registry_path": str(paths.registry_path),
        "stdout_path": str(paths.stdout_path),
        "metadata_path": str(paths.metadata_path),
        "resolved_config_path": str(paths.resolved_config_path) if resolved_config_written else None,
        "command": list(command),
        "summary": {},
    }
    write_json(paths.metadata_path, base_record)

    env = os.environ.copy()
    env["ARC_EXPERIMENT_DIR"] = str(paths.run_dir)
    env["ARC_RUN_ID"] = run_id
    env["ARC_RUN_FAMILY"] = _sanitize_token(family)
    env["PYTHONUNBUFFERED"] = "1"

    return_code = 1
    status = "failed"
    with paths.stdout_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            if process.stdout is None:
                raise RuntimeError("Subprocess stdout pipe is unavailable.")

            for line in process.stdout:
                sys.stdout.write(line)
                log_handle.write(line)
            process.stdout.close()
            return_code = process.wait()
            status = "completed" if return_code == 0 else "failed"
        except Exception as exc:
            message = f"Failed to launch experiment: {exc}\n"
            sys.stdout.write(message)
            log_handle.write(message)
            status = "failed"
            return_code = 1

    summary = summarize_run_artifacts(paths.run_dir)
    finished_at = iso_now()
    record = {
        **base_record,
        "status": status,
        "finished_at_utc": finished_at,
        "return_code": return_code,
        "summary": summary,
    }
    write_json(paths.metadata_path, record)
    append_registry_entry(paths.registry_path, record)
    return record


def format_registry_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> str:
    visible_rows = list(rows)
    if limit is not None:
        visible_rows = visible_rows[-limit:]

    headers = ["run_id", "status", "family", "train_solve", "eval_solve", "return_code"]
    table_rows: list[list[str]] = [headers]
    for row in visible_rows:
        summary = row.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        table_rows.append(
            [
                str(row.get("run_id", "")),
                str(row.get("status", "")),
                str(row.get("family", "")),
                str(summary.get("train_solve_rate", "")),
                str(summary.get("eval_solve_rate", "")),
                str(row.get("return_code", "")),
            ]
        )

    widths = [max(len(rendered[index]) for rendered in table_rows) for index in range(len(headers))]
    lines: list[str] = []
    for row_index, row in enumerate(table_rows):
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        lines.append(" | ".join(padded))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)
