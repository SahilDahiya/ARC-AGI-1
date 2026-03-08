#!/usr/bin/env python3
"""Evaluate a trained neural baseline checkpoint on ARC splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from arc_agi_1.data import load_split
from arc_agi_1.model import ArcGridBaselineModel
from arc_agi_1.training import current_commit, evaluate_task_solve_rate, iso_now, resolve_device, save_json


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="results/neural_baseline/model.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", choices=["training", "evaluation", "all"], default="all")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=str,
        default="results/neural_baseline/eval_from_checkpoint.json",
    )
    return parser.parse_args()


def _config_or_default(checkpoint: dict[str, Any], key: str, default: Any) -> Any:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            return model_cfg.get(key, default)
    return default


def main() -> int:
    args = parse_args()
    project_root = _project_root()
    device = resolve_device(args.device)

    checkpoint_path = _abs_path(project_root, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Invalid checkpoint: missing model_state_dict.")

    max_grid = int(_config_or_default(checkpoint, "max_grid", 30))
    num_colors = int(_config_or_default(checkpoint, "num_colors", 10))
    pad_color = int(_config_or_default(checkpoint, "pad_color", 10))
    d_model = int(_config_or_default(checkpoint, "d_model", 192))
    n_heads = int(_config_or_default(checkpoint, "n_heads", 6))
    n_layers = int(_config_or_default(checkpoint, "n_layers", 6))
    dropout = float(_config_or_default(checkpoint, "dropout", 0.1))

    model = ArcGridBaselineModel(
        max_grid=max_grid,
        num_colors=num_colors,
        pad_color=pad_color,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_root = _abs_path(project_root, args.data_root)
    splits = ["training", "evaluation"] if args.split == "all" else [args.split]

    rows: list[dict[str, Any]] = []
    for split in splits:
        tasks = load_split(data_root, split)
        metrics = evaluate_task_solve_rate(
            model,
            tasks,
            split=split,
            device=device,
            max_grid=max_grid,
            pad_color=pad_color,
        )
        rows.append(asdict(metrics))
        print(
            f"{split}: solved {metrics.solved_tasks}/{metrics.total_tasks} "
            f"({100.0 * metrics.solve_rate:.2f}%), pair-acc {100.0 * metrics.pair_accuracy:.2f}%"
        )

    payload = {
        "timestamp_utc": iso_now(),
        "commit": current_commit(project_root),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "results": rows,
    }
    output_path = _abs_path(project_root, args.output)
    save_json(output_path, payload)
    print(f"Wrote evaluation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

