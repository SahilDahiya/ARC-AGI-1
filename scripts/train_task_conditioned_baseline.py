#!/usr/bin/env python3
"""Train a task-conditioned no-TTT baseline for ARC-AGI-1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_agi_1.data import load_split
from arc_agi_1.dataset import ArcTaskDataset, build_task_conditioned_samples, collate_task_batch
from arc_agi_1.model import ArcTaskConditionedModel
from arc_agi_1.training import (
    autocast_context,
    compute_loss,
    current_commit,
    evaluate_task_conditioned_solve_rate,
    iso_now,
    move_batch_to_device,
    resolve_device,
    save_json,
    set_seed,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="conf/train_task_conditioned.yaml",
        help="Path to OmegaConf YAML config file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, for example optim.epochs=1",
    )
    return parser.parse_args()


def load_config(project_root: Path, config_path: str, overrides: list[str]) -> DictConfig:
    config_file = _abs_path(project_root, config_path)
    cfg = OmegaConf.load(config_file)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected DictConfig from OmegaConf.")
    return cfg


def train(cfg: DictConfig) -> dict[str, Any]:
    project_root = _project_root()
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.runtime.device))
    use_amp = bool(cfg.runtime.amp) and device.type == "cuda"

    data_root = _abs_path(project_root, str(cfg.data.root))
    train_split = str(cfg.data.train_split)
    eval_split = str(cfg.data.eval_split)

    train_tasks = load_split(data_root, train_split)
    eval_tasks = load_split(data_root, eval_split)

    train_samples = build_task_conditioned_samples(
        train_tasks,
        split=train_split,
        query_sets=list(cfg.data.train_query_sets),
    )
    train_dataset = ArcTaskDataset(
        train_samples,
        max_grid=int(cfg.model.max_grid),
        max_demos=int(cfg.model.max_demos),
        pad_color=int(cfg.model.pad_color),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.optim.batch_size),
        shuffle=True,
        num_workers=int(cfg.runtime.num_workers),
        collate_fn=collate_task_batch,
        pin_memory=device.type == "cuda",
    )

    model = ArcTaskConditionedModel(
        max_grid=int(cfg.model.max_grid),
        max_demos=int(cfg.model.max_demos),
        num_colors=int(cfg.model.num_colors),
        pad_color=int(cfg.model.pad_color),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        dropout=float(cfg.model.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    epochs = int(cfg.optim.epochs)
    log_every = int(cfg.runtime.log_every)
    size_loss_weight = float(cfg.optim.size_loss_weight)
    eval_every = int(cfg.eval.every_n_epochs)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Training tasks: {len(train_tasks)} | Evaluation tasks: {len(eval_tasks)}")

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_cell_loss = 0.0
        running_height_loss = 0.0
        running_width_loss = 0.0
        steps = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device.type, use_amp):
                model_out = model(
                    demo_input_grids=batch["demo_input_grids"],
                    demo_input_masks=batch["demo_input_masks"],
                    demo_output_grids=batch["demo_output_grids"],
                    demo_output_masks=batch["demo_output_masks"],
                    demo_mask=batch["demo_mask"],
                    query_input_grid=batch["query_input_grid"],
                    query_input_mask=batch["query_input_mask"],
                )
                loss, parts = compute_loss(
                    model_out,
                    batch,
                    size_loss_weight=size_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += parts["total_loss"]
            running_cell_loss += parts["cell_loss"]
            running_height_loss += parts["height_loss"]
            running_width_loss += parts["width_loss"]
            steps += 1

            if step % log_every == 0:
                progress.set_postfix({"loss": f"{parts['total_loss']:.4f}"})

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": running_total_loss / max(1, steps),
            "train_cell_loss": running_cell_loss / max(1, steps),
            "train_height_loss": running_height_loss / max(1, steps),
            "train_width_loss": running_width_loss / max(1, steps),
        }

        if epoch % eval_every == 0:
            train_eval = evaluate_task_conditioned_solve_rate(
                model,
                train_tasks,
                split=train_split,
                device=device,
                max_grid=int(cfg.model.max_grid),
                max_demos=int(cfg.model.max_demos),
                pad_color=int(cfg.model.pad_color),
            )
            eval_eval = evaluate_task_conditioned_solve_rate(
                model,
                eval_tasks,
                split=eval_split,
                device=device,
                max_grid=int(cfg.model.max_grid),
                max_demos=int(cfg.model.max_demos),
                pad_color=int(cfg.model.pad_color),
            )
            epoch_metrics["train_task_metrics"] = asdict(train_eval)
            epoch_metrics["eval_task_metrics"] = asdict(eval_eval)

            print(
                f"Epoch {epoch}: "
                f"train solve {train_eval.solved_tasks}/{train_eval.total_tasks} "
                f"({100.0 * train_eval.solve_rate:.2f}%), "
                f"eval solve {eval_eval.solved_tasks}/{eval_eval.total_tasks} "
                f"({100.0 * eval_eval.solve_rate:.2f}%)"
            )

        history.append(epoch_metrics)

    checkpoint_path = _abs_path(project_root, str(cfg.output.checkpoint_path))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "timestamp_utc": iso_now(),
            "commit": current_commit(project_root),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    final_train_eval = evaluate_task_conditioned_solve_rate(
        model,
        train_tasks,
        split=train_split,
        device=device,
        max_grid=int(cfg.model.max_grid),
        max_demos=int(cfg.model.max_demos),
        pad_color=int(cfg.model.pad_color),
    )
    final_eval_eval = evaluate_task_conditioned_solve_rate(
        model,
        eval_tasks,
        split=eval_split,
        device=device,
        max_grid=int(cfg.model.max_grid),
        max_demos=int(cfg.model.max_demos),
        pad_color=int(cfg.model.pad_color),
    )

    metrics_payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "commit": current_commit(project_root),
        "device": str(device),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "history": history,
        "final": {
            "train": asdict(final_train_eval),
            "evaluation": asdict(final_eval_eval),
        },
        "checkpoint_path": str(checkpoint_path),
    }

    metrics_path = _abs_path(project_root, str(cfg.output.metrics_path))
    eval_path = _abs_path(project_root, str(cfg.output.eval_path))
    save_json(metrics_path, metrics_payload)
    save_json(
        eval_path,
        {
            "timestamp_utc": metrics_payload["timestamp_utc"],
            "commit": metrics_payload["commit"],
            "device": metrics_payload["device"],
            "final": metrics_payload["final"],
            "checkpoint_path": metrics_payload["checkpoint_path"],
        },
    )

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote final eval: {eval_path}")
    return metrics_payload


def main() -> int:
    args = parse_args()
    cfg = load_config(_project_root(), args.config, args.overrides)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
