"""Training and evaluation utilities for neural ARC baselines."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from arc_agi_1.data import Grid, TaskMap
from arc_agi_1.dataset import decode_grid, encode_grid
from arc_agi_1.model import ArcGridBaselineModel


@dataclass(slots=True)
class TaskEvalMetrics:
    split: str
    solved_tasks: int
    total_tasks: int
    solve_rate: float
    exact_pair_matches: int
    total_pairs: int
    pair_accuracy: float


def resolve_device(device_spec: str) -> torch.device:
    """Resolve a device string with an 'auto' option."""
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_commit(project_root: Path) -> str:
    """Best-effort git commit hash."""
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


def autocast_context(device_type: str, enabled: bool):
    """Return an autocast context manager for the active device type."""
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=enabled)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move all tensors in a batch dictionary to the target device."""
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_loss(
    model_out: dict[str, Tensor],
    batch: dict[str, object],
    *,
    size_loss_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute combined dimension and cell losses."""
    target_h = batch["output_h_idx"]
    target_w = batch["output_w_idx"]
    target_cells = batch["output_grid"]
    target_cell_mask = batch["output_mask"]

    if not isinstance(target_h, Tensor):
        raise TypeError("output_h_idx must be a Tensor.")
    if not isinstance(target_w, Tensor):
        raise TypeError("output_w_idx must be a Tensor.")
    if not isinstance(target_cells, Tensor):
        raise TypeError("output_grid must be a Tensor.")
    if not isinstance(target_cell_mask, Tensor):
        raise TypeError("output_mask must be a Tensor.")

    height_loss = F.cross_entropy(model_out["height_logits"], target_h)
    width_loss = F.cross_entropy(model_out["width_logits"], target_w)

    cell_logits = model_out["cell_logits"].reshape(-1, model_out["cell_logits"].shape[-1])
    cell_targets = target_cells.reshape(-1)
    cell_mask = target_cell_mask.reshape(-1).to(dtype=cell_logits.dtype)

    cell_losses = F.cross_entropy(cell_logits, cell_targets, reduction="none")
    masked_cell_loss = (cell_losses * cell_mask).sum() / cell_mask.sum().clamp(min=1.0)

    total_loss = masked_cell_loss + size_loss_weight * (height_loss + width_loss)

    parts = {
        "total_loss": float(total_loss.detach().cpu()),
        "cell_loss": float(masked_cell_loss.detach().cpu()),
        "height_loss": float(height_loss.detach().cpu()),
        "width_loss": float(width_loss.detach().cpu()),
    }
    return total_loss, parts


@torch.no_grad()
def predict_single_output(
    model: ArcGridBaselineModel,
    input_grid: Grid,
    *,
    device: torch.device,
    max_grid: int,
    pad_color: int,
) -> Grid:
    """Predict one output grid from one input grid."""
    model.eval()

    input_tensor, input_mask = encode_grid(input_grid, max_grid=max_grid, pad_color=pad_color)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_mask = input_mask.unsqueeze(0).to(device)

    out = model(input_tensor, input_mask)
    pred_h = int(torch.argmax(out["height_logits"], dim=-1).item()) + 1
    pred_w = int(torch.argmax(out["width_logits"], dim=-1).item()) + 1
    pred_cells = torch.argmax(out["cell_logits"][0], dim=-1).to(dtype=torch.long).cpu()
    return decode_grid(pred_cells, height=pred_h, width=pred_w)


@torch.no_grad()
def evaluate_task_solve_rate(
    model: ArcGridBaselineModel,
    tasks: TaskMap,
    *,
    split: str,
    device: torch.device,
    max_grid: int,
    pad_color: int,
) -> TaskEvalMetrics:
    """Evaluate strict ARC solved-task rate for a task split."""
    solved_tasks = 0
    total_tasks = len(tasks)
    exact_pair_matches = 0
    total_pairs = 0

    for task in tasks.values():
        task_solved = True
        for pair in task["test"]:
            predicted = predict_single_output(
                model,
                pair["input"],
                device=device,
                max_grid=max_grid,
                pad_color=pad_color,
            )
            expected = pair["output"]
            total_pairs += 1
            if predicted == expected:
                exact_pair_matches += 1
            else:
                task_solved = False
        if task_solved:
            solved_tasks += 1

    solve_rate = solved_tasks / total_tasks if total_tasks else 0.0
    pair_accuracy = exact_pair_matches / total_pairs if total_pairs else 0.0

    return TaskEvalMetrics(
        split=split,
        solved_tasks=solved_tasks,
        total_tasks=total_tasks,
        solve_rate=solve_rate,
        exact_pair_matches=exact_pair_matches,
        total_pairs=total_pairs,
        pair_accuracy=pair_accuracy,
    )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def iso_now() -> str:
    return datetime.now(UTC).isoformat()
