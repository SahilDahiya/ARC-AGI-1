"""Neural baseline model for ARC grid-to-grid prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ArcGridEncoder(nn.Module):
    """Shared 2D grid encoder used by ARC models."""

    def __init__(
        self,
        *,
        max_grid: int,
        num_colors: int,
        pad_color: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_grid = max_grid
        self.num_colors = num_colors
        self.pad_color = pad_color

        self.token_embedding = nn.Embedding(num_colors + 1, d_model, padding_idx=pad_color)
        self.row_embedding = nn.Embedding(max_grid, d_model)
        self.col_embedding = nn.Embedding(max_grid, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def encode(self, grid: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, height, width = grid.shape
        if height != self.max_grid or width != self.max_grid:
            raise ValueError(
                f"Expected grid shape [B, {self.max_grid}, {self.max_grid}], "
                f"got {tuple(grid.shape)}."
            )

        x = self.token_embedding(grid)

        row_ids = torch.arange(self.max_grid, device=grid.device)
        col_ids = torch.arange(self.max_grid, device=grid.device)
        row_emb = self.row_embedding(row_ids).view(1, self.max_grid, 1, -1)
        col_emb = self.col_embedding(col_ids).view(1, 1, self.max_grid, -1)
        x = x + row_emb + col_emb

        x = x.view(batch_size, self.max_grid * self.max_grid, -1)
        pad_mask = ~mask.view(batch_size, self.max_grid * self.max_grid)

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.output_norm(x)
        return x, pad_mask

    @staticmethod
    def masked_mean_pool(x: Tensor, pad_mask: Tensor) -> Tensor:
        valid_mask = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
        return (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)


class ArcGridBaselineModel(nn.Module):
    """
    2D-aware Transformer encoder baseline.

    The model predicts:
    - output height (1..max_grid)
    - output width (1..max_grid)
    - output color logits at each cell in a max_grid x max_grid canvas
    """

    def __init__(
        self,
        *,
        max_grid: int = 30,
        num_colors: int = 10,
        pad_color: int = 10,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_grid = max_grid
        self.num_colors = num_colors
        self.pad_color = pad_color

        self.grid_encoder = _ArcGridEncoder(
            max_grid=max_grid,
            num_colors=num_colors,
            pad_color=pad_color,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.height_head = nn.Linear(d_model, max_grid)
        self.width_head = nn.Linear(d_model, max_grid)
        self.cell_head = nn.Linear(d_model, num_colors)

    def forward(self, input_grid: Tensor, input_mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
        - input_grid: [batch, max_grid, max_grid], token ids
        - input_mask: [batch, max_grid, max_grid], True for valid input cells
        """
        batch_size = input_grid.shape[0]
        x, pad_mask = self.grid_encoder.encode(input_grid, input_mask)
        pooled = self.grid_encoder.masked_mean_pool(x, pad_mask)

        height_logits = self.height_head(pooled)
        width_logits = self.width_head(pooled)
        cell_logits = self.cell_head(x).view(batch_size, self.max_grid, self.max_grid, self.num_colors)

        return {
            "height_logits": height_logits,
            "width_logits": width_logits,
            "cell_logits": cell_logits,
        }


class ArcTaskConditionedModel(nn.Module):
    """
    Task-conditioned ARC model.

    The model encodes:
    - demonstration input grids
    - demonstration output grids
    - the current query input grid

    It then conditions query predictions on the pooled demo context.
    """

    def __init__(
        self,
        *,
        max_grid: int = 30,
        max_demos: int = 4,
        num_colors: int = 10,
        pad_color: int = 10,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_grid = max_grid
        self.max_demos = max_demos
        self.num_colors = num_colors
        self.pad_color = pad_color

        self.grid_encoder = _ArcGridEncoder(
            max_grid=max_grid,
            num_colors=num_colors,
            pad_color=pad_color,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.demo_projection = nn.Linear(d_model * 2, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.context_to_cells = nn.Linear(d_model, d_model)
        self.height_head = nn.Linear(d_model, max_grid)
        self.width_head = nn.Linear(d_model, max_grid)
        self.cell_head = nn.Linear(d_model, num_colors)

    def forward(
        self,
        demo_input_grids: Tensor,
        demo_input_masks: Tensor,
        demo_output_grids: Tensor,
        demo_output_masks: Tensor,
        demo_mask: Tensor,
        query_input_grid: Tensor,
        query_input_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, num_demos, height, width = demo_input_grids.shape
        if num_demos > self.max_demos:
            raise ValueError(f"Received {num_demos} demos, exceeds max_demos={self.max_demos}.")
        if height != self.max_grid or width != self.max_grid:
            raise ValueError(
                f"Expected demo grid shape [B, M, {self.max_grid}, {self.max_grid}], "
                f"got {tuple(demo_input_grids.shape)}."
            )

        flat_demo_inputs = demo_input_grids.view(batch_size * num_demos, self.max_grid, self.max_grid)
        flat_demo_input_masks = demo_input_masks.view(batch_size * num_demos, self.max_grid, self.max_grid)
        flat_demo_outputs = demo_output_grids.view(batch_size * num_demos, self.max_grid, self.max_grid)
        flat_demo_output_masks = demo_output_masks.view(batch_size * num_demos, self.max_grid, self.max_grid)

        demo_input_tokens, demo_input_pad = self.grid_encoder.encode(flat_demo_inputs, flat_demo_input_masks)
        demo_output_tokens, demo_output_pad = self.grid_encoder.encode(flat_demo_outputs, flat_demo_output_masks)

        demo_input_pooled = self.grid_encoder.masked_mean_pool(demo_input_tokens, demo_input_pad).view(
            batch_size,
            num_demos,
            -1,
        )
        demo_output_pooled = self.grid_encoder.masked_mean_pool(demo_output_tokens, demo_output_pad).view(
            batch_size,
            num_demos,
            -1,
        )
        demo_repr = self.demo_projection(torch.cat([demo_input_pooled, demo_output_pooled], dim=-1))

        demo_valid = demo_mask.unsqueeze(-1).to(dtype=demo_repr.dtype)
        demo_context = (demo_repr * demo_valid).sum(dim=1) / demo_valid.sum(dim=1).clamp(min=1.0)

        query_tokens, query_pad = self.grid_encoder.encode(query_input_grid, query_input_mask)
        query_pooled = self.grid_encoder.masked_mean_pool(query_tokens, query_pad)

        global_context = self.context_norm(query_pooled + demo_context)
        cell_context = self.context_to_cells(global_context).unsqueeze(1)
        query_tokens = query_tokens + cell_context

        height_logits = self.height_head(global_context)
        width_logits = self.width_head(global_context)
        cell_logits = self.cell_head(query_tokens).view(batch_size, self.max_grid, self.max_grid, self.num_colors)

        return {
            "height_logits": height_logits,
            "width_logits": width_logits,
            "cell_logits": cell_logits,
        }
