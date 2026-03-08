"""Neural baseline model for ARC grid-to-grid prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


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

        self.height_head = nn.Linear(d_model, max_grid)
        self.width_head = nn.Linear(d_model, max_grid)
        self.cell_head = nn.Linear(d_model, num_colors)

    def forward(self, input_grid: Tensor, input_mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
        - input_grid: [batch, max_grid, max_grid], token ids
        - input_mask: [batch, max_grid, max_grid], True for valid input cells
        """
        batch_size, height, width = input_grid.shape
        if height != self.max_grid or width != self.max_grid:
            raise ValueError(
                f"Expected grid shape [B, {self.max_grid}, {self.max_grid}], "
                f"got {tuple(input_grid.shape)}."
            )

        x = self.token_embedding(input_grid)

        row_ids = torch.arange(self.max_grid, device=input_grid.device)
        col_ids = torch.arange(self.max_grid, device=input_grid.device)
        row_emb = self.row_embedding(row_ids).view(1, self.max_grid, 1, -1)
        col_emb = self.col_embedding(col_ids).view(1, 1, self.max_grid, -1)
        x = x + row_emb + col_emb

        x = x.view(batch_size, self.max_grid * self.max_grid, -1)
        pad_mask = ~input_mask.view(batch_size, self.max_grid * self.max_grid)

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.output_norm(x)

        valid_mask = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        height_logits = self.height_head(pooled)
        width_logits = self.width_head(pooled)
        cell_logits = self.cell_head(x).view(batch_size, self.max_grid, self.max_grid, self.num_colors)

        return {
            "height_logits": height_logits,
            "width_logits": width_logits,
            "cell_logits": cell_logits,
        }
