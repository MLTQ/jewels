"""Linear-memory coupling for cell-local and neighboring jewel birth sets."""

from __future__ import annotations

import torch
import torch.nn as nn

from sol.streaming_model import ContextRasterEncoder


def rasterize_set_moments(
    values: torch.Tensor,
    cell_indices: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """Pool a variable-size hidden set into per-cell mean, variance, and occupancy."""
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("set values must have shape (N,D) with D positive")
    if cell_indices.shape != (len(values),) or cell_indices.dtype != torch.long:
        raise ValueError("cell_indices must be one int64 value per set row")
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if len(cell_indices) and (
        int(cell_indices.min()) < 0 or int(cell_indices.max()) >= n_cells
    ):
        raise ValueError("cell indices exceed the declared raster")
    total = values.new_zeros(n_cells, values.shape[1])
    square = torch.zeros_like(total)
    count = values.new_zeros(n_cells, 1)
    expanded = cell_indices[:, None].expand_as(values)
    total.scatter_add_(0, expanded, values)
    square.scatter_add_(0, expanded, values.square())
    count.scatter_add_(0, cell_indices[:, None], values.new_ones(len(values), 1))
    mean = total / count.clamp_min(1)
    variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
    return torch.cat((mean, variance, count.log1p(), (count > 0).to(values)), dim=1)


class NeighborhoodBirthSetBlock(nn.Module):
    """Return a permutation-equivariant residual from cell and neighbor set state."""

    def __init__(
        self,
        model_dim: int,
        grid_shape: tuple[int, int, int],
        *,
        raster_depth: int = 0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or model_dim % 8:
            raise ValueError("model_dim must be positive and divisible by eight")
        if raster_depth < 0:
            raise ValueError("raster_depth must be non-negative")
        self.grid_shape = grid_shape
        self.n_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.set_encoder = ContextRasterEncoder(
            model_dim * 2 + 2,
            model_dim,
            grid_shape,
            raster_depth,
        )
        self.row_update = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        nn.init.zeros_(self.row_update[-1].weight)
        nn.init.zeros_(self.row_update[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden birth rows must have shape (N,D)")
        moments = rasterize_set_moments(hidden, cell_indices, self.n_cells)
        neighborhood = self.set_encoder(moments, spatial=True)[0]
        update = self.row_update(
            torch.cat((hidden, neighborhood[cell_indices]), dim=1)
        )
        return hidden + update
