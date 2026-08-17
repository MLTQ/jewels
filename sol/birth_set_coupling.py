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


class SsogBirthSetBlock(nn.Module):
    """Steered separable Gaussian-field coupling over the birth-cell raster.

    Each atom is a Gaussian over *relative* cell displacement (offset, width,
    weight); per-cell content applies bounded, cold-started residuals to all
    three (SSOG: Pisoni 2026). The field is applied with three 1D passes and an
    exact separable normalizer, so no cell-pair matrix is ever built, and the
    learned reach can span the whole raster --- the long-range structure that a
    fixed 3x3x3 convolution cannot express.
    """

    def __init__(
        self,
        model_dim: int,
        grid_shape: tuple[int, int, int],
        *,
        atoms: int = 4,
        max_offset: float = 4.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or model_dim % 8:
            raise ValueError("model_dim must be positive and divisible by eight")
        if atoms <= 0 or max_offset <= 0:
            raise ValueError("atoms and max_offset must be positive")
        self.grid_shape = grid_shape
        self.n_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.atoms = atoms
        self.max_offset = float(max_offset)
        self.moment_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 2 + 2),
            nn.Linear(model_dim * 2 + 2, model_dim),
            nn.SiLU(),
        )
        generator = torch.Generator().manual_seed(atoms * 7919 + model_dim)
        self.mu0 = nn.Parameter(
            torch.randn(atoms, 3, generator=generator)
        )
        self.log_sigma0 = nn.Parameter(
            torch.full((atoms, 3), float(torch.log(torch.tensor(1.5))))
        )
        self.log_lambda0 = nn.Parameter(torch.zeros(atoms))
        self.steer = nn.Linear(model_dim, atoms * 7)
        nn.init.zeros_(self.steer.weight)
        nn.init.zeros_(self.steer.bias)
        self.gate_mu = nn.Parameter(torch.full((), 0.01))
        self.gate_sigma = nn.Parameter(torch.full((), 0.01))
        self.gate_lambda = nn.Parameter(torch.full((), 0.01))
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(grid_shape[0], dtype=torch.float32),
                torch.arange(grid_shape[1], dtype=torch.float32),
                torch.arange(grid_shape[2], dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(self.n_cells, 3)
        self.register_buffer("cell_coordinates", coordinates)
        self.row_update = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        nn.init.zeros_(self.row_update[-1].weight)
        nn.init.zeros_(self.row_update[-1].bias)

    def _field_context(self, moments: torch.Tensor) -> torch.Tensor:
        """Gather one steered Gaussian-mixture context vector per cell."""
        state = self.moment_projection(moments.float())
        raw = self.steer(state).reshape(self.n_cells, self.atoms, 7)
        mu = self.mu0[None] + self.gate_mu * self.max_offset * torch.tanh(
            raw[..., 0:3]
        )
        sigma = torch.exp(
            self.log_sigma0[None] + self.gate_sigma * torch.tanh(raw[..., 3:6])
        ).clamp(0.3, float(max(self.grid_shape)))
        weights = torch.softmax(
            self.log_lambda0[None] + self.gate_lambda * torch.tanh(raw[..., 6]),
            dim=-1,
        )
        occupancy = moments[:, -1:].float()
        masked = torch.cat((state * occupancy, occupancy), dim=1)
        volume = masked.reshape(*self.grid_shape, masked.shape[-1])
        context = state.new_zeros(self.n_cells, state.shape[-1])
        for atom in range(self.atoms):
            kernels = []
            for axis, length in enumerate(self.grid_shape):
                sources = torch.arange(length, device=state.device, dtype=state.dtype)
                center = self.cell_coordinates[:, axis] + mu[:, atom, axis]
                kernels.append(
                    torch.exp(
                        -0.5
                        * (
                            (sources[None] - center[:, None])
                            / sigma[:, atom, axis][:, None]
                        ).square()
                    )
                )
            gathered = torch.einsum("cu,uvtd->cvtd", kernels[0], volume)
            gathered = torch.einsum("cv,cvtd->ctd", kernels[1], gathered)
            gathered = torch.einsum("ct,ctd->cd", kernels[2], gathered)
            normalizer = gathered[:, -1:].clamp_min(1e-6)
            context = context + weights[:, atom, None] * (
                gathered[:, :-1] / normalizer
            )
        return context

    def forward(
        self,
        hidden: torch.Tensor,
        cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden birth rows must have shape (N,D)")
        moments = rasterize_set_moments(hidden, cell_indices, self.n_cells)
        context = self._field_context(moments).to(hidden.dtype)
        update = self.row_update(
            torch.cat((hidden, context[cell_indices]), dim=1)
        )
        return hidden + update


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
