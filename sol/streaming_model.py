"""Prefix-conditioned sparse birth model for persistent jewel continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.streaming_data import BirthTarget
from sol.token_grid import GridSpec


def _rank_basis(
    slot_indices: torch.Tensor, slots_per_cell: int, dtype: torch.dtype
) -> torch.Tensor:
    rank = slot_indices.to(dtype)
    normalized = rank / max(slots_per_cell - 1, 1)
    log_rank = torch.log1p(rank) / math.log(slots_per_cell + 1)
    wavelengths = rank.new_tensor([4.0, 16.0, 64.0])
    phases = rank[..., None] * (2 * math.pi) / wavelengths
    return torch.cat(
        (normalized[:, None], log_rank[:, None], phases.sin(), phases.cos()), dim=1
    )


class ResidualMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class ContextRasterEncoder(nn.Module):
    """Compress count/moment prefix rasters with local 3D convolutions."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        grid_shape: tuple[int, int, int],
        depth: int,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.input = nn.Conv3d(input_dim, model_dim, 3, padding=1)
        self.position = nn.Parameter(
            torch.randn(1, model_dim, *grid_shape) * 0.02
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(8, model_dim),
                nn.SiLU(),
                nn.Conv3d(model_dim, model_dim, 3, padding=1),
                nn.GroupNorm(8, model_dim),
                nn.SiLU(),
                nn.Conv3d(model_dim, model_dim, 3, padding=1),
            )
            for _ in range(depth)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.SiLU(),
        )

    def forward(self, raster: torch.Tensor, *, spatial: bool = False) -> torch.Tensor:
        cells = math.prod(self.grid_shape)
        if raster.ndim == 2:
            raster = raster[None]
        if raster.ndim != 3 or raster.shape[1] != cells:
            raise ValueError(f"context raster must have shape (B,{cells},C)")
        batch = raster.shape[0]
        volume = raster.reshape(batch, *self.grid_shape, raster.shape[-1]).permute(
            0, 4, 1, 2, 3
        )
        hidden = self.input(volume) + self.position
        for block in self.blocks:
            hidden = hidden + block(hidden)
        mean = hidden.mean(dim=(2, 3, 4))
        maximum = hidden.amax(dim=(2, 3, 4))
        global_context = self.out(torch.cat((mean, maximum), dim=1))
        if not spatial:
            return global_context
        local_context = hidden.permute(0, 2, 3, 4, 1).reshape(
            batch, cells, hidden.shape[1]
        )
        return local_context + global_context[:, None]


@dataclass
class BirthModelOutput:
    occupied_features: torch.Tensor
    log_count: torch.Tensor
    context: torch.Tensor


@dataclass
class DecodedBirths:
    values: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    counts: torch.Tensor


class BirthContinuationModel(nn.Module):
    """Predict future birth counts and marks without modifying carried jewels."""

    def __init__(
        self,
        feature_dim: int = 22,
        context_dim: int = 46,
        model_dim: int = 128,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 256),
        context_depth: int = 2,
        cell_depth: int = 3,
        slot_depth: int = 3,
        context_mode: str = "global",
        text_dim: int = 0,
    ) -> None:
        super().__init__()
        if model_dim % 8:
            raise ValueError("model_dim must be divisible by eight")
        if context_mode not in {"global", "local"}:
            raise ValueError("context_mode must be 'global' or 'local'")
        if text_dim < 0:
            raise ValueError("text_dim must be non-negative")
        self.feature_dim = feature_dim
        self.grid_spec = grid_spec
        self.context_mode = context_mode
        self.text_dim = text_dim
        self.context_encoder = ContextRasterEncoder(
            context_dim, model_dim, grid_spec.shape, context_depth
        )
        gu, gv, gt = grid_spec.shape
        self.u_embedding = nn.Parameter(torch.randn(gu, model_dim) * 0.02)
        self.v_embedding = nn.Parameter(torch.randn(gv, model_dim) * 0.02)
        self.t_embedding = nn.Parameter(torch.randn(gt, model_dim) * 0.02)
        self.context_projection = nn.Linear(model_dim, model_dim)
        self.text_projection = (
            nn.Linear(text_dim, model_dim) if text_dim else None
        )
        self.null_text_condition = (
            nn.Parameter(torch.zeros(model_dim)) if text_dim else None
        )
        self.cell_blocks = nn.ModuleList(
            ResidualMLP(model_dim) for _ in range(cell_depth)
        )
        self.count_head = nn.Linear(model_dim, 1)
        self.rank_projection = nn.Linear(8, model_dim)
        self.slot_blocks = nn.ModuleList(
            ResidualMLP(model_dim) for _ in range(slot_depth)
        )
        self.feature_head = nn.Linear(model_dim, feature_dim)

    def encode_context(self, raster: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(raster, spatial=self.context_mode == "local")

    def _project_text(
        self,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None,
        *,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.text_projection is None or self.null_text_condition is None:
            if text_condition is not None or drop_condition is not None:
                raise ValueError("model was constructed without text conditioning")
            return None
        null = self.null_text_condition.to(reference)[None].expand(batch, -1)
        if text_condition is None:
            if drop_condition is not None:
                raise ValueError("drop_condition requires a text condition")
            return null
        if text_condition.ndim == 1:
            text_condition = text_condition[None]
        if text_condition.shape != (batch, self.text_dim):
            raise ValueError(
                f"text condition must have shape ({batch},{self.text_dim})"
            )
        projected = self.text_projection(text_condition.to(reference))
        if drop_condition is None:
            return projected
        if drop_condition.shape != (batch,) or drop_condition.dtype != torch.bool:
            raise ValueError("drop_condition must be boolean with one value per batch")
        return torch.where(drop_condition[:, None].to(reference.device), null, projected)

    def cell_states(
        self,
        context: torch.Tensor,
        text_condition: torch.Tensor | None = None,
        drop_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim == 1:
            context = context[None]
        gu, gv, gt = self.grid_spec.shape
        positional = (
            self.u_embedding[:, None, None]
            + self.v_embedding[None, :, None]
            + self.t_embedding[None, None, :]
        ).reshape(self.grid_spec.n_cells, -1)
        if context.ndim == 2:
            conditioned = self.context_projection(context)[:, None]
        elif context.ndim == 3 and context.shape[1] == self.grid_spec.n_cells:
            conditioned = self.context_projection(context)
        else:
            raise ValueError(
                "context must have shape (B,D) or (B,n_cells,D)"
            )
        text = self._project_text(
            text_condition,
            drop_condition,
            batch=context.shape[0],
            reference=context,
        )
        if text is not None:
            conditioned = conditioned + text[:, None]
        hidden = positional[None] + conditioned
        for block in self.cell_blocks:
            hidden = block(hidden)
        return hidden

    def decode_indices(
        self,
        cell_states: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        if cell_states.shape[0] != 1:
            raise ValueError("sparse continuation decoding currently expects batch size one")
        hidden = cell_states[0, cell_indices] + self.rank_projection(
            _rank_basis(slot_indices, self.grid_spec.slots_per_cell, cell_states.dtype)
        )
        for block in self.slot_blocks:
            hidden = block(hidden)
        return self.feature_head(hidden)

    def forward_training(
        self,
        context_raster: torch.Tensor,
        target: BirthTarget,
        text_condition: torch.Tensor | None = None,
        drop_condition: torch.Tensor | None = None,
    ) -> BirthModelOutput:
        context = self.encode_context(context_raster)
        return self.forward_from_context(
            context, target, text_condition, drop_condition
        )

    def forward_from_context(
        self,
        context: torch.Tensor,
        target: BirthTarget,
        text_condition: torch.Tensor | None = None,
        drop_condition: torch.Tensor | None = None,
    ) -> BirthModelOutput:
        """Evaluate target ranks from a precomputed or ablated context embedding."""
        cells = self.cell_states(context, text_condition, drop_condition)
        return BirthModelOutput(
            occupied_features=self.decode_indices(
                cells, target.cell_indices, target.slot_indices
            ),
            log_count=self.count_head(cells).squeeze(0).squeeze(-1),
            context=context,
        )

    def loss(
        self,
        output: BirthModelOutput,
        normalized_target_values: torch.Tensor,
        target_counts: torch.Tensor,
        *,
        count_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature = F.smooth_l1_loss(
            output.occupied_features, normalized_target_values
        )
        count = F.smooth_l1_loss(
            output.log_count, torch.log1p(target_counts.to(output.log_count.dtype))
        )
        return feature + count_weight * count, {
            "feature": feature.detach(),
            "count": count.detach(),
        }

    @torch.no_grad()
    def decode(
        self,
        context_raster: torch.Tensor,
        text_condition: torch.Tensor | None = None,
    ) -> DecodedBirths:
        context = self.encode_context(context_raster)
        cells = self.cell_states(context, text_condition)
        maximum = cells.new_tensor(float(self.grid_spec.slots_per_cell + 1)).log()
        counts = self.count_head(cells).squeeze(0).squeeze(-1).clamp(0, maximum)
        counts = counts.expm1().round().long()
        cell_indices = torch.repeat_interleave(
            torch.arange(self.grid_spec.n_cells, device=cells.device), counts
        )
        offsets = counts.cumsum(0) - counts
        slot_indices = torch.arange(len(cell_indices), device=cells.device) - torch.repeat_interleave(
            offsets, counts
        )
        values = self.decode_indices(cells, cell_indices, slot_indices)
        return DecodedBirths(values, cell_indices, slot_indices, counts)
