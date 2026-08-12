"""Video-scaffold-conditioned occupied-cell and birth-count prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.streaming_model import ContextRasterEncoder, ResidualMLP
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class ScaffoldTopologyOutput:
    occupancy_logits: torch.Tensor
    positive_count_raw: torch.Tensor

    @property
    def positive_counts(self) -> torch.Tensor:
        return F.softplus(self.positive_count_raw) + 1.0

    @property
    def expected_counts(self) -> torch.Tensor:
        return torch.sigmoid(self.occupancy_logits) * self.positive_counts


class ScaffoldTopologyModel(nn.Module):
    """Predict discrete frontier topology from aligned video and carried state."""

    def __init__(
        self,
        *,
        guide_dim: int = 3,
        carry_dim: int = 3,
        model_dim: int = 64,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
        encoder_depth: int = 3,
        cell_depth: int = 2,
    ) -> None:
        super().__init__()
        if guide_dim <= 0 or carry_dim < 0:
            raise ValueError("guide_dim must be positive and carry_dim non-negative")
        if model_dim % 8:
            raise ValueError("model_dim must be divisible by eight")
        self.guide_dim = guide_dim
        self.carry_dim = carry_dim
        self.grid_spec = grid_spec
        self.encoder = ContextRasterEncoder(
            guide_dim + carry_dim,
            model_dim,
            grid_spec.shape,
            encoder_depth,
        )
        self.cell_blocks = nn.ModuleList(
            ResidualMLP(model_dim) for _ in range(cell_depth)
        )
        self.occupancy_head = nn.Linear(model_dim, 1)
        self.positive_count_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        guide_raster: torch.Tensor,
        carry_raster: torch.Tensor | None = None,
    ) -> ScaffoldTopologyOutput:
        squeeze = guide_raster.ndim == 2
        guide = guide_raster[None] if squeeze else guide_raster
        if guide.ndim != 3 or guide.shape[1:] != (
            self.grid_spec.n_cells,
            self.guide_dim,
        ):
            raise ValueError(
                "guide_raster must have shape (cells,guide_dim) or "
                "(batch,cells,guide_dim)"
            )
        if self.carry_dim:
            if carry_raster is None:
                carry = guide.new_zeros(
                    guide.shape[0], self.grid_spec.n_cells, self.carry_dim
                )
            else:
                carry = carry_raster[None] if carry_raster.ndim == 2 else carry_raster
                if carry.shape != (
                    guide.shape[0],
                    self.grid_spec.n_cells,
                    self.carry_dim,
                ):
                    raise ValueError("carry_raster does not match the guide batch/grid")
                carry = carry.to(guide)
            encoded_input = torch.cat((guide, carry), dim=-1)
        else:
            if carry_raster is not None:
                raise ValueError("model was constructed without carried-state channels")
            encoded_input = guide
        hidden = self.encoder(encoded_input, spatial=True)
        for block in self.cell_blocks:
            hidden = block(hidden)
        occupancy = self.occupancy_head(hidden).squeeze(-1)
        positive = self.positive_count_head(hidden).squeeze(-1)
        if squeeze:
            occupancy = occupancy[0]
            positive = positive[0]
        return ScaffoldTopologyOutput(occupancy, positive)

    def loss(
        self,
        output: ScaffoldTopologyOutput,
        target_counts: torch.Tensor,
        *,
        occupancy_weight: float = 1.0,
        positive_count_weight: float = 1.0,
        total_count_weight: float = 0.25,
        distribution_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if min(
            occupancy_weight,
            positive_count_weight,
            total_count_weight,
            distribution_weight,
        ) < 0:
            raise ValueError("topology loss weights must be non-negative")
        counts = target_counts.to(output.occupancy_logits)
        if counts.shape != output.occupancy_logits.shape:
            raise ValueError("target_counts must match the output cell shape")
        batched = counts.ndim == 2
        counts_batch = counts if batched else counts[None]
        logits_batch = (
            output.occupancy_logits if batched else output.occupancy_logits[None]
        )
        positive_batch = output.positive_counts if batched else output.positive_counts[None]
        expected_batch = output.expected_counts if batched else output.expected_counts[None]

        occupied = counts_batch > 0
        bce = F.binary_cross_entropy_with_logits(
            logits_batch, occupied.to(logits_batch.dtype), reduction="none"
        )
        occupancy_terms = []
        for row, row_occupied in zip(bce, occupied, strict=True):
            pieces = [row[row_occupied].mean()]
            if (~row_occupied).any():
                pieces.append(row[~row_occupied].mean())
            occupancy_terms.append(torch.stack(pieces).mean())
        occupancy = torch.stack(occupancy_terms).mean()

        positive_count = F.smooth_l1_loss(
            torch.log1p(positive_batch[occupied]),
            torch.log1p(counts_batch[occupied]),
        )
        total_count = F.smooth_l1_loss(
            torch.log1p(expected_batch.sum(dim=1)),
            torch.log1p(counts_batch.sum(dim=1)),
        )
        predicted_distribution = expected_batch / expected_batch.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        target_distribution = counts_batch / counts_batch.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        distribution = (
            predicted_distribution - target_distribution
        ).abs().sum(dim=1).mean()
        total = (
            occupancy_weight * occupancy
            + positive_count_weight * positive_count
            + total_count_weight * total_count
            + distribution_weight * distribution
        )
        return total, {
            "occupancy": occupancy.detach(),
            "positive_count": positive_count.detach(),
            "total_count": total_count.detach(),
            "distribution": distribution.detach(),
        }

    @torch.no_grad()
    def decode_counts(
        self,
        output: ScaffoldTopologyOutput,
        *,
        occupancy_threshold: float = 0.5,
    ) -> torch.Tensor:
        if not 0 < occupancy_threshold < 1:
            raise ValueError("occupancy_threshold must lie inside (0,1)")
        occupied = torch.sigmoid(output.occupancy_logits) >= occupancy_threshold
        positive = output.positive_counts.round().long().clamp(
            1, self.grid_spec.slots_per_cell
        )
        return torch.where(occupied, positive, torch.zeros_like(positive))
