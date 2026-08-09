"""Compact occupied-group tokenizer for dense fitted jewel fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.sparse_autoencoder import _cell_basis, _rank_basis
from sol.token_grid import CompactGrid, GridSpec, OccupancyGrid


@dataclass
class GroupedLatents:
    """Ragged occupied token stream plus exact discrete topology."""

    values: torch.Tensor
    batch_indices: torch.Tensor
    cell_indices: torch.Tensor
    group_indices: torch.Tensor
    group_counts: torch.Tensor
    batch_size: int


@dataclass
class GroupedAutoencoderOutput:
    occupied_features: torch.Tensor


class GroupedTokenEncoder(nn.Module):
    """Encode small canonical jewel groups without allocating empty raster tokens."""

    def __init__(
        self,
        feature_dim: int,
        model_dim: int,
        latent_dim: int,
        spec: GridSpec,
        jewels_per_token: int,
        depth: int,
    ) -> None:
        super().__init__()
        if jewels_per_token <= 0:
            raise ValueError("jewels per token must be positive")
        if depth < 0:
            raise ValueError("encoder depth cannot be negative")
        self.spec = spec
        self.grid = OccupancyGrid(spec)
        self.jewels_per_token = jewels_per_token
        self.groups_per_cell = math.ceil(spec.slots_per_cell / jewels_per_token)
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 8, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.stats_proj = nn.Linear(model_dim * 2 + 1, model_dim)
        self.cell_position_proj = nn.Linear(27, model_dim)
        self.group_position_proj = nn.Linear(8, model_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim * 4),
                nn.GELU(),
                nn.Linear(model_dim * 4, model_dim),
            )
            for _ in range(depth)
        )
        self.out = nn.Linear(model_dim, latent_dim)

    def forward(
        self, features: torch.Tensor, target: CompactGrid | None = None
    ) -> GroupedLatents:
        if features.ndim != 3:
            raise ValueError("encoder expects (B,N,F)")
        compact = self.grid.pack_compact(features) if target is None else target
        batch, jewels = compact.cell_indices.shape
        batch_indices = torch.arange(batch, device=features.device)[:, None].expand(
            batch, jewels
        )
        group_indices = compact.slot_indices // self.jewels_per_token
        within_group = compact.slot_indices % self.jewels_per_token
        stride = self.spec.n_cells * self.groups_per_cell
        keys = (
            batch_indices * stride
            + compact.cell_indices * self.groups_per_cell
            + group_indices
        ).reshape(-1)
        unique_keys, inverse, group_counts = torch.unique(
            keys, sorted=True, return_inverse=True, return_counts=True
        )
        flat_values = compact.values.reshape(-1, compact.values.shape[-1])
        rank = _rank_basis(
            within_group.reshape(-1), self.jewels_per_token, flat_values.dtype
        )
        projected = self.in_proj(torch.cat([flat_values, rank], dim=-1)).float()
        total = projected.new_zeros(unique_keys.shape[0], projected.shape[-1])
        square = torch.zeros_like(total)
        scatter = inverse[:, None].expand_as(projected)
        total.scatter_add_(0, scatter, projected)
        square.scatter_add_(0, scatter, projected.square())
        count = group_counts.to(projected.dtype)[:, None]
        mean = total / count
        variance = (square / count - mean.square()).clamp_min(0)
        normalized_count = torch.log1p(count) / math.log(self.jewels_per_token + 1)
        unique_batch = unique_keys // stride
        remainder = unique_keys % stride
        unique_cells = remainder // self.groups_per_cell
        unique_groups = remainder % self.groups_per_cell
        hidden = self.stats_proj(
            torch.cat([mean, variance, normalized_count], dim=-1)
        )
        hidden = (
            hidden
            + self.cell_position_proj(
                _cell_basis(unique_cells, self.spec, hidden.dtype)
            )
            + self.group_position_proj(
                _rank_basis(unique_groups, self.groups_per_cell, hidden.dtype)
            )
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return GroupedLatents(
            values=self.out(hidden),
            batch_indices=unique_batch,
            cell_indices=unique_cells,
            group_indices=unique_groups,
            group_counts=group_counts,
            batch_size=batch,
        )


class GroupedTokenDecoder(nn.Module):
    """Decode each occupied group token into its exact number of local jewels."""

    def __init__(
        self,
        feature_dim: int,
        model_dim: int,
        latent_dim: int,
        spec: GridSpec,
        jewels_per_token: int,
        depth: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("decode chunk size must be positive")
        self.spec = spec
        self.jewels_per_token = jewels_per_token
        self.groups_per_cell = math.ceil(spec.slots_per_cell / jewels_per_token)
        self.chunk_size = chunk_size
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        self.cell_position_proj = nn.Linear(27, model_dim)
        self.group_position_proj = nn.Linear(8, model_dim)
        self.rank_proj = nn.Linear(8, model_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim * 4),
                nn.GELU(),
                nn.Linear(model_dim * 4, model_dim),
            )
            for _ in range(depth)
        )
        self.feature_head = nn.Linear(model_dim, feature_dim)

    def _decode_chunk(
        self,
        latents: GroupedLatents,
        token_indices: torch.Tensor,
        within_group: torch.Tensor,
    ) -> torch.Tensor:
        cells = latents.cell_indices[token_indices]
        groups = latents.group_indices[token_indices]
        hidden = (
            self.latent_proj(latents.values[token_indices])
            + self.cell_position_proj(_cell_basis(cells, self.spec, latents.values.dtype))
            + self.group_position_proj(
                _rank_basis(groups, self.groups_per_cell, latents.values.dtype)
            )
            + self.rank_proj(
                _rank_basis(within_group, self.jewels_per_token, latents.values.dtype)
            )
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        raw = self.feature_head(hidden)
        gu, gv, gt = self.spec.shape
        t = cells % gt
        v = (cells // gt) % gv
        u = cells // (gv * gt)
        coordinate = torch.stack([u, v, t], dim=-1).to(latents.values.dtype)
        cell_size = latents.values.new_tensor([2 / gu, 2 / gv, 2 / gt])
        center = -1 + (coordinate + 0.5) * cell_size
        constrained = center + raw[:, :3].tanh() * cell_size * 0.5
        return torch.cat([constrained, raw[:, 3:]], dim=-1)

    def decode_tokens(
        self, latents: GroupedLatents, token_indices: torch.Tensor
    ) -> torch.Tensor:
        counts = latents.group_counts[token_indices]
        repeated = torch.repeat_interleave(token_indices, counts)
        offsets = counts.cumsum(0) - counts
        within_group = torch.arange(
            len(repeated), device=latents.values.device
        ) - torch.repeat_interleave(offsets, counts)
        pieces = []
        for start in range(0, len(repeated), self.chunk_size):
            part = slice(start, start + self.chunk_size)
            pieces.append(
                self._decode_chunk(latents, repeated[part], within_group[part])
            )
        return (
            torch.cat(pieces)
            if pieces
            else latents.values.new_empty((0, self.feature_head.out_features))
        )

    def forward_training(self, latents: GroupedLatents) -> GroupedAutoencoderOutput:
        token_indices = torch.arange(len(latents.values), device=latents.values.device)
        decoded = self.decode_tokens(latents, token_indices)
        totals = torch.bincount(
            latents.batch_indices,
            weights=latents.group_counts.to(latents.values.dtype),
            minlength=latents.batch_size,
        ).long()
        if not torch.equal(totals, totals[:1].expand_as(totals)):
            raise ValueError("training batches must contain equal jewel counts")
        return GroupedAutoencoderOutput(
            occupied_features=decoded.reshape(latents.batch_size, int(totals[0]), -1)
        )

    @torch.no_grad()
    def decode(self, latents: GroupedLatents) -> list[torch.Tensor]:
        outputs = []
        for batch_index in range(latents.batch_size):
            token_indices = torch.nonzero(
                latents.batch_indices == batch_index, as_tuple=False
            ).flatten()
            outputs.append(self.decode_tokens(latents, token_indices))
        return outputs


class GroupedSparseJewelAutoencoder(nn.Module):
    """Sparse topology plus multiple local content tokens per occupied raster cell."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 32,
        spec: GridSpec = GridSpec((64, 64, 64), 128),
        jewels_per_token: int = 8,
        enc_depth: int = 1,
        dec_depth: int = 4,
        decode_chunk_size: int = 32_768,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.grid = OccupancyGrid(spec)
        self.jewels_per_token = jewels_per_token
        self.encoder = GroupedTokenEncoder(
            feature_dim,
            model_dim,
            latent_dim,
            spec,
            jewels_per_token,
            enc_depth,
        )
        self.decoder = GroupedTokenDecoder(
            feature_dim,
            model_dim,
            latent_dim,
            spec,
            jewels_per_token,
            dec_depth,
            decode_chunk_size,
        )

    def forward_compact(
        self, features: torch.Tensor, target: CompactGrid
    ) -> GroupedAutoencoderOutput:
        return self.decoder.forward_training(self.encoder(features, target))

    def decode(self, latents: GroupedLatents) -> list[torch.Tensor]:
        return self.decoder.decode(latents)

    def structural_loss(
        self,
        output: GroupedAutoencoderOutput,
        target: CompactGrid,
        **_: object,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature_error = F.smooth_l1_loss(output.occupied_features, target.values)
        return feature_error, {
            "feature": feature_error.detach(),
            "count": feature_error.detach().new_zeros(()),
        }
