"""Variable-count sparse decoder for dense fitted jewel fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.autoencoder import OccupancyAwareEncoder
from sol.token_grid import CompactGrid, GridSpec, OccupancyGrid


@dataclass
class SparseAutoencoderOutput:
    occupied_features: torch.Tensor
    log_count: torch.Tensor


def _rank_basis(
    slot_indices: torch.Tensor, slots_per_cell: int, dtype: torch.dtype
) -> torch.Tensor:
    """Encode integer canonical ranks at local and cell-wide frequencies."""
    rank = slot_indices.to(dtype)
    normalized = rank / max(slots_per_cell - 1, 1)
    log_rank = torch.log1p(rank) / math.log(slots_per_cell + 1)
    wavelengths = rank.new_tensor([4.0, 16.0, 64.0])
    phases = rank[..., None] * (2 * math.pi) / wavelengths
    return torch.cat(
        [normalized[..., None], log_rank[..., None], phases.sin(), phases.cos()],
        dim=-1,
    )


def _cell_basis(
    cell_indices: torch.Tensor, spec: GridSpec, dtype: torch.dtype
) -> torch.Tensor:
    """Encode raster-cell centers with shared multiscale 3D Fourier features."""
    gu, gv, gt = spec.shape
    t = cell_indices % gt
    v = (cell_indices // gt) % gv
    u = cell_indices // (gv * gt)
    coordinate = torch.stack([u, v, t], dim=-1).to(dtype)
    shape = coordinate.new_tensor([gu, gv, gt])
    normalized = (coordinate + 0.5) * (2 / shape) - 1
    frequencies = normalized.new_tensor([1.0, 2.0, 4.0, 8.0])
    phases = normalized[..., None] * (math.pi * frequencies)
    return torch.cat(
        [normalized, phases.sin().flatten(-2), phases.cos().flatten(-2)], dim=-1
    )


class RankConditionedEncoder(nn.Module):
    """Pool nonlinear `(feature, canonical-rank)` bindings into raster tokens."""

    def __init__(
        self,
        feature_dim: int,
        model_dim: int,
        latent_dim: int,
        spec: GridSpec,
        depth: int,
        heads: int,
        position_mode: str = "learned",
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("encoder depth cannot be negative")
        if position_mode not in {"learned", "fourier"}:
            raise ValueError(f"unknown position mode: {position_mode}")
        self.spec = spec
        self.grid = OccupancyGrid(spec)
        self.position_mode = position_mode
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim + 8, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.stats_proj = nn.Linear(model_dim * 2 + 2, model_dim)
        if position_mode == "learned":
            self.cell_embed = nn.Parameter(torch.randn(spec.n_cells, model_dim) * 0.02)
            self.cell_position_proj = None
        else:
            self.register_parameter("cell_embed", None)
            self.cell_position_proj = nn.Linear(27, model_dim)
        if depth:
            layer = nn.TransformerEncoderLayer(
                model_dim,
                heads,
                model_dim * 4,
                batch_first=True,
                norm_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(layer, depth)
        else:
            self.blocks = nn.Identity()
        self.out = nn.Linear(model_dim, latent_dim)

    def _position(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.cell_embed is not None:
            return self.cell_embed
        indices = torch.arange(self.spec.n_cells, device=device)
        return self.cell_position_proj(_cell_basis(indices, self.spec, dtype))

    def forward(
        self, features: torch.Tensor, target: CompactGrid | None = None
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("encoder expects (B,N,F)")
        compact = self.grid.pack_compact(features) if target is None else target
        rank = _rank_basis(
            compact.slot_indices, self.spec.slots_per_cell, features.dtype
        )
        projected = self.in_proj(torch.cat([compact.values, rank], dim=-1)).float()
        batch = features.shape[0]
        total = projected.new_zeros(batch, self.spec.n_cells, projected.shape[-1])
        square = torch.zeros_like(total)
        indices = compact.cell_indices[..., None].expand_as(projected)
        total.scatter_add_(1, indices, projected)
        square.scatter_add_(1, indices, projected.square())
        count = compact.counts.to(projected.dtype)[..., None]
        mean = total / count.clamp_min(1)
        variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
        occupied = (count > 0).to(projected.dtype)
        log_count = torch.log1p(count) / math.log(self.spec.slots_per_cell + 1)
        cells = self.stats_proj(
            torch.cat([mean, variance, log_count, occupied], dim=-1)
        ) + self._position(projected.device, projected.dtype)[None]
        return self.out(self.blocks(cells))


class SparseSlotDecoder(nn.Module):
    """Evaluate only requested canonical cell/rank pairs, never padded slots."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 128,
        spec: GridSpec = GridSpec((12, 12, 6), 512),
        depth: int = 4,
        chunk_size: int = 32_768,
        position_mode: str = "learned",
    ) -> None:
        super().__init__()
        if position_mode not in {"learned", "fourier"}:
            raise ValueError(f"unknown position mode: {position_mode}")
        self.spec = spec
        self.chunk_size = chunk_size
        self.position_mode = position_mode
        if position_mode == "learned":
            self.cell_embed = nn.Parameter(torch.randn(spec.n_cells, model_dim) * 0.02)
            self.cell_position_proj = None
        else:
            self.register_parameter("cell_embed", None)
            self.cell_position_proj = nn.Linear(27, model_dim)
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        self.slot_proj = nn.Linear(8, model_dim)
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
        self.count_head = nn.Linear(latent_dim, 1)

    def _slot_basis(self, slot_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return _rank_basis(slot_indices, self.spec.slots_per_cell, dtype)

    def _position(
        self, cell_indices: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.cell_embed is not None:
            return self.cell_embed[cell_indices]
        return self.cell_position_proj(_cell_basis(cell_indices, self.spec, dtype))

    def _decode_chunk(
        self,
        latents: torch.Tensor,
        batch_indices: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.latent_proj(latents[batch_indices, cell_indices])
            + self._position(cell_indices, latents.dtype)
            + self.slot_proj(self._slot_basis(slot_indices, latents.dtype))
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        raw = self.feature_head(hidden)
        gu, gv, gt = self.spec.shape
        t = cell_indices % gt
        v = (cell_indices // gt) % gv
        u = cell_indices // (gv * gt)
        coordinate = torch.stack([u, v, t], dim=-1).to(latents.dtype)
        cell_size = latents.new_tensor([2 / gu, 2 / gv, 2 / gt])
        center = -1 + (coordinate + 0.5) * cell_size
        constrained = center + raw[:, :3].tanh() * cell_size * 0.5
        return torch.cat([constrained, raw[:, 3:]], dim=-1)

    def decode_indices(
        self,
        latents: torch.Tensor,
        batch_indices: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a flat list of requested `(batch,cell,rank)` tuples in chunks."""
        if not (
            batch_indices.shape == cell_indices.shape == slot_indices.shape
            and batch_indices.ndim == 1
        ):
            raise ValueError("sparse decoder indices must be matching vectors")
        pieces = []
        for start in range(0, len(batch_indices), self.chunk_size):
            part = slice(start, start + self.chunk_size)
            pieces.append(
                self._decode_chunk(
                    latents,
                    batch_indices[part],
                    cell_indices[part],
                    slot_indices[part],
                )
            )
        return (
            torch.cat(pieces)
            if pieces
            else latents.new_empty((0, self.feature_head.out_features))
        )

    def forward_training(
        self,
        latents: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> SparseAutoencoderOutput:
        batch, jewels = cell_indices.shape
        batch_indices = torch.arange(batch, device=latents.device)[:, None].expand(
            batch, jewels
        )
        occupied = self.decode_indices(
            latents,
            batch_indices.reshape(-1),
            cell_indices.reshape(-1),
            slot_indices.reshape(-1),
        ).reshape(batch, jewels, -1)
        return SparseAutoencoderOutput(
            occupied_features=occupied,
            log_count=self.count_head(latents).squeeze(-1),
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> list[torch.Tensor]:
        """Predict counts, then materialize only those canonical ranks."""
        log_count = self.count_head(latents).squeeze(-1)
        maximum = log_count.new_tensor(float(self.spec.slots_per_cell + 1)).log()
        counts = log_count.clamp(0, maximum).expm1().round().long()
        outputs = []
        for batch_index in range(latents.shape[0]):
            cells = torch.repeat_interleave(
                torch.arange(self.spec.n_cells, device=latents.device),
                counts[batch_index],
            )
            if not len(cells):
                outputs.append(latents.new_empty((0, self.feature_head.out_features)))
                continue
            offsets = counts[batch_index].cumsum(0) - counts[batch_index]
            slots = torch.arange(len(cells), device=latents.device) - torch.repeat_interleave(
                offsets, counts[batch_index]
            )
            batches = torch.full_like(cells, batch_index)
            outputs.append(self.decode_indices(latents, batches, cells, slots))
        return outputs


class SparseJewelAutoencoder(nn.Module):
    """Occupancy-aware raster encoder plus variable-count sparse slot decoder."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 128,
        spec: GridSpec = GridSpec((12, 12, 6), 512),
        enc_depth: int = 3,
        dec_depth: int = 4,
        heads: int = 8,
        decode_chunk_size: int = 32_768,
        encoder_mode: str = "pooled",
        position_mode: str = "learned",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.grid = OccupancyGrid(spec)
        if encoder_mode == "pooled":
            if position_mode != "learned":
                raise ValueError("fourier positions require encoder_mode='rank'")
            self.encoder = OccupancyAwareEncoder(
                feature_dim, model_dim, latent_dim, spec, enc_depth, heads
            )
        elif encoder_mode == "rank":
            self.encoder = RankConditionedEncoder(
                feature_dim,
                model_dim,
                latent_dim,
                spec,
                enc_depth,
                heads,
                position_mode,
            )
        else:
            raise ValueError(f"unknown encoder mode: {encoder_mode}")
        self.encoder_mode = encoder_mode
        self.position_mode = position_mode
        self.decoder = SparseSlotDecoder(
            feature_dim,
            model_dim,
            latent_dim,
            spec,
            dec_depth,
            decode_chunk_size,
            position_mode,
        )

    def forward_compact(
        self, features: torch.Tensor, target: CompactGrid
    ) -> SparseAutoencoderOutput:
        latents = (
            self.encoder(features, target)
            if self.encoder_mode == "rank"
            else self.encoder(features)
        )
        return self.decoder.forward_training(
            latents, target.cell_indices, target.slot_indices
        )

    def structural_loss(
        self,
        output: SparseAutoencoderOutput,
        target: CompactGrid,
        *,
        count_weight: float = 0.25,
        balance_count: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature_error = F.smooth_l1_loss(output.occupied_features, target.values)
        target_log_count = torch.log1p(target.counts.to(output.log_count.dtype))
        count_errors = F.smooth_l1_loss(
            output.log_count, target_log_count, reduction="none"
        )
        if balance_count:
            occupied = target.counts > 0
            groups = [
                count_errors[mask].mean()
                for mask in (occupied, ~occupied)
                if mask.any()
            ]
            count_error = torch.stack(groups).mean()
        else:
            count_error = count_errors.mean()
        total = feature_error + count_weight * count_error
        return total, {
            "feature": feature_error.detach(),
            "count": count_error.detach(),
        }

    def loss_from_compact(
        self, features: torch.Tensor, target: CompactGrid, count_weight: float = 0.25
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.structural_loss(
            self.forward_compact(features, target), target, count_weight=count_weight
        )

    def loss(
        self, features: torch.Tensor, count_weight: float = 0.25
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_from_compact(
            features, self.grid.pack_compact(features), count_weight=count_weight
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> list[torch.Tensor]:
        return self.decoder.decode(latents)
