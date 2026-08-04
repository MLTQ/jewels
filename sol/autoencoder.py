"""Count-aware encoder and deterministic slot decoder for structured jewels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.token_grid import CompactGrid, GridSpec, OccupancyGrid


class OccupancyAwareEncoder(nn.Module):
    """Permutation-invariant grid encoder retaining count and second moments."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 64,
        spec: GridSpec = GridSpec(),
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.stats_proj = nn.Linear(model_dim * 2 + 2, model_dim)
        self.cell_embed = nn.Parameter(torch.randn(spec.n_cells, model_dim) * 0.02)
        if depth < 0:
            raise ValueError("encoder depth cannot be negative")
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("encoder expects (B,N,F)")
        batch = features.shape[0]
        indices = self.spec.cell_index(features[..., :3])
        projected = self.in_proj(features)
        statistics = projected.float()
        total = statistics.new_zeros(batch, self.spec.n_cells, statistics.shape[-1])
        square = torch.zeros_like(total)
        count = statistics.new_zeros(batch, self.spec.n_cells, 1)
        total.scatter_add_(1, indices[..., None].expand_as(statistics), statistics)
        square.scatter_add_(
            1, indices[..., None].expand_as(statistics), statistics.square()
        )
        count.scatter_add_(
            1,
            indices[..., None],
            torch.ones_like(indices, dtype=statistics.dtype)[..., None],
        )
        mean = total / count.clamp_min(1)
        variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
        occupied = (count > 0).to(statistics.dtype)
        log_count = torch.log1p(count) / torch.log(
            statistics.new_tensor(float(self.spec.slots_per_cell + 1))
        )
        stats = torch.cat([mean, variance, log_count, occupied], dim=-1)
        cells = self.stats_proj(stats) + self.cell_embed[None]
        return self.out(self.blocks(cells))


@dataclass
class AutoencoderOutput:
    features: torch.Tensor
    existence_logits: torch.Tensor
    log_count: torch.Tensor


class StructuredSlotDecoder(nn.Module):
    """Deterministically expands each raster latent into canonical jewel slots."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 64,
        spec: GridSpec = GridSpec(),
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.slot_embed = nn.Parameter(
            torch.randn(spec.slots_per_cell, model_dim) * 0.02
        )
        self.cell_embed = nn.Parameter(torch.randn(spec.n_cells, model_dim) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, model_dim)
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
        self.existence_head = nn.Linear(model_dim, 1)
        self.count_head = nn.Linear(latent_dim, 1)

    def forward(self, latents: torch.Tensor) -> AutoencoderOutput:
        batch, cells, _ = latents.shape
        if cells != self.spec.n_cells:
            raise ValueError(f"expected {self.spec.n_cells} cells, got {cells}")
        hidden = (
            self.slot_embed[None, None]
            + self.cell_embed[None, :, None]
            + self.latent_proj(latents)[:, :, None]
        )
        for block in self.blocks:
            hidden = hidden + block(hidden)
        raw_features = self.feature_head(hidden)
        gu, gv, gt = self.spec.shape
        ids = torch.arange(cells, device=latents.device)
        t = ids % gt
        v = (ids // gt) % gv
        u = ids // (gv * gt)
        coordinate = torch.stack([u, v, t], dim=-1).to(latents.dtype)
        cell_size = latents.new_tensor([2 / gu, 2 / gv, 2 / gt])
        cell_center = -1 + (coordinate + 0.5) * cell_size
        constrained_center = (
            cell_center[None, :, None]
            + raw_features[..., :3].tanh() * cell_size[None, None, None] * 0.5
        )
        decoded_features = torch.cat(
            [constrained_center, raw_features[..., 3:]], dim=-1
        )
        return AutoencoderOutput(
            features=decoded_features,
            existence_logits=self.existence_head(hidden).squeeze(-1),
            log_count=self.count_head(latents).squeeze(-1),
        )


class StructuredJewelAutoencoder(nn.Module):
    """Count-aware encoder plus deterministic raster-cell slot decoder."""

    def __init__(
        self,
        feature_dim: int = 22,
        model_dim: int = 256,
        latent_dim: int = 64,
        spec: GridSpec = GridSpec(),
        enc_depth: int = 4,
        dec_depth: int = 3,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.grid = OccupancyGrid(spec)
        self.encoder = OccupancyAwareEncoder(
            feature_dim, model_dim, latent_dim, spec, enc_depth, heads
        )
        self.decoder = StructuredSlotDecoder(
            feature_dim, model_dim, latent_dim, spec, dec_depth
        )

    def forward(self, features: torch.Tensor) -> AutoencoderOutput:
        return self.decoder(self.encoder(features))

    def loss(
        self,
        features: torch.Tensor,
        *,
        existence_weight: float = 1.0,
        count_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.grid.pack_compact(features)
        return self.loss_from_compact(features, target, existence_weight, count_weight)

    def loss_from_compact(
        self,
        features: torch.Tensor,
        target: CompactGrid,
        existence_weight: float = 1.0,
        count_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train against precomputed occupied slots without dense target storage."""
        output = self(features)
        return self.structural_loss(
            output, target, existence_weight=existence_weight, count_weight=count_weight
        )

    def structural_loss(
        self,
        output: AutoencoderOutput,
        target: CompactGrid,
        *,
        existence_weight: float = 1.0,
        count_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score a decoded slot field while allowing external perceptual losses."""
        batch_indices = torch.arange(
            output.features.shape[0], device=output.features.device
        )[:, None]
        predicted_occupied = output.features[
            batch_indices, target.cell_indices, target.slot_indices
        ]
        feature_error = F.smooth_l1_loss(
            predicted_occupied, target.values
        )
        existence_target = torch.zeros_like(output.existence_logits)
        existence_target[
            batch_indices, target.cell_indices, target.slot_indices
        ] = 1
        positives = existence_target.sum().clamp_min(1)
        negatives = existence_target.numel() - positives
        positive_weight = (negatives / positives).clamp_min(1)
        existence_error = F.binary_cross_entropy_with_logits(
            output.existence_logits,
            existence_target,
            pos_weight=positive_weight,
        )
        target_log_count = torch.log1p(target.counts.to(output.log_count.dtype))
        count_error = F.smooth_l1_loss(output.log_count, target_log_count)
        total = feature_error + existence_weight * existence_error + count_weight * count_error
        return total, {
            "feature": feature_error.detach(),
            "existence": existence_error.detach(),
            "count": count_error.detach(),
        }

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> list[torch.Tensor]:
        """Decode variable-size sets using the count head and ranked existence logits."""
        output = self.decoder(latents)
        maximum_log_count = output.log_count.new_tensor(
            float(self.spec.slots_per_cell + 1)
        ).log()
        requested = output.log_count.clamp(0, maximum_log_count).expm1().round().long()
        decoded = []
        for batch_index in range(latents.shape[0]):
            pieces = []
            for cell in range(self.spec.n_cells):
                count = int(requested[batch_index, cell])
                if count == 0:
                    continue
                chosen = output.existence_logits[batch_index, cell].topk(count).indices
                pieces.append(output.features[batch_index, cell, chosen])
            decoded.append(
                torch.cat(pieces, dim=0)
                if pieces
                else output.features.new_empty(0, output.features.shape[-1])
            )
        return decoded
