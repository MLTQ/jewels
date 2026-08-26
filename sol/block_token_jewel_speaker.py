"""Expand local spacetime block tokens into irregular continuous Jewel casts."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from sol.block_token_language import block_centers, block_local_coordinates
from sol.prompt_jewel_caster import ACTIVE_FACTORS
from sol.token_grid import GridSpec


class BlockTokenJewelSpeaker(nn.Module):
    """Decode one shared discrete block token into local density and Jewel marks."""

    def __init__(
        self,
        *,
        block_vocabulary_size: int = 256,
        jewel_vocabulary_size: int = 1024,
        block_shape: tuple[int, int, int] = (8, 8, 4),
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if min(block_vocabulary_size, jewel_vocabulary_size, hidden_dim, depth) <= 0:
            raise ValueError("block speaker dimensions must be positive")
        self.block_vocabulary_size = block_vocabulary_size
        self.jewel_vocabulary_size = jewel_vocabulary_size
        self.block_shape = block_shape
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.spec = GridSpec(block_shape, slots_per_cell=1)
        coordinate_dim = 3 * (1 + 2 * 4)
        self.block_token_embedding = nn.Embedding(
            block_vocabulary_size, hidden_dim
        )
        self.block_address_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.token_head = nn.Linear(
            hidden_dim, len(ACTIVE_FACTORS) * jewel_vocabulary_size
        )
        self.intensity_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def coordinate_features(coordinates: torch.Tensor) -> torch.Tensor:
        output = [coordinates]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            output.extend(
                [
                    torch.sin(torch.pi * frequency * coordinates),
                    torch.cos(torch.pi * frequency * coordinates),
                ]
            )
        return torch.cat(output, dim=1)

    def program_tokens(
        self, program: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Look up the shared block token without changing continuous centers."""
        if program.shape != (self.spec.n_cells,):
            raise ValueError("block program must contain one token per routing block")
        if program.dtype != torch.long:
            raise ValueError("block program must contain integer token IDs")
        cells = self.spec.cell_index(centers)
        return program[cells]

    def hidden(
        self, block_tokens: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        if block_tokens.shape != (len(centers),):
            raise ValueError("block-token and centroid rows must align")
        if block_tokens.dtype != torch.long:
            raise ValueError("block tokens must be integer IDs")
        cells, local = block_local_coordinates(centers, self.spec)
        addresses = block_centers(self.spec, device=centers.device)[cells].to(centers)
        combined = (
            self.block_token_embedding(block_tokens)
            + self.block_address_projection(self.coordinate_features(addresses))
            + self.local_coordinate_projection(self.coordinate_features(local))
        )
        return self.trunk(self.normalization(combined))

    def token_logits(
        self, block_tokens: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        logits = self.token_head(self.hidden(block_tokens, centers))
        return logits.reshape(
            len(centers), len(ACTIVE_FACTORS), self.jewel_vocabulary_size
        )

    def intensity_logits(
        self, block_tokens: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        return self.intensity_head(
            self.hidden(block_tokens, centers)
        ).squeeze(1)

    def loss(
        self,
        positive_block_tokens: torch.Tensor,
        centers: torch.Tensor,
        jewel_tokens: torch.Tensor,
        negative_block_tokens: torch.Tensor,
        negative_centers: torch.Tensor,
        *,
        density_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_nll = F.cross_entropy(
            self.token_logits(positive_block_tokens, centers).flatten(0, 1),
            jewel_tokens.flatten(),
        )
        positive = self.intensity_logits(positive_block_tokens, centers)
        negative = self.intensity_logits(
            negative_block_tokens, negative_centers
        )
        density_nce = 0.5 * (
            F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
            + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        )
        return token_nll + density_weight * density_nce, {
            "token_nll": token_nll,
            "density_nce": density_nce,
        }

    @torch.no_grad()
    def sample_centers(
        self,
        program: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
        proposal_multiplier: int = 4,
        chunk: int = 16384,
    ) -> torch.Tensor:
        if count <= 0 or proposal_multiplier <= 1:
            raise ValueError("center sampling requires positive count and excess proposals")
        proposal_count = count * proposal_multiplier
        proposals = torch.rand(
            proposal_count, 3, generator=generator,
            device=self.block_token_embedding.weight.device,
        ) * 1.998 - 0.999
        logits = []
        for start in range(0, proposal_count, chunk):
            part = proposals[start : start + chunk]
            logits.append(
                self.intensity_logits(self.program_tokens(program, part), part)
            )
        selected = torch.multinomial(
            torch.cat(logits).softmax(dim=0), count,
            replacement=False, generator=generator,
        )
        return proposals[selected]

    @torch.no_grad()
    def sample_tokens(
        self,
        program: torch.Tensor,
        centers: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 0.9,
        top_k: int = 64,
        chunk: int = 8192,
    ) -> torch.Tensor:
        if temperature <= 0 or not 0 < top_k <= self.jewel_vocabulary_size:
            raise ValueError("block Jewel token sampling settings are invalid")
        outputs = []
        for start in range(0, len(centers), chunk):
            part = centers[start : start + chunk]
            logits = self.token_logits(
                self.program_tokens(program, part), part
            ) / temperature
            values, indices = torch.topk(logits, top_k, dim=2)
            sampled = torch.multinomial(
                values.softmax(dim=2).reshape(-1, top_k),
                1, generator=generator,
            ).reshape(len(part), len(ACTIVE_FACTORS))
            outputs.append(indices.gather(2, sampled[:, :, None]).squeeze(2))
        return torch.cat(outputs)

    def architecture(self) -> dict:
        return {
            "block_vocabulary_size": self.block_vocabulary_size,
            "jewel_vocabulary_size": self.jewel_vocabulary_size,
            "block_shape": self.block_shape,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "density": "block_token_plus_continuous_local_fourier_intensity_nce",
            "conditioning": "one_discrete_token_per_local_spacetime_block",
        }
