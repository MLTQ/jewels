"""Text-conditioned rectified flow over frozen-encoder jewel latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.latent_prior import timestep_embedding

ARCHITECTURE = "latent_text_prior_v1"


class LatentStandardizer:
    """Per-channel statistics so the flow operates on unit-scale targets."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != std.shape:
            raise ValueError("standardizer mean and std must share a shape")
        self.mean = mean
        self.std = std.clamp_min(1e-5)

    @classmethod
    def fit(cls, values: torch.Tensor) -> "LatentStandardizer":
        flat = values.reshape(-1, values.shape[-1]).float()
        return cls(flat.mean(0), flat.std(0))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values) + self.mean.to(values)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "LatentStandardizer":
        return cls(state["mean"], state["std"])


class _Block(nn.Module):
    """Self-attention over cells, cross-attention to text, then an MLP."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, model_dim * 3)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        gate_self, gate_cross, gate_mlp = self.modulation(conditioning).chunk(
            3, dim=-1
        )
        normed = self.norm_self(hidden)
        attended, _ = self.self_attention(normed, normed, normed, need_weights=False)
        hidden = hidden + gate_self.tanh() * attended
        normed = self.norm_cross(hidden)
        crossed, _ = self.cross_attention(
            normed, text, text, key_padding_mask=text_mask, need_weights=False
        )
        hidden = hidden + gate_cross.tanh() * crossed
        hidden = hidden + gate_mlp.tanh() * self.mlp(self.norm_mlp(hidden))
        return hidden


class LatentTextPrior(nn.Module):
    """Predict flow velocity for a cell-token latent given prompt tokens."""

    def __init__(
        self,
        *,
        n_cells: int,
        cell_dim: int,
        seed_dim: int,
        text_dim: int,
        model_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("heads must divide model_dim")
        self.n_cells = n_cells
        self.cell_dim = cell_dim
        self.seed_dim = seed_dim
        self.feature_dim = cell_dim + seed_dim
        self.input_projection = nn.Linear(self.feature_dim, model_dim)
        self.position = nn.Parameter(torch.randn(n_cells, model_dim) * 0.02)
        self.text_projection = nn.Linear(text_dim, model_dim)
        self.null_text = nn.Parameter(torch.randn(1, model_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.blocks = nn.ModuleList(_Block(model_dim, heads) for _ in range(depth))
        self.norm_out = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, self.feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        flow_time: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (self.n_cells, self.feature_dim):
            raise ValueError("noisy latent must have shape (B,n_cells,feature_dim)")
        hidden = self.input_projection(noisy) + self.position[None]
        conditioning = self.time_mlp(
            timestep_embedding(flow_time, self.position.shape[-1])
        )[:, None]
        if text_tokens is None:
            text = self.null_text[None].expand(len(noisy), -1, -1)
            mask = None
        else:
            text = self.text_projection(text_tokens)
            mask = None if text_mask is None else ~text_mask.bool()
        for block in self.blocks:
            hidden = block(hidden, text, mask, conditioning)
        return self.output_projection(self.norm_out(hidden))

    @torch.no_grad()
    def sample(
        self,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None = None,
        *,
        steps: int = 32,
        guidance: float = 1.0,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Euler-integrate the velocity field from noise to a latent."""
        batch = 1 if text_tokens is None else len(text_tokens)
        state = torch.randn(
            batch,
            self.n_cells,
            self.feature_dim,
            device=device,
            generator=generator,
        )
        times = torch.linspace(0, 1, steps + 1, device=device)
        for index in range(steps):
            time = times[index : index + 1].expand(batch)
            velocity = self(state, time, text_tokens, text_mask)
            if guidance != 1.0 and text_tokens is not None:
                unconditional = self(state, time, None, None)
                velocity = unconditional + guidance * (velocity - unconditional)
            state = state + (times[index + 1] - times[index]) * velocity
        return state
