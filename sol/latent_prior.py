"""Text-conditioned rectified-flow prior over raster-ordered jewel latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    """Encode normalized flow time as sinusoidal features."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = time[:, None].float() * 1000.0 * frequencies[None]
    embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class ConditionalBlock(nn.Module):
    """Raster self-attention with adaLN-Zero text/time modulation."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dimension, elementwise_affine=False)
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_out = nn.Linear(dimension, dimension)
        self.norm_mlp = nn.LayerNorm(dimension, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
        )
        self.modulation = nn.Linear(dimension, dimension * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, cells: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, count, dimension = cells.shape
        scale_a, bias_a, gate_a, scale_m, bias_m, gate_m = self.modulation(
            F.silu(condition)
        )[:, None].chunk(6, dim=-1)
        hidden = self.norm_attention(cells) * (1 + scale_a) + bias_a
        qkv = self.qkv(hidden).reshape(
            batch, count, 3, self.heads, dimension // self.heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden.transpose(1, 2).reshape(batch, count, dimension)
        cells = cells + gate_a * self.attention_out(hidden)
        hidden = self.norm_mlp(cells) * (1 + scale_m) + bias_m
        return cells + gate_m * self.mlp(hidden)


class RasterFlowPrior(nn.Module):
    """Conditional velocity field over canonical raster-cell latents."""

    def __init__(
        self,
        n_cells: int,
        latent_dim: int = 64,
        model_dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        text_dim: int = 512,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.input = nn.Linear(latent_dim, model_dim)
        self.cell_embedding = nn.Parameter(torch.randn(n_cells, model_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.text_projection = nn.Linear(text_dim, model_dim)
        self.null_condition = nn.Parameter(torch.zeros(model_dim))
        self.blocks = nn.ModuleList(
            ConditionalBlock(model_dim, heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.output_modulation = nn.Linear(model_dim, model_dim * 2)
        self.output = nn.Linear(model_dim, latent_dim)
        nn.init.zeros_(self.output_modulation.weight)
        nn.init.zeros_(self.output_modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.n_cells = n_cells
        self.model_dim = model_dim

    def forward(
        self,
        latents: torch.Tensor,
        time: torch.Tensor,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1] != self.n_cells:
            raise ValueError(f"latents must have shape (B,{self.n_cells},D)")
        condition = self.time_mlp(timestep_embedding(time, self.model_dim))
        if text_condition is None:
            condition = condition + self.null_condition[None]
        else:
            text = self.text_projection(text_condition)
            if drop_condition is not None:
                text = torch.where(
                    drop_condition[:, None], self.null_condition[None], text
                )
            condition = condition + text
        hidden = self.input(latents) + self.cell_embedding[None]
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale, bias = self.output_modulation(F.silu(condition))[:, None].chunk(2, -1)
        return self.output(self.output_norm(hidden) * (1 + scale) + bias)


def flow_matching_loss(
    model: RasterFlowPrior,
    target_latents: torch.Tensor,
    text_condition: torch.Tensor,
    *,
    condition_dropout: float = 0.1,
) -> torch.Tensor:
    """Rectified-flow loss from Gaussian noise to encoded jewel latents."""
    if not 0 <= condition_dropout <= 1:
        raise ValueError("condition_dropout must be in [0,1]")
    noise = torch.randn_like(target_latents)
    time = torch.rand(target_latents.shape[0], device=target_latents.device)
    drop = torch.rand(target_latents.shape[0], device=target_latents.device) < condition_dropout
    return flow_matching_objective(
        model,
        target_latents,
        text_condition,
        noise=noise,
        time=time,
        drop_condition=drop,
    )


def flow_matching_objective(
    model: RasterFlowPrior,
    target_latents: torch.Tensor,
    text_condition: torch.Tensor,
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
    drop_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score explicit flow paths so held-out protocols can remain byte-stable."""
    if noise.shape != target_latents.shape:
        raise ValueError("noise must match target latents")
    if time.shape != (target_latents.shape[0],):
        raise ValueError("time must have one value per sample")
    noised = (1 - time[:, None, None]) * noise + time[:, None, None] * target_latents
    target_velocity = target_latents - noise
    predicted = model(noised, time, text_condition, drop_condition)
    return F.mse_loss(predicted.float(), target_velocity.float())


def masked_flow_matching_loss(
    model: RasterFlowPrior,
    target_latents: torch.Tensor,
    text_condition: torch.Tensor,
    dirty_mask: torch.Tensor,
    *,
    condition_dropout: float = 0.1,
) -> torch.Tensor:
    """Train noise-to-target flow only where an editor requested repair."""
    if not 0 <= condition_dropout <= 1:
        raise ValueError("condition_dropout must be in [0,1]")
    noise = torch.randn_like(target_latents)
    time = torch.rand(target_latents.shape[0], device=target_latents.device)
    drop = torch.rand(target_latents.shape[0], device=target_latents.device) < condition_dropout
    return masked_flow_matching_objective(
        model,
        target_latents,
        text_condition,
        dirty_mask,
        noise=noise,
        time=time,
        drop_condition=drop,
    )


def masked_flow_matching_objective(
    model: RasterFlowPrior,
    target_latents: torch.Tensor,
    text_condition: torch.Tensor,
    dirty_mask: torch.Tensor,
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
    drop_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score fixed masked paths with clean context held at target for every time."""
    if noise.shape != target_latents.shape:
        raise ValueError("noise must match target latents")
    if time.shape != (target_latents.shape[0],):
        raise ValueError("time must have one value per sample")
    if dirty_mask.ndim == 1:
        dirty_mask = dirty_mask[None].expand(target_latents.shape[0], -1)
    if dirty_mask.shape != target_latents.shape[:2]:
        raise ValueError("dirty_mask must have shape (C,) or (B,C)")
    dirty = dirty_mask.to(device=target_latents.device, dtype=torch.bool)[..., None]
    if not bool(dirty.any()):
        raise ValueError("masked flow requires at least one dirty cell")
    interpolated = (
        (1 - time[:, None, None]) * noise
        + time[:, None, None] * target_latents
    )
    path = torch.where(dirty, interpolated, target_latents)
    target_velocity = target_latents - noise
    if getattr(model, "mask_conditioning", False):
        predicted = model(
            path,
            time,
            text_condition,
            drop_condition,
            edit_mask=dirty_mask.to(device=target_latents.device, dtype=torch.bool),
        )
    else:
        predicted = model(path, time, text_condition, drop_condition)
    squared_error = (predicted.float() - target_velocity.float()).square()
    return squared_error[dirty.expand_as(squared_error)].mean()


@torch.no_grad()
def sample_flow(
    model: RasterFlowPrior,
    condition: torch.Tensor | None,
    *,
    batch: int,
    n_cells: int,
    latent_dim: int,
    device: torch.device | str,
    steps: int = 50,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-sample normalized raster latents with optional classifier-free guidance."""
    if batch <= 0 or steps <= 0:
        raise ValueError("batch and steps must be positive")
    if condition is not None and condition.shape[0] != batch:
        raise ValueError("condition batch must match requested samples")
    target_device = torch.device(device)
    state = torch.randn(
        batch,
        n_cells,
        latent_dim,
        device=target_device,
        generator=generator,
    )
    times = torch.linspace(0, 1, steps + 1, device=target_device)
    was_training = model.training
    model.eval()
    for index in range(steps):
        time = times[index].expand(batch)
        if condition is not None and cfg_scale != 1.0:
            conditioned = model(state, time, condition)
            unconditioned = model(state, time, None)
            velocity = unconditioned + cfg_scale * (conditioned - unconditioned)
        else:
            velocity = model(state, time, condition)
        state = state + (times[index + 1] - times[index]) * velocity
    if was_training:
        model.train()
    return state
