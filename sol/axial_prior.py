"""Text-conditioned axial rectified flow for hierarchical jewel latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.latent_prior import timestep_embedding


class AxialConditionalBlock(nn.Module):
    """One axis-attention update plus adaLN-Zero conditioned pointwise MLP."""

    def __init__(self, dimension: int, heads: int, axis: int) -> None:
        super().__init__()
        if axis not in (1, 2, 3):
            raise ValueError("axis must identify u, v, or t")
        self.axis = axis
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

    def _attention(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, gu, gv, gt, dimension = hidden.shape
        permutation = [0] + [axis for axis in (1, 2, 3) if axis != self.axis] + [
            self.axis,
            4,
        ]
        sequences = hidden.permute(permutation).reshape(-1, hidden.shape[self.axis], dimension)
        count = sequences.shape[1]
        qkv = self.qkv(sequences).reshape(
            sequences.shape[0], count, 3, self.heads, dimension // self.heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape_as(sequences)
        attended = self.attention_out(attended).reshape(
            [batch]
            + [hidden.shape[axis] for axis in (1, 2, 3) if axis != self.axis]
            + [hidden.shape[self.axis], dimension]
        )
        inverse = [permutation.index(axis) for axis in range(5)]
        return attended.permute(inverse)

    def forward(self, cells: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        modulation = self.modulation(F.silu(condition))
        scale_a, bias_a, gate_a, scale_m, bias_m, gate_m = modulation.chunk(6, dim=-1)
        expand = (slice(None), None, None, None, slice(None))
        hidden = self.norm_attention(cells)
        hidden = hidden * (1 + scale_a[expand]) + bias_a[expand]
        cells = cells + gate_a[expand] * self._attention(hidden)
        hidden = self.norm_mlp(cells)
        hidden = hidden * (1 + scale_m[expand]) + bias_m[expand]
        return cells + gate_m[expand] * self.mlp(hidden)


class AxialFlowPrior(nn.Module):
    """Conditional velocity field with rotating u/v/t attention sweeps."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        latent_dim: int = 96,
        model_dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        text_dim: int = 512,
        mask_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if depth <= 0 or any(axis <= 0 for axis in grid_shape):
            raise ValueError("depth and grid axes must be positive")
        self.grid_shape = tuple(grid_shape)
        self.n_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.model_dim = model_dim
        self.mask_conditioning = bool(mask_conditioning)
        self.input = nn.Linear(latent_dim, model_dim)
        self.u_embedding = nn.Parameter(torch.randn(grid_shape[0], model_dim) * 0.02)
        self.v_embedding = nn.Parameter(torch.randn(grid_shape[1], model_dim) * 0.02)
        self.t_embedding = nn.Parameter(torch.randn(grid_shape[2], model_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.text_projection = nn.Linear(text_dim, model_dim)
        self.null_condition = nn.Parameter(torch.zeros(model_dim))
        if self.mask_conditioning:
            self.edit_mask_embedding = nn.Embedding(2, model_dim)
            nn.init.zeros_(self.edit_mask_embedding.weight)
        self.blocks = nn.ModuleList(
            AxialConditionalBlock(model_dim, heads, 1 + index % 3)
            for index in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.output_modulation = nn.Linear(model_dim, model_dim * 2)
        self.output = nn.Linear(model_dim, latent_dim)
        nn.init.zeros_(self.output_modulation.weight)
        nn.init.zeros_(self.output_modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        latents: torch.Tensor,
        time: torch.Tensor,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
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
        batch = latents.shape[0]
        gu, gv, gt = self.grid_shape
        hidden = self.input(latents).reshape(batch, gu, gv, gt, self.model_dim)
        hidden = (
            hidden
            + self.u_embedding[None, :, None, None]
            + self.v_embedding[None, None, :, None]
            + self.t_embedding[None, None, None, :]
        )
        if self.mask_conditioning:
            if edit_mask is None:
                edit_mask = torch.ones(
                    batch, self.n_cells, dtype=torch.bool, device=latents.device
                )
            if edit_mask.shape != (batch, self.n_cells):
                raise ValueError(f"edit_mask must have shape (B,{self.n_cells})")
            mask_ids = edit_mask.to(device=latents.device, dtype=torch.long)
            hidden = hidden + self.edit_mask_embedding(mask_ids).reshape_as(hidden)
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale, bias = self.output_modulation(F.silu(condition)).chunk(2, dim=-1)
        expand = (slice(None), None, None, None, slice(None))
        hidden = self.output_norm(hidden) * (1 + scale[expand]) + bias[expand]
        return self.output(hidden).reshape_as(latents)
