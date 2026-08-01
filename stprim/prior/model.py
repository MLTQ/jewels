"""Set-flow transformer: the jewel emitter.

A DiT-style transformer velocity field for conditional flow matching over
primitive SETS. Deliberately NO positional encodings anywhere: with tokens
unordered, the network is permutation-equivariant and the learned
distribution is permutation-invariant — exactly what the canonicalization
experiments said a prior over fitted sets must be (individual primitives
don't correspond across fits; the distribution does).

Conditioning enters through adaLN (DiT-style): timestep embedding plus a
CLIP embedding projected to model width. A learned null embedding replaces
the CLIP vector under conditioning dropout, which is what makes
classifier-free guidance — and therefore t2v prompting — a sampling-time
option rather than a retrain.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) in [0,1] -> (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    """Pre-norm attention + MLP with adaLN-Zero modulation."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        # adaLN-Zero: modulation starts at identity, gates start closed.
        self.ada = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(cond))[:, None].chunk(6, -1)

        h = self.norm1(x) * (1 + s1) + b1
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(b, n, d)
        x = x + g1 * self.proj(h)

        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class SetDiT(nn.Module):
    def __init__(
        self,
        feat_dim: int = 22,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        cond_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.c_proj = nn.Linear(cond_dim, dim)
        self.null_cond = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, feat_dim)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,          # (B, N, feat_dim) noisy sets
        t: torch.Tensor,          # (B,) flow time in [0, 1]
        c: torch.Tensor | None,   # (B, cond_dim) CLIP embeddings or None
        drop: torch.Tensor | None = None,  # (B,) bool: use null cond
    ) -> torch.Tensor:
        cond = self.t_mlp(timestep_embedding(t, self.dim))
        if c is None:
            cond = cond + self.null_cond[None]
        else:
            ce = self.c_proj(c)
            if drop is not None:
                ce = torch.where(drop[:, None], self.null_cond[None], ce)
            cond = cond + ce

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, cond)
        s, b = self.ada_out(F.silu(cond))[:, None].chunk(2, -1)
        return self.out_proj(self.norm_out(h) * (1 + s) + b)
