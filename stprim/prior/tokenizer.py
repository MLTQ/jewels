"""Jewel tokenizer: set of N primitives <-> ~256 latent tokens, linear in N.

The O(N^2) wall: SetDiT's self-attention is quadratic in set size, fine at
6.5k jewels, impossible at 45k (dense spec). This module is the wall-breaker,
in the L3DG/GaussianCube tradition (primitive diffusion works best in a
structured/latent space) adapted to sets:

  encoder  GridPoolEncoder: bucket jewels into a coarse (u,v,t) grid, pool
           per cell, small transformer over CELLS (fixed count) -> M latents.
           Linear in N; permutation-invariant because bucketing is order-free.
           (The grid embedding is structure over SPACE, not token order — it
           does not reintroduce the ordering gauge.)
  decoder  LatentSetDecoder: conditional flow-matching velocity field where
           each output jewel cross-attends to the latents and NEVER to other
           jewels. Linear in N. Inter-jewel coordination must flow through
           the latents — that is the compression bet being tested.

Reconstruction loss = conditional flow matching, which is already
permutation-invariant — no Chamfer, no Hungarian.

Downstream: the stage-2 prior then models the M latent tokens (cheap
attention), and the decoder expands latents -> jewels at any N. The VQ
variant of the latent space is the "vocabulary of anisotropies" and the
codec entropy model; it lands after continuous v0 proves reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prior.model import timestep_embedding


class GridPoolEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 22,
        dim: int = 256,
        latent_dim: int = 32,
        grid: tuple[int, int, int] = (8, 8, 4),
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.n_cells = grid[0] * grid[1] * grid[2]
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.cell_embed = nn.Parameter(torch.randn(self.n_cells, dim) * 0.02)
        # Features arrive standardized; bucketing needs physical (u,v,t).
        # Set from the corpus stats by the train script.
        self.register_buffer("mu_mean", torch.zeros(3))
        self.register_buffer("mu_std", torch.ones(3))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, norm_first=True,
            dropout=0.0, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(dim, latent_dim)

    def cell_index(self, mu: torch.Tensor) -> torch.Tensor:
        mu = mu * self.mu_std + self.mu_mean  # back to physical (u,v,t)
        gu, gv, gt = self.grid
        u = ((mu[..., 0].clamp(-1, 1) + 1) / 2 * (gu - 1e-4)).long()
        v = ((mu[..., 1].clamp(-1, 1) + 1) / 2 * (gv - 1e-4)).long()
        t = ((mu[..., 2].clamp(-1, 1) + 1) / 2 * (gt - 1e-4)).long()
        return (u * gv + v) * gt + t  # (B, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, feat) -> (B, n_cells, latent_dim)."""
        b, n, _ = x.shape
        h = self.in_proj(x)
        idx = self.cell_index(x[..., :3])  # mu is dims 0:3 by feature layout
        cells = h.new_zeros(b, self.n_cells, h.shape[-1])
        count = h.new_zeros(b, self.n_cells, 1)
        cells.scatter_add_(1, idx[..., None].expand_as(h), h)
        count.scatter_add_(1, idx[..., None], torch.ones_like(idx, dtype=h.dtype)[..., None])
        cells = cells / count.clamp_min(1.0)
        cells = cells + self.cell_embed[None]
        return self.out(self.blocks(cells))


class DecoderBlock(nn.Module):
    """Cross-attention to latents + MLP, adaLN-modulated by flow time."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ada = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, lat_kv, cond):
        b, n, d = x.shape
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(cond))[:, None].chunk(6, -1)
        h = self.norm1(x) * (1 + s1) + b1
        q = self.q(h).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = lat_kv.chunk(2, -1)
        k = k.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        x = x + g1 * self.proj(h.transpose(1, 2).reshape(b, n, d))
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class LatentSetDecoder(nn.Module):
    """Velocity field v(x_t, t | latents); NO jewel-to-jewel attention."""

    def __init__(
        self,
        feat_dim: int = 22,
        dim: int = 256,
        latent_dim: int = 32,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, dim)
        self.lat_proj = nn.Linear(latent_dim, dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(DecoderBlock(dim, heads) for _ in range(depth))
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, feat_dim)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.dim = dim

    def forward(self, x, t, latents):
        cond = self.t_mlp(timestep_embedding(t, self.dim))
        lat_kv = self.kv_proj(self.lat_proj(latents))
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, lat_kv, cond)
        s, b = self.ada_out(F.silu(cond))[:, None].chunk(2, -1)
        return self.out(self.norm_out(h) * (1 + s) + b)


class JewelTokenizer(nn.Module):
    def __init__(self, **kw) -> None:
        super().__init__()
        enc_kw = {k: v for k, v in kw.items() if k in
                  ("feat_dim", "dim", "latent_dim", "grid", "heads")}
        dec_kw = {k: v for k, v in kw.items() if k in
                  ("feat_dim", "dim", "latent_dim", "heads")}
        self.encoder = GridPoolEncoder(
            depth=kw.get("enc_depth", 4), **enc_kw)
        self.decoder = LatentSetDecoder(
            depth=kw.get("dec_depth", 6), **dec_kw)

    def flow_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """End-to-end reconstruction objective (conditional flow matching)."""
        z = self.encoder(x1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = self.decoder(xt, t, z)
        return F.mse_loss(v, x1 - x0)

    @torch.no_grad()
    def reconstruct(
        self, x1: torch.Tensor, *, steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode then sample the decoder: the round-trip through the latents."""
        z = self.encoder(x1)
        x = torch.randn(x1.shape, generator=generator, device=x1.device,
                        dtype=x1.dtype)
        ts = torch.linspace(0, 1, steps + 1, device=x1.device)
        for i in range(steps):
            t = ts[i].expand(x1.shape[0])
            x = x + (ts[i + 1] - ts[i]) * self.decoder(x, t, z)
        return x
