"""Feed-forward video-to-jewel-field encoder: amortized fitting, not set sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.token_grid import GridSpec
from stprim.models.tiled_support import (
    build_support_tile_index,
    query_support_pairs,
)

_TRIU = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def cholesky_render(
    centers: torch.Tensor,
    cholesky: torch.Tensor,
    colors: torch.Tensor,
    color_grads: torch.Tensor,
    logit_w: torch.Tensor,
    points: torch.Tensor,
    background: torch.Tensor,
    *,
    point_chunk: int = 1024,
    cull_mode: str = "exact",
    support_sigma: float = 5.0,
    support_capacity: int = 1024,
    support_base_resolution: int = 32,
    support_level_scale: float = 1.55,
) -> torch.Tensor:
    """Additively render jewels parameterized by a precision Cholesky factor.

    Mirrors `sol.render.render_exact` (same logits, P1 color, background mix)
    but keeps `torch.linalg.eigh` out of the training graph: the Mahalanobis
    term is ||L^T d||^2 with L the lower-triangular precision factor.
    """
    def _exact_block(block: torch.Tensor) -> torch.Tensor:
        delta = block[:, None, :] - centers[None, :, :]
        projected = torch.einsum("nij,mnj->mni", cholesky.transpose(1, 2), delta)
        mahalanobis = projected.square().sum(-1)
        logits = -0.5 * mahalanobis + F.logsigmoid(logit_w)[None]
        color = colors[None] + torch.einsum("nij,mnj->mni", color_grads, delta)
        alpha = logits.exp()
        return (alpha[..., None] * color).sum(dim=1) + background[None]

    def _support_block(
        block: torch.Tensor,
        owners: torch.Tensor,
        primitive_indices: torch.Tensor,
    ) -> torch.Tensor:
        if owners.numel() == 0:
            return background[None].expand(len(block), -1).clone()
        delta = block[owners] - centers[primitive_indices]
        projected = torch.einsum(
            "pji,pj->pi", cholesky[primitive_indices], delta
        )
        mahalanobis = projected.square().sum(-1)
        logits = -0.5 * mahalanobis + F.logsigmoid(
            logit_w[primitive_indices]
        )
        logits = logits.masked_fill(
            mahalanobis > support_sigma * support_sigma, -torch.inf
        )
        color = colors[primitive_indices] + torch.einsum(
            "pij,pj->pi", color_grads[primitive_indices], delta
        )
        contribution = logits.exp()[:, None] * color
        return torch.zeros(
            len(block), 3, dtype=contribution.dtype, device=block.device
        ).index_add(0, owners, contribution) + background[None]

    if point_chunk <= 0:
        raise ValueError("point_chunk must be positive")
    if cull_mode not in {"exact", "support_tiled"}:
        raise ValueError(f"unknown cull_mode {cull_mode!r}")

    tile_index = None
    if cull_mode == "support_tiled":
        detached_cholesky = cholesky.detach()
        inverse = torch.linalg.inv(detached_cholesky)
        half_extent = support_sigma * torch.sqrt(
            inverse.square().sum(dim=1).clamp_min(1e-16)
        )
        tile_index = build_support_tile_index(
            centers.detach(),
            half_extent.amax(dim=1) / support_sigma,
            half_extent=half_extent,
            metric_matrix=detached_cholesky.transpose(1, 2),
            support_sigma=support_sigma,
            base_resolution=support_base_resolution,
            level_scale=support_level_scale,
        )

    outputs = []
    needs_graph = torch.is_grad_enabled() and (
        centers.requires_grad or cholesky.requires_grad or colors.requires_grad
    )
    for start in range(0, len(points), point_chunk):
        block = points[start : start + point_chunk]
        if cull_mode == "support_tiled":
            owners, primitive_indices = query_support_pairs(
                tile_index, block, capacity=support_capacity
            )
            outputs.append(_support_block(block, owners, primitive_indices))
        elif needs_graph:
            outputs.append(
                torch.utils.checkpoint.checkpoint(
                    _exact_block, block, use_reentrant=False
                )
            )
        else:
            outputs.append(_exact_block(block))
    return torch.cat(outputs)


def cholesky_to_log_covariance(
    cholesky: torch.Tensor, *, chunk: int = 16384
) -> torch.Tensor:
    """Convert precision factors to canonical upper-triangular logSigma features.

    Runs on CPU in chunks: batched CUDA eigh requests a per-matrix solver
    workspace that dwarfs the actual data for large batches of 3x3 matrices.
    """
    outputs = []
    for start in range(0, len(cholesky), chunk):
        part = cholesky[start : start + chunk].double().cpu()
        precision = part @ part.transpose(1, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(precision)
        log_sigma = torch.einsum(
            "nij,nj,nkj->nik",
            eigenvectors,
            -eigenvalues.clamp_min(1e-12).log(),
            eigenvectors,
        )
        outputs.append(
            torch.stack([log_sigma[:, i, j] for i, j in _TRIU], dim=1)
        )
    return torch.cat(outputs).to(device=cholesky.device, dtype=cholesky.dtype)


class VideoToJewelEncoder(nn.Module):
    """Predict a fixed per-cell budget of jewels from one video window."""

    def __init__(
        self,
        *,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
        slots_per_cell: int = 36,
        model_dim: int = 256,
        initial_scale: float = 0.06,
    ) -> None:
        super().__init__()
        if slots_per_cell <= 0 or model_dim % 8:
            raise ValueError("slots must be positive and model_dim divisible by 8")
        self.grid_spec = grid_spec
        self.slots_per_cell = slots_per_cell
        gu, gv, gt = grid_spec.shape
        self.trunk = nn.Sequential(
            nn.Conv3d(3, 64, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, (3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, model_dim, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((gt, gv, gu)),
            nn.Conv3d(model_dim, model_dim, 3, padding=1),
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            nn.Conv3d(model_dim, model_dim, 3, padding=1),
        )
        self.head = nn.Linear(model_dim, slots_per_cell * 22)
        nn.init.zeros_(self.head.weight)
        self.background_head = nn.Linear(model_dim, 3)
        head_bias = torch.zeros(slots_per_cell, 22)
        head_bias[:, 21] = -2.7
        self.head.bias = nn.Parameter(head_bias.reshape(-1))
        coordinates = torch.stack(
            torch.meshgrid(
                (torch.arange(gu) + 0.5) / gu * 2 - 1,
                (torch.arange(gv) + 0.5) / gv * 2 - 1,
                (torch.arange(gt) + 0.5) / gt * 2 - 1,
                indexing="ij",
            ),
            dim=-1,
        ).reshape(grid_spec.n_cells, 3)
        self.register_buffer("cell_centers", coordinates)
        extents = torch.tensor([2.0 / gu, 2.0 / gv, 2.0 / gt])
        self.register_buffer("cell_extents", extents)
        side = max(1, round(slots_per_cell ** (1 / 3)))
        self.register_buffer(
            "log_precision_init", torch.log(2.0 * side / extents)
        )
        self.initial_scale = initial_scale

    def slot_lattice(self) -> torch.Tensor:
        """Deterministic stratified slot positions inside every cell."""
        slots = self.slots_per_cell
        side = max(1, round(slots ** (1 / 3)))
        offsets = []
        for index in range(slots):
            u = (index % side + 0.5) / side
            v = ((index // side) % side + 0.5) / side
            t = ((index // (side * side)) % side + 0.5) / side
            offsets.append((u, v, t))
        lattice = torch.tensor(offsets, dtype=torch.float32) * 2 - 1
        return (
            self.cell_centers[:, None]
            + 0.5 * self.cell_extents * lattice[None].to(self.cell_centers)
        )

    @staticmethod
    def sample_video_colors(
        video: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Trilinearly sample the window's RGB at normalized (u,v,t) positions."""
        volume = video.permute(3, 0, 1, 2)[None]
        query = positions.reshape(1, -1, 1, 1, 3)
        sampled = F.grid_sample(volume, query, align_corners=True)
        return sampled[0, :, :, 0, 0].transpose(0, 1).reshape(*positions.shape)

    def encode(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map a window to the generatable latent: cell features plus slot seeds.

        Both parts are video-derived, so a text-conditioned generator must emit
        both to synthesize without a video. The seed tensor is effectively a
        coarse RGB volume sampled on the slot lattice.
        """
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must have shape (T,H,W,3)")
        volume = video.permute(3, 0, 1, 2)[None]
        hidden = self.trunk(volume)[0]
        cells = hidden.permute(3, 2, 1, 0).reshape(self.grid_spec.n_cells, -1)
        seed = self.sample_video_colors(video, self.slot_lattice()).clamp(
            1e-3, 1 - 1e-3
        )
        return {"cells": cells, "seed": seed}

    def decode(self, latent: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Turn a latent into jewel parameters without touching a video."""
        cells, seeded_colors = latent["cells"], latent["seed"]
        if cells.shape[0] != self.grid_spec.n_cells:
            raise ValueError("latent cells do not match the encoder grid")
        if seeded_colors.shape != (
            self.grid_spec.n_cells,
            self.slots_per_cell,
            3,
        ):
            raise ValueError("latent seed colors do not match the slot lattice")
        raw = self.head(cells).reshape(
            self.grid_spec.n_cells, self.slots_per_cell, 22
        )
        lattice = self.slot_lattice()
        seeded_colors = seeded_colors.clamp(1e-3, 1 - 1e-3)
        centers = lattice + 0.75 * self.cell_extents * torch.tanh(raw[..., 0:3])
        log_diagonal = (raw[..., 3:6] + self.log_precision_init).clamp(-1.0, 7.0)
        off_diagonal = 0.2 * raw[..., 6:9]
        colors = torch.sigmoid(
            raw[..., 9:12] + torch.logit(seeded_colors)
        )
        color_grads = 0.2 * raw[..., 12:21]
        logit_w = raw[..., 21].clamp(-9.0, 6.0)
        flat = lambda value: value.reshape(-1, *value.shape[2:])  # noqa: E731
        cholesky = torch.zeros(
            self.grid_spec.n_cells * self.slots_per_cell,
            3,
            3,
            device=cells.device,
            dtype=cells.dtype,
        )
        diagonal = flat(log_diagonal).exp()
        offdiag = flat(off_diagonal)
        cholesky[:, 0, 0] = diagonal[:, 0]
        cholesky[:, 1, 1] = diagonal[:, 1]
        cholesky[:, 2, 2] = diagonal[:, 2]
        cholesky[:, 1, 0] = offdiag[:, 0] * diagonal[:, 0]
        cholesky[:, 2, 0] = offdiag[:, 1] * diagonal[:, 0]
        cholesky[:, 2, 1] = offdiag[:, 2] * diagonal[:, 1]
        background = torch.sigmoid(self.background_head(cells.mean(dim=0)))
        return {
            "centers": flat(centers),
            "cholesky": cholesky,
            "colors": flat(colors),
            "color_grads": flat(color_grads).reshape(-1, 3, 3),
            "logit_w": flat(logit_w).reshape(-1),
            "background": background,
        }

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode one (T,H,W,3) window into per-slot jewel parameters."""
        return self.decode(self.encode(video))

    def canonical_features(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the canonical 22-D feature matrix for saving and exact audits."""
        with torch.no_grad():
            log_sigma = cholesky_to_log_covariance(prediction["cholesky"])
            return torch.cat(
                (
                    prediction["centers"],
                    log_sigma,
                    prediction["colors"],
                    prediction["color_grads"].reshape(-1, 9),
                    prediction["logit_w"][:, None],
                ),
                dim=1,
            )
