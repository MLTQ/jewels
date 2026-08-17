"""Feed-forward video-to-jewel-field encoder: amortized fitting, not set sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.token_grid import GridSpec

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
) -> torch.Tensor:
    """Additively render jewels parameterized by a precision Cholesky factor.

    Mirrors `sol.render.render_exact` (same logits, P1 color, background mix)
    but keeps `torch.linalg.eigh` out of the training graph: the Mahalanobis
    term is ||L^T d||^2 with L the lower-triangular precision factor.
    """
    outputs = []
    for start in range(0, len(points), point_chunk):
        block = points[start : start + point_chunk]
        delta = block[:, None, :] - centers[None, :, :]
        projected = torch.einsum("nij,mnj->mni", cholesky.transpose(1, 2), delta)
        mahalanobis = projected.square().sum(-1)
        logits = -0.5 * mahalanobis + F.logsigmoid(logit_w)[None]
        color = colors[None] + torch.einsum("nij,mnj->mni", color_grads, delta)
        alpha = logits.exp()
        rendered = (alpha[..., None] * color).sum(dim=1)
        outputs.append(rendered + background[None])
    return torch.cat(outputs)


def cholesky_to_log_covariance(cholesky: torch.Tensor) -> torch.Tensor:
    """Convert precision factors to canonical upper-triangular logSigma features."""
    precision = cholesky @ cholesky.transpose(1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(precision.double())
    log_sigma = torch.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        -eigenvalues.clamp_min(1e-12).log(),
        eigenvectors,
    ).to(cholesky.dtype)
    return torch.stack([log_sigma[:, i, j] for i, j in _TRIU], dim=1)


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
        head_bias[:, 3:6] = 2.5
        head_bias[:, 21] = -4.0
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
        self.initial_scale = initial_scale

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode one (T,H,W,3) window into per-slot jewel parameters."""
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must have shape (T,H,W,3)")
        volume = video.permute(3, 0, 1, 2)[None]
        hidden = self.trunk(volume)[0]
        gu, gv, gt = self.grid_spec.shape
        cells = hidden.permute(3, 2, 1, 0).reshape(self.grid_spec.n_cells, -1)
        raw = self.head(cells).reshape(
            self.grid_spec.n_cells, self.slots_per_cell, 22
        )
        centers = self.cell_centers[:, None] + 1.5 * self.cell_extents * torch.tanh(
            raw[..., 0:3]
        )
        log_diagonal = (raw[..., 3:6] + torch.log(
            1.0 / (self.initial_scale * self.cell_extents.mean())
        )).clamp(-1.0, 7.0)
        off_diagonal = 0.2 * raw[..., 6:9]
        colors = torch.sigmoid(raw[..., 9:12])
        color_grads = 0.2 * raw[..., 12:21]
        logit_w = raw[..., 21].clamp(-9.0, 6.0)
        flat = lambda value: value.reshape(-1, *value.shape[2:])  # noqa: E731
        cholesky = torch.zeros(
            self.grid_spec.n_cells * self.slots_per_cell,
            3,
            3,
            device=video.device,
            dtype=video.dtype,
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
