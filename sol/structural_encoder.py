"""Feed-forward encoder whose jewels must describe content rather than sample it.

Path B. Three deliberate departures from `amortized_encoder.VideoToJewelEncoder`:

1. **Scarcity** — ~10k jewels per window instead of 73k, so a primitive must
   stretch to cover structure rather than tile a small patch.
2. **No content lookup** — colors are pure network outputs. The lattice encoder
   sampled video RGB at each slot, which is why its learned features contributed
   only 0.15 dB and why its field is a re-encoded video.
3. **Tube-capable shape** — quaternion plus three independent log scales (the
   fitter's own parameterization) instead of a near-diagonal Cholesky whose
   small off-diagonals capped anisotropy near 2.

Geometric initialization is retained: lattice starting positions and scales
calibrated for unity coverage. Positions are free to migrate across cells, so
density can follow content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.token_grid import GridSpec

ARCHITECTURE = "structural_jewel_encoder_v1"


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Unit-normalize a quaternion batch and expand to rotation matrices."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def precision_factor(
    quaternion: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """Return M with precision = M M^T for covariance R diag(s^2) R^T.

    The renderer evaluates ||M^T d||^2, which only needs M M^T to equal the
    precision — M need not be triangular — so rotation times inverse scale is a
    valid factor and keeps arbitrary anisotropy expressible.
    """
    rotation = quaternion_to_matrix(quaternion)
    return rotation * torch.exp(-log_scale)[..., None, :]


class StructuralJewelEncoder(nn.Module):
    """Predict a scarce set of tube-capable jewels from one video window."""

    def __init__(
        self,
        *,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
        slots_per_cell: int = 5,
        model_dim: int = 256,
        coverage: float = 1.0,
    ) -> None:
        super().__init__()
        if slots_per_cell <= 0 or model_dim % 8:
            raise ValueError("slots must be positive and model_dim divisible by 8")
        self.grid_spec = grid_spec
        self.slots_per_cell = slots_per_cell
        self.n_jewels = grid_spec.n_cells * slots_per_cell
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
        # 3 offset + 4 quaternion + 3 log-scale + 3 colour + 9 gradient + 1 opacity
        self.head = nn.Linear(model_dim, slots_per_cell * 23)
        self.background_head = nn.Linear(model_dim, 3)
        bias = torch.zeros(slots_per_cell, 23)
        bias[:, 3] = 1.0  # identity quaternion (w = 1)
        bias[:, 13:16] = 0.0  # mid-grey colour before sigmoid
        bias[:, 22] = -2.2  # dim opacity so overlapping jewels sum near unity
        self.head.bias = nn.Parameter(bias.reshape(-1))
        nn.init.normal_(self.head.weight, std=0.01)
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
        # one jewel per slot should cover roughly its share of a cell
        share = float(slots_per_cell) ** (1 / 3)
        self.register_buffer("log_scale_init", torch.log(extents / (2.0 * share) * coverage))
        self.register_buffer(
            "slot_offsets", (torch.rand(slots_per_cell, 3) * 2 - 1) * 0.5
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must have shape (T,H,W,3)")
        hidden = self.trunk(video.permute(3, 0, 1, 2)[None])[0]
        cells = hidden.permute(3, 2, 1, 0).reshape(self.grid_spec.n_cells, -1)
        raw = self.head(cells).reshape(
            self.grid_spec.n_cells, self.slots_per_cell, 23
        )
        anchor = (
            self.cell_centers[:, None]
            + self.slot_offsets[None] * self.cell_extents
        )
        # positions may migrate well beyond their own cell so density can cluster
        centers = anchor + 2.0 * self.cell_extents * torch.tanh(raw[..., 0:3])
        quaternion = raw[..., 3:7]
        log_scale = (self.log_scale_init + 1.5 * torch.tanh(raw[..., 7:10]) * 2.0).clamp(
            -9.0, 1.0
        )
        colors = torch.sigmoid(raw[..., 10:13])
        color_grads = 0.2 * raw[..., 13:22]
        logit_w = raw[..., 22].clamp(-9.0, 6.0)
        flat = lambda value: value.reshape(-1, *value.shape[2:])  # noqa: E731
        return {
            "centers": flat(centers),
            "precision_factor": precision_factor(
                flat(quaternion), flat(log_scale)
            ),
            "log_scale": flat(log_scale),
            "quaternion": flat(quaternion),
            "colors": flat(colors),
            "color_grads": flat(color_grads).reshape(-1, 3, 3),
            "logit_w": flat(logit_w).reshape(-1),
            "background": torch.sigmoid(self.background_head(cells.mean(dim=0))),
        }

    def canonical_features(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the canonical 22-D layout so existing tools apply unchanged."""
        with torch.no_grad():
            rotation = quaternion_to_matrix(prediction["quaternion"])
            log_sigma = torch.einsum(
                "nij,nj,nkj->nik",
                rotation.double(),
                2.0 * prediction["log_scale"].double(),
                rotation.double(),
            ).to(prediction["centers"].dtype)
            triu = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
            packed = torch.stack([log_sigma[:, i, j] for i, j in triu], dim=1)
            return torch.cat(
                (
                    prediction["centers"],
                    packed,
                    prediction["colors"],
                    prediction["color_grads"].reshape(-1, 9),
                    prediction["logit_w"][:, None],
                ),
                dim=1,
            )


def render_structural(
    prediction: dict[str, torch.Tensor],
    points: torch.Tensor,
    *,
    point_chunk: int = 1024,
) -> torch.Tensor:
    """Additive render matching `sol.render.render_exact` semantics."""
    centers = prediction["centers"]
    factor = prediction["precision_factor"]
    colors = prediction["colors"]
    grads = prediction["color_grads"]
    logit_w = prediction["logit_w"]
    background = prediction["background"]

    def block(query: torch.Tensor) -> torch.Tensor:
        delta = query[:, None, :] - centers[None]
        projected = torch.einsum("nij,mnj->mni", factor.transpose(1, 2), delta)
        logits = -0.5 * projected.square().sum(-1) + F.logsigmoid(logit_w)[None]
        colour = colors[None] + torch.einsum("nij,mnj->mni", grads, delta)
        return (logits.exp()[..., None] * colour).sum(dim=1) + background[None]

    needs_graph = torch.is_grad_enabled() and centers.requires_grad
    outputs = []
    for start in range(0, len(points), point_chunk):
        chunk = points[start : start + point_chunk]
        outputs.append(
            torch.utils.checkpoint.checkpoint(block, chunk, use_reentrant=False)
            if needs_graph
            else block(chunk)
        )
    return torch.cat(outputs)
