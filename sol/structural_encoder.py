"""Feed-forward encoder whose jewels must describe content rather than sample it.

Path B. Three deliberate departures from `amortized_encoder.VideoToJewelEncoder`:

1. **Scarcity** — ~10k jewels per window instead of 73k, so a primitive must
   stretch to cover structure rather than tile a small patch.
2. **Optional continuous colour seeding** — the lattice encoder sampled video
   RGB on its fixed slots. The irregular arm may instead seed colour at its
   predicted, mobile centres to retain fidelity without forcing colours onto a
   spatial grid; the choice is checkpointed for the later generative model.
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

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.token_grid import GridSpec

ARCHITECTURE = "structural_jewel_encoder_v2"


def stratified_slot_offsets(slots: int) -> torch.Tensor:
    """Return deterministic non-lattice offsets inside one cell.

    Three irrational rotations avoid wraparound duplication in the legacy
    ``round(cuberoot(slots))`` layout while keeping every coordinate strictly
    inside ``[-0.5, 0.5]``.
    """
    if slots <= 0:
        raise ValueError("slots must be positive")
    index = torch.arange(slots, dtype=torch.float32) + 0.5
    rotations = torch.tensor((0.754877666, 0.569840296, 0.438579021))
    return torch.frac(index[:, None] * rotations[None]) - 0.5


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
        max_offset_cells: float = 2.0,
        seed_video_colors: bool = False,
    ) -> None:
        super().__init__()
        if slots_per_cell <= 0 or model_dim % 8:
            raise ValueError("slots must be positive and model_dim divisible by 8")
        self.grid_spec = grid_spec
        self.slots_per_cell = slots_per_cell
        self.n_jewels = grid_spec.n_cells * slots_per_cell
        self.max_offset_cells = float(max_offset_cells)
        self.seed_video_colors = bool(seed_video_colors)
        if self.max_offset_cells <= 0:
            raise ValueError("max_offset_cells must be positive")
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
            "slot_offsets", stratified_slot_offsets(slots_per_cell)
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
        centers = anchor + self.max_offset_cells * self.cell_extents * torch.tanh(
            raw[..., 0:3]
        )
        quaternion = raw[..., 3:7]
        log_scale = (self.log_scale_init + 1.5 * torch.tanh(raw[..., 7:10]) * 2.0).clamp(
            -9.0, 1.0
        )
        if self.seed_video_colors:
            seeded = VideoToJewelEncoder.sample_video_colors(video, centers).clamp(
                1e-3, 1 - 1e-3
            )
            colors = torch.sigmoid(raw[..., 10:13] + torch.logit(seeded))
        else:
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
    cull_mode: str = "exact",
    support_sigma: float = 5.0,
    support_capacity: int = 1024,
    support_base_resolution: int = 32,
    support_level_scale: float = 1.55,
) -> torch.Tensor:
    """Render arbitrary tube factors through the support-complete training path."""
    return cholesky_render(
        prediction["centers"],
        prediction["precision_factor"],
        prediction["colors"],
        prediction["color_grads"],
        prediction["logit_w"],
        points,
        prediction["background"],
        point_chunk=point_chunk,
        cull_mode=cull_mode,
        support_sigma=support_sigma,
        support_capacity=support_capacity,
        support_base_resolution=support_base_resolution,
        support_level_scale=support_level_scale,
    )
