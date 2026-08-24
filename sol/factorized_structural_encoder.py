"""Factorized mobile-jewel encoder with independent geometry and appearance paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.amortized_encoder import VideoToJewelEncoder
from sol.structural_encoder import (
    precision_factor,
    quaternion_to_matrix,
    stratified_slot_offsets,
)
from sol.token_grid import GridSpec

ARCHITECTURE = "factorized_structural_jewel_encoder_v3"
GEOMETRY_CHANNELS = (*range(10), 22)
APPEARANCE_CONTRACTS = ("bounded", "residual")


def spacetime_trunk(model_dim: int, shape: tuple[int, int, int]) -> nn.Sequential:
    """Build the geometry feature trunk used by the compatible v2 transplant."""
    gu, gv, gt = shape
    return nn.Sequential(
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


def sample_feature_volume(volume: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample a CxTxHxW feature volume at normalized x/y/time centres."""
    query = centers.reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(volume, query, align_corners=True)
    return sampled[0, :, :, 0, 0].transpose(0, 1)


class FactorizedStructuralJewelEncoder(nn.Module):
    """Emit geometry and appearance through disjoint parameter branches."""

    architecture = ARCHITECTURE

    def __init__(
        self,
        *,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
        slots_per_cell: int = 10,
        model_dim: int = 256,
        max_offset_cells: float = 4.0,
        appearance_dim: int = 32,
        appearance_hidden: int = 64,
        appearance_contract: str = "bounded",
    ) -> None:
        super().__init__()
        if slots_per_cell <= 0 or model_dim % 8 or appearance_dim % 8:
            raise ValueError("slots must be positive and feature dimensions divisible by 8")
        if max_offset_cells <= 0 or appearance_hidden <= 0:
            raise ValueError("mobility and appearance hidden size must be positive")
        if appearance_contract not in APPEARANCE_CONTRACTS:
            raise ValueError(
                f"appearance contract must be one of {APPEARANCE_CONTRACTS}"
            )
        self.grid_spec = grid_spec
        self.slots_per_cell = slots_per_cell
        self.n_jewels = grid_spec.n_cells * slots_per_cell
        self.max_offset_cells = float(max_offset_cells)
        self.appearance_dim = int(appearance_dim)
        self.appearance_hidden = int(appearance_hidden)
        self.appearance_contract = appearance_contract
        gu, gv, gt = grid_spec.shape

        self.geometry_trunk = spacetime_trunk(model_dim, grid_spec.shape)
        self.geometry_head = nn.Linear(model_dim, slots_per_cell * 11)
        geometry_bias = torch.zeros(slots_per_cell, 11)
        geometry_bias[:, 3] = 1.0
        geometry_bias[:, 10] = -2.2
        self.geometry_head.bias = nn.Parameter(geometry_bias.reshape(-1))
        nn.init.normal_(self.geometry_head.weight, std=0.01)

        self.appearance_fine = nn.Sequential(
            nn.Conv3d(
                3, appearance_dim, 3, stride=(1, 2, 2), padding=1
            ),
            nn.GroupNorm(8, appearance_dim),
            nn.SiLU(),
            nn.Conv3d(appearance_dim, appearance_dim, 3, padding=1),
            nn.GroupNorm(8, appearance_dim),
            nn.SiLU(),
        )
        self.appearance_coarse = nn.Sequential(
            nn.Conv3d(
                appearance_dim, appearance_dim, 3,
                stride=(2, 2, 2), padding=1,
            ),
            nn.GroupNorm(8, appearance_dim),
            nn.SiLU(),
        )
        self.appearance_head = nn.Sequential(
            nn.Linear(2 * appearance_dim + 3, appearance_hidden),
            nn.SiLU(),
            nn.Linear(appearance_hidden, 12),
        )
        self.background_head = nn.Linear(appearance_dim, 3)
        nn.init.zeros_(self.appearance_head[-1].weight)
        nn.init.zeros_(self.appearance_head[-1].bias)
        nn.init.zeros_(self.background_head.weight)
        nn.init.zeros_(self.background_head.bias)
        if appearance_contract == "residual":
            self.appearance_residual_head = nn.Linear(
                2 * appearance_dim + 3, 12
            )
            nn.init.zeros_(self.appearance_residual_head.weight)
            nn.init.zeros_(self.appearance_residual_head.bias)

        coordinates = torch.stack(
            torch.meshgrid(
                (torch.arange(gu) + 0.5) / gu * 2 - 1,
                (torch.arange(gv) + 0.5) / gv * 2 - 1,
                (torch.arange(gt) + 0.5) / gt * 2 - 1,
                indexing="ij",
            ),
            dim=-1,
        ).reshape(grid_spec.n_cells, 3)
        extents = torch.tensor([2.0 / gu, 2.0 / gv, 2.0 / gt])
        share = float(slots_per_cell) ** (1 / 3)
        self.register_buffer("cell_centers", coordinates)
        self.register_buffer("cell_extents", extents)
        self.register_buffer("slot_offsets", stratified_slot_offsets(slots_per_cell))
        self.register_buffer(
            "log_scale_init", torch.log(extents / (2.0 * share))
        )

    @property
    def model_args(self) -> dict[str, int | float | str]:
        """Constructor arguments that define checkpoint compatibility."""
        return {
            "slots_per_cell": self.slots_per_cell,
            "model_dim": self.geometry_head.in_features,
            "max_offset_cells": self.max_offset_cells,
            "appearance_dim": self.appearance_dim,
            "appearance_hidden": self.appearance_hidden,
            "appearance_contract": self.appearance_contract,
        }

    def freeze_geometry(self) -> None:
        """Make the geometry path immutable while appearance remains trainable."""
        for module in (self.geometry_trunk, self.geometry_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def load_v2_geometry(self, state: dict[str, torch.Tensor]) -> None:
        """Transplant the exactly compatible v2 trunk and geometry output rows."""
        trunk = {
            key.removeprefix("trunk."): value
            for key, value in state.items()
            if key.startswith("trunk.")
        }
        self.geometry_trunk.load_state_dict(trunk)
        source_weight = state["head.weight"].reshape(
            self.slots_per_cell, 23, self.geometry_head.in_features
        )
        source_bias = state["head.bias"].reshape(self.slots_per_cell, 23)
        self.geometry_head.weight.copy_(
            source_weight[:, GEOMETRY_CHANNELS].reshape_as(self.geometry_head.weight)
        )
        self.geometry_head.bias.copy_(
            source_bias[:, GEOMETRY_CHANNELS].reshape_as(self.geometry_head.bias)
        )

    def load_bounded_appearance_expansion(
        self, state: dict[str, torch.Tensor]
    ) -> None:
        """Load a bounded v3 state while retaining a zero residual expansion."""
        if self.appearance_contract != "residual":
            raise ValueError("bounded appearance expansion requires residual contract")
        incompatible = self.load_state_dict(state, strict=False)
        expected_missing = {
            "appearance_residual_head.weight",
            "appearance_residual_head.bias",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "bounded checkpoint differs beyond the declared residual head: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError("video must have shape (T,H,W,3)")
        volume = video.permute(3, 0, 1, 2)[None]
        hidden = self.geometry_trunk(volume)[0]
        cells = hidden.permute(3, 2, 1, 0).reshape(self.grid_spec.n_cells, -1)
        raw = self.geometry_head(cells).reshape(
            self.grid_spec.n_cells, self.slots_per_cell, 11
        )
        anchor = self.cell_centers[:, None] + self.slot_offsets[None] * self.cell_extents
        centers = anchor + self.max_offset_cells * self.cell_extents * torch.tanh(
            raw[..., :3]
        )
        quaternion = raw[..., 3:7]
        log_scale = (
            self.log_scale_init + 3.0 * torch.tanh(raw[..., 7:10])
        ).clamp(-9.0, 1.0)
        logit_w = raw[..., 10].clamp(-9.0, 6.0)

        flat = lambda value: value.reshape(-1, *value.shape[2:])  # noqa: E731
        centers_flat = flat(centers)
        appearance_centers = centers_flat.detach()
        fine = self.appearance_fine(volume)
        coarse = self.appearance_coarse(fine)
        fine_sample = sample_feature_volume(fine, appearance_centers)
        coarse_sample = sample_feature_volume(coarse, appearance_centers)
        seed = VideoToJewelEncoder.sample_video_colors(
            video, appearance_centers
        ).reshape(-1, 3).clamp(1e-3, 1 - 1e-3)
        appearance_input = torch.cat((fine_sample, coarse_sample, seed), dim=1)
        appearance = self.appearance_head(appearance_input)
        colors = torch.sigmoid(torch.logit(seed) + appearance[:, :3])
        color_grads = 0.25 * torch.tanh(appearance[:, 3:].reshape(-1, 3, 3))
        appearance_residual = torch.zeros_like(appearance)
        if self.appearance_contract == "residual":
            appearance_residual = self.appearance_residual_head(appearance_input)
            colors = colors + appearance_residual[:, :3]
            color_grads = color_grads + appearance_residual[:, 3:].reshape(-1, 3, 3)
        base_background = video.mean(dim=(0, 1, 2)).clamp(1e-3, 1 - 1e-3)
        background = torch.sigmoid(
            torch.logit(base_background) + self.background_head(coarse.mean(dim=(0, 2, 3, 4)))
        )
        return {
            "centers": centers_flat,
            "precision_factor": precision_factor(
                flat(quaternion), flat(log_scale)
            ),
            "log_scale": flat(log_scale),
            "quaternion": flat(quaternion),
            "colors": colors,
            "color_grads": color_grads,
            "appearance_residual": appearance_residual,
            "logit_w": flat(logit_w).reshape(-1),
            "background": background,
        }

    def canonical_features(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the canonical 22-D field representation used by all audits."""
        with torch.no_grad():
            rotation = quaternion_to_matrix(prediction["quaternion"])
            log_sigma = torch.einsum(
                "nij,nj,nkj->nik", rotation.double(),
                2.0 * prediction["log_scale"].double(), rotation.double(),
            ).to(prediction["centers"].dtype)
            triu = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
            packed = torch.stack([log_sigma[:, i, j] for i, j in triu], dim=1)
            return torch.cat((
                prediction["centers"], packed, prediction["colors"],
                prediction["color_grads"].reshape(-1, 9),
                prediction["logit_w"][:, None],
            ), dim=1)
