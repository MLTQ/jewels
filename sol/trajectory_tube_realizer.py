"""Compose persistent foreground and background Jewel programs through a moving tube."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.coherent_source_realizer import (
    CoherentSourceRealizer,
    fit_coherent_source_realizer,
)
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class TrajectoryTubeRealizer:
    """Cast two coherent donor fields through one connected moving XY-time tube."""

    coherent: CoherentSourceRealizer
    tube_radius: float

    @property
    def device(self) -> torch.device:
        return self.coherent.field_centers.device

    def ranked_sources(self, scene_token: int, program: torch.Tensor) -> torch.Tensor:
        eligible, distances = self.coherent.source_distances(scene_token, program)
        return eligible[torch.argsort(distances)]

    def select_donors(
        self,
        scene_token: int,
        program: torch.Tensor,
        *,
        foreground_scene_token: int | None = None,
    ) -> tuple[int, int]:
        """Choose a foreground owner and a distinct correct-scene background owner."""
        foreground_scene = (
            scene_token if foreground_scene_token is None else foreground_scene_token
        )
        foreground = int(self.ranked_sources(foreground_scene, program)[0])
        background_ranked = self.ranked_sources(scene_token, program)
        background = next(int(row) for row in background_ranked if int(row) != foreground)
        return foreground, background

    def tube_centers(self, foreground: int, background: int) -> torch.Tensor:
        """Compute the eight-slab moving tube path from donor disagreement only."""
        spec = GridSpec(self.coherent.block_shape, slots_per_cell=1)
        descriptors = self.coherent.normalized_descriptors
        disagreement = (
            descriptors[foreground] - descriptors[background]
        ).square().mean(dim=1).reshape(spec.shape)
        x = (torch.arange(spec.shape[0], device=self.device) + 0.5) / spec.shape[0] * 2 - 1
        y = (torch.arange(spec.shape[1], device=self.device) + 0.5) / spec.shape[1] * 2 - 1
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        centrality = torch.exp(-0.5 * ((xx / 0.85).square() + (yy / 0.85).square()))
        weights = disagreement * centrality[:, :, None]
        total = weights.sum(dim=(0, 1)).clamp_min(1e-8)
        centers = torch.stack([
            (weights * xx[:, :, None]).sum(dim=(0, 1)) / total,
            (weights * yy[:, :, None]).sum(dim=(0, 1)) / total,
        ], dim=1)
        padded = torch.cat([centers[:1], centers, centers[-1:]], dim=0)
        return (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4

    def inside_tube(self, centers: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Assign continuous centroids to the nearest fixed time slab and tube cross-section."""
        slabs = len(path)
        time = (((centers[:, 2] + 1) * 0.5 * slabs).floor().long()).clamp(0, slabs - 1)
        delta = centers[:, :2] - path[time]
        return delta.square().sum(dim=1) <= self.tube_radius ** 2

    @torch.no_grad()
    def sample(
        self,
        scene_token: int,
        program: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
        foreground_scene_token: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | list]]:
        """Compose two complete active-token fields and adjust to an exact row count."""
        if count <= 0:
            raise ValueError("trajectory-tube sampling requires a positive count")
        foreground, background = self.select_donors(
            scene_token, program, foreground_scene_token=foreground_scene_token
        )
        path = self.tube_centers(foreground, background)
        foreground_centers = self.coherent.field_centers[foreground]
        background_centers = self.coherent.field_centers[background]
        foreground_mask = self.inside_tube(foreground_centers, path)
        background_mask = ~self.inside_tube(background_centers, path)
        centers = torch.cat([
            foreground_centers[foreground_mask], background_centers[background_mask]
        ])
        tokens = torch.cat([
            self.coherent.field_jewel_tokens[foreground][foreground_mask],
            self.coherent.field_jewel_tokens[background][background_mask],
        ])
        foreground_count = int(foreground_mask.sum())
        background_count = int(background_mask.sum())
        unadjusted = len(centers)
        if unadjusted > count:
            rows = torch.randperm(unadjusted, generator=generator, device=self.device)[:count]
            centers, tokens = centers[rows], tokens[rows]
        elif unadjusted < count:
            rows = torch.randint(
                unadjusted, (count - unadjusted,), generator=generator, device=self.device
            )
            centers = torch.cat([centers, centers[rows]])
            tokens = torch.cat([tokens, tokens[rows]])
        if self.coherent.jitter_std > 0:
            centers = (
                centers
                + torch.randn(
                    centers.shape, generator=generator, device=self.device
                )
                * self.coherent.jitter_std
            ).clamp(-0.999, 0.999)
        return centers, tokens, {
            "scene_token": scene_token,
            "foreground_scene_token": (
                scene_token if foreground_scene_token is None else foreground_scene_token
            ),
            "foreground_training_field": foreground,
            "background_training_field": background,
            "foreground_unadjusted_jewels": foreground_count,
            "background_unadjusted_jewels": background_count,
            "foreground_fraction": foreground_count / unadjusted,
            "background_fraction": background_count / unadjusted,
            "unadjusted_jewels": unadjusted,
            "requested_jewels": count,
            "emitted_jewels": len(centers),
            "adjustment_fraction": abs(unadjusted - count) / count,
            "tube_radius": self.tube_radius,
            "tube_centers": path.cpu().tolist(),
        }


def fit_trajectory_tube_realizer(
    fields: list[torch.Tensor],
    scene_owners: torch.Tensor,
    *,
    block_codebook,
    physical_codebook,
    jitter_std: float = 0.005,
    tube_radius: float = 0.78,
) -> tuple[TrajectoryTubeRealizer, dict]:
    """Fit the train-owned coherent base used by the trajectory compositor."""
    if tube_radius <= 0:
        raise ValueError("tube radius must be positive")
    coherent, report = fit_coherent_source_realizer(
        fields,
        scene_owners,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=jitter_std,
        smoothing=0.1,
    )
    return TrajectoryTubeRealizer(coherent=coherent, tube_radius=tube_radius), {
        **report,
        "tube_radius": tube_radius,
        "tube_path": "donor_disagreement_centroid_temporally_smoothed_121",
        "foreground_background_are_distinct": True,
    }
