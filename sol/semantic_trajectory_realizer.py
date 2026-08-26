"""Compose coherent Jewel programs through semantic, density-balanced trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.coherent_source_realizer import CoherentSourceRealizer, fit_coherent_source_realizer
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class SemanticTrajectoryRealizer:
    """Cast a prompt-discriminative foreground tube over a distinct coherent background."""

    coherent: CoherentSourceRealizer
    scene_paths: torch.Tensor
    radius_candidates: torch.Tensor
    minimum_donor_fraction: float
    preferred_radius: float

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
        foreground_scene = (
            scene_token if foreground_scene_token is None else foreground_scene_token
        )
        foreground = int(self.ranked_sources(foreground_scene, program)[0])
        background = next(
            int(row) for row in self.ranked_sources(scene_token, program)
            if int(row) != foreground
        )
        return foreground, background

    def squared_tube_distance(self, centers: torch.Tensor, scene_token: int) -> torch.Tensor:
        if not 0 <= scene_token < len(self.scene_paths):
            raise ValueError("semantic scene token is out of range")
        path = self.scene_paths[scene_token]
        slabs = len(path)
        time = (((centers[:, 2] + 1) * 0.5 * slabs).floor().long()).clamp(0, slabs - 1)
        return (centers[:, :2] - path[time]).square().sum(dim=1)

    def balanced_masks(
        self,
        foreground: int,
        background: int,
        scene_token: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, int, int]:
        """Choose the preregistered radius with minimum valid count mismatch."""
        foreground_distance = self.squared_tube_distance(
            self.coherent.field_centers[foreground], scene_token
        )
        background_distance = self.squared_tube_distance(
            self.coherent.field_centers[background], scene_token
        )
        target_count = len(foreground_distance)
        candidates = []
        for radius in self.radius_candidates:
            foreground_count = int((foreground_distance <= radius.square()).sum())
            background_count = int((background_distance > radius.square()).sum())
            total = foreground_count + background_count
            donor_fraction = min(foreground_count, background_count) / max(total, 1)
            if donor_fraction >= self.minimum_donor_fraction:
                candidates.append((
                    abs(total - target_count),
                    abs(float(radius) - self.preferred_radius),
                    float(radius),
                    foreground_count,
                    background_count,
                ))
        if not candidates:
            raise ValueError("no radius satisfies the material donor contribution floor")
        _, _, radius, foreground_count, background_count = min(candidates)
        return (
            foreground_distance <= radius ** 2,
            background_distance > radius ** 2,
            radius,
            foreground_count,
            background_count,
        )

    @torch.no_grad()
    def sample_from_donors(
        self,
        scene_token: int,
        foreground: int,
        background: int,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | list]]:
        """Compose explicit source-level tokens without any target-derived program."""
        if count <= 0 or foreground == background:
            raise ValueError("semantic trajectory needs a positive count and distinct donors")
        if not 0 <= scene_token < self.coherent.semantic_scene_tokens:
            raise ValueError("semantic trajectory scene token is out of range")
        foreground_mask, background_mask, radius, foreground_count, background_count = (
            self.balanced_masks(foreground, background, scene_token)
        )
        centers = torch.cat([
            self.coherent.field_centers[foreground][foreground_mask],
            self.coherent.field_centers[background][background_mask],
        ])
        tokens = torch.cat([
            self.coherent.field_jewel_tokens[foreground][foreground_mask],
            self.coherent.field_jewel_tokens[background][background_mask],
        ])
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
            "foreground_scene_token": int(self.coherent.scene_owners[foreground]),
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
            "selected_radius": radius,
            "tube_centers": self.scene_paths[scene_token].cpu().tolist(),
        }

    @torch.no_grad()
    def sample_rank_balanced_from_donors(
        self,
        scene_token: int,
        foreground: int,
        background: int,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | list]]:
        """Cast equal foreground/background quotas ranked around the semantic tube."""
        if count <= 0 or count % 2 or foreground == background:
            raise ValueError("rank-balanced casting needs an even count and distinct donors")
        quota = count // 2
        if quota > len(self.coherent.field_centers[foreground]):
            raise ValueError("rank-balanced quota exceeds the donor field")
        foreground_distance = self.squared_tube_distance(
            self.coherent.field_centers[foreground], scene_token
        )
        background_distance = self.squared_tube_distance(
            self.coherent.field_centers[background], scene_token
        )
        foreground_rows = torch.topk(
            foreground_distance, quota, largest=False
        ).indices
        background_rows = torch.topk(
            background_distance, quota, largest=True
        ).indices
        centers = torch.cat([
            self.coherent.field_centers[foreground][foreground_rows],
            self.coherent.field_centers[background][background_rows],
        ])
        tokens = torch.cat([
            self.coherent.field_jewel_tokens[foreground][foreground_rows],
            self.coherent.field_jewel_tokens[background][background_rows],
        ])
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
            "foreground_scene_token": int(self.coherent.scene_owners[foreground]),
            "foreground_training_field": foreground,
            "background_training_field": background,
            "foreground_unadjusted_jewels": quota,
            "background_unadjusted_jewels": quota,
            "foreground_fraction": 0.5,
            "background_fraction": 0.5,
            "unadjusted_jewels": count,
            "requested_jewels": count,
            "emitted_jewels": len(centers),
            "adjustment_fraction": 0.0,
            "foreground_boundary_radius": float(
                foreground_distance[foreground_rows].max().sqrt()
            ),
            "background_boundary_radius": float(
                background_distance[background_rows].min().sqrt()
            ),
            "tube_centers": self.scene_paths[scene_token].cpu().tolist(),
        }

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
        """Select donors from an oracle program, then use the explicit-token sampler."""
        foreground, background = self.select_donors(
            scene_token, program, foreground_scene_token=foreground_scene_token
        )
        return self.sample_from_donors(
            scene_token, foreground, background, count, generator=generator
        )


def _semantic_scene_paths(coherent: CoherentSourceRealizer) -> torch.Tensor:
    spec = GridSpec(coherent.block_shape, slots_per_cell=1)
    x = (torch.arange(spec.shape[0], device=coherent.field_centers.device) + 0.5)
    x = x / spec.shape[0] * 2 - 1
    y = (torch.arange(spec.shape[1], device=coherent.field_centers.device) + 0.5)
    y = y / spec.shape[1] * 2 - 1
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    centrality = torch.exp(-0.5 * ((xx / 0.85).square() + (yy / 0.85).square()))
    paths = []
    for scene in range(coherent.semantic_scene_tokens):
        same = coherent.normalized_descriptors[coherent.scene_owners == scene].mean(dim=0)
        other = coherent.normalized_descriptors[coherent.scene_owners != scene].mean(dim=0)
        saliency = (same - other).square().mean(dim=1).reshape(spec.shape)
        weights = saliency * centrality[:, :, None]
        total = weights.sum(dim=(0, 1)).clamp_min(1e-8)
        centers = torch.stack([
            (weights * xx[:, :, None]).sum(dim=(0, 1)) / total,
            (weights * yy[:, :, None]).sum(dim=(0, 1)) / total,
        ], dim=1)
        padded = torch.cat([centers[:1], centers, centers[-1:]], dim=0)
        paths.append((padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4)
    return torch.stack(paths)


def fit_semantic_trajectory_realizer(
    fields: list[torch.Tensor],
    scene_owners: torch.Tensor,
    *,
    block_codebook,
    physical_codebook,
    jitter_std: float = 0.005,
) -> tuple[SemanticTrajectoryRealizer, dict]:
    """Fit training-only semantic paths over the coherent source representation."""
    coherent, coherent_report = fit_coherent_source_realizer(
        fields,
        scene_owners,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=jitter_std,
        smoothing=0.1,
    )
    radii = torch.linspace(0.45, 1.10, 132, device=fields[0].device)
    realizer = SemanticTrajectoryRealizer(
        coherent=coherent,
        scene_paths=_semantic_scene_paths(coherent),
        radius_candidates=radii,
        minimum_donor_fraction=0.20,
        preferred_radius=0.78,
    )
    return realizer, {
        **coherent_report,
        "semantic_path": "scene_mean_vs_other_mean_descriptor_saliency_smoothed_121",
        "radius_candidates": {
            "minimum": 0.45,
            "maximum": 1.10,
            "count": 132,
        },
        "minimum_donor_fraction": 0.20,
        "preferred_tie_break_radius": 0.78,
        "density_balanced_boundary": True,
    }
