"""Deterministic cuboid masks and held-out metrics for local latent repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from sol.inpaint import VelocityModel, masked_flow_inpaint


def sample_cuboid_masks(
    batch: int,
    grid_shape: tuple[int, int, int],
    min_extent: tuple[int, int, int],
    max_extent: tuple[int, int, int],
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample filled axis-aligned cuboids in canonical coarse raster order."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    if not (
        len(grid_shape) == len(min_extent) == len(max_extent) == 3
        and all(0 < lo <= hi <= axis for axis, lo, hi in zip(grid_shape, min_extent, max_extent))
    ):
        raise ValueError("extents must be positive, ordered, and fit the grid")
    target = torch.device(device)
    masks = torch.zeros((batch, *grid_shape), dtype=torch.bool, device=target)
    for batch_index in range(batch):
        spans = [
            int(
                torch.randint(
                    lo, hi + 1, (1,), device=target, generator=generator
                ).item()
            )
            for lo, hi in zip(min_extent, max_extent, strict=True)
        ]
        starts = [
            int(
                torch.randint(
                    0, axis - span + 1, (1,), device=target, generator=generator
                ).item()
            )
            for axis, span in zip(grid_shape, spans, strict=True)
        ]
        u, v, t = starts
        su, sv, st = spans
        masks[batch_index, u : u + su, v : v + sv, t : t + st] = True
    return masks.reshape(batch, -1)


@dataclass
class RepairEvaluation:
    dirty_latent_mse: float
    zero_fill_latent_mse: float
    clean_max_abs_error: float
    mean_dirty_fraction: float
    examples: int

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_masked_repair(
    model: VelocityModel,
    latents: torch.Tensor,
    conditions: torch.Tensor,
    grid_shape: tuple[int, int, int],
    min_extent: tuple[int, int, int],
    max_extent: tuple[int, int, int],
    *,
    device: torch.device | str,
    examples: int = 8,
    steps: int = 25,
    seed: int = 0,
) -> RepairEvaluation:
    """Compare clamped repair against normalized-space mean filling."""
    count = min(examples, len(latents))
    if count <= 0 or steps <= 0:
        raise ValueError("evaluation requires positive examples and steps")
    target_device = torch.device(device)
    target = latents[:count].to(target_device)
    condition = conditions[:count].to(target_device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    dirty = sample_cuboid_masks(
        count,
        grid_shape,
        min_extent,
        max_extent,
        device=target_device,
        generator=generator,
    )
    repaired = masked_flow_inpaint(
        model,
        target,
        dirty,
        condition=condition,
        steps=steps,
        generator=generator,
    )
    expanded = dirty[..., None].expand_as(target)
    dirty_mse = (repaired[expanded].float() - target[expanded].float()).square().mean()
    zero_mse = target[expanded].float().square().mean()
    clean = ~expanded
    clean_error = (repaired[clean] - target[clean]).abs().max()
    return RepairEvaluation(
        dirty_latent_mse=float(dirty_mse),
        zero_fill_latent_mse=float(zero_mse),
        clean_max_abs_error=float(clean_error),
        mean_dirty_fraction=float(dirty.float().mean()),
        examples=count,
    )
