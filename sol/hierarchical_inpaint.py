"""Map fine edit regions onto block codes and clamp hierarchical flow repair."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.inpaint import VelocityModel, masked_flow_inpaint


def _batched_mask(mask: torch.Tensor, cells: int) -> tuple[torch.Tensor, bool]:
    squeeze = mask.ndim == 1
    batched = mask[None] if squeeze else mask
    if batched.ndim != 2 or batched.shape[1] != cells:
        raise ValueError(f"mask must have shape ({cells},) or (B,{cells})")
    return batched.bool(), squeeze


def coarsen_dirty_mask(
    fine_mask: torch.Tensor,
    fine_shape: tuple[int, int, int],
    block_size: int,
) -> torch.Tensor:
    """Mark a coarse block dirty when any constituent fine cell is dirty."""
    if block_size <= 0 or any(axis % block_size for axis in fine_shape):
        raise ValueError("block size must be positive and divide every fine axis")
    fine_cells = fine_shape[0] * fine_shape[1] * fine_shape[2]
    batched, squeeze = _batched_mask(fine_mask, fine_cells)
    batch = batched.shape[0]
    cu, cv, ct = (axis // block_size for axis in fine_shape)
    blocked = batched.reshape(
        batch, cu, block_size, cv, block_size, ct, block_size
    )
    coarse = blocked.any(dim=(2, 4, 6)).reshape(batch, -1)
    return coarse[0] if squeeze else coarse


def expand_coarse_mask(
    coarse_mask: torch.Tensor,
    fine_shape: tuple[int, int, int],
    block_size: int,
) -> torch.Tensor:
    """Expand dirty block codes back to every fine cell they can influence."""
    if block_size <= 0 or any(axis % block_size for axis in fine_shape):
        raise ValueError("block size must be positive and divide every fine axis")
    cu, cv, ct = (axis // block_size for axis in fine_shape)
    coarse_cells = cu * cv * ct
    batched, squeeze = _batched_mask(coarse_mask, coarse_cells)
    volume = batched.reshape(-1, cu, cv, ct)
    fine = volume.repeat_interleave(block_size, 1).repeat_interleave(
        block_size, 2
    ).repeat_interleave(block_size, 3)
    fine = fine.reshape(batched.shape[0], -1)
    return fine[0] if squeeze else fine


def restore_clean_codes(
    repaired_codes: torch.Tensor,
    known_codes: torch.Tensor,
    dirty_mask: torch.Tensor,
) -> torch.Tensor:
    """Reclamp raw codes after normalization round-trips introduce float error."""
    if repaired_codes.shape != known_codes.shape or repaired_codes.ndim != 3:
        raise ValueError("repaired and known codes must share shape (B,C,D)")
    batched, _ = _batched_mask(dirty_mask, repaired_codes.shape[1])
    if batched.shape[0] == 1 and repaired_codes.shape[0] != 1:
        batched = batched.expand(repaired_codes.shape[0], -1)
    if batched.shape[0] != repaired_codes.shape[0]:
        raise ValueError("dirty-mask batch must match code batch")
    dirty = batched.to(device=repaired_codes.device)[..., None]
    return torch.where(dirty, repaired_codes, known_codes)


@dataclass
class HierarchicalInpaintResult:
    normalized_coarse: torch.Tensor
    dirty_coarse: torch.Tensor
    affected_fine: torch.Tensor


@torch.no_grad()
def hierarchical_masked_flow_inpaint(
    model: VelocityModel,
    known_normalized_coarse: torch.Tensor,
    fine_dirty_mask: torch.Tensor,
    fine_shape: tuple[int, int, int],
    block_size: int,
    *,
    condition: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    steps: int = 50,
    generator: torch.Generator | None = None,
) -> HierarchicalInpaintResult:
    """Repair touched blocks and preserve all other normalized codes exactly."""
    dirty_coarse = coarsen_dirty_mask(fine_dirty_mask, fine_shape, block_size)
    repaired = masked_flow_inpaint(
        model,
        known_normalized_coarse,
        dirty_coarse,
        condition=condition,
        cfg_scale=cfg_scale,
        steps=steps,
        generator=generator,
    )
    affected_fine = expand_coarse_mask(dirty_coarse, fine_shape, block_size)
    return HierarchicalInpaintResult(repaired, dirty_coarse, affected_fine)
