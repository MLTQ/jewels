"""Low-resolution video guidance aligned with the jewel birth raster."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sol.token_grid import GridSpec


def video_to_cell_raster(video: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """Area/trilinear-resample ``(T,H,W,3)`` video into ``(cells,3)`` grid order."""
    if video.ndim != 4 or video.shape[-1] != 3 or min(video.shape[:3]) <= 0:
        raise ValueError("video must have non-empty shape (T,H,W,3)")
    gu, gv, gt = spec.shape
    volume = video.float().permute(3, 0, 1, 2)[None]
    guide = F.interpolate(
        volume,
        size=(gt, gv, gu),
        mode="trilinear",
        align_corners=False,
    )
    return guide.permute(0, 4, 3, 2, 1).reshape(spec.n_cells, 3)
