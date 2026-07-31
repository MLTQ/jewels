"""Spacetime coordinate handling for a (T, H, W, 3) video volume.

The only genuinely arbitrary choice in this whole representation lives here:
the exchange rate between pixels and frames. A primitive's extent in t and its
extent in u are measured in different units, and nothing in the data fixes the
conversion. We normalize each axis to [-1, 1] and expose `t_scale` as an
explicit knob on top of that.

Note that per-primitive anisotropy can absorb a global mis-set t_scale — it
only costs the optimizer some work. It matters most for kNN culling, which is
isotropic and therefore does depend on the axes being comparably scaled.
"""

from __future__ import annotations

import torch


def make_grid(
    shape: tuple[int, int, int],
    *,
    t_scale: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(T, H, W) -> (T*H*W, 3) coordinates in normalized (u, v, t), C-order.

    Ordering matches `video.reshape(-1, 3)` for a (T, H, W, 3) tensor.
    """
    T, H, W = shape
    ts = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype) * t_scale
    vs = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    us = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    tt, vv, uu = torch.meshgrid(ts, vs, us, indexing="ij")
    return torch.stack([uu, vv, tt], dim=-1).reshape(-1, 3)


def sample_indices(
    n_total: int,
    n_sample: int,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform random voxel indices for stochastic fitting.

    Fitting against random voxels rather than whole frames is what keeps VRAM
    flat regardless of clip length — the whole point of treating the video as a
    volume instead of a sequence.
    """
    return torch.randint(
        0, n_total, (n_sample,), device=device, generator=generator
    )
