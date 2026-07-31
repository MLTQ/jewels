"""Anisotropic spacetime primitive field.

The parameter container for the additive-splat renderer (models/render.py) and
the unit the stage-2 generative model will emit: a video is a set of these.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix.

    Quaternions are normalized here, so the raw parameter is unconstrained.
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


class PrimitiveField(nn.Module):
    """N anisotropic primitives living in normalized (u, v, t) space.

    Every primitive carries:
      mu          (N, 3)     center, normalized coords in [-1, 1]^3
      log_scale   (N, 3)     log of principal-axis extents
      quat        (N, 4)     orientation; tilt into t IS velocity
      color       (N, 3)     RGB (P0 term)
      color_grad  (N, 3, 3)  optional P1 term: d(color)/d(position)
      logit_w     (N,)       weight / opacity logit

    The (scale, quat) pair defines the per-primitive Gaussian covariance. Its
    normalized form (the anisotropy tensor) is what the stage-2 vocabulary is
    meant to quantize — note the covariance itself, not (scale, quat), is the
    canonical object: quaternions double-cover rotations (q and -q are the same
    primitive) and axis order is permutable, so raw parameters must not be fed
    to a prior as-is.
    """

    def __init__(
        self,
        num_primitives: int,
        *,
        p1_color: bool = True,
        init_scale: float = 0.12,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.p1_color = p1_color
        g = generator

        n = num_primitives
        # Random draws happen on the generator's device (CUDA ops reject a CPU
        # generator), then move; a seed therefore gives the same init on every
        # target device.
        init_dev = g.device if g is not None else torch.device(device)
        mu = (torch.rand(n, 3, generator=g, device=init_dev) * 2.0) - 1.0
        self.mu = nn.Parameter(mu.to(device))
        self.log_scale = nn.Parameter(
            torch.full((n, 3), math.log(init_scale), device=device)
        )

        quat = torch.zeros(n, 4, device=init_dev)
        quat[:, 0] = 1.0
        quat = quat + 0.01 * torch.randn(n, 4, generator=g, device=init_dev)
        self.quat = nn.Parameter(quat.to(device))

        self.color = nn.Parameter(
            (0.5 + 0.1 * torch.randn(n, 3, generator=g, device=init_dev)).to(device)
        )
        if p1_color:
            self.color_grad = nn.Parameter(torch.zeros(n, 3, 3, device=device))
        else:
            self.register_parameter("color_grad", None)

        self.logit_w = nn.Parameter(torch.zeros(n, device=device))

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.mu.shape[0]

    @property
    def device(self) -> torch.device:
        return self.mu.device

    def scales(self) -> torch.Tensor:
        """(N, 3) positive extents, clamped away from degeneracy."""
        return self.log_scale.clamp(-8.0, 2.0).exp()

    def rotations(self) -> torch.Tensor:
        """(N, 3, 3)."""
        return quat_to_rotmat(self.quat)

    def weights(self) -> torch.Tensor:
        """(N,) positive weights."""
        return torch.sigmoid(self.logit_w)

    # ------------------------------------------------------------------ #

    def gather(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather parameters for a (M, K) index tensor of selected primitives.

        Returns tensors shaped (M, K, ...). Used by the renderer after culling.
        """
        out = {
            "mu": self.mu[idx],
            "scale": self.scales()[idx],
            "rot": self.rotations()[idx],
            "color": self.color[idx],
            "logit_w": self.logit_w[idx],
        }
        if self.p1_color:
            out["color_grad"] = self.color_grad[idx]
        return out

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def subset_(self, keep: torch.Tensor) -> None:
        """In-place prune to a boolean/index mask. Optimizer must be rebuilt."""
        self.mu = nn.Parameter(self.mu[keep].contiguous())
        self.log_scale = nn.Parameter(self.log_scale[keep].contiguous())
        self.quat = nn.Parameter(self.quat[keep].contiguous())
        self.color = nn.Parameter(self.color[keep].contiguous())
        self.logit_w = nn.Parameter(self.logit_w[keep].contiguous())
        if self.p1_color:
            self.color_grad = nn.Parameter(self.color_grad[keep].contiguous())

    @torch.no_grad()
    def append_(self, other: dict[str, torch.Tensor]) -> None:
        """Append new primitives from a dict of tensors. Rebuild optimizer."""
        self.mu = nn.Parameter(torch.cat([self.mu, other["mu"]], 0))
        self.log_scale = nn.Parameter(
            torch.cat([self.log_scale, other["log_scale"]], 0)
        )
        self.quat = nn.Parameter(torch.cat([self.quat, other["quat"]], 0))
        self.color = nn.Parameter(torch.cat([self.color, other["color"]], 0))
        self.logit_w = nn.Parameter(torch.cat([self.logit_w, other["logit_w"]], 0))
        if self.p1_color:
            self.color_grad = nn.Parameter(
                torch.cat([self.color_grad, other["color_grad"]], 0)
            )
