"""Selection and translation operations in normalized (u, v, t) space."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Parallelepiped:
    """An oriented selection volume represented by a center and three half-edges.

    `basis[:, i]` is the world-space vector from the center to the positive face on
    local axis `i`. Points are inside when their solved local coordinates are all
    in [-1, 1].
    """

    center: torch.Tensor
    basis: torch.Tensor

    def __post_init__(self) -> None:
        if self.center.shape != (3,) or self.basis.shape != (3, 3):
            raise ValueError("center must be (3,) and basis must be (3,3)")
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.basis).all():
            raise ValueError("selection geometry must be finite")
        if torch.linalg.det(self.basis.float()).abs() < 1e-8:
            raise ValueError("basis must contain three independent half-edges")

    @classmethod
    def axis_aligned(
        cls,
        center: torch.Tensor,
        half_extent: torch.Tensor,
    ) -> "Parallelepiped":
        if half_extent.shape != (3,) or (half_extent <= 0).any():
            raise ValueError("half_extent must be three positive values")
        return cls(center=center, basis=torch.diag(half_extent))

    def contains(self, points: torch.Tensor, *, tolerance: float = 1e-6) -> torch.Tensor:
        """Return a boolean mask for points shaped (..., 3)."""
        basis = self.basis.to(device=points.device, dtype=points.dtype)
        center = self.center.to(device=points.device, dtype=points.dtype)
        local = torch.linalg.solve(basis, (points - center).unsqueeze(-1)).squeeze(-1)
        return local.abs().amax(dim=-1) <= 1.0 + tolerance

    def translated(self, delta: torch.Tensor) -> "Parallelepiped":
        if delta.shape != (3,):
            raise ValueError("delta must have shape (3,)")
        return Parallelepiped(self.center + delta.to(self.center), self.basis)

    def world_aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Conservative world-axis bounds containing the oriented volume."""
        extent = self.basis.abs().sum(dim=1)
        return self.center - extent, self.center + extent


def translate_selected(
    features: torch.Tensor,
    selection: Parallelepiped,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate selected jewel centers and return (edited_features, selected_mask).

    Feature layout follows `stprim.prior.featurize`: center coordinates occupy
    dimensions 0:3. Covariance and world-frame color gradients are unchanged by a
    pure translation.
    """
    if features.ndim != 2 or features.shape[-1] < 3:
        raise ValueError("features must have shape (N,F), F>=3")
    mask = selection.contains(features[:, :3])
    edited = features.clone()
    edited[mask, :3] += delta.to(device=features.device, dtype=features.dtype)
    return edited, mask
