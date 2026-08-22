"""Local fitted-teacher attributes and detached soft correspondence losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.render import covariance_terms


@dataclass(frozen=True)
class LocalTeacherAttributes:
    """Source-owned attributes for an opacity-weighted fitted-field sample."""

    centers: torch.Tensor
    log_scale: torch.Tensor
    axis: torch.Tensor
    opacity: torch.Tensor
    colors: torch.Tensor
    color_grads: torch.Tensor
    active_count: float

    def to(self, device: torch.device) -> LocalTeacherAttributes:
        """Move tensor attributes while retaining the physical active count."""
        return LocalTeacherAttributes(
            centers=self.centers.to(device),
            log_scale=self.log_scale.to(device),
            axis=self.axis.to(device),
            opacity=self.opacity.to(device),
            colors=self.colors.to(device),
            color_grads=self.color_grads.to(device),
            active_count=self.active_count,
        )


def extract_local_teacher_attributes(
    features: torch.Tensor,
    keep: int,
    generator: torch.Generator,
) -> LocalTeacherAttributes:
    """Opacity-sample fitted jewels and expose local covariance/appearance targets."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("teacher features must have shape (N,22)")
    if keep <= 0:
        raise ValueError("teacher keep count must be positive")
    opacity = torch.sigmoid(features[:, 21])
    index = torch.multinomial(
        opacity.clamp_min(1e-6), min(keep, len(features)),
        replacement=False, generator=generator,
    )
    chosen = features[index]
    covariance, _ = covariance_terms(chosen)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    log_scale = 0.5 * eigenvalues.clamp_min(1e-12).log()
    return LocalTeacherAttributes(
        centers=chosen[:, :3],
        log_scale=log_scale.to(chosen.dtype),
        axis=eigenvectors[:, :, -1].to(chosen.dtype),
        opacity=opacity[index],
        colors=chosen[:, 9:12],
        color_grads=chosen[:, 12:21].reshape(-1, 3, 3),
        active_count=float((opacity > 0.02).sum()),
    )


@torch.no_grad()
def soft_local_correspondence(
    student_centers: torch.Tensor,
    teacher_centers: torch.Tensor,
    *,
    neighbors: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-teacher indices and normalized distance-kernel weights."""
    if neighbors <= 0 or temperature <= 0:
        raise ValueError("local neighbors and temperature must be positive")
    if student_centers.ndim != 2 or student_centers.shape[1] != 3:
        raise ValueError("student centers must have shape (N,3)")
    if teacher_centers.ndim != 2 or teacher_centers.shape[1] != 3:
        raise ValueError("teacher centers must have shape (M,3)")
    count = min(neighbors, len(teacher_centers))
    distance, index = torch.cdist(
        student_centers.detach(), teacher_centers.detach()
    ).topk(count, dim=1, largest=False, sorted=True)
    weight = torch.softmax(-distance.square() / (temperature * temperature), dim=1)
    return index, weight


def _weighted_smooth_l1(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Expected smooth-L1 over a student's soft local teacher set."""
    error = F.smooth_l1_loss(
        student[:, None].expand_as(teacher), teacher, reduction="none"
    )
    while error.ndim > weight.ndim:
        error = error.mean(dim=-1)
    return (weight * error).sum(dim=1).mean()


def local_teacher_attribute_losses(
    *,
    student_centers: torch.Tensor,
    student_log_scale: torch.Tensor,
    student_axis: torch.Tensor,
    student_opacity: torch.Tensor,
    student_colors: torch.Tensor,
    student_color_grads: torch.Tensor,
    teacher: LocalTeacherAttributes,
    neighbors: int,
    temperature: float,
    size_offset: float,
    opacity_mass_ratio: float,
) -> dict[str, torch.Tensor]:
    """Score local shape, direction, optical mass, and appearance independently."""
    if opacity_mass_ratio <= 0:
        raise ValueError("opacity mass ratio must be positive")
    index, weight = soft_local_correspondence(
        student_centers, teacher.centers,
        neighbors=neighbors, temperature=temperature,
    )
    teacher_scale = teacher.log_scale[index] + size_offset
    student_scale = student_log_scale.sort(dim=1).values
    scale = _weighted_smooth_l1(student_scale, teacher_scale, weight)

    teacher_axis = teacher.axis[index]
    cosine = (
        student_axis[:, None, :] * teacher_axis
    ).sum(dim=-1).abs().clamp(max=1.0)
    axis = (weight * (1.0 - cosine)).sum(dim=1).mean()

    teacher_tau = -torch.log1p(-teacher.opacity[index].clamp(max=1 - 1e-6))
    target_tau = opacity_mass_ratio * (weight * teacher_tau).sum(dim=1)
    student_tau = -torch.log1p(-student_opacity.clamp(max=1 - 1e-6))
    opacity = F.smooth_l1_loss(student_tau, target_tau)

    color = _weighted_smooth_l1(student_colors, teacher.colors[index], weight)
    gradient = _weighted_smooth_l1(
        student_color_grads, teacher.color_grads[index], weight
    )
    return {
        "scale": scale,
        "axis": axis,
        "opacity": opacity,
        "color": color,
        "gradient": gradient,
    }
