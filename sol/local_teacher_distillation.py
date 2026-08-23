"""Local fitted-teacher attributes and detached renderer-responsibility losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.render import covariance_terms


@dataclass(frozen=True)
class LocalTeacherAttributes:
    """Source-owned attributes for an opacity-weighted fitted-field sample."""

    centers: torch.Tensor
    covariance: torch.Tensor
    precision: torch.Tensor
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
            covariance=self.covariance.to(device),
            precision=self.precision.to(device),
            log_scale=self.log_scale.to(device),
            axis=self.axis.to(device),
            opacity=self.opacity.to(device),
            colors=self.colors.to(device),
            color_grads=self.color_grads.to(device),
            active_count=self.active_count,
        )


@dataclass(frozen=True)
class TeacherResponsibilityTargets:
    """Composited teacher moments evaluated at detached student centers."""

    log_scale: torch.Tensor
    axis: torch.Tensor
    optical_tau: torch.Tensor
    colors: torch.Tensor
    color_grads: torch.Tensor
    effective_count: torch.Tensor
    support_count: torch.Tensor
    used_fallback: torch.Tensor


def extract_local_teacher_attributes(
    features: torch.Tensor,
    keep: int,
    generator: torch.Generator,
    *,
    sampling: str = "opacity",
) -> LocalTeacherAttributes:
    """Opacity-sample fitted jewels and expose local covariance/appearance targets."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("teacher features must have shape (N,22)")
    if keep <= 0:
        raise ValueError("teacher keep count must be positive")
    opacity = torch.sigmoid(features[:, 21])
    count = min(keep, len(features))
    if sampling == "opacity":
        index = torch.multinomial(
            opacity.clamp_min(1e-6), count,
            replacement=False, generator=generator,
        )
    elif sampling == "active_uniform":
        pool = (opacity > 0.02).nonzero(as_tuple=False).reshape(-1)
        if not len(pool):
            pool = torch.arange(len(features), device=features.device)
        order = torch.randperm(len(pool), generator=generator, device=pool.device)
        index = pool[order[: min(count, len(pool))]]
    else:
        raise ValueError("teacher sampling must be opacity or active_uniform")
    chosen = features[index]
    covariance, precision = covariance_terms(chosen)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    log_scale = 0.5 * eigenvalues.clamp_min(1e-12).log()
    return LocalTeacherAttributes(
        centers=chosen[:, :3],
        covariance=covariance,
        precision=precision,
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


@torch.no_grad()
def renderer_responsibility_targets(
    student_centers: torch.Tensor,
    teacher: LocalTeacherAttributes,
    *,
    support_sigma: float,
    temperature: float,
) -> TeacherResponsibilityTargets:
    """Evaluate normalized fitted-field contribution moments at student centers."""
    if support_sigma <= 0 or temperature <= 0:
        raise ValueError("responsibility support and temperature must be positive")
    if student_centers.ndim != 2 or student_centers.shape[1] != 3:
        raise ValueError("student centers must have shape (N,3)")
    if not len(teacher.centers):
        raise ValueError("responsibility teacher must not be empty")

    centers = student_centers.detach()
    delta = centers[:, None, :] - teacher.centers[None]
    mahalanobis = torch.einsum(
        "nmi,mij,nmj->nm", delta, teacher.precision, delta
    )
    logits = teacher.opacity.clamp_min(1e-12).log()[None] - 0.5 * mahalanobis
    support = mahalanobis <= support_sigma * support_sigma
    missing = ~support.any(dim=1)
    if bool(missing.any()):
        fallback = mahalanobis[missing].argmin(dim=1)
        support[missing] = False
        support[missing, fallback] = True
    weight = torch.softmax(
        logits.masked_fill(~support, -torch.inf) / temperature, dim=1
    )

    mean_center = torch.einsum("nm,mi->ni", weight, teacher.centers)
    centered = teacher.centers[None] - mean_center[:, None]
    second_moment = teacher.covariance[None] + torch.einsum(
        "nmi,nmj->nmij", centered, centered
    )
    mixture_covariance = torch.einsum("nm,nmij->nij", weight, second_moment)
    eigenvalues, eigenvectors = torch.linalg.eigh(mixture_covariance.double())
    log_scale = 0.5 * eigenvalues.clamp_min(1e-12).log()

    local_colors = teacher.colors[None] + torch.einsum(
        "mij,nmj->nmi", teacher.color_grads, delta
    )
    colors = torch.einsum("nm,nmi->ni", weight, local_colors)
    log_weight_gradient = -torch.einsum(
        "mij,nmj->nmi", teacher.precision, delta
    )
    mean_log_weight_gradient = torch.einsum(
        "nm,nmi->ni", weight, log_weight_gradient
    )
    centered_gradient = log_weight_gradient - mean_log_weight_gradient[:, None]
    local_jacobian = teacher.color_grads[None] + torch.einsum(
        "nmi,nmj->nmij", local_colors, centered_gradient
    )
    color_grads = torch.einsum("nm,nmij->nij", weight, local_jacobian)

    optical_tau = torch.einsum(
        "nm,m->n", weight,
        -torch.log1p(-teacher.opacity.clamp(max=1 - 1e-6)),
    )
    return TeacherResponsibilityTargets(
        log_scale=log_scale.to(centers.dtype),
        axis=eigenvectors[:, :, -1].to(centers.dtype),
        optical_tau=optical_tau,
        colors=colors,
        color_grads=color_grads,
        effective_count=weight.square().sum(dim=1).reciprocal(),
        support_count=support.sum(dim=1).to(centers.dtype),
        used_fallback=missing,
    )


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


def responsibility_teacher_moment_losses(
    *,
    student_centers: torch.Tensor,
    student_log_scale: torch.Tensor,
    student_axis: torch.Tensor,
    student_opacity: torch.Tensor,
    student_colors: torch.Tensor,
    student_color_grads: torch.Tensor,
    teacher: LocalTeacherAttributes,
    support_sigma: float,
    temperature: float,
    size_offset: float,
    opacity_mass_ratio: float,
) -> tuple[dict[str, torch.Tensor], TeacherResponsibilityTargets]:
    """Score student attributes against composited teacher responsibility moments."""
    if opacity_mass_ratio <= 0:
        raise ValueError("opacity mass ratio must be positive")
    targets = renderer_responsibility_targets(
        student_centers, teacher,
        support_sigma=support_sigma, temperature=temperature,
    )
    scale = F.smooth_l1_loss(
        student_log_scale.sort(dim=1).values,
        (targets.log_scale + size_offset).clamp(-9.0, 1.0),
    )
    cosine = (
        student_axis * targets.axis
    ).sum(dim=-1).abs().clamp(max=1.0)
    axis = (1.0 - cosine).mean()
    student_tau = -torch.log1p(-student_opacity.clamp(max=1 - 1e-6))
    opacity = F.smooth_l1_loss(
        student_tau, (opacity_mass_ratio * targets.optical_tau).clamp(max=6.0)
    )
    color = F.smooth_l1_loss(student_colors, targets.colors.clamp(0.0, 1.0))
    gradient = F.smooth_l1_loss(
        student_color_grads, targets.color_grads.clamp(-0.25, 0.25)
    )
    return {
        "scale": scale,
        "axis": axis,
        "opacity": opacity,
        "color": color,
        "gradient": gradient,
    }, targets
