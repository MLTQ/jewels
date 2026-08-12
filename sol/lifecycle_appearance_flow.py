"""Coupled mark-flow sampling with a frozen lifecycle reference trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.birth_mark_flow import BirthMarkFlowModel


# Canonical features are center (0:3), symmetric log covariance (3:9), RGB,
# RGB gradient, and opacity.  The temporal center plus every log-covariance
# entry touching time are owned by the lifecycle stream.
LIFECYCLE_DIMENSIONS = (2, 5, 7, 8)
SPATIAL_APPEARANCE_DIMENSIONS = tuple(
    index for index in range(22) if index not in LIFECYCLE_DIMENSIONS
)
TEMPORAL_COLOR_GRADIENT_DIMENSIONS = (14, 17, 20)
SPATIAL_GEOMETRY_DIMENSIONS = (0, 1, 3, 4, 6)
RGB_DIMENSIONS = (9, 10, 11)
SPATIAL_COLOR_GRADIENT_DIMENSIONS = (12, 13, 15, 16, 18, 19)
APPEARANCE_DIMENSION_SETS = {
    "all": SPATIAL_APPEARANCE_DIMENSIONS,
    "static-detail": (
        *SPATIAL_GEOMETRY_DIMENSIONS,
        *RGB_DIMENSIONS,
        *SPATIAL_COLOR_GRADIENT_DIMENSIONS,
    ),
    "color-detail": (*RGB_DIMENSIONS, *SPATIAL_COLOR_GRADIENT_DIMENSIONS),
    "color": RGB_DIMENSIONS,
    "geometry": SPATIAL_GEOMETRY_DIMENSIONS,
    "no-temporal-gradient": tuple(
        index
        for index in SPATIAL_APPEARANCE_DIMENSIONS
        if index not in TEMPORAL_COLOR_GRADIENT_DIMENSIONS
    ),
    "no-opacity": tuple(
        index for index in SPATIAL_APPEARANCE_DIMENSIONS if index != 21
    ),
}


@dataclass(frozen=True)
class LifecycleLockedSample:
    """Matched base/candidate standardized marks and their trajectory audit."""

    base: torch.Tensor
    appearance: torch.Tensor
    max_step_lifecycle_error: float
    appearance_dimensions: tuple[int, ...] = SPATIAL_APPEARANCE_DIMENSIONS

    @property
    def lifecycle_exact(self) -> bool:
        return torch.equal(
            self.base[:, LIFECYCLE_DIMENSIONS],
            self.appearance[:, LIFECYCLE_DIMENSIONS],
        ) and self.max_step_lifecycle_error == 0.0

    @property
    def spatial_appearance_mae(self) -> float:
        if not len(self.base):
            return 0.0
        return float(
            (
                self.base[:, SPATIAL_APPEARANCE_DIMENSIONS]
                - self.appearance[:, SPATIAL_APPEARANCE_DIMENSIONS]
            )
            .abs()
            .mean()
        )


def copy_lifecycle_dimensions(
    reference: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """Return candidate marks with canonical lifecycle state copied bit-exactly."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate marks must have identical matrices")
    if reference.shape[1] != 22:
        raise ValueError("the lifecycle contract requires 22-D canonical marks")
    copied = candidate.clone()
    copied[:, LIFECYCLE_DIMENSIONS] = reference[:, LIFECYCLE_DIMENSIONS]
    return copied


def constrain_appearance_dimensions(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    appearance_dimensions: tuple[int, ...],
    appearance_strengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Copy every non-selected coordinate from a matched frozen reference."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate marks must have identical matrices")
    if reference.shape[1] != 22:
        raise ValueError("the appearance contract requires 22-D canonical marks")
    mutable = tuple(appearance_dimensions)
    if not mutable or len(set(mutable)) != len(mutable):
        raise ValueError("appearance dimensions must be non-empty and unique")
    if any(index not in SPATIAL_APPEARANCE_DIMENSIONS for index in mutable):
        raise ValueError("appearance dimensions cannot include lifecycle or unknown features")
    frozen = tuple(index for index in range(22) if index not in mutable)
    constrained = candidate.clone()
    constrained[:, frozen] = reference[:, frozen]
    if appearance_strengths is not None:
        if appearance_strengths.shape != (len(reference),):
            raise ValueError("appearance strengths require one value per mark")
        strengths = appearance_strengths.to(reference)
        if not torch.isfinite(strengths).all() or bool(
            ((strengths < 0) | (strengths > 1)).any()
        ):
            raise ValueError("appearance strengths must be finite values inside [0,1]")
        constrained[:, mutable] = reference[:, mutable] + strengths[:, None] * (
            candidate[:, mutable] - reference[:, mutable]
        )
    return constrained


def _validate_compatible_models(
    base_model: BirthMarkFlowModel, appearance_model: BirthMarkFlowModel
) -> None:
    attributes = ("feature_dim", "text_dim", "guide_dim", "guide_token_dim")
    if any(
        getattr(base_model, name) != getattr(appearance_model, name)
        for name in attributes
    ):
        raise ValueError("base and appearance flows have incompatible feature contracts")
    if base_model.grid_spec != appearance_model.grid_spec:
        raise ValueError("base and appearance flows use different topology grids")


def _guided_velocity(
    model: BirthMarkFlowModel,
    context_raster: torch.Tensor,
    state: torch.Tensor,
    time: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    *,
    cfg_scale: float,
    guide_raster: torch.Tensor | None,
    guide_tokens: torch.Tensor | None,
) -> torch.Tensor:
    conditioned = model(
        context_raster,
        state,
        time,
        cell_indices,
        slot_indices,
        text_condition,
        guide_raster=guide_raster,
        guide_tokens=guide_tokens,
    )
    if text_condition is None or cfg_scale == 1.0:
        return conditioned
    unconditioned = model(
        context_raster,
        state,
        time,
        cell_indices,
        slot_indices,
        None,
        guide_raster=guide_raster,
        guide_tokens=guide_tokens,
    )
    return unconditioned + cfg_scale * (conditioned - unconditioned)


@torch.no_grad()
def sample_lifecycle_locked_birth_marks(
    base_model: BirthMarkFlowModel,
    appearance_model: BirthMarkFlowModel,
    base_context_raster: torch.Tensor,
    appearance_context_raster: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    *,
    steps: int = 20,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
    guide_raster: torch.Tensor | None = None,
    guide_tokens: torch.Tensor | None = None,
    appearance_dimensions: tuple[int, ...] = SPATIAL_APPEARANCE_DIMENSIONS,
    appearance_strengths: torch.Tensor | None = None,
) -> LifecycleLockedSample:
    """Euler-integrate a frozen base and lifecycle-clamped appearance stream."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    _validate_compatible_models(base_model, appearance_model)
    if base_context_raster.device != appearance_context_raster.device:
        raise ValueError("base and appearance contexts must share one device")
    if cell_indices.device != base_context_raster.device or (
        slot_indices.device != base_context_raster.device
    ):
        raise ValueError("topology indices and contexts must share one device")
    device = base_context_raster.device
    initial = torch.randn(
        len(cell_indices),
        base_model.feature_dim,
        device=device,
        generator=generator,
    )
    base_state = initial.clone()
    appearance_state = initial.clone()
    times = torch.linspace(0, 1, steps + 1, device=device)
    base_was_training = base_model.training
    appearance_was_training = appearance_model.training
    base_model.eval()
    appearance_model.eval()
    max_error = 0.0
    for index in range(steps):
        time = times[index : index + 1]
        base_velocity = _guided_velocity(
            base_model,
            base_context_raster,
            base_state,
            time,
            cell_indices,
            slot_indices,
            text_condition,
            cfg_scale=cfg_scale,
            guide_raster=guide_raster,
            guide_tokens=guide_tokens,
        )
        appearance_velocity = _guided_velocity(
            appearance_model,
            appearance_context_raster,
            appearance_state,
            time,
            cell_indices,
            slot_indices,
            text_condition,
            cfg_scale=cfg_scale,
            guide_raster=guide_raster,
            guide_tokens=guide_tokens,
        )
        delta = times[index + 1] - times[index]
        base_state = base_state + delta * base_velocity
        appearance_state = appearance_state + delta * appearance_velocity
        appearance_state = constrain_appearance_dimensions(
            base_state,
            appearance_state,
            appearance_dimensions,
            appearance_strengths,
        )
        if len(base_state):
            step_error = float(
                (
                    base_state[:, LIFECYCLE_DIMENSIONS]
                    - appearance_state[:, LIFECYCLE_DIMENSIONS]
                )
                .abs()
                .max()
            )
            max_error = max(max_error, step_error)
    if base_was_training:
        base_model.train()
    if appearance_was_training:
        appearance_model.train()
    sample = LifecycleLockedSample(
        base_state,
        appearance_state,
        max_error,
        tuple(appearance_dimensions),
    )
    if not sample.lifecycle_exact:
        raise RuntimeError("appearance flow escaped the frozen lifecycle trajectory")
    return sample
