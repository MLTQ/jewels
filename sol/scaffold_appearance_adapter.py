"""Compact RGB-only residual adapter for a frozen scaffold mark flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.latent_prior import timestep_embedding
from sol.streaming_model import ResidualMLP, _rank_basis
from sol.token_grid import GridSpec


RGB_DIMENSIONS = (9, 10, 11)
NON_RGB_DIMENSIONS = tuple(index for index in range(22) if index not in RGB_DIMENSIONS)


class ScaffoldAppearanceAdapter(nn.Module):
    """Predict an addressed RGB velocity residual over a frozen base flow."""

    def __init__(
        self,
        *,
        feature_dim: int = 22,
        context_dim: int = 46,
        guide_dim: int = 3,
        text_dim: int = 512,
        model_dim: int = 48,
        depth: int = 2,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
    ) -> None:
        super().__init__()
        if feature_dim != 22:
            raise ValueError("the canonical appearance adapter requires 22 features")
        if min(context_dim, guide_dim, text_dim, model_dim, depth) <= 0:
            raise ValueError("adapter dimensions and depth must be positive")
        if model_dim % 8:
            raise ValueError("model_dim must be divisible by eight")
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.guide_dim = guide_dim
        self.text_dim = text_dim
        self.model_dim = model_dim
        self.grid_spec = grid_spec
        self.mark_projection = nn.Linear(feature_dim, model_dim)
        self.base_velocity_projection = nn.Linear(feature_dim, model_dim)
        self.context_projection = nn.Linear(context_dim, model_dim)
        self.guide_projection = nn.Linear(guide_dim, model_dim)
        self.rank_projection = nn.Linear(8, model_dim)
        self.text_projection = nn.Linear(text_dim, model_dim)
        self.null_text_condition = nn.Parameter(torch.zeros(model_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        gu, gv, gt = grid_spec.shape
        self.u_embedding = nn.Parameter(torch.randn(gu, model_dim) * 0.02)
        self.v_embedding = nn.Parameter(torch.randn(gv, model_dim) * 0.02)
        self.t_embedding = nn.Parameter(torch.randn(gt, model_dim) * 0.02)
        self.blocks = nn.ModuleList(ResidualMLP(model_dim) for _ in range(depth))
        self.rgb_velocity_head = nn.Linear(model_dim, len(RGB_DIMENSIONS))
        nn.init.zeros_(self.rgb_velocity_head.weight)
        nn.init.zeros_(self.rgb_velocity_head.bias)

    def _text_condition(
        self,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        null = self.null_text_condition.to(reference)
        if text_condition is None:
            if drop_condition is not None:
                raise ValueError("drop_condition requires a text condition")
            return null
        if text_condition.ndim == 2:
            if text_condition.shape[0] != 1:
                raise ValueError("appearance adapter currently expects batch size one")
            text_condition = text_condition[0]
        if text_condition.shape != (self.text_dim,):
            raise ValueError(f"text condition must have shape ({self.text_dim},)")
        projected = self.text_projection(text_condition.to(reference))
        if drop_condition is None:
            return projected
        if drop_condition.shape != (1,) or drop_condition.dtype != torch.bool:
            raise ValueError("drop_condition must have one boolean value")
        return torch.where(drop_condition[0], null, projected)

    def forward(
        self,
        context_raster: torch.Tensor,
        noisy_values: torch.Tensor,
        base_velocity: torch.Tensor,
        flow_time: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None = None,
        guide_raster: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_values.ndim != 2 or noisy_values.shape[1] != self.feature_dim:
            raise ValueError("noisy values must have shape (N,feature_dim)")
        if base_velocity.shape != noisy_values.shape:
            raise ValueError("base velocity must match noisy values")
        if context_raster.shape != (self.grid_spec.n_cells, self.context_dim):
            raise ValueError("context raster does not match the adapter grid")
        if cell_indices.shape != (len(noisy_values),) or slot_indices.shape != (
            len(noisy_values),
        ):
            raise ValueError("cell and slot indices must align with noisy values")
        if cell_indices.dtype != torch.long or slot_indices.dtype != torch.long:
            raise ValueError("cell and slot indices must be int64")
        if len(cell_indices) and (
            int(cell_indices.min()) < 0
            or int(cell_indices.max()) >= self.grid_spec.n_cells
        ):
            raise ValueError("cell indices exceed the adapter grid")
        if len(slot_indices) and (
            int(slot_indices.min()) < 0
            or int(slot_indices.max()) >= self.grid_spec.slots_per_cell
        ):
            raise ValueError("slot indices exceed the adapter rank capacity")
        if flow_time.ndim == 0:
            flow_time = flow_time[None]
        if flow_time.shape != (1,):
            raise ValueError("flow time must contain one value")
        if guide_raster is None:
            guide_raster = noisy_values.new_zeros(
                self.grid_spec.n_cells, self.guide_dim
            )
        if guide_raster.shape != (self.grid_spec.n_cells, self.guide_dim):
            raise ValueError("guide raster does not match the adapter grid")
        gu, gv, gt = self.grid_spec.shape
        t_index = cell_indices % gt
        v_index = (cell_indices // gt) % gv
        u_index = cell_indices // (gv * gt)
        hidden = (
            self.mark_projection(noisy_values)
            + self.base_velocity_projection(base_velocity)
            + self.context_projection(context_raster[cell_indices])
            + self.guide_projection(guide_raster[cell_indices])
            + self.rank_projection(
                _rank_basis(
                    slot_indices,
                    self.grid_spec.slots_per_cell,
                    noisy_values.dtype,
                )
            )
            + self.u_embedding[u_index]
            + self.v_embedding[v_index]
            + self.t_embedding[t_index]
            + self._text_condition(
                text_condition, drop_condition, reference=noisy_values
            )[None]
            + self.time_mlp(timestep_embedding(flow_time, self.model_dim))[0][None]
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.rgb_velocity_head(hidden)


def top_fraction_cell_gate(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return a binary gate for an exact top fraction of finite cell scores."""
    if scores.ndim != 1 or not len(scores):
        raise ValueError("cell scores must be a non-empty vector")
    if not torch.isfinite(scores).all():
        raise ValueError("cell scores must be finite")
    if not 0 < fraction <= 1:
        raise ValueError("gate fraction must lie inside (0,1]")
    if fraction == 1:
        return torch.ones_like(scores)
    count = max(1, round(len(scores) * fraction))
    selected = scores.topk(count).indices
    gate = torch.zeros_like(scores)
    gate[selected] = 1
    return gate


def apply_scaffold_rgb_residual(
    base_velocity: torch.Tensor,
    rgb_residual: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply addressed RGB residuals while copying all non-RGB velocities."""
    if base_velocity.ndim != 2 or base_velocity.shape[1] != 22:
        raise ValueError("base velocity must have shape (N,22)")
    if rgb_residual.shape != (len(base_velocity), len(RGB_DIMENSIONS)):
        raise ValueError("RGB residual must have shape (N,3)")
    if cell_indices.shape != (len(base_velocity),) or cell_indices.dtype != torch.long:
        raise ValueError("one int64 cell index is required per velocity")
    if cell_weights.ndim != 1 or (
        len(cell_indices) and int(cell_indices.max()) >= len(cell_weights)
    ):
        raise ValueError("cell weights do not cover the addressed cells")
    corrected = base_velocity.clone()
    corrected[:, RGB_DIMENSIONS] = (
        base_velocity[:, RGB_DIMENSIONS]
        + cell_weights.to(base_velocity)[cell_indices, None]
        * rgb_residual.to(base_velocity)
    )
    return corrected


def appearance_feature_loss(
    corrected_velocity: torch.Tensor,
    expected_velocity: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_gate: torch.Tensor,
    cell_saliency: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score corrected RGB velocity on gate-selected jewels."""
    if corrected_velocity.shape != expected_velocity.shape or (
        corrected_velocity.ndim != 2 or corrected_velocity.shape[1] != 22
    ):
        raise ValueError("appearance feature loss expects matching (N,22) values")
    if cell_indices.shape != (len(corrected_velocity),):
        raise ValueError("appearance feature loss requires one cell per mark")
    if cell_gate.ndim != 1 or (
        len(cell_indices) and int(cell_indices.max()) >= len(cell_gate)
    ):
        raise ValueError("appearance gate does not cover addressed cells")
    selected = cell_gate.to(corrected_velocity)[cell_indices] > 0
    if not bool(selected.any()):
        raise ValueError("appearance gate selects no jewels")
    error = (
        corrected_velocity[selected][:, RGB_DIMENSIONS].float()
        - expected_velocity[selected][:, RGB_DIMENSIONS].float()
    ).square().mean(dim=1)
    if cell_saliency is None:
        return error.mean()
    if cell_saliency.shape != cell_gate.shape:
        raise ValueError("cell saliency and gate must share a shape")
    weights = cell_saliency.to(error)[cell_indices[selected]].clamp_min(0)
    return (error * weights).sum() / weights.sum().clamp_min(1e-8)


@dataclass(frozen=True)
class AppearanceAdaptedSample:
    """Matched base and RGB-adapted standardized samples."""

    base: torch.Tensor
    appearance: torch.Tensor
    max_non_rgb_error: float


def _validate_compatible_models(
    base_flow: BirthMarkFlowModel, adapter: ScaffoldAppearanceAdapter
) -> None:
    if base_flow.feature_dim != adapter.feature_dim:
        raise ValueError("base flow and adapter use different feature dimensions")
    if base_flow.grid_spec != adapter.grid_spec:
        raise ValueError("base flow and adapter use different grids")
    if base_flow.text_dim != adapter.text_dim:
        raise ValueError("base flow and adapter use different text dimensions")
    if base_flow.guide_dim != adapter.guide_dim:
        raise ValueError("base flow and adapter use different guide dimensions")


def _guided_base_velocity(
    base_flow: BirthMarkFlowModel,
    context_raster: torch.Tensor,
    state: torch.Tensor,
    flow_time: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    cfg_scale: float,
    guide_raster: torch.Tensor | None,
) -> torch.Tensor:
    conditioned = base_flow(
        context_raster,
        state,
        flow_time,
        cell_indices,
        slot_indices,
        text_condition,
        guide_raster=guide_raster,
    )
    if text_condition is None or cfg_scale == 1:
        return conditioned
    unconditioned = base_flow(
        context_raster,
        state,
        flow_time,
        cell_indices,
        slot_indices,
        None,
        guide_raster=guide_raster,
    )
    return unconditioned + cfg_scale * (conditioned - unconditioned)


@torch.no_grad()
def sample_appearance_adapted_birth_marks(
    base_flow: BirthMarkFlowModel,
    adapter: ScaffoldAppearanceAdapter,
    base_context_raster: torch.Tensor,
    appearance_context_raster: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    *,
    cell_weights: torch.Tensor,
    steps: int = 20,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
    guide_raster: torch.Tensor | None = None,
) -> AppearanceAdaptedSample:
    """Euler-sample a base and RGB-only adapted stream from shared noise."""
    _validate_compatible_models(base_flow, adapter)
    if steps <= 0 or cfg_scale < 0:
        raise ValueError("sampling steps must be positive and CFG scale non-negative")
    if cell_weights.shape != (base_flow.grid_spec.n_cells,):
        raise ValueError("cell weights must contain one value per scaffold cell")
    device = base_context_raster.device
    base_state = torch.randn(
        len(cell_indices),
        base_flow.feature_dim,
        device=device,
        generator=generator,
    )
    appearance_state = base_state.clone()
    times = torch.linspace(0, 1, steps + 1, device=device)
    base_was_training = base_flow.training
    adapter_was_training = adapter.training
    base_flow.eval()
    adapter.eval()
    maximum_error = 0.0
    for index in range(steps):
        time = times[index : index + 1]
        base_velocity = _guided_base_velocity(
            base_flow,
            base_context_raster,
            base_state,
            time,
            cell_indices,
            slot_indices,
            text_condition,
            cfg_scale,
            guide_raster,
        )
        residual = adapter(
            appearance_context_raster,
            appearance_state,
            base_velocity,
            time,
            cell_indices,
            slot_indices,
            text_condition,
            guide_raster=guide_raster,
        )
        if text_condition is not None and cfg_scale != 1:
            unconditioned_residual = adapter(
                appearance_context_raster,
                appearance_state,
                base_velocity,
                time,
                cell_indices,
                slot_indices,
                None,
                guide_raster=guide_raster,
            )
            residual = unconditioned_residual + cfg_scale * (
                residual - unconditioned_residual
            )
        step_size = times[index + 1] - times[index]
        base_next = base_state + step_size * base_velocity
        weighted_residual = cell_weights.to(residual)[cell_indices, None] * residual
        appearance_next = base_next.clone()
        appearance_rgb = (
            appearance_state[:, RGB_DIMENSIONS]
            + step_size * base_velocity[:, RGB_DIMENSIONS]
        )
        appearance_next[:, RGB_DIMENSIONS] = (
            appearance_rgb + step_size * weighted_residual
        )
        if len(base_next):
            error = float(
                (
                    base_next[:, NON_RGB_DIMENSIONS]
                    - appearance_next[:, NON_RGB_DIMENSIONS]
                )
                .abs()
                .max()
            )
            maximum_error = max(maximum_error, error)
        base_state = base_next
        appearance_state = appearance_next
    if base_was_training:
        base_flow.train()
    if adapter_was_training:
        adapter.train()
    if maximum_error:
        raise RuntimeError("appearance adapter modified a non-RGB feature")
    return AppearanceAdaptedSample(base_state, appearance_state, maximum_error)
