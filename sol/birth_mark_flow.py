"""Conditional rectified flow over jewel birth marks with externally owned topology."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sol.latent_prior import timestep_embedding
from sol.splat_density import temporal_standard_deviation
from sol.streaming_model import ContextRasterEncoder, ResidualMLP, _rank_basis
from sol.token_grid import GridSpec


def rasterize_noisy_marks(
    values: torch.Tensor, cell_indices: torch.Tensor, n_cells: int
) -> torch.Tensor:
    """Pool one noisy variable-cardinality mark set into local raster statistics."""
    if values.ndim != 2 or values.shape[1] != 22:
        raise ValueError("noisy marks must have shape (N,22)")
    if cell_indices.shape != (len(values),) or cell_indices.dtype != torch.long:
        raise ValueError("cell_indices must be one int64 value per mark")
    if n_cells <= 0 or (len(cell_indices) and int(cell_indices.max()) >= n_cells):
        raise ValueError("cell indices exceed the declared raster")
    total = values.new_zeros(n_cells, values.shape[1])
    square = torch.zeros_like(total)
    count = values.new_zeros(n_cells, 1)
    expanded = cell_indices[:, None].expand_as(values)
    total.scatter_add_(0, expanded, values)
    square.scatter_add_(0, expanded, values.square())
    count.scatter_add_(0, cell_indices[:, None], values.new_ones(len(values), 1))
    mean = total / count.clamp_min(1)
    variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
    return torch.cat((mean, variance, count.log1p(), (count > 0).to(values)), dim=1)


class BirthMarkFlowModel(nn.Module):
    """Generate correlated birth marks while cell/rank topology stays explicit."""

    def __init__(
        self,
        *,
        feature_dim: int = 22,
        context_dim: int = 46,
        model_dim: int = 64,
        grid_spec: GridSpec = GridSpec((16, 16, 8), 512),
        context_depth: int = 2,
        noisy_depth: int = 2,
        guide_depth: int = 2,
        cell_depth: int = 2,
        mark_depth: int = 3,
        text_dim: int = 512,
        guide_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim != 22:
            raise ValueError("the canonical birth-mark contract has 22 features")
        if model_dim % 8:
            raise ValueError("model_dim must be divisible by eight")
        if text_dim <= 0 or guide_dim < 0:
            raise ValueError("text_dim must be positive and guide_dim non-negative")
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.grid_spec = grid_spec
        self.text_dim = text_dim
        self.guide_dim = guide_dim
        self.context_encoder = ContextRasterEncoder(
            context_dim, model_dim, grid_spec.shape, context_depth
        )
        self.noisy_encoder = ContextRasterEncoder(
            feature_dim * 2 + 2, model_dim, grid_spec.shape, noisy_depth
        )
        self.guide_encoder = (
            ContextRasterEncoder(guide_dim, model_dim, grid_spec.shape, guide_depth)
            if guide_dim
            else None
        )
        gu, gv, gt = grid_spec.shape
        self.u_embedding = nn.Parameter(torch.randn(gu, model_dim) * 0.02)
        self.v_embedding = nn.Parameter(torch.randn(gv, model_dim) * 0.02)
        self.t_embedding = nn.Parameter(torch.randn(gt, model_dim) * 0.02)
        self.text_projection = nn.Linear(text_dim, model_dim)
        self.null_text_condition = nn.Parameter(torch.zeros(model_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.cell_blocks = nn.ModuleList(
            ResidualMLP(model_dim) for _ in range(cell_depth)
        )
        self.rank_projection = nn.Linear(8, model_dim)
        self.noisy_mark_projection = nn.Linear(feature_dim, model_dim)
        self.mark_blocks = nn.ModuleList(
            ResidualMLP(model_dim) for _ in range(mark_depth)
        )
        self.velocity_head = nn.Linear(model_dim, feature_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

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
                raise ValueError("birth mark flow currently expects batch size one")
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
        flow_time: torch.Tensor,
        cell_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        text_condition: torch.Tensor | None,
        drop_condition: torch.Tensor | None = None,
        guide_raster: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_values.ndim != 2 or noisy_values.shape[1] != self.feature_dim:
            raise ValueError("noisy values must have shape (N,feature_dim)")
        if cell_indices.shape != (len(noisy_values),) or slot_indices.shape != (
            len(noisy_values),
        ):
            raise ValueError("cell and slot indices must align with noisy values")
        if flow_time.ndim == 0:
            flow_time = flow_time[None]
        if flow_time.shape != (1,):
            raise ValueError("flow time must contain one value")
        context = self.context_encoder(context_raster, spatial=True)
        noisy_raster = rasterize_noisy_marks(
            noisy_values, cell_indices, self.grid_spec.n_cells
        )
        noisy_context = self.noisy_encoder(noisy_raster, spatial=True)
        if self.guide_encoder is None:
            if guide_raster is not None:
                raise ValueError("model was constructed without video guidance")
            guide_context: torch.Tensor | float = 0.0
        else:
            if guide_raster is None:
                guide_raster = noisy_values.new_zeros(
                    self.grid_spec.n_cells, self.guide_dim
                )
            guide_context = self.guide_encoder(guide_raster, spatial=True)
        gu, gv, gt = self.grid_spec.shape
        position = (
            self.u_embedding[:, None, None]
            + self.v_embedding[None, :, None]
            + self.t_embedding[None, None, :]
        ).reshape(self.grid_spec.n_cells, -1)
        cells = context + noisy_context + guide_context + position[None]
        text = self._text_condition(
            text_condition, drop_condition, reference=noisy_values
        )
        time = self.time_mlp(timestep_embedding(flow_time, self.model_dim))[0]
        cells = cells + text[None, None] + time[None, None]
        for block in self.cell_blocks:
            cells = block(cells)
        hidden = (
            cells[0, cell_indices]
            + self.rank_projection(
                _rank_basis(
                    slot_indices, self.grid_spec.slots_per_cell, noisy_values.dtype
                )
            )
            + self.noisy_mark_projection(noisy_values)
        )
        for block in self.mark_blocks:
            hidden = block(hidden)
        return self.velocity_head(hidden)


def birth_mark_flow_objective(
    model: BirthMarkFlowModel,
    context_raster: torch.Tensor,
    target_values: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    *,
    noise: torch.Tensor,
    flow_time: torch.Tensor,
    drop_condition: torch.Tensor | None = None,
    guide_raster: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score one explicit noise-to-mark path with topology held fixed."""
    if noise.shape != target_values.shape:
        raise ValueError("noise must match target values")
    if flow_time.ndim == 0:
        flow_time = flow_time[None]
    if flow_time.shape != (1,):
        raise ValueError("flow time must contain one value")
    noised = (1 - flow_time) * noise + flow_time * target_values
    expected_velocity = target_values - noise
    predicted = model(
        context_raster,
        noised,
        flow_time,
        cell_indices,
        slot_indices,
        text_condition,
        drop_condition,
        guide_raster,
    )
    return F.mse_loss(predicted.float(), expected_velocity.float())


@torch.no_grad()
def sample_birth_marks(
    model: BirthMarkFlowModel,
    context_raster: torch.Tensor,
    cell_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    text_condition: torch.Tensor | None,
    *,
    steps: int = 20,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
    guide_raster: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler-sample standardized marks for one externally supplied topology."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = context_raster.device
    state = torch.randn(
        len(cell_indices),
        model.feature_dim,
        device=device,
        generator=generator,
    )
    times = torch.linspace(0, 1, steps + 1, device=device)
    was_training = model.training
    model.eval()
    for index in range(steps):
        time = times[index : index + 1]
        conditioned = model(
            context_raster,
            state,
            time,
            cell_indices,
            slot_indices,
            text_condition,
            guide_raster=guide_raster,
        )
        if text_condition is not None and cfg_scale != 1.0:
            unconditioned = model(
                context_raster,
                state,
                time,
                cell_indices,
                slot_indices,
                None,
                guide_raster=guide_raster,
            )
            velocity = unconditioned + cfg_scale * (conditioned - unconditioned)
        else:
            velocity = conditioned
        state = state + (times[index + 1] - times[index]) * velocity
    if was_training:
        model.train()
    return state


def project_birth_topology(
    local_features: torch.Tensor,
    cell_indices: torch.Tensor,
    *,
    spec: GridSpec,
    support_sigma: float,
    stride_frames: int,
    covariance_chunk: int = 4096,
) -> torch.Tensor:
    """Clamp sampled centers and finite-support starts to their declared birth cells."""
    if local_features.ndim != 2 or local_features.shape[1] != 22:
        raise ValueError("local features must have shape (N,22)")
    if cell_indices.shape != (len(local_features),):
        raise ValueError("every feature requires one cell index")
    if support_sigma <= 0 or stride_frames <= 0 or covariance_chunk <= 0:
        raise ValueError("support sigma, stride, and covariance chunk must be positive")
    if not len(local_features):
        return local_features.clone()
    output = local_features.clone()
    gu, gv, gt = spec.shape
    t = cell_indices % gt
    v = (cell_indices // gt) % gv
    u = cell_indices // (gv * gt)
    coordinates = torch.stack((u, v), dim=1).to(output)
    size = output.new_tensor((2 / gu, 2 / gv))
    lower = -1 + coordinates * size
    upper = lower + size
    epsilon = torch.finfo(output.dtype).eps * 16
    lower_bound = lower + epsilon
    upper_bound = upper - epsilon
    lower_bound[:, 0] = torch.where(
        u == 0, output.new_full((len(output),), -torch.inf), lower_bound[:, 0]
    )
    upper_bound[:, 0] = torch.where(
        u == gu - 1,
        output.new_full((len(output),), torch.inf),
        upper_bound[:, 0],
    )
    lower_bound[:, 1] = torch.where(
        v == 0, output.new_full((len(output),), -torch.inf), lower_bound[:, 1]
    )
    upper_bound[:, 1] = torch.where(
        v == gv - 1,
        output.new_full((len(output),), torch.inf),
        upper_bound[:, 1],
    )
    output[:, :2] = torch.maximum(
        lower_bound, torch.minimum(output[:, :2], upper_bound)
    )
    temporal_sigma = torch.cat(
        [
            temporal_standard_deviation(output[start : start + covariance_chunk])
            for start in range(0, len(output), covariance_chunk)
        ]
    )
    support_start = output[:, 2] - support_sigma * temporal_sigma
    first_frame = torch.div(
        t * stride_frames + gt - 1, gt, rounding_mode="floor"
    )
    stop_frame = torch.div(
        (t + 1) * stride_frames + gt - 1, gt, rounding_mode="floor"
    )
    last_frame = stop_frame - 1
    lower_t = (first_frame.to(output) - 1) / stride_frames + epsilon
    upper_t = last_frame.to(output) / stride_frames
    projected_start = torch.maximum(
        lower_t, torch.minimum(support_start, upper_t)
    )
    output[:, 2] += projected_start - support_start
    return output
