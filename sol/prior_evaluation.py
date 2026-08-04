"""Leakage-safe latent-flow diagnostics and retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.latent_prior import RasterFlowPrior, sample_flow


@dataclass(frozen=True)
class PriorEvaluation:
    conditional_flow_mse: float
    shuffled_flow_mse: float
    unconditional_flow_mse: float
    conditional_gain: float
    scene_mean_latent_mse: float
    clip_retrieval_latent_mse: float
    sampled_latent_mse: float
    scene_mean_energy: float
    clip_retrieval_energy: float
    sampled_energy: float
    sampled_examples: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def energy_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Empirical energy distance on flattened tensors, scaled per feature."""
    if first.ndim < 2 or second.ndim != first.ndim:
        raise ValueError("energy inputs must have matching ranks with a sample axis")
    if first.shape[1:] != second.shape[1:]:
        raise ValueError("energy input feature shapes must match")
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    scale = first_flat.shape[1] ** 0.5
    cross = torch.cdist(first_flat, second_flat).mean() / scale
    within_first = torch.cdist(first_flat, first_flat).mean() / scale
    within_second = torch.cdist(second_flat, second_flat).mean() / scale
    return float(2 * cross - within_first - within_second)


@torch.no_grad()
def evaluate_prior(
    model: RasterFlowPrior,
    train_latents: torch.Tensor,
    train_conditions: torch.Tensor,
    validation_latents: torch.Tensor,
    validation_conditions: torch.Tensor,
    *,
    device: torch.device | str,
    sample_steps: int = 25,
    sample_examples: int = 4,
    cfg_scale: float = 1.5,
    seed: int = 0,
    evaluation_batch: int = 2,
) -> PriorEvaluation:
    """Compare conditional flow to shuffled/unconditional paths and simple baselines."""
    if sample_steps <= 0 or sample_examples <= 0 or evaluation_batch <= 0:
        raise ValueError("evaluation counts must be positive")
    target_device = torch.device(device)
    train_latents = train_latents.float()
    train_conditions = train_conditions.float()
    validation_latents = validation_latents.float()
    validation_conditions = validation_conditions.float()
    generator = torch.Generator(device=target_device).manual_seed(seed)
    flow_totals = torch.zeros(3, dtype=torch.float64)
    flow_values = 0
    was_training = model.training
    model.eval()
    for start in range(0, len(validation_latents), evaluation_batch):
        targets = validation_latents[start : start + evaluation_batch].to(target_device)
        conditions = validation_conditions[start : start + evaluation_batch].to(target_device)
        batch = len(targets)
        noise = torch.randn(targets.shape, device=target_device, generator=generator)
        time = torch.rand(batch, device=target_device, generator=generator)
        noised = (1 - time[:, None, None]) * noise + time[:, None, None] * targets
        expected = targets - noise
        shuffled = validation_conditions.roll(1, dims=0)[
            start : start + evaluation_batch
        ].to(target_device)
        predictions = (
            model(noised, time, conditions),
            model(noised, time, shuffled),
            model(noised, time, None),
        )
        for index, prediction in enumerate(predictions):
            flow_totals[index] += float((prediction.float() - expected).square().sum())
        flow_values += expected.numel()

    scene_mean = train_latents.mean(dim=0, keepdim=True)
    scene_mean_mse = float((validation_latents - scene_mean).square().mean())
    similarities = validation_conditions @ train_conditions.T
    nearest = similarities.argmax(dim=1)
    retrieval_mse = float(
        (validation_latents - train_latents[nearest]).square().mean()
    )

    sample_count = min(sample_examples, len(validation_latents))
    picks = torch.linspace(0, len(validation_latents) - 1, sample_count).long()
    sample_losses = []
    generated_samples = []
    for pick in picks:
        condition = validation_conditions[pick : pick + 1].to(target_device)
        generated = sample_flow(
            model,
            condition,
            batch=1,
            n_cells=validation_latents.shape[1],
            latent_dim=validation_latents.shape[2],
            device=target_device,
            steps=sample_steps,
            cfg_scale=cfg_scale,
            generator=generator,
        ).cpu()
        sample_losses.append(
            (generated - validation_latents[pick : pick + 1]).square().mean()
        )
        generated_samples.append(generated[0])
    if was_training:
        model.train()
    conditional, shuffled, unconditional = (
        float(value / flow_values) for value in flow_totals
    )
    generated_tensor = torch.stack(generated_samples)
    selected_targets = validation_latents[picks]
    selected_retrieval = train_latents[nearest[picks]]
    selected_mean = scene_mean.expand_as(selected_targets)
    return PriorEvaluation(
        conditional_flow_mse=conditional,
        shuffled_flow_mse=shuffled,
        unconditional_flow_mse=unconditional,
        conditional_gain=unconditional - conditional,
        scene_mean_latent_mse=scene_mean_mse,
        clip_retrieval_latent_mse=retrieval_mse,
        sampled_latent_mse=float(torch.stack(sample_losses).mean()),
        scene_mean_energy=energy_distance(selected_mean, selected_targets),
        clip_retrieval_energy=energy_distance(selected_retrieval, selected_targets),
        sampled_energy=energy_distance(generated_tensor, selected_targets),
        sampled_examples=sample_count,
    )
