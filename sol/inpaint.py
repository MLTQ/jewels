"""Classifier-free guided flow sampling with exact clean-cell clamping."""

from __future__ import annotations

from collections.abc import Callable

import torch


VelocityModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@torch.no_grad()
def masked_flow_inpaint(
    model: VelocityModel,
    known_latents: torch.Tensor,
    dirty_mask: torch.Tensor,
    *,
    condition: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    steps: int = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Resample dirty raster cells while preserving every clean cell exactly.

    The velocity model has the same conceptual condition slot as the existing
    CLIP-conditioned prior. It may additionally encode protected moved jewels
    into that condition before this function is called.
    """
    if known_latents.ndim != 3:
        raise ValueError("known_latents must have shape (B,C,D)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if dirty_mask.ndim == 1:
        dirty_mask = dirty_mask[None].expand(known_latents.shape[0], -1)
    if dirty_mask.shape != known_latents.shape[:2]:
        raise ValueError("dirty_mask must have shape (C,) or (B,C)")
    if condition is not None and condition.shape[0] != known_latents.shape[0]:
        raise ValueError("condition batch must match known_latents")
    dirty = dirty_mask.to(device=known_latents.device, dtype=torch.bool)[..., None]
    noise = torch.randn(
        known_latents.shape,
        device=known_latents.device,
        dtype=known_latents.dtype,
        generator=generator,
    )
    state = torch.where(dirty, noise, known_latents)
    times = torch.linspace(0, 1, steps + 1, device=known_latents.device)
    for index in range(steps):
        time = times[index].expand(known_latents.shape[0])
        model_args = {"edit_mask": dirty_mask} if getattr(
            model, "mask_conditioning", False
        ) else {}
        if condition is not None and cfg_scale != 1.0:
            conditioned = model(state, time, condition, **model_args)
            unconditioned = model(state, time, None, **model_args)
            velocity = unconditioned + cfg_scale * (conditioned - unconditioned)
        else:
            velocity = model(state, time, condition, **model_args)
        if velocity.shape != state.shape:
            raise ValueError("velocity model must return the latent input shape")
        state = state + (times[index + 1] - times[index]) * velocity
        state = torch.where(dirty, state, known_latents)
    return state
