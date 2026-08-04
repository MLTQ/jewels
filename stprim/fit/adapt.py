"""Adaptive primitive-count control: densify where error is high, prune dead.

GaussianImage's known weakness was a fixed primitive count; the follow-up work
added densification precisely because a fixed budget can't follow content. In a
spacetime volume this matters more than in an image: a static background wants
a handful of enormous primitives, a moving limb wants hundreds of small ones,
and the ratio between those regions is far more extreme than within a frame.

Positional-gradient magnitude is the densification signal: a primitive that
"wants" to be in several places at once is one that should become several
primitives.
"""

from __future__ import annotations

import torch

from core.params import PrimitiveField


class GradientTracker:
    """Accumulates |d(loss)/d(mu)| per primitive between adaptation steps."""

    def __init__(self, n: int, device: str | torch.device) -> None:
        self.accum = torch.zeros(n, device=device)
        self.count = 0

    def update(self, field: PrimitiveField) -> None:
        if field.mu.grad is None:
            return
        g = field.mu.grad.norm(dim=-1)
        if g.shape[0] != self.accum.shape[0]:  # count changed under us
            self.reset(g.shape[0], g.device)
        self.accum += g
        self.count += 1

    def reset(self, n: int, device: str | torch.device) -> None:
        self.accum = torch.zeros(n, device=device)
        self.count = 0

    def mean(self) -> torch.Tensor:
        return self.accum / max(self.count, 1)


@torch.no_grad()
def adapt(
    field: PrimitiveField,
    tracker: GradientTracker,
    *,
    max_primitives: int,
    densify_frac: float = 0.05,
    prune_weight: float = 0.01,
    split_shrink: float = 1.6,
    split_mode: str = "isotropic",
    jitter: float = 0.5,
    generator: torch.Generator | None = None,
) -> dict[str, int]:
    """Prune low-weight primitives, then split the highest-gradient survivors.

    Returns a small stats dict for logging. The caller MUST rebuild the
    optimizer afterwards — parameter tensors are replaced, so existing Adam
    moment buffers no longer correspond to them.
    """
    if split_mode not in {"isotropic", "spatial"}:
        raise ValueError(f"unsupported split mode: {split_mode}")
    stats = {"pruned": 0, "split": 0, "n": len(field)}

    # --- prune -------------------------------------------------------- #
    w = field.weights()
    keep = w > prune_weight
    if keep.sum() < 8:  # never prune into oblivion
        keep = torch.ones_like(keep)
    if (~keep).any():
        grad_mean = tracker.mean()[keep]
        field.subset_(keep)
        stats["pruned"] = int((~keep).sum())
    else:
        grad_mean = tracker.mean()

    # --- densify ------------------------------------------------------ #
    n = len(field)
    room = max_primitives - n
    if room > 0 and densify_frac > 0:
        n_split = min(room, max(1, int(n * densify_frac)))
        if grad_mean.shape[0] != n:  # tracker was stale
            grad_mean = torch.zeros(n, device=field.device)
        sel = grad_mean.topk(min(n_split, n)).indices

        scale = field.scales()[sel]
        if generator is None:
            noise = torch.randn_like(scale)
        else:
            # One CPU random stream makes exact recovery independent of the
            # accelerator's RNG implementation.
            noise = torch.randn(
                scale.shape,
                dtype=scale.dtype,
                device=generator.device,
                generator=generator,
            ).to(scale.device)
        if split_mode == "spatial":
            rotations = field.rotations()[sel]
            temporal_axis = rotations[:, 2].abs().argmax(dim=-1)
            spatial_axes = torch.ones_like(scale, dtype=torch.bool)
            spatial_axes.scatter_(1, temporal_axis[:, None], False)
            local_offset = noise * scale * jitter * spatial_axes
            offset = torch.einsum("nij,nj->ni", rotations, local_offset)
            shrink = torch.where(
                spatial_axes,
                torch.full_like(scale, 2.0**0.5),
                torch.ones_like(scale),
            )
            log_shrink = shrink.log()
        else:
            offset = noise * scale * jitter
            log_shrink = torch.log(
                torch.tensor(split_shrink, device=field.device)
            )

        new = {
            "mu": (field.mu[sel] + offset).clone(),
            "log_scale": (field.log_scale[sel] - log_shrink).clone(),
            "quat": field.quat[sel].clone(),
            "color": field.color[sel].clone(),
            "logit_w": field.logit_w[sel].clone(),
        }
        if field.p1_color:
            new["color_grad"] = field.color_grad[sel].clone()

        # shrink the parents too, so total energy is roughly conserved
        field.log_scale[sel] -= log_shrink
        field.append_(new)
        stats["split"] = int(sel.shape[0])

    stats["n"] = len(field)
    tracker.reset(len(field), field.device)
    return stats
