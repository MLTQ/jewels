"""Read-only gradient calibration for frozen-geometry appearance objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.appearance_objective import (
    appearance_objective,
    range_diagnostics,
    range_excess_loss,
    residual_energy,
)
from sol.audit_irregular_encoder import load_candidate
from sol.distill_structural_encoder import appearance_frame_indices
from sol.factorized_structural_encoder import FactorizedStructuralJewelEncoder
from sol.render_streaming_continuation import frame_points
from sol.structural_encoder import render_structural
from sol.train_amortized_encoder import sample_voxels
from stprim.data.video_io import load_video


def gradient_l2_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    """Return the joint L2 norm without accumulating `.grad` or taking a step."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    square = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            square = square + gradient.square().sum()
    return float(square.sqrt().detach())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--grid-frames", type=int, default=4)
    parser.add_argument("--grid-height", type=int, default=48)
    parser.add_argument("--grid-width", type=int, default=72)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--support-capacity", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(
        args.height,
        args.width,
        args.grid_frames,
        args.grid_height,
        args.grid_width,
        args.points,
        args.point_chunk,
        args.support_capacity,
    ) <= 0:
        raise ValueError("calibration dimensions and capacities must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    matches = [
        item for item in manifest["examples"] if item["source_id"] == args.source_id
    ]
    if len(matches) != 1:
        raise ValueError("calibration source must identify exactly one manifest example")
    item = matches[0]
    video = load_video(
        item["video"],
        max_frames=int(item["frames"]),
        start_frame=int(item.get("start_frame", 0)),
        resize=(args.height, args.width),
        device="cpu",
    ).to(device)
    model = load_candidate(Path(args.checkpoint), device)
    if not isinstance(model, FactorizedStructuralJewelEncoder):
        raise ValueError("calibration requires a factorized checkpoint")
    if model.appearance_contract != "residual":
        raise ValueError("calibration requires the residual appearance contract")
    model.freeze_geometry()
    model.train()
    prediction = model(video)
    points, target = sample_voxels(video, args.points, generator)
    rendered = render_structural(
        prediction,
        points,
        point_chunk=args.point_chunk,
        cull_mode="support_tiled",
        support_capacity=args.support_capacity,
    )
    sampled_render = torch.nn.functional.mse_loss(rendered, target)
    indices = appearance_frame_indices(
        len(video), args.grid_frames, args.seed,
        contiguous=True,
    )
    grid_points = frame_points(
        len(video), indices,
        args.grid_height, args.grid_width,
        device=device,
    )
    grid_rendered = render_structural(
        prediction,
        grid_points,
        point_chunk=args.point_chunk,
        cull_mode="support_tiled",
        support_capacity=args.support_capacity,
    ).reshape(args.grid_frames, args.grid_height, args.grid_width, 3)
    grid_target = torch.nn.functional.interpolate(
        video[indices].permute(0, 3, 1, 2),
        size=(args.grid_height, args.grid_width),
        mode="bilinear",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    terms = appearance_objective(
        grid_rendered,
        grid_target,
        rgb_weight=1,
        spatial_weight=1,
        temporal_weight=1,
        structure_weight=1,
        range_weight=1,
    )
    sampled_range = range_excess_loss(rendered)
    residual_color, residual_gradient = residual_energy(
        prediction["appearance_residual"]
    )
    components = {
        "sampled_render": sampled_render,
        "grid_rgb": terms.rgb,
        "grid_spatial": terms.spatial,
        "grid_temporal": terms.temporal,
        "grid_structure": terms.structure,
        "grid_range": terms.range,
        "sampled_range": sampled_range,
        "residual_color": residual_color,
        "residual_gradient": residual_gradient,
    }
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    gradient_norms = {}
    names = list(components)
    for index, name in enumerate(names):
        gradient_norms[name] = gradient_l2_norm(
            components[name],
            parameters,
            retain_graph=index < len(names) - 1,
        )
    rendered_diagnostics = range_diagnostics(rendered)
    grid_diagnostics = range_diagnostics(grid_rendered)
    report = {
        "schema": "frozen-appearance-calibration-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_id": args.source_id,
        "seed": args.seed,
        "frame_indices": indices.detach().cpu().tolist(),
        "losses": {name: float(value.detach()) for name, value in components.items()},
        "appearance_gradient_l2": gradient_norms,
        "sampled_render_range": {
            name: float(value.detach()) for name, value in rendered_diagnostics.items()
        },
        "grid_render_range": {
            name: float(value.detach()) for name, value in grid_diagnostics.items()
        },
        "geometry_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("geometry_") and parameter.requires_grad
        ),
        "appearance_trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
