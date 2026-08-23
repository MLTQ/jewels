"""Measure renderer-responsibility targets and gradients without optimizing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from sol.factorized_structural_encoder import FactorizedStructuralJewelEncoder
from sol.local_teacher_distillation import (
    extract_local_teacher_attributes,
    responsibility_teacher_moment_losses,
)
from sol.structural_encoder import render_structural
from sol.token_grid import GridSpec
from sol.train_amortized_encoder import sample_voxels
from stprim.data.video_io import load_video


def parameter_gradient_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    """Return the joint L2 norm of a loss gradient over one parameter group."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    total = sum(
        gradient.detach().float().square().sum()
        for gradient in gradients if gradient is not None
    )
    return math.sqrt(float(total)) if not isinstance(total, int) else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-sample", type=int, default=4000)
    parser.add_argument("--student-sample", type=int, default=1024)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--support-sigma", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--size-offset", type=float, default=0.0)
    parser.add_argument("--target-active-fraction", type=float, default=0.58)
    parser.add_argument("--support-capacity", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cpu_generator = torch.Generator().manual_seed(args.seed + 100003)
    manifest = json.loads(Path(args.manifest).read_text())
    example = next(
        item for item in manifest["examples"] if item["source_id"] == args.source_id
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model = FactorizedStructuralJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024),
        **meta["model_args"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    video = load_video(
        example["video"],
        max_frames=int(example["frames"]),
        start_frame=int(example.get("start_frame", 0)),
        resize=(meta["train_args"]["height"], meta["train_args"]["width"]),
        device="cpu",
    ).to(device)

    fit_path = None
    for root in args.checkpoint_root:
        for candidate in Path(root).glob("*.pt"):
            saved = torch.load(candidate, map_location="cpu", weights_only=False)
            if saved.get("source", {}).get("source_id") == args.source_id:
                fit_path = candidate
                fitted = saved
                break
        if fit_path is not None:
            break
    if fit_path is None:
        raise ValueError(f"no fitted teacher for {args.source_id}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stprim"))
    from prior.featurize import state_to_features  # noqa: PLC0415

    teacher = extract_local_teacher_attributes(
        state_to_features(fitted["state"]).float(),
        args.teacher_sample,
        cpu_generator,
        sampling="active_uniform",
    ).to(device)
    prediction = model(video)
    opacity = torch.sigmoid(prediction["logit_w"])
    count = min(args.student_sample, len(opacity))
    subset = torch.multinomial(
        opacity.clamp_min(1e-6), count,
        replacement=False, generator=generator,
    )
    from sol.distill_structural_encoder import principal_axis  # noqa: PLC0415

    log_scale = prediction["log_scale"][subset]
    losses, targets = responsibility_teacher_moment_losses(
        student_centers=prediction["centers"][subset],
        student_log_scale=log_scale,
        student_axis=principal_axis(prediction["quaternion"][subset], log_scale),
        student_opacity=opacity[subset],
        student_colors=prediction["colors"][subset],
        student_color_grads=prediction["color_grads"][subset],
        teacher=teacher,
        support_sigma=args.support_sigma,
        temperature=args.temperature,
        size_offset=args.size_offset,
        opacity_mass_ratio=(
            teacher.active_count / (model.n_jewels * args.target_active_fraction)
        ),
    )
    points, target = sample_voxels(video, args.points, generator)
    rendered = render_structural(
        prediction, points, point_chunk=256,
        cull_mode="support_tiled", support_capacity=args.support_capacity,
    )
    losses = {"render": torch.nn.functional.mse_loss(rendered, target), **losses}
    geometry = list(model.geometry_trunk.parameters()) + list(
        model.geometry_head.parameters()
    )
    appearance = (
        list(model.appearance_fine.parameters())
        + list(model.appearance_coarse.parameters())
        + list(model.appearance_head.parameters())
        + list(model.background_head.parameters())
    )
    report = {
        "source_id": args.source_id,
        "checkpoint": args.checkpoint,
        "teacher_checkpoint": str(fit_path),
        "teacher_sample": len(teacher.centers),
        "student_sample": count,
        "support_sigma": args.support_sigma,
        "temperature": args.temperature,
        "losses": {},
        "targets": {
            "support_count_mean": float(targets.support_count.mean()),
            "support_count_min": float(targets.support_count.min()),
            "effective_count_mean": float(targets.effective_count.mean()),
            "effective_count_median": float(targets.effective_count.median()),
            "fallback_fraction": float(targets.used_fallback.float().mean()),
            "color_clip_fraction": float(
                ((targets.colors < 0) | (targets.colors > 1)).float().mean()
            ),
            "gradient_clip_fraction": float(
                (targets.color_grads.abs() > 0.25).float().mean()
            ),
            "scale_clip_fraction": float(
                (
                    ((targets.log_scale + args.size_offset) < -9)
                    | ((targets.log_scale + args.size_offset) > 1)
                ).float().mean()
            ),
        },
    }
    for name, loss in losses.items():
        report["losses"][name] = {
            "value": float(loss.detach()),
            "geometry_gradient": parameter_gradient_norm(
                loss, geometry, retain_graph=True
            ),
            "appearance_gradient": parameter_gradient_norm(
                loss, appearance, retain_graph=True
            ),
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
