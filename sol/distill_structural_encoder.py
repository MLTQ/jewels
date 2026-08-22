"""Distill the fitter's field structure into the feed-forward structural encoder.

Render loss alone actively drives the field toward a uniform lattice (measured:
extent IQR 2.10 -> 1.35 and occupancy uniformity 0.9961 -> 0.9992 while PSNR
climbed). The fitter escapes that optimum only because densify/prune is an
explicit structural mechanism outside the loss. This supervises the student
against fitted fields directly, which is what PROMPTABLE_ROADMAP Phase 3 always
specified.

Two structural terms accompany the render loss:

* **Chamfer on positions, symmetric** — teacher->student forces students to
  cover regions the fitter densified (clustering); student->teacher keeps
  students off empty space.
* **Anisotropy spread matching** — each student matches its nearest teacher's
  log-eigenvalue spread, a scale-invariant shape descriptor, so the student
  learns tube-ness without inheriting a 72k-budget jewel's absolute size.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

from sol.compare_field_structure import structure_report
from sol.factorized_structural_encoder import (
    ARCHITECTURE as FACTORIZED_ARCHITECTURE,
    FactorizedStructuralJewelEncoder,
)
from sol.local_teacher_distillation import (
    LocalTeacherAttributes,
    extract_local_teacher_attributes,
    local_teacher_attribute_losses,
)
from sol.render_streaming_continuation import frame_points
from sol.structural_encoder import (
    ARCHITECTURE as STRUCTURAL_ARCHITECTURE,
    StructuralJewelEncoder,
    quaternion_to_matrix,
    render_structural,
)
from sol.token_grid import GridSpec
from sol.train_amortized_encoder import sample_voxels
from stprim.data.video_io import load_video


def teacher_descriptors(
    features: torch.Tensor, keep: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opacity-weighted subsample: centres, log-eigenvalue spreads, principal axes.

    The axis is the eigenvector of the largest eigenvalue — the direction a
    sheared tube points. Supervising spread alone produced elongation without
    direction (visible as horizontal smearing), so orientation is carried too.
    """
    teacher = extract_local_teacher_attributes(features, keep, generator)
    spread = teacher.log_scale[:, -1] - teacher.log_scale[:, 0]
    size = teacher.log_scale.mean(dim=1)
    return teacher.centers, spread, teacher.axis, teacher.opacity, size


def principal_axis(
    quaternion: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """The rotation column belonging to the largest scale."""
    rotation = quaternion_to_matrix(quaternion)
    largest = log_scale.argmax(dim=1)
    index = largest.view(-1, 1, 1).expand(-1, 3, 1)
    return torch.gather(rotation, 2, index).squeeze(-1)


def orientation_loss(
    student_axis: torch.Tensor, teacher_axis: torch.Tensor
) -> torch.Tensor:
    """1 - |cos| between axes; absolute because an axis has no sign."""
    cosine = (student_axis * teacher_axis).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cosine).mean()


def mixed_spacetime_tilt(axis: torch.Tensor) -> torch.Tensor:
    """Return zero for pure space/time axes and one for 45-degree trajectories."""
    temporal = axis[:, 2].abs().clamp(max=1.0)
    spatial_square = axis[:, :2].square().sum(dim=1)
    spatial = torch.where(
        spatial_square > 0,
        spatial_square.clamp_min(1e-12).sqrt(),
        torch.zeros_like(spatial_square),
    )
    return 2.0 * temporal * spatial


def soft_occupancy(
    centres: torch.Tensor,
    grid: GridSpec,
    weights: torch.Tensor | None = None,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Differentiable per-cell occupancy share via soft trilinear-style binning.

    Chamfer correspondence teaches shape but not placement: with enough students
    the nearest-teacher distance is small everywhere, so nothing pushes mass into
    dense regions. Matching binned densities penalises exactly that — too few
    jewels where the fitter put many.
    """
    gu, gv, gt = grid.shape
    scaled = (centres.clamp(-1, 1) + 1) * 0.5
    axes = []
    for axis, count in enumerate((gu, gv, gt)):
        edges = (torch.arange(count, device=centres.device, dtype=centres.dtype) + 0.5) / count
        distance = (scaled[:, axis : axis + 1] - edges[None]).abs()
        axes.append(torch.softmax(-distance / (temperature / count), dim=1))
    spatial_weights = (
        axes[0][:, :, None, None] * axes[1][:, None, :, None] * axes[2][:, None, None, :]
    )
    assignment = weights
    if assignment is None:
        assignment = torch.ones(len(centres), device=centres.device, dtype=centres.dtype)
    if assignment.shape != (len(centres),):
        raise ValueError("occupancy weights must have one scalar per centre")
    occupancy = (
        spatial_weights.reshape(len(centres), -1) * assignment[:, None]
    ).sum(dim=0)
    return occupancy / occupancy.sum().clamp_min(1e-8)


def density_loss(
    student_centres: torch.Tensor,
    teacher_centres: torch.Tensor,
    grid: GridSpec,
    student_weights: torch.Tensor | None = None,
    teacher_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric KL between student and teacher occupancy distributions."""
    student = soft_occupancy(
        student_centres, grid, weights=student_weights
    ).clamp_min(1e-8)
    with torch.no_grad():
        teacher = soft_occupancy(
            teacher_centres, grid, weights=teacher_weights
        ).clamp_min(1e-8)
    return 0.5 * (
        (student * (student / teacher).log()).sum()
        + (teacher * (teacher / student).log()).sum()
    )


def soft_active_fraction(
    logit_w: torch.Tensor,
    *,
    opacity_threshold: float = 0.02,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Differentiable fraction of proposals above the exported opacity floor."""
    if not 0 < opacity_threshold < 1 or temperature <= 0:
        raise ValueError("active-fraction parameters are outside their valid range")
    threshold = math.log(opacity_threshold / (1.0 - opacity_threshold))
    return torch.sigmoid((logit_w - threshold) / temperature).mean()


def select_validation_ids(
    videos: dict[str, tuple[torch.Tensor, str]],
    requested: tuple[str, ...],
    limit: int,
) -> list[str]:
    """Select an explicit, ordered evaluation subset without changing training data."""
    if limit < 0:
        raise ValueError("validation limit must be non-negative")
    available = [
        source_id for source_id, (_, split) in videos.items()
        if split == "validation"
    ]
    if requested:
        missing = [source_id for source_id in requested if source_id not in available]
        if missing:
            raise ValueError(f"requested validation sources are unavailable: {missing}")
        available = list(requested)
    if limit:
        available = available[:limit]
    if not available:
        raise ValueError("validation selection is empty")
    return available


def schedule_multiplier(step: int, start_step: int, ramp_steps: int) -> float:
    """Return a zero-to-one linear multiplier for delayed structural pressure."""
    if step < 0 or start_step < 0 or ramp_steps < 0:
        raise ValueError("schedule steps must be non-negative")
    if step <= start_step:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - start_step) / ramp_steps)


def freeze_geometry_state(model: StructuralJewelEncoder) -> dict[str, torch.Tensor]:
    """Freeze shared features and snapshot geometry/opacity output rows exactly."""
    for parameter in model.trunk.parameters():
        parameter.requires_grad_(False)
    channels = torch.arange(23, device=model.head.weight.device)
    geometry_channels = (channels < 10) | (channels == 22)
    rows = geometry_channels.repeat(model.slots_per_cell)
    return {
        "rows": rows,
        "weight": model.head.weight.detach()[rows].clone(),
        "bias": model.head.bias.detach()[rows].clone(),
    }


def mask_geometry_gradients(
    model: StructuralJewelEncoder, frozen: dict[str, torch.Tensor]
) -> None:
    """Remove mixed-head gradients that could change frozen field geometry."""
    rows = frozen["rows"]
    if model.head.weight.grad is not None:
        model.head.weight.grad[rows] = 0
    if model.head.bias.grad is not None:
        model.head.bias.grad[rows] = 0


@torch.no_grad()
def restore_geometry_state(
    model: StructuralJewelEncoder, frozen: dict[str, torch.Tensor]
) -> None:
    """Undo decoupled AdamW weight decay on frozen rows after every update."""
    rows = frozen["rows"]
    model.head.weight[rows] = frozen["weight"]
    model.head.bias[rows] = frozen["bias"]


def multiscale_image_loss(
    rendered: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Charbonnier image pyramids plus first-order edges for appearance detail."""
    if rendered.shape != target.shape or rendered.ndim != 4 or rendered.shape[-1] != 3:
        raise ValueError("rendered and target must share NHWC image shapes")
    prediction = rendered.permute(0, 3, 1, 2)
    reference = target.permute(0, 3, 1, 2)
    loss = torch.zeros((), device=rendered.device, dtype=rendered.dtype)
    weight_sum = 0.0
    for weight in (1.0, 0.5, 0.25):
        difference = prediction - reference
        loss = loss + weight * torch.sqrt(difference.square() + 1e-6).mean()
        weight_sum += weight
        if min(prediction.shape[-2:]) < 2:
            break
        prediction = torch.nn.functional.avg_pool2d(prediction, 2)
        reference = torch.nn.functional.avg_pool2d(reference, 2)
    prediction = rendered.permute(0, 3, 1, 2)
    reference = target.permute(0, 3, 1, 2)
    horizontal = torch.nn.functional.l1_loss(
        prediction[..., 1:] - prediction[..., :-1],
        reference[..., 1:] - reference[..., :-1],
    )
    vertical = torch.nn.functional.l1_loss(
        prediction[..., 1:, :] - prediction[..., :-1, :],
        reference[..., 1:, :] - reference[..., :-1, :],
    )
    return loss / weight_sum + 0.25 * (horizontal + vertical)


def chamfer(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric squared-distance Chamfer plus a->b nearest indices."""
    forward, indices = [], []
    for start in range(0, len(a), chunk):
        distance = torch.cdist(a[start : start + chunk], b)
        value, index = distance.min(dim=1)
        forward.append(value.square())
        indices.append(index)
    backward = []
    for start in range(0, len(b), chunk):
        distance = torch.cdist(b[start : start + chunk], a)
        backward.append(distance.min(dim=1).values.square())
    return (
        torch.cat(forward).mean() + torch.cat(backward).mean(),
        torch.cat(indices),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-geometry-checkpoint")
    parser.add_argument(
        "--factorized", action="store_true",
        help="use the v3 disjoint geometry/appearance architecture",
    )
    parser.add_argument(
        "--freeze-geometry", action="store_true",
        help=(
            "freeze trunk plus center/quaternion/scale/opacity rows; train only "
            "colour, colour-gradient, and background parameters"
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots-per-cell", type=int, default=10)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--max-offset-cells", type=float, default=4.0)
    parser.add_argument("--seed-video-colors", action="store_true")
    parser.add_argument("--appearance-dim", type=int, default=32)
    parser.add_argument("--appearance-hidden", type=int, default=64)
    parser.add_argument("--appearance-grid-weight", type=float, default=0.0)
    parser.add_argument("--appearance-grid-every", type=int, default=4)
    parser.add_argument("--appearance-grid-frames", type=int, default=2)
    parser.add_argument("--appearance-grid-height", type=int, default=64)
    parser.add_argument("--appearance-grid-width", type=int, default=96)
    parser.add_argument("--points-per-step", type=int, default=8192)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--teacher-sample", type=int, default=12000)
    parser.add_argument(
        "--chamfer-sample",
        type=int,
        default=0,
        help=(
            "random student jewels used for the correspondence terms each step "
            "(0 = all). Autograd retains the full pairwise distance matrix, so "
            "this bounds memory independently of the jewel budget."
        ),
    )
    parser.add_argument(
        "--chamfer-chunk",
        type=int,
        default=2048,
        help="rows per cdist block; scales memory with jewel count",
    )
    parser.add_argument("--chamfer-weight", type=float, default=3.0)
    parser.add_argument("--spread-weight", type=float, default=0.05)
    parser.add_argument("--size-weight", type=float, default=0.0)
    parser.add_argument(
        "--size-offset", type=float, default=0.0,
        help="additive log-scale offset applied to nearest-teacher absolute size",
    )
    parser.add_argument("--local-neighbors", type=int, default=4)
    parser.add_argument("--local-temperature", type=float, default=0.08)
    parser.add_argument("--local-scale-weight", type=float, default=0.0)
    parser.add_argument("--local-axis-weight", type=float, default=0.0)
    parser.add_argument("--local-opacity-weight", type=float, default=0.0)
    parser.add_argument("--local-color-weight", type=float, default=0.0)
    parser.add_argument("--local-gradient-weight", type=float, default=0.0)
    parser.add_argument("--local-start-step", type=int, default=0)
    parser.add_argument("--local-ramp-steps", type=int, default=0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--tilt-weight", type=float, default=0.0)
    parser.add_argument("--orientation-start-step", type=int, default=0)
    parser.add_argument("--orientation-ramp-steps", type=int, default=0)
    parser.add_argument("--density-weight", type=float, default=0.0)
    parser.add_argument("--density-grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--density-start-step", type=int, default=0)
    parser.add_argument("--density-ramp-steps", type=int, default=0)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--target-active-fraction", type=float, default=0.5)
    parser.add_argument("--polarization-weight", type=float, default=0.0)
    parser.add_argument("--sparsity-start-step", type=int, default=0)
    parser.add_argument("--sparsity-ramp-steps", type=int, default=0)
    parser.add_argument("--teacher-probability", type=float, default=0.5)
    parser.add_argument("--validation-source", action="append", default=[])
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument(
        "--renderer", choices=("exact", "support_tiled"), default="support_tiled"
    )
    parser.add_argument("--support-capacity", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.appearance_grid_weight < 0 or min(
        args.appearance_grid_every,
        args.appearance_grid_frames,
        args.appearance_grid_height,
        args.appearance_grid_width,
    ) <= 0:
        raise ValueError("appearance-grid settings must be positive and weight non-negative")
    local_weights = (
        args.local_scale_weight,
        args.local_axis_weight,
        args.local_opacity_weight,
        args.local_color_weight,
        args.local_gradient_weight,
    )
    if (
        min(local_weights) < 0
        or args.local_neighbors <= 0
        or args.local_temperature <= 0
    ):
        raise ValueError(
            "local teacher weights must be non-negative and matching settings positive"
        )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cpu_generator = torch.Generator().manual_seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stprim"))
    from prior.featurize import state_to_features  # noqa: PLC0415

    fits_by_name, fits_by_source = {}, {}
    for root in args.checkpoint_root:
        for path in Path(root).glob("*.pt"):
            fits_by_name.setdefault(path.name, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            source = saved.get("source", {})
            if source.get("source_id"):
                fits_by_source.setdefault(source["source_id"], path)
    manifest = json.loads(Path(args.manifest).read_text())
    spec = GridSpec(tuple(args.grid), 1024)
    density_grid = GridSpec(tuple(args.density_grid), 1)

    videos: dict[str, tuple[torch.Tensor, str]] = {}
    teachers: dict[str, LocalTeacherAttributes] = {}
    for example in manifest["examples"]:
        videos[example["source_id"]] = (
            load_video(
                example["video"],
                max_frames=int(example["frames"]),
                start_frame=int(example.get("start_frame", 0)),
                resize=(args.height, args.width),
                device="cpu",
            ),
            example["split"],
        )
        name = f"{Path(example['video']).stem}_w000000.pt"
        fit_path = fits_by_source.get(example["source_id"], fits_by_name.get(name))
        if fit_path is None:
            continue
        fitted = torch.load(fit_path, map_location="cpu", weights_only=False)
        features = state_to_features(fitted["state"]).float()
        teacher = extract_local_teacher_attributes(
            features, args.teacher_sample, cpu_generator
        )
        teachers[example["source_id"]] = teacher.to(device)
        print("loaded teacher", example["source_id"], flush=True)

    train_ids = [k for k, (_, s) in videos.items() if s == "train"]
    validation_ids = select_validation_ids(
        videos, tuple(args.validation_source), args.validation_limit
    )
    teacher_train_ids = [source_id for source_id in train_ids if source_id in teachers]
    if not teacher_train_ids:
        raise ValueError("at least one training example needs a fitted teacher")
    if args.init_checkpoint and args.init_geometry_checkpoint:
        raise ValueError("full and geometry-only initialization are mutually exclusive")
    if args.factorized:
        model = FactorizedStructuralJewelEncoder(
            grid_spec=spec,
            slots_per_cell=args.slots_per_cell,
            model_dim=args.model_dim,
            max_offset_cells=args.max_offset_cells,
            appearance_dim=args.appearance_dim,
            appearance_hidden=args.appearance_hidden,
        ).to(device)
        checkpoint_architecture = FACTORIZED_ARCHITECTURE
        expected_args = model.model_args
    else:
        model = StructuralJewelEncoder(
            grid_spec=spec,
            slots_per_cell=args.slots_per_cell,
            model_dim=args.model_dim,
            max_offset_cells=args.max_offset_cells,
            seed_video_colors=args.seed_video_colors,
        ).to(device)
        checkpoint_architecture = STRUCTURAL_ARCHITECTURE
        expected_args = {
            "slots_per_cell": args.slots_per_cell,
            "model_dim": args.model_dim,
            "max_offset_cells": args.max_offset_cells,
            "seed_video_colors": args.seed_video_colors,
        }
    if args.init_checkpoint:
        initialized = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        meta = initialized.get("meta", {})
        if meta.get("architecture") != checkpoint_architecture:
            raise ValueError("initial checkpoint architecture does not match")
        if tuple(meta.get("grid_shape", ())) != spec.shape:
            raise ValueError("initial checkpoint grid does not match")
        if meta.get("model_args") != expected_args:
            raise ValueError("initial checkpoint model arguments do not match")
        model.load_state_dict(initialized["model"])
        print(
            f"initialized from {args.init_checkpoint} step={initialized.get('step')}",
            flush=True,
        )
    if args.init_geometry_checkpoint:
        if not args.factorized:
            raise ValueError("geometry-only initialization requires --factorized")
        initialized_geometry = torch.load(
            args.init_geometry_checkpoint, map_location=device, weights_only=False
        )
        meta = initialized_geometry.get("meta", {})
        source_args = meta.get("model_args", {})
        if meta.get("architecture") != STRUCTURAL_ARCHITECTURE:
            raise ValueError("geometry source is not a v2 structural checkpoint")
        if tuple(meta.get("grid_shape", ())) != spec.shape:
            raise ValueError("geometry source grid does not match")
        for key in ("slots_per_cell", "model_dim", "max_offset_cells"):
            if source_args.get(key) != expected_args[key]:
                raise ValueError(f"geometry source {key} does not match")
        model.load_v2_geometry(initialized_geometry["model"])
        print(
            f"initialized geometry from {args.init_geometry_checkpoint} "
            f"step={initialized_geometry.get('step')}", flush=True,
        )
    frozen_geometry = None
    if args.freeze_geometry:
        if args.factorized:
            model.freeze_geometry()
        else:
            frozen_geometry = freeze_geometry_state(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        report: dict = {}
        evaluation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        structures = []
        for source_id in validation_ids:
            video = videos[source_id][0].to(device)
            prediction = model(video)
            points, target = sample_voxels(video, 16384, evaluation_generator)
            rendered = render_structural(
                prediction, points, point_chunk=args.point_chunk,
                cull_mode=args.renderer, support_capacity=args.support_capacity,
            )
            mse = float(torch.nn.functional.mse_loss(rendered, target))
            report[source_id] = {"psnr": -10.0 * math.log10(max(mse, 1e-10))}
            structures.append(structure_report(model.canonical_features(prediction).cpu()))
        report["macro_psnr"] = sum(
            v["psnr"] for v in report.values() if isinstance(v, dict)
        ) / len(validation_ids)
        report["structure"] = {
            key: sum(s[key] for s in structures) / len(structures)
            for key in (
                "anisotropy_median",
                "anisotropy_p90",
                "extent_median",
                "extent_iqr_ratio",
                "occupancy_uniformity",
                "jewels_above_2pct_opacity",
                "mixed_spacetime_tilt_median",
                "mixed_spacetime_tilt_p90",
            )
        }
        report["structure"]["active_fraction"] = (
            report["structure"]["jewels_above_2pct_opacity"] / model.n_jewels
        )
        model.train()
        return report

    print(
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"jewels={model.n_jewels} teachers={len(teacher_train_ids)} "
        f"teacher_sample={args.teacher_sample}",
        flush=True,
    )
    latest = None
    history: list[tuple[float, ...]] = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        use_teacher = bool(
            torch.rand((), generator=generator, device=device) < args.teacher_probability
        )
        choices = teacher_train_ids if use_teacher else train_ids
        source_id = choices[
            int(torch.randint(0, len(choices), (1,), generator=generator, device=device))
        ]
        video = videos[source_id][0].to(device)
        if step <= args.warmup:
            rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        prediction = model(video)
        points, target = sample_voxels(video, args.points_per_step, generator)
        rendered = render_structural(
            prediction, points, point_chunk=args.point_chunk,
            cull_mode=args.renderer, support_capacity=args.support_capacity,
        )
        render_loss = torch.nn.functional.mse_loss(rendered, target)
        zero = torch.zeros((), device=device)
        image_loss = zero
        if args.appearance_grid_weight and step % args.appearance_grid_every == 0:
            stride = max(len(video) // args.appearance_grid_frames, 1)
            frame_indices = (
                torch.arange(args.appearance_grid_frames) * stride + step
            ).remainder(len(video)).sort().values
            grid_points = frame_points(
                len(video), frame_indices,
                args.appearance_grid_height, args.appearance_grid_width,
                device=device,
            )
            grid_rendered = render_structural(
                prediction, grid_points, point_chunk=args.point_chunk,
                cull_mode=args.renderer, support_capacity=args.support_capacity,
            ).reshape(
                args.appearance_grid_frames,
                args.appearance_grid_height,
                args.appearance_grid_width,
                3,
            )
            grid_target = torch.nn.functional.interpolate(
                video[frame_indices].permute(0, 3, 1, 2),
                size=(args.appearance_grid_height, args.appearance_grid_width),
                mode="bilinear", align_corners=True,
            ).permute(0, 2, 3, 1)
            image_loss = multiscale_image_loss(grid_rendered, grid_target)
        chamfer_loss = spread_loss = size_loss = axis_loss = tilt_loss = zero
        occupancy_loss = zero
        local_losses = {
            "scale": zero,
            "axis": zero,
            "opacity": zero,
            "color": zero,
            "gradient": zero,
        }
        needs_teacher_structure = any((
            args.chamfer_weight,
            args.spread_weight,
            args.size_weight,
            args.orientation_weight,
            args.tilt_weight,
            args.density_weight,
            *local_weights,
        ))
        if source_id in teachers and needs_teacher_structure:
            teacher = teachers[source_id]
            teacher_centres = teacher.centers
            teacher_spread = teacher.log_scale[:, -1] - teacher.log_scale[:, 0]
            teacher_axis = teacher.axis
            teacher_weight = teacher.opacity
            teacher_size = teacher.log_scale.mean(dim=1)
            opacity = torch.sigmoid(prediction["logit_w"])
            if args.chamfer_sample and args.chamfer_sample < len(prediction["centers"]):
                subset = torch.multinomial(
                    opacity.clamp_min(1e-6), args.chamfer_sample,
                    replacement=False, generator=generator,
                )
            else:
                subset = slice(None)
            chamfer_loss, nearest = chamfer(
                prediction["centers"][subset], teacher_centres,
                chunk=args.chamfer_chunk,
            )
            log_scale = prediction["log_scale"][subset]
            student_spread = log_scale.max(dim=1).values - log_scale.min(dim=1).values
            spread_loss = torch.nn.functional.mse_loss(
                student_spread, teacher_spread[nearest]
            )
            size_loss = torch.nn.functional.smooth_l1_loss(
                log_scale.mean(dim=1), teacher_size[nearest] + args.size_offset
            )
            student_axis = principal_axis(
                prediction["quaternion"][subset], log_scale
            )
            matched_teacher_axis = teacher_axis[nearest]
            axis_loss = orientation_loss(student_axis, matched_teacher_axis)
            tilt_loss = torch.nn.functional.smooth_l1_loss(
                mixed_spacetime_tilt(student_axis),
                mixed_spacetime_tilt(matched_teacher_axis),
            )
            if args.density_weight:
                occupancy_loss = density_loss(
                    prediction["centers"], teacher_centres, density_grid,
                    student_weights=opacity, teacher_weights=teacher_weight,
                )
            if any(local_weights):
                local_losses = local_teacher_attribute_losses(
                    student_centers=prediction["centers"][subset],
                    student_log_scale=log_scale,
                    student_axis=student_axis,
                    student_opacity=opacity[subset],
                    student_colors=prediction["colors"][subset],
                    student_color_grads=prediction["color_grads"][subset],
                    teacher=teacher,
                    neighbors=args.local_neighbors,
                    temperature=args.local_temperature,
                    size_offset=args.size_offset,
                    opacity_mass_ratio=(
                        teacher.active_count
                        / (model.n_jewels * args.target_active_fraction)
                    ),
                )
        active_fraction = soft_active_fraction(prediction["logit_w"])
        sparsity_loss = (active_fraction - args.target_active_fraction).square()
        soft_presence = torch.sigmoid(
            (prediction["logit_w"] - math.log(0.02 / 0.98)) / 0.5
        )
        polarization_loss = (soft_presence * (1.0 - soft_presence)).mean()
        sparsity_multiplier = schedule_multiplier(
            step, args.sparsity_start_step, args.sparsity_ramp_steps
        )
        density_multiplier = schedule_multiplier(
            step, args.density_start_step, args.density_ramp_steps
        )
        orientation_multiplier = schedule_multiplier(
            step, args.orientation_start_step, args.orientation_ramp_steps
        )
        local_multiplier = schedule_multiplier(
            step, args.local_start_step, args.local_ramp_steps
        )
        loss = (
            render_loss
            + args.appearance_grid_weight * image_loss
            + args.chamfer_weight * chamfer_loss
            + args.spread_weight * spread_loss
            + args.size_weight * size_loss
            + orientation_multiplier * args.orientation_weight * axis_loss
            + orientation_multiplier * args.tilt_weight * tilt_loss
            + density_multiplier * args.density_weight * occupancy_loss
            + local_multiplier * args.local_scale_weight * local_losses["scale"]
            + local_multiplier * args.local_axis_weight * local_losses["axis"]
            + local_multiplier * args.local_opacity_weight * local_losses["opacity"]
            + local_multiplier * args.local_color_weight * local_losses["color"]
            + local_multiplier * args.local_gradient_weight * local_losses["gradient"]
            + sparsity_multiplier * args.sparsity_weight * sparsity_loss
            + sparsity_multiplier * args.polarization_weight * polarization_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if frozen_geometry is not None:
            mask_geometry_gradients(model, frozen_geometry)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if frozen_geometry is not None:
            restore_geometry_state(model, frozen_geometry)
        history.append(
            (
                float(render_loss.detach()),
                float(image_loss.detach()),
                float(chamfer_loss.detach()),
                float(spread_loss.detach()),
                float(size_loss.detach()),
                float(axis_loss.detach()),
                float(tilt_loss.detach()),
                float(occupancy_loss.detach()),
                float(local_losses["scale"].detach()),
                float(local_losses["axis"].detach()),
                float(local_losses["opacity"].detach()),
                float(local_losses["color"].detach()),
                float(local_losses["gradient"].detach()),
                float(sparsity_loss.detach()),
                float(active_fraction.detach()),
            )
        )
        if step % args.log_every == 0 or step == args.steps:
            recent = history[-args.log_every :]
            render_mean = sum(row[0] for row in recent) / len(recent)
            record = {
                "step": step,
                "render_loss": render_mean,
                "train_psnr": -10.0 * math.log10(max(render_mean, 1e-10)),
                "appearance_grid": sum(row[1] for row in recent) / len(recent),
                "chamfer": sum(row[2] for row in recent) / len(recent),
                "spread": sum(row[3] for row in recent) / len(recent),
                "size": sum(row[4] for row in recent) / len(recent),
                "orientation": sum(row[5] for row in recent) / len(recent),
                "tilt": sum(row[6] for row in recent) / len(recent),
                "density": sum(row[7] for row in recent) / len(recent),
                "local_scale": sum(row[8] for row in recent) / len(recent),
                "local_axis": sum(row[9] for row in recent) / len(recent),
                "local_opacity": sum(row[10] for row in recent) / len(recent),
                "local_color": sum(row[11] for row in recent) / len(recent),
                "local_gradient": sum(row[12] for row in recent) / len(recent),
                "sparsity": sum(row[13] for row in recent) / len(recent),
                "active_fraction": sum(row[14] for row in recent) / len(recent),
                "sparsity_multiplier": sparsity_multiplier,
                "density_multiplier": density_multiplier,
                "orientation_multiplier": orientation_multiplier,
                "local_multiplier": local_multiplier,
                "gradient_norm": float(gradient_norm),
                "lr": rate,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate()
            with log_path.open("a") as stream:
                stream.write(json.dumps({"step": step, "evaluation": latest}) + "\n")
            print(
                json.dumps(
                    {
                        "step": step,
                        "macro_psnr": round(latest["macro_psnr"], 3),
                        "structure": {k: round(v, 4) for k, v in latest["structure"].items()},
                    }
                ),
                flush=True,
            )
            checkpoint = {
                "model": model.state_dict(),
                "step": step,
                "meta": {
                    "architecture": checkpoint_architecture,
                    "grid_shape": spec.shape,
                    "slots_per_cell": args.slots_per_cell,
                    "model_args": expected_args,
                    "distilled": True,
                    "train_args": vars(args),
                    "latest_evaluation": latest,
                    "initialized_from": args.init_checkpoint,
                    "initialized_geometry_from": args.init_geometry_checkpoint,
                },
            }
            torch.save(checkpoint, output_dir / "encoder.pt")
            torch.save(checkpoint, output_dir / f"encoder_step{step:06}.pt")
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "jewels_per_window": model.n_jewels,
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
