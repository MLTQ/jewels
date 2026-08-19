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
from sol.render import covariance_terms
from sol.structural_encoder import (
    ARCHITECTURE,
    StructuralJewelEncoder,
    quaternion_to_matrix,
    render_structural,
)
from sol.token_grid import GridSpec
from sol.train_amortized_encoder import sample_voxels
from stprim.data.video_io import load_video


def teacher_descriptors(
    features: torch.Tensor, keep: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opacity-weighted subsample: centres, log-eigenvalue spreads, principal axes.

    The axis is the eigenvector of the largest eigenvalue — the direction a
    sheared tube points. Supervising spread alone produced elongation without
    direction (visible as horizontal smearing), so orientation is carried too.
    """
    weight = torch.sigmoid(features[:, 21])
    index = torch.multinomial(
        weight.clamp_min(1e-6), min(keep, len(features)), replacement=False,
        generator=generator,
    )
    chosen = features[index]
    covariance, _ = covariance_terms(chosen)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    eigenvalues = eigenvalues.clamp_min(1e-12)
    log_scale = 0.5 * eigenvalues.log()
    spread = (log_scale[:, -1] - log_scale[:, 0]).to(chosen.dtype)
    axis = eigenvectors[:, :, -1].to(chosen.dtype)
    return chosen[:, :3], spread, axis


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


def soft_occupancy(
    centres: torch.Tensor, grid: GridSpec, temperature: float = 0.35
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
    weights = (
        axes[0][:, :, None, None] * axes[1][:, None, :, None] * axes[2][:, None, None, :]
    )
    occupancy = weights.reshape(len(centres), -1).sum(dim=0)
    return occupancy / occupancy.sum().clamp_min(1e-8)


def density_loss(
    student_centres: torch.Tensor,
    teacher_centres: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """Symmetric KL between student and teacher occupancy distributions."""
    student = soft_occupancy(student_centres, grid).clamp_min(1e-8)
    with torch.no_grad():
        teacher = soft_occupancy(teacher_centres, grid).clamp_min(1e-8)
    return 0.5 * (
        (student * (student / teacher).log()).sum()
        + (teacher * (teacher / student).log()).sum()
    )


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
    parser.add_argument("--points-per-step", type=int, default=8192)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--teacher-sample", type=int, default=12000)
    parser.add_argument("--chamfer-weight", type=float, default=3.0)
    parser.add_argument("--spread-weight", type=float, default=0.05)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--density-weight", type=float, default=0.0)
    parser.add_argument("--density-grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cpu_generator = torch.Generator().manual_seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stprim"))
    from prior.featurize import state_to_features  # noqa: PLC0415

    fits = {}
    for root in args.checkpoint_root:
        for path in Path(root).glob("*_w000000.pt"):
            fits.setdefault(path.name, path)
    manifest = json.loads(Path(args.manifest).read_text())
    spec = GridSpec(tuple(args.grid), 1024)
    density_grid = GridSpec(tuple(args.density_grid), 1)

    videos, teachers = {}, {}
    for example in manifest["examples"]:
        name = f"{Path(example['video']).stem}_w000000.pt"
        if name not in fits:
            raise FileNotFoundError(f"missing fitted field for {name}")
        videos[example["source_id"]] = (
            load_video(
                example["video"],
                max_frames=int(example["frames"]),
                start_frame=int(example.get("start_frame", 0)),
                resize=(args.height, args.width),
                device="cpu",
            ).to(device),
            example["split"],
        )
        fitted = torch.load(fits[name], map_location="cpu", weights_only=False)
        features = state_to_features(fitted["state"]).float()
        centres, spread, axis = teacher_descriptors(
            features, args.teacher_sample, cpu_generator
        )
        teachers[example["source_id"]] = (
            centres.to(device), spread.to(device), axis.to(device)
        )
        print("loaded teacher", example["source_id"], flush=True)

    train_ids = [k for k, (_, s) in videos.items() if s == "train"]
    validation_ids = [k for k, (_, s) in videos.items() if s == "validation"]
    model = StructuralJewelEncoder(
        grid_spec=spec, slots_per_cell=args.slots_per_cell, model_dim=args.model_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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
            video = videos[source_id][0]
            prediction = model(video)
            points, target = sample_voxels(video, 16384, evaluation_generator)
            rendered = render_structural(prediction, points, point_chunk=args.point_chunk)
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
                "extent_iqr_ratio",
                "occupancy_uniformity",
            )
        }
        model.train()
        return report

    print(
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"jewels={model.n_jewels} teacher_sample={args.teacher_sample}",
        flush=True,
    )
    latest = None
    history: list[tuple[float, float, float, float, float]] = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        source_id = train_ids[
            int(torch.randint(0, len(train_ids), (1,), generator=generator, device=device))
        ]
        video = videos[source_id][0]
        teacher_centres, teacher_spread, teacher_axis = teachers[source_id]
        if step <= args.warmup:
            rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        prediction = model(video)
        points, target = sample_voxels(video, args.points_per_step, generator)
        rendered = render_structural(prediction, points, point_chunk=args.point_chunk)
        render_loss = torch.nn.functional.mse_loss(rendered, target)
        chamfer_loss, nearest = chamfer(prediction["centers"], teacher_centres)
        student_spread = (
            prediction["log_scale"].max(dim=1).values
            - prediction["log_scale"].min(dim=1).values
        )
        spread_loss = torch.nn.functional.mse_loss(
            student_spread, teacher_spread[nearest]
        )
        axis_loss = orientation_loss(
            principal_axis(prediction["quaternion"], prediction["log_scale"]),
            teacher_axis[nearest],
        )
        occupancy_loss = (
            density_loss(prediction["centers"], teacher_centres, density_grid)
            if args.density_weight
            else torch.zeros((), device=device)
        )
        loss = (
            render_loss
            + args.chamfer_weight * chamfer_loss
            + args.spread_weight * spread_loss
            + args.orientation_weight * axis_loss
            + args.density_weight * occupancy_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(
            (
                float(render_loss.detach()),
                float(chamfer_loss.detach()),
                float(spread_loss.detach()),
                float(axis_loss.detach()),
                float(occupancy_loss.detach()),
            )
        )
        if step % args.log_every == 0 or step == args.steps:
            recent = history[-args.log_every :]
            render_mean = sum(r for r, _, _, _, _ in recent) / len(recent)
            record = {
                "step": step,
                "render_loss": render_mean,
                "train_psnr": -10.0 * math.log10(max(render_mean, 1e-10)),
                "chamfer": sum(c for _, c, _, _, _ in recent) / len(recent),
                "spread": sum(s for _, _, s, _, _ in recent) / len(recent),
                "orientation": sum(a for _, _, _, a, _ in recent) / len(recent),
                "density": sum(o for _, _, _, _, o in recent) / len(recent),
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "meta": {
                        "architecture": ARCHITECTURE,
                        "grid_shape": spec.shape,
                        "slots_per_cell": args.slots_per_cell,
                        "model_args": {
                            "slots_per_cell": args.slots_per_cell,
                            "model_dim": args.model_dim,
                        },
                        "distilled": True,
                        "train_args": vars(args),
                        "latest_evaluation": latest,
                    },
                },
                output_dir / "encoder.pt",
            )
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
