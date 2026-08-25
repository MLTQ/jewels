"""Compare sparse structural encoders with the lattice baseline and fitted ceiling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageDraw
import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.compare_field_structure import structure_report
from sol.factorized_structural_encoder import (
    ARCHITECTURE as FACTORIZED_ARCHITECTURE,
    FactorizedStructuralJewelEncoder,
)
from sol.perceptual_eval import lpips_metric, score_arms
from sol.render import covariance_terms
from sol.render_streaming_continuation import frame_points
from sol.structural_encoder import (
    ARCHITECTURE as STRUCTURAL_ARCHITECTURE,
    StructuralJewelEncoder,
    render_structural,
)
from sol.token_grid import GridSpec

ROOT = Path(__file__).resolve().parents[1]
STPRIM_ROOT = ROOT / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import PrimitiveField  # noqa: E402
from data.video_io import load_video  # noqa: E402
from models.render import render_points  # noqa: E402


def load_baseline(path: Path, device: torch.device) -> VideoToJewelEncoder:
    """Restore the frozen lattice encoder without changing checkpoint semantics."""
    saved = torch.load(path, map_location=device, weights_only=False)
    meta = saved["meta"]
    model = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024), **meta["model_args"]
    ).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def load_candidate(
    path: Path, device: torch.device
) -> StructuralJewelEncoder | FactorizedStructuralJewelEncoder:
    """Restore one declared irregular structural checkpoint."""
    saved = torch.load(path, map_location=device, weights_only=False)
    meta = saved["meta"]
    architecture = meta["architecture"]
    constructors = {
        STRUCTURAL_ARCHITECTURE: StructuralJewelEncoder,
        FACTORIZED_ARCHITECTURE: FactorizedStructuralJewelEncoder,
    }
    if architecture not in constructors:
        raise ValueError(f"candidate {path} has unsupported architecture {architecture}")
    model = constructors[architecture](
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024),
        **meta["model_args"],
    ).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def load_teacher(path: Path, device: torch.device) -> tuple[PrimitiveField, dict]:
    """Restore a source-owned support-correct fitted field."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state"]
    field = PrimitiveField(
        len(state["mu"]), p1_color="color_grad" in state, device=device
    )
    field.load_state_dict(state)
    return field.eval(), saved


def structure(features: torch.Tensor) -> dict[str, float]:
    """Add mixed spacetime tilt and active fraction to the standard structure report."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        base = structure_report(features)
    weight = torch.sigmoid(features[:, 21])
    active = features[weight > 0.02]
    if len(active) > 20000:
        index = torch.linspace(0, len(active) - 1, 20000).long()
        active = active[index]
    covariance, _ = covariance_terms(active)
    eigenvectors = torch.linalg.eigh(covariance.double()).eigenvectors
    temporal = eigenvectors[:, 2, -1].abs()
    mixed = 2 * temporal * torch.sqrt((1 - temporal.square()).clamp_min(0))
    return {
        **base,
        "active_fraction": base["jewels_above_2pct_opacity"] / base["jewels_total"],
        "mixed_spacetime_tilt_median": float(mixed.median()),
        "mixed_spacetime_tilt_p90": float(mixed.quantile(0.9)),
    }


def summarize_gate(
    perceptual_macro: dict[str, dict[str, float]],
    structure_macro: dict[str, dict[str, float]],
    structure_by_seed: dict[str, dict[str, float]],
) -> dict:
    """Apply the preregistered representation gate without post-hoc thresholds."""
    candidate_labels = sorted(
        label for label in perceptual_macro if label.startswith("irregular_seed")
    )
    candidate_perceptual = {
        key: statistics.mean(perceptual_macro[label][key] for label in candidate_labels)
        for key in ("lpips", "psnr", "ssim", "layout_psnr", "layout_ssim")
    }
    checks = {
        "each_seed_occupancy_at_most_0.985": all(
            row["occupancy_uniformity"] <= 0.985
            for row in structure_by_seed.values()
        ),
        "each_seed_less_uniform_than_lattice": all(
            row["occupancy_uniformity"]
            < structure_macro["lattice"]["occupancy_uniformity"]
            for row in structure_by_seed.values()
        ),
        "mean_active_fraction_at_most_0.70": (
            structure_macro["irregular"]["active_fraction"] <= 0.70
        ),
        "mean_mixed_tilt_at_least_0.25": (
            structure_macro["irregular"]["mixed_spacetime_tilt_median"] >= 0.25
        ),
        "mean_psnr_at_least_20db": candidate_perceptual["psnr"] >= 20.0,
        "mean_lpips_at_most_0.40": candidate_perceptual["lpips"] <= 0.40,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_perceptual": candidate_perceptual,
    }


def labeled(frame: torch.Tensor, label: str) -> Image.Image:
    """Create one compact qualitative panel."""
    pixels = (frame.clamp(0, 1) * 255).round().byte().cpu().numpy()
    image = Image.fromarray(pixels)
    canvas = Image.new("RGB", (image.width, image.height + 24), "black")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 6), label, fill="white")
    return canvas


def audit_arm_labels(candidate_count: int) -> list[str]:
    """Return the stable visual/report order for every audited arm."""
    if candidate_count < 1:
        raise ValueError("an audit needs at least one candidate")
    return [
        "lattice",
        *(f"irregular_seed{seed}" for seed in range(candidate_count)),
        "teacher",
    ]


def audit_display_labels(
    candidate_count: int, candidate_labels: list[str] | None = None
) -> dict[str, str]:
    """Map stable report keys to optional human-readable visual labels."""
    labels = candidate_labels or []
    if labels and len(labels) != candidate_count:
        raise ValueError("candidate labels must match candidate checkpoints")
    return {
        "lattice": "lattice",
        **{
            f"irregular_seed{seed}": (
                labels[seed] if labels else f"irregular_seed{seed}"
            )
            for seed in range(candidate_count)
        },
        "teacher": "teacher",
    }


def layout_slice(
    features: torch.Tensor,
    fixed_axis: int,
    band: float = 0.12,
    max_points: int = 10000,
) -> torch.Tensor:
    """Select visible jewels near one coordinate plane for a layout-only plot."""
    opacity = torch.sigmoid(features[:, 21])
    selected = features[(opacity > 0.02) & (features[:, fixed_axis].abs() <= band), :3]
    if len(selected) > max_points:
        index = torch.linspace(0, len(selected) - 1, max_points).long()
        selected = selected[index]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--candidate-label", action="append")
    parser.add_argument("--teacher-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--point-chunk", type=int, default=512)
    parser.add_argument("--support-capacity", type=int, default=2048)
    parser.add_argument("--teacher-support-capacity", type=int, default=16384)
    args = parser.parse_args()
    output, device = Path(args.out), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    validation = {
        item["source_id"]: item for item in manifest["examples"]
        if item["split"] == "validation"
    }
    teacher_paths = {}
    for path in Path(args.teacher_root).glob("*.pt"):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        source_id = saved.get("source", {}).get("source_id")
        if source_id in validation:
            teacher_paths[source_id] = path
    if not teacher_paths:
        raise ValueError("no validation teachers match the manifest")

    baseline = load_baseline(Path(args.baseline), device)
    candidates = [load_candidate(Path(path), device) for path in args.candidate]
    display_labels = audit_display_labels(len(candidates), args.candidate_label)
    metric = lpips_metric(device)
    perceptual_records, structure_records = [], []
    qualitative_rows = []
    layout_features: dict[str, torch.Tensor] = {}

    for source_id in sorted(teacher_paths):
        item = validation[source_id]
        video = load_video(
            item["video"], max_frames=int(item["frames"]),
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width), device="cpu",
        )
        indices = torch.linspace(0, len(video) - 1, args.frames).long()
        points = frame_points(
            len(video), indices, args.height, args.width, device=device
        )
        target = video[indices]
        arms = {}
        with torch.no_grad():
            baseline_prediction = baseline(video.to(device))
            arms["lattice"] = cholesky_render(
                baseline_prediction["centers"], baseline_prediction["cholesky"],
                baseline_prediction["colors"], baseline_prediction["color_grads"],
                baseline_prediction["logit_w"], points,
                baseline_prediction["background"], point_chunk=args.point_chunk,
                cull_mode="support_tiled", support_capacity=1024,
            ).reshape(args.frames, args.height, args.width, 3).cpu()
            structure_records.append({
                "arm": "lattice", "seed": 0, "source_id": source_id,
                **structure(baseline.canonical_features(baseline_prediction).cpu()),
            })
            if not layout_features:
                layout_features["lattice"] = baseline.canonical_features(
                    baseline_prediction
                ).cpu()
            for seed, candidate in enumerate(candidates):
                prediction = candidate(video.to(device))
                label = f"irregular_seed{seed}"
                arms[label] = render_structural(
                    prediction, points, point_chunk=args.point_chunk,
                    cull_mode="support_tiled",
                    support_capacity=args.support_capacity,
                ).reshape(args.frames, args.height, args.width, 3).cpu()
                structure_records.append({
                    "arm": "irregular", "seed": seed, "source_id": source_id,
                    **structure(candidate.canonical_features(prediction).cpu()),
                })
                if label not in layout_features:
                    layout_features[label] = candidate.canonical_features(
                        prediction
                    ).cpu()
            teacher, teacher_saved = load_teacher(teacher_paths[source_id], device)
            background = torch.as_tensor(
                teacher_saved["info"]["background"], device=device
            )
            arms["teacher"] = render_points(
                teacher, points, cull_mode="support_tiled", support_sigma=5.0,
                support_capacity=args.teacher_support_capacity,
                support_point_chunk=args.point_chunk, support_base_resolution=32,
                support_level_scale=1.55, background=background,
            ).reshape(args.frames, args.height, args.width, 3).cpu()
            if "teacher" not in layout_features:
                teacher_features = torch.zeros(len(teacher.mu), 22)
                teacher_features[:, :3] = teacher.mu.detach().cpu()
                teacher_features[:, 21] = teacher.logit_w.detach().cpu()
                layout_features["teacher"] = teacher_features

        scored = score_arms(target, arms, metric)
        for arm, row in scored.items():
            perceptual_records.append({
                "source_id": source_id,
                "style": item.get("style"),
                "arm": arm,
                **row,
            })
        middle = args.frames // 2
        panels = [labeled(target[middle], f"{item.get('style')}: target")]
        for arm in audit_arm_labels(len(candidates)):
            panels.append(labeled(
                arms[arm][middle], f"{item.get('style')}: {display_labels[arm]}"
            ))
        row_image = Image.new(
            "RGB", (sum(panel.width for panel in panels), panels[0].height), "white"
        )
        offset = 0
        for panel in panels:
            row_image.paste(panel, (offset, 0)); offset += panel.width
        qualitative_rows.append(row_image)
        print("audited", source_id, flush=True)

    perceptual_macro = {}
    for arm in sorted({row["arm"] for row in perceptual_records}):
        rows = [row for row in perceptual_records if row["arm"] == arm]
        perceptual_macro[arm] = {
            "lpips": statistics.mean(row["lpips_mean"] for row in rows),
            "psnr": statistics.mean(row["render_signature"]["psnr"] for row in rows),
            "ssim": statistics.mean(row["render_signature"]["ssim"] for row in rows),
            "layout_psnr": statistics.mean(
                row["layout_signature"]["layout_psnr"] for row in rows
            ),
            "layout_ssim": statistics.mean(
                row["layout_signature"]["layout_ssim"] for row in rows
            ),
        }
    structure_macro = {}
    for arm in ("lattice", "irregular"):
        rows = [row for row in structure_records if row["arm"] == arm]
        structure_macro[arm] = {
            key: statistics.mean(row[key] for row in rows)
            for key in (
                "active_fraction", "anisotropy_median", "anisotropy_p90",
                "extent_iqr_ratio", "occupancy_uniformity",
                "mixed_spacetime_tilt_median", "mixed_spacetime_tilt_p90",
            )
        }
    structure_by_seed = {}
    for seed in range(len(candidates)):
        rows = [
            row for row in structure_records
            if row["arm"] == "irregular" and row["seed"] == seed
        ]
        structure_by_seed[f"seed{seed}"] = {
            key: statistics.mean(row[key] for row in rows)
            for key in (
                "active_fraction", "anisotropy_median", "anisotropy_p90",
                "extent_iqr_ratio", "occupancy_uniformity",
                "mixed_spacetime_tilt_median", "mixed_spacetime_tilt_p90",
            )
        }
    gate = summarize_gate(perceptual_macro, structure_macro, structure_by_seed)
    report = {
        "schema": "irregular-encoder-audit-v1",
        "protocol": {
            "sources": sorted(teacher_paths), "frames": args.frames,
            "candidate_checkpoints": args.candidate,
            "candidate_labels": args.candidate_label,
            "renderer": "support_tiled", "support_sigma": 5.0,
        },
        "perceptual_macro": perceptual_macro,
        "structure_macro": structure_macro,
        "structure_by_seed": structure_by_seed,
        "gate": gate,
        "perceptual_records": perceptual_records,
        "structure_records": structure_records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    sheet = Image.new(
        "RGB", (qualitative_rows[0].width, sum(row.height for row in qualitative_rows)),
        "white",
    )
    offset = 0
    for row in qualitative_rows:
        sheet.paste(row, (0, offset)); offset += row.height
    sheet.save(output / "qualitative.png")

    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = audit_arm_labels(len(candidates))
    visual_labels = [display_labels[label] for label in labels]
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].bar(
        visual_labels, [perceptual_macro[label]["lpips"] for label in labels]
    )
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[0].axhline(0.40, color="tab:red", linestyle="--", linewidth=1)
    axes[1].bar(
        visual_labels, [perceptual_macro[label]["psnr"] for label in labels]
    )
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].axhline(20.0, color="tab:red", linestyle="--", linewidth=1)
    structure_labels = ["lattice", *structure_by_seed]
    axes[2].bar(
        structure_labels,
        [
            structure_macro["lattice"]["occupancy_uniformity"],
            *(row["occupancy_uniformity"] for row in structure_by_seed.values()),
        ],
    )
    axes[2].set_ylabel("Occupancy uniformity (lower is irregular)")
    axes[2].set_ylim(0.95, 1.001)
    axes[2].axhline(0.985, color="tab:red", linestyle="--", linewidth=1)
    axes[3].bar(
        structure_labels,
        [
            structure_macro["lattice"]["active_fraction"],
            *(row["active_fraction"] for row in structure_by_seed.values()),
        ],
    )
    axes[3].set_ylabel("Active proposal fraction")
    axes[3].set_ylim(0, 1.05)
    axes[3].axhline(0.70, color="tab:red", linestyle="--", linewidth=1)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout(); figure.savefig(output / "comparison.png", dpi=180)

    layout_arms = audit_arm_labels(len(candidates))
    layout_figure, layout_axes = plt.subplots(
        2, len(layout_arms), figsize=(3.65 * len(layout_arms), 7.2), squeeze=False,
    )
    for column, arm in enumerate(layout_arms):
        xy = layout_slice(layout_features[arm], fixed_axis=2)
        xt = layout_slice(layout_features[arm], fixed_axis=1)
        layout_axes[0, column].scatter(
            xy[:, 0], xy[:, 1], c=xy[:, 2], s=2, alpha=0.35,
            cmap="coolwarm", rasterized=True,
        )
        layout_axes[1, column].scatter(
            xt[:, 0], xt[:, 2], c=xt[:, 1], s=2, alpha=0.35,
            cmap="viridis", rasterized=True,
        )
        layout_axes[0, column].set_title(display_labels[arm])
        layout_axes[0, column].set_xlabel("x")
        layout_axes[0, column].set_ylabel("y")
        layout_axes[1, column].set_xlabel("x")
        layout_axes[1, column].set_ylabel("time")
        for axis in layout_axes[:, column]:
            axis.set_xlim(-1, 1); axis.set_ylim(-1, 1)
            axis.set_aspect("equal"); axis.grid(alpha=0.12)
    layout_figure.suptitle(
        "Active jewel centers: XY near t=0 (top), XT near y=0 (bottom)"
    )
    layout_figure.tight_layout(); layout_figure.savefig(
        output / "field_layout.png", dpi=200
    )
    print(json.dumps({
        "perceptual_macro": perceptual_macro,
        "structure_macro": structure_macro,
        "gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
