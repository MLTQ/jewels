"""Build the real fitted-Jewel isolation clip used by explainer episode 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw
import torch


STPRIM_ROOT = Path(__file__).resolve().parent.parent / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import PrimitiveField  # noqa: E402
from core.volume import make_grid  # noqa: E402
from models.render import render_volume  # noqa: E402


EGGSHELL = torch.tensor((244, 240, 230), dtype=torch.float32) / 255.0
LABEL_COLORS = ("#087f6d", "#b52d68", "#ae4f0b", "#2463a7")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--display-gain", type=float, default=8.0)
    return parser.parse_args()


def _field_from_state(
    state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    indices: torch.Tensor | None = None,
) -> PrimitiveField:
    selected = state if indices is None else {
        key: value[indices] for key, value in state.items()
    }
    p1_color = "color_grad" in selected and selected["color_grad"] is not None
    field = PrimitiveField(
        selected["mu"].shape[0], p1_color=p1_color, device="cpu"
    )
    field.load_state_dict(selected)
    return field.to(device)


def covariance_and_velocity(
    field: PrimitiveField,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return covariance, conditional screen velocity, and 2-D slice covariance."""
    rotation = field.rotations()
    scale_squared = field.scales().square()
    covariance = rotation @ torch.diag_embed(scale_squared) @ rotation.transpose(1, 2)
    time_variance = covariance[:, 2, 2].clamp_min(1e-10)
    cross = covariance[:, :2, 2]
    velocity = cross / time_variance[:, None]
    slice_covariance = covariance[:, :2, :2] - (
        cross[:, :, None] * cross[:, None, :] / time_variance[:, None, None]
    )
    return covariance, velocity, slice_covariance


@torch.no_grad()
def select_visible_moving_jewels(
    field: PrimitiveField,
    *,
    count: int,
    t_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select strong, visible splats using a declared non-manual ranking."""
    covariance, velocity, slice_covariance = covariance_and_velocity(field)
    time_sigma = covariance[:, 2, 2].clamp_min(1e-10).sqrt()
    motion = velocity.norm(dim=1)
    determinant = torch.linalg.det(slice_covariance).clamp_min(1e-12)
    color_energy = field.color.detach().square().mean(dim=1).sqrt()
    proxy = (
        field.weights().detach()
        * determinant.sqrt()
        * time_sigma
        * color_energy
        * (1.0 + 0.45 * motion.clamp(max=2.0))
    )
    eligible = (
        (field.mu[:, :2].abs().amax(dim=1) <= 0.88)
        & (time_sigma >= 0.08)
        & (motion >= 0.015)
    )
    eligible_indices = eligible.nonzero(as_tuple=False).flatten()
    if eligible_indices.numel() < count:
        raise ValueError(f"only {eligible_indices.numel()} jewels satisfy visibility filters")
    pool_count = min(192, eligible_indices.numel())
    pool = eligible_indices[proxy[eligible_indices].topk(pool_count).indices]

    # Measure actual rendered contribution on a coarse spacetime grid instead
    # of treating scale or opacity alone as visibility.
    coarse_grid = make_grid((16, 40, 77), t_scale=t_scale, device=field.device)
    gathered = field.gather(pool.expand(coarse_grid.shape[0], -1))
    offset = coarse_grid[:, None, :] - gathered["mu"]
    local = torch.einsum("mkji,mkj->mki", gathered["rot"], offset)
    local = local / (gathered["scale"] + 1e-8)
    q = local.square().sum(dim=-1)
    weight = torch.exp(
        -0.5 * q + torch.nn.functional.logsigmoid(gathered["logit_w"])
    )
    color = gathered["color"]
    if field.p1_color:
        color = color + torch.einsum(
            "mkij,mkj->mki", gathered["color_grad"], offset
        )
    contribution = weight[..., None] * color
    energy = contribution.square().mean(dim=(0, 2)).sqrt()
    temporal_energy = contribution.reshape(16, 40, 77, pool_count, 3).square()
    temporal_energy = temporal_energy.mean(dim=(1, 2, 4)).sqrt()
    score = energy * (1.0 + 0.25 * motion[pool].clamp(max=2.0))

    # Choose one strong, long-lived contribution near each of four time anchors.
    # This reflects the field's temporal locality while keeping the complete
    # 64-frame pass populated instead of cherry-picking one busy moment.
    selected: list[int] = []
    anchors = torch.linspace(-0.78 * t_scale, 0.78 * t_scale, count, device=field.device)
    coarse_times = torch.linspace(-t_scale, t_scale, 16, device=field.device)
    for anchor in anchors:
        time_index = int((coarse_times - anchor).abs().argmin())
        anchored_score = temporal_energy[time_index] * (
            1.0 + 0.25 * motion[pool].clamp(max=2.0)
        )
        for pool_position in anchored_score.argsort(descending=True).tolist():
            candidate = int(pool[pool_position])
            if candidate in selected or float(time_sigma[candidate]) < 0.12:
                continue
            if abs(float(anchor - field.mu[candidate, 2])) > 2.25 * float(
                time_sigma[candidate]
            ):
                continue
            center = field.mu[candidate, :2] + velocity[candidate] * (
                anchor - field.mu[candidate, 2]
            )
            if float(center.abs().amax()) > 0.92:
                continue
            if all(
                float(torch.linalg.vector_norm(center - (
                    field.mu[other, :2]
                    + velocity[other] * (anchor - field.mu[other, 2])
                ))) >= 0.14
                for other in selected
            ):
                selected.append(candidate)
                break
    if len(selected) < count:
        for candidate in pool[score.argsort(descending=True)].tolist():
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == count:
                break
    indices = torch.tensor(selected, dtype=torch.long, device=field.device)
    diagnostics = {
        "proxy": proxy,
        "energy": torch.zeros(len(field), device=field.device).index_copy(0, pool, energy),
        "motion": motion,
        "velocity": velocity,
        "covariance": covariance,
        "slice_covariance": slice_covariance,
    }
    return indices, diagnostics


def _to_image(frame: torch.Tensor, upscale: int) -> Image.Image:
    array = (frame.clamp(0, 1).cpu().numpy() * 255).round().astype("uint8")
    image = Image.fromarray(array, "RGB")
    if upscale > 1:
        image = image.resize(
            (image.width * upscale, image.height * upscale),
            Image.Resampling.LANCZOS,
        )
    return image


def _screen_point(
    normalized: torch.Tensor, width: int, height: int, upscale: int
) -> tuple[float, float]:
    return (
        float((normalized[0] + 1.0) * 0.5 * (width - 1) * upscale),
        float((normalized[1] + 1.0) * 0.5 * (height - 1) * upscale),
    )


def _decorate(
    image: Image.Image,
    *,
    frame_index: int,
    shape: tuple[int, int, int],
    field: PrimitiveField,
    selected: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    t_scale: float,
    upscale: int,
    alpha: float,
) -> Image.Image:
    if alpha <= 0:
        return image
    frames, height, width = shape
    time = -t_scale + (2.0 * t_scale * frame_index / max(frames - 1, 1))
    draw = ImageDraw.Draw(image)
    for order, primitive_index in enumerate(selected.tolist()):
        temporal_delta = time - float(field.mu[primitive_index, 2])
        temporal_sigma = math.sqrt(
            max(float(diagnostics["covariance"][primitive_index, 2, 2]), 1e-10)
        )
        if abs(temporal_delta) > 3.2 * temporal_sigma:
            continue
        center = (
            field.mu[primitive_index, :2]
            + diagnostics["velocity"][primitive_index] * temporal_delta
        )
        covariance = diagnostics["slice_covariance"][primitive_index]
        eigenvalue, eigenvector = torch.linalg.eigh(covariance)
        angles = torch.linspace(0, 2 * math.pi, 49, device=field.device)
        unit = torch.stack((angles.cos(), angles.sin()), dim=0)
        ellipse = center[:, None] + eigenvector @ (
            2.35 * eigenvalue.clamp_min(1e-10).sqrt()[:, None] * unit
        )
        polygon = [
            _screen_point(ellipse[:, point], width, height, upscale)
            for point in range(ellipse.shape[1])
        ]
        color = LABEL_COLORS[order % len(LABEL_COLORS)]
        draw.line(
            polygon,
            fill=color,
            width=max(2, round(3 * upscale * alpha)),
            joint="curve",
        )
        x, y = _screen_point(center, width, height, upscale)
        radius = 11 * upscale
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="#f4f0e6",
            width=max(1, upscale),
        )
        draw.text((x, y), str(order + 1), fill="#ffffff", anchor="mm")
    return image


def _isolated_display(contribution: torch.Tensor, gain: float) -> torch.Tensor:
    """Matte exact selected contributions over eggshell with declared exposure."""
    positive = contribution.clamp_min(0)
    magnitude = positive.amax(dim=-1, keepdim=True)
    alpha = (magnitude * gain).clamp(0.0, 0.92)
    hue = positive / magnitude.clamp_min(1e-8)
    pigment = 0.10 + 0.90 * hue
    cream = EGGSHELL.to(contribution.device).view(1, 1, 1, 3)
    return cream * (1.0 - alpha) + pigment * alpha


def _timeline() -> list[tuple[str, int, float, float]]:
    timeline: list[tuple[str, int, float, float]] = []
    timeline.extend(("full", index, 0.0, 0.0) for index in range(12))
    timeline.extend(("identify", 12, 0.0, (index + 1) / 8) for index in range(8))
    timeline.extend(("fade-out", 12, (index + 1) / 12, 1.0) for index in range(12))
    timeline.extend(("isolated", index, 1.0, 1.0) for index in range(64))
    timeline.extend(
        (
            "fade-in",
            63,
            1.0 - (index + 1) / 12,
            1.0 - (index + 1) / 12,
        )
        for index in range(12)
    )
    return timeline


def _encode(frames: list[Image.Image], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{frames[0].width}x{frames[0].height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr}")


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.count != 4:
        raise ValueError("the episode layout currently requires exactly four jewels")
    if min(args.fps, args.upscale) <= 0 or args.display_gain <= 0:
        raise ValueError("fps, upscale, and display gain must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    shape = tuple(int(value) for value in checkpoint["info"]["shape"])
    if shape[0] != 64:
        raise ValueError(f"the explanatory timeline expects 64 fitted frames, got {shape[0]}")
    config = checkpoint.get("cfg", {})
    t_scale = float(config.get("t_scale", 1.0))
    device = torch.device(args.device)
    field = _field_from_state(state, device=device)
    selected, diagnostics = select_visible_moving_jewels(
        field, count=args.count, t_scale=t_scale
    )
    selected_field = _field_from_state(
        state, device=device, indices=selected.cpu()
    )
    grid = make_grid(shape, t_scale=t_scale, device=device)
    background = torch.tensor(checkpoint["info"]["background"], device=device)
    full = render_volume(
        field,
        grid,
        knn=int(config.get("knn", 64)),
        background=background,
    ).reshape(*shape, 3).clamp(0, 1)
    selected_contribution = render_volume(
        selected_field, grid, knn=len(selected_field)
    ).reshape(*shape, 3)
    isolated = _isolated_display(selected_contribution, args.display_gain)

    output_frames: list[Image.Image] = []
    timeline = _timeline()
    for stage, frame_index, isolate_amount, label_alpha in timeline:
        blended = full[frame_index] * (1.0 - isolate_amount) + isolated[frame_index] * isolate_amount
        image = _to_image(blended, args.upscale)
        output_frames.append(
            _decorate(
                image,
                frame_index=frame_index,
                shape=shape,
                field=field,
                selected=selected,
                diagnostics=diagnostics,
                t_scale=t_scale,
                upscale=args.upscale,
                alpha=label_alpha,
            )
        )

    _encode(output_frames, args.out, args.fps)
    picks = (0, 19, 37, 64, len(output_frames) - 1)
    contact = Image.new(
        "RGB",
        (output_frames[0].width * len(picks), output_frames[0].height),
        "#f4f0e6",
    )
    for column, pick in enumerate(picks):
        contact.paste(output_frames[pick], (column * output_frames[pick].width, 0))
    contact.save(args.out.with_name("actual_jewel_isolation_contact.png"))

    selection: list[dict[str, Any]] = []
    for order, primitive_index in enumerate(selected.tolist(), start=1):
        selection.append(
            {
                "label": order,
                "field_index": primitive_index,
                "label_color": LABEL_COLORS[order - 1],
                "mu_uvt": field.mu[primitive_index].detach().cpu().tolist(),
                "covariance": diagnostics["covariance"][primitive_index].detach().cpu().tolist(),
                "conditional_uv_velocity": diagnostics["velocity"][primitive_index].detach().cpu().tolist(),
                "coarse_render_energy": float(diagnostics["energy"][primitive_index]),
                "selection_proxy": float(diagnostics["proxy"][primitive_index]),
                "motion_score": float(diagnostics["motion"][primitive_index]),
            }
        )
    metadata = {
        "schema": "jewel-isolation-explainer-v1",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "field_shape": list(shape),
        "field_jewels": len(field),
        "selected_jewels": selection,
        "selection": "four timeline anchors; visible temporal extent + conditional motion; ranked by coarse exact rendered energy with label spacing",
        "full_render": {"culling": "checkpoint legacy center-KNN", "knn": int(config.get("knn", 64))},
        "isolated_render": {"culling": "all 4 selected primitives", "background": "eggshell matte", "display_gain": args.display_gain},
        "timeline": [
            {"output_frame": index, "stage": row[0], "source_frame": row[1], "isolate_amount": row[2]}
            for index, row in enumerate(timeline)
        ],
        "fps": args.fps,
        "output_frames": len(output_frames),
        "resolution": list(output_frames[0].size),
    }
    args.out.with_name("actual_jewel_isolation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
