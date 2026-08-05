"""Render target and prefix-control fields for a learned jewel continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from sol.corpus import _state_to_features
from sol.render import render_exact
from sol.streaming import frame_times
from sol.streaming_continuation_eval import _nonoverlapping_context_index
from sol.streaming_data import BirthTarget, build_continuation_dataset, rasterize_context
from sol.streaming_features import to_global_time
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


def frame_points(
    total_frames: int,
    frame_indices: torch.Tensor,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Create a low-resolution global `(u,v,t)` render grid."""
    times = frame_times(total_frames)[frame_indices].to(device)
    vertical = torch.linspace(-1, 1, height, device=device)
    horizontal = torch.linspace(-1, 1, width, device=device)
    tt, vv, uu = torch.meshgrid(times, vertical, horizontal, indexing="ij")
    return torch.stack((uu, vv, tt), dim=-1).reshape(-1, 3)


def _panel(frame: torch.Tensor, label: str, upscale: int) -> Image.Image:
    pixels = (frame.clamp(0, 1) * 255).round().byte().numpy()
    image = Image.fromarray(pixels).resize(
        (frame.shape[1] * upscale, frame.shape[0] * upscale),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (image.width, image.height + 20), "black")
    canvas.paste(image, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="white")
    return canvas


def _row(images: list[Image.Image], pad: int = 2) -> Image.Image:
    width = sum(image.width for image in images) + pad * (len(images) - 1)
    output = Image.new("RGB", (width, max(image.height for image in images)), "white")
    offset = 0
    for image in images:
        output.paste(image, (offset, 0))
        offset += image.width + pad
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--fit-checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--upscale", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.height, args.width, args.upscale) <= 0:
        raise ValueError("render dimensions and upscale must be positive")
    device = torch.device(args.device)
    saved = torch.load(args.continuation, map_location="cpu", weights_only=False)
    meta = saved["meta"]
    fit_path = args.fit_checkpoint or meta["source_checkpoint"]
    fitted = torch.load(fit_path, map_location="cpu", weights_only=False)
    features = _state_to_features(fitted["state"]).float()
    total_frames = int(fitted["info"]["shape"][0])
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    dataset = build_continuation_dataset(
        features,
        total_frames,
        prefix_frames=int(meta["prefix_frames"]),
        stride_frames=int(meta["stride_frames"]),
        support_sigma=float(meta["support_sigma"]),
        grid_spec=spec,
    )
    if not 0 <= args.view < len(dataset.views):
        raise ValueError(f"view must be in [0,{len(dataset.views) - 1}]")
    model = BirthContinuationModel(grid_spec=spec, **meta["model_args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    rasters = [
        rasterize_context(
            view.context_features,
            dataset.context_standardizer,
            prefix_frames=dataset.prefix_frames,
            stride_frames=dataset.stride_frames,
            grid_shape=spec.shape,
        ).to(device)
        for view in dataset.views
    ]
    contexts = [model.encode_context(raster) for raster in rasters]
    view = dataset.views[args.view]
    births = view.births
    target = BirthTarget(
        values=dataset.birth_standardizer.normalize(births.values).to(device),
        cell_indices=births.cell_indices.to(device),
        slot_indices=births.slot_indices.to(device),
        counts=births.counts.to(device),
        global_ids=births.global_ids.to(device),
        birth_frames=births.birth_frames.to(device),
    )
    shuffled_index = _nonoverlapping_context_index(
        dataset.views, args.view, dataset.prefix_frames
    )
    conditions = {
        "correct prefix": contexts[args.view],
        "shuffled prefix": contexts[shuffled_index],
        "null prefix": torch.zeros_like(contexts[args.view]),
    }
    fields = {"fitted target": view.target_active_global_features.to(device)}
    for name, context in conditions.items():
        output = model.forward_from_context(context, target)
        predicted_local = dataset.birth_standardizer.denormalize(
            output.occupied_features
        )
        predicted_global = to_global_time(
            predicted_local,
            total_frames,
            view.frontier,
            dataset.stride_frames,
        )
        fields[name] = torch.cat(
            (view.carried_global_features.to(device), predicted_global), dim=0
        )

    frame_indices = torch.arange(view.frontier, view.commit_stop)
    points = frame_points(
        total_frames,
        frame_indices,
        args.height,
        args.width,
        device=device,
    )
    background = torch.tensor(fitted["info"]["background"], device=device)
    rendered = {
        name: render_exact(field, points, background=background)
        .reshape(len(frame_indices), args.height, args.width, 3)
        .cpu()
        for name, field in fields.items()
    }
    names = list(rendered)
    frames = [
        _row([_panel(rendered[name][i], name, args.upscale) for name in names])
        for i in range(len(frame_indices))
    ]
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_dir / "continuation_controls.gif",
        save_all=True,
        append_images=frames[1:],
        duration=83,
        loop=0,
    )
    picks = (0, len(frames) // 2, len(frames) - 1)
    contact = Image.new(
        "RGB",
        (frames[0].width, sum(frames[index].height for index in picks)),
        "white",
    )
    offset = 0
    for index in picks:
        contact.paste(frames[index], (0, offset))
        offset += frames[index].height
    contact.save(output_dir / "continuation_controls_contact.png")

    target_render = rendered["fitted target"]
    report = {
        "continuation": args.continuation,
        "fit_checkpoint": fit_path,
        "view": args.view,
        "frontier": view.frontier,
        "commit_stop": view.commit_stop,
        "shuffled_view": shuffled_index,
        "render_shape": [len(frame_indices), args.height, args.width],
        "field_psnr": {
            name: float(
                -10
                * torch.log10(
                    F.mse_loss(value, target_render).clamp_min(1e-10)
                )
            )
            for name, value in rendered.items()
            if name != "fitted target"
        },
    }
    (output_dir / "visual_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
