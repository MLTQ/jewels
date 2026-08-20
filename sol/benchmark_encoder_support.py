"""Benchmark support-complete encoder training against the all-center oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.streaming import frame_times
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def fixed_points(frames: int, height: int, width: int, count: int, device: str) -> torch.Tensor:
    """Return a deterministic flat traversal of video voxel coordinates."""
    linear = torch.arange(count, device=device) % (frames * height * width)
    t = linear // (height * width)
    y = (linear // width) % height
    x = linear % width
    return torch.stack(
        (
            torch.linspace(-1, 1, width, device=device)[x],
            torch.linspace(-1, 1, height, device=device)[y],
            frame_times(frames, device=device)[t],
        ),
        dim=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--support-capacity", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = saved["meta"]
    model = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024), **meta["model_args"]
    ).to(device)
    model.load_state_dict(saved["model"])
    video = load_video(
        args.video,
        max_frames=args.frames,
        resize=(args.height, args.width),
        device="cpu",
    ).to(device)
    points = fixed_points(
        len(video), args.height, args.width, args.points, args.device
    )
    target = video.reshape(-1, 3)[: args.points]

    def render(prediction: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        return cholesky_render(
            prediction["centers"], prediction["cholesky"], prediction["colors"],
            prediction["color_grads"], prediction["logit_w"], points,
            prediction["background"], point_chunk=args.point_chunk,
            cull_mode=mode, support_capacity=args.support_capacity,
        )

    with torch.no_grad():
        prediction = model(video)
        oracle = render(prediction, "exact")
        support = render(prediction, "support_tiled")
        max_abs_vs_infinite_oracle = float((support - oracle).abs().max())

    results = {}
    for mode in ("exact", "support_tiled"):
        samples = []
        peaks = []
        for repeat in range(args.warmup + args.repeats):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(video)
            loss = torch.nn.functional.mse_loss(render(prediction, mode), target)
            loss.backward()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if repeat >= args.warmup:
                samples.append(elapsed)
                peaks.append(torch.cuda.max_memory_allocated(device) / 2**30)
        results[mode] = {
            "seconds": samples,
            "median_seconds": float(torch.tensor(samples).median()),
            "peak_allocated_gib": max(peaks),
        }

    report = {
        "schema": "encoder-support-benchmark-v1",
        "checkpoint": args.checkpoint,
        "video": args.video,
        "points": args.points,
        "support_sigma": 5.0,
        "support_capacity": args.support_capacity,
        "max_abs_vs_infinite_oracle": max_abs_vs_infinite_oracle,
        "results": results,
        "speedup": results["exact"]["median_seconds"]
        / results["support_tiled"]["median_seconds"],
        "gpu": torch.cuda.get_device_name(device),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["all-center", "support tiled"]
    modes = ["exact", "support_tiled"]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    time_bars = axes[0].bar(
        labels, [results[mode]["median_seconds"] for mode in modes]
    )
    axes[0].bar_label(time_bars, fmt="%.3f s")
    axes[0].set_ylabel("Forward + loss + backward (s)")
    memory_bars = axes[1].bar(
        labels, [results[mode]["peak_allocated_gib"] for mode in modes]
    )
    axes[1].bar_label(memory_bars, fmt="%.2f GiB")
    axes[1].set_ylabel("Peak allocated GPU memory (GiB)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Support-correct encoder training cost (73,728 splats)")
    figure.tight_layout()
    figure.savefig(output.parent / "benchmark.png", dpi=180)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
