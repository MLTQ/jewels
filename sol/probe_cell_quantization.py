"""Probe whether quantized cell latents still decode into watchable video.

Precondition for the LLM-emission architecture: a window must survive being
reduced to a short sequence of discrete cell codes. This measures the loss
against the unquantized encoder ceiling across codebook sizes and codes/cell.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.audit_prompted_washout import render_signature
from sol.perceptual_eval import layout_signature
from sol.streaming import frame_times
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def kmeans(values: torch.Tensor, codes: int, iterations: int = 25, seed: int = 0):
    """Plain Lloyd k-means; returns the codebook."""
    generator = torch.Generator(device=values.device).manual_seed(seed)
    start = torch.randperm(len(values), generator=generator, device=values.device)
    book = values[start[:codes]].clone()
    for _ in range(iterations):
        distance = torch.cdist(values, book)
        assignment = distance.argmin(dim=1)
        for index in range(codes):
            mask = assignment == index
            if bool(mask.any()):
                book[index] = values[mask].mean(dim=0)
    return book


def quantize(values: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return book[torch.cdist(values, book).argmin(dim=1)]


def split_groups(values: torch.Tensor, groups: int) -> list[torch.Tensor]:
    """Split the channel axis into `groups` contiguous product-quantizer blocks."""
    size = values.shape[-1] // groups
    return [values[..., i * size : (i + 1) * size] for i in range(groups)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codes", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=144)
    args = parser.parse_args()

    device = torch.device(args.device)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    records = cache["records"]
    encoder_meta = cache["encoder"]
    encoder_saved = torch.load(
        encoder_meta["checkpoint"], map_location=device, weights_only=False
    )
    encoder = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(encoder_meta["grid_shape"]), 1024),
        **encoder_meta["model_args"],
    ).to(device)
    encoder.load_state_dict(encoder_saved["model"])
    encoder.eval()
    manifest = {
        item["source_id"]: item
        for item in json.loads(Path(args.manifest).read_text())["examples"]
    }

    # Codebooks are fitted on TRAIN cells only, then applied to held-out windows.
    train_cells = torch.cat(
        [cache["cells"][i] for i, r in enumerate(records) if r["split"] == "train"]
    ).float()
    held = [(i, r) for i, r in enumerate(records) if r["split"] != "train"]
    by_style: dict[str, list] = {}
    for entry in held:
        by_style.setdefault(entry[1]["style"], []).append(entry)
    chosen, depth = [], 0
    while len(chosen) < args.windows and any(
        depth < len(v) for v in by_style.values()
    ):
        for style in sorted(by_style):
            if depth < len(by_style[style]) and len(chosen) < args.windows:
                chosen.append(by_style[style][depth])
        depth += 1

    times = frame_times(args.frames, device=device)
    vertical = torch.linspace(-1, 1, args.height, device=device)
    horizontal = torch.linspace(-1, 1, args.width, device=device)
    tt, vv, uu = torch.meshgrid(times, vertical, horizontal, indexing="ij")
    points = torch.stack((uu, vv, tt), dim=-1).reshape(-1, 3)

    def render(cells: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        prediction = encoder.decode(
            {"cells": cells.to(device), "seed": seed.to(device)}
        )
        out = cholesky_render(
            prediction["centers"],
            prediction["cholesky"],
            prediction["colors"],
            prediction["color_grads"],
            prediction["logit_w"],
            points,
            prediction["background"],
            point_chunk=512,
        )
        return out.reshape(args.frames, args.height, args.width, 3).clamp(0, 1).cpu()

    books: dict[tuple[int, int], list[torch.Tensor]] = {}
    for groups in args.groups:
        blocks = split_groups(train_cells, groups)
        for codes in args.codes:
            books[(groups, codes)] = [
                kmeans(block.to(device), codes).cpu() for block in blocks
            ]
            print(f"fitted codebook groups={groups} codes={codes}", flush=True)

    results = []
    for index, record in chosen:
        item = manifest[record["source_id"]]
        target = load_video(
            item["video"], max_frames=args.frames, start_frame=0,
            resize=(args.height, args.width), device="cpu",
        )
        cells = cache["cells"][index].float()
        seed = cache["seed"][index].float().clamp(1e-3, 1 - 1e-3)
        entry = {
            "source_id": record["source_id"],
            "style": record["style"],
            "class_name": record["class_name"],
            "arms": {},
        }
        exact = render(cells, seed)
        entry["arms"]["unquantized"] = {
            "render_signature": asdict(render_signature(exact, target)),
            "layout": layout_signature(exact, target, 8)["layout_psnr"],
            "tokens_per_window": None,
        }
        for (groups, codes), book in books.items():
            blocks = split_groups(cells, groups)
            rebuilt = torch.cat(
                [quantize(b.to(device), k.to(device)).cpu() for b, k in zip(blocks, book)],
                dim=-1,
            )
            video = render(rebuilt, seed)
            entry["arms"][f"g{groups}_c{codes}"] = {
                "render_signature": asdict(render_signature(video, target)),
                "layout": layout_signature(video, target, 8)["layout_psnr"],
                "tokens_per_window": cells.shape[0] * groups,
                "bits_per_window": cells.shape[0] * groups * (codes.bit_length() - 1),
            }
        results.append(entry)
        print(
            record["source_id"],
            {k: round(v["render_signature"]["psnr"], 2) for k, v in entry["arms"].items()},
            flush=True,
        )

    arms = list(results[0]["arms"])
    macro = {
        name: {
            "psnr": sum(r["arms"][name]["render_signature"]["psnr"] for r in results)
            / len(results),
            "ssim": sum(r["arms"][name]["render_signature"]["ssim"] for r in results)
            / len(results),
            "layout_psnr": sum(r["arms"][name]["layout"] for r in results) / len(results),
            "tokens_per_window": results[0]["arms"][name]["tokens_per_window"],
        }
        for name in arms
    }
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "schema": "cell-quantization-probe-v1",
                "protocol": {
                    "frames": args.frames,
                    "height": args.height,
                    "width": args.width,
                    "codebooks_fitted_on": "train split cells only",
                    "seed_left_unquantized": True,
                },
                "macro": macro,
                "records": results,
            },
            indent=1,
        )
    )
    print(json.dumps(macro, indent=1))


if __name__ == "__main__":
    main()
