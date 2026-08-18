"""Gate the text-conditioned latent prior at the rendered level."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.audit_prompted_washout import render_signature
from sol.latent_text_prior import LatentStandardizer, LatentTextPrior
from sol.perceptual_eval import layout_signature
from sol.streaming import frame_times
from sol.token_grid import GridSpec
from sol.train_latent_text_prior import unpack
from stprim.data.video_io import load_video


def render_latent(
    encoder: VideoToJewelEncoder,
    packed: torch.Tensor,
    standardizer: LatentStandardizer,
    cell_dim: int,
    slots: int,
    *,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode one generated latent into a rendered video."""
    cells, seed = unpack(standardizer.denormalize(packed.cpu()), cell_dim, slots)
    prediction = encoder.decode(
        {"cells": cells.to(device), "seed": seed.clamp(1e-3, 1 - 1e-3).to(device)}
    )
    times = frame_times(frames, device=device)
    vertical = torch.linspace(-1, 1, height, device=device)
    horizontal = torch.linspace(-1, 1, width, device=device)
    tt, vv, uu = torch.meshgrid(times, vertical, horizontal, indexing="ij")
    points = torch.stack((uu, vv, tt), dim=-1).reshape(-1, 3)
    rendered = cholesky_render(
        prediction["centers"],
        prediction["cholesky"],
        prediction["colors"],
        prediction["color_grads"],
        prediction["logit_w"],
        points,
        prediction["background"],
        point_chunk=512,
    )
    return rendered.reshape(frames, height, width, 3).clamp(0, 1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    saved = torch.load(args.prior, map_location=device, weights_only=False)
    meta = saved["meta"]
    prior = LatentTextPrior(**meta["model_args"]).to(device)
    prior.load_state_dict(saved["model"])
    prior.eval()
    standardizer = LatentStandardizer.from_state_dict(meta["standardizer"])

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
    records = cache["records"]
    text_tokens = cache["text_tokens"].float().to(device)
    text_mask = cache["text_mask"].to(device)
    validation = [
        (i, r) for i, r in enumerate(records) if r["split"] != "train"
    ][: args.limit]
    slots = int(encoder_meta["slots_per_cell"])
    cell_dim = int(meta["model_args"]["cell_dim"])

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in validation:
        item = manifest[record["source_id"]]
        target = load_video(
            item["video"],
            max_frames=args.frames,
            start_frame=0,
            resize=(args.height, args.width),
            device="cpu",
        )
        correct = record["prompt_index"]
        alternate = records[(index + 7) % len(records)]["prompt_index"]
        arms = {}
        for name, prompt in (("correct", correct), ("shuffled", alternate)):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            latent = prior.sample(
                text_tokens[prompt : prompt + 1],
                text_mask[prompt : prompt + 1],
                steps=args.steps,
                guidance=args.guidance,
                generator=generator,
                device=device,
            )[0]
            video = render_latent(
                encoder, latent, standardizer, cell_dim, slots,
                frames=args.frames, height=args.height, width=args.width,
                device=device,
            )
            arms[name] = {
                "render_signature": asdict(render_signature(video, target)),
                "layout_signature": layout_signature(
                    video, target, min(8, args.height // 2, args.width // 2)
                ),
            }
        results.append(
            {
                "source_id": record["source_id"],
                "style": record["style"],
                "class_name": record["class_name"],
                "prompt": record["prompt"],
                "arms": arms,
            }
        )
        print(
            record["source_id"],
            "correct", round(arms["correct"]["render_signature"]["psnr"], 2),
            "shuffled", round(arms["shuffled"]["render_signature"]["psnr"], 2),
            flush=True,
        )

    macro = {
        name: {
            "psnr": sum(r["arms"][name]["render_signature"]["psnr"] for r in results)
            / len(results),
            "ssim": sum(r["arms"][name]["render_signature"]["ssim"] for r in results)
            / len(results),
            "layout_psnr": sum(
                r["arms"][name]["layout_signature"]["layout_psnr"] for r in results
            )
            / len(results),
        }
        for name in ("correct", "shuffled")
    }
    macro["correct_minus_shuffled_psnr"] = (
        macro["correct"]["psnr"] - macro["shuffled"]["psnr"]
    )
    report = {
        "schema": "latent-text-prior-render-gate-v1",
        "protocol": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "sampling_steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "note": "correct vs shuffled prompt, identical noise seed per pair",
        },
        "inputs": {"prior": args.prior, "latents": args.latents},
        "macro": macro,
        "records": results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(macro, indent=1))


if __name__ == "__main__":
    main()
