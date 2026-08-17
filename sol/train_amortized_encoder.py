"""Train the feed-forward video-to-jewel encoder with stochastic-voxel render loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.prompt_embeddings import load_prompt_cache, manifest_digest
from sol.streaming import frame_times
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots-per-cell", type=int, default=36)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--points-per-step", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_voxels(
    video: torch.Tensor, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw random voxel centers and their RGB values in normalized coordinates."""
    frames, height, width, _ = video.shape
    t = torch.randint(0, frames, (count,), device=video.device, generator=generator)
    y = torch.randint(0, height, (count,), device=video.device, generator=generator)
    x = torch.randint(0, width, (count,), device=video.device, generator=generator)
    times = frame_times(frames, device=video.device)[t]
    vertical = torch.linspace(-1, 1, height, device=video.device)[y]
    horizontal = torch.linspace(-1, 1, width, device=video.device)[x]
    points = torch.stack((horizontal, vertical, times), dim=-1)
    return points, video[t, y, x]


def _atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.lr <= 0 or args.points_per_step <= 0:
        raise ValueError("training schedule is outside its valid range")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    spec = GridSpec(tuple(args.grid), 1024)
    videos = {}
    for example in manifest["examples"]:
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
    train_ids = [k for k, (_, split) in videos.items() if split == "train"]
    validation_ids = [k for k, (_, split) in videos.items() if split == "validation"]
    if not train_ids or not validation_ids:
        raise ValueError("manifest must provide train and validation examples")
    model = VideoToJewelEncoder(
        grid_spec=spec,
        slots_per_cell=args.slots_per_cell,
        model_dim=args.model_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    checkpoint_path = output_dir / "encoder.pt"

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        report = {}
        eval_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        for source_id in validation_ids:
            video = videos[source_id][0]
            prediction = model(video)
            points, target = sample_voxels(video, 16384, eval_generator)
            rendered = cholesky_render(
                prediction["centers"],
                prediction["cholesky"],
                prediction["colors"],
                prediction["color_grads"],
                prediction["logit_w"],
                points,
                prediction["background"],
            )
            mse = float(torch.nn.functional.mse_loss(rendered, target))
            report[source_id] = {
                "mse": mse,
                "psnr": -10.0 * math.log10(max(mse, 1e-10)),
            }
        report["macro_psnr"] = sum(row["psnr"] for row in report.values()) / len(
            validation_ids
        )
        model.train()
        return report

    def save(step: int, evaluation: dict | None) -> None:
        _atomic_save(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "video_to_jewel_encoder_v0",
                    "grid_shape": spec.shape,
                    "slots_per_cell": args.slots_per_cell,
                    "model_args": {
                        "slots_per_cell": args.slots_per_cell,
                        "model_dim": args.model_dim,
                    },
                    "manifest": args.manifest,
                    "manifest_sha256": manifest_digest(manifest),
                    "train_args": vars(args),
                    "latest_evaluation": evaluation,
                },
            },
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"jewels_per_window={spec.n_cells * args.slots_per_cell} "
        f"model={parameters / 1e6:.2f}M",
        flush=True,
    )
    latest = None
    history = []
    started = time.time()
    interval = started
    model.train()
    for step in range(1, args.steps + 1):
        source_id = train_ids[
            int(torch.randint(0, len(train_ids), (1,), generator=generator, device=device))
        ]
        video = videos[source_id][0]
        if step <= args.warmup:
            rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        prediction = model(video)
        points, target = sample_voxels(video, args.points_per_step, generator)
        rendered = cholesky_render(
            prediction["centers"],
            prediction["cholesky"],
            prediction["colors"],
            prediction["color_grads"],
            prediction["logit_w"],
            points,
            prediction["background"],
        )
        loss = torch.nn.functional.mse_loss(rendered, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            recent = history[-args.log_every :]
            mse = sum(recent) / len(recent)
            record = {
                "step": step,
                "loss": mse,
                "train_psnr": -10.0 * math.log10(max(mse, 1e-10)),
                "gradient_norm": float(gradient_norm),
                "lr": rate,
                "seconds_per_step": (now - interval) / len(recent),
            }
            interval = now
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate()
            record = {"step": step, "evaluation": latest}
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "parameters": parameters,
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
