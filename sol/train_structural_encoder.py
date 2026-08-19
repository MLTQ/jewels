"""Train the scarce, tube-capable structural encoder with render loss only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from sol.compare_field_structure import structure_report
from sol.structural_encoder import (
    ARCHITECTURE,
    StructuralJewelEncoder,
    render_structural,
)
from sol.token_grid import GridSpec
from sol.train_amortized_encoder import sample_voxels
from stprim.data.video_io import load_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots-per-cell", type=int, default=5)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--points-per-step", type=int, default=8192)
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
    train_ids = [k for k, (_, s) in videos.items() if s == "train"]
    validation_ids = [k for k, (_, s) in videos.items() if s == "validation"]
    if not train_ids or not validation_ids:
        raise ValueError("manifest must provide train and validation examples")

    model = StructuralJewelEncoder(
        grid_spec=spec,
        slots_per_cell=args.slots_per_cell,
        model_dim=args.model_dim,
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
            rendered = render_structural(
                prediction, points, point_chunk=args.point_chunk
            )
            mse = float(torch.nn.functional.mse_loss(rendered, target))
            report[source_id] = {
                "mse": mse,
                "psnr": -10.0 * math.log10(max(mse, 1e-10)),
            }
            structures.append(
                structure_report(model.canonical_features(prediction).cpu())
            )
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
                "jewels_above_2pct_opacity",
            )
        }
        model.train()
        return report

    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"train={len(train_ids)} validation={len(validation_ids)} "
        f"jewels_per_window={model.n_jewels} model={parameters / 1e6:.2f}M",
        flush=True,
    )
    latest = None
    history: list[float] = []
    started = time.time()
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
        rendered = render_structural(prediction, points, point_chunk=args.point_chunk)
        loss = torch.nn.functional.mse_loss(rendered, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            recent = history[-args.log_every :]
            mse = sum(recent) / len(recent)
            record = {
                "step": step,
                "loss": mse,
                "train_psnr": -10.0 * math.log10(max(mse, 1e-10)),
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
                        "structure": {
                            k: round(v, 4) for k, v in latest["structure"].items()
                        },
                    }
                ),
                flush=True,
            )
        if step % args.checkpoint_every == 0 or step == args.steps:
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
                        "manifest": args.manifest,
                        "train_args": vars(args),
                        "latest_evaluation": latest,
                    },
                },
                output_dir / "encoder.pt",
            )
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "parameters": parameters,
        "jewels_per_window": model.n_jewels,
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
