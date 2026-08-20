"""Run a reproducible multi-size, multi-seed encoder convergence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _ids(manifest: dict, split: str) -> list[str]:
    return [item["source_id"] for item in manifest["examples"] if item["split"] == split]


def validate_nested_manifests(root: Path, sizes: list[int]) -> dict:
    """Prove train prefixes are nested and validation identity is frozen."""
    manifests = {
        size: json.loads((root / f"n{size}" / "manifest.json").read_text())
        for size in sizes
    }
    train_ids = {size: _ids(manifest, "train") for size, manifest in manifests.items()}
    validation_ids = {
        size: _ids(manifest, "validation") for size, manifest in manifests.items()
    }
    for previous, current in zip(sizes, sizes[1:]):
        if train_ids[current][:previous] != train_ids[previous]:
            raise ValueError(f"n{previous} is not a prefix of n{current}")
    baseline = validation_ids[sizes[0]]
    if any(validation_ids[size] != baseline for size in sizes[1:]):
        raise ValueError("validation identities differ across curve points")
    return {
        "train_sizes": sizes,
        "validation_count": len(baseline),
        "validation_sha256": hashlib.sha256("\n".join(baseline).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(12, 60, 180))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-root")
    parser.add_argument(
        "--resume-checkpoint",
        choices=("encoder.pt", "latest.pt"),
        default="encoder.pt",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-delta-db", type=float, default=0.03)
    parser.add_argument("--points-per-step", type=int, default=4096)
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument("--support-capacity", type=int, default=1024)
    args = parser.parse_args()

    root = Path(args.root)
    sizes = sorted(args.sizes)
    protocol = validate_nested_manifests(root, sizes)
    protocol.update(
        {
            "schema": "encoder-convergence-protocol-v2",
            "seeds": args.seeds,
            "renderer": "support_tiled",
            "resume_root": args.resume_root,
            "resume_checkpoint": args.resume_checkpoint,
            "lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "support_sigma": 5.0,
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "eval_every_epochs": args.eval_every_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta_db": args.early_stop_min_delta_db,
            "points_per_step": args.points_per_step,
            "point_chunk": args.point_chunk,
            "support_capacity": args.support_capacity,
            "runs": [],
        }
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    for size in sizes:
        for seed in args.seeds:
            output = root / f"n{size}" / f"seed{seed}"
            summary = output / "summary.json"
            if summary.exists():
                print(f"skip complete n{size} seed{seed}", flush=True)
            else:
                command = [
                    sys.executable,
                    "-m", "sol.train_amortized_encoder",
                    "--manifest", str(root / f"n{size}" / "manifest.json"),
                    "--out", str(output),
                    "--device", args.device,
                    "--lr", str(args.lr),
                    "--warmup-epochs", str(args.warmup_epochs),
                    "--max-epochs", str(args.max_epochs),
                    "--min-epochs", str(args.min_epochs),
                    "--eval-every-epochs", str(args.eval_every_epochs),
                    "--early-stop-patience", str(args.early_stop_patience),
                    "--early-stop-min-delta-db", str(args.early_stop_min_delta_db),
                    "--points-per-step", str(args.points_per_step),
                    "--point-chunk", str(args.point_chunk),
                    "--support-capacity", str(args.support_capacity),
                    "--checkpoint-every", str(size * args.eval_every_epochs),
                    "--log-every", str(size),
                    "--seed", str(seed),
                ]
                if args.resume_root:
                    command.extend([
                        "--resume",
                        str(Path(args.resume_root) / f"n{size}" / f"seed{seed}"
                            / args.resume_checkpoint),
                    ])
                print("run", f"n{size}", f"seed{seed}", flush=True)
                subprocess.run(command, check=True)
            protocol["runs"].append(
                {
                    "train_size": size,
                    "seed": seed,
                    "output": str(output),
                    "summary": json.loads(summary.read_text()),
                }
            )
            (root / "protocol.json").write_text(
                json.dumps(protocol, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
