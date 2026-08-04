"""CLI for auditing persistent carry/commit semantics on one fitted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from sol.streaming_metrics import audit_carry_commit_render, measure_streaming_contract


stprim_root = Path(__file__).resolve().parent.parent / "stprim"
if str(stprim_root) not in sys.path:
    sys.path.insert(0, str(stprim_root))

from prior.featurize import state_to_features  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--points-per-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = state_to_features(checkpoint["state"]).to(args.device)
    shape = tuple(checkpoint["info"]["shape"])
    report, _, windows = measure_streaming_contract(
        features,
        shape,
        fps=args.fps,
        prefix_frames=args.prefix_frames,
        stride_frames=args.stride_frames,
        support_sigma=args.support_sigma,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["render_equivalence"] = audit_carry_commit_render(
        features,
        shape[0],
        windows,
        support_sigma=args.support_sigma,
        points_per_frame=args.points_per_frame,
        seed=args.seed,
    )
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
