"""Encode manifest windows into canonical jewel fields with a trained encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.amortized_encoder import VideoToJewelEncoder
from sol.prompt_embeddings import manifest_digest
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = saved["meta"]
    if meta.get("architecture") != "video_to_jewel_encoder_v0":
        raise ValueError("checkpoint is not a video-to-jewel encoder")
    manifest = json.loads(Path(args.manifest).read_text())
    if meta.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("encoder checkpoint does not own this manifest")
    spec = GridSpec(tuple(meta["grid_shape"]), 1024)
    model = VideoToJewelEncoder(grid_spec=spec, **meta["model_args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = []
    for example in manifest["examples"]:
        if example["split"] != args.split:
            continue
        video = load_video(
            example["video"],
            max_frames=int(example["frames"]),
            start_frame=int(example.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        ).to(device)
        with torch.no_grad():
            prediction = model(video)
            features = model.canonical_features(prediction)
        torch.save(
            {
                "features": features.cpu(),
                "background": prediction["background"].detach().cpu(),
                "encoder": {
                    "checkpoint": args.checkpoint,
                    "step": int(saved["step"]),
                    "architecture": meta["architecture"],
                },
            },
            output_dir / f"{example['source_id']}_generated_field.pt",
        )
        encoded.append(example["source_id"])
        print("encoded", example["source_id"], "jewels", len(features), flush=True)
    if not encoded:
        raise ValueError(f"manifest has no {args.split!r} examples")
    (output_dir / "encode_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "step": int(saved["step"]),
                "split": args.split,
                "sources": encoded,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
