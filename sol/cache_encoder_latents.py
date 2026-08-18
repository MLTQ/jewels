"""Cache frozen-encoder latents and prompt embeddings for text-conditioned generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.amortized_encoder import VideoToJewelEncoder
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video

SCHEMA = "jewel-encoder-latent-cache-v1"


def encode_prompts(
    prompts: list[str], model_name: str, device: torch.device, max_tokens: int = 32
) -> dict[str, torch.Tensor]:
    """Return padded token embeddings and a mask for every unique prompt.

    Token sequences rather than one pooled vector: pooled sentence embeddings
    are exactly the representation the earlier CLIP-conditioned attempts failed
    with, because verbs and composition survive tokenization but not pooling.
    """
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    batch = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = model(**batch).last_hidden_state
    return {
        "tokens": output.cpu(),
        "mask": batch["attention_mask"].cpu(),
        "dim": output.shape[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--text-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.encoder, map_location=device, weights_only=False)
    meta = saved["meta"]
    if meta.get("architecture") != "video_to_jewel_encoder_v0":
        raise ValueError("checkpoint is not a video-to-jewel encoder")
    spec = GridSpec(tuple(meta["grid_shape"]), 1024)
    model = VideoToJewelEncoder(grid_spec=spec, **meta["model_args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    manifest = json.loads(Path(args.manifest).read_text())
    examples = manifest["examples"]
    prompts = sorted({item.get("source_prompt", "") for item in examples})
    if not all(prompts):
        raise ValueError("every example needs a non-empty source_prompt")
    text = encode_prompts(prompts, args.text_model, device, args.max_tokens)
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, cells, seeds = [], [], []
    for item in examples:
        video = load_video(
            item["video"],
            max_frames=int(item["frames"]),
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        ).to(device)
        with torch.no_grad():
            latent = model.encode(video)
        cells.append(latent["cells"].cpu())
        seeds.append(latent["seed"].cpu())
        records.append(
            {
                "source_id": item["source_id"],
                "style": item.get("style", "default"),
                "class_id": int(item["class_id"]),
                "class_name": item["class_name"],
                "split": item["split"],
                "prompt": item["source_prompt"],
                "prompt_index": prompt_index[item["source_prompt"]],
            }
        )
        print("encoded", item["source_id"], flush=True)

    torch.save(
        {
            "schema": SCHEMA,
            "cells": torch.stack(cells),
            "seed": torch.stack(seeds),
            "text_tokens": text["tokens"],
            "text_mask": text["mask"],
            "text_dim": int(text["dim"]),
            "prompts": prompts,
            "records": records,
            "encoder": {
                "checkpoint": args.encoder,
                "step": int(saved["step"]),
                "grid_shape": list(spec.shape),
                "slots_per_cell": int(meta["slots_per_cell"]),
                "model_args": meta["model_args"],
            },
            "text_model": args.text_model,
        },
        output_dir / "latents.pt",
    )
    print(
        json.dumps(
            {
                "out": str(output_dir / "latents.pt"),
                "windows": len(records),
                "prompts": len(prompts),
                "cells_shape": list(cells[0].shape),
                "seed_shape": list(seeds[0].shape),
                "text_dim": int(text["dim"]),
            }
        )
    )


if __name__ == "__main__":
    main()
