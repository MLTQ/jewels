"""Encode every prompt in a UCF streaming manifest with frozen OpenCLIP text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.prompt_embeddings import build_prompt_cache, collect_prompts, save_prompt_cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    manifest = json.loads(Path(args.manifest).read_text())
    prompts = collect_prompts(manifest)
    encoder = manifest["text_encoder"]
    if encoder.get("library") != "open_clip":
        raise ValueError("only the declared open_clip encoder is supported")
    import open_clip

    model = open_clip.create_model(
        encoder["model"], pretrained=encoder["pretrained"], device=args.device
    )
    tokenizer = open_clip.get_tokenizer(encoder["model"])
    model.eval()
    encoded = []
    for start in range(0, len(prompts), args.batch_size):
        tokens = tokenizer(list(prompts[start : start + args.batch_size])).to(args.device)
        values = model.encode_text(tokens).float()
        encoded.append(values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    embeddings = torch.cat(encoded).cpu()
    cache = build_prompt_cache(manifest, prompts, embeddings)
    save_prompt_cache(cache, args.out)
    similarity = embeddings @ embeddings.T
    off_diagonal = similarity[~torch.eye(len(prompts), dtype=torch.bool)]
    print(
        json.dumps(
            {
                "cache": args.out,
                "prompts": len(prompts),
                "dimension": embeddings.shape[1],
                "encoder": cache.encoder,
                "minimum_off_diagonal_cosine": float(off_diagonal.min()),
                "maximum_off_diagonal_cosine": float(off_diagonal.max()),
                "manifest_sha256": cache.manifest_sha256,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
