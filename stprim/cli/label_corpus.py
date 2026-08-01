"""Attach CLIP image embeddings to a fitted corpus (t2v conditioning data).

    python cli/label_corpus.py --corpus corpus/avenue --videos-root .

For each `<stem>_w<start>.pt` checkpoint, samples a few frames from the source
window, encodes them with CLIP, and writes the mean embedding to
`<stem>_w<start>.clip.npy` next to the checkpoint. Resumable; skips existing.

The stage-2 prior conditions on this embedding (with dropout, so unconditional
generation is the special case). When a diverse captioned corpus exists, text
embeddings from the same CLIP model drop into the same interface — that is the
point of using CLIP rather than a bare video encoder.

Requires open_clip_torch (pip install open-clip-torch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.video_io import load_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True,
                    help="directory of fit checkpoints")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--frames-per-window", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import open_clip  # local import: heavy, and only this tool needs it

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    model.eval()

    from PIL import Image

    ckpts = sorted(Path(args.corpus).glob("*_w*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints in {args.corpus}")

    done = skipped = 0
    for ckpt in ckpts:
        out = ckpt.with_suffix(".clip.npy")
        if out.exists():
            skipped += 1
            continue
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        src = meta["source"]
        T = meta["info"]["shape"][0]

        # A few frames spread across the window; CLIP is frame-level, and the
        # mean over frames is the standard cheap video embedding.
        picks = np.linspace(0, T - 1, args.frames_per_window).astype(int)
        vid = load_video(src["video"], max_frames=T,
                         start_frame=src["start_frame"])
        ims = torch.stack([
            preprocess(Image.fromarray(
                (vid[t].numpy() * 255).astype("uint8")))
            for t in picks
        ]).to(args.device)
        with torch.no_grad():
            emb = model.encode_image(ims)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        np.save(out, emb.mean(0).cpu().numpy().astype(np.float32))
        done += 1
        if done % 25 == 0:
            print(f"{done} labeled...", flush=True)

    print(f"done: {done} labeled, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
