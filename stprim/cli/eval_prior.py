"""Score a prior checkpoint: CLIP-Fréchet distance, generated vs fitted renders.

    python cli/eval_prior.py --ckpt prior/v1/prior.pt --corpus corpus/avenue_train

Protocol (FIXED — comparability across checkpoints is the entire point):
  * sample `--n-samples` sets (seed 0, cfg 1.5, 100 flow steps, cond-index 0),
    render each, take `--frames-per` evenly spaced frames
  * reference: `--n-real` corpus windows decoded from their checkpoints and
    rendered the same way (renders vs renders — this isolates PRIOR quality
    from fitter softness)
  * CLIP ViT-B/32 image embeddings of all frames; Fréchet distance between
    the two Gaussians. Small-sample FID is biased, but the bias is shared
    across checkpoints under a fixed protocol, so the CURVE is meaningful
    even where an absolute number would not be.

Reference embeddings are cached in the corpus dir (real_clip_stats.pt) so
every checkpoint is scored against the identical reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.volume import make_grid  # noqa: E402
from models.render import render_volume  # noqa: E402
from prior.featurize import features_to_field, load_corpus  # noqa: E402
from prior.model import SetDiT  # noqa: E402
from cli.sample_prior import sample_sets  # noqa: E402


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fréchet distance between Gaussians fit to rows of a and b (double)."""
    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = torch.cov(a.T) + 1e-6 * torch.eye(a.shape[1], dtype=torch.float64)
    cb = torch.cov(b.T) + 1e-6 * torch.eye(b.shape[1], dtype=torch.float64)
    # Tr sqrt(Ca Cb) via the symmetric sandwich Ca^1/2 Cb Ca^1/2.
    la, ua = torch.linalg.eigh(ca)
    sq_a = ua @ torch.diag(la.clamp_min(0).sqrt()) @ ua.T
    ls, _ = torch.linalg.eigh(sq_a @ cb @ sq_a)
    tr_sqrt = ls.clamp_min(0).sqrt().sum()
    return float((mu_a - mu_b).square().sum() + ca.trace() + cb.trace()
                 - 2 * tr_sqrt)


@torch.no_grad()
def clip_frames(frames: torch.Tensor, model, preprocess, device) -> torch.Tensor:
    from PIL import Image
    ims = torch.stack([
        preprocess(Image.fromarray((f.cpu().numpy() * 255).astype("uint8")))
        for f in frames
    ]).to(device)
    emb = model.encode_image(ims)
    return (emb / emb.norm(dim=-1, keepdim=True)).float().cpu()


@torch.no_grad()
def render_frames(field, shape, background, picks, device):
    T, H, W = shape
    grid = make_grid((T, H, W), device=device)
    out = render_volume(field, grid, background=background)
    return out.reshape(T, H, W, 3).clamp(0, 1)[picks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--n-real", type=int, default=32)
    ap.add_argument("--frames-per", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import open_clip

    device = torch.device(args.device)
    corpus = load_corpus(args.corpus)
    S = corpus["feats"].shape[0]
    shape = corpus["shape"]
    picks = torch.linspace(0, shape[0] - 1, args.frames_per).long()

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    clip_model.eval()

    # --- reference embeddings (cached; identical across evals) ------------ #
    cache = Path(args.corpus) / "real_clip_stats.pt"
    if cache.exists():
        real_emb = torch.load(cache, weights_only=False)["emb"]
    else:
        g = torch.Generator().manual_seed(123)
        idxs = torch.randperm(S, generator=g)[: args.n_real]
        chunks = []
        for i in idxs:
            f = features_to_field(corpus["feats"][i], device=device)
            fr = render_frames(f, shape, corpus["bg"][i].to(device), picks,
                               device)
            chunks.append(clip_frames(fr, clip_model, preprocess, device))
            del f
        real_emb = torch.cat(chunks)
        torch.save({"emb": real_emb, "n_real": args.n_real,
                    "frames_per": args.frames_per}, cache)
    print(f"reference: {real_emb.shape[0]} frame embeddings", flush=True)

    # --- generated embeddings --------------------------------------------- #
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = SetDiT(**meta["model_args"]).to(device)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.eval()

    cond = corpus["clip"][0][None].to(device).expand(args.n_samples, -1)
    g = torch.Generator(device=device).manual_seed(0)
    xs = sample_sets(model, n_samples=args.n_samples,
                     n_prims=meta["n_primitives"],
                     feat_dim=meta["feat_mean"].numel(), cond=cond, cfg=1.5,
                     steps=100, device=device, generator=g)
    xs = xs.cpu() * meta["feat_std"] + meta["feat_mean"]
    bg = meta["bg_mean"].to(device)

    chunks = []
    for k in range(args.n_samples):
        f = features_to_field(xs[k], device=device)
        fr = render_frames(f, shape, bg, picks, device)
        chunks.append(clip_frames(fr, clip_model, preprocess, device))
        del f
    gen_emb = torch.cat(chunks)

    cfd = frechet(gen_emb, real_emb)
    print(f"CFD (CLIP-Frechet, gen renders vs fitted renders): {cfd:.2f}")
    print(f"  ckpt: {args.ckpt}  step: {ck.get('step')}  "
          f"gen frames: {gen_emb.shape[0]}")


if __name__ == "__main__":
    main()
