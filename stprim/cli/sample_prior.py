"""Sample the set prior and render the results: jewels -> video.

    python cli/sample_prior.py --ckpt ~/jewels/prior/avenue_v0/prior.pt \
        --corpus ~/jewels/corpus/avenue_train --out ~/jewels/prior/samples

Per sample: Euler-integrate the learned velocity field from noise (optionally
with classifier-free guidance toward a CLIP embedding), un-normalize,
decode features -> PrimitiveField, render the full volume. Writes one GIF per
sample as [real fitted window | generated] so the eye has an anchor, plus the
decoded field checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.render_recon import hstack, to_pil  # noqa: E402
from core.volume import make_grid  # noqa: E402
from models.render import render_volume  # noqa: E402
from prior.featurize import features_to_field, load_corpus  # noqa: E402
from prior.model import SetDiT  # noqa: E402


@torch.no_grad()
def sample_sets(
    model, *, n_samples: int, n_prims: int, feat_dim: int,
    cond: torch.Tensor | None, cfg: float, steps: int, device,
    generator: torch.Generator,
) -> torch.Tensor:
    x = torch.randn(n_samples, n_prims, feat_dim,
                    generator=generator, device=device)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n_samples)
        if cond is not None and cfg != 1.0:
            v_c = model(x, t, cond)
            v_u = model(x, t, None)
            v = v_u + cfg * (v_c - v_u)
        else:
            v = model(x, t, cond)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


@torch.no_grad()
def render_field(field, shape, background, *, device, chunk=65536):
    T, H, W = shape
    grid = make_grid((T, H, W), device=device)
    out = render_volume(field, grid, chunk=chunk, background=background)
    return out.reshape(T, H, W, 3).clamp(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--corpus", type=str, required=True,
                    help="for the real-fit reference panel + conditioning")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--flow-steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--cond-index", type=int, default=0,
                    help="corpus window whose CLIP embedding conditions "
                         "samples; -1 for unconditional")
    ap.add_argument("--prompt", type=str, default=None,
                    help="text prompt, encoded with the CLIP text tower "
                         "(overrides --cond-index). NB: the prior trains on "
                         "IMAGE embeddings; CLIP's text-image modality gap "
                         "means prompts steer weakly until a corpus is "
                         "trained with caption embeddings in the mix")
    ap.add_argument("--raw-weights", action="store_true",
                    help="sample the raw weights instead of EMA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = SetDiT(**meta["model_args"]).to(device)
    weights = ck["model"] if args.raw_weights or "ema" not in ck else ck["ema"]
    model.load_state_dict(weights)
    model.eval()

    corpus = load_corpus(args.corpus)
    shape = meta["shape"]
    bg = meta["bg_mean"].to(device)

    # Real reference: the conditioning window itself (or window 0).
    ref_i = max(args.cond_index, 0)
    ref_field = features_to_field(corpus["feats"][ref_i], device=device)
    ref_bg = corpus["bg"][ref_i].to(device)
    ref_frames = render_field(ref_field, shape, ref_bg, device=device)

    cond = None
    if args.prompt:
        import open_clip

        cm, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        cm.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            te = cm.encode_text(tok([args.prompt]).to(device))
            te = te / te.norm(dim=-1, keepdim=True)
        cond = te.float().expand(args.n_samples, -1)
        del cm
        print(f"conditioning on prompt: {args.prompt!r}", flush=True)
    elif args.cond_index >= 0:
        cond = corpus["clip"][args.cond_index][None].to(device) \
            .expand(args.n_samples, -1)

    g = torch.Generator(device=device).manual_seed(args.seed)
    xs = sample_sets(
        model, n_samples=args.n_samples, n_prims=meta["n_primitives"],
        feat_dim=meta["feat_mean"].numel(), cond=cond, cfg=args.cfg,
        steps=args.flow_steps, device=device, generator=g,
    )
    xs = xs.cpu() * meta["feat_std"] + meta["feat_mean"]

    T = shape[0]
    for k in range(args.n_samples):
        field = features_to_field(xs[k], device=device)
        frames = render_field(field, shape, bg, device=device)
        gif = [hstack([to_pil(ref_frames[t], 1), to_pil(frames[t], 1)])
               for t in range(T)]
        gif[0].save(outdir / f"sample_{k}.gif", save_all=True,
                    append_images=gif[1:], duration=83, loop=0)
        torch.save({"features": xs[k], "meta_ref": str(args.ckpt)},
                   outdir / f"sample_{k}_features.pt")
        print(f"sample {k}: N={len(field)}  wrote sample_{k}.gif", flush=True)

    # Marginal sanity: generated vs corpus feature stats.
    gen = xs.reshape(-1, xs.shape[-1])
    real = corpus["feats"].reshape(-1, xs.shape[-1])
    with np.printoptions(precision=3, suppress=True):
        print("mu/scale marginals (real vs gen):")
        print("  real mean[:9]:", real.mean(0)[:9].numpy())
        print("  gen  mean[:9]:", gen.mean(0)[:9].numpy())
        print("  real std[:9]: ", real.std(0)[:9].numpy())
        print("  gen  std[:9]: ", gen.std(0)[:9].numpy())


if __name__ == "__main__":
    main()
