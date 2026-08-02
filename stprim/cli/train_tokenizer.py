"""Train the jewel tokenizer (set autoencoder) on a fitted corpus.

    python cli/train_tokenizer.py --corpus ~/jewels/corpus/avenue_train \
        --out ~/jewels/tokenizer/v0 --device cuda:1

Ends with a round-trip check: encode a corpus window to latents, sample the
decoder back to jewels, render both, report PSNR between the two renders.
That number is the compression question answered: can ~256 latent tokens
carry a scene the prior currently needs 6471 tokens to express?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.volume import make_grid  # noqa: E402
from models.render import render_volume  # noqa: E402
from prior.featurize import FEAT_DIM, features_to_field, load_corpus  # noqa: E402
from prior.tokenizer import JewelTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--grid", type=int, nargs=3, default=[8, 8, 4])
    ap.add_argument("--enc-depth", type=int, default=4)
    ap.add_argument("--dec-depth", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--flip-u", action="store_true")
    ap.add_argument("--no-bf16", action="store_true",
                    help="fp32 training — required on pre-Ampere GPUs, where "
                         "autocast steers SDPA into the FlashAttention kernel "
                         "(Ampere+) and crashes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=2500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if (device.type == "cuda"
            and torch.cuda.get_device_capability(device) < (8, 0)):
        # Turing and older have no FlashAttention kernels, and this torch
        # build's SDPA dispatcher selects flash before the arch check.
        torch.backends.cuda.enable_flash_sdp(False)
        print("pre-Ampere GPU: flash SDP disabled", flush=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus)
    feats = corpus["feats"]
    if args.flip_u:
        flipped = feats.clone()
        for d in (0, 4, 5, 12, 15, 18):
            flipped[..., d] = -flipped[..., d]
        feats = torch.cat([feats, flipped])
    S, N, _ = feats.shape
    flat = feats.reshape(-1, FEAT_DIM)
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-4)
    x_data = ((feats - mean) / std).to(device)
    print(f"corpus: {S} sets x {N} jewels", flush=True)

    tok = JewelTokenizer(
        feat_dim=FEAT_DIM, dim=args.dim, latent_dim=args.latent_dim,
        grid=tuple(args.grid), enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
    ).to(device)
    tok.encoder.mu_mean.copy_(mean[:3].to(device))
    tok.encoder.mu_std.copy_(std[:3].to(device))
    n_params = sum(p.numel() for p in tok.parameters())
    n_lat = tok.encoder.n_cells * args.latent_dim
    print(f"tokenizer: {n_params / 1e6:.2f}M params; "
          f"{tok.encoder.n_cells} latents x {args.latent_dim} = {n_lat} numbers "
          f"({N * FEAT_DIM / n_lat:.0f}x compression at N={N})", flush=True)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = (not args.no_bf16) and device.type == "cuda"

    meta = {
        "feat_mean": mean, "feat_std": std, "shape": corpus["shape"],
        "bg_mean": corpus["bg"].mean(0), "n_primitives": N,
        "model_args": {"feat_dim": FEAT_DIM, "dim": args.dim,
                       "latent_dim": args.latent_dim,
                       "grid": tuple(args.grid),
                       "enc_depth": args.enc_depth,
                       "dec_depth": args.dec_depth},
        "train_args": vars(args),
    }

    t0 = time.time()
    t_last = t0
    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, S, (args.batch,), device=device)
        lr = (args.lr * step / args.warmup if step < args.warmup else
              args.lr * (0.1 + 0.45 * (1 + math.cos(
                  math.pi * (step - args.warmup) / (args.steps - args.warmup)))))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss = tok.flow_loss(x_data[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

        if step % args.log_every == 0:
            now = time.time()
            rate = (now - t_last) / args.log_every
            t_last = now
            avg = sum(losses[-args.log_every:]) / args.log_every
            print(f"step {step:6d}  loss {avg:.4f}  ({rate:.2f}s/step, "
                  f"~{rate * (args.steps - step) / 60:.0f}m left)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": tok.state_dict(), "meta": meta,
                        "step": step}, outdir / "tokenizer.pt")

    # --- round-trip check ------------------------------------------------- #
    tok.eval()
    g = torch.Generator(device=device).manual_seed(0)
    x1 = x_data[0:1]
    with torch.no_grad():
        xr = tok.reconstruct(x1, steps=100, generator=g)
    orig = (x1[0].cpu() * std + mean)
    recon = (xr[0].float().cpu() * std + mean)
    T, H, W = corpus["shape"]
    grid_pts = make_grid((T, H, W), device=device)
    bg = corpus["bg"][0].to(device)
    with torch.no_grad():
        fa = render_volume(features_to_field(orig, device=device),
                           grid_pts, background=bg)
        fb = render_volume(features_to_field(recon, device=device),
                           grid_pts, background=bg)
    psnr = -10 * torch.log10(((fa - fb) ** 2).mean().clamp_min(1e-10))
    print(f"round-trip render PSNR (fit render vs tokenized render): "
          f"{float(psnr):.2f} dB", flush=True)
    (outdir / "train_log.json").write_text(json.dumps(
        {"final_loss": sum(losses[-200:]) / 200,
         "roundtrip_psnr": float(psnr),
         "seconds": time.time() - t0}))
    print(f"done in {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
