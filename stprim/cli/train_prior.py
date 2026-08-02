"""Train the v0 set prior on a fitted corpus (conditional flow matching).

    python cli/train_prior.py --corpus ~/jewels/corpus/avenue_train \
        --out ~/jewels/prior/avenue_v0

Rectified-flow objective: x_t = (1-t) noise + t data, target velocity is
(data - noise), MSE. CLIP conditioning with dropout so classifier-free
guidance works at sampling time. On a 231-set corpus the model WILL largely
memorize — intended for v0: the question this run answers is whether
sample -> decode -> render produces coherent video at all.
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

from prior.featurize import FEAT_DIM, load_corpus  # noqa: E402
from prior.model import SetDiT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--cond-dropout", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (same math, fused kernels)")
    ap.add_argument("--flip-u", action="store_true",
                    help="mirror augmentation in u (doubles the corpus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus)
    feats, clip = corpus["feats"], corpus["clip"]

    if args.flip_u:
        # Mirror in u, applied in RAW feature space: congruence by
        # diag(-1,1,1) negates mu_u, the uv/ut log-covariance entries, and
        # the d/du column of the P1 ramp. CLIP embeddings are reused — CLIP
        # is nearly mirror-invariant and exact flip embeddings aren't worth
        # a second encode pass at this corpus size.
        flipped = feats.clone()
        for d in (0, 4, 5, 12, 15, 18):
            flipped[..., d] = -flipped[..., d]
        feats = torch.cat([feats, flipped])
        clip = torch.cat([clip, clip])

    S, N, F_ = feats.shape
    assert F_ == FEAT_DIM

    flat = feats.reshape(-1, FEAT_DIM)
    mean = flat.mean(0)
    std = flat.std(0).clamp_min(1e-4)
    x_data = ((feats - mean) / std).to(device)
    c_data = clip.to(device)
    print(f"corpus: {S} sets x {N} primitives x {FEAT_DIM} dims "
          f"(flip_u={args.flip_u})", flush=True)

    model = SetDiT(
        feat_dim=FEAT_DIM, dim=args.dim, depth=args.depth,
        heads=args.heads, cond_dim=clip.shape[1],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # EMA and checkpoints always use the raw module's state dict — a compiled
    # wrapper prefixes keys with _orig_mod. and would poison the sampler.
    raw = model
    if args.compile:
        model = torch.compile(model)

    ema = {k: v.detach().clone() for k, v in raw.state_dict().items()}
    use_amp = (not args.no_bf16) and device.type == "cuda"

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        a = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * a)))

    meta = {
        "feat_mean": mean, "feat_std": std,
        "bg_mean": corpus["bg"].mean(0), "shape": corpus["shape"],
        "n_primitives": N, "cond_dim": clip.shape[1],
        "model_args": {"feat_dim": FEAT_DIM, "dim": args.dim,
                       "depth": args.depth, "heads": args.heads,
                       "cond_dim": clip.shape[1]},
        "corpus": str(args.corpus), "train_args": vars(args),
    }

    t0 = time.time()
    t_last = t0
    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, S, (args.batch,), device=device)
        x1 = x_data[idx]
        c = c_data[idx]
        drop = torch.rand(args.batch, device=device) < args.cond_dropout

        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch, device=device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0

        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            v_pred = model(xt, t, c, drop)
            loss = torch.nn.functional.mse_loss(
                v_pred.float(), v_target.float()
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))

        with torch.no_grad():
            d = args.ema_decay
            for k, v in raw.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(d).add_(v, alpha=1 - d)
                else:
                    ema[k].copy_(v)

        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / args.log_every
            now = time.time()
            rate = (now - t_last) / args.log_every  # interval, not cumulative:
            t_last = now                            # compile warmup would skew it
            print(f"step {step:6d}  loss {avg:.4f}  "
                  f"({rate:.2f}s/step, ~{rate * (args.steps - step) / 60:.0f}m left)",
                  flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": raw.state_dict(), "ema": ema,
                        "meta": meta, "step": step}, outdir / "prior.pt")

    (outdir / "train_log.json").write_text(json.dumps(
        {"final_loss_avg100": sum(losses[-100:]) / 100,
         "seconds": time.time() - t0, "steps": args.steps}))
    print(f"done in {(time.time() - t0) / 60:.1f}m -> {outdir/'prior.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
