"""Train a text-conditioned rectified flow over frozen jewel raster latents."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from sol.latent_data import load_latent_cache
from sol.latent_prior import RasterFlowPrior, flow_matching_loss
from sol.prior_evaluation import PriorEvaluation, evaluate_prior


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-sample-steps", type=int, default=25)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("steps and batch must be positive")
    if not 0 <= args.condition_dropout <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError("dropout and EMA decay are outside their valid ranges")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)
    cache = load_latent_cache(args.cache)
    train_latents, train_conditions, _ = cache.split(train=True)
    validation_latents, validation_conditions, _ = cache.split(train=False)
    train_latents = train_latents.to(device)
    train_conditions = train_conditions.to(device)
    model_args = {
        "n_cells": train_latents.shape[1],
        "latent_dim": train_latents.shape[2],
        "model_dim": args.model_dim,
        "depth": args.depth,
        "heads": args.heads,
        "text_dim": train_conditions.shape[1],
    }
    model = RasterFlowPrior(**model_args).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "prior.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        print(f"resumed step {start_step}", flush=True)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train={len(train_latents)} validation={len(validation_latents)} "
        f"latent={tuple(train_latents.shape[1:])} condition={train_conditions.shape[1]} "
        f"model={parameters / 1e6:.2f}M amp={use_amp}",
        flush=True,
    )

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    def run_evaluation() -> PriorEvaluation:
        return evaluate_prior(
            ema_model,
            train_latents.cpu(),
            train_conditions.cpu(),
            validation_latents,
            validation_conditions,
            device=device,
            sample_steps=args.eval_sample_steps,
            sample_examples=args.eval_samples,
            seed=args.seed,
        )

    def save(step: int, latest: PriorEvaluation | None) -> None:
        _atomic_checkpoint(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "model_args": model_args,
                    "cache": str(args.cache),
                    "cache_metadata": cache.metadata,
                    "train_args": vars(args),
                    "latest_evaluation": latest.to_dict() if latest else None,
                },
            },
        )

    latest = run_evaluation() if start_step == 0 else None
    if latest:
        record = {"step": 0, "evaluation": latest.to_dict()}
        _append_json(log_path, record)
        print(json.dumps(record), flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    losses = []
    started = time.time()
    interval_started = started
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            0, len(train_latents), (args.batch,), device=device, generator=generator
        )
        learning_rate = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = flow_matching_loss(
                model,
                train_latents[indices],
                train_conditions[indices],
                condition_dropout=args.condition_dropout,
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                ema_model.parameters(), model.parameters(), strict=True
            ):
                ema_parameter.lerp_(parameter, 1 - args.ema_decay)
            for ema_buffer, buffer in zip(
                ema_model.buffers(), model.buffers(), strict=True
            ):
                ema_buffer.copy_(buffer)

        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            count = min(args.log_every, len(losses))
            record = {
                "step": step,
                "loss": sum(losses[-count:]) / count,
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / count,
            }
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = run_evaluation()
            record = {"step": step, "evaluation": latest.to_dict()}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)

    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "latest_evaluation": latest.to_dict() if latest else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
