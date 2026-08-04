"""Fine-tune an axial flow prior on clamped cuboid-repair paths."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from sol.axial_prior import AxialFlowPrior
from sol.latent_data import load_latent_cache
from sol.latent_prior import flow_matching_loss, masked_flow_matching_loss
from sol.repair_evaluation import (
    RepairEvaluation,
    evaluate_masked_repair,
    sample_cuboid_masks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-extent", nargs=3, type=int, default=(3, 4, 4))
    parser.add_argument("--max-extent", nargs=3, type=int, default=(8, 10, 10))
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--full-loss-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=25)
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
    if args.steps <= 0 or args.batch <= 0 or args.full_loss_weight < 0:
        raise ValueError("steps/batch must be positive and full-loss weight nonnegative")
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
    base_checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if base_checkpoint["meta"].get("architecture") != "axial_flow_v1":
        raise ValueError("masked fine-tuning requires an axial-flow checkpoint")
    model_args = dict(base_checkpoint["meta"]["model_args"])
    model_args["mask_conditioning"] = True
    model = AxialFlowPrior(**model_args).to(device)
    incompatible = model.load_state_dict(
        base_checkpoint.get("ema", base_checkpoint["model"]), strict=False
    )
    if incompatible.unexpected_keys or incompatible.missing_keys != [
        "edit_mask_embedding.weight"
    ]:
        raise ValueError(f"base checkpoint is incompatible: {incompatible}")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    grid_shape = tuple(model_args["grid_shape"])
    min_extent = tuple(args.min_extent)
    max_extent = tuple(args.max_extent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "prior.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    base_evaluation: RepairEvaluation | None = None
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        stored = checkpoint["meta"].get("base_repair_evaluation")
        if stored:
            base_evaluation = RepairEvaluation(**stored)

    def run_evaluation() -> RepairEvaluation:
        return evaluate_masked_repair(
            ema_model,
            validation_latents,
            validation_conditions,
            grid_shape,
            min_extent,
            max_extent,
            device=device,
            examples=args.eval_examples,
            steps=args.eval_steps,
            seed=args.seed,
        )

    if base_evaluation is None:
        base_evaluation = run_evaluation()
        record = {"step": start_step, "base_repair_evaluation": base_evaluation.to_dict()}
        _append_json(log_path, record)
        print(json.dumps(record), flush=True)

    def save(step: int, latest: RepairEvaluation | None) -> None:
        _atomic_checkpoint(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "axial_flow_masked_v2",
                    "training_mode": "mask_conditioned_cuboid_repair_v2",
                    "model_args": model_args,
                    "cache": str(args.cache),
                    "cache_metadata": cache.metadata,
                    "base_checkpoint": str(args.base),
                    "train_args": vars(args),
                    "base_repair_evaluation": base_evaluation.to_dict(),
                    "latest_repair_evaluation": latest.to_dict() if latest else None,
                },
            },
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train={len(train_latents)} validation={len(validation_latents)} "
        f"grid={grid_shape} model={parameters / 1e6:.2f}M amp={use_amp} "
        f"cuboids={min_extent}:{max_extent}",
        flush=True,
    )

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    losses: list[float] = []
    masked_losses: list[float] = []
    full_losses: list[float] = []
    latest: RepairEvaluation | None = None
    started = time.time()
    interval_started = started
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            0, len(train_latents), (args.batch,), device=device, generator=generator
        )
        dirty = sample_cuboid_masks(
            args.batch,
            grid_shape,
            min_extent,
            max_extent,
            device=device,
            generator=generator,
        )
        learning_rate = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            masked_loss = masked_flow_matching_loss(
                model,
                train_latents[indices],
                train_conditions[indices],
                dirty,
                condition_dropout=args.condition_dropout,
            )
            if args.full_loss_weight:
                full_loss = flow_matching_loss(
                    model,
                    train_latents[indices],
                    train_conditions[indices],
                    condition_dropout=args.condition_dropout,
                )
            else:
                full_loss = masked_loss.new_zeros(())
            loss = masked_loss + args.full_loss_weight * full_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        masked_losses.append(float(masked_loss.detach()))
        full_losses.append(float(full_loss.detach()))
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
                "masked_loss": sum(masked_losses[-count:]) / count,
                "full_loss": sum(full_losses[-count:]) / count,
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / count,
            }
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = run_evaluation()
            record = {"step": step, "repair_evaluation": latest.to_dict()}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "base_repair_evaluation": base_evaluation.to_dict(),
        "latest_repair_evaluation": latest.to_dict() if latest else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
