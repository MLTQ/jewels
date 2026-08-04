"""Train and validate the structured jewel autoencoder on fitted checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, FittedExample, load_fitted_corpus, split_by_source
from sol.evaluation import EvaluationReport, evaluate_roundtrip
from sol.render import render_exact
from sol.token_grid import CompactGrid, GridSpec, OccupancyGrid


@dataclass
class PreparedExample:
    features: torch.Tensor
    target: CompactGrid


def _prepare_examples(
    examples: list[FittedExample],
    normalizer: FeatureNormalizer,
    grid: OccupancyGrid,
) -> list[PreparedExample]:
    prepared = []
    for example in examples:
        features = normalizer.normalize(example.features)
        prepared.append(PreparedExample(features, grid.pack_compact(features)))
    return prepared


def _batch(
    prepared: list[PreparedExample],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, CompactGrid]:
    selected = [prepared[int(index)] for index in indices]
    features = torch.stack([example.features for example in selected]).to(device)
    target = CompactGrid(
        values=torch.stack([example.target.values for example in selected]).to(device),
        cell_indices=torch.stack(
            [example.target.cell_indices for example in selected]
        ).to(device),
        slot_indices=torch.stack(
            [example.target.slot_indices for example in selected]
        ).to(device),
        counts=torch.stack([example.target.counts for example in selected]).to(device),
    )
    return features, target


def _sampled_render_loss(
    decoded_features: torch.Tensor,
    target: CompactGrid,
    normalizer: FeatureNormalizer,
    *,
    points_per_example: int,
) -> torch.Tensor:
    """Compare differentiable renders using the target's occupied slot correspondence."""
    if points_per_example <= 0:
        raise ValueError("render points must be positive")
    batch = decoded_features.shape[0]
    batch_indices = torch.arange(batch, device=decoded_features.device)[:, None]
    predicted_normalized = decoded_features[
        batch_indices, target.cell_indices, target.slot_indices
    ]
    mean = normalizer.mean.to(decoded_features.device, dtype=torch.float32)
    std = normalizer.std.to(decoded_features.device, dtype=torch.float32)
    predicted = predicted_normalized.float() * std + mean
    expected = target.values.float() * std + mean
    losses = []
    for batch_index in range(batch):
        points = torch.rand(
            points_per_example, 3, device=decoded_features.device, dtype=torch.float32
        ) * 2 - 1
        with torch.no_grad():
            expected_render = render_exact(expected[batch_index], points).clamp(0, 1)
        predicted_render = render_exact(predicted[batch_index], points).clamp(0, 1)
        losses.append((predicted_render - expected_render).square().mean())
    return torch.stack(losses).mean()


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--grid", type=int, nargs=3, default=(12, 12, 6))
    parser.add_argument("--slots", type=int, default=80)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--enc-depth", type=int, default=2)
    parser.add_argument("--dec-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--validation-sources", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-examples", type=int, default=3)
    parser.add_argument("--eval-points", type=int, default=512)
    parser.add_argument("--render-weight", type=float, default=0.1)
    parser.add_argument("--render-points", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.batch <= 0:
        raise ValueError("steps and batch must be positive")
    if args.render_weight < 0 or args.render_points <= 0:
        raise ValueError("render weight must be non-negative and points positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "autoencoder.pt"
    log_path = output_dir / "train_log.jsonl"

    examples = load_fitted_corpus(args.corpus, limit=args.limit)
    split = split_by_source(
        examples,
        validation_sources=args.validation_sources,
        seed=args.seed,
    )
    normalizer = FeatureNormalizer.fit(split.train)
    spec = GridSpec(tuple(args.grid), args.slots)
    grid = OccupancyGrid(spec)
    reports = [grid.capacity_report(example.features) for example in examples]
    max_occupancy = max(report.max_cell_occupancy for report in reports)
    overflow = sum(report.overflow_cells for report in reports)
    if overflow:
        raise ValueError(
            f"grid under-capacity: {overflow} overflowing example-cells; "
            f"observed maximum {max_occupancy}, configured slots {args.slots}"
        )
    print(
        f"corpus={len(examples)} train={len(split.train)} val={len(split.validation)} "
        f"held_out={split.validation_sources}",
        flush=True,
    )
    print(
        f"grid={spec.shape} cells={spec.n_cells} slots={spec.slots_per_cell} "
        f"observed_max={max_occupancy}",
        flush=True,
    )
    prepared = _prepare_examples(split.train, normalizer, grid)

    model_args = {
        "feature_dim": examples[0].features.shape[-1],
        "model_dim": args.model_dim,
        "latent_dim": args.latent_dim,
        "spec": spec,
        "enc_depth": args.enc_depth,
        "dec_depth": args.dec_depth,
        "heads": args.heads,
    }
    model = StructuredJewelAutoencoder(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        print(f"resumed step {start_step}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    raw_numbers = examples[0].features.numel()
    latent_numbers = spec.n_cells * args.latent_dim
    print(
        f"model={parameter_count / 1e6:.2f}M latent={latent_numbers} numbers "
        f"compression={raw_numbers / latent_numbers:.2f}x amp={use_amp}",
        flush=True,
    )

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    def save(step: int, latest_evaluation: EvaluationReport | None) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "meta": {
                "model_args": {
                    key: value for key, value in model_args.items() if key != "spec"
                },
                "grid_shape": spec.shape,
                "slots_per_cell": spec.slots_per_cell,
                "normalizer": normalizer.state_dict(),
                "validation_sources": split.validation_sources,
                "corpus": str(args.corpus),
                "train_args": vars(args),
                "latest_evaluation": (
                    latest_evaluation.to_dict() if latest_evaluation else None
                ),
            },
        }
        _atomic_checkpoint(checkpoint_path, payload)

    losses = []
    latest_evaluation = None
    if start_step == 0:
        latest_evaluation = evaluate_roundtrip(
            model,
            split.validation,
            normalizer,
            device=device,
            points_per_example=args.eval_points,
            max_examples=args.eval_examples,
            seed=args.seed,
        )
        baseline_record = {"step": 0, "evaluation": latest_evaluation.to_dict()}
        _append_json(log_path, baseline_record)
        print(json.dumps(baseline_record), flush=True)
    started = time.time()
    interval_started = started
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(start_step + 1, args.steps + 1):
        sample_indices = torch.randint(
            0, len(prepared), (args.batch,), generator=generator
        )
        features, target = _batch(prepared, sample_indices, device)
        learning_rate = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(features)
            loss, terms = model.structural_loss(output, target)
        if args.render_weight:
            with torch.autocast(device_type=device.type, enabled=False):
                render_error = _sampled_render_loss(
                    output.features,
                    target,
                    normalizer,
                    points_per_example=args.render_points,
                )
                loss = loss.float() + args.render_weight * render_error
            terms["render"] = render_error.detach()
        else:
            terms["render"] = loss.detach().new_zeros(())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))

        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            seconds_per_step = (now - interval_started) / args.log_every
            interval_started = now
            record = {
                "step": step,
                "loss": sum(losses[-args.log_every :]) / min(len(losses), args.log_every),
                "feature": float(terms["feature"]),
                "existence": float(terms["existence"]),
                "count": float(terms["count"]),
                "render": float(terms["render"]),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": seconds_per_step,
            }
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)

        should_evaluate = step % args.eval_every == 0 or step == args.steps
        if should_evaluate:
            latest_evaluation = evaluate_roundtrip(
                model,
                split.validation,
                normalizer,
                device=device,
                points_per_example=args.eval_points,
                max_examples=args.eval_examples,
                seed=args.seed,
            )
            evaluation_record = {"step": step, "evaluation": latest_evaluation.to_dict()}
            _append_json(log_path, evaluation_record)
            print(json.dumps(evaluation_record), flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest_evaluation)

    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "latest_evaluation": latest_evaluation.to_dict() if latest_evaluation else None,
        "max_cell_occupancy": max_occupancy,
        "validation_sources": split.validation_sources,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
