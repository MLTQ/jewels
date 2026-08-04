"""Train the sparse variable-count tokenizer on dense fitted checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from sol.corpus import (
    FeatureNormalizer,
    FittedExample,
    SourceSplit,
    load_fitted_corpus,
    split_by_source,
)
from sol.domain_sampling import sample_domain_balanced_indices
from sol.evaluation import EvaluationReport, evaluate_roundtrip
from sol.render import render_exact
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import CompactGrid, GridSpec, OccupancyGrid


@dataclass
class PreparedDenseExample:
    features: torch.Tensor
    target: CompactGrid
    motion_points: torch.Tensor | None = None


def _prepare_examples(
    examples: list[FittedExample],
    normalizer: FeatureNormalizer,
    grid: OccupancyGrid,
    motion_points_dir: Path | None = None,
) -> list[PreparedDenseExample]:
    prepared = []
    for example in examples:
        features = normalizer.normalize(example.features)
        motion_points = None
        if motion_points_dir is not None:
            sidecar = motion_points_dir / f"{Path(example.name).stem}.motion.pt"
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            motion_points = payload["points"].float()
            if motion_points.ndim != 2 or motion_points.shape[1] != 3:
                raise ValueError(f"invalid motion point pool in {sidecar}")
        prepared.append(
            PreparedDenseExample(
                features, grid.pack_compact(features), motion_points
            )
        )
    return prepared


def _batch(
    prepared: list[PreparedDenseExample],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, CompactGrid, torch.Tensor | None]:
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
    pools = [example.motion_points for example in selected]
    motion_points = None
    if any(pool is not None for pool in pools):
        if not all(pool is not None for pool in pools):
            raise ValueError("batch mixes examples with and without motion pools")
        motion_points = torch.stack(pools).to(device)  # type: ignore[arg-type]
    return features, target, motion_points


def _sampled_render_loss(
    predicted_normalized: torch.Tensor,
    target: CompactGrid,
    normalizer: FeatureNormalizer,
    *,
    points_per_example: int,
    motion_fraction: float = 0.0,
    motion_candidate_multiplier: int = 8,
    motion_time_delta: float = 0.05,
    motion_point_pool: torch.Tensor | None = None,
) -> torch.Tensor:
    if points_per_example <= 0:
        raise ValueError("render points must be positive")
    if not 0 <= motion_fraction <= 1:
        raise ValueError("motion fraction must be in [0,1]")
    if motion_candidate_multiplier <= 0 or motion_time_delta <= 0:
        raise ValueError("motion sampling settings must be positive")
    mean = normalizer.mean.to(predicted_normalized.device, dtype=torch.float32)
    std = normalizer.std.to(predicted_normalized.device, dtype=torch.float32)
    predicted = predicted_normalized.float() * std + mean
    expected = target.values.float() * std + mean
    losses = []
    for batch_index in range(len(predicted)):
        motion_points = round(points_per_example * motion_fraction)
        uniform_points = points_per_example - motion_points
        pieces = []
        if uniform_points:
            pieces.append(
                torch.rand(
                    uniform_points,
                    3,
                    device=predicted.device,
                    dtype=torch.float32,
                )
                * 2
                - 1
            )
        if motion_points:
            if motion_point_pool is not None:
                pool = motion_point_pool[batch_index]
                selected = torch.randint(
                    len(pool), (motion_points,), device=predicted.device
                )
                pieces.append(pool[selected].float())
            else:
                candidate_count = max(
                    motion_points, points_per_example * motion_candidate_multiplier
                )
                candidates = torch.rand(
                    candidate_count,
                    3,
                    device=predicted.device,
                    dtype=torch.float32,
                ) * 2 - 1
                before = candidates.clone()
                after = candidates.clone()
                before[:, 2] = (before[:, 2] - motion_time_delta).clamp(-1, 1)
                after[:, 2] = (after[:, 2] + motion_time_delta).clamp(-1, 1)
                with torch.no_grad():
                    before_render = render_exact(expected[batch_index], before).clamp(0, 1)
                    after_render = render_exact(expected[batch_index], after).clamp(0, 1)
                    salience = (after_render - before_render).abs().mean(dim=-1)
                selected = salience.topk(motion_points).indices
                pieces.append(candidates[selected])
        points = torch.cat(pieces, dim=0)
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
    parser.add_argument("--extra-corpus", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--grid", type=int, nargs=3, default=(12, 12, 6))
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--enc-depth", type=int, default=3)
    parser.add_argument("--encoder-mode", choices=("pooled", "rank"), default="pooled")
    parser.add_argument("--dec-depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decode-chunk", type=int, default=32768)
    parser.add_argument("--validation-sources", type=int, default=2)
    parser.add_argument("--validation-source-ids", nargs="*", default=[])
    parser.add_argument(
        "--sampling",
        choices=("window", "domain-balanced"),
        default="window",
    )
    parser.add_argument("--overfit-name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-examples", type=int, default=2)
    parser.add_argument("--eval-points", type=int, default=256)
    parser.add_argument("--render-weight", type=float, default=0.5)
    parser.add_argument("--render-points", type=int, default=16)
    parser.add_argument("--motion-render-fraction", type=float, default=0.0)
    parser.add_argument("--motion-candidate-multiplier", type=int, default=8)
    parser.add_argument("--motion-time-delta", type=float, default=0.05)
    parser.add_argument("--motion-points-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.batch <= 0 or args.decode_chunk <= 0:
        raise ValueError("steps, batch, and decode chunk must be positive")
    if args.render_weight < 0 or args.render_points <= 0:
        raise ValueError("render settings are invalid")
    if not 0 <= args.motion_render_fraction <= 1:
        raise ValueError("motion render fraction must be in [0,1]")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "autoencoder.pt"
    log_path = output_dir / "train_log.jsonl"

    corpus_paths = [args.corpus, *args.extra_corpus]
    examples = load_fitted_corpus(corpus_paths, limit=args.limit)
    if args.overfit_name:
        selected = [example for example in examples if example.name == args.overfit_name]
        if len(selected) != 1:
            raise ValueError(
                f"overfit name must identify exactly one example, found {len(selected)}"
            )
        split = SourceSplit(selected, selected, (selected[0].source_id,))
    else:
        split = split_by_source(
            examples,
            validation_sources=args.validation_sources,
            seed=args.seed,
            held_out_sources=args.validation_source_ids or None,
        )
    normalizer = FeatureNormalizer.fit(
        split.train, balance_domains=args.sampling == "domain-balanced"
    )
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
        f"held_out={split.validation_sources} sampling={args.sampling}",
        flush=True,
    )
    domain_counts: dict[str, int] = {}
    for example in split.train:
        domain_counts[example.domain_id] = domain_counts.get(example.domain_id, 0) + 1
    print(f"train_domains={dict(sorted(domain_counts.items()))}", flush=True)
    print(
        f"grid={spec.shape} cells={spec.n_cells} capacity={spec.slots_per_cell} "
        f"observed_max={max_occupancy} actual_jewels={examples[0].features.shape[0]}",
        flush=True,
    )
    prepared = _prepare_examples(
        split.train,
        normalizer,
        grid,
        Path(args.motion_points_dir) if args.motion_points_dir else None,
    )
    model_args = {
        "feature_dim": examples[0].features.shape[-1],
        "model_dim": args.model_dim,
        "latent_dim": args.latent_dim,
        "spec": spec,
        "enc_depth": args.enc_depth,
        "dec_depth": args.dec_depth,
        "heads": args.heads,
        "decode_chunk_size": args.decode_chunk,
        "encoder_mode": args.encoder_mode,
    }
    model = SparseJewelAutoencoder(**model_args).to(device)
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
    parameters = sum(parameter.numel() for parameter in model.parameters())
    raw_numbers = examples[0].features.numel()
    latent_numbers = spec.n_cells * args.latent_dim
    padded_slots = spec.n_cells * spec.slots_per_cell
    print(
        f"model={parameters / 1e6:.2f}M latent={latent_numbers} numbers "
        f"compression={raw_numbers / latent_numbers:.2f}x "
        f"requested/padded={examples[0].features.shape[0]}/{padded_slots} amp={use_amp}",
        flush=True,
    )

    def lr_at(step: int) -> float:
        if step <= args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    def save(step: int, latest: EvaluationReport | None) -> None:
        _atomic_checkpoint(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "sparse_variable_count_v1",
                    "model_args": {
                        key: value for key, value in model_args.items() if key != "spec"
                    },
                    "grid_shape": spec.shape,
                    "slots_per_cell": spec.slots_per_cell,
                    "normalizer": normalizer.state_dict(),
                    "validation_sources": split.validation_sources,
                    "corpus": corpus_paths,
                    "training_domains": dict(sorted(domain_counts.items())),
                    "train_args": vars(args),
                    "latest_evaluation": latest.to_dict() if latest else None,
                },
            },
        )

    latest = evaluate_roundtrip(
        model,
        split.validation,
        normalizer,
        device=device,
        points_per_example=args.eval_points,
        max_examples=args.eval_examples,
        seed=args.seed,
    ) if start_step == 0 else None
    if latest:
        record = {"step": 0, "evaluation": latest.to_dict()}
        _append_json(log_path, record)
        print(json.dumps(record), flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    started = time.time()
    interval_started = started
    for step in range(start_step + 1, args.steps + 1):
        if args.sampling == "domain-balanced":
            indices = sample_domain_balanced_indices(
                [example.domain_id for example in split.train],
                args.batch,
                step,
                generator,
            )
        else:
            indices = torch.randint(0, len(prepared), (args.batch,), generator=generator)
        features, target, motion_point_pool = _batch(prepared, indices, device)
        learning_rate = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model.forward_compact(features, target)
            loss, terms = model.structural_loss(output, target)
        if args.render_weight:
            with torch.autocast(device_type=device.type, enabled=False):
                render_error = _sampled_render_loss(
                    output.occupied_features,
                    target,
                    normalizer,
                    points_per_example=args.render_points,
                    motion_fraction=args.motion_render_fraction,
                    motion_candidate_multiplier=args.motion_candidate_multiplier,
                    motion_time_delta=args.motion_time_delta,
                    motion_point_pool=motion_point_pool,
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
            count = min(args.log_every, len(losses))
            record = {
                "step": step,
                "loss": sum(losses[-count:]) / count,
                "feature": float(terms["feature"]),
                "count": float(terms["count"]),
                "render": float(terms["render"]),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / count,
            }
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate_roundtrip(
                model,
                split.validation,
                normalizer,
                device=device,
                points_per_example=args.eval_points,
                max_examples=args.eval_examples,
                seed=args.seed,
            )
            record = {"step": step, "evaluation": latest.to_dict()}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "latest_evaluation": latest.to_dict() if latest else None,
        "max_cell_occupancy": max_occupancy,
        "validation_sources": split.validation_sources,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
