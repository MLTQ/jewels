"""Train scaffold-conditioned discrete birth topology on UCF and evaluate on LTX."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

import torch

from sol.checkpoint_transfer import load_compatible_model_weights
from sol.prompt_embeddings import load_prompt_cache
from sol.scaffold_topology import ScaffoldTopologyModel, ScaffoldTopologyOutput
from sol.scaffold_topology_data import (
    ScaffoldTopologyView,
    build_scaffold_topology_views,
    rasterize_carried_state,
)
from sol.scaffold_topology_eval import (
    TopologyControlView,
    calibrate_occupancy_threshold,
    evaluate_topology_controls,
)
from sol.scaffold_topology_rollout import rollout_oracle_matched_topology
from sol.streaming_corpus import PromptedField, load_prompted_fields
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


@dataclass(frozen=True)
class PreparedTopologyView:
    topology: ScaffoldTopologyView
    guide_raster: torch.Tensor
    carry_raster: torch.Tensor


@dataclass(frozen=True)
class PreparedTopologySource:
    field: PromptedField
    views: tuple[PreparedTopologyView, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=3)
    parser.add_argument("--cell-depth", type=int, default=2)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--guide-height", type=int, default=24)
    parser.add_argument("--guide-width", type=int, default=40)
    parser.add_argument("--occupancy-weight", type=float, default=1.0)
    parser.add_argument("--positive-count-weight", type=float, default=1.0)
    parser.add_argument("--total-count-weight", type=float, default=0.25)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--transfer-from",
        help="model-only initialization from a compatible checkpoint on another manifest",
    )
    parser.add_argument(
        "--diagnostic-validation-source-id",
        action="append",
        default=[],
        help=(
            "diagnostic-only split override: hold out these source IDs and train on "
            "every other loaded field"
        ),
    )
    return parser.parse_args()


def _prepare_sources(
    fields: list[PromptedField],
    manifest: dict,
    spec: GridSpec,
    *,
    stride_frames: int,
    support_sigma: float,
    guide_height: int,
    guide_width: int,
) -> tuple[PreparedTopologySource, ...]:
    manifest_sources = {item["source_id"]: item for item in manifest["examples"]}
    prepared = []
    for field in fields:
        source = manifest_sources.get(field.source_id)
        if source is None:
            raise ValueError(f"manifest is missing source {field.source_id!r}")
        video = load_video(
            source["video"],
            max_frames=field.frames,
            start_frame=int(source.get("start_frame", 0)),
            resize=(guide_height, guide_width),
            device="cpu",
        )
        if len(video) != field.frames:
            raise ValueError(f"video length disagrees with field: {field.source_id}")
        topology_views = build_scaffold_topology_views(
            field.features,
            field.frames,
            stride_frames=stride_frames,
            support_sigma=support_sigma,
            grid_spec=spec,
        )
        views = []
        for view in topology_views:
            views.append(
                PreparedTopologyView(
                    topology=view,
                    guide_raster=video_to_cell_raster(
                        video[view.frontier : view.commit_stop], spec
                    ),
                    carry_raster=rasterize_carried_state(
                        view.carried_global_features,
                        field.frames,
                        view.frontier,
                        stride_frames,
                        spec,
                        support_sigma=support_sigma,
                    ),
                )
            )
        prepared.append(PreparedTopologySource(field, tuple(views)))
    return tuple(prepared)


def _flatten(
    sources: tuple[PreparedTopologySource, ...], split: str
) -> list[tuple[PreparedTopologySource, PreparedTopologyView]]:
    return [
        (source, view)
        for source in sources
        if source.field.split == split
        for view in source.views
    ]


def _diagnostic_split_override(
    sources: tuple[PreparedTopologySource, ...],
    validation_source_ids: list[str],
) -> tuple[PreparedTopologySource, ...]:
    """Apply an explicit source-level cross-validation split for diagnostics."""
    if not validation_source_ids:
        return sources
    requested = set(validation_source_ids)
    if len(requested) != len(validation_source_ids):
        raise ValueError("diagnostic validation source IDs must be unique")
    available = {source.field.source_id for source in sources}
    missing = requested - available
    if missing:
        raise ValueError(f"unknown diagnostic validation source IDs: {sorted(missing)}")
    if requested == available:
        raise ValueError("diagnostic split must retain at least one training source")
    return tuple(
        replace(
            source,
            field=replace(
                source.field,
                split=(
                    "validation"
                    if source.field.source_id in requested
                    else "train"
                ),
            ),
        )
        for source in sources
    )


def _mean_counts_by_index(
    rows: list[tuple[PreparedTopologySource, PreparedTopologyView]],
) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for _, view in rows:
        grouped.setdefault(view.topology.index, []).append(
            view.topology.births.counts.float()
        )
    if not grouped:
        raise ValueError("train-mean topology requires training rows")
    return {index: torch.stack(values).mean(dim=0) for index, values in grouped.items()}


def _control_views(
    rows: list[tuple[PreparedTopologySource, PreparedTopologyView]],
) -> list[TopologyControlView]:
    return [
        TopologyControlView(
            source_id=source.field.source_id,
            class_id=source.field.class_id,
            class_name=source.field.class_name,
            index=view.topology.index,
            guide_raster=view.guide_raster,
            carry_raster=view.carry_raster,
            target_counts=view.topology.births.counts,
        )
        for source, view in rows
    ]


def _atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


@torch.no_grad()
def _calibrate(
    model: ScaffoldTopologyModel,
    rows: list[tuple[PreparedTopologySource, PreparedTopologyView]],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    outputs = []
    targets = []
    for _, view in rows:
        output = model(
            view.guide_raster.to(device), view.carry_raster.to(device)
        )
        outputs.append(
            ScaffoldTopologyOutput(
                output.occupancy_logits.cpu(), output.positive_count_raw.cpu()
            )
        )
        targets.append(view.topology.births.counts)
    return calibrate_occupancy_threshold(
        outputs, targets, slots_per_cell=model.grid_spec.slots_per_cell
    )


def main() -> None:
    args = _parse_args()
    weights = (
        args.occupancy_weight,
        args.positive_count_weight,
        args.total_count_weight,
        args.distribution_weight,
    )
    if args.steps <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.warmup < 0:
        raise ValueError("training schedule is outside its valid range")
    if min(args.guide_height, args.guide_width) <= 0 or min(weights) < 0:
        raise ValueError("guide dimensions and loss weights are invalid")
    if args.resume and args.transfer_from:
        raise ValueError("resume and transfer-from are mutually exclusive")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    spec = GridSpec(tuple(args.grid), args.slots)
    sources = _prepare_sources(
        fields,
        manifest,
        spec,
        stride_frames=args.stride_frames,
        support_sigma=args.support_sigma,
        guide_height=args.guide_height,
        guide_width=args.guide_width,
    )
    sources = _diagnostic_split_override(
        sources, args.diagnostic_validation_source_id
    )
    train_rows = _flatten(sources, "train")
    validation_rows = _flatten(sources, "validation")
    if not train_rows or not validation_rows:
        raise ValueError("topology training requires non-empty train and validation splits")
    mean_counts = _mean_counts_by_index(train_rows)
    validation_controls = _control_views(validation_rows)

    model_args = {
        "guide_dim": 3,
        "carry_dim": 3,
        "model_dim": args.model_dim,
        "encoder_depth": args.encoder_depth,
        "cell_depth": args.cell_depth,
    }
    model = ScaffoldTopologyModel(grid_spec=spec, **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "scaffold_topology.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    latest = None
    initialization = None
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])
        initialization = saved.get("meta", {}).get("initialization")
    elif args.transfer_from:
        initialization = load_compatible_model_weights(
            model,
            args.transfer_from,
            map_location=device,
            architecture="scaffold_topology_v1",
            model_args=model_args,
            grid_spec=spec,
            destination_manifest_sha256=prompt_cache.manifest_sha256,
            allow_cross_manifest=True,
        )

    def save(step: int, evaluation: dict | None) -> None:
        _atomic_save(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "scaffold_topology_v1",
                    "model_args": model_args,
                    "grid_shape": spec.shape,
                    "slots_per_cell": spec.slots_per_cell,
                    "manifest": args.manifest,
                    "manifest_sha256": prompt_cache.manifest_sha256,
                    "training_sources": [
                        source.field.source_id
                        for source in sources
                        if source.field.split == "train"
                    ],
                    "validation_sources": [
                        source.field.source_id
                        for source in sources
                        if source.field.split == "validation"
                    ],
                    "split_policy": (
                        "diagnostic_source_cross_validation"
                        if args.diagnostic_validation_source_id
                        else "manifest"
                    ),
                    "transferred_from": args.transfer_from,
                    "initialization": initialization,
                    "train_args": vars(args),
                    "latest_evaluation": evaluation,
                },
            },
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train_sources={sum(source.field.split == 'train' for source in sources)} "
        f"validation_sources={sum(source.field.split == 'validation' for source in sources)} "
        f"train_views={len(train_rows)} validation_views={len(validation_rows)} "
        f"model={parameters / 1e6:.2f}M amp={use_amp}",
        flush=True,
    )
    losses = []
    interval_started = time.time()
    started = interval_started
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            len(train_rows), (args.batch_size,), generator=generator
        ).tolist()
        batch = [train_rows[index][1] for index in indices]
        guide = torch.stack([view.guide_raster for view in batch]).to(device)
        carry = torch.stack([view.carry_raster for view in batch]).to(device)
        target = torch.stack(
            [view.topology.births.counts for view in batch]
        ).to(device)
        if step <= args.warmup:
            learning_rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            learning_rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(guide, carry)
            loss, terms = model.loss(
                output,
                target,
                occupancy_weight=args.occupancy_weight,
                positive_count_weight=args.positive_count_weight,
                total_count_weight=args.total_count_weight,
                distribution_weight=args.distribution_weight,
            )
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
                **{name: float(value) for name, value in terms.items()},
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / count,
            }
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            threshold, calibration = _calibrate(model, train_rows, device)
            controls = evaluate_topology_controls(
                model,
                validation_controls,
                mean_counts,
                occupancy_threshold=threshold,
                device=device,
            )
            latest = {"train_calibration": calibration, "validation": controls}
            record = {"step": step, "evaluation": latest}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)

    threshold, calibration = _calibrate(model, train_rows, device)
    controls = evaluate_topology_controls(
        model,
        validation_controls,
        mean_counts,
        occupancy_threshold=threshold,
        device=device,
    )
    rollouts = {}
    for source in sources:
        if source.field.split != "validation":
            continue
        result = rollout_oracle_matched_topology(
            model,
            tuple(view.topology for view in source.views),
            [view.guide_raster for view in source.views],
            source.field.features,
            source.field.frames,
            spec,
            stride_frames=args.stride_frames,
            support_sigma=args.support_sigma,
            occupancy_threshold=threshold,
            device=device,
        )
        rollouts[source.field.source_id] = result.report
        torch.save(
            {"features": result.features, "global_ids": result.global_ids},
            output_dir / f"{source.field.source_id}_oracle_rollout.pt",
        )
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "train_views": len(train_rows),
        "validation_views": len(validation_rows),
        "train_calibration": calibration,
        "validation": controls,
        "oracle_rollouts": rollouts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save(args.steps, {"train_calibration": calibration, "validation": controls})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
