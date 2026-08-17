"""Train one 1,024-rank mark flow across initial and continuation strides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, project_birth_topology
from sol.checkpoint_transfer import (
    load_augmented_model_weights,
    load_compatible_model_weights,
)
from sol.frontier_contribution_loss import frontier_contribution_loss
from sol.prompt_embeddings import load_prompt_cache
from sol.realizer_render_loss import (
    estimate_target_marks,
    realizer_render_loss,
    scaffold_saliency_weights,
)
from sol.scaffold_mark_data import (
    ScaffoldMarkCorpus,
    build_scaffold_mark_corpus,
    rasterize_scaffold_context,
)
from sol.scaffold_mark_eval import evaluate_scaffold_mark_flow
from sol.streaming_corpus import load_prompted_fields
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


SPATIAL_APPEARANCE_DIMENSIONS = (0, 1, 3, 4, 6, *range(9, 22))


@dataclass(frozen=True)
class PreparedScaffoldMarkView:
    """One normalized variable-cardinality training row resident on the device."""

    source_id: str
    index: int
    frontier: int
    context_raster: torch.Tensor
    target_values: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    guide_raster: torch.Tensor
    prompt_indices: tuple[int, ...]
    carried_global: torch.Tensor
    background: torch.Tensor
    total_frames: int
    stride_frames: int
    support_sigma: float
    cell_saliency: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--noisy-depth", type=int, default=2)
    parser.add_argument("--guide-depth", type=int, default=2)
    parser.add_argument("--cell-depth", type=int, default=2)
    parser.add_argument("--mark-depth", type=int, default=3)
    parser.add_argument("--set-depth", type=int, default=0)
    parser.add_argument("--set-raster-depth", type=int, default=0)
    parser.add_argument(
        "--set-coupling", choices=("neighborhood", "ssog"), default="neighborhood"
    )
    parser.add_argument("--set-atoms", type=int, default=4)
    parser.add_argument("--set-max-offset", type=float, default=4.0)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--guide-height", type=int, default=24)
    parser.add_argument("--guide-width", type=int, default=40)
    parser.add_argument("--text-dropout", type=float, default=0.15)
    parser.add_argument("--context-dropout", type=float, default=0.15)
    parser.add_argument("--guide-dropout", type=float, default=0.10)
    parser.add_argument("--initial-repeat", type=int, default=1)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--feature-saliency-weight", type=float, default=0.0)
    parser.add_argument(
        "--feature-saliency-mode",
        choices=("all", "spatial-appearance"),
        default="all",
    )
    parser.add_argument("--render-weight", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--render-patches", type=int, default=2)
    parser.add_argument("--render-patch-frames", type=int, default=2)
    parser.add_argument("--render-patch-height", type=int, default=4)
    parser.add_argument("--render-patch-width", type=int, default=4)
    parser.add_argument("--render-rgb-weight", type=float, default=1.0)
    parser.add_argument("--render-edge-weight", type=float, default=0.25)
    parser.add_argument("--render-chroma-weight", type=float, default=0.25)
    parser.add_argument("--render-structure-weight", type=float, default=0.25)
    parser.add_argument("--render-saliency-weight", type=float, default=0.0)
    parser.add_argument("--render-motion-weight", type=float, default=0.0)
    parser.add_argument("--render-stability-weight", type=float, default=0.0)
    parser.add_argument("--render-saliency-fraction", type=float, default=0.0)
    parser.add_argument("--render-anchor-frontier", action="store_true")
    parser.add_argument(
        "--learn-single-background",
        action="store_true",
        help=(
            "learn one RGB background from a causal scaffold initialization; "
            "requires exactly one physical training field and render supervision"
        ),
    )
    parser.add_argument("--frontier-weight", type=float, default=0.0)
    parser.add_argument("--frontier-every", type=int, default=1)
    parser.add_argument("--frontier-visible-threshold", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="retain immutable step checkpoints in addition to the resumable latest file",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from")
    parser.add_argument(
        "--augment-from",
        help="same-manifest base checkpoint for a zero-residual coupled-set model",
    )
    parser.add_argument(
        "--freeze-base-on-augment",
        action="store_true",
        help="optimize only the new set blocks after exact base initialization",
    )
    parser.add_argument(
        "--transfer-from",
        help="model-only initialization from a compatible checkpoint on another manifest",
    )
    return parser.parse_args()


def _load_guides(
    corpus: ScaffoldMarkCorpus,
    manifest: dict,
    *,
    height: int,
    width: int,
) -> dict[tuple[str, int], torch.Tensor]:
    """Load and align every complete scaffold stride with the birth grid."""
    if min(height, width) <= 0:
        raise ValueError("guide dimensions must be positive")
    manifest_sources = {item["source_id"]: item for item in manifest["examples"]}
    guides = {}
    for source in corpus.sources:
        item = manifest_sources.get(source.field.source_id)
        if item is None:
            raise ValueError(f"manifest is missing source {source.field.source_id!r}")
        video = load_video(
            item["video"],
            max_frames=source.field.frames,
            start_frame=int(item.get("start_frame", 0)),
            resize=(height, width),
            device="cpu",
        )
        if len(video) != source.field.frames:
            raise ValueError(f"video length disagrees with field: {source.field.source_id}")
        for view in source.views:
            guides[(source.field.source_id, view.index)] = video_to_cell_raster(
                video[view.frontier : view.commit_stop], corpus.grid_spec
            )
    return guides


def _prepare(
    corpus: ScaffoldMarkCorpus,
    guide_rasters: dict[tuple[str, int], torch.Tensor],
    device: torch.device,
    backgrounds: dict[str, torch.Tensor] | None = None,
) -> list[PreparedScaffoldMarkView]:
    prepared = []
    for source in corpus.train:
        for view in source.views:
            if not len(view.births.values):
                continue
            key = (source.field.source_id, view.index)
            if key not in guide_rasters:
                raise ValueError(f"missing scaffold guide for {key}")
            background = (
                backgrounds[source.field.source_id]
                if backgrounds is not None
                else torch.zeros(3)
            )
            cell_saliency = scaffold_saliency_weights(
                guide_rasters[key], corpus.grid_spec.shape, background
            )
            cell_saliency = cell_saliency / cell_saliency.mean().clamp_min(1e-4)
            prepared.append(
                PreparedScaffoldMarkView(
                    source.field.source_id,
                    view.index,
                    view.frontier,
                    rasterize_scaffold_context(
                        view.context_features,
                        corpus.context_standardizer,
                        stride_frames=corpus.stride_frames,
                        grid_spec=corpus.grid_spec,
                    ).to(device),
                    corpus.birth_standardizer.normalize(view.births.values).to(device),
                    view.births.cell_indices.to(device),
                    view.births.slot_indices.to(device),
                    guide_rasters[key].to(device),
                    source.field.train_prompt_indices,
                    view.carried_global_features.to(device),
                    background.to(device),
                    source.field.frames,
                    corpus.stride_frames,
                    corpus.support_sigma,
                    cell_saliency.to(device),
                )
            )
    if not prepared or not any(view.frontier == 0 for view in prepared):
        raise ValueError("training rows require birth-bearing initial and later views")
    return prepared


def _feature_objective(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_saliency: torch.Tensor,
    saliency_weight: float,
    salient_dimensions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return mean-normalized scaffold-salient per-jewel velocity MSE."""
    if predicted.shape != expected.shape or predicted.ndim != 2:
        raise ValueError("feature objective expects matching mark matrices")
    if cell_indices.shape != (len(predicted),):
        raise ValueError("feature objective requires one cell per mark")
    if saliency_weight < 0 or cell_saliency.ndim != 1:
        raise ValueError("feature saliency contract is invalid")
    error = (predicted.float() - expected.float()).square()
    if not saliency_weight:
        return error.mean()
    row_weights = 1 + saliency_weight * cell_saliency[cell_indices]
    weights = torch.ones_like(error)
    if salient_dimensions is None:
        weights *= row_weights[:, None]
    else:
        if not salient_dimensions or min(salient_dimensions) < 0 or max(
            salient_dimensions
        ) >= error.shape[1]:
            raise ValueError("salient feature dimensions are invalid")
        weights[:, salient_dimensions] = row_weights[:, None]
    return (error * weights).sum() / weights.sum().clamp_min(1e-8)


def _load_backgrounds(
    manifest: dict, checkpoint_roots: list[str]
) -> dict[str, torch.Tensor]:
    """Load fitted training targets for optional render-supervised fine-tuning."""
    checkpoints = {}
    for root in checkpoint_roots:
        for path in Path(root).glob("*_w000000.pt"):
            if path.name in checkpoints:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            checkpoints[path.name] = path
    backgrounds = {}
    for item in manifest["examples"]:
        name = f"{Path(item['video']).stem}_w000000.pt"
        path = checkpoints.get(name)
        if path is None:
            raise FileNotFoundError(f"missing fitted checkpoint: {name}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        backgrounds[item["source_id"]] = torch.as_tensor(
            saved["info"]["background"], dtype=torch.float32
        ).clone()
    return backgrounds


def _single_background_initialization(
    prepared: list[PreparedScaffoldMarkView],
) -> torch.Tensor:
    """Initialize one learned background from the first causal scaffold stride."""
    source_ids = {view.source_id for view in prepared}
    initial = [view for view in prepared if view.frontier == 0]
    if len(source_ids) != 1 or len(initial) != 1:
        raise ValueError(
            "single-background learning requires one source with one initial view"
        )
    return initial[0].guide_raster.float().mean(dim=0).clamp(1e-4, 1 - 1e-4)


def _training_amp_enabled(
    device: torch.device, *, no_amp: bool, render_weight: float
) -> bool:
    """Keep differentiable jewel-render gradients out of unsafe loss scaling."""
    return device.type == "cuda" and not no_amp and not render_weight


def _atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.lr <= 0 or args.warmup < 0:
        raise ValueError("training schedule is outside its valid range")
    if args.set_depth < 0 or args.set_raster_depth < 0:
        raise ValueError("set depths must be non-negative")
    if not args.set_depth and args.set_raster_depth:
        raise ValueError("set-raster-depth requires set-depth")
    if args.augment_from and not args.set_depth:
        raise ValueError("augment-from requires a positive set-depth")
    if args.freeze_base_on_augment and not (args.augment_from or args.resume):
        raise ValueError("freeze-base-on-augment requires augment-from or resume")
    if args.freeze_base_on_augment and args.learn_single_background:
        raise ValueError(
            "freeze-base-on-augment cannot optimize a newly learned background"
        )
    if args.snapshot_every < 0:
        raise ValueError("snapshot-every must be non-negative")
    if args.initial_repeat <= 0 or args.feature_weight <= 0:
        raise ValueError("initial repeat and feature weight must be positive")
    if args.feature_saliency_weight < 0:
        raise ValueError("feature saliency weight must be non-negative")
    if args.render_weight < 0 or args.render_every <= 0:
        raise ValueError("render weight/cadence are invalid")
    if args.learn_single_background and not args.render_weight:
        raise ValueError("single-background learning requires render supervision")
    if args.frontier_weight < 0 or args.frontier_every <= 0:
        raise ValueError("frontier contribution weight/cadence are invalid")
    if not 0 < args.frontier_visible_threshold < 1:
        raise ValueError("frontier visible threshold must lie inside (0,1)")
    render_weights = (
        args.render_rgb_weight,
        args.render_edge_weight,
        args.render_chroma_weight,
        args.render_structure_weight,
        args.render_saliency_weight,
        args.render_motion_weight,
        args.render_stability_weight,
    )
    if any(weight < 0 for weight in render_weights) or not any(render_weights):
        raise ValueError("render component weights must be non-negative and not all zero")
    if not 0 <= args.render_saliency_fraction <= 1:
        raise ValueError("render saliency fraction must be in [0,1]")
    initialization_paths = (
        args.resume,
        args.initialize_from,
        args.transfer_from,
        args.augment_from,
    )
    if sum(bool(value) for value in initialization_paths) > 1:
        raise ValueError(
            "resume, initialize-from, transfer-from, and augment-from are mutually exclusive"
        )
    dropouts = (args.text_dropout, args.context_dropout, args.guide_dropout)
    if any(value < 0 or value > 1 for value in dropouts):
        raise ValueError("conditioning dropout probabilities must be in [0,1]")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    render_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    spec = GridSpec(tuple(args.grid), args.slots)
    corpus = build_scaffold_mark_corpus(
        fields,
        prompt_cache.embeddings,
        stride_frames=args.stride_frames,
        support_sigma=args.support_sigma,
        grid_spec=spec,
    )
    guides = _load_guides(
        corpus, manifest, height=args.guide_height, width=args.guide_width
    )
    backgrounds = _load_backgrounds(manifest, args.checkpoint_root)
    prepared = _prepare(corpus, guides, device, backgrounds)
    training_schedule = [
        view
        for view in prepared
        for _ in range(args.initial_repeat if view.frontier == 0 else 1)
    ]
    model_args = {
        "feature_dim": 22,
        "context_dim": 46,
        "model_dim": args.model_dim,
        "context_depth": args.context_depth,
        "noisy_depth": args.noisy_depth,
        "guide_depth": args.guide_depth,
        "cell_depth": args.cell_depth,
        "mark_depth": args.mark_depth,
        "text_dim": int(prompt_cache.embeddings.shape[1]),
        "guide_dim": 3,
        "guide_token_dim": 0,
        "guide_heads": 8,
    }
    if args.set_depth:
        model_args.update(
            {
                "set_depth": args.set_depth,
                "set_raster_depth": args.set_raster_depth,
            }
        )
        if args.set_coupling != "neighborhood":
            model_args.update(
                {
                    "set_coupling": args.set_coupling,
                    "set_atoms": args.set_atoms,
                    "set_max_offset": args.set_max_offset,
                }
            )
    model = BirthMarkFlowModel(grid_spec=spec, **model_args).to(device)
    if args.freeze_base_on_augment:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.set_blocks.parameters():
            parameter.requires_grad_(True)
    background_logit = None
    if args.learn_single_background:
        initial_background = _single_background_initialization(prepared).to(device)
        background_logit = torch.nn.Parameter(torch.logit(initial_background))
    model_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not model_parameters:
        raise ValueError("training configuration has no trainable model parameters")
    trainable_parameters = list(model_parameters)
    parameter_groups = [{"params": model_parameters, "weight_decay": 0.01}]
    if background_logit is not None:
        trainable_parameters.append(background_logit)
        parameter_groups.append({"params": [background_logit], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr)
    use_amp = _training_amp_enabled(
        device, no_amp=args.no_amp, render_weight=args.render_weight
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "scaffold_mark_flow.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    latest = None
    initialization = None
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if background_logit is not None:
            if saved.get("background_logit") is None:
                raise ValueError("resume checkpoint lacks learned background state")
            background_logit.data.copy_(saved["background_logit"].to(device))
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])
        initialization = saved.get("meta", {}).get("initialization")
    elif args.initialize_from:
        initialization = load_compatible_model_weights(
            model,
            args.initialize_from,
            map_location=device,
            architecture="scaffold_birth_mark_flow_v1",
            model_args=model_args,
            grid_spec=spec,
            destination_manifest_sha256=prompt_cache.manifest_sha256,
            allow_cross_manifest=False,
        )
    elif args.transfer_from:
        initialization = load_compatible_model_weights(
            model,
            args.transfer_from,
            map_location=device,
            architecture="scaffold_birth_mark_flow_v1",
            model_args=model_args,
            grid_spec=spec,
            destination_manifest_sha256=prompt_cache.manifest_sha256,
            allow_cross_manifest=True,
        )
    elif args.augment_from:
        initialization = load_augmented_model_weights(
            model,
            args.augment_from,
            map_location=device,
            architecture="scaffold_birth_mark_flow_v1",
            model_args=model_args,
            added_model_args={
                "set_depth": args.set_depth,
                "set_raster_depth": args.set_raster_depth,
                **(
                    {
                        "set_coupling": args.set_coupling,
                        "set_atoms": args.set_atoms,
                        "set_max_offset": args.set_max_offset,
                    }
                    if args.set_coupling != "neighborhood"
                    else {}
                ),
            },
            added_state_prefixes=("set_blocks.",),
            grid_spec=spec,
            destination_manifest_sha256=prompt_cache.manifest_sha256,
        )

    def save(step: int, evaluation: dict | None) -> None:
        state = {
            "model": model.state_dict(),
            "background_logit": (
                background_logit.detach().cpu()
                if background_logit is not None
                else None
            ),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "meta": {
                "architecture": "scaffold_birth_mark_flow_v1",
                "model_args": model_args,
                "grid_shape": spec.shape,
                "slots_per_cell": spec.slots_per_cell,
                "manifest": args.manifest,
                "manifest_sha256": prompt_cache.manifest_sha256,
                "prompt_encoder": prompt_cache.encoder,
                "training_sources": [source.field.source_id for source in corpus.train],
                "validation_sources": [
                    source.field.source_id for source in corpus.validation
                ],
                "context_standardizer": corpus.context_standardizer.state_dict(),
                "birth_standardizer": corpus.birth_standardizer.state_dict(),
                "background_contract": (
                    "single_field_learned_rgb"
                    if background_logit is not None
                    else "initial_scaffold_rgb_mean"
                ),
                "learned_background": (
                    torch.sigmoid(background_logit).detach().cpu()
                    if background_logit is not None
                    else None
                ),
                "initialized_from": args.initialize_from,
                "transferred_from": args.transfer_from,
                "augmented_from": (
                    args.augment_from
                    or (
                        initialization.get("path")
                        if initialization is not None
                        and initialization.get("mode")
                        == "same_manifest_architecture_augmentation"
                        else None
                    )
                ),
                "frozen_base_on_augment": args.freeze_base_on_augment,
                "initialization": initialization,
                "train_args": vars(args),
                "latest_evaluation": evaluation,
            },
        }
        _atomic_save(checkpoint_path, state)
        if args.snapshot_every and step % args.snapshot_every == 0:
            _atomic_save(
                output_dir / f"scaffold_mark_flow_step{step:06d}.pt",
                state,
            )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"train_sources={len(corpus.train)} validation_sources={len(corpus.validation)} "
        f"train_views={len(prepared)} initial_views="
        f"{sum(view.frontier == 0 for view in prepared)} model={parameters / 1e6:.2f}M "
        f"trainable={trainable / 1e6:.2f}M schedule_views={len(training_schedule)} "
        f"amp={use_amp}",
        flush=True,
    )
    history = []
    interval_started = time.time()
    started = interval_started
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        view = training_schedule[(step - 1) % len(training_schedule)]
        prompt_offset = ((step - 1) // len(training_schedule)) % len(
            view.prompt_indices
        )
        text = corpus.prompt_embeddings[view.prompt_indices[prompt_offset]].to(device)
        drop_text = torch.rand(1, device=device, generator=generator) < args.text_dropout
        drop_context = bool(
            torch.rand(1, device=device, generator=generator).item()
            < args.context_dropout
        )
        drop_guide = bool(
            torch.rand(1, device=device, generator=generator).item()
            < args.guide_dropout
        )
        context = torch.zeros_like(view.context_raster) if drop_context else view.context_raster
        guide = torch.zeros_like(view.guide_raster) if drop_guide else view.guide_raster
        noise = torch.randn(view.target_values.shape, device=device, generator=generator)
        flow_time = torch.rand(1, device=device, generator=generator)
        if step <= args.warmup:
            learning_rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            learning_rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            noised = (1 - flow_time) * noise + flow_time * view.target_values
            expected_velocity = view.target_values - noise
            predicted_velocity = model(
                context,
                noised,
                flow_time,
                view.cell_indices,
                view.slot_indices,
                text,
                drop_condition=drop_text,
                guide_raster=guide,
            )
            feature_loss = _feature_objective(
                predicted_velocity,
                expected_velocity,
                view.cell_indices,
                view.cell_saliency,
                args.feature_saliency_weight,
                (
                    SPATIAL_APPEARANCE_DIMENSIONS
                    if args.feature_saliency_mode == "spatial-appearance"
                    else None
                ),
            )
        render_terms = None
        frontier_terms = None
        loss = args.feature_weight * feature_loss
        render_update = bool(args.render_weight and step % args.render_every == 0)
        frontier_update = bool(
            args.frontier_weight and step % args.frontier_every == 0
        )
        if render_update or frontier_update:
            estimated = estimate_target_marks(
                noised.float(), predicted_velocity.float(), flow_time.float()
            )
            estimated_local = corpus.birth_standardizer.denormalize(estimated)
            estimated_local = project_birth_topology(
                estimated_local,
                view.cell_indices,
                spec=spec,
                support_sigma=view.support_sigma,
                stride_frames=view.stride_frames,
                allow_prefrontier_support=view.frontier == 0,
            )
            target_local = corpus.birth_standardizer.denormalize(
                view.target_values.float()
            )
        if frontier_update:
            frontier_terms = frontier_contribution_loss(
                estimated_local,
                target_local,
                view.cell_indices,
                n_cells=spec.n_cells,
                visible_threshold=args.frontier_visible_threshold,
            )
            loss = (
                loss
                + args.frontier_weight
                * args.frontier_every
                * frontier_terms.total
            )
        if render_update:
            render_terms = realizer_render_loss(
                estimated_local,
                target_local,
                view.carried_global,
                total_frames=view.total_frames,
                frontier=view.frontier,
                stride_frames=view.stride_frames,
                background=view.background,
                candidate_background=(
                    torch.sigmoid(background_logit)
                    if background_logit is not None
                    else None
                ),
                render_height=args.guide_height,
                render_width=args.guide_width,
                patches=args.render_patches,
                patch_frames=args.render_patch_frames,
                patch_height=args.render_patch_height,
                patch_width=args.render_patch_width,
                rgb_weight=args.render_rgb_weight,
                edge_weight=args.render_edge_weight,
                chroma_weight=args.render_chroma_weight,
                structure_weight=args.render_structure_weight,
                saliency_weight=args.render_saliency_weight,
                motion_weight=args.render_motion_weight,
                stability_weight=args.render_stability_weight,
                guide_raster=view.guide_raster,
                guide_grid_shape=spec.shape,
                saliency_fraction=args.render_saliency_fraction,
                anchor_frontier=args.render_anchor_frontier,
                generator=render_generator,
            )
            loss = loss + args.render_weight * args.render_every * render_terms.total
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "loss": float(loss.detach()),
                "feature_loss": float(feature_loss.detach()),
                "render_loss": (
                    float(render_terms.total.detach()) if render_terms is not None else None
                ),
                "render_rgb": (
                    float(render_terms.rgb.detach()) if render_terms is not None else None
                ),
                "render_saliency": (
                    float(render_terms.saliency_rgb.detach())
                    if render_terms is not None
                    else None
                ),
                "render_motion": (
                    float(render_terms.motion.detach())
                    if render_terms is not None
                    else None
                ),
                "render_stability": (
                    float(render_terms.stability.detach())
                    if render_terms is not None
                    else None
                ),
                "frontier_loss": (
                    float(frontier_terms.total.detach())
                    if frontier_terms is not None
                    else None
                ),
                "frontier_per_jewel": (
                    float(frontier_terms.per_jewel.detach())
                    if frontier_terms is not None
                    else None
                ),
                "frontier_cell_alpha": (
                    float(frontier_terms.cell_alpha.detach())
                    if frontier_terms is not None
                    else None
                ),
                "frontier_visible_count": (
                    float(frontier_terms.visible_count.detach())
                    if frontier_terms is not None
                    else None
                ),
            }
        )
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            recent = history[-args.log_every :]
            render_recent = [row for row in recent if row["render_loss"] is not None]
            frontier_recent = [
                row for row in recent if row["frontier_loss"] is not None
            ]
            record = {
                "step": step,
                "loss": sum(row["loss"] for row in recent) / len(recent),
                "feature_loss": sum(row["feature_loss"] for row in recent)
                / len(recent),
                "render_loss": (
                    sum(float(row["render_loss"]) for row in render_recent)
                    / len(render_recent)
                    if render_recent
                    else None
                ),
                "render_updates": len(render_recent),
                "frontier_updates": len(frontier_recent),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / len(recent),
                "learned_background": (
                    torch.sigmoid(background_logit).detach().cpu().tolist()
                    if background_logit is not None
                    else None
                ),
            }
            for name in (
                "render_rgb",
                "render_saliency",
                "render_motion",
                "render_stability",
            ):
                record[name] = (
                    sum(float(row[name]) for row in render_recent)
                    / len(render_recent)
                    if render_recent
                    else None
                )
            for name in (
                "frontier_loss",
                "frontier_per_jewel",
                "frontier_cell_alpha",
                "frontier_visible_count",
            ):
                record[name] = (
                    sum(float(row[name]) for row in frontier_recent)
                    / len(frontier_recent)
                    if frontier_recent
                    else None
                )
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate_scaffold_mark_flow(
                model, corpus, guides, device=device, seed=args.seed
            )
            record = {"step": step, "evaluation": latest}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
            model.train()
        snapshot_step = bool(args.snapshot_every and step % args.snapshot_every == 0)
        if step % args.checkpoint_every == 0 or snapshot_step or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "train_views": len(prepared),
        "schedule_views": len(training_schedule),
        "initial_train_views": sum(view.frontier == 0 for view in prepared),
        "latest_evaluation": latest,
        "learned_background": (
            torch.sigmoid(background_logit).detach().cpu().tolist()
            if background_logit is not None
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
