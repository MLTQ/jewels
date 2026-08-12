"""Train prompt-conditioned stochastic jewel marks under oracle birth topology."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from sol.birth_mark_flow import BirthMarkFlowModel, project_birth_topology
from sol.multiscale_video_guide import (
    MULTISCALE_GUIDE_FEATURE_DIM,
    video_to_multiscale_cell_tokens,
)
from sol.prompt_embeddings import load_prompt_cache
from sol.prompted_mark_flow_eval import evaluate_prompted_mark_flow
from sol.realizer_render_loss import estimate_target_marks, realizer_render_loss
from sol.streaming_corpus import (
    PromptedContinuationCorpus,
    build_prompted_continuation_corpus,
    load_prompted_fields,
)
from sol.streaming_data import rasterize_context
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


@dataclass(frozen=True)
class PreparedMarkFlowView:
    context_raster: torch.Tensor
    target_values: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    prompt_indices: tuple[int, ...]
    guide_raster: torch.Tensor | None
    guide_tokens: torch.Tensor | None
    carried_global: torch.Tensor
    background: torch.Tensor
    total_frames: int
    frontier: int
    stride_frames: int
    support_sigma: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--noisy-depth", type=int, default=2)
    parser.add_argument("--guide-depth", type=int, default=2)
    parser.add_argument("--cell-depth", type=int, default=2)
    parser.add_argument("--mark-depth", type=int, default=3)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--prefix-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--text-dropout", type=float, default=0.15)
    parser.add_argument("--context-dropout", type=float, default=0.25)
    parser.add_argument("--oracle-video-guide", action="store_true")
    parser.add_argument("--oracle-multiscale-guide", action="store_true")
    parser.add_argument("--oracle-hybrid-guide", action="store_true")
    parser.add_argument("--guide-height", type=int, default=24)
    parser.add_argument("--guide-width", type=int, default=40)
    parser.add_argument("--guide-scales", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--guide-subgrid", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--guide-heads", type=int, default=8)
    parser.add_argument("--feature-weight", type=float, default=1.0)
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
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _prepare(
    corpus: PromptedContinuationCorpus,
    device: torch.device,
    guide_rasters: dict[tuple[str, int], torch.Tensor] | None = None,
    guide_tokens: dict[tuple[str, int], torch.Tensor] | None = None,
    backgrounds: dict[str, torch.Tensor] | None = None,
) -> list[PreparedMarkFlowView]:
    prepared = []
    for example in corpus.train:
        for view in example.dataset.views:
            if not len(view.births.values):
                continue
            prepared.append(
                PreparedMarkFlowView(
                    context_raster=rasterize_context(
                        view.context_features,
                        corpus.context_standardizer,
                        prefix_frames=example.dataset.prefix_frames,
                        stride_frames=example.dataset.stride_frames,
                        grid_shape=example.dataset.grid_spec.shape,
                    ).to(device),
                    target_values=corpus.birth_standardizer.normalize(
                        view.births.values
                    ).to(device),
                    cell_indices=view.births.cell_indices.to(device),
                    slot_indices=view.births.slot_indices.to(device),
                    prompt_indices=example.train_prompt_indices,
                    guide_raster=(
                        guide_rasters[(example.source_id, view.index)].to(device)
                        if guide_rasters is not None
                        else None
                    ),
                    guide_tokens=(
                        guide_tokens[(example.source_id, view.index)].to(device)
                        if guide_tokens is not None
                        else None
                    ),
                    carried_global=view.carried_global_features.to(device),
                    background=(
                        backgrounds[example.source_id].to(device)
                        if backgrounds is not None
                        else torch.zeros(3, device=device)
                    ),
                    total_frames=example.dataset.total_frames,
                    frontier=view.frontier,
                    stride_frames=example.dataset.stride_frames,
                    support_sigma=example.dataset.support_sigma,
                )
            )
    if not prepared:
        raise ValueError("training corpus contains no birth-bearing views")
    return prepared


def _load_oracle_guides(
    corpus: PromptedContinuationCorpus,
    manifest: dict,
    spec: GridSpec,
    *,
    height: int,
    width: int,
) -> dict[tuple[str, int], torch.Tensor]:
    if min(height, width) <= 0:
        raise ValueError("guide dimensions must be positive")
    sources = {item["source_id"]: item for item in manifest["examples"]}
    guides = {}
    for example in corpus.examples:
        source = sources[example.source_id]
        video = load_video(
            source["video"],
            max_frames=example.dataset.total_frames,
            start_frame=int(source.get("start_frame", 0)),
            resize=(height, width),
            device="cpu",
        )
        if len(video) != example.dataset.total_frames:
            raise ValueError(f"video length disagrees with fitted field: {example.source_id}")
        for view in example.dataset.views:
            guides[(example.source_id, view.index)] = video_to_cell_raster(
                video[view.frontier : view.commit_stop], spec
            )
    return guides


def _load_backgrounds(
    manifest: dict, checkpoint_roots: list[str]
) -> dict[str, torch.Tensor]:
    checkpoints = {}
    for root in checkpoint_roots:
        for path in Path(root).glob("*_w000000.pt"):
            if path.name in checkpoints:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            checkpoints[path.name] = path
    backgrounds = {}
    for example in manifest["examples"]:
        name = f"{Path(example['video']).stem}_w000000.pt"
        if name not in checkpoints:
            raise FileNotFoundError(f"missing fitted checkpoint: {name}")
        saved = torch.load(checkpoints[name], map_location="cpu", weights_only=False)
        backgrounds[example["source_id"]] = torch.as_tensor(
            saved["info"]["background"], dtype=torch.float32
        ).clone()
    return backgrounds


def _load_multiscale_guides(
    corpus: PromptedContinuationCorpus,
    manifest: dict,
    spec: GridSpec,
    *,
    height: int,
    width: int,
    scales: tuple[int, ...],
    subgrid: tuple[int, int, int],
) -> dict[tuple[str, int], torch.Tensor]:
    if min(height, width) <= 0:
        raise ValueError("guide dimensions must be positive")
    sources = {item["source_id"]: item for item in manifest["examples"]}
    guides = {}
    for example in corpus.examples:
        source = sources[example.source_id]
        video = load_video(
            source["video"],
            max_frames=example.dataset.total_frames,
            start_frame=int(source.get("start_frame", 0)),
            resize=(height, width),
            device="cpu",
        )
        if len(video) != example.dataset.total_frames:
            raise ValueError(f"video length disagrees with fitted field: {example.source_id}")
        for view in example.dataset.views:
            guides[(example.source_id, view.index)] = video_to_multiscale_cell_tokens(
                video[view.frontier : view.commit_stop],
                spec,
                scales=scales,
                subgrid=subgrid,
            )
    return guides


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
    if not 0 <= args.text_dropout <= 1 or not 0 <= args.context_dropout <= 1:
        raise ValueError("condition dropout probabilities must be in [0,1]")
    guide_modes = (
        args.oracle_video_guide,
        args.oracle_multiscale_guide,
        args.oracle_hybrid_guide,
    )
    if sum(guide_modes) > 1:
        raise ValueError("select at most one oracle guide mode")
    if args.feature_weight <= 0 or args.render_weight < 0 or args.render_every <= 0:
        raise ValueError("feature/render weights and render cadence are invalid")
    render_components = (
        args.render_rgb_weight,
        args.render_edge_weight,
        args.render_chroma_weight,
        args.render_structure_weight,
    )
    if any(weight < 0 for weight in render_components) or not any(render_components):
        raise ValueError("render component weights must be non-negative and not all zero")
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
    corpus = build_prompted_continuation_corpus(
        fields,
        prompt_cache.embeddings,
        prefix_frames=args.prefix_frames,
        stride_frames=args.stride_frames,
        support_sigma=args.support_sigma,
        grid_spec=spec,
    )
    guide_rasters = (
        _load_oracle_guides(
            corpus,
            manifest,
            spec,
            height=args.guide_height,
            width=args.guide_width,
        )
        if args.oracle_video_guide or args.oracle_hybrid_guide
        else None
    )
    guide_tokens = (
        _load_multiscale_guides(
            corpus,
            manifest,
            spec,
            height=args.guide_height,
            width=args.guide_width,
            scales=tuple(args.guide_scales),
            subgrid=tuple(args.guide_subgrid),
        )
        if args.oracle_multiscale_guide or args.oracle_hybrid_guide
        else None
    )
    backgrounds = _load_backgrounds(manifest, args.checkpoint_root)
    prepared = _prepare(
        corpus, device, guide_rasters, guide_tokens, backgrounds
    )
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
        "guide_dim": (
            3 if args.oracle_video_guide or args.oracle_hybrid_guide else 0
        ),
        "guide_token_dim": (
            MULTISCALE_GUIDE_FEATURE_DIM
            if args.oracle_multiscale_guide or args.oracle_hybrid_guide
            else 0
        ),
        "guide_heads": args.guide_heads,
    }
    model = BirthMarkFlowModel(grid_spec=spec, **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "prompted_mark_flow.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    if args.resume and model_path.exists():
        saved = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])

    def save(step: int, evaluation: dict | None) -> None:
        _atomic_save(
            model_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "prompted_birth_mark_flow_v1",
                    "model_args": model_args,
                    "grid_shape": spec.shape,
                    "slots_per_cell": spec.slots_per_cell,
                    "manifest": args.manifest,
                    "manifest_sha256": prompt_cache.manifest_sha256,
                    "prompt_encoder": prompt_cache.encoder,
                    "training_sources": [example.source_id for example in corpus.train],
                    "validation_sources": [
                        example.source_id for example in corpus.validation
                    ],
                    "context_standardizer": corpus.context_standardizer.state_dict(),
                    "birth_standardizer": corpus.birth_standardizer.state_dict(),
                    "train_args": vars(args),
                    "latest_evaluation": evaluation,
                },
            },
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train_sources={len(corpus.train)} validation_sources={len(corpus.validation)} "
        f"train_views={len(prepared)} model={parameters / 1e6:.2f}M "
        f"text_dim={model.text_dim} amp={use_amp}",
        flush=True,
    )
    history = []
    latest = None
    interval_started = time.time()
    started = interval_started
    for step in range(start_step + 1, args.steps + 1):
        prepared_view = prepared[(step - 1) % len(prepared)]
        prompt_offset = ((step - 1) // len(prepared)) % len(
            prepared_view.prompt_indices
        )
        text = corpus.prompt_embeddings[
            prepared_view.prompt_indices[prompt_offset]
        ].to(device)
        drop_text = torch.rand(1, device=device, generator=generator) < args.text_dropout
        drop_context = bool(
            torch.rand(1, device=device, generator=generator).item()
            < args.context_dropout
        )
        context = (
            torch.zeros_like(prepared_view.context_raster)
            if drop_context
            else prepared_view.context_raster
        )
        noise = torch.randn(
            prepared_view.target_values.shape,
            device=device,
            generator=generator,
        )
        flow_time = torch.rand(1, device=device, generator=generator)
        if step <= args.warmup:
            learning_rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            learning_rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            noised = (1 - flow_time) * noise + flow_time * prepared_view.target_values
            expected_velocity = prepared_view.target_values - noise
            predicted_velocity = model(
                context,
                noised,
                flow_time,
                prepared_view.cell_indices,
                prepared_view.slot_indices,
                text,
                drop_condition=drop_text,
                guide_raster=prepared_view.guide_raster,
                guide_tokens=prepared_view.guide_tokens,
            )
            feature_loss = F.mse_loss(
                predicted_velocity.float(), expected_velocity.float()
            )
        render_terms = None
        loss = args.feature_weight * feature_loss
        if args.render_weight and step % args.render_every == 0:
            estimated = estimate_target_marks(
                noised.float(), predicted_velocity.float(), flow_time.float()
            )
            estimated_local = corpus.birth_standardizer.denormalize(estimated)
            estimated_local = project_birth_topology(
                estimated_local,
                prepared_view.cell_indices,
                spec=spec,
                support_sigma=prepared_view.support_sigma,
                stride_frames=prepared_view.stride_frames,
            )
            target_local = corpus.birth_standardizer.denormalize(
                prepared_view.target_values.float()
            )
            render_terms = realizer_render_loss(
                estimated_local,
                target_local,
                prepared_view.carried_global,
                total_frames=prepared_view.total_frames,
                frontier=prepared_view.frontier,
                stride_frames=prepared_view.stride_frames,
                background=prepared_view.background,
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
                generator=render_generator,
            )
            loss = loss + args.render_weight * args.render_every * render_terms.total
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                "render_edge": (
                    float(render_terms.edge.detach()) if render_terms is not None else None
                ),
                "render_chroma": (
                    float(render_terms.chroma.detach()) if render_terms is not None else None
                ),
                "render_structure": (
                    float(render_terms.structure.detach())
                    if render_terms is not None
                    else None
                ),
            }
        )
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            recent = history[-args.log_every :]
            render_recent = [item for item in recent if item["render_loss"] is not None]
            record = {
                "step": step,
                "loss": sum(item["loss"] for item in recent) / len(recent),
                "feature_loss": sum(item["feature_loss"] for item in recent)
                / len(recent),
                "render_updates": len(render_recent),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / len(recent),
            }
            for name in (
                "render_loss",
                "render_rgb",
                "render_edge",
                "render_chroma",
                "render_structure",
            ):
                record[name] = (
                    sum(float(item[name]) for item in render_recent) / len(render_recent)
                    if render_recent
                    else None
                )
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate_prompted_mark_flow(
                model,
                corpus,
                device=device,
                seed=args.seed,
                guide_rasters=guide_rasters,
                guide_tokens=guide_tokens,
            ).to_dict()
            record = {"step": step, "evaluation": latest}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "train_views": len(prepared),
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
