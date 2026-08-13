"""Train a compact scaffold-gated RGB adapter over a frozen mark flow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, project_birth_topology
from sol.prompt_embeddings import load_prompt_cache
from sol.realizer_render_loss import (
    estimate_target_marks,
    realizer_render_loss,
    scaffold_saliency_weights,
)
from sol.scaffold_appearance_adapter import (
    ScaffoldAppearanceAdapter,
    appearance_feature_loss,
    apply_scaffold_rgb_residual,
    top_fraction_cell_gate,
)
from sol.scaffold_mark_data import (
    ScaffoldMarkCorpus,
    ScaffoldMarkSource,
    build_scaffold_mark_corpus,
    rasterize_scaffold_context,
)
from sol.streaming_corpus import load_prompted_fields
from sol.streaming_data import FeatureStandardizer
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


@dataclass(frozen=True)
class PreparedAppearanceView:
    """One adapter training or evaluation row resident on the selected device."""

    source_id: str
    index: int
    frontier: int
    context_raster: torch.Tensor
    target_values: torch.Tensor
    target_local: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    guide_raster: torch.Tensor
    prompt_indices: tuple[int, ...]
    carried_global: torch.Tensor
    fitted_background: torch.Tensor
    cell_gate: torch.Tensor
    cell_saliency: torch.Tensor
    total_frames: int
    stride_frames: int
    support_sigma: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-flow", required=True)
    parser.add_argument(
        "--teacher-flow",
        help="optional compatible full appearance flow to distill on gated RGB",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-fraction", type=float, default=0.20)
    parser.add_argument("--text-dropout", type=float, default=0.10)
    parser.add_argument("--context-dropout", type=float, default=0.10)
    parser.add_argument("--guide-dropout", type=float, default=0.10)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--render-weight", type=float, default=0.02)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--render-height", type=int, default=192)
    parser.add_argument("--render-width", type=int, default=288)
    parser.add_argument("--render-patches", type=int, default=2)
    parser.add_argument("--render-patch-frames", type=int, default=4)
    parser.add_argument("--render-patch-height", type=int, default=12)
    parser.add_argument("--render-patch-width", type=int, default=12)
    parser.add_argument("--render-rgb-weight", type=float, default=1.0)
    parser.add_argument("--render-edge-weight", type=float, default=0.5)
    parser.add_argument("--render-chroma-weight", type=float, default=0.25)
    parser.add_argument("--render-structure-weight", type=float, default=0.25)
    parser.add_argument("--render-saliency-weight", type=float, default=1.0)
    parser.add_argument("--render-motion-weight", type=float, default=0.5)
    parser.add_argument("--render-stability-weight", type=float, default=1.0)
    parser.add_argument("--render-saliency-fraction", type=float, default=0.5)
    parser.add_argument("--initial-repeat", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_guides(
    corpus: ScaffoldMarkCorpus,
    manifest: dict,
    *,
    height: int,
    width: int,
) -> dict[tuple[str, int], torch.Tensor]:
    """Load every stride through a native-aspect high-resolution intermediate."""
    if min(height, width) <= 0 or width * 2 != height * 3:
        raise ValueError("appearance adapter render resolution must preserve 3:2 aspect")
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


def _load_fitted_backgrounds(
    manifest: dict, checkpoint_roots: list[str]
) -> dict[str, torch.Tensor]:
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


def _causal_backgrounds(
    corpus: ScaffoldMarkCorpus,
    guides: dict[tuple[str, int], torch.Tensor],
) -> dict[str, torch.Tensor]:
    backgrounds = {}
    for source in corpus.sources:
        initial = [view for view in source.views if view.frontier == 0]
        if len(initial) != 1:
            raise ValueError("every source requires exactly one initial scaffold view")
        backgrounds[source.field.source_id] = guides[
            (source.field.source_id, initial[0].index)
        ].float().mean(dim=0)
    return backgrounds


def _prepare(
    sources: tuple[ScaffoldMarkSource, ...],
    corpus: ScaffoldMarkCorpus,
    guides: dict[tuple[str, int], torch.Tensor],
    fitted_backgrounds: dict[str, torch.Tensor],
    causal_backgrounds: dict[str, torch.Tensor],
    device: torch.device,
    *,
    gate_fraction: float,
    evaluation: bool,
) -> list[PreparedAppearanceView]:
    prepared = []
    for source in sources:
        prompt_indices = (
            source.field.evaluation_prompt_indices
            if evaluation
            else source.field.train_prompt_indices
        )
        for view in source.views:
            if not len(view.births.values):
                continue
            key = (source.field.source_id, view.index)
            guide = guides[key]
            saliency = scaffold_saliency_weights(
                guide,
                corpus.grid_spec.shape,
                causal_backgrounds[source.field.source_id],
            )
            saliency = saliency / saliency.mean().clamp_min(1e-4)
            gate = top_fraction_cell_gate(saliency, gate_fraction)
            prepared.append(
                PreparedAppearanceView(
                    source_id=source.field.source_id,
                    index=view.index,
                    frontier=view.frontier,
                    context_raster=rasterize_scaffold_context(
                        view.context_features,
                        corpus.context_standardizer,
                        stride_frames=corpus.stride_frames,
                        grid_spec=corpus.grid_spec,
                    ).to(device),
                    target_values=corpus.birth_standardizer.normalize(
                        view.births.values
                    ).to(device),
                    target_local=view.births.values.to(device),
                    cell_indices=view.births.cell_indices.to(device),
                    slot_indices=view.births.slot_indices.to(device),
                    guide_raster=guide.to(device),
                    prompt_indices=prompt_indices,
                    carried_global=view.carried_global_features.to(device),
                    fitted_background=fitted_backgrounds[
                        source.field.source_id
                    ].to(device),
                    cell_gate=gate.to(device),
                    cell_saliency=saliency.to(device),
                    total_frames=source.field.frames,
                    stride_frames=corpus.stride_frames,
                    support_sigma=corpus.support_sigma,
                )
            )
    if not prepared:
        raise ValueError("appearance adapter preparation produced no birth views")
    return prepared


@torch.no_grad()
def _evaluate_feature_control(
    base_flow: BirthMarkFlowModel,
    adapter: ScaffoldAppearanceAdapter,
    teacher_flow: BirthMarkFlowModel | None,
    views: list[PreparedAppearanceView],
    prompt_embeddings: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Compare deterministic base/adapted RGB velocity error on held-out views."""
    generator = torch.Generator(device=device).manual_seed(seed + 100_003)
    base_errors = []
    adapted_errors = []
    teacher_errors = []
    distillation_errors = []
    was_training = adapter.training
    adapter.eval()
    for view in views:
        noise = torch.randn(
            view.target_values.shape, device=device, generator=generator
        )
        flow_time = torch.tensor([0.5], device=device)
        noised = 0.5 * noise + 0.5 * view.target_values
        expected = view.target_values - noise
        text = prompt_embeddings[view.prompt_indices[0]].to(device)
        base_velocity = base_flow(
            view.context_raster,
            noised,
            flow_time,
            view.cell_indices,
            view.slot_indices,
            text,
            guide_raster=view.guide_raster,
        )
        residual = adapter(
            view.context_raster,
            noised,
            base_velocity,
            flow_time,
            view.cell_indices,
            view.slot_indices,
            text,
            guide_raster=view.guide_raster,
        )
        adapted = apply_scaffold_rgb_residual(
            base_velocity,
            residual,
            view.cell_indices,
            view.cell_gate,
        )
        base_errors.append(
            appearance_feature_loss(
                base_velocity,
                expected,
                view.cell_indices,
                view.cell_gate,
                view.cell_saliency,
            )
        )
        adapted_errors.append(
            appearance_feature_loss(
                adapted,
                expected,
                view.cell_indices,
                view.cell_gate,
                view.cell_saliency,
            )
        )
        if teacher_flow is not None:
            teacher_velocity = teacher_flow(
                view.context_raster,
                noised,
                flow_time,
                view.cell_indices,
                view.slot_indices,
                text,
                guide_raster=view.guide_raster,
            )
            teacher_control = apply_scaffold_rgb_residual(
                base_velocity,
                teacher_velocity[:, 9:12] - base_velocity[:, 9:12],
                view.cell_indices,
                view.cell_gate,
            )
            teacher_errors.append(
                appearance_feature_loss(
                    teacher_control,
                    expected,
                    view.cell_indices,
                    view.cell_gate,
                    view.cell_saliency,
                )
            )
            distillation_errors.append(
                appearance_feature_loss(
                    adapted,
                    teacher_control,
                    view.cell_indices,
                    view.cell_gate,
                    view.cell_saliency,
                )
            )
    if was_training:
        adapter.train()
    base_mse = float(torch.stack(base_errors).mean())
    adapted_mse = float(torch.stack(adapted_errors).mean())
    result = {
        "base_gated_rgb_mse": base_mse,
        "adapted_gated_rgb_mse": adapted_mse,
        "relative_improvement": (base_mse - adapted_mse) / max(base_mse, 1e-12),
        "views": len(views),
    }
    if teacher_errors:
        teacher_mse = float(torch.stack(teacher_errors).mean())
        result.update(
            {
                "teacher_gated_rgb_mse": teacher_mse,
                "teacher_relative_improvement": (base_mse - teacher_mse)
                / max(base_mse, 1e-12),
                "adapter_to_teacher_gated_rgb_mse": float(
                    torch.stack(distillation_errors).mean()
                ),
            }
        )
    return result


def _atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.lr <= 0 or args.warmup < 0:
        raise ValueError("training schedule is outside its valid range")
    if args.model_dim <= 0 or args.depth <= 0 or args.initial_repeat <= 0:
        raise ValueError("adapter architecture/repeat values must be positive")
    if not 0 < args.gate_fraction <= 1:
        raise ValueError("gate fraction must lie inside (0,1]")
    if args.feature_weight <= 0 or args.render_weight < 0 or args.render_every <= 0:
        raise ValueError("feature/render loss configuration is invalid")
    if args.render_width * 2 != args.render_height * 3:
        raise ValueError("render dimensions must preserve the source 3:2 aspect")
    if min(
        args.render_height,
        args.render_width,
        args.render_patches,
        args.render_patch_frames,
        args.render_patch_height,
        args.render_patch_width,
    ) <= 0:
        raise ValueError("render and patch dimensions must be positive")
    if not 0 <= args.render_saliency_fraction <= 1:
        raise ValueError("render saliency fraction must lie inside [0,1]")
    dropouts = (args.text_dropout, args.context_dropout, args.guide_dropout)
    if any(value < 0 or value > 1 for value in dropouts):
        raise ValueError("conditioning dropout probabilities must lie inside [0,1]")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    render_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    base_saved = torch.load(args.base_flow, map_location="cpu", weights_only=False)
    base_meta = base_saved["meta"]
    if base_meta.get("architecture") != "scaffold_birth_mark_flow_v1":
        raise ValueError("base checkpoint is not a scaffold birth mark flow")
    if base_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
        raise ValueError("base flow and prompt manifest differ")
    spec = GridSpec(
        tuple(base_meta["grid_shape"]), int(base_meta["slots_per_cell"])
    )
    base_flow = BirthMarkFlowModel(
        grid_spec=spec, **base_meta["model_args"]
    ).to(device)
    base_flow.load_state_dict(base_saved["model"])
    base_flow.eval()
    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    teacher_flow = None
    teacher_sha256 = None
    if args.teacher_flow:
        teacher_saved = torch.load(
            args.teacher_flow, map_location="cpu", weights_only=False
        )
        teacher_meta = teacher_saved["meta"]
        if teacher_meta.get("architecture") != "scaffold_birth_mark_flow_v1":
            raise ValueError("teacher checkpoint is not a scaffold birth mark flow")
        if teacher_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
            raise ValueError("teacher flow and prompt manifest differ")
        teacher_spec = GridSpec(
            tuple(teacher_meta["grid_shape"]),
            int(teacher_meta["slots_per_cell"]),
        )
        if teacher_spec != spec or teacher_meta["model_args"] != base_meta["model_args"]:
            raise ValueError("teacher flow is architecturally incompatible with the base")
        for name in ("context_standardizer", "birth_standardizer"):
            for statistic in ("mean", "std"):
                if not torch.equal(
                    teacher_meta[name][statistic], base_meta[name][statistic]
                ):
                    raise ValueError(f"teacher and base disagree on {name}")
        teacher_flow = BirthMarkFlowModel(
            grid_spec=teacher_spec, **teacher_meta["model_args"]
        ).to(device)
        teacher_flow.load_state_dict(teacher_saved["model"])
        teacher_flow.eval()
        for parameter in teacher_flow.parameters():
            parameter.requires_grad_(False)
        teacher_sha256 = _sha256(args.teacher_flow)
    base_args = base_meta["train_args"]
    corpus = build_scaffold_mark_corpus(
        fields,
        prompt_cache.embeddings,
        stride_frames=int(base_args["stride_frames"]),
        support_sigma=float(base_args["support_sigma"]),
        grid_spec=spec,
    )
    for actual, name in (
        (corpus.context_standardizer, "context_standardizer"),
        (corpus.birth_standardizer, "birth_standardizer"),
    ):
        saved = FeatureStandardizer.from_state_dict(base_meta[name])
        if not torch.equal(actual.mean, saved.mean) or not torch.equal(
            actual.std, saved.std
        ):
            raise ValueError(f"reconstructed {name} differs from the base flow")
    guides = _load_guides(
        corpus,
        manifest,
        height=args.render_height,
        width=args.render_width,
    )
    fitted_backgrounds = _load_fitted_backgrounds(
        manifest, args.checkpoint_root
    )
    causal_backgrounds = _causal_backgrounds(corpus, guides)
    training_views = _prepare(
        corpus.train,
        corpus,
        guides,
        fitted_backgrounds,
        causal_backgrounds,
        device,
        gate_fraction=args.gate_fraction,
        evaluation=False,
    )
    validation_views = _prepare(
        corpus.validation,
        corpus,
        guides,
        fitted_backgrounds,
        causal_backgrounds,
        device,
        gate_fraction=args.gate_fraction,
        evaluation=True,
    )
    training_schedule = [
        view
        for view in training_views
        for _ in range(args.initial_repeat if view.frontier == 0 else 1)
    ]
    adapter_args = {
        "feature_dim": 22,
        "context_dim": 46,
        "guide_dim": int(base_meta["model_args"]["guide_dim"]),
        "text_dim": int(prompt_cache.embeddings.shape[1]),
        "model_dim": args.model_dim,
        "depth": args.depth,
    }
    adapter = ScaffoldAppearanceAdapter(grid_spec=spec, **adapter_args).to(device)
    adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    base_parameters = sum(parameter.numel() for parameter in base_flow.parameters())
    if adapter_parameters >= base_parameters:
        raise ValueError("appearance adapter is not smaller than its frozen base")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "appearance_adapter.pt"
    log_path = output_dir / "train_log.jsonl"
    base_sha256 = _sha256(args.base_flow)
    start_step = 0
    latest_evaluation = None
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["meta"].get("base_flow_sha256") != base_sha256:
            raise ValueError("resume checkpoint was trained over a different base flow")
        adapter.load_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        latest_evaluation = saved["meta"].get("latest_evaluation")

    def save(step: int) -> None:
        state = {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "meta": {
                "architecture": "scaffold_appearance_adapter_v1",
                "adapter_args": adapter_args,
                "adapter_parameters": adapter_parameters,
                "base_parameters": base_parameters,
                "grid_shape": spec.shape,
                "slots_per_cell": spec.slots_per_cell,
                "base_flow_checkpoint": args.base_flow,
                "base_flow_sha256": base_sha256,
                "teacher_flow_checkpoint": args.teacher_flow,
                "teacher_flow_sha256": teacher_sha256,
                "feature_objective": (
                    "gated_teacher_rgb_velocity_distillation"
                    if teacher_flow is not None
                    else "gated_ground_truth_rgb_velocity"
                ),
                "manifest": args.manifest,
                "manifest_sha256": prompt_cache.manifest_sha256,
                "prompt_encoder": prompt_cache.encoder,
                "training_sources": [
                    source.field.source_id for source in corpus.train
                ],
                "validation_sources": [
                    source.field.source_id for source in corpus.validation
                ],
                "context_standardizer": corpus.context_standardizer.state_dict(),
                "birth_standardizer": corpus.birth_standardizer.state_dict(),
                "mutable_dimensions": [9, 10, 11],
                "gate_fraction": args.gate_fraction,
                "guide_resolution": [args.render_height, args.render_width],
                "render_coordinate_resolution": [
                    args.render_height,
                    args.render_width,
                ],
                "background_contract": "initial_scaffold_rgb_mean",
                "render_target_contract": "fitted_carry_and_background_only",
                "train_args": vars(args),
                "latest_evaluation": latest_evaluation,
            },
        }
        _atomic_save(checkpoint_path, state)
        _atomic_save(output_dir / f"appearance_adapter_step{step:06d}.pt", state)

    print(
        f"train_views={len(training_views)} validation_views={len(validation_views)} "
        f"adapter={adapter_parameters / 1e3:.1f}K base={base_parameters / 1e6:.2f}M "
        f"ratio={adapter_parameters / base_parameters:.4f} "
        f"render={args.render_width}x{args.render_height} fp32=True",
        flush=True,
    )
    history = []
    interval_started = time.time()
    started = interval_started
    adapter.train()
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
        adapter_context = (
            torch.zeros_like(view.context_raster)
            if drop_context
            else view.context_raster
        )
        adapter_guide = (
            torch.zeros_like(view.guide_raster) if drop_guide else view.guide_raster
        )
        noise = torch.randn(view.target_values.shape, device=device, generator=generator)
        flow_time = torch.rand(1, device=device, generator=generator)
        noised = (1 - flow_time) * noise + flow_time * view.target_values
        expected_velocity = view.target_values - noise
        with torch.no_grad():
            base_velocity = base_flow(
                view.context_raster,
                noised,
                flow_time,
                view.cell_indices,
                view.slot_indices,
                text,
                guide_raster=view.guide_raster,
            )
            teacher_velocity = (
                teacher_flow(
                    view.context_raster,
                    noised,
                    flow_time,
                    view.cell_indices,
                    view.slot_indices,
                    text,
                    guide_raster=view.guide_raster,
                )
                if teacher_flow is not None
                else None
            )
        residual = adapter(
            adapter_context,
            noised,
            base_velocity,
            flow_time,
            view.cell_indices,
            view.slot_indices,
            text,
            drop_condition=drop_text,
            guide_raster=adapter_guide,
        )
        corrected_velocity = apply_scaffold_rgb_residual(
            base_velocity,
            residual,
            view.cell_indices,
            view.cell_gate,
        )
        feature_loss = appearance_feature_loss(
            corrected_velocity,
            teacher_velocity if teacher_velocity is not None else expected_velocity,
            view.cell_indices,
            view.cell_gate,
            view.cell_saliency,
        )
        loss = args.feature_weight * feature_loss
        render_terms = None
        render_update = bool(args.render_weight and step % args.render_every == 0)
        if render_update:
            estimated = estimate_target_marks(
                noised.float(), corrected_velocity.float(), flow_time.float()
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
            render_terms = realizer_render_loss(
                estimated_local,
                view.target_local,
                view.carried_global,
                total_frames=view.total_frames,
                frontier=view.frontier,
                stride_frames=view.stride_frames,
                background=view.fitted_background,
                render_height=args.render_height,
                render_width=args.render_width,
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
                anchor_frontier=True,
                generator=render_generator,
            )
            loss = loss + args.render_weight * args.render_every * render_terms.total
        if step <= args.warmup:
            learning_rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            learning_rate = args.lr * (
                0.1 + 0.45 * (1 + math.cos(math.pi * progress))
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "loss": float(loss.detach()),
                "feature_loss": float(feature_loss.detach()),
                "render_loss": (
                    float(render_terms.total.detach()) if render_terms else None
                ),
                "render_rgb": float(render_terms.rgb.detach()) if render_terms else None,
                "render_saliency": (
                    float(render_terms.saliency_rgb.detach()) if render_terms else None
                ),
                "render_motion": (
                    float(render_terms.motion.detach()) if render_terms else None
                ),
                "render_stability": (
                    float(render_terms.stability.detach()) if render_terms else None
                ),
            }
        )
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            recent = history[-args.log_every :]
            rendered = [row for row in recent if row["render_loss"] is not None]
            record = {
                "step": step,
                "loss": sum(row["loss"] for row in recent) / len(recent),
                "feature_loss": sum(row["feature_loss"] for row in recent)
                / len(recent),
                "render_loss": (
                    sum(float(row["render_loss"]) for row in rendered) / len(rendered)
                    if rendered
                    else None
                ),
                "render_updates": len(rendered),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / len(recent),
            }
            for name in (
                "render_rgb",
                "render_saliency",
                "render_motion",
                "render_stability",
            ):
                record[name] = (
                    sum(float(row[name]) for row in rendered) / len(rendered)
                    if rendered
                    else None
                )
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest_evaluation = _evaluate_feature_control(
                base_flow,
                adapter,
                teacher_flow,
                validation_views,
                corpus.prompt_embeddings,
                seed=args.seed,
                device=device,
            )
            record = {"step": step, "evaluation": latest_evaluation}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
            adapter.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "adapter_parameters": adapter_parameters,
        "base_parameters": base_parameters,
        "parameter_ratio": adapter_parameters / base_parameters,
        "gate_fraction": args.gate_fraction,
        "render_resolution": [args.render_height, args.render_width],
        "latest_evaluation": latest_evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
