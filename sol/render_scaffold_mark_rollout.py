"""Render and audit autonomous initial-plus-two-stride scaffold mark rollouts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import torch

from sol.audit_prompted_washout import render_signature
from sol.birth_mark_flow import BirthMarkFlowModel
from sol.lifecycle_appearance_rollout import (
    LifecycleAppearanceRollout,
    rollout_lifecycle_appearance_marks,
)
from sol.lifecycle_appearance_flow import APPEARANCE_DIMENSION_SETS
from sol.prompt_embeddings import load_prompt_cache
from sol.realizer_render_loss import scaffold_saliency_weights
from sol.render import render_exact
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.saliency_metrics import saliency_render_signature
from sol.scaffold_appearance_adapter import (
    RGB_DIMENSIONS,
    ScaffoldAppearanceAdapter,
)
from sol.scaffold_appearance_adapter_rollout import (
    AppearanceAdapterRollout,
    rollout_scaffold_appearance_adapter,
)
from sol.scaffold_mark_data import build_scaffold_mark_corpus
from sol.scaffold_mark_rollout import ScaffoldMarkRollout, rollout_scaffold_marks
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.scaffold_topology_eval import topology_metrics
from sol.splat_density import measure_frame_splat_density, summarize_counts
from sol.streaming_corpus import load_prompted_fields
from sol.streaming_data import FeatureStandardizer
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


BASE_PANELS = (
    "LTX scaffold",
    "fitted jewel ceiling",
    "generated correct",
    "generated shuffled",
    "generated null",
)


def _source_seed_map(source_ids: Sequence[str], base_seed: int) -> dict[str, int]:
    """Assign stable seeds before any evaluation-source filtering."""
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("validation source IDs must be unique")
    return {source_id: base_seed + index for index, source_id in enumerate(source_ids)}


def _panel_names(lifecycle_appearance: bool) -> tuple[str, ...]:
    """Insert the matched frozen branch only for two-stream experiments."""
    if not lifecycle_appearance:
        return BASE_PANELS
    return (
        "LTX scaffold",
        "fitted jewel ceiling",
        "generated frozen base",
        "generated correct",
        "generated shuffled",
        "generated null",
    )


def _base_lock_report(
    base_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    *,
    added_prefixes: tuple[str, ...],
) -> dict[str, int | bool]:
    """Require every non-augmentation tensor to remain exact."""
    if not added_prefixes:
        raise ValueError("base-lock validation requires added state prefixes")
    shared = {
        name: value
        for name, value in candidate_state.items()
        if not name.startswith(added_prefixes)
    }
    added = [name for name in candidate_state if name.startswith(added_prefixes)]
    if not added:
        raise ValueError("candidate checkpoint contains no augmented state")
    if shared.keys() != base_state.keys():
        raise ValueError("candidate and paired base have different shared state keys")
    mismatched = [
        name for name, value in shared.items() if not torch.equal(value, base_state[name])
    ]
    if mismatched:
        raise ValueError("candidate modified tensors owned by the paired base")
    return {
        "shared_tensors_exact": True,
        "shared_tensor_count": len(shared),
        "added_tensor_count": len(added),
    }


def _correct_panel_names(paired_appearance: bool) -> tuple[str, ...]:
    if not paired_appearance:
        return ("LTX scaffold", "fitted jewel ceiling", "generated correct")
    return (
        "LTX scaffold",
        "fitted jewel ceiling",
        "generated frozen base",
        "generated correct",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--mark-flow", required=True)
    parser.add_argument(
        "--paired-base-flow",
        help="base mark flow that owns exact counts/ranks for an augmented-model audit",
    )
    parser.add_argument("--appearance-flow")
    parser.add_argument("--appearance-adapter")
    parser.add_argument(
        "--appearance-dimension-set",
        choices=tuple(APPEARANCE_DIMENSION_SETS),
        default="all",
    )
    parser.add_argument("--appearance-saliency-fraction", type=float, default=1.0)
    parser.add_argument("--appearance-strength", type=float, default=1.0)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--strict-initial-boundary", action="store_true")
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="skip shuffled/null rollouts and panels for a matched high-resolution gate",
    )
    return parser.parse_args()


def _configure_determinism(enabled: bool, device: torch.device) -> None:
    """Enable repeatable CUDA kernels before models or generators are created."""
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _fit_lookup(roots: list[str]) -> dict[str, Path]:
    paths = {}
    for root in roots:
        for path in Path(root).glob("*_w000000.pt"):
            if path.name in paths:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            paths[path.name] = path
    return paths


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _appearance_saliency_gates(
    guides: tuple[torch.Tensor, ...],
    grid_shape: tuple[int, int, int],
    background: torch.Tensor,
    fraction: float,
    strength: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Select a fixed fraction of highest-scoring guide cells in every stride."""
    if not 0 < fraction <= 1:
        raise ValueError("appearance saliency fraction must lie inside (0,1]")
    if not 0 <= strength <= 1:
        raise ValueError("appearance strength must lie inside [0,1]")
    gates = []
    for guide in guides:
        scores = scaffold_saliency_weights(guide, grid_shape, background)
        if fraction == 1.0:
            gates.append(torch.full_like(scores, strength))
            continue
        count = max(1, round(len(scores) * fraction))
        selected = scores.topk(count).indices
        gate = torch.zeros_like(scores)
        gate[selected] = strength
        gates.append(gate)
    return tuple(gates)


def _causal_background(initial_video: torch.Tensor) -> torch.Tensor:
    """Derive one persistent RGB background from the first scaffold stride only."""
    if initial_video.ndim != 4 or initial_video.shape[-1] != 3 or not len(initial_video):
        raise ValueError("initial video must have shape (frames,height,width,3)")
    return initial_video.float().mean(dim=(0, 1, 2))


def _seam_report(
    candidate: torch.Tensor, target: torch.Tensor, stride_frames: int
) -> dict[str, float]:
    """Compare boundary changes with ordinary temporal changes and target seams."""
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("candidate and target videos must share (T,H,W,3)")
    if stride_frames <= 0 or len(candidate) <= stride_frames:
        raise ValueError("seam report requires at least two complete strides")
    candidate_change = (candidate[1:] - candidate[:-1]).abs().mean((1, 2, 3))
    target_change = (target[1:] - target[:-1]).abs().mean((1, 2, 3))
    seam_indices = torch.arange(
        stride_frames - 1,
        len(candidate_change),
        stride_frames,
        device=candidate.device,
    )
    nonseam = torch.ones(len(candidate_change), dtype=torch.bool, device=candidate.device)
    nonseam[seam_indices] = False
    candidate_seam = candidate_change[seam_indices].mean()
    target_seam = target_change[seam_indices].mean()
    candidate_regular = candidate_change[nonseam].mean()
    target_regular = target_change[nonseam].mean()
    return {
        "candidate_seam_change": float(candidate_seam),
        "target_seam_change": float(target_seam),
        "candidate_regular_change": float(candidate_regular),
        "target_regular_change": float(target_regular),
        "seam_to_target_ratio": float(candidate_seam / target_seam.clamp_min(1e-8)),
        "seam_to_regular_ratio": float(
            candidate_seam / candidate_regular.clamp_min(1e-8)
        ),
    }


def _density_report(
    features: torch.Tensor,
    total_frames: int,
    completed_frames: int,
    stride_frames: int,
    support_sigma: float,
) -> dict:
    density = measure_frame_splat_density(
        features.cpu(), total_frames, support_sigma=support_sigma
    )
    effective = density.effective_peak_alpha_counts[:completed_frames]
    visible = density.peak_alpha_counts[0.05][:completed_frames]
    frontiers = range(0, completed_frames, stride_frames)
    return {
        "effective": summarize_counts(
            effective
        ),
        "above_5_percent_alpha": summarize_counts(
            visible
        ),
        "frontier_effective": {str(frame): float(effective[frame]) for frame in frontiers},
        "frontier_above_5_percent_alpha": {
            str(frame): int(visible[frame]) for frame in frontiers
        },
        "initial_stride_effective": [float(value) for value in effective[:stride_frames]],
        "initial_stride_above_5_percent_alpha": [
            int(value) for value in visible[:stride_frames]
        ],
    }


def _macro_average(records: list[dict], section: str) -> dict[str, float]:
    if not records:
        raise ValueError("macro average requires source records")
    names = tuple(records[0][section])
    if any(tuple(record[section]) != names for record in records):
        raise ValueError("macro-average sections must have identical keys")
    return {
        name: sum(float(record[section][name]) for record in records) / len(records)
        for name in names
    }


def _render_field(
    features: torch.Tensor,
    points: torch.Tensor,
    background: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        render_exact(features.to(points.device), points, background=background.to(points.device))
        .reshape(frames, height, width, 3)
        .cpu()
    )


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.steps, args.height, args.width, args.upscale) <= 0:
        raise ValueError("sampling and render dimensions must be positive")
    if not 0 < args.appearance_saliency_fraction <= 1:
        raise ValueError("appearance saliency fraction must lie inside (0,1]")
    if not 0 <= args.appearance_strength <= 1:
        raise ValueError("appearance strength must lie inside [0,1]")
    if args.appearance_flow and args.appearance_adapter:
        raise ValueError("appearance flow and compact adapter are mutually exclusive")
    if args.paired_base_flow and (args.appearance_flow or args.appearance_adapter):
        raise ValueError(
            "paired base flow and appearance mechanisms are mutually exclusive"
        )
    device = torch.device(args.device)
    _configure_determinism(args.deterministic, device)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)

    topology_saved = torch.load(args.topology, map_location="cpu", weights_only=False)
    topology_meta = topology_saved["meta"]
    if topology_meta.get("architecture") != "scaffold_topology_v1":
        raise ValueError("checkpoint is not a scaffold topology model")
    if topology_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
        raise ValueError("topology checkpoint does not own this manifest")
    topology_spec = GridSpec(
        tuple(topology_meta["grid_shape"]), int(topology_meta["slots_per_cell"])
    )
    topology = ScaffoldTopologyModel(
        grid_spec=topology_spec, **topology_meta["model_args"]
    ).to(device)
    topology.load_state_dict(topology_saved["model"])
    topology.eval()
    topology_evaluation = topology_meta.get("latest_evaluation")
    if not topology_evaluation:
        raise ValueError("topology checkpoint lacks train-calibrated evaluation")
    occupancy_threshold = float(
        topology_evaluation["validation"]["occupancy_threshold"]
    )

    flow_saved = torch.load(args.mark_flow, map_location="cpu", weights_only=False)
    flow_meta = flow_saved["meta"]
    if flow_meta.get("architecture") != "scaffold_birth_mark_flow_v1":
        raise ValueError("checkpoint is not an initial-compatible scaffold mark flow")
    if flow_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
        raise ValueError("mark checkpoint does not own this manifest")
    flow_spec = GridSpec(
        tuple(flow_meta["grid_shape"]), int(flow_meta["slots_per_cell"])
    )
    if flow_spec.shape != topology_spec.shape or (
        flow_spec.slots_per_cell < topology_spec.slots_per_cell
    ):
        raise ValueError("mark flow cannot realize the topology checkpoint")
    flow = BirthMarkFlowModel(grid_spec=flow_spec, **flow_meta["model_args"]).to(device)
    flow.load_state_dict(flow_saved["model"])
    flow.eval()
    flow_args = flow_meta["train_args"]
    corpus = build_scaffold_mark_corpus(
        fields,
        prompt_cache.embeddings,
        stride_frames=int(flow_args["stride_frames"]),
        support_sigma=float(flow_args["support_sigma"]),
        grid_spec=flow_spec,
    )
    context_standardizer = FeatureStandardizer.from_state_dict(
        flow_meta["context_standardizer"]
    )
    birth_standardizer = FeatureStandardizer.from_state_dict(
        flow_meta["birth_standardizer"]
    )
    for actual, saved in (
        (corpus.context_standardizer, context_standardizer),
        (corpus.birth_standardizer, birth_standardizer),
    ):
        if not torch.equal(actual.mean, saved.mean) or not torch.equal(
            actual.std, saved.std
        ):
            raise ValueError("reconstructed standardizers differ from the mark flow")
    if flow_meta.get("background_contract") != "initial_scaffold_rgb_mean":
        raise ValueError("mark flow declares an unsupported background contract")
    paired_base_flow = None
    paired_base_meta = None
    base_lock = None
    if args.paired_base_flow:
        paired_base_saved = torch.load(
            args.paired_base_flow, map_location="cpu", weights_only=False
        )
        paired_base_meta = paired_base_saved["meta"]
        if paired_base_meta.get("architecture") != "scaffold_birth_mark_flow_v1":
            raise ValueError("paired base checkpoint is not a scaffold mark flow")
        if paired_base_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
            raise ValueError("paired base checkpoint does not own this manifest")
        paired_base_spec = GridSpec(
            tuple(paired_base_meta["grid_shape"]),
            int(paired_base_meta["slots_per_cell"]),
        )
        candidate_base_args = {
            name: value
            for name, value in flow_meta["model_args"].items()
            if name
            not in {
                "set_depth",
                "set_raster_depth",
                "set_coupling",
                "set_atoms",
                "set_max_offset",
            }
        }
        if int(flow_meta["model_args"].get("set_depth", 0)) <= 0:
            raise ValueError("paired base audit requires an augmented candidate")
        if paired_base_spec != flow_spec or (
            paired_base_meta["model_args"] != candidate_base_args
        ):
            raise ValueError("candidate and paired base have incompatible models")
        paired_base_args = paired_base_meta["train_args"]
        for name in ("stride_frames", "support_sigma"):
            if paired_base_args[name] != flow_args[name]:
                raise ValueError(f"candidate and paired base disagree on {name}")
        for name in ("context_standardizer", "birth_standardizer"):
            for statistic in ("mean", "std"):
                if not torch.equal(
                    paired_base_meta[name][statistic], flow_meta[name][statistic]
                ):
                    raise ValueError(
                        f"candidate and paired base disagree on {name}"
                    )
        if paired_base_meta.get("background_contract") != flow_meta.get(
            "background_contract"
        ):
            raise ValueError("candidate and paired base background contracts differ")
        base_lock = _base_lock_report(
            paired_base_saved["model"],
            flow_saved["model"],
            added_prefixes=("set_blocks.",),
        )
        paired_base_flow = BirthMarkFlowModel(
            grid_spec=paired_base_spec, **paired_base_meta["model_args"]
        ).to(device)
        paired_base_flow.load_state_dict(paired_base_saved["model"])
        paired_base_flow.eval()
    appearance_flow = None
    appearance_meta = None
    if args.appearance_flow:
        appearance_saved = torch.load(
            args.appearance_flow, map_location="cpu", weights_only=False
        )
        appearance_meta = appearance_saved["meta"]
        if appearance_meta.get("architecture") != "scaffold_birth_mark_flow_v1":
            raise ValueError("appearance checkpoint is not a scaffold mark flow")
        if appearance_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
            raise ValueError("appearance checkpoint does not own this manifest")
        appearance_spec = GridSpec(
            tuple(appearance_meta["grid_shape"]),
            int(appearance_meta["slots_per_cell"]),
        )
        if appearance_spec != flow_spec or (
            appearance_meta["model_args"] != flow_meta["model_args"]
        ):
            raise ValueError("base and appearance checkpoints have different models")
        appearance_args = appearance_meta["train_args"]
        for name in ("stride_frames", "support_sigma"):
            if appearance_args[name] != flow_args[name]:
                raise ValueError(
                    f"base and appearance checkpoints disagree on {name}"
                )
        for name in ("context_standardizer", "birth_standardizer"):
            for statistic in ("mean", "std"):
                if not torch.equal(
                    appearance_meta[name][statistic], flow_meta[name][statistic]
                ):
                    raise ValueError(
                        f"base and appearance checkpoints disagree on {name}"
                    )
        if appearance_meta.get("background_contract") != flow_meta.get(
            "background_contract"
        ):
            raise ValueError("base and appearance background contracts differ")
        appearance_flow = BirthMarkFlowModel(
            grid_spec=appearance_spec, **appearance_meta["model_args"]
        ).to(device)
        appearance_flow.load_state_dict(appearance_saved["model"])
        appearance_flow.eval()
    appearance_adapter = None
    adapter_meta = None
    if args.appearance_adapter:
        adapter_saved = torch.load(
            args.appearance_adapter, map_location="cpu", weights_only=False
        )
        adapter_meta = adapter_saved["meta"]
        if adapter_meta.get("architecture") != "scaffold_appearance_adapter_v1":
            raise ValueError("appearance checkpoint is not a compact RGB adapter")
        if adapter_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
            raise ValueError("appearance adapter does not own this manifest")
        if adapter_meta.get("base_flow_sha256") != _sha256(args.mark_flow):
            raise ValueError("appearance adapter was trained over a different base flow")
        adapter_spec = GridSpec(
            tuple(adapter_meta["grid_shape"]),
            int(adapter_meta["slots_per_cell"]),
        )
        if adapter_spec != flow_spec:
            raise ValueError("base flow and appearance adapter use different grids")
        if tuple(adapter_meta.get("mutable_dimensions", ())) != RGB_DIMENSIONS:
            raise ValueError("appearance adapter is not restricted to canonical RGB")
        for name in ("context_standardizer", "birth_standardizer"):
            for statistic in ("mean", "std"):
                if not torch.equal(
                    adapter_meta[name][statistic], flow_meta[name][statistic]
                ):
                    raise ValueError(
                        f"base flow and appearance adapter disagree on {name}"
                    )
        if abs(
            float(adapter_meta["gate_fraction"])
            - args.appearance_saliency_fraction
        ) > 1e-12:
            raise ValueError(
                "evaluation saliency fraction must match adapter training"
            )
        appearance_adapter = ScaffoldAppearanceAdapter(
            grid_spec=adapter_spec, **adapter_meta["adapter_args"]
        ).to(device)
        appearance_adapter.load_state_dict(adapter_saved["adapter"])
        appearance_adapter.eval()

    validation = sorted(
        corpus.validation,
        key=lambda source: (source.field.class_id, source.field.source_id),
    )
    source_seeds = _source_seed_map(
        [source.field.source_id for source in validation], args.seed
    )
    if len({source.field.class_id for source in validation}) < 2:
        raise ValueError("shuffled rollout requires two held-out classes")
    alternate = {}
    for index, source in enumerate(validation):
        candidate = validation[(index + 1) % len(validation)]
        if candidate.field.class_id == source.field.class_id:
            raise ValueError("shuffled rollout must use another action class")
        alternate[source.field.source_id] = candidate.field.source_id
    source_filter = set(args.source_id)
    available = {source.field.source_id for source in validation}
    if source_filter - available:
        raise ValueError(f"unknown validation sources: {sorted(source_filter - available)}")
    selected_sources = [
        source
        for source in validation
        if not source_filter or source.field.source_id in source_filter
    ]
    manifest_sources = {item["source_id"]: item for item in manifest["examples"]}
    videos = {}
    guides = {}
    for source in validation:
        item = manifest_sources[source.field.source_id]
        video = load_video(
            item["video"],
            max_frames=source.field.frames,
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        )
        if len(video) != source.field.frames:
            raise ValueError(f"video length disagrees with field: {source.field.source_id}")
        videos[source.field.source_id] = video
        guides[source.field.source_id] = tuple(
            video_to_cell_raster(
                video[view.frontier : view.commit_stop], topology_spec
            )
            for view in source.views
        )

    fits = _fit_lookup(args.checkpoint_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    aggregate_rows = []
    controls = (
        ("correct",)
        if args.correct_only
        else ("correct", "shuffled", "null")
    )
    predicted_counts = {name: [] for name in controls}
    target_counts = []
    paired_appearance = appearance_flow is not None or appearance_adapter is not None
    paired_topology = paired_base_flow is not None
    paired_control = paired_appearance or paired_topology
    panels = (
        _correct_panel_names(paired_control)
        if args.correct_only
        else _panel_names(paired_control)
    )
    for source in selected_sources:
        source_id = source.field.source_id
        source_seed = source_seeds[source_id]
        target_video = videos[source_id]
        background = _causal_background(target_video[: corpus.stride_frames])
        source_guides = {
            "correct": guides[source_id],
        }
        if not args.correct_only:
            source_guides.update(
                {
                    "shuffled": guides[alternate[source_id]],
                    "null": tuple(
                        torch.zeros_like(guide) for guide in guides[source_id]
                    ),
                }
            )
        text = corpus.prompt_embeddings[
            source.field.evaluation_prompt_indices[0]
        ].to(device)
        rollouts: dict[str, ScaffoldMarkRollout] = {}
        paired_rollouts: dict[
            str, LifecycleAppearanceRollout | AppearanceAdapterRollout
        ] = {}
        paired_base_rollouts: dict[str, ScaffoldMarkRollout] = {}
        for name, control_guides in source_guides.items():
            generator = torch.Generator(device=device).manual_seed(
                source_seed
            )
            rollout_arguments = {
                "total_frames": source.field.frames,
                "stride_frames": corpus.stride_frames,
                "support_sigma": corpus.support_sigma,
                "topology_spec": topology_spec,
                "occupancy_threshold": occupancy_threshold,
                "device": device,
                "steps": args.steps,
                "generator": generator,
                "allow_initial_prefrontier": not args.strict_initial_boundary,
            }
            if paired_topology:
                if paired_base_flow is None:
                    raise AssertionError("paired topology mode lacks a base flow")
                paired_base_rollouts[name] = rollout_scaffold_marks(
                    topology,
                    paired_base_flow,
                    control_guides,
                    text,
                    context_standardizer,
                    birth_standardizer,
                    **rollout_arguments,
                )
                candidate_arguments = dict(rollout_arguments)
                candidate_arguments["generator"] = torch.Generator(
                    device=device
                ).manual_seed(source_seed)
                candidate_arguments["owned_counts"] = paired_base_rollouts[
                    name
                ].counts
                rollouts[name] = rollout_scaffold_marks(
                    topology,
                    flow,
                    control_guides,
                    text,
                    context_standardizer,
                    birth_standardizer,
                    **candidate_arguments,
                )
            elif not paired_appearance:
                rollouts[name] = rollout_scaffold_marks(
                    topology,
                    flow,
                    control_guides,
                    text,
                    context_standardizer,
                    birth_standardizer,
                    **rollout_arguments,
                )
            elif appearance_flow is not None:
                paired = rollout_lifecycle_appearance_marks(
                    topology,
                    flow,
                    appearance_flow,
                    control_guides,
                    text,
                    context_standardizer,
                    birth_standardizer,
                    appearance_dimensions=APPEARANCE_DIMENSION_SETS[
                        args.appearance_dimension_set
                    ],
                    appearance_cell_weights=_appearance_saliency_gates(
                        tuple(control_guides),
                        topology_spec.shape,
                        background,
                        args.appearance_saliency_fraction,
                        args.appearance_strength,
                    ),
                    **rollout_arguments,
                )
                paired_rollouts[name] = paired
                rollouts[name] = paired.appearance
            else:
                if appearance_adapter is None:
                    raise AssertionError("paired adapter mode lacks an adapter")
                paired = rollout_scaffold_appearance_adapter(
                    topology,
                    flow,
                    appearance_adapter,
                    control_guides,
                    text,
                    context_standardizer,
                    birth_standardizer,
                    appearance_cell_weights=_appearance_saliency_gates(
                        tuple(control_guides),
                        topology_spec.shape,
                        background,
                        args.appearance_saliency_fraction,
                        args.appearance_strength,
                    ),
                    **rollout_arguments,
                )
                paired_rollouts[name] = paired
                rollouts[name] = paired.appearance
            predicted_counts[name].extend(rollouts[name].counts)
        target_counts.extend(view.births.counts for view in source.views)
        completed_frames = rollouts["correct"].completed_frames
        target_video = target_video[:completed_frames]
        item = manifest_sources[source_id]
        fit_name = f"{Path(item['video']).stem}_w000000.pt"
        fit_path = fits.get(fit_name)
        if fit_path is None:
            raise FileNotFoundError(f"missing fitted checkpoint {fit_name}")
        fitted = torch.load(fit_path, map_location="cpu", weights_only=False)
        fitted_background = torch.as_tensor(
            fitted["info"]["background"], dtype=torch.float32
        )
        points = frame_points(
            source.field.frames,
            torch.arange(completed_frames),
            args.height,
            args.width,
            device=device,
        )
        rendered = {
            "LTX scaffold": target_video,
            "fitted jewel ceiling": _render_field(
                source.field.features,
                points,
                fitted_background,
                frames=completed_frames,
                height=args.height,
                width=args.width,
            ),
        }
        if paired_rollouts:
            rendered["generated frozen base"] = _render_field(
                paired_rollouts["correct"].base.features,
                points,
                background,
                frames=completed_frames,
                height=args.height,
                width=args.width,
            )
        elif paired_base_rollouts:
            rendered["generated frozen base"] = _render_field(
                paired_base_rollouts["correct"].features,
                points,
                background,
                frames=completed_frames,
                height=args.height,
                width=args.width,
            )
        rendered_controls = (
            (("correct", "generated correct"),)
            if args.correct_only
            else (
                ("correct", "generated correct"),
                ("shuffled", "generated shuffled"),
                ("null", "generated null"),
            )
        )
        for control, panel in rendered_controls:
            rendered[panel] = _render_field(
                rollouts[control].features,
                points,
                background,
                frames=completed_frames,
                height=args.height,
                width=args.width,
            )
        signatures = {
            panel: asdict(render_signature(frames, target_video))
            for panel, frames in rendered.items()
            if panel != "LTX scaffold"
        }
        seams = {
            panel: _seam_report(frames, target_video, corpus.stride_frames)
            for panel, frames in rendered.items()
            if panel != "LTX scaffold"
        }
        saliency_signatures = {
            panel: asdict(
                saliency_render_signature(
                    frames, target_video, background=fitted_background
                )
            )
            for panel, frames in rendered.items()
            if panel != "LTX scaffold"
        }
        density = {
            "fitted jewel ceiling": _density_report(
                source.field.features,
                source.field.frames,
                completed_frames,
                corpus.stride_frames,
                corpus.support_sigma,
            ),
            **{
                f"generated {name}": _density_report(
                    rollout.features,
                    source.field.frames,
                    completed_frames,
                    corpus.stride_frames,
                    corpus.support_sigma,
                )
                for name, rollout in rollouts.items()
            },
        }
        if paired_rollouts:
            density["generated frozen base"] = _density_report(
                paired_rollouts["correct"].base.features,
                source.field.frames,
                completed_frames,
                corpus.stride_frames,
                corpus.support_sigma,
            )
        elif paired_base_rollouts:
            density["generated frozen base"] = _density_report(
                paired_base_rollouts["correct"].features,
                source.field.frames,
                completed_frames,
                corpus.stride_frames,
                corpus.support_sigma,
            )
        animation = [
            _row([_panel(rendered[name][frame], name, args.upscale) for name in panels])
            for frame in range(completed_frames)
        ]
        gif_name = f"{source_id}_three_window_rollout.gif"
        animation[0].save(
            output_dir / gif_name,
            save_all=True,
            append_images=animation[1:],
            duration=83,
            loop=0,
        )
        picks = sorted(
            {
                0,
                corpus.stride_frames - 1,
                corpus.stride_frames,
                2 * corpus.stride_frames - 1,
                2 * corpus.stride_frames,
                completed_frames - 1,
            }
        )
        contact = Image.new(
            "RGB",
            (animation[0].width, sum(animation[index].height for index in picks)),
            "white",
        )
        offset = 0
        for index in picks:
            contact.paste(animation[index], (0, offset))
            offset += animation[index].height
        contact_name = f"{source_id}_three_window_rollout_contact.png"
        contact.save(output_dir / contact_name)
        aggregate_rows.append(animation[2 * corpus.stride_frames])
        field_name = f"{source_id}_generated_field.pt"
        mark_only_attribution = None
        if paired_base_rollouts:
            mark_only_attribution = {}
            for name in controls:
                base_rollout = paired_base_rollouts[name]
                candidate_rollout = rollouts[name]
                mark_only_attribution[name] = {
                    "counts_exact": all(
                        torch.equal(base, candidate)
                        for base, candidate in zip(
                            base_rollout.counts, candidate_rollout.counts
                        )
                    ),
                    "birth_budget_exact": len(base_rollout.features)
                    == len(candidate_rollout.features),
                    "base_stable_ids_exact": base_rollout.report[
                        "stable_ids_exact"
                    ],
                    "candidate_stable_ids_exact": candidate_rollout.report[
                        "stable_ids_exact"
                    ],
                    "base_topology_contract": base_rollout.topology_contract,
                    "candidate_topology_contract": candidate_rollout.topology_contract,
                }
        field_payload = {
            "features": rollouts["correct"].features,
            "global_ids": rollouts["correct"].global_ids,
            "background": background,
            "rollout": rollouts["correct"].report,
        }
        if paired_rollouts:
            field_payload.update(
                {
                    "frozen_base_features": paired_rollouts[
                        "correct"
                    ].base.features,
                    "lifecycle_appearance": paired_rollouts["correct"].report,
                    "appearance_adapter_checkpoint": args.appearance_adapter,
                }
            )
        elif paired_base_rollouts:
            field_payload.update(
                {
                    "frozen_base_features": paired_base_rollouts[
                        "correct"
                    ].features,
                    "mark_only_attribution": mark_only_attribution["correct"],
                    "paired_base_flow_checkpoint": args.paired_base_flow,
                }
            )
        torch.save(field_payload, output_dir / field_name)
        record = {
            "source_id": source_id,
            "class_name": source.field.class_name,
            "sampling_seed": source_seed,
            "shuffled_source_id": (
                None if args.correct_only else alternate[source_id]
            ),
            "completed_frames": completed_frames,
            "background": {
                "contract": "initial_scaffold_rgb_mean",
                "predicted": background.tolist(),
                "fitted_reference": fitted_background.tolist(),
                "mae": float((background - fitted_background).abs().mean()),
            },
            "render_signatures": signatures,
            "saliency_signatures": saliency_signatures,
            "seams": seams,
            "density": density,
            "rollouts": {name: rollout.report for name, rollout in rollouts.items()},
            "lifecycle_appearance": (
                {name: rollout.report for name, rollout in paired_rollouts.items()}
                if paired_rollouts
                else None
            ),
            "mark_only_attribution": mark_only_attribution,
            "artifacts": {
                "gif": gif_name,
                "contact_sheet": contact_name,
                "generated_field": field_name,
            },
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    aggregate_contact = Image.new(
        "RGB",
        (
            max(row.width for row in aggregate_rows),
            sum(row.height for row in aggregate_rows),
        ),
        "white",
    )
    offset = 0
    for row in aggregate_rows:
        aggregate_contact.paste(row, (0, offset))
        offset += row.height
    aggregate_contact.save(output_dir / "three_window_rollout_contact.png")
    topology_report = {
        name: topology_metrics(values, target_counts)
        for name, values in predicted_counts.items()
    }
    macro_render = {
        panel: _macro_average(
            [
                {"signature": record["render_signatures"][panel]}
                for record in records
            ],
            "signature",
        )
        for panel in panels
        if panel != "LTX scaffold"
    }
    macro_seams = {
        panel: _macro_average(
            [{"seam": record["seams"][panel]} for record in records], "seam"
        )
        for panel in panels
        if panel != "LTX scaffold"
    }
    macro_saliency = {
        panel: _macro_average(
            [
                {"signature": record["saliency_signatures"][panel]}
                for record in records
            ],
            "signature",
        )
        for panel in panels
        if panel != "LTX scaffold"
    }
    macro_density = {
        panel: sum(record["density"][panel]["effective"]["mean"] for record in records)
        / len(records)
        for panel in panels
        if panel != "LTX scaffold"
    }
    lifecycle_appearance_report = None
    if paired_appearance:
        paired = [
            control
            for record in records
            for control in record["lifecycle_appearance"].values()
        ]
        lifecycle_appearance_report = {
            "all_lifecycle_exact": all(item["lifecycle_exact"] for item in paired),
            "all_non_appearance_exact": all(
                item.get("non_appearance_exact", True) for item in paired
            ),
            "all_stable_ids_exact": all(item["stable_ids_exact"] for item in paired),
            "all_topology_exact": all(item["topology_exact"] for item in paired),
            "controls_audited": len(paired),
            "mechanism": (
                "compact_rgb_adapter"
                if appearance_adapter is not None
                else "second_full_flow"
            ),
        }
    mark_only_report = None
    if paired_topology:
        paired = [
            control
            for record in records
            for control in record["mark_only_attribution"].values()
        ]
        mark_only_report = {
            "all_counts_exact": all(item["counts_exact"] for item in paired),
            "all_birth_budgets_exact": all(
                item["birth_budget_exact"] for item in paired
            ),
            "all_stable_ids_exact": all(
                item["base_stable_ids_exact"]
                and item["candidate_stable_ids_exact"]
                for item in paired
            ),
            "controls_audited": len(paired),
            "base_lock": base_lock,
        }
    summary = {
        "schema": (
            "scaffold-mark-paired-topology-rollout-v1"
            if paired_topology
            else (
                "scaffold-mark-rgb-adapter-rollout-v1"
                if appearance_adapter is not None
                else (
                    "scaffold-mark-lifecycle-appearance-rollout-v1"
                    if appearance_flow is not None
                    else "scaffold-mark-three-window-rollout-v1"
                )
            )
        ),
        "sources": len(records),
        "sampling_steps": args.steps,
        "sampling_seed": args.seed,
        "source_seed_contract": "base_seed_plus_full_validation_order",
        "deterministic": args.deterministic,
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "inputs": {
            "topology_checkpoint": args.topology,
            "mark_flow_checkpoint": args.mark_flow,
            "mark_flow_sha256": _sha256(args.mark_flow),
            "paired_base_flow_checkpoint": args.paired_base_flow,
            "paired_base_flow_sha256": (
                _sha256(args.paired_base_flow) if args.paired_base_flow else None
            ),
            "appearance_flow_checkpoint": args.appearance_flow,
            "appearance_adapter_checkpoint": args.appearance_adapter,
            "appearance_dimension_set": (
                args.appearance_dimension_set if appearance_flow is not None else None
            ),
            "appearance_dimensions": (
                list(APPEARANCE_DIMENSION_SETS[args.appearance_dimension_set])
                if appearance_flow is not None
                else (list(RGB_DIMENSIONS) if appearance_adapter is not None else None)
            ),
            "appearance_saliency_fraction": (
                args.appearance_saliency_fraction
                if paired_appearance
                else None
            ),
            "appearance_strength": (
                args.appearance_strength if paired_appearance else None
            ),
            "manifest": args.manifest,
            "prompt_cache": args.prompt_cache,
            "checkpoint_roots": args.checkpoint_root,
            "manifest_sha256": prompt_cache.manifest_sha256,
        },
        "initial_boundary_contract": (
            "strict_finite_support"
            if args.strict_initial_boundary
            else "left_censored_clip_start"
        ),
        "completed_strides": 3,
        "controls": list(controls),
        "occupancy_threshold": occupancy_threshold,
        "topology": topology_report,
        "macro_render_signatures": macro_render,
        "macro_saliency_signatures": macro_saliency,
        "macro_seams": macro_seams,
        "macro_effective_density": macro_density,
        "lifecycle_appearance": lifecycle_appearance_report,
        "mark_only_attribution": mark_only_report,
        "mean_background_mae": sum(record["background"]["mae"] for record in records)
        / len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
