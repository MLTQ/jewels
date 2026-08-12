"""Render frozen mark flow under learned, oracle, and shuffled scaffold topology."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import torch

from sol.audit_prompted_washout import render_signature, topology_adherence
from sol.birth_mark_flow import BirthMarkFlowModel
from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.scaffold_topology_eval import topology_metrics
from sol.scaffold_topology_realizer import (
    predict_realizer_topology,
    realize_topology_marks,
    validate_realizer_topology,
)
from sol.splat_density import measure_frame_splat_density, summarize_counts
from sol.streaming_corpus import build_prompted_continuation_corpus, load_prompted_fields
from sol.streaming_data import FeatureStandardizer, rasterize_context
from sol.streaming_features import to_global_time
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


PANELS = (
    "fitted target",
    "carried only",
    "oracle topology",
    "learned topology",
    "shuffled topology",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--mark-flow", required=True)
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
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--source-id", action="append", default=[])
    return parser.parse_args()


def _fit_lookup(roots: list[str]) -> dict[str, Path]:
    paths = {}
    for root in roots:
        for path in Path(root).glob("*_w000000.pt"):
            if path.name in paths:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            paths[path.name] = path
    return paths


def _macro_average(records: list[dict], key: str) -> dict[str, float]:
    """Average one flat numeric record section equally over held-out sources."""
    if not records or any(key not in record for record in records):
        raise ValueError("macro aggregation requires aligned non-empty records")
    names = tuple(records[0][key])
    if any(tuple(record[key]) != names for record in records):
        raise ValueError("macro aggregation sections have different keys")
    return {
        name: sum(float(record[key][name]) for record in records) / len(records)
        for name in names
    }


def _density_report(
    features: torch.Tensor,
    total_frames: int,
    frontier: int,
    commit_stop: int,
    support_sigma: float,
) -> dict:
    density = measure_frame_splat_density(
        features.cpu(), total_frames, support_sigma=support_sigma
    )
    effective = density.effective_peak_alpha_counts[frontier:commit_stop]
    alpha = density.peak_alpha_counts[0.05][frontier:commit_stop]
    return {
        "effective": summarize_counts(effective),
        "above_5_percent_alpha": summarize_counts(alpha),
    }


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.steps, args.height, args.width, args.upscale) <= 0 or args.view < 0:
        raise ValueError("sampling, render dimensions, and view must be valid")
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)

    topology_saved = torch.load(args.topology, map_location="cpu", weights_only=False)
    topology_meta = topology_saved["meta"]
    if topology_meta.get("architecture") != "scaffold_topology_v1":
        raise ValueError("checkpoint is not a scaffold topology model")
    if topology_meta.get("manifest_sha256") != prompt_cache.manifest_sha256:
        raise ValueError("topology checkpoint does not own this scaffold manifest")
    topology_spec = GridSpec(
        tuple(topology_meta["grid_shape"]), int(topology_meta["slots_per_cell"])
    )
    topology_model = ScaffoldTopologyModel(
        grid_spec=topology_spec, **topology_meta["model_args"]
    ).to(device)
    topology_model.load_state_dict(topology_saved["model"])
    topology_model.eval()
    latest = topology_meta.get("latest_evaluation")
    if not latest:
        raise ValueError("topology checkpoint has no train-calibrated evaluation")
    occupancy_threshold = float(latest["validation"]["occupancy_threshold"])

    flow_saved = torch.load(args.mark_flow, map_location="cpu", weights_only=False)
    flow_meta = flow_saved["meta"]
    if flow_meta.get("architecture") != "prompted_birth_mark_flow_v1":
        raise ValueError("checkpoint is not a prompted birth-mark flow")
    if int(flow_meta["model_args"].get("guide_dim", 0)) != 3 or int(
        flow_meta["model_args"].get("guide_token_dim", 0)
    ):
        raise ValueError("coupling gate requires the selected cell-RGB-only realizer")
    realizer_spec = GridSpec(
        tuple(flow_meta["grid_shape"]), int(flow_meta["slots_per_cell"])
    )
    if topology_spec.shape != realizer_spec.shape:
        raise ValueError("topology and realizer checkpoints use different grid shapes")
    manifest_training = {
        item["source_id"] for item in manifest["examples"] if item["split"] == "train"
    }
    if manifest_training != set(flow_meta["training_sources"]):
        raise ValueError("manifest training sources differ from the frozen realizer")
    flow = BirthMarkFlowModel(
        grid_spec=realizer_spec, **flow_meta["model_args"]
    ).to(device)
    flow.load_state_dict(flow_saved["model"])
    flow.eval()
    flow_args = flow_meta["train_args"]
    corpus = build_prompted_continuation_corpus(
        fields,
        prompt_cache.embeddings,
        prefix_frames=int(flow_args["prefix_frames"]),
        stride_frames=int(flow_args["stride_frames"]),
        support_sigma=float(flow_args["support_sigma"]),
        grid_spec=realizer_spec,
    )
    context_standardizer = FeatureStandardizer.from_state_dict(
        flow_meta["context_standardizer"]
    )
    birth_standardizer = FeatureStandardizer.from_state_dict(
        flow_meta["birth_standardizer"]
    )
    for actual, expected in (
        (corpus.context_standardizer, context_standardizer),
        (corpus.birth_standardizer, birth_standardizer),
    ):
        if not torch.equal(actual.mean, expected.mean) or not torch.equal(
            actual.std, expected.std
        ):
            raise ValueError("reconstructed corpus standardizers differ from the flow")

    validation = sorted(corpus.validation, key=lambda item: item.class_id)
    if len({example.class_id for example in validation}) < 2:
        raise ValueError("shuffled topology requires at least two validation classes")
    source_filter = set(args.source_id)
    available = {example.source_id for example in validation}
    if source_filter - available:
        raise ValueError(f"unknown validation source IDs: {sorted(source_filter - available)}")
    selected = [
        example
        for example in validation
        if not source_filter or example.source_id in source_filter
    ]
    manifest_examples = {item["source_id"]: item for item in manifest["examples"]}
    fits = _fit_lookup(args.checkpoint_root)
    guides = {}
    videos = {}
    for example in validation:
        if not 0 <= args.view < len(example.dataset.views):
            raise ValueError(
                f"view {args.view} is unavailable for source {example.source_id}"
            )
        view = example.dataset.views[args.view]
        source = manifest_examples[example.source_id]
        video = load_video(
            source["video"],
            max_frames=example.dataset.total_frames,
            start_frame=int(source.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        )
        if len(video) != example.dataset.total_frames:
            raise ValueError(f"video/field length mismatch for {example.source_id}")
        videos[example.source_id] = video
        guides[example.source_id] = video_to_cell_raster(
            video[view.frontier : view.commit_stop], topology_spec
        )
    shuffled = {
        example.source_id: validation[(index + 1) % len(validation)].source_id
        for index, example in enumerate(validation)
    }

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    predicted_counts = {"learned": [], "shuffled": [], "null": []}
    target_counts = []
    aggregate_rows = []

    for example_index, example in enumerate(selected):
        view = example.dataset.views[args.view]
        guide = guides[example.source_id]
        alternate_guide = guides[shuffled[example.source_id]]
        common = {
            "total_frames": example.dataset.total_frames,
            "frontier": view.frontier,
            "stride_frames": example.dataset.stride_frames,
            "support_sigma": example.dataset.support_sigma,
            "topology_spec": topology_spec,
            "realizer_spec": realizer_spec,
            "occupancy_threshold": occupancy_threshold,
            "device": device,
        }
        learned_topology = predict_realizer_topology(
            topology_model,
            guide,
            view.carried_global_features,
            **common,
        )
        shuffled_topology = predict_realizer_topology(
            topology_model,
            alternate_guide,
            view.carried_global_features,
            **common,
        )
        null_topology = predict_realizer_topology(
            topology_model,
            torch.zeros_like(guide),
            view.carried_global_features,
            **common,
        )
        oracle_topology = validate_realizer_topology(
            view.births.counts.long().cpu(), topology_spec, realizer_spec
        )
        target_counts.append(oracle_topology.counts)
        predicted_counts["learned"].append(learned_topology.counts)
        predicted_counts["shuffled"].append(shuffled_topology.counts)
        predicted_counts["null"].append(null_topology.counts)

        context = rasterize_context(
            view.context_features,
            context_standardizer,
            prefix_frames=example.dataset.prefix_frames,
            stride_frames=example.dataset.stride_frames,
            grid_shape=realizer_spec.shape,
        ).to(device)
        text = prompt_cache.embeddings[example.evaluation_prompt_indices[0]].to(device)

        def realize(topology, seed_offset: int) -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(
                args.seed + example_index + seed_offset
            )
            return realize_topology_marks(
                flow,
                context,
                topology,
                text,
                birth_standardizer,
                guide_raster=guide.to(device),
                support_sigma=example.dataset.support_sigma,
                stride_frames=example.dataset.stride_frames,
                steps=args.steps,
                generator=generator,
            )

        local_marks = {
            "oracle topology": realize(oracle_topology, 0),
            "learned topology": realize(learned_topology, 0),
            "shuffled topology": realize(shuffled_topology, 0),
        }
        carried = view.carried_global_features.cpu()
        fields_to_render = {
            "fitted target": view.target_active_global_features.cpu(),
            "carried only": carried,
        }
        carry_error = {}
        for name, local in local_marks.items():
            generated = to_global_time(
                local.detach().cpu(),
                example.dataset.total_frames,
                view.frontier,
                example.dataset.stride_frames,
            )
            combined = torch.cat((carried, generated), dim=0)
            fields_to_render[name] = combined
            carry_error[name] = (
                float((combined[: len(carried)] - carried).abs().max())
                if len(carried)
                else 0.0
            )

        source = manifest_examples[example.source_id]
        fit_name = f"{Path(source['video']).stem}_w000000.pt"
        fitted_path = fits.get(fit_name)
        if fitted_path is None:
            raise FileNotFoundError(f"missing fitted checkpoint {fit_name}")
        fitted = torch.load(fitted_path, map_location="cpu", weights_only=False)
        background = torch.tensor(fitted["info"]["background"], device=device)
        frame_indices = torch.arange(view.frontier, view.commit_stop)
        points = frame_points(
            example.dataset.total_frames,
            frame_indices,
            args.height,
            args.width,
            device=device,
        )
        rendered = {}
        for name in PANELS:
            rendered[name] = (
                render_exact(
                    fields_to_render[name].to(device), points, background=background
                )
                .reshape(len(frame_indices), args.height, args.width, 3)
                .cpu()
            )
        signatures = {
            name: asdict(render_signature(rendered[name], rendered["fitted target"]))
            for name in PANELS
            if name != "fitted target"
        }
        adherence = {
            name: asdict(
                topology_adherence(
                    local,
                    (
                        oracle_topology.cell_indices
                        if name == "oracle topology"
                        else learned_topology.cell_indices
                        if name == "learned topology"
                        else shuffled_topology.cell_indices
                    ),
                    spec=realizer_spec,
                    total_frames=example.dataset.total_frames,
                    frontier=view.frontier,
                    stride_frames=example.dataset.stride_frames,
                    support_sigma=example.dataset.support_sigma,
                )
            )
            for name, local in local_marks.items()
        }
        density = {
            name: _density_report(
                field,
                example.dataset.total_frames,
                view.frontier,
                view.commit_stop,
                example.dataset.support_sigma,
            )
            for name, field in fields_to_render.items()
        }
        animation = [
            _row([_panel(rendered[name][index], name, args.upscale) for name in PANELS])
            for index in range(len(frame_indices))
        ]
        gif_name = f"{example.source_id}_autonomous_topology.gif"
        animation[0].save(
            output_dir / gif_name,
            save_all=True,
            append_images=animation[1:],
            duration=83,
            loop=0,
        )
        picks = (0, len(animation) // 2, len(animation) - 1)
        contact = Image.new(
            "RGB",
            (animation[0].width, sum(animation[index].height for index in picks)),
            "white",
        )
        offset = 0
        for index in picks:
            contact.paste(animation[index], (0, offset))
            offset += animation[index].height
        contact_name = f"{example.source_id}_autonomous_topology_contact.png"
        contact.save(output_dir / contact_name)
        aggregate_rows.append(animation[len(animation) // 2])
        record = {
            "source_id": example.source_id,
            "class_name": example.class_name,
            "view": args.view,
            "frontier": view.frontier,
            "commit_stop": view.commit_stop,
            "shuffled_source_id": shuffled[example.source_id],
            "births": {
                "target": int(oracle_topology.counts.sum()),
                "learned": int(learned_topology.counts.sum()),
                "shuffled": int(shuffled_topology.counts.sum()),
                "null": int(null_topology.counts.sum()),
                "target_max_cell": int(oracle_topology.counts.max()),
                "learned_max_cell": int(learned_topology.counts.max()),
            },
            "render_signatures": signatures,
            "topology_adherence": adherence,
            "density": density,
            "max_carry_feature_error": carry_error,
            "artifact": gif_name,
            "contact_sheet": contact_name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    aggregate_contact = Image.new(
        "RGB",
        (max(row.width for row in aggregate_rows), sum(row.height for row in aggregate_rows)),
        "white",
    )
    offset = 0
    for row in aggregate_rows:
        aggregate_contact.paste(row, (0, offset))
        offset += row.height
    aggregate_contact.save(output_dir / "autonomous_topology_contact.png")

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
        for panel in PANELS
        if panel != "fitted target"
    }
    macro_effective_density = {
        panel: sum(
            record["density"][panel]["effective"]["mean"] for record in records
        )
        / len(records)
        for panel in PANELS
    }
    summary = {
        "schema": "scaffold-topology-realizer-gate-v1",
        "sources": len(records),
        "sampling_steps": args.steps,
        "occupancy_threshold": occupancy_threshold,
        "topology": topology_report,
        "macro_render_signatures": macro_render,
        "macro_effective_density": macro_effective_density,
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
