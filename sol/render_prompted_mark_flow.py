"""Render oracle-topology stochastic jewel marks against deterministic washout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import torch

from sol.audit_prompted_washout import render_signature, topology_adherence
from sol.birth_mark_flow import (
    BirthMarkFlowModel,
    project_birth_topology,
    sample_birth_marks,
)
from sol.multiscale_video_guide import video_to_multiscale_cell_tokens
from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.streaming_corpus import build_prompted_continuation_corpus, load_prompted_fields
from sol.streaming_data import BirthTarget, rasterize_context
from sol.streaming_features import to_global_time
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-flow", required=True)
    parser.add_argument("--deterministic-continuation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--view", type=int, default=0)
    return parser.parse_args()


def _fit_lookup(roots: list[str]) -> dict[str, Path]:
    paths = {}
    for root in roots:
        for path in Path(root).glob("*_w000000.pt"):
            if path.name in paths:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            paths[path.name] = path
    return paths


def _global_for_render(
    local_features: torch.Tensor,
    *,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    device: torch.device,
) -> torch.Tensor:
    return to_global_time(
        local_features.detach().cpu(), total_frames, frontier, stride_frames
    ).to(device)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.steps, args.height, args.width, args.upscale) <= 0 or args.view < 0:
        raise ValueError("sampling, render dimensions, and view must be valid")
    device = torch.device(args.device)
    flow_saved = torch.load(args.mark_flow, map_location="cpu", weights_only=False)
    flow_meta = flow_saved["meta"]
    if flow_meta.get("architecture") != "prompted_birth_mark_flow_v1":
        raise ValueError("checkpoint is not a prompted birth-mark flow")
    deterministic_saved = torch.load(
        args.deterministic_continuation, map_location="cpu", weights_only=False
    )
    deterministic_meta = deterministic_saved["meta"]
    if deterministic_meta.get("architecture") != "prompted_birth_continuation_v1":
        raise ValueError("baseline is not a prompted deterministic continuation")
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    spec = GridSpec(
        tuple(flow_meta["grid_shape"]), int(flow_meta["slots_per_cell"])
    )
    if (
        tuple(deterministic_meta["grid_shape"]) != spec.shape
        or int(deterministic_meta["slots_per_cell"]) != spec.slots_per_cell
    ):
        raise ValueError("flow and deterministic baseline use different topology")
    train_args = flow_meta["train_args"]
    corpus = build_prompted_continuation_corpus(
        fields,
        prompt_cache.embeddings,
        prefix_frames=int(train_args["prefix_frames"]),
        stride_frames=int(train_args["stride_frames"]),
        support_sigma=float(train_args["support_sigma"]),
        grid_spec=spec,
    )
    flow = BirthMarkFlowModel(grid_spec=spec, **flow_meta["model_args"]).to(device)
    flow.load_state_dict(flow_saved["model"])
    flow.eval()
    deterministic = BirthContinuationModel(
        grid_spec=spec, **deterministic_meta["model_args"]
    ).to(device)
    deterministic.load_state_dict(deterministic_saved["model"])
    deterministic.eval()
    validation = sorted(corpus.validation, key=lambda example: example.class_id)
    classes = [example.class_id for example in validation]
    prompt_by_class = {
        example.class_id: corpus.prompt_embeddings[example.evaluation_prompt_indices[0]].to(
            device
        )
        for example in validation
    }
    shuffled_class = {
        class_id: classes[(index + 1) % len(classes)]
        for index, class_id in enumerate(classes)
    }
    fits = _fit_lookup(args.checkpoint_root)
    manifest_examples = {item["source_id"]: item for item in manifest["examples"]}
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for example_index, example in enumerate(validation):
        view = example.dataset.views[args.view]
        context = rasterize_context(
            view.context_features,
            corpus.context_standardizer,
            prefix_frames=example.dataset.prefix_frames,
            stride_frames=example.dataset.stride_frames,
            grid_shape=spec.shape,
        ).to(device)
        births = view.births
        target = BirthTarget(
            values=corpus.birth_standardizer.normalize(births.values).to(device),
            cell_indices=births.cell_indices.to(device),
            slot_indices=births.slot_indices.to(device),
            counts=births.counts.to(device),
            global_ids=births.global_ids.to(device),
            birth_frames=births.birth_frames.to(device),
        )
        correct_text = prompt_by_class[example.class_id]
        wrong_text = prompt_by_class[shuffled_class[example.class_id]]
        source = manifest_examples[example.source_id]
        guide = None
        guide_tokens = None
        if flow.guide_dim or flow.guide_token_dim:
            video = load_video(
                source["video"],
                max_frames=example.dataset.total_frames,
                start_frame=int(source.get("start_frame", 0)),
                resize=(args.height, args.width),
                device="cpu",
            )
            future_video = video[view.frontier : view.commit_stop]
            if flow.guide_dim:
                guide = video_to_cell_raster(future_video, spec).to(device)
            if flow.guide_token_dim:
                guide_tokens = video_to_multiscale_cell_tokens(
                    future_video,
                    spec,
                    scales=tuple(train_args["guide_scales"]),
                    subgrid=tuple(train_args["guide_subgrid"]),
                ).to(device)
        deterministic_output = deterministic.forward_training(
            context, target, correct_text
        )
        deterministic_local = corpus.birth_standardizer.denormalize(
            deterministic_output.occupied_features
        )

        def sample(
            text: torch.Tensor,
            context_raster: torch.Tensor,
            guide_raster: torch.Tensor | None,
            multiscale_tokens: torch.Tensor | None,
        ) -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(
                args.seed + example_index
            )
            normalized = sample_birth_marks(
                flow,
                context_raster,
                target.cell_indices,
                target.slot_indices,
                text,
                steps=args.steps,
                generator=generator,
                guide_raster=guide_raster,
                guide_tokens=multiscale_tokens,
            )
            return corpus.birth_standardizer.denormalize(normalized)

        correct_raw = sample(correct_text, context, guide, guide_tokens)
        shuffled_raw = sample(wrong_text, context, guide, guide_tokens)
        correct_projected = project_birth_topology(
            correct_raw,
            target.cell_indices,
            spec=spec,
            support_sigma=example.dataset.support_sigma,
            stride_frames=example.dataset.stride_frames,
        )
        shuffled_projected = project_birth_topology(
            shuffled_raw,
            target.cell_indices,
            spec=spec,
            support_sigma=example.dataset.support_sigma,
            stride_frames=example.dataset.stride_frames,
        )
        target_local = births.values.to(device)
        target_projected = project_birth_topology(
            target_local,
            target.cell_indices,
            spec=spec,
            support_sigma=example.dataset.support_sigma,
            stride_frames=example.dataset.stride_frames,
        )
        if flow.guide_dim or flow.guide_token_dim:
            no_guide_raw = sample(
                correct_text,
                context,
                (
                    torch.zeros(spec.n_cells, flow.guide_dim, device=device)
                    if flow.guide_dim
                    else None
                ),
                (
                    torch.zeros_like(guide_tokens)
                    if guide_tokens is not None
                    else None
                ),
            )
            no_guide_projected = project_birth_topology(
                no_guide_raw,
                target.cell_indices,
                spec=spec,
                support_sigma=example.dataset.support_sigma,
                stride_frames=example.dataset.stride_frames,
            )
            local_fields = {
                "projected target": target_projected,
                "deterministic marks": deterministic_local,
                "flow no guide": no_guide_projected,
                "flow guided raw": correct_raw,
                "flow guided": correct_projected,
                "guided shuffled": shuffled_projected,
            }
        else:
            text_only_raw = sample(
                correct_text, torch.zeros_like(context), None, None
            )
            text_only_projected = project_birth_topology(
                text_only_raw,
                target.cell_indices,
                spec=spec,
                support_sigma=example.dataset.support_sigma,
                stride_frames=example.dataset.stride_frames,
            )
            local_fields = {
                "projected target": target_projected,
                "deterministic marks": deterministic_local,
                "flow raw": correct_raw,
                "flow projected": correct_projected,
                "flow shuffled": shuffled_projected,
                "flow text only": text_only_projected,
            }
        carried = view.carried_global_features.to(device)
        fields_to_render = {
            "fitted target": view.target_active_global_features.to(device),
            "carried only": carried,
        }
        for name, local in local_fields.items():
            fields_to_render[name] = torch.cat(
                (
                    carried,
                    _global_for_render(
                        local,
                        total_frames=example.dataset.total_frames,
                        frontier=view.frontier,
                        stride_frames=example.dataset.stride_frames,
                        device=device,
                    ),
                )
            )
        frame_indices = torch.arange(view.frontier, view.commit_stop)
        points = frame_points(
            example.dataset.total_frames,
            frame_indices,
            args.height,
            args.width,
            device=device,
        )
        fit_name = f"{Path(source['video']).stem}_w000000.pt"
        fitted = torch.load(fits[fit_name], map_location="cpu", weights_only=False)
        background = torch.tensor(fitted["info"]["background"], device=device)
        rendered = {
            name: render_exact(field, points, background=background)
            .reshape(len(frame_indices), args.height, args.width, 3)
            .cpu()
            for name, field in fields_to_render.items()
        }
        target_render = rendered["fitted target"]
        signatures = {
            name: asdict(render_signature(value, target_render))
            for name, value in rendered.items()
            if name != "fitted target"
        }
        adherence = {
            name: asdict(
                topology_adherence(
                    local,
                    target.cell_indices,
                    spec=spec,
                    total_frames=example.dataset.total_frames,
                    frontier=view.frontier,
                    stride_frames=example.dataset.stride_frames,
                    support_sigma=example.dataset.support_sigma,
                )
            )
            for name, local in local_fields.items()
        }
        names = list(rendered)
        frames = [
            _row([_panel(rendered[name][index], name, args.upscale) for name in names])
            for index in range(len(frame_indices))
        ]
        artifact = f"{example.source_id}_mark_flow_controls.gif"
        frames[0].save(
            output_dir / artifact,
            save_all=True,
            append_images=frames[1:],
            duration=83,
            loop=0,
        )
        picks = (0, len(frames) // 2, len(frames) - 1)
        contact = Image.new(
            "RGB",
            (frames[0].width, sum(frames[index].height for index in picks)),
            "white",
        )
        offset = 0
        for index in picks:
            contact.paste(frames[index], (0, offset))
            offset += frames[index].height
        contact_name = f"{example.source_id}_mark_flow_controls_contact.png"
        contact.save(output_dir / contact_name)
        record = {
            "source_id": example.source_id,
            "class_name": example.class_name,
            "target_births": len(target_local),
            "sampling_steps": args.steps,
            "render_signatures": signatures,
            "topology_adherence": adherence,
            "artifact": artifact,
            "contact_sheet": contact_name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "mark_flow_visual_report.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
