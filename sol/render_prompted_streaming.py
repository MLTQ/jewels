"""Render free-count prompt controls for direct jewel continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.streaming_corpus import (
    build_prompted_continuation_corpus,
    load_prompted_fields,
)
from sol.streaming_data import rasterize_context
from sol.streaming_features import to_global_time
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
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


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.height, args.width, args.upscale) <= 0 or args.view < 0:
        raise ValueError("render dimensions and view must be valid")
    device = torch.device(args.device)
    saved = torch.load(args.continuation, map_location="cpu", weights_only=False)
    meta = saved["meta"]
    if meta.get("architecture") != "prompted_birth_continuation_v1":
        raise ValueError("checkpoint is not a prompted birth continuation model")
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    train_args = meta["train_args"]
    corpus = build_prompted_continuation_corpus(
        fields,
        prompt_cache.embeddings,
        prefix_frames=int(train_args["prefix_frames"]),
        stride_frames=int(train_args["stride_frames"]),
        support_sigma=float(train_args["support_sigma"]),
        grid_spec=spec,
    )
    expected_context = meta["context_standardizer"]
    expected_birth = meta["birth_standardizer"]
    context_matches = torch.equal(
        corpus.context_standardizer.mean, expected_context["mean"]
    ) and torch.equal(corpus.context_standardizer.std, expected_context["std"])
    if not context_matches:
        raise ValueError("rebuilt context standardizer disagrees with checkpoint")
    birth_matches = torch.equal(
        corpus.birth_standardizer.mean, expected_birth["mean"]
    ) and torch.equal(corpus.birth_standardizer.std, expected_birth["std"])
    if not birth_matches:
        raise ValueError("rebuilt birth standardizer disagrees with checkpoint")
    model = BirthContinuationModel(grid_spec=spec, **meta["model_args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
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
    manifest_examples = {
        item["source_id"]: item for item in manifest["examples"]
    }
    fits = _fit_lookup(args.checkpoint_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for example in validation:
        if args.view >= len(example.dataset.views):
            raise ValueError(
                f"view {args.view} is unavailable for {example.source_id}"
            )
        view = example.dataset.views[args.view]
        context = rasterize_context(
            view.context_features,
            corpus.context_standardizer,
            prefix_frames=example.dataset.prefix_frames,
            stride_frames=example.dataset.stride_frames,
            grid_shape=spec.shape,
        ).to(device)
        zero_context = torch.zeros_like(context)
        correct_text = prompt_by_class[example.class_id]
        wrong_text = prompt_by_class[shuffled_class[example.class_id]]
        conditions = {
            "prefix + correct text": (context, correct_text),
            "text only: correct": (zero_context, correct_text),
            "text only: shuffled": (zero_context, wrong_text),
            "text only: null": (zero_context, None),
        }
        fields_to_render = {
            "fitted target": view.target_active_global_features.to(device)
        }
        counts = {}
        for name, (condition_context, text) in conditions.items():
            decoded = model.decode(condition_context, text)
            local = corpus.birth_standardizer.denormalize(decoded.values)
            global_births = to_global_time(
                local,
                example.dataset.total_frames,
                view.frontier,
                example.dataset.stride_frames,
            )
            fields_to_render[name] = torch.cat(
                (view.carried_global_features.to(device), global_births), dim=0
            )
            counts[name] = int(decoded.counts.sum())
        frame_indices = torch.arange(view.frontier, view.commit_stop)
        points = frame_points(
            example.dataset.total_frames,
            frame_indices,
            args.height,
            args.width,
            device=device,
        )
        source = manifest_examples[example.source_id]
        fit_name = f"{Path(source['video']).stem}_w000000.pt"
        fitted = torch.load(fits[fit_name], map_location="cpu", weights_only=False)
        background = torch.tensor(fitted["info"]["background"], device=device)
        rendered = {
            name: render_exact(field, points, background=background)
            .reshape(len(frame_indices), args.height, args.width, 3)
            .cpu()
            for name, field in fields_to_render.items()
        }
        names = list(rendered)
        frames = [
            _row(
                [
                    _panel(
                        rendered[name][index],
                        name if name == "fitted target" else f"{name} ({counts[name]:,} births)",
                        args.upscale,
                    )
                    for name in names
                ]
            )
            for index in range(len(frame_indices))
        ]
        stem = example.source_id
        artifact = f"{stem}_prompt_controls.gif"
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
        contact_artifact = f"{stem}_prompt_controls_contact.png"
        contact.save(output_dir / contact_artifact)
        target_render = rendered["fitted target"]
        record = {
            "source_id": example.source_id,
            "class_name": example.class_name,
            "view": args.view,
            "target_births": len(view.births.values),
            "predicted_births": counts,
            "field_psnr": {
                name: float(
                    -10
                    * torch.log10(
                        F.mse_loss(value, target_render).clamp_min(1e-10)
                    )
                )
                for name, value in rendered.items()
                if name != "fitted target"
            },
            "artifact": artifact,
            "contact_sheet": contact_artifact,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "visual_report.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
