"""Render a matched single-field, exact-topology mark memorization audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import torch

from sol.audit_prompted_washout import render_signature
from sol.birth_mark_flow import (
    BirthMarkFlowModel,
    project_birth_topology,
    sample_birth_marks,
)
from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_scaffold_mark_rollout import _configure_determinism, _seam_report
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.saliency_metrics import saliency_render_signature
from sol.scaffold_mark_data import (
    ScaffoldMarkCorpus,
    build_scaffold_mark_corpus,
    rasterize_scaffold_context,
)
from sol.splat_density import measure_frame_splat_density, summarize_counts
from sol.streaming_corpus import load_prompted_fields
from sol.streaming_data import FeatureStandardizer
from sol.streaming_features import to_global_time
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


PANELS = (
    "LTX source",
    "fitted jewel ceiling",
    "feature-only",
    "render-trained fitted bg",
    "render-trained learned bg",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-flow", required=True)
    parser.add_argument("--render-flow", required=True)
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
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _single_field_identity(manifest: dict, corpus: ScaffoldMarkCorpus) -> tuple:
    """Return the validation source/item after proving physical field overlap."""
    if len(corpus.train) != 1 or len(corpus.validation) != 1:
        raise ValueError("memorization audit requires one train and one validation alias")
    examples = manifest.get("examples", [])
    if len(examples) != 2 or {item.get("split") for item in examples} != {
        "train",
        "validation",
    }:
        raise ValueError("manifest must contain one train/validation alias pair")
    stems = {item.get("shared_field_stem") for item in examples}
    if len(stems) != 1 or None in stems:
        raise ValueError("memorization aliases must share one physical field stem")
    if not manifest.get("single_field_overfit_class"):
        raise ValueError("manifest lacks the explicit single-field overfit contract")
    validation_item = next(item for item in examples if item["split"] == "validation")
    validation = corpus.validation[0]
    if validation.field.source_id != validation_item["source_id"]:
        raise ValueError("validation source ownership disagrees with manifest")
    return validation, validation_item


def _fit_path(item: dict, roots: list[str]) -> Path:
    name = f"{Path(item['video']).stem}_w000000.pt"
    matches = [Path(root) / name for root in roots if (Path(root) / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one fitted checkpoint named {name}: {matches}")
    return matches[0]


def _standardizers_match(
    expected: FeatureStandardizer, saved: dict[str, torch.Tensor]
) -> bool:
    actual = FeatureStandardizer.from_state_dict(saved)
    return torch.equal(expected.mean, actual.mean) and torch.equal(
        expected.std, actual.std
    )


def _load_flow(
    path: str,
    corpus: ScaffoldMarkCorpus,
    manifest_sha256: str,
    device: torch.device,
) -> tuple[BirthMarkFlowModel, dict, torch.Tensor | None]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved.get("meta", {})
    if meta.get("architecture") != "scaffold_birth_mark_flow_v1":
        raise ValueError(f"checkpoint is not a scaffold mark flow: {path}")
    if meta.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"checkpoint does not own this manifest: {path}")
    if tuple(meta.get("grid_shape", ())) != corpus.grid_spec.shape or int(
        meta.get("slots_per_cell", -1)
    ) != corpus.grid_spec.slots_per_cell:
        raise ValueError("checkpoint grid disagrees with the memorization corpus")
    if not _standardizers_match(
        corpus.context_standardizer, meta["context_standardizer"]
    ) or not _standardizers_match(
        corpus.birth_standardizer, meta["birth_standardizer"]
    ):
        raise ValueError("checkpoint normalizers disagree with the memorization corpus")
    model = BirthMarkFlowModel(
        grid_spec=corpus.grid_spec, **meta["model_args"]
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    learned = meta.get("learned_background")
    if learned is not None:
        learned = torch.as_tensor(learned, dtype=torch.float32)
    return model, meta, learned


def _render_field(
    features: torch.Tensor,
    points: torch.Tensor,
    background: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    frames, height, width = shape
    return (
        render_exact(
            features.to(points.device), points, background=background.to(points.device)
        )
        .reshape(frames, height, width, 3)
        .detach()
        .cpu()
    )


def _teacher_forced_density(
    fields: list[torch.Tensor],
    *,
    total_frames: int,
    stride_frames: int,
    support_sigma: float,
) -> dict:
    """Stitch per-window density only over the stride each field commits."""
    effective = []
    visible = []
    for index, features in enumerate(fields):
        density = measure_frame_splat_density(
            features.cpu(), total_frames, support_sigma=support_sigma
        )
        start = index * stride_frames
        stop = min(start + stride_frames, total_frames)
        if stop - start != stride_frames:
            raise ValueError("teacher-forced density requires complete aligned strides")
        effective.append(density.effective_peak_alpha_counts[start:stop])
        visible.append(density.peak_alpha_counts[0.05][start:stop])
    effective_values = torch.cat(effective)
    visible_values = torch.cat(visible)
    return {
        "effective": summarize_counts(effective_values),
        "above_5_percent_alpha": summarize_counts(visible_values),
        "per_frame_effective": [float(value) for value in effective_values],
        "per_frame_above_5_percent_alpha": [int(value) for value in visible_values],
    }


def _metric_bundle(
    candidate: torch.Tensor,
    target: torch.Tensor,
    *,
    background: torch.Tensor,
    stride_frames: int,
) -> dict:
    return {
        "render": asdict(render_signature(candidate, target)),
        "saliency": asdict(
            saliency_render_signature(candidate, target, background=background)
        ),
        "seams": _seam_report(candidate, target, stride_frames),
    }


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.steps, args.height, args.width, args.upscale) <= 0:
        raise ValueError("sampling and render dimensions must be positive")
    device = torch.device(args.device)
    _configure_determinism(args.deterministic, device)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)

    feature_saved = torch.load(args.feature_flow, map_location="cpu", weights_only=False)
    feature_meta = feature_saved.get("meta", {})
    spec = GridSpec(
        tuple(feature_meta.get("grid_shape", ())),
        int(feature_meta.get("slots_per_cell", -1)),
    )
    train_args = feature_meta.get("train_args", {})
    corpus = build_scaffold_mark_corpus(
        fields,
        prompt_cache.embeddings,
        stride_frames=int(train_args.get("stride_frames", 16)),
        support_sigma=float(train_args.get("support_sigma", 3.0)),
        grid_spec=spec,
    )
    source, item = _single_field_identity(manifest, corpus)
    feature_model, feature_meta, feature_background = _load_flow(
        args.feature_flow, corpus, prompt_cache.manifest_sha256, device
    )
    render_model, render_meta, learned_background = _load_flow(
        args.render_flow, corpus, prompt_cache.manifest_sha256, device
    )
    if feature_background is not None:
        raise ValueError("feature-only control unexpectedly contains a learned background")
    if render_meta.get("background_contract") != "single_field_learned_rgb" or (
        learned_background is None
    ):
        raise ValueError("render checkpoint lacks its single-field learned background")
    if feature_meta["model_args"] != render_meta["model_args"]:
        raise ValueError("matched checkpoints must use the same model architecture")

    fit_path = _fit_path(item, args.checkpoint_root)
    fitted = torch.load(fit_path, map_location="cpu", weights_only=False)
    fitted_background = torch.as_tensor(
        fitted["info"]["background"], dtype=torch.float32
    )
    target_video = load_video(
        item["video"],
        max_frames=source.field.frames,
        start_frame=int(item.get("start_frame", 0)),
        resize=(args.height, args.width),
        device="cpu",
    ).float()
    if len(target_video) != source.field.frames:
        raise ValueError("source video length disagrees with fitted field")
    prompt_indices = source.field.evaluation_prompt_indices
    if not prompt_indices:
        raise ValueError("validation alias has no evaluation prompt")
    prompt = corpus.prompt_embeddings[prompt_indices[0]].to(device)

    rendered_windows = {name: [] for name in PANELS}
    density_fields = {
        "fitted jewel ceiling": [],
        "feature-only": [],
        "render-trained fitted bg": [],
        "render-trained learned bg": [],
    }
    for view in source.views:
        frame_indices = torch.arange(view.frontier, view.commit_stop)
        points = frame_points(
            source.field.frames,
            frame_indices,
            args.height,
            args.width,
            device=device,
        )
        shape = (len(frame_indices), args.height, args.width)
        guide = video_to_cell_raster(
            target_video[view.frontier : view.commit_stop], spec
        ).to(device)
        context = rasterize_scaffold_context(
            view.context_features,
            corpus.context_standardizer,
            stride_frames=corpus.stride_frames,
            grid_spec=spec,
        ).to(device)
        cells = view.births.cell_indices.to(device)
        slots = view.births.slot_indices.to(device)

        generated_fields = {}
        for name, model in (
            ("feature-only", feature_model),
            ("render-trained", render_model),
        ):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + view.index
            )
            standardized = sample_birth_marks(
                model,
                context,
                cells,
                slots,
                prompt,
                steps=args.steps,
                generator=generator,
                guide_raster=guide,
            )
            local = corpus.birth_standardizer.denormalize(standardized)
            local = project_birth_topology(
                local,
                cells,
                spec=spec,
                support_sigma=corpus.support_sigma,
                stride_frames=corpus.stride_frames,
                allow_prefrontier_support=view.frontier == 0,
            )
            births = to_global_time(
                local.cpu(),
                source.field.frames,
                view.frontier,
                corpus.stride_frames,
            )
            generated_fields[name] = torch.cat(
                (view.carried_global_features, births), dim=0
            )

        exact = view.target_active_global_features
        rendered_windows["LTX source"].append(
            target_video[view.frontier : view.commit_stop]
        )
        rendered_windows["fitted jewel ceiling"].append(
            _render_field(exact, points, fitted_background, shape)
        )
        rendered_windows["feature-only"].append(
            _render_field(
                generated_fields["feature-only"], points, fitted_background, shape
            )
        )
        for panel, background in (
            ("render-trained fitted bg", fitted_background),
            ("render-trained learned bg", learned_background),
        ):
            rendered_windows[panel].append(
                _render_field(
                    generated_fields["render-trained"], points, background, shape
                )
            )
        density_fields["fitted jewel ceiling"].append(exact)
        density_fields["feature-only"].append(generated_fields["feature-only"])
        density_fields["render-trained fitted bg"].append(
            generated_fields["render-trained"]
        )
        density_fields["render-trained learned bg"].append(
            generated_fields["render-trained"]
        )

    rendered = {name: torch.cat(windows) for name, windows in rendered_windows.items()}
    completed_frames = len(rendered["LTX source"])
    if completed_frames != len(source.views) * corpus.stride_frames:
        raise AssertionError("rendered windows do not form complete strides")
    animation = [
        _row([_panel(rendered[name][frame], name, args.upscale) for name in PANELS])
        for frame in range(completed_frames)
    ]
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    animation[0].save(
        output_dir / "single_field_memorization.gif",
        save_all=True,
        append_images=animation[1:],
        duration=83,
        loop=0,
    )
    picks = sorted(
        {
            0,
            *(index * corpus.stride_frames - 1 for index in range(1, len(source.views))),
            *(index * corpus.stride_frames for index in range(1, len(source.views))),
            completed_frames - 1,
        }
    )
    contact = Image.new(
        "RGB", (animation[0].width, sum(animation[index].height for index in picks)), "white"
    )
    offset = 0
    for index in picks:
        contact.paste(animation[index], (0, offset))
        offset += animation[index].height
    contact.save(output_dir / "single_field_memorization_contact.png")

    candidates = {name: frames for name, frames in rendered.items() if name != "LTX source"}
    report = {
        "experiment": "single-field-exact-topology-fitted-carry-memorization",
        "class_name": source.field.class_name,
        "source_id": source.field.source_id,
        "physical_field_stem": item["shared_field_stem"],
        "feature_flow": args.feature_flow,
        "render_flow": args.render_flow,
        "fit_checkpoint": str(fit_path),
        "completed_frames": completed_frames,
        "sampling_steps": args.steps,
        "deterministic": args.deterministic,
        "background": {
            "fitted": fitted_background.tolist(),
            "learned": learned_background.tolist(),
            "mae": float((learned_background - fitted_background).abs().mean()),
        },
        "against_source": {
            name: _metric_bundle(
                frames,
                rendered["LTX source"],
                background=fitted_background,
                stride_frames=corpus.stride_frames,
            )
            for name, frames in candidates.items()
        },
        "against_fitted_ceiling": {
            name: _metric_bundle(
                frames,
                rendered["fitted jewel ceiling"],
                background=fitted_background,
                stride_frames=corpus.stride_frames,
            )
            for name, frames in candidates.items()
            if name != "fitted jewel ceiling"
        },
        "density": {
            name: _teacher_forced_density(
                values,
                total_frames=source.field.frames,
                stride_frames=corpus.stride_frames,
                support_sigma=corpus.support_sigma,
            )
            for name, values in density_fields.items()
        },
        "artifacts": {
            "gif": "single_field_memorization.gif",
            "contact_sheet": "single_field_memorization_contact.png",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
