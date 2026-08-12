"""Decompose prompted continuation washout into topology and jewel-mark errors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_streaming_continuation import _panel, _row, frame_points
from sol.splat_density import temporal_standard_deviation
from sol.streaming_corpus import build_prompted_continuation_corpus, load_prompted_fields
from sol.streaming_data import BirthTarget, rasterize_context
from sol.streaming_features import to_global_time
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


FEATURE_GROUPS = {
    "center": slice(0, 3),
    "covariance": slice(3, 9),
    "color": slice(9, 12),
    "gradient": slice(12, 21),
    "opacity": slice(21, 22),
}


@dataclass(frozen=True)
class TopologyAdherence:
    spatial_cell_fraction: float
    birth_cell_fraction: float
    birth_commit_fraction: float


@dataclass(frozen=True)
class RenderSignature:
    psnr: float
    ssim: float
    contrast_ratio: float
    edge_ratio: float
    saturation_ratio: float
    temporal_change_ratio: float


def replace_groups(
    base: torch.Tensor, source: torch.Tensor, groups: tuple[str, ...]
) -> torch.Tensor:
    """Copy named canonical feature groups from ``source`` into ``base``."""
    if base.shape != source.shape or base.ndim != 2 or base.shape[1] != 22:
        raise ValueError("feature tensors must have matching (N,22) shapes")
    unknown = set(groups) - FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    output = base.clone()
    for group in groups:
        output[:, FEATURE_GROUPS[group]] = source[:, FEATURE_GROUPS[group]]
    return output


def _spatial_cells(values: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    gu, gv, gt = spec.shape
    scaled = (values[:, :2].clamp(-1, 1) + 1) * 0.5
    u = (scaled[:, 0] * gu).long().clamp_max(gu - 1)
    v = (scaled[:, 1] * gv).long().clamp_max(gv - 1)
    return torch.stack((u, v), dim=1)


def topology_adherence(
    local_features: torch.Tensor,
    assigned_cells: torch.Tensor,
    *,
    spec: GridSpec,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    support_sigma: float,
) -> TopologyAdherence:
    """Measure whether decoded marks honor their assigned spatial and birth cells."""
    if len(local_features) != len(assigned_cells):
        raise ValueError("every feature requires one assigned cell")
    if not len(local_features):
        return TopologyAdherence(1.0, 1.0, 1.0)
    local_features = local_features.detach().cpu()
    assigned_cells = assigned_cells.detach().cpu()
    gu, gv, gt = spec.shape
    assigned_u = assigned_cells // (gv * gt)
    assigned_v = (assigned_cells // gt) % gv
    assigned_t = assigned_cells % gt
    actual_spatial = _spatial_cells(local_features, spec)
    spatial_match = (actual_spatial[:, 0] == assigned_u) & (
        actual_spatial[:, 1] == assigned_v
    )

    global_features = to_global_time(
        local_features, total_frames, frontier, stride_frames
    )
    temporal_sigma = temporal_standard_deviation(global_features)
    continuous_start = (
        global_features[:, 2] - support_sigma * temporal_sigma + 1
    ) * ((total_frames - 1) / 2)
    first_active = continuous_start.ceil().long()
    commit_match = (first_active >= frontier) & (
        first_active < frontier + stride_frames
    )
    relative = (first_active - frontier).clamp(0, stride_frames - 1)
    actual_t = (relative * gt // stride_frames).clamp_max(gt - 1)
    birth_match = spatial_match & commit_match & (actual_t == assigned_t)
    return TopologyAdherence(
        spatial_cell_fraction=float(spatial_match.float().mean()),
        birth_cell_fraction=float(birth_match.float().mean()),
        birth_commit_fraction=float(commit_match.float().mean()),
    )


def _to_global_for_render(
    local_features: torch.Tensor,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the batched symmetric time transform on CPU, then return to the renderer."""
    global_features = to_global_time(
        local_features.detach().cpu(), total_frames, frontier, stride_frames
    )
    return global_features.to(device)


def render_signature(
    candidate: torch.Tensor, target: torch.Tensor
) -> RenderSignature:
    """Compare fidelity and visible-detail energy against a matched target render."""
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("renders must have matching (T,H,W,3) shapes")
    candidate = candidate.clamp(0, 1).float()
    target = target.clamp(0, 1).float()

    def _mean_luma_std(frames: torch.Tensor) -> torch.Tensor:
        luma = (frames * frames.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)
        return luma.flatten(1).std(dim=1).mean()

    def _edge_energy(frames: torch.Tensor) -> torch.Tensor:
        horizontal = (frames[:, :, 1:] - frames[:, :, :-1]).abs().mean()
        vertical = (frames[:, 1:] - frames[:, :-1]).abs().mean()
        return horizontal + vertical

    def _saturation(frames: torch.Tensor) -> torch.Tensor:
        return (frames.amax(-1) - frames.amin(-1)).mean()

    def _temporal_change(frames: torch.Tensor) -> torch.Tensor:
        if len(frames) < 2:
            return frames.new_tensor(0.0)
        return (frames[1:] - frames[:-1]).abs().mean()

    def _ratio(candidate_value: torch.Tensor, target_value: torch.Tensor) -> float:
        return float(candidate_value / target_value.clamp_min(1e-8))

    def _ssim(candidate_frames: torch.Tensor, target_frames: torch.Tensor) -> float:
        dimensions = (1, 2)
        candidate_mean = candidate_frames.mean(dim=dimensions)
        target_mean = target_frames.mean(dim=dimensions)
        candidate_centered = candidate_frames - candidate_mean[:, None, None]
        target_centered = target_frames - target_mean[:, None, None]
        candidate_variance = candidate_centered.square().mean(dim=dimensions)
        target_variance = target_centered.square().mean(dim=dimensions)
        covariance = (candidate_centered * target_centered).mean(dim=dimensions)
        c1 = 0.01**2
        c2 = 0.03**2
        score = (
            (2 * candidate_mean * target_mean + c1)
            * (2 * covariance + c2)
            / (
                (candidate_mean.square() + target_mean.square() + c1)
                * (candidate_variance + target_variance + c2)
            )
        )
        return float(score.mean())

    mse = F.mse_loss(candidate, target).clamp_min(1e-10)
    return RenderSignature(
        psnr=float(-10 * torch.log10(mse)),
        ssim=_ssim(candidate, target),
        contrast_ratio=_ratio(_mean_luma_std(candidate), _mean_luma_std(target)),
        edge_ratio=_ratio(_edge_energy(candidate), _edge_energy(target)),
        saturation_ratio=_ratio(_saturation(candidate), _saturation(target)),
        temporal_change_ratio=_ratio(
            _temporal_change(candidate), _temporal_change(target)
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
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
    model = BirthContinuationModel(grid_spec=spec, **meta["model_args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    fits = _fit_lookup(args.checkpoint_root)
    manifest_examples = {item["source_id"]: item for item in manifest["examples"]}
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for example in sorted(corpus.validation, key=lambda item: item.class_id):
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
        text = corpus.prompt_embeddings[example.evaluation_prompt_indices[0]].to(device)
        output = model.forward_training(context, target, text)
        predicted_local = corpus.birth_standardizer.denormalize(
            output.occupied_features
        )
        target_local = births.values.to(device)
        free = model.decode(context, text)
        free_local = corpus.birth_standardizer.denormalize(free.values)

        local_fields = {
            "oracle predicted marks": predicted_local,
            "predicted geometry": replace_groups(
                target_local, predicted_local, ("center", "covariance")
            ),
            "predicted color": replace_groups(
                target_local, predicted_local, ("color", "gradient")
            ),
            "predicted opacity": replace_groups(
                target_local, predicted_local, ("opacity",)
            ),
        }
        carried = view.carried_global_features.to(device)
        fields_to_render = {
            "fitted target": view.target_active_global_features.to(device),
            "carried only": carried,
        }
        for name, values in local_fields.items():
            fields_to_render[name] = torch.cat(
                (
                    carried,
                    _to_global_for_render(
                        values,
                        example.dataset.total_frames,
                        view.frontier,
                        example.dataset.stride_frames,
                        device,
                    ),
                )
            )
        fields_to_render["free prediction"] = torch.cat(
            (
                carried,
                _to_global_for_render(
                    free_local,
                    example.dataset.total_frames,
                    view.frontier,
                    example.dataset.stride_frames,
                    device,
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
        target_render = rendered["fitted target"]
        signatures = {
            name: asdict(render_signature(value, target_render))
            for name, value in rendered.items()
            if name != "fitted target"
        }
        adherence = asdict(
            topology_adherence(
                predicted_local,
                target.cell_indices,
                spec=spec,
                total_frames=example.dataset.total_frames,
                frontier=view.frontier,
                stride_frames=example.dataset.stride_frames,
                support_sigma=example.dataset.support_sigma,
            )
        )
        normalized_error = (output.occupied_features - target.values).square()
        group_mse = {
            name: float(normalized_error[:, part].mean())
            for name, part in FEATURE_GROUPS.items()
        }

        names = list(rendered)
        frames = [
            _row([_panel(rendered[name][index], name, args.upscale) for name in names])
            for index in range(len(frame_indices))
        ]
        artifact = f"{example.source_id}_washout_decomposition.gif"
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
        contact_name = f"{example.source_id}_washout_decomposition_contact.png"
        contact.save(output_dir / contact_name)
        record = {
            "source_id": example.source_id,
            "class_name": example.class_name,
            "target_births": len(target_local),
            "free_births": len(free_local),
            "topology_adherence": adherence,
            "normalized_group_mse": group_mse,
            "render_signatures": signatures,
            "artifact": artifact,
            "contact_sheet": contact_name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    (output_dir / "washout_report.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
