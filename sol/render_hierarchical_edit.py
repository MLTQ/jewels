"""Render a cursor translation followed by local hierarchical flow repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.block_codec import BlockPCACodec
from sol.cache_latents import _restore_tokenizer
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.edit import plan_translation_edit
from sol.evaluate_latent_prior import _restore_prior
from sol.geometry import Parallelepiped, translate_selected
from sol.hierarchical_inpaint import (
    hierarchical_masked_flow_inpaint,
    restore_clean_codes,
)
from sol.latent_data import load_latent_cache
from sol.render_prior_samples import _panel, _render, _row
from sol.token_grid import GridSpec


def _triple(values: list[float]) -> torch.Tensor:
    if len(values) != 3:
        raise ValueError("expected three coordinates")
    return torch.tensor(values, dtype=torch.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--fine-cache", required=True)
    parser.add_argument("--block-codec", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--center", nargs=3, type=float, default=(0.25, 0.0, 0.0))
    parser.add_argument(
        "--half-extent", nargs=3, type=float, default=(0.16, 0.35, 0.35)
    )
    parser.add_argument("--delta", nargs=3, type=float, default=(-0.30, 0.0, 0.0))
    parser.add_argument("--halo-cells", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.frames, args.steps, args.knn) <= 0:
        raise ValueError("render and sampling counts must be positive")
    device = torch.device(args.device)
    coarse_cache = load_latent_cache(args.coarse_cache)
    fine_cache = load_latent_cache(args.fine_cache)
    if coarse_cache.names != fine_cache.names:
        raise ValueError("fine and coarse caches are not sample-aligned")
    codec = BlockPCACodec.from_state_dict(
        torch.load(args.block_codec, map_location="cpu", weights_only=False)
    )
    prior_checkpoint = torch.load(args.prior, map_location="cpu", weights_only=False)
    prior = _restore_prior(prior_checkpoint).to(device).eval()
    tokenizer_checkpoint = torch.load(
        args.tokenizer, map_location="cpu", weights_only=False
    )
    tokenizer_meta = tokenizer_checkpoint["meta"]
    spec = GridSpec(
        tuple(tokenizer_meta["grid_shape"]),
        int(tokenizer_meta["slots_per_cell"]),
    )
    if spec.shape != codec.grid_shape:
        raise ValueError("tokenizer and block codec grids do not match")
    tokenizer = _restore_tokenizer(tokenizer_checkpoint, spec).to(device).eval()
    feature_normalizer = FeatureNormalizer.from_state_dict(
        tokenizer_meta["normalizer"]
    )
    examples = load_fitted_corpus(args.corpus)
    held_out = set(coarse_cache.metadata["validation_sources"])
    candidates = [example for example in examples if example.source_id in held_out]
    if args.name:
        candidates = [example for example in candidates if example.name == args.name]
    if not candidates:
        raise ValueError("no matching held-out example")
    example = candidates[0]
    index = coarse_cache.names.index(example.name)

    center = _triple(list(args.center))
    half_extent = _triple(list(args.half_extent))
    delta = _triple(list(args.delta))
    selection = Parallelepiped.axis_aligned(center, half_extent)
    plan = plan_translation_edit(
        example.features, selection, delta, spec, halo_cells=args.halo_cells
    )
    edited, selected = translate_selected(example.features, selection, delta)
    condition = coarse_cache.normalized_conditions[index : index + 1].to(device)
    known_normalized = coarse_cache.normalized_latents[index : index + 1].to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    repair = hierarchical_masked_flow_inpaint(
        prior,
        known_normalized,
        plan.dirty_cells,
        codec.grid_shape,
        codec.block_size,
        condition=condition,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        generator=generator,
    )
    clean_coarse = ~repair.dirty_coarse.to(device)
    clean_coarse_error = float(
        (repair.normalized_coarse[:, clean_coarse] - known_normalized[:, clean_coarse])
        .abs()
        .max()
    )
    original_codes = coarse_cache.latents[index : index + 1].to(device)
    repaired_codes = coarse_cache.denormalize(repair.normalized_coarse)
    repaired_codes = restore_clean_codes(
        repaired_codes, original_codes, repair.dirty_coarse
    )
    original_fine_normalized = codec.decode(original_codes)
    repaired_fine_normalized = codec.decode(repaired_codes)
    affected_fine = repair.affected_fine.to(device)
    clean_fine_error = float(
        (
            repaired_fine_normalized[:, ~affected_fine]
            - original_fine_normalized[:, ~affected_fine]
        )
        .abs()
        .max()
    )

    original_fine = fine_cache.denormalize(original_fine_normalized)
    repaired_fine = fine_cache.denormalize(repaired_fine_normalized)
    hierarchy = feature_normalizer.denormalize(tokenizer.decode(original_fine)[0]).cpu()
    generated = feature_normalizer.denormalize(tokenizer.decode(repaired_fine)[0]).cpu()
    affected_cpu = repair.affected_fine.cpu()
    generated_cells = spec.cell_index(generated[:, :3])
    generated_dirty = generated[affected_cpu[generated_cells]]
    edited_cells = spec.cell_index(edited[:, :3])
    clean_context = edited[~affected_cpu[edited_cells] & ~selected]
    merged = torch.cat([clean_context, generated_dirty, edited[selected]], dim=0)

    picks = torch.linspace(0, example.shape[0] - 1, args.frames).round().long()
    target_frames = _render(
        example.features, example.shape, picks, example.background, device, args.knn
    )
    hierarchy_frames = _render(
        hierarchy, example.shape, picks, example.background, device, args.knn
    )
    moved_frames = _render(
        edited, example.shape, picks, example.background, device, args.knn
    )
    repaired_frames = _render(
        merged, example.shape, picks, example.background, device, args.knn
    )
    frames = [
        _row(
            [
                _panel(target_frames[i], "fitted target"),
                _panel(hierarchy_frames[i], "hierarchy round-trip"),
                _panel(moved_frames[i], "selected jewels moved"),
                _panel(repaired_frames[i], "moved + local repair"),
            ]
        )
        for i in range(len(picks))
    ]
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{Path(example.name).stem}_hierarchical_edit.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    record = {
        "name": example.name,
        "source_id": example.source_id,
        "selection_center": center.tolist(),
        "selection_half_extent": half_extent.tolist(),
        "delta": delta.tolist(),
        "selected_jewels": int(selected.sum()),
        "fine_dirty_cells": int(plan.dirty_cells.sum()),
        "coarse_dirty_codes": int(repair.dirty_coarse.sum()),
        "affected_fine_cells": int(repair.affected_fine.sum()),
        "coarse_dirty_fraction": float(repair.dirty_coarse.float().mean()),
        "clean_coarse_max_abs_error": clean_coarse_error,
        "clean_fine_max_abs_error": clean_fine_error,
        "hierarchy_jewels": int(hierarchy.shape[0]),
        "generated_dirty_jewels": int(generated_dirty.shape[0]),
        "merged_jewels": int(merged.shape[0]),
        "frames": picks.tolist(),
        "artifact": path.name,
    }
    (output_dir / "edit_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
