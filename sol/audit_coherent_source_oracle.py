"""Audit the one-source-per-window coherent Jewel language upper bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import load_field_records
from sol.audit_scene_block_constellation_oracle import (
    _lowest_fits,
    evaluate_generation,
    hierarchical_control_metrics,
    scene_key,
)
from sol.audit_scene_posterior_oracle import select_oracle_sources
from sol.block_token_language import BlockTokenCodebook, encode_block_tokens
from sol.coherent_source_realizer import fit_coherent_source_realizer
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.train_block_token_oracle import cyclic_shuffled_programs
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=108)
    args = parser.parse_args()
    if min(
        args.generation_jewels, args.voxel_points, args.frames,
        args.height, args.width,
    ) <= 0:
        raise ValueError("coherent source oracle render settings must be positive")
    device = torch.device(args.device)
    seed = 20260911
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if (
        int(architecture["block_vocabulary_size"]) != 1024
        or tuple(architecture["block_shape"]) != (16, 16, 8)
    ):
        raise ValueError("Gate 2a8 requires the frozen fine K=1024 language")
    block_codebook = BlockTokenCodebook.from_state_dict(
        checkpoint["block_codebook"], device
    )
    training_programs = checkpoint["training_programs"].to(device)
    split_protocol = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split_protocol["roots"]])
    training, validation = select_prompt_splits(
        records,
        set(split_protocol["validation_sources"]),
        set(split_protocol["training_sources"]),
    )
    training = sorted(training, key=lambda record: record.source_id)
    validation = sorted(validation, key=lambda record: (record.source_id, record.fit_seed))
    if [record.source_id for record in training] != list(checkpoint["training_sources"]):
        raise ValueError("fine programs do not align with registered training sources")
    keys = sorted({scene_key(record) for record in training})
    if len(keys) != 3:
        raise ValueError("Gate 2a8 requires three semantic scene tokens")
    key_to_scene = {key: index for index, key in enumerate(keys)}
    training_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in training],
        dtype=torch.long, device=device,
    )
    validation_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in validation],
        dtype=torch.long, device=device,
    )
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_coherent_source_realizer(
        [record.features.to(device) for record in training],
        training_scenes,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=0.005,
        smoothing=0.1,
    )
    null_token = realizer.most_frequent_nonempty_token(training_programs)
    validation_programs = torch.stack([
        encode_block_tokens(record.features.to(device), block_codebook)[0]
        for record in validation
    ])
    validation_shuffled = cyclic_shuffled_programs(validation, validation_programs)
    validation_controls = hierarchical_control_metrics(
        realizer, validation, validation_scenes,
        validation_programs, validation_shuffled, null_token,
        physical_codebook=physical_codebook, device=device,
    )
    direct_records = select_oracle_sources(
        training, [record.source_id for record in training]
    )
    program_by_source = {
        record.source_id: training_programs[index]
        for index, record in enumerate(training)
    }
    direct_programs = torch.stack([
        program_by_source[record.source_id] for record in direct_records
    ])
    direct_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in direct_records],
        dtype=torch.long, device=device,
    )
    direct_shuffled = torch.roll(direct_programs, shifts=-1, dims=0)
    direct_controls = hierarchical_control_metrics(
        realizer, direct_records, direct_scenes,
        direct_programs, direct_shuffled, null_token,
        physical_codebook=physical_codebook, device=device,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    direct_rows, direct_generation = evaluate_generation(
        realizer, direct_records, direct_scenes,
        direct_programs, direct_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 1000,
        image_path=output / "qualitative_direct.png",
    )
    heldout_records = _lowest_fits(validation)
    validation_by_path = {
        record.path: (validation_programs[index], validation_scenes[index])
        for index, record in enumerate(validation)
    }
    heldout_programs = torch.stack([
        validation_by_path[record.path][0] for record in heldout_records
    ])
    heldout_scenes = torch.stack([
        validation_by_path[record.path][1] for record in heldout_records
    ])
    heldout_shuffled = torch.roll(heldout_programs, shifts=-1, dims=0)
    heldout_rows, heldout_generation = evaluate_generation(
        realizer, heldout_records, heldout_scenes,
        heldout_programs, heldout_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 2000,
        image_path=output / "qualitative_source_disjoint.png",
    )
    competing = ("shuffled scene", "null hierarchy")
    correct_nll = validation_controls["oracle hierarchy"]["token_nll_macro"]
    correct_hist = heldout_generation["oracle hierarchy"]["target_histogram_cosine"]
    selected_rows = [
        int(row["arms"]["oracle hierarchy"]["realization"]["selected_training_field"])
        for row in heldout_rows
    ]
    checks = {
        "source_disjoint_correct_beats_shuffled_scene_and_null_token_nll": all(
            correct_nll < validation_controls[arm]["token_nll_macro"]
            for arm in competing
        ),
        "source_disjoint_correct_beats_shuffled_scene_and_null_histogram": all(
            correct_hist > heldout_generation[arm]["target_histogram_cosine"]
            for arm in competing
        ),
        "all_generated_counts_exact": all(
            row["arms"][arm]["realization"]["emitted_jewels"]
            == args.generation_jewels
            for row in direct_rows + heldout_rows
            for arm in ("oracle hierarchy", "shuffled scene", "shuffled blocks", "null hierarchy")
        ),
        "generated_centers_not_grid_locked": max(
            row["arms"][arm]["audit"]["center_irregularity"]["cell_center_lock_fraction"]
            for row in direct_rows + heldout_rows
            for arm in ("oracle hierarchy", "shuffled scene", "shuffled blocks", "null hierarchy")
        ) < 0.01,
        "all_generated_renders_finite": all(
            math.isfinite(row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"])
            for row in direct_rows + heldout_rows
            for arm in ("oracle hierarchy", "shuffled scene", "shuffled blocks", "null hierarchy")
        ),
    }
    report = {
        "schema": "coherent-source-program-oracle-v1",
        "protocol": {
            "block_checkpoint": args.block_checkpoint,
            "split_report": args.split_report,
            "scene_keys": keys,
            "training_sources": [record.source_id for record in training],
            "validation_sources": sorted({record.source_id for record in validation}),
            "generation_jewels": args.generation_jewels,
            "one_training_source_program_per_window": True,
            "target_program_selects_training_source": True,
            "oracle_is_retrieval_not_prompt_inference": True,
            "oracle_is_not_novel_generation": True,
            "sample_time_target_field": False,
            "emitted_centroids_are_continuous": True,
        },
        "fit": fit_report,
        "null_token": null_token,
        "teacher_forced": {
            "direct_comparison": direct_controls,
            "source_disjoint": validation_controls,
        },
        "generation": {
            "direct_comparison_macro": direct_generation,
            "source_disjoint_macro": heldout_generation,
            "direct_records": direct_rows,
            "source_disjoint_records": heldout_rows,
        },
        "source_disjoint_selected_training_rows": selected_rows,
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_recognizability_review_required": True,
            "pass_does_not_license_retrieval_as_text_to_video": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "validation_controls": validation_controls,
        "heldout_generation": heldout_generation,
        "selected_training_rows": selected_rows,
        "gate": report["gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
