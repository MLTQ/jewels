"""Audit semantic density-balanced trajectory-tube Jewel composition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import load_field_records
from sol.audit_scene_block_constellation_oracle import _lowest_fits, scene_key
from sol.audit_trajectory_tube_oracle import ARMS, evaluate_source_disjoint
from sol.block_token_language import BlockTokenCodebook, encode_block_tokens
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.semantic_trajectory_realizer import fit_semantic_trajectory_realizer
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
        raise ValueError("semantic trajectory audit settings must be positive")
    device = torch.device(args.device)
    seed = 20260913
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if (
        int(architecture["block_vocabulary_size"]) != 1024
        or tuple(architecture["block_shape"]) != (16, 16, 8)
    ):
        raise ValueError("Gate 2a10 requires the frozen fine K=1024 language")
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
    key_to_scene = {key: index for index, key in enumerate(keys)}
    training_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in training],
        dtype=torch.long, device=device,
    )
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_semantic_trajectory_realizer(
        [record.features.to(device) for record in training],
        training_scenes,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=0.005,
    )
    null_token = realizer.coherent.most_frequent_nonempty_token(training_programs)
    heldout_records = _lowest_fits(validation)
    validation_programs = {
        record.path: encode_block_tokens(record.features.to(device), block_codebook)[0]
        for record in validation
    }
    programs = torch.stack([validation_programs[record.path] for record in heldout_records])
    scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in heldout_records],
        dtype=torch.long, device=device,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows, macro = evaluate_source_disjoint(
        realizer,
        heldout_records,
        scenes,
        programs,
        physical_codebook=physical_codebook,
        training_sources=[record.source_id for record in training],
        null_token=null_token,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames,
        height=args.height,
        width=args.width,
        seed=seed,
        image_path=output / "qualitative_source_disjoint.png",
    )
    composites = [row["arms"]["tube composite"]["realization"] for row in rows]
    checks = {
        "distinct_foreground_background_donors": all(
            row["foreground_training_field"] != row["background_training_field"]
            for row in composites
        ),
        "both_donors_contribute_at_least_20pct": all(
            min(row["foreground_fraction"], row["background_fraction"]) >= 0.20
            for row in composites
        ),
        "count_adjustment_below_5pct": max(
            row["adjustment_fraction"] for row in composites
        ) < 0.05,
        "all_counts_exact": all(
            row["arms"][arm]["realization"]["emitted_jewels"]
            == args.generation_jewels
            for row in rows for arm in ARMS
        ),
        "correct_histogram_beats_wrong_object_and_null": all(
            macro["tube composite"]["target_histogram_cosine"]
            > macro[arm]["target_histogram_cosine"]
            for arm in ("wrong-object tube", "pooled null")
        ),
        "generated_centers_not_grid_locked": max(
            row["arms"][arm]["audit"]["center_irregularity"][
                "cell_center_lock_fraction"
            ] for row in rows for arm in ARMS
        ) < 0.01,
        "all_generated_renders_finite": all(
            math.isfinite(
                row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"]
            ) for row in rows for arm in ARMS
        ),
    }
    report = {
        "schema": "semantic-density-balanced-trajectory-oracle-v1",
        "protocol": {
            "block_checkpoint": args.block_checkpoint,
            "split_report": args.split_report,
            "scene_keys": keys,
            "training_sources": [record.source_id for record in training],
            "validation_sources": [record.source_id for record in heldout_records],
            "target_program_selects_training_donors": True,
            "target_contributes_jewel_rows_or_path": False,
            "scene_path_is_training_only": True,
            "radius_uses_donor_density_only": True,
            "two_distinct_training_programs_per_composite": True,
            "oracle_is_not_prompt_inference": True,
            "oracle_is_template_backed": True,
            "emitted_centroids_are_continuous": True,
            "generation_jewels": args.generation_jewels,
        },
        "fit": fit_report,
        "macro": macro,
        "source_disjoint_records": rows,
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_recognizability_and_wrong_object_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"macro": macro, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
