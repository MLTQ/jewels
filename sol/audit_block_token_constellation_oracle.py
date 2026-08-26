"""Audit complete predefined medoid constellations for K=1024 block tokens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sol.audit_block_token_empirical_oracle import (
    empirical_control_metrics,
    evaluate_generation,
)
from sol.audit_jewel_casting_language import FieldRecord, load_field_records
from sol.audit_scene_posterior_oracle import select_oracle_sources
from sol.block_token_constellation_realizer import fit_constellation_block_realizer
from sol.block_token_language import BlockTokenCodebook, encode_block_tokens
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.train_block_token_oracle import ARMS, cyclic_shuffled_programs
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


def adjustment_macro(rows: list[dict], arm: str = "oracle block") -> dict[str, float]:
    """Summarize how much exact-count normalization changed assembled constellations."""
    values = [row["arms"][arm]["realization"] for row in rows]
    return {
        "mean_unadjusted_jewels": sum(row["unadjusted_jewels"] for row in values) / len(values),
        "mean_adjustment_fraction": sum(row["adjustment_fraction"] for row in values) / len(values),
        "max_adjustment_fraction": max(row["adjustment_fraction"] for row in values),
    }


def _lowest_fits(records: list[FieldRecord]) -> list[FieldRecord]:
    return [
        min((record for record in records if record.source_id == source),
            key=lambda record: record.fit_seed)
        for source in sorted({record.source_id for record in records})
    ]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--posterior-oracle-report", required=True)
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
        raise ValueError("constellation oracle render settings must be positive")
    device = torch.device(args.device)
    seed = 20260907
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["architecture"]["block_vocabulary_size"]) != 1024:
        raise ValueError("Gate 2a4 requires the frozen K=1024 block checkpoint")
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
        raise ValueError("K=1024 programs do not align with registered training sources")
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_constellation_block_realizer(
        [record.features.to(device) for record in training],
        training_programs,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        smoothing=0.1,
        jitter_std=0.005,
    )
    null_token = realizer.most_frequent_nonempty_token(training_programs)
    validation_programs = torch.stack([
        encode_block_tokens(record.features.to(device), block_codebook)[0]
        for record in validation
    ])
    validation_shuffled = cyclic_shuffled_programs(validation, validation_programs)
    validation_controls = empirical_control_metrics(
        realizer, validation, validation_programs, validation_shuffled, null_token,
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
    direct_shuffled = torch.roll(direct_programs, shifts=-1, dims=0)
    direct_controls = empirical_control_metrics(
        realizer, direct_records, direct_programs, direct_shuffled, null_token,
        physical_codebook=physical_codebook, device=device,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    direct_rows, direct_generation = evaluate_generation(
        realizer, direct_records, direct_programs, direct_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 1000,
        image_path=output / "qualitative_direct.png",
    )
    heldout_records = _lowest_fits(validation)
    validation_by_path = {
        record.path: validation_programs[index]
        for index, record in enumerate(validation)
    }
    heldout_programs = torch.stack([
        validation_by_path[record.path] for record in heldout_records
    ])
    heldout_shuffled = torch.roll(heldout_programs, shifts=-1, dims=0)
    heldout_rows, heldout_generation = evaluate_generation(
        realizer, heldout_records, heldout_programs, heldout_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 2000,
        image_path=output / "qualitative_source_disjoint.png",
    )
    posterior = json.loads(Path(args.posterior_oracle_report).read_text())
    baseline = posterior["macro"]["posterior oracle"]
    direct_token = direct_controls["oracle block"]["token_nll_macro"]
    direct_histogram = direct_generation["oracle block"]["target_histogram_cosine"]
    token_improvement = (
        baseline["token_nll_macro"] - direct_token
    ) / baseline["token_nll_macro"]
    histogram_improvement = direct_histogram - baseline["target_histogram_cosine"]
    checks = {
        "direct_token_nll_improves_at_least_2pct_over_global_posterior": token_improvement >= 0.02,
        "direct_histogram_improves_at_least_002_over_global_posterior": histogram_improvement >= 0.02,
        "direct_oracle_beats_shuffled_and_null_token_nll": all(
            direct_token < direct_controls[arm]["token_nll_macro"]
            for arm in ("shuffled block", "null block")
        ),
        "direct_oracle_beats_shuffled_and_null_histogram": all(
            direct_histogram > direct_generation[arm]["target_histogram_cosine"]
            for arm in ("shuffled block", "null block")
        ),
        "source_disjoint_oracle_beats_shuffled_and_null_token_nll": all(
            validation_controls["oracle block"]["token_nll_macro"]
            < validation_controls[arm]["token_nll_macro"]
            for arm in ("shuffled block", "null block")
        ),
        "source_disjoint_oracle_beats_shuffled_and_null_histogram": all(
            heldout_generation["oracle block"]["target_histogram_cosine"]
            > heldout_generation[arm]["target_histogram_cosine"]
            for arm in ("shuffled block", "null block")
        ),
        "generated_centers_not_grid_locked": max(
            direct_generation["oracle block"]["cell_center_lock_fraction"],
            heldout_generation["oracle block"]["cell_center_lock_fraction"],
        ) < 0.01,
        "all_generated_renders_finite": all(
            math.isfinite(row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"])
            for row in direct_rows + heldout_rows for arm in ARMS
        ),
    }
    torch.save(
        {
            "realizer": realizer.state_dict(),
            "block_codebook": checkpoint["block_codebook"],
            "physical_codebook": args.codebook,
            "null_token": null_token,
            "seed": seed,
        },
        output / "realizer.pt",
    )
    report = {
        "schema": "predefined-medoid-block-constellation-oracle-v1",
        "protocol": {
            "block_checkpoint": args.block_checkpoint,
            "split_report": args.split_report,
            "block_vocabulary_size": 1024,
            "blocks_per_program": 256,
            "training_sources": [record.source_id for record in training],
            "validation_sources": sorted({record.source_id for record in validation}),
            "generation_jewels": args.generation_jewels,
            "oracle_uses_target_block_descriptors": True,
            "oracle_is_not_valid_prompt_inference": True,
            "sample_time_target_field": False,
            "one_complete_medoid_constellation_per_token": True,
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
            "direct_adjustment": adjustment_macro(direct_rows),
            "source_disjoint_adjustment": adjustment_macro(heldout_rows),
            "direct_records": direct_rows,
            "source_disjoint_records": heldout_rows,
        },
        "frozen_global_posterior_baseline": baseline,
        "improvement_over_global_posterior": {
            "token_nll_fraction": token_improvement,
            "histogram_cosine_absolute": histogram_improvement,
        },
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_constellation_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "fit": fit_report,
        "direct_controls": direct_controls,
        "validation_controls": validation_controls,
        "direct_generation": direct_generation,
        "heldout_generation": heldout_generation,
        "adjustment": {
            "direct": report["generation"]["direct_adjustment"],
            "source_disjoint": report["generation"]["source_disjoint_adjustment"],
        },
        "improvement_over_global_posterior": report["improvement_over_global_posterior"],
        "gate": report["gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
