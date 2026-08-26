"""Audit predefined macro-Jewel realization for the frozen K=256 block language."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch

from sol.audit_jewel_casting_language import (
    FieldRecord,
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.audit_scene_posterior_oracle import select_oracle_sources
from sol.block_token_empirical_realizer import fit_empirical_block_realizer
from sol.block_token_language import BlockTokenCodebook, encode_block_tokens
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.jewel_casting_language import histogram_cosine
from sol.prompt_jewel_caster import (
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec
from sol.train_block_token_oracle import ARMS, cyclic_shuffled_programs
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits
from sol.train_prompt_jewel_caster import _metadata


def empirical_control_metrics(
    realizer,
    records: list[FieldRecord],
    programs: torch.Tensor,
    shuffled_programs: torch.Tensor,
    null_token: int,
    *,
    physical_codebook,
    device: torch.device,
) -> dict:
    """Measure macro-token empirical likelihood under matched program controls."""
    accumulators = {
        arm: {name: [] for name in ("covariance", "surface", "gradient")}
        for arm in ARMS
    }
    for owner, record in enumerate(records):
        features = record.features.to(device)
        target_tokens = encode_active_jewel_tokens(features, physical_codebook)
        arm_programs = {
            "oracle block": programs[owner],
            "shuffled block": shuffled_programs[owner],
            "null block": torch.full_like(programs[owner], null_token),
        }
        for arm, program in arm_programs.items():
            row = realizer.token_nll(program, features[:, :3], target_tokens)
            for name, value in row["token_nll"].items():
                accumulators[arm][name].append(value)
    output = {}
    for arm in ARMS:
        by_role = {
            name: sum(values) / len(values)
            for name, values in accumulators[arm].items()
        }
        output[arm] = {
            "token_nll": by_role,
            "token_nll_macro": sum(by_role.values()) / len(by_role),
        }
    return output


def _generation_macro(rows: list[dict]) -> dict:
    def mean(arm: str, path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value = row["arms"][arm]
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    return {
        arm: {
            "target_histogram_cosine": mean(arm, ("target_histogram_cosine",)),
            "voxel_psnr_diagnostic": mean(
                arm, ("audit", "voxel_psnr_to_continuous_source")
            ),
            "cell_center_lock_fraction": mean(
                arm, ("audit", "center_irregularity", "cell_center_lock_fraction")
            ),
        }
        for arm in ARMS
    }


def realizer_device(realizer) -> torch.device:
    """Resolve the storage device shared by empirical and constellation realizers."""
    if hasattr(realizer, "phrase_local_centers"):
        return realizer.phrase_local_centers.device
    if hasattr(realizer, "local_centers"):
        return realizer.local_centers.device
    raise TypeError("unsupported block realizer")


@torch.no_grad()
def evaluate_generation(
    realizer,
    records: list[FieldRecord],
    programs: torch.Tensor,
    shuffled_programs: torch.Tensor,
    null_token: int,
    *,
    physical_codebook,
    generation_jewels: int,
    voxel_points: int,
    frames: int,
    height: int,
    width: int,
    seed: int,
    image_path: Path,
) -> tuple[list[dict], dict]:
    """Render matched oracle/shuffled/null empirical macro-token programs."""
    device = realizer_device(realizer)
    spec = GridSpec(realizer.block_shape, slots_per_cell=1)
    frame_indices = torch.linspace(0, frames - 1, 3).round().long()
    render_points = frame_points(frames, frame_indices, height, width, device=device)
    rows, image_rows = [], []
    for owner, record in enumerate(records):
        target = record.features.to(device)
        target_tokens = encode_active_jewel_tokens(target, physical_codebook)
        target_histogram = active_cell_histogram(
            target[:, :3], target_tokens, spec=spec,
            vocabulary_size=physical_codebook.vocabulary_size,
        )
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        points = torch.rand(
            voxel_points, 3, generator=generator, device=device
        ) * 2 - 1
        reference_render = _render(target, points, record.background)
        arm_programs = {
            "oracle block": programs[owner],
            "shuffled block": shuffled_programs[owner],
            "null block": torch.full_like(programs[owner], null_token),
        }
        arms, candidates = {}, {"target": target}
        for arm in ARMS:
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 10000 + _seed_for(record)
            )
            sampled = realizer.sample(
                arm_programs[arm], generation_jewels, generator=arm_generator
            )
            if len(sampled) == 2:
                centers, tokens = sampled
                realization = {}
            else:
                centers, tokens, realization = sampled
            features = active_tokens_to_features(centers, tokens, physical_codebook)
            candidates[arm] = features
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=physical_codebook.vocabulary_size,
            )
            arms[arm] = {
                "target_histogram_cosine": histogram_cosine(
                    target_histogram, histogram
                ),
                "realization": realization,
                "audit": _audit_candidate(
                    target, features, reference_render=reference_render,
                    points=points, background=record.background, spec=spec,
                ),
            }
        rows.append({"source_id": record.source_id, "fit_seed": record.fit_seed, "arms": arms})
        label = _metadata(record.path)["style"]
        for arm in ("target",) + ARMS:
            rendered = _render(
                candidates[arm], render_points, record.background
            ).reshape(3, height, width, 3)
            image_rows.append(_row([
                _panel(frame, f"{label} / {arm} / t={int(index)}")
                for frame, index in zip(rendered, frame_indices)
            ]))
        print("rendered empirical block oracle", record.source_id, flush=True)
    sheet = Image.new(
        "RGB",
        (image_rows[0].width,
         sum(row.height for row in image_rows) + 3 * (len(image_rows) - 1)),
        "white",
    )
    offset = 0
    for row in image_rows:
        sheet.paste(row, (0, offset))
        offset += row.height + 3
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path)
    return rows, _generation_macro(rows)


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
    parser.add_argument("--block-vocabulary-size", type=int, default=256)
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
        raise ValueError("empirical block audit render settings must be positive")
    device = torch.device(args.device)
    seed = 20260906
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if args.block_vocabulary_size not in (256, 1024):
        raise ValueError("empirical block audit permits only frozen K=256 or K=1024")
    if int(architecture["block_vocabulary_size"]) != args.block_vocabulary_size:
        raise ValueError("requested block vocabulary does not match checkpoint")
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
        raise ValueError("block checkpoint programs do not align with training sources")
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_empirical_block_realizer(
        [record.features.to(device) for record in training],
        training_programs,
        physical_codebook=physical_codebook,
        block_vocabulary_size=args.block_vocabulary_size,
        smoothing=0.1,
        jitter_std=0.01,
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
    histogram_improvement = (
        direct_histogram - baseline["target_histogram_cosine"]
    )
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
        "schema": "empirical-macro-jewel-block-oracle-v1",
        "protocol": {
            "block_checkpoint": args.block_checkpoint,
            "split_report": args.split_report,
            "block_vocabulary_size": args.block_vocabulary_size,
            "blocks_per_program": 256,
            "training_sources": [record.source_id for record in training],
            "validation_sources": sorted({record.source_id for record in validation}),
            "generation_jewels": args.generation_jewels,
            "oracle_uses_target_block_descriptors": True,
            "oracle_is_not_valid_prompt_inference": True,
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
        "frozen_global_posterior_baseline": baseline,
        "improvement_over_global_posterior": {
            "token_nll_fraction": token_improvement,
            "histogram_cosine_absolute": histogram_improvement,
        },
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_locality_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "fit": fit_report,
        "direct_controls": direct_controls,
        "validation_controls": validation_controls,
        "direct_generation": direct_generation,
        "heldout_generation": heldout_generation,
        "improvement_over_global_posterior": report["improvement_over_global_posterior"],
        "gate": report["gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
