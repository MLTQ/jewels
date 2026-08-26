"""Fit the preregistered fine-routing K=1024 block language without a neural decoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import load_field_records
from sol.block_token_language import (
    block_serialization_order,
    encode_block_tokens,
    fit_block_token_codebook,
)
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


def validate_fine_language_settings(
    block_shape: tuple[int, int, int], vocabulary_size: int
) -> None:
    if block_shape != (16, 16, 8) or vocabulary_size != 1024:
        raise ValueError("Gate 2a6 is frozen to 16x16x8 routing and K=1024")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-shape", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--block-vocabulary-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    block_shape = tuple(args.block_shape)
    validate_fine_language_settings(block_shape, args.block_vocabulary_size)
    if args.iterations != 20:
        raise ValueError("Gate 2a6 is frozen to 20 Lloyd iterations")
    device = torch.device(args.device)
    seed = 20260909
    split = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split["roots"]])
    training, validation = select_prompt_splits(
        records,
        set(split["validation_sources"]),
        set(split["training_sources"]),
    )
    training = sorted(training, key=lambda record: record.source_id)
    if len(training) != 18 or len(validation) != 9:
        raise ValueError("Gate 2a6 requires the immutable 18/9 source split")
    physical_codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(block_shape, slots_per_cell=1)
    fields = [record.features.to(device) for record in training]
    print("fitting fine block vocabulary", spec.shape, args.block_vocabulary_size, flush=True)
    codebook, fit_report = fit_block_token_codebook(
        fields,
        normalizer=physical_codebook.normalizer,
        spec=spec,
        vocabulary_size=args.block_vocabulary_size,
        iterations=args.iterations,
        seed=seed,
    )
    programs, assignment_distances = [], []
    for record, field in zip(training, fields):
        program, distances = encode_block_tokens(field, codebook)
        programs.append(program)
        assignment_distances.append(float(distances.mean()))
        print("encoded fine program", record.source_id, flush=True)
    programs = torch.stack(programs)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": {
                "block_vocabulary_size": args.block_vocabulary_size,
                "block_shape": block_shape,
                "conditioning": "fine_block_language_without_neural_decoder",
            },
            "block_codebook": codebook.state_dict(),
            "training_programs": programs.cpu(),
            "training_sources": [record.source_id for record in training],
            "serialization_order": block_serialization_order(spec),
            "physical_codebook": args.codebook,
            "seed": seed,
        },
        output / "language.pt",
    )
    report = {
        "schema": "fine-routing-block-token-language-v1",
        "protocol": {
            "split_report": args.split_report,
            "training_sources": [record.source_id for record in training],
            "validation_sources": sorted({record.source_id for record in validation}),
            "block_shape": list(block_shape),
            "blocks_per_program": spec.n_cells,
            "block_vocabulary_size": args.block_vocabulary_size,
            "iterations": args.iterations,
            "serialization": "time_major_xy_morton",
            "decisions_per_49_frame_window": 1 + spec.n_cells,
            "decisions_per_8_frames": (1 + spec.n_cells) * 8 / 49,
            "gate0f_active_role_decisions_per_49_frames": 216000,
            "decision_reduction_fraction": 1 - (1 + spec.n_cells) / 216000,
        },
        "fit": fit_report,
        "assignment_distance": {
            "per_training_source": assignment_distances,
            "mean": sum(assignment_distances) / len(assignment_distances),
        },
        "checks": {
            "all_training_blocks_serialized": programs.shape == (18, spec.n_cells),
            "program_tokens_in_range": int(programs.max()) < args.block_vocabulary_size,
            "finite_assignment_distances": all(math.isfinite(value) for value in assignment_distances),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
