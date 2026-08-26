"""Audit the fresh-source hierarchical Jewel casting language gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.audit_factorized_jewel_casting_language import _save_codebook
from sol.audit_jewel_casting_language import (
    _audit_candidate,
    _render,
    _seed_for,
    center_irregularity,
    load_field_records,
)
from sol.compare_field_structure import structure_report
from sol.factorized_jewel_casting_language import (
    FactorizedCodebook,
    FactorizedProgram,
    decode_factorized_program,
    encode_factorized_program,
    factor_histograms,
    fit_factorized_codebook,
)
from sol.jewel_casting_language import histogram_cosine, quantize_centers_to_cells
from sol.token_grid import GridSpec


def compose_hierarchical_features(
    pair_features: torch.Tensor, individual_features: torch.Tensor
) -> torch.Tensor:
    """Take geometry from pair casts and appearance from individual casts."""
    if pair_features.shape != individual_features.shape:
        raise ValueError("hierarchical feature arms must share a canonical shape")
    result = pair_features.clone()
    result[:, 9:] = individual_features[:, 9:]
    return result


def hierarchical_decisions(
    pair_program: FactorizedProgram, individual_program: FactorizedProgram
) -> int:
    """Count only pair layout/covariance and individual surface/gradient tokens."""
    return pair_program.casts * 2 + individual_program.casts * 2


def _normalized(histogram: torch.Tensor) -> torch.Tensor:
    return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)


def hierarchical_histogram(
    pair_program: FactorizedProgram,
    individual_program: FactorizedProgram,
    *,
    n_cells: int,
    vocabulary_size: int,
) -> torch.Tensor:
    """Concatenate only the three content-bearing registered token roles."""
    pair = factor_histograms(
        pair_program, n_cells=n_cells, vocabulary_size=vocabulary_size
    )
    individual = factor_histograms(
        individual_program, n_cells=n_cells, vocabulary_size=vocabulary_size
    )
    return torch.cat(
        [
            _normalized(pair["covariance"]).flatten(),
            _normalized(individual["surface"]).flatten(),
            _normalized(individual["gradient"]).flatten(),
        ]
    )


def pairwise_hierarchical_similarity(
    programs: list[tuple[str, FactorizedProgram, FactorizedProgram]],
    *,
    n_cells: int,
    vocabulary_size: int,
) -> dict:
    """Compare fresh same-source hierarchical programs against unrelated controls."""
    histograms = [
        hierarchical_histogram(
            pair, individual, n_cells=n_cells, vocabulary_size=vocabulary_size
        )
        for _, pair, individual in programs
    ]
    pairs = []
    for first in range(len(programs)):
        for second in range(first + 1, len(programs)):
            value = histogram_cosine(histograms[first], histograms[second])
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "same_source": programs[first][0] == programs[second][0],
                    "cell_conditional_cosine": value,
                }
            )
    same = [row["cell_conditional_cosine"] for row in pairs if row["same_source"]]
    different = [row["cell_conditional_cosine"] for row in pairs if not row["same_source"]]
    if not same or not different:
        raise ValueError("hierarchical canonicality needs same and different source pairs")
    same_mean = sum(same) / len(same)
    different_mean = sum(different) / len(different)
    return {
        "pairs": pairs,
        "summary": {
            "same_source": same_mean,
            "different_source": different_mean,
            "margin": same_mean - different_mean,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--max-fit-casts", type=int, default=100000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device(args.device)
    spec = GridSpec((8, 8, 4), slots_per_cell=1)
    records = load_field_records([Path(root) for root in args.root])
    validation_sources = set(args.validation_source)
    training = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    validation_counts = {
        source: sum(row.source_id == source for row in validation)
        for source in validation_sources
    }
    if not training or any(count != 3 for count in validation_counts.values()):
        raise ValueError("hierarchical gate requires train fields and exactly three fresh fits per source")
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("hierarchical vocabulary training permits one field per source")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_fields = [record.features.to(device) for record in training]
    pair_codebook, pair_fit = fit_factorized_codebook(
        train_fields,
        spec=spec,
        bundle_size=2,
        vocabulary_size=1024,
        iterations=args.iterations,
        max_casts=args.max_fit_casts,
        assignment_chunk=512,
        seed=20260828,
    )
    individual_codebook, individual_fit = fit_factorized_codebook(
        train_fields,
        spec=spec,
        bundle_size=1,
        vocabulary_size=1024,
        iterations=args.iterations,
        max_casts=args.max_fit_casts,
        assignment_chunk=512,
        seed=20260829,
    )
    _save_codebook(pair_codebook, output / "codebook_pair_k1024.pt")
    _save_codebook(individual_codebook, output / "codebook_individual_k1024.pt")
    rows = []
    programs = []
    for record in validation:
        features = record.features.to(device)
        pair_program = encode_factorized_program(features, pair_codebook)
        individual_program = encode_factorized_program(features, individual_codebook)
        programs.append((record.source_id, pair_program, individual_program))
        generator = torch.Generator(device=device).manual_seed(_seed_for(record))
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2.0 - 1.0
        source_render = _render(features, points, record.background)
        candidates = {}
        for residual_scale in (0.0, 0.5, 1.0):
            pair_features = decode_factorized_program(
                pair_program, pair_codebook, residual_scale=residual_scale
            )
            individual_features = decode_factorized_program(
                individual_program,
                individual_codebook,
                residual_scale=residual_scale,
            )
            candidate = compose_hierarchical_features(
                pair_features, individual_features
            )
            candidates[str(residual_scale)] = _audit_candidate(
                features,
                candidate,
                reference_render=source_render,
                points=points,
                background=record.background,
                spec=spec,
            )
        grid_control = _audit_candidate(
            features,
            quantize_centers_to_cells(features, spec),
            reference_render=source_render,
            points=points,
            background=record.background,
            spec=spec,
        )
        torch.manual_seed(0)
        rows.append(
            {
                "path": record.path,
                "source_id": record.source_id,
                "style": record.style,
                "fit_seed": record.fit_seed,
                "source_jewels": len(features),
                "pair_casts": pair_program.casts,
                "individual_casts": individual_program.casts,
                "hierarchical_decisions": hierarchical_decisions(
                    pair_program, individual_program
                ),
                "pair_serialized_jewels": int(pair_program.counts.sum()),
                "individual_serialized_jewels": int(individual_program.counts.sum()),
                "source_structure": structure_report(features),
                "source_center_irregularity": center_irregularity(features, spec),
                "grid_center_control": grid_control,
                "candidates": candidates,
            }
        )
        print("audited", record.source_id, record.fit_seed, flush=True)
    canonicality = pairwise_hierarchical_similarity(
        programs, n_cells=spec.n_cells, vocabulary_size=1024
    )

    def macro(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    macro_report = {
        "pair_casts": macro(("pair_casts",)),
        "individual_casts": macro(("individual_casts",)),
        "hierarchical_decisions": macro(("hierarchical_decisions",)),
        "eight_frame_decisions": macro(("hierarchical_decisions",)) * 8 / 49,
        "token_only_voxel_psnr": macro(("candidates", "0.0", "voxel_psnr_to_continuous_source")),
        "half_residual_voxel_psnr": macro(("candidates", "0.5", "voxel_psnr_to_continuous_source")),
        "full_residual_voxel_psnr": macro(("candidates", "1.0", "voxel_psnr_to_continuous_source")),
        "token_only_mixed_tilt_retention": macro(("candidates", "0.0", "mixed_tilt_retention")),
        "half_residual_mixed_tilt_retention": macro(("candidates", "0.5", "mixed_tilt_retention")),
        "token_only_cell_center_lock_fraction": macro(("candidates", "0.0", "center_irregularity", "cell_center_lock_fraction")),
        "grid_control_voxel_psnr": macro(("grid_center_control", "voxel_psnr_to_continuous_source")),
    }
    checks = {
        "all_jewels_serialized": all(
            row["source_jewels"]
            == row["pair_serialized_jewels"]
            == row["individual_serialized_jewels"]
            for row in rows
        ),
        "full_residual_numerically_exact": macro_report["full_residual_voxel_psnr"] >= 80,
        "token_centers_not_grid_locked": macro_report["token_only_cell_center_lock_fraction"] < 0.01,
        "token_only_at_least_20db": macro_report["token_only_voxel_psnr"] >= 20,
        "half_residual_at_least_25db": macro_report["half_residual_voxel_psnr"] >= 25,
        "half_residual_tilt_within_10pct": 0.9 <= macro_report["half_residual_mixed_tilt_retention"] <= 1.1,
        "same_source_margin_at_least_005": canonicality["summary"]["margin"] >= 0.05,
        "eight_frame_decisions_at_most_40000": macro_report["eight_frame_decisions"] <= 40000,
    }
    report = {
        "schema": "hierarchical-jewel-casting-language-gate-v1",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": sorted(validation_sources),
            "grid_shape": spec.shape,
            "vocabulary_size_per_factor": 1024,
            "pair_bundle_size": 2,
            "individual_bundle_size": 1,
            "iterations": args.iterations,
            "max_fit_casts": args.max_fit_casts,
            "voxel_points": args.voxel_points,
            "residual_scales": [0.0, 0.5, 1.0],
        },
        "fit": {"pair": pair_fit, "individual": individual_fit},
        "macro": macro_report,
        "canonicality": canonicality,
        "records": rows,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "macro": macro_report,
                      "canonicality": canonicality["summary"]}, indent=2))


if __name__ == "__main__":
    main()
