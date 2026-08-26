"""Audit the preregistered compositional Jewel casting phrase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import (
    FieldRecord,
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
    composed_prototypes,
    decode_factorized_program,
    encode_factorized_program,
    factor_histograms,
    fit_factorized_codebook,
)
from sol.jewel_casting_language import histogram_cosine, quantize_centers_to_cells
from sol.token_grid import GridSpec


def factorized_residual_metrics(
    program: FactorizedProgram, codebook: FactorizedCodebook
) -> dict:
    """Measure token-owned energy overall and for each physical role."""
    row = torch.arange(codebook.bundle_size, device=program.counts.device)
    valid = row[None] < program.counts[:, None]
    prototype = composed_prototypes(program, codebook)
    target = prototype + program.residuals

    def energy(dimensions: tuple[int, ...]) -> dict[str, float]:
        factor_valid = valid[:, :, None].expand(
            -1, -1, len(dimensions)
        )
        residual = program.residuals[:, :, list(dimensions)][factor_valid]
        reference = target[:, :, list(dimensions)][factor_valid]
        residual_energy = residual.square().mean()
        target_energy = reference.square().mean().clamp_min(1e-12)
        return {
            "standardized_residual_mse": float(residual_energy),
            "standardized_target_energy": float(target_energy),
            "motif_explained_fraction": float(1.0 - residual_energy / target_energy),
        }

    all_dimensions = tuple(range(22))
    return {
        **energy(all_dimensions),
        "factors": {
            factor.name: energy(factor.dimensions) for factor in codebook.factors
        },
        "casts": program.casts,
        "discrete_decisions": program.discrete_decisions,
        "source_jewels": program.source_jewels,
        "serialized_jewels": int(program.counts.sum()),
        "jewels_per_cast": float(program.source_jewels / program.casts),
    }


def _cell_normalized(histogram: torch.Tensor) -> torch.Tensor:
    return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)


def pairwise_factorized_similarity(
    programs: list[tuple[str, FactorizedProgram]],
    *,
    n_cells: int,
    vocabulary_size: int,
) -> dict:
    """Compare same-source compositional programs with different-source controls."""
    histograms = [
        factor_histograms(
            program, n_cells=n_cells, vocabulary_size=vocabulary_size
        )
        for _, program in programs
    ]
    factor_names = tuple(histograms[0])
    pairs = []
    for first in range(len(programs)):
        for second in range(first + 1, len(programs)):
            first_source, first_program = programs[first]
            second_source, second_program = programs[second]
            factor_cosines = {
                name: histogram_cosine(
                    _cell_normalized(histograms[first][name]),
                    _cell_normalized(histograms[second][name]),
                )
                for name in factor_names
            }
            first_composite = torch.cat(
                [_cell_normalized(histograms[first][name]).flatten() for name in factor_names]
            )
            second_composite = torch.cat(
                [_cell_normalized(histograms[second][name]).flatten() for name in factor_names]
            )
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "same_source": first_source == second_source,
                    "composite_cell_conditional_cosine": histogram_cosine(
                        first_composite, second_composite
                    ),
                    "factor_cell_conditional_cosine": factor_cosines,
                    "occupancy_cosine": histogram_cosine(
                        torch.bincount(
                            first_program.cells,
                            weights=first_program.counts.float(),
                            minlength=n_cells,
                        ),
                        torch.bincount(
                            second_program.cells,
                            weights=second_program.counts.float(),
                            minlength=n_cells,
                        ),
                    ),
                }
            )
    same = [row for row in pairs if row["same_source"]]
    different = [row for row in pairs if not row["same_source"]]
    if not same or not different:
        raise ValueError("factorized canonicality needs same and different source pairs")

    def summarize(values_same: list[float], values_different: list[float]) -> dict:
        same_mean = sum(values_same) / len(values_same)
        different_mean = sum(values_different) / len(values_different)
        return {
            "same_source": same_mean,
            "different_source": different_mean,
            "margin": same_mean - different_mean,
        }

    summary = {
        "composite_cell_conditional_cosine": summarize(
            [row["composite_cell_conditional_cosine"] for row in same],
            [row["composite_cell_conditional_cosine"] for row in different],
        ),
        "occupancy_cosine": summarize(
            [row["occupancy_cosine"] for row in same],
            [row["occupancy_cosine"] for row in different],
        ),
        "factors": {
            name: summarize(
                [row["factor_cell_conditional_cosine"][name] for row in same],
                [row["factor_cell_conditional_cosine"][name] for row in different],
            )
            for name in factor_names
        },
    }
    return {"pairs": pairs, "summary": summary}


def _save_codebook(codebook: FactorizedCodebook, path: Path) -> None:
    torch.save(
        {
            "factors": {
                factor.name: {
                    "dimensions": factor.dimensions,
                    "prototypes": factor.prototypes.cpu(),
                    "prototype_count_coordinates": factor.prototype_count_coordinates.cpu(),
                }
                for factor in codebook.factors
            },
            "normalizer": {
                "intrinsic_mean": codebook.normalizer.intrinsic_mean.cpu(),
                "intrinsic_std": codebook.normalizer.intrinsic_std.cpu(),
            },
            "grid_shape": codebook.grid_shape,
            "bundle_size": codebook.bundle_size,
            "count_weight": codebook.count_weight,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--joint-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocabulary-size", action="append", type=int, default=[])
    parser.add_argument("--bundle-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--max-fit-casts", type=int, default=100000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    args = parser.parse_args()
    vocabulary_sizes = args.vocabulary_size or [64, 256, 1024]
    if sorted(set(vocabulary_sizes)) != vocabulary_sizes:
        raise ValueError("vocabulary sizes must be unique and increasing")
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
    if not training or any(count < 2 for count in validation_counts.values()):
        raise ValueError("factorized split needs train fields and two fits per validation source")
    joint_report = json.loads(Path(args.joint_report).read_text())
    joint_baseline = joint_report["vocabularies"]["1024"]["macro"]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_fields = [record.features.to(device) for record in training]
    report = {
        "schema": "factorized-jewel-casting-language-gate-v1",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": sorted(validation_sources),
            "grid_shape": spec.shape,
            "bundle_size": args.bundle_size,
            "vocabulary_sizes_per_factor": vocabulary_sizes,
            "factor_names": ["layout", "covariance", "surface", "gradient"],
            "iterations": args.iterations,
            "max_fit_casts": args.max_fit_casts,
            "voxel_points": args.voxel_points,
            "residual_scales": [0.0, 0.5, 1.0],
            "joint_baseline_report": args.joint_report,
        },
        "joint_baseline_macro": joint_baseline,
        "vocabularies": {},
    }
    source_render_cache = {}
    grid_control_cache = {}
    for vocabulary_size in vocabulary_sizes:
        codebook, fit_report = fit_factorized_codebook(
            train_fields,
            spec=spec,
            bundle_size=args.bundle_size,
            vocabulary_size=vocabulary_size,
            iterations=args.iterations,
            max_casts=args.max_fit_casts,
            assignment_chunk=512,
            seed=20260827,
        )
        _save_codebook(codebook, output / f"codebook_factorized_k{vocabulary_size}.pt")
        rows = []
        programs = []
        for record in validation:
            features = record.features.to(device)
            program = encode_factorized_program(features, codebook)
            programs.append((record.source_id, program))
            cache_key = (record.source_id, record.fit_seed, record.path)
            if cache_key not in source_render_cache:
                generator = torch.Generator(device=device).manual_seed(_seed_for(record))
                points = torch.rand(
                    args.voxel_points, 3, generator=generator, device=device
                ) * 2.0 - 1.0
                source_render_cache[cache_key] = (
                    points, _render(features, points, record.background)
                )
            points, source_render = source_render_cache[cache_key]
            if cache_key not in grid_control_cache:
                grid_control_cache[cache_key] = _audit_candidate(
                    features,
                    quantize_centers_to_cells(features, spec),
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                )
            candidates = {}
            for residual_scale in (0.0, 0.5, 1.0):
                candidate = decode_factorized_program(
                    program, codebook, residual_scale=residual_scale
                )
                candidates[str(residual_scale)] = _audit_candidate(
                    features,
                    candidate,
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                )
            torch.manual_seed(0)
            source_structure = structure_report(features)
            rows.append(
                {
                    "path": record.path,
                    "source_id": record.source_id,
                    "style": record.style,
                    "fit_seed": record.fit_seed,
                    "source_jewels": len(features),
                    "program": factorized_residual_metrics(program, codebook),
                    "source_structure": source_structure,
                    "source_center_irregularity": center_irregularity(features, spec),
                    "grid_center_control": grid_control_cache[cache_key],
                    "candidates": candidates,
                }
            )
            print("audited", vocabulary_size, record.source_id, record.fit_seed, flush=True)
        canonicality = pairwise_factorized_similarity(
            programs, n_cells=spec.n_cells, vocabulary_size=vocabulary_size
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
            "casts": macro(("program", "casts")),
            "discrete_decisions": macro(("program", "discrete_decisions")),
            "jewels_per_cast": macro(("program", "jewels_per_cast")),
            "motif_explained_fraction": macro(("program", "motif_explained_fraction")),
            "factor_explained_fraction": {
                name: macro(("program", "factors", name, "motif_explained_fraction"))
                for name in report["protocol"]["factor_names"]
            },
            "token_only_voxel_psnr": macro(("candidates", "0.0", "voxel_psnr_to_continuous_source")),
            "half_residual_voxel_psnr": macro(("candidates", "0.5", "voxel_psnr_to_continuous_source")),
            "full_residual_voxel_psnr": macro(("candidates", "1.0", "voxel_psnr_to_continuous_source")),
            "token_only_mixed_tilt_retention": macro(("candidates", "0.0", "mixed_tilt_retention")),
            "half_residual_mixed_tilt_retention": macro(("candidates", "0.5", "mixed_tilt_retention")),
            "token_only_cell_center_lock_fraction": macro(("candidates", "0.0", "center_irregularity", "cell_center_lock_fraction")),
            "grid_control_voxel_psnr": macro(("grid_center_control", "voxel_psnr_to_continuous_source")),
            "grid_control_cell_center_lock_fraction": macro(("grid_center_control", "center_irregularity", "cell_center_lock_fraction")),
        }
        report["vocabularies"][str(vocabulary_size)] = {
            "fit": fit_report,
            "macro": macro_report,
            "canonicality": canonicality,
            "records": rows,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        del codebook
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = report["vocabularies"][str(vocabulary_sizes[-1])]
    macro = selected["macro"]
    margin = selected["canonicality"]["summary"][
        "composite_cell_conditional_cosine"
    ]["margin"]
    checks = {
        "all_jewels_serialized": all(
            row["source_jewels"]
            == row["program"]["source_jewels"]
            == row["program"]["serialized_jewels"]
            for row in selected["records"]
        ),
        "casts_at_most_12000": macro["casts"] <= 12000,
        "discrete_decisions_at_most_40000": macro["discrete_decisions"] <= 40000,
        "phrase_explains_at_least_35pct": macro["motif_explained_fraction"] >= 0.35,
        "token_only_improves_joint_by_2db": (
            macro["token_only_voxel_psnr"] - joint_baseline["token_only_voxel_psnr"] >= 2.0
        ),
        "half_residual_at_least_20db": macro["half_residual_voxel_psnr"] >= 20.0,
        "half_residual_improves_joint_by_3db": (
            macro["half_residual_voxel_psnr"] - joint_baseline["half_residual_voxel_psnr"] >= 3.0
        ),
        "half_residual_tilt_within_15pct": 0.85 <= macro["half_residual_mixed_tilt_retention"] <= 1.15,
        "token_centers_not_grid_locked": macro["token_only_cell_center_lock_fraction"] < 0.01,
        "same_source_composite_margin_at_least_005": margin >= 0.05,
        "full_residual_numerically_exact": macro["full_residual_voxel_psnr"] >= 80.0,
    }
    report["gate"] = {"checks": checks, "passed": all(checks.values())}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "selected_macro": macro}, indent=2))


if __name__ == "__main__":
    main()
