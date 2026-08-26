"""Audit the active one-Jewel-plus-centroid language on fresh fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import (
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.factorized_jewel_casting_language import (
    FactorizedProgram,
    decode_factorized_program,
    encode_factorized_program,
    factor_histograms,
    load_factorized_codebook,
)
from sol.jewel_casting_language import histogram_cosine
from sol.token_grid import GridSpec


ACTIVE_FACTORS = ("covariance", "surface", "gradient")


def active_individual_histogram(
    program: FactorizedProgram, *, n_cells: int, vocabulary_size: int
) -> torch.Tensor:
    """Concatenate only nonconstant individual-Jewel factor histograms."""
    histograms = factor_histograms(
        program, n_cells=n_cells, vocabulary_size=vocabulary_size
    )
    output = []
    for name in ACTIVE_FACTORS:
        histogram = histograms[name]
        normalized = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)
        output.append(normalized.flatten())
    return torch.cat(output)


def pairwise_active_similarity(
    programs: list[tuple[str, FactorizedProgram]],
    *,
    n_cells: int,
    vocabulary_size: int,
) -> dict:
    """Compare active individual languages for same and different sources."""
    histograms = [
        active_individual_histogram(
            program, n_cells=n_cells, vocabulary_size=vocabulary_size
        )
        for _, program in programs
    ]
    pairs = []
    for first in range(len(programs)):
        for second in range(first + 1, len(programs)):
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "same_source": programs[first][0] == programs[second][0],
                    "cell_conditional_cosine": histogram_cosine(
                        histograms[first], histograms[second]
                    ),
                }
            )
    same = [row["cell_conditional_cosine"] for row in pairs if row["same_source"]]
    different = [row["cell_conditional_cosine"] for row in pairs if not row["same_source"]]
    if not same or not different:
        raise ValueError("active individual audit needs same and different source pairs")
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
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-points", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device(args.device)
    codebook = load_factorized_codebook(args.codebook, device)
    if codebook.bundle_size != 1:
        raise ValueError("active individual audit requires a bundle-1 codebook")
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    validation_sources = set(args.validation_source)
    records = [
        record
        for record in load_field_records([Path(root) for root in args.root])
        if record.source_id in validation_sources
    ]
    counts = {
        source: sum(record.source_id == source for record in records)
        for source in validation_sources
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError("active individual audit requires three fresh fits per source")
    rows = []
    programs = []
    for record in records:
        features = record.features.to(device)
        program = encode_factorized_program(features, codebook)
        programs.append((record.source_id, program))
        generator = torch.Generator(device=device).manual_seed(_seed_for(record))
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2.0 - 1.0
        source_render = _render(features, points, record.background)
        token_features = decode_factorized_program(
            program, codebook, residual_scale=0.0
        )
        full_features = decode_factorized_program(
            program, codebook, residual_scale=1.0
        )
        rows.append(
            {
                "path": record.path,
                "source_id": record.source_id,
                "style": record.style,
                "fit_seed": record.fit_seed,
                "source_jewels": len(features),
                "serialized_jewels": int(program.counts.sum()),
                "active_role_decisions": program.casts * len(ACTIVE_FACTORS),
                "token_only": _audit_candidate(
                    features,
                    token_features,
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                ),
                "full_residual_audit": _audit_candidate(
                    features,
                    full_features,
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                ),
            }
        )
        print("audited", record.source_id, record.fit_seed, flush=True)
    canonicality = pairwise_active_similarity(
        programs, n_cells=spec.n_cells, vocabulary_size=codebook.vocabulary_size
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
        "token_only_voxel_psnr": macro(("token_only", "voxel_psnr_to_continuous_source")),
        "token_only_mixed_tilt_retention": macro(("token_only", "mixed_tilt_retention")),
        "token_only_cell_center_lock_fraction": macro(("token_only", "center_irregularity", "cell_center_lock_fraction")),
        "full_residual_voxel_psnr": macro(("full_residual_audit", "voxel_psnr_to_continuous_source")),
        "active_role_decisions": macro(("active_role_decisions",)),
        "eight_frame_decisions": macro(("active_role_decisions",)) * 8 / 49,
    }
    checks = {
        "all_jewels_serialized": all(
            row["source_jewels"] == row["serialized_jewels"] for row in rows
        ),
        "token_render_at_least_20db": macro_report["token_only_voxel_psnr"] >= 20,
        "token_tilt_within_15pct": 0.85 <= macro_report["token_only_mixed_tilt_retention"] <= 1.15,
        "active_same_source_margin_at_least_005": canonicality["summary"]["margin"] >= 0.05,
        "token_centers_not_grid_locked": macro_report["token_only_cell_center_lock_fraction"] < 0.01,
        "full_residual_numerically_exact": macro_report["full_residual_voxel_psnr"] >= 80,
        "eight_frame_decisions_at_most_40000": macro_report["eight_frame_decisions"] <= 40000,
    }
    report = {
        "schema": "active-individual-jewel-language-gate-v1",
        "protocol": {
            "roots": args.root,
            "validation_sources": sorted(validation_sources),
            "validation_fields": len(records),
            "codebook": args.codebook,
            "vocabulary_size_per_factor": codebook.vocabulary_size,
            "active_factors": ACTIVE_FACTORS,
            "voxel_points": args.voxel_points,
            "exact_centroid": True,
            "exact_residual_emitted": False,
        },
        "macro": macro_report,
        "canonicality": canonicality,
        "records": rows,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "macro": macro_report,
                      "canonicality": canonicality["summary"]}, indent=2))


if __name__ == "__main__":
    main()
