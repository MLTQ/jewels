"""Audit whether irregular Jewel fields admit a stable casting vocabulary."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from sol.compare_field_structure import structure_report
from sol.jewel_casting_language import (
    CastProgram,
    MotifCodebook,
    decode_program,
    encode_program,
    fit_motif_codebook,
    histogram_cosine,
    program_histogram,
    quantize_centers_to_cells,
)
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class FieldRecord:
    path: str
    source_id: str
    style: str
    fit_seed: int
    features: torch.Tensor
    background: torch.Tensor


def _featurizer():
    stprim_root = Path(__file__).resolve().parents[1] / "stprim"
    if str(stprim_root) not in sys.path:
        sys.path.insert(0, str(stprim_root))
    from prior.featurize import features_to_field, state_to_features

    return state_to_features, features_to_field


def load_field_records(roots: list[Path]) -> list[FieldRecord]:
    """Load source-owned fitted fields while rejecting duplicate paths."""
    state_to_features, _ = _featurizer()
    paths = sorted({path.resolve() for root in roots for path in root.glob("*.pt")})
    if not paths:
        raise FileNotFoundError("casting audit roots contain no checkpoints")
    records = []
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if "state" not in saved or "source" not in saved:
            continue
        source = saved["source"]
        records.append(
            FieldRecord(
                path=str(path),
                source_id=str(source["source_id"]),
                style=str(source.get("style", "unknown")),
                fit_seed=int(saved.get("cfg", {}).get("seed", -1)),
                features=state_to_features(saved["state"]).float(),
                background=torch.as_tensor(saved["info"]["background"]).float(),
            )
        )
    if not records:
        raise ValueError("casting roots contain no fitted-field checkpoints")
    return records


def residual_metrics(program: CastProgram, codebook: MotifCodebook) -> dict[str, float | int]:
    """Measure how much standardized bundle energy remains outside the motif."""
    row = torch.arange(codebook.bundle_size, device=program.counts.device)
    valid = row[None] < program.counts[:, None]
    prototype = codebook.prototypes.to(program.residuals)[program.motifs]
    target = prototype + program.residuals
    residual_energy = program.residuals[valid].square().mean()
    target_energy = target[valid].square().mean().clamp_min(1e-12)
    return {
        "standardized_residual_mse": float(residual_energy),
        "standardized_target_energy": float(target_energy),
        "motif_explained_fraction": float(1.0 - residual_energy / target_energy),
        "casts": program.casts,
        "source_jewels": program.source_jewels,
        "serialized_jewels": int(program.counts.sum()),
        "jewels_per_cast": float(program.source_jewels / program.casts),
    }


def center_irregularity(features: torch.Tensor, spec: GridSpec) -> dict[str, float]:
    """Report within-cell spread and exact locking to addressed cell centers."""
    scaled = (features[:, :3] + 1.0) * 0.5 * features.new_tensor(spec.shape)
    fraction = torch.remainder(scaled, 1.0)
    center_distance = (fraction - 0.5).abs()
    return {
        "within_cell_fraction_std": float(fraction.std()),
        "cell_center_lock_fraction": float((center_distance < 1e-4).all(dim=1).float().mean()),
        "near_cell_center_fraction": float((center_distance < 0.02).all(dim=1).float().mean()),
    }


def _normalized_histogram(histogram: torch.Tensor) -> torch.Tensor:
    return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)


def pairwise_language_similarity(
    programs: list[tuple[str, CastProgram]],
    *,
    n_cells: int,
    vocabulary_size: int,
) -> dict:
    """Compare equivalent-source motif programs against mismatched controls."""
    histograms = [
        program_histogram(
            program, n_cells=n_cells, vocabulary_size=vocabulary_size
        )
        for _, program in programs
    ]
    pairs = []
    for first in range(len(programs)):
        for second in range(first + 1, len(programs)):
            source_first, _ = programs[first]
            source_second, _ = programs[second]
            raw = histogram_cosine(histograms[first], histograms[second])
            conditional = histogram_cosine(
                _normalized_histogram(histograms[first]),
                _normalized_histogram(histograms[second]),
            )
            occupancy = histogram_cosine(
                histograms[first].sum(dim=1), histograms[second].sum(dim=1)
            )
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "same_source": source_first == source_second,
                    "raw_motif_cosine": raw,
                    "cell_conditional_motif_cosine": conditional,
                    "occupancy_cosine": occupancy,
                }
            )
    same = [row for row in pairs if row["same_source"]]
    different = [row for row in pairs if not row["same_source"]]
    if not same or not different:
        raise ValueError("canonicality audit needs same-source and different-source pairs")

    def mean(rows: list[dict], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    summary = {}
    for key in (
        "raw_motif_cosine",
        "cell_conditional_motif_cosine",
        "occupancy_cosine",
    ):
        same_mean = mean(same, key)
        different_mean = mean(different, key)
        summary[key] = {
            "same_source": same_mean,
            "different_source": different_mean,
            "margin": same_mean - different_mean,
        }
    return {"pairs": pairs, "summary": summary}


def _render(
    features: torch.Tensor,
    points: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    _, features_to_field = _featurizer()
    from stprim.models.render import render_points

    field = features_to_field(features, device=points.device)
    with torch.no_grad():
        return render_points(
            field,
            points,
            cull_mode="support_tiled",
            support_sigma=5.0,
            support_capacity=16384,
            support_point_chunk=min(8192, len(points)),
            support_base_resolution=32,
            support_level_scale=1.55,
            background=background.to(points),
        )


def _psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = (reference - candidate).square().mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def _audit_candidate(
    source: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_render: torch.Tensor,
    points: torch.Tensor,
    background: torch.Tensor,
    spec: GridSpec,
) -> dict:
    rendered = _render(candidate, points, background)
    torch.manual_seed(0)
    source_structure = structure_report(source)
    torch.manual_seed(0)
    candidate_structure = structure_report(candidate)
    return {
        "voxel_psnr_to_continuous_source": _psnr(reference_render, rendered),
        # This is deliberately a marginal diagnostic: Jewel rows are a set and
        # primitive-to-primitive correspondence is not identifiable.
        "feature_marginal_mae": float(
            (source.sort(dim=0).values - candidate.sort(dim=0).values).abs().mean()
        ),
        "structure": candidate_structure,
        "mixed_tilt_retention": (
            candidate_structure["mixed_spacetime_tilt_median"]
            / max(source_structure["mixed_spacetime_tilt_median"], 1e-12)
        ),
        "center_irregularity": center_irregularity(candidate, spec),
    }


def _seed_for(record: FieldRecord) -> int:
    digest = hashlib.sha256(
        f"{record.source_id}:{record.fit_seed}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
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
    if args.voxel_points <= 0:
        raise ValueError("voxel points must be positive")
    device = torch.device(args.device)
    spec = GridSpec((8, 8, 4), slots_per_cell=1)
    records = load_field_records([Path(root) for root in args.root])
    validation_sources = set(args.validation_source)
    training = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    if not training or not validation:
        raise ValueError("casting split needs train and validation fields")
    validation_counts = {
        source: sum(row.source_id == source for row in validation)
        for source in validation_sources
    }
    if any(count < 2 for count in validation_counts.values()):
        raise ValueError("every validation source needs at least two independent fields")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train_fields = [record.features.to(device) for record in training]
    report = {
        "schema": "jewel-casting-language-gate-v0",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": sorted(validation_sources),
            "grid_shape": spec.shape,
            "bundle_size": args.bundle_size,
            "vocabulary_sizes": vocabulary_sizes,
            "iterations": args.iterations,
            "max_fit_casts": args.max_fit_casts,
            "voxel_points": args.voxel_points,
            "renderer": "support_tiled",
            "support_sigma": 5.0,
            "residual_scales": [0.0, 0.5, 1.0],
        },
        "vocabularies": {},
    }
    source_render_cache = {}
    grid_control_cache = {}
    for vocabulary_size in vocabulary_sizes:
        codebook, fit_report = fit_motif_codebook(
            train_fields,
            spec=spec,
            bundle_size=args.bundle_size,
            vocabulary_size=vocabulary_size,
            iterations=args.iterations,
            max_casts=args.max_fit_casts,
            assignment_chunk=512,
            seed=20260826,
        )
        torch.save(
            {
                "prototypes": codebook.prototypes.cpu(),
                "prototype_count_coordinates": codebook.prototype_count_coordinates.cpu(),
                "normalizer": {
                    "intrinsic_mean": codebook.normalizer.intrinsic_mean.cpu(),
                    "intrinsic_std": codebook.normalizer.intrinsic_std.cpu(),
                },
                "grid_shape": codebook.grid_shape,
                "bundle_size": codebook.bundle_size,
                "count_weight": codebook.count_weight,
            },
            output / f"codebook_k{vocabulary_size}.pt",
        )
        audit_rows = []
        programs = []
        for record in validation:
            features = record.features.to(device)
            program = encode_program(features, codebook)
            programs.append((record.source_id, program))
            cache_key = (record.source_id, record.fit_seed, record.path)
            if cache_key not in source_render_cache:
                generator = torch.Generator(device=device).manual_seed(_seed_for(record))
                points = torch.rand(
                    args.voxel_points, 3, generator=generator, device=device
                ) * 2.0 - 1.0
                source_render_cache[cache_key] = (
                    points,
                    _render(features, points, record.background),
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
                candidate = decode_program(
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
            audit_rows.append(
                {
                    "path": record.path,
                    "source_id": record.source_id,
                    "style": record.style,
                    "fit_seed": record.fit_seed,
                    "source_jewels": len(features),
                    "program": residual_metrics(program, codebook),
                    "source_structure": source_structure,
                    "source_center_irregularity": center_irregularity(features, spec),
                    "grid_center_control": grid_control_cache[cache_key],
                    "candidates": candidates,
                }
            )
            print(
                "audited", vocabulary_size, record.source_id, record.fit_seed,
                flush=True,
            )
        pairwise = pairwise_language_similarity(
            programs,
            n_cells=spec.n_cells,
            vocabulary_size=vocabulary_size,
        )

        def macro(path: tuple[str, ...]) -> float:
            values = []
            for row in audit_rows:
                value = row
                for key in path:
                    value = value[key]
                values.append(float(value))
            return sum(values) / len(values)

        macro_report = {
            "casts": macro(("program", "casts")),
            "jewels_per_cast": macro(("program", "jewels_per_cast")),
            "motif_explained_fraction": macro(("program", "motif_explained_fraction")),
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
            "canonicality": pairwise,
            "records": audit_rows,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        del codebook
        torch.cuda.empty_cache() if device.type == "cuda" else None

    selected = report["vocabularies"][str(vocabulary_sizes[-1])]
    canonical_margin = selected["canonicality"]["summary"][
        "cell_conditional_motif_cosine"
    ]["margin"]
    checks = {
        "all_jewels_serialized": all(
            row["source_jewels"]
            == row["program"]["source_jewels"]
            == row["program"]["serialized_jewels"]
            for row in selected["records"]
        ),
        "casts_at_most_12000": selected["macro"]["casts"] <= 12000,
        "motif_explains_at_least_25pct": selected["macro"]["motif_explained_fraction"] >= 0.25,
        "half_residual_at_least_30db": selected["macro"]["half_residual_voxel_psnr"] >= 30.0,
        "half_residual_tilt_within_10pct": 0.9 <= selected["macro"]["half_residual_mixed_tilt_retention"] <= 1.1,
        "token_centers_not_grid_locked": selected["macro"]["token_only_cell_center_lock_fraction"] < 0.01,
        "same_source_conditional_margin_at_least_005": canonical_margin >= 0.05,
        "full_residual_numerically_exact": selected["macro"]["full_residual_voxel_psnr"] >= 80.0,
    }
    report["gate"] = {"checks": checks, "passed": all(checks.values())}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "selected_macro": selected["macro"]}, indent=2))


if __name__ == "__main__":
    main()
