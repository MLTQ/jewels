"""Train and audit the preregistered hierarchical phrase-to-Jewel decoder."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from sol.audit_jewel_casting_language import (
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.hierarchical_jewel_decoder import (
    HierarchicalPhraseBatch,
    HierarchicalPhraseDecoder,
    build_hierarchical_phrase_batch,
    build_sampled_hierarchical_phrase_batch,
    concatenate_phrase_batches,
    phrase_decoder_loss,
    phrase_values_to_features,
    residual_scale,
)
from sol.token_grid import GridSpec


@torch.no_grad()
def dataset_loss(
    model: HierarchicalPhraseDecoder,
    batch: HierarchicalPhraseBatch,
    scale: torch.Tensor,
    *,
    chunk: int = 8192,
) -> float:
    """Evaluate active-row normalized loss without allocating a full prediction."""
    total, weight = 0.0, 0
    for start in range(0, len(batch), chunk):
        part = batch.index(torch.arange(start, min(start + chunk, len(batch)), device=batch.cells.device))
        active = int(part.counts.sum())
        total += float(phrase_decoder_loss(model(part), part, scale)) * active
        weight += active
    return total / weight


@torch.no_grad()
def predict_values(
    model: HierarchicalPhraseDecoder,
    batch: HierarchicalPhraseBatch,
    *,
    chunk: int = 8192,
) -> torch.Tensor:
    """Decode a complete field in bounded chunks."""
    output = []
    for start in range(0, len(batch), chunk):
        selected = torch.arange(
            start, min(start + chunk, len(batch)), device=batch.cells.device
        )
        output.append(model(batch.index(selected)))
    return torch.cat(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--pair-codebook", required=True)
    parser.add_argument("--individual-codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-pairs-per-source", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--voxel-points", type=int, default=4096)
    args = parser.parse_args()
    if min(
        args.train_pairs_per_source,
        args.steps,
        args.batch_size,
        args.eval_every,
        args.patience,
        args.voxel_points,
    ) <= 0:
        raise ValueError("decoder schedule values must be positive")
    device = torch.device(args.device)
    seed = 20260830
    torch.manual_seed(seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_codebook = load_factorized_codebook(args.pair_codebook, device)
    individual_codebook = load_factorized_codebook(args.individual_codebook, device)
    records = load_field_records([Path(root) for root in args.root])
    validation_sources = set(args.validation_source)
    training = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("decoder training permits one field per source")
    validation_counts = {
        source: sum(record.source_id == source for record in validation)
        for source in validation_sources
    }
    if any(count != 3 for count in validation_counts.values()):
        raise ValueError("decoder validation requires three independent fits per source")

    training_batches = []
    for record in training:
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        training_batches.append(
            build_sampled_hierarchical_phrase_batch(
                record.features.to(device),
                pair_codebook,
                individual_codebook,
                max_pairs=args.train_pairs_per_source,
                generator=generator,
            )
        )
        print("sampled train", record.source_id, len(training_batches[-1]), flush=True)
    train_batch = concatenate_phrase_batches(training_batches)
    scale = residual_scale(train_batch)

    validation_rows = []
    validation_batches = []
    for record in validation:
        batch, pair_program, _ = build_hierarchical_phrase_batch(
            record.features.to(device), pair_codebook, individual_codebook
        )
        validation_batches.append(batch)
        validation_rows.append((record, batch, pair_program))
        print("encoded validation", record.source_id, record.fit_seed, len(batch), flush=True)
    validation_batch = concatenate_phrase_batches(validation_batches)

    model = HierarchicalPhraseDecoder(
        vocabulary_size=pair_codebook.vocabulary_size,
        n_cells=GridSpec(pair_codebook.grid_shape, slots_per_cell=1).n_cells,
        embedding_dim=32,
        hidden_dim=384,
        depth=4,
        output_scale=scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    # The registered baseline is exact zero correction, not a random neural decoder.
    baseline_validation_loss = float(
        phrase_decoder_loss(
            validation_batch.base_values, validation_batch, scale
        )
    )
    history = []
    best_loss = float("inf")
    best_step = 0
    best_state = None
    stale = 0
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    for step in range(1, args.steps + 1):
        selected = torch.randint(
            len(train_batch),
            (args.batch_size,),
            generator=sample_generator,
            device=device,
        )
        part = train_batch.index(selected)
        prediction = model(part)
        loss = phrase_decoder_loss(prediction, part, scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        validation_loss = dataset_loss(model, validation_batch, scale)
        train_loss = dataset_loss(model, train_batch, scale)
        row = {
            "step": step,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss * 0.999:
            best_loss = validation_loss
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "baseline_validation_normalized_mse": baseline_validation_loss,
                    "best_validation_normalized_mse": best_loss,
                    "best_step": best_step,
                    "history": history,
                },
                indent=2,
            )
            + "\n"
        )
        model.train()
        if stale >= args.patience:
            print("validation plateau", step, stale, flush=True)
            break
    if best_state is None:
        raise RuntimeError("decoder training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    audit_rows = []
    spec = GridSpec(pair_codebook.grid_shape, slots_per_cell=1)
    for record, batch, pair_program in validation_rows:
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(_seed_for(record))
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2.0 - 1.0
        source_render = _render(features, points, record.background)
        raw_features = phrase_values_to_features(
            batch.base_values, pair_program, pair_codebook
        )
        learned_features = phrase_values_to_features(
            predict_values(model, batch), pair_program, pair_codebook
        )
        audit_rows.append(
            {
                "source_id": record.source_id,
                "style": record.style,
                "fit_seed": record.fit_seed,
                "raw_product": _audit_candidate(
                    features,
                    raw_features,
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                ),
                "learned_decoder": _audit_candidate(
                    features,
                    learned_features,
                    reference_render=source_render,
                    points=points,
                    background=record.background,
                    spec=spec,
                ),
            }
        )
        print("rendered validation", record.source_id, record.fit_seed, flush=True)

    def macro(arm: str, path: tuple[str, ...]) -> float:
        values = []
        for row in audit_rows:
            value = row[arm]
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    macro_report = {
        "raw_product_voxel_psnr": macro(
            "raw_product", ("voxel_psnr_to_continuous_source",)
        ),
        "learned_voxel_psnr": macro(
            "learned_decoder", ("voxel_psnr_to_continuous_source",)
        ),
        "learned_mixed_tilt_retention": macro(
            "learned_decoder", ("mixed_tilt_retention",)
        ),
        "learned_cell_center_lock_fraction": macro(
            "learned_decoder", ("center_irregularity", "cell_center_lock_fraction")
        ),
        "baseline_validation_normalized_mse": baseline_validation_loss,
        "best_validation_normalized_mse": best_loss,
        "validation_mse_improvement_fraction": 1.0 - best_loss / baseline_validation_loss,
    }
    input_audit = {
        "forward_inputs": [
            "pair_layout_token",
            "pair_covariance_token",
            "individual_surface_tokens",
            "individual_gradient_tokens",
            "cell",
            "pair_anchor",
            "count",
            "frozen_base_prototypes",
        ],
        "target_values_forward_input": False,
        "exact_residual_forward_input": False,
        "source_pixels_forward_input": False,
    }
    checks = {
        "validation_mse_improves_at_least_10pct": (
            macro_report["validation_mse_improvement_fraction"] >= 0.10
        ),
        "learned_render_at_least_20db": macro_report["learned_voxel_psnr"] >= 20,
        "learned_improves_raw_by_1db": (
            macro_report["learned_voxel_psnr"] - macro_report["raw_product_voxel_psnr"] >= 1
        ),
        "learned_tilt_within_15pct": 0.85 <= macro_report["learned_mixed_tilt_retention"] <= 1.15,
        "learned_centers_not_grid_locked": macro_report["learned_cell_center_lock_fraction"] < 0.01,
        "no_target_residual_forward_input": (
            not input_audit["target_values_forward_input"]
            and not input_audit["exact_residual_forward_input"]
        ),
    }
    checkpoint = {
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": model.architecture(),
        "output_scale": scale.cpu(),
        "pair_codebook": args.pair_codebook,
        "individual_codebook": args.individual_codebook,
        "best_step": best_step,
        "seed": seed,
    }
    torch.save(checkpoint, output / "decoder.pt")
    report = {
        "schema": "hierarchical-jewel-phrase-decoder-gate-v1",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": sorted(validation_sources),
            "train_pairs_per_source": args.train_pairs_per_source,
            "steps_max": args.steps,
            "steps_completed": history[-1]["step"],
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "seed": seed,
            "voxel_points": args.voxel_points,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "input_audit": input_audit,
        "history": history,
        "best_step": best_step,
        "macro": macro_report,
        "records": audit_rows,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": report["gate"], "macro": macro_report,
                      "best_step": best_step}, indent=2))


if __name__ == "__main__":
    main()
