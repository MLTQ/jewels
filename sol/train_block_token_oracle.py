"""Train and audit the preregistered local spacetime block-token oracle."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import (
    FieldRecord,
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.audit_scene_posterior_oracle import select_oracle_sources
from sol.block_token_jewel_speaker import BlockTokenJewelSpeaker
from sol.block_token_language import (
    block_serialization_order,
    encode_block_tokens,
    fit_block_token_codebook,
    most_frequent_block_token,
)
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.jewel_casting_language import histogram_cosine
from sol.prompt_jewel_caster import (
    ACTIVE_FACTORS,
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits
from sol.train_prompt_jewel_caster import _metadata


ARMS = ("oracle block", "shuffled block", "null block")


@dataclass(frozen=True)
class BlockOracleBatch:
    centers: torch.Tensor
    negative_centers: torch.Tensor
    jewel_tokens: torch.Tensor
    owners: torch.Tensor

    def index(self, selected: torch.Tensor) -> "BlockOracleBatch":
        return BlockOracleBatch(
            self.centers[selected], self.negative_centers[selected],
            self.jewel_tokens[selected], self.owners[selected],
        )

    def __len__(self) -> int:
        return int(len(self.centers))


def cyclic_shuffled_programs(
    records: list[FieldRecord], programs: torch.Tensor
) -> torch.Tensor:
    """Align each fit with the same-rank fit from the next prompt source."""
    if programs.shape[0] != len(records):
        raise ValueError("record and block-program rows must align")
    source_order = sorted({record.source_id for record in records})
    if len(source_order) < 2:
        raise ValueError("shuffled block control requires multiple prompt sources")
    by_source = {
        source: sorted(
            (index for index, record in enumerate(records) if record.source_id == source),
            key=lambda index: records[index].fit_seed,
        )
        for source in source_order
    }
    output = []
    for index, record in enumerate(records):
        rows = by_source[record.source_id]
        rank = rows.index(index)
        shifted_source = source_order[(source_order.index(record.source_id) + 1) % len(source_order)]
        shifted_rows = by_source[shifted_source]
        output.append(programs[shifted_rows[rank % len(shifted_rows)]])
    return torch.stack(output)


def oracle_control_metrics(
    model: BlockTokenJewelSpeaker,
    batch: BlockOracleBatch,
    programs: torch.Tensor,
    shuffled_programs: torch.Tensor,
    null_token: int,
    *,
    chunk: int = 4096,
) -> dict:
    """Measure oracle, cyclic-shuffled, and prompt-blind block conditions."""
    if programs.shape != shuffled_programs.shape:
        raise ValueError("correct and shuffled block programs must align")
    output = {}
    with torch.no_grad():
        for arm in ARMS:
            token_sum = torch.zeros(len(ACTIVE_FACTORS), device=batch.centers.device)
            density_sum = 0.0
            count = 0
            for start in range(0, len(batch), chunk):
                selected = torch.arange(
                    start, min(start + chunk, len(batch)), device=batch.centers.device
                )
                part = batch.index(selected)
                positive_cells = model.spec.cell_index(part.centers)
                negative_cells = model.spec.cell_index(part.negative_centers)
                if arm == "oracle block":
                    arm_programs = programs
                elif arm == "shuffled block":
                    arm_programs = shuffled_programs
                else:
                    arm_programs = None
                if arm_programs is None:
                    positive_blocks = torch.full_like(positive_cells, null_token)
                    negative_blocks = torch.full_like(negative_cells, null_token)
                else:
                    positive_blocks = arm_programs[part.owners, positive_cells]
                    negative_blocks = arm_programs[part.owners, negative_cells]
                logits = model.token_logits(positive_blocks, part.centers)
                for role in range(len(ACTIVE_FACTORS)):
                    token_sum[role] += F.cross_entropy(
                        logits[:, role], part.jewel_tokens[:, role], reduction="sum"
                    )
                positive = model.intensity_logits(positive_blocks, part.centers)
                negative = model.intensity_logits(
                    negative_blocks, part.negative_centers
                )
                density_sum += float(
                    0.5 * (
                        F.binary_cross_entropy_with_logits(
                            positive, torch.ones_like(positive), reduction="sum"
                        )
                        + F.binary_cross_entropy_with_logits(
                            negative, torch.zeros_like(negative), reduction="sum"
                        )
                    )
                )
                count += len(part)
            output[arm] = {
                "token_nll": {
                    name: float(token_sum[index] / count)
                    for index, name in enumerate(ACTIVE_FACTORS)
                },
                "token_nll_macro": float(token_sum.mean() / count),
                "density_nce": density_sum / count,
            }
    return output


def make_batch(
    records: list[FieldRecord],
    *,
    physical_codebook,
    device: torch.device,
    rows_per_field: int,
    seed: int,
) -> BlockOracleBatch:
    """Build deterministic source-owned positive/negative Jewel rows."""
    parts = []
    for owner, record in enumerate(records):
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(rows_per_field, len(features))]
        sampled = features[selected]
        parts.append(
            BlockOracleBatch(
                centers=sampled[:, :3],
                negative_centers=torch.rand(
                    len(sampled), 3, generator=generator, device=device
                ) * 1.998 - 0.999,
                jewel_tokens=encode_active_jewel_tokens(sampled, physical_codebook),
                owners=torch.full(
                    (len(sampled),), owner, dtype=torch.long, device=device
                ),
            )
        )
    return BlockOracleBatch(
        *(torch.cat([getattr(part, field) for part in parts])
          for field in ("centers", "negative_centers", "jewel_tokens", "owners"))
    )


def programs_for_records(
    records: list[FieldRecord], block_codebook, device: torch.device
) -> tuple[torch.Tensor, list[float]]:
    programs, distances = [], []
    for record in records:
        program, distance = encode_block_tokens(record.features.to(device), block_codebook)
        programs.append(program)
        distances.append(float(distance.mean()))
    return torch.stack(programs), distances


def generation_macro(rows: list[dict]) -> dict:
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


@torch.no_grad()
def evaluate_generation(
    model: BlockTokenJewelSpeaker,
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
    """Free-run all block controls with matched randomness and render a contact sheet."""
    device = model.block_token_embedding.weight.device
    spec = model.spec
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
            centers = model.sample_centers(
                arm_programs[arm], generation_jewels,
                generator=arm_generator, proposal_multiplier=4,
            )
            tokens = model.sample_tokens(
                arm_programs[arm], centers, generator=arm_generator,
                temperature=0.9, top_k=64,
            )
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
        print("generated block oracle", record.source_id, flush=True)
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
    return rows, generation_macro(rows)


def _select_lowest_fit(records: list[FieldRecord]) -> list[FieldRecord]:
    return [
        min((record for record in records if record.source_id == source),
            key=lambda record: record.fit_seed)
        for source in sorted({record.source_id for record in records})
    ]


def resolve_split_arguments(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Resolve roots and source IDs from an immutable prior report or explicit CLI rows."""
    if args.split_report:
        if args.root or args.validation_source or args.training_source:
            raise ValueError("split report cannot be mixed with explicit split arguments")
        protocol = json.loads(Path(args.split_report).read_text())["protocol"]
        return (
            list(protocol["roots"]),
            list(protocol["validation_sources"]),
            list(protocol["training_sources"]),
        )
    if not args.root or not args.validation_source or not args.training_source:
        raise ValueError("explicit split requires roots plus training and validation sources")
    return list(args.root), list(args.validation_source), list(args.training_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append")
    parser.add_argument("--validation-source", action="append")
    parser.add_argument("--training-source", action="append")
    parser.add_argument("--split-report")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--posterior-oracle-report", required=True)
    parser.add_argument("--shared-scene-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-vocabulary-size", type=int, default=256)
    parser.add_argument("--block-iterations", type=int, default=20)
    parser.add_argument("--train-jewels-per-source", type=int, default=16384)
    parser.add_argument("--validation-jewels-per-field", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=108)
    args = parser.parse_args()
    if min(
        args.block_vocabulary_size, args.block_iterations,
        args.train_jewels_per_source, args.validation_jewels_per_field,
        args.steps, args.batch_size, args.eval_every, args.patience,
        args.generation_jewels, args.voxel_points, args.frames,
        args.height, args.width,
    ) <= 0:
        raise ValueError("block oracle schedule values must be positive")
    if args.block_vocabulary_size not in (64, 256, 1024):
        raise ValueError("Gate 2a permits only preregistered block vocabularies")
    device = torch.device(args.device)
    seed = 20260905 + args.block_vocabulary_size
    torch.manual_seed(seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    physical_codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec((8, 8, 4), slots_per_cell=1)
    roots, validation_sources, training_sources = resolve_split_arguments(args)
    records = load_field_records([Path(root) for root in roots])
    training, validation = select_prompt_splits(
        records, set(validation_sources), set(training_sources)
    )
    training = sorted(training, key=lambda record: record.source_id)
    validation = sorted(validation, key=lambda record: (record.source_id, record.fit_seed))
    if len(training) != 18 or len(validation) != 9:
        raise ValueError(
            f"Gate 2a requires 18 training and 9 validation fields; got {len(training)} and {len(validation)}"
        )
    print("fitting block vocabulary", args.block_vocabulary_size, flush=True)
    training_fields = [record.features.to(device) for record in training]
    block_codebook, block_fit = fit_block_token_codebook(
        training_fields,
        normalizer=physical_codebook.normalizer,
        spec=spec,
        vocabulary_size=args.block_vocabulary_size,
        iterations=args.block_iterations,
        seed=seed,
    )
    del training_fields
    training_programs, training_assignment = programs_for_records(
        training, block_codebook, device
    )
    validation_programs, validation_assignment = programs_for_records(
        validation, block_codebook, device
    )
    null_token = most_frequent_block_token(
        training_programs, args.block_vocabulary_size
    )
    validation_shuffled = cyclic_shuffled_programs(validation, validation_programs)
    train_batch = make_batch(
        training, physical_codebook=physical_codebook, device=device,
        rows_per_field=args.train_jewels_per_source, seed=seed,
    )
    validation_batch = make_batch(
        validation, physical_codebook=physical_codebook, device=device,
        rows_per_field=args.validation_jewels_per_field, seed=seed + 1000,
    )
    model = BlockTokenJewelSpeaker(
        block_vocabulary_size=args.block_vocabulary_size,
        jewel_vocabulary_size=physical_codebook.vocabulary_size,
        block_shape=spec.shape,
        hidden_dim=512,
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    best_score, best_step, best_state, stale = float("inf"), 0, None, 0
    history = []
    for step in range(1, args.steps + 1):
        selected = torch.randint(
            len(train_batch), (args.batch_size,), generator=generator, device=device
        )
        part = train_batch.index(selected)
        cells = spec.cell_index(part.centers)
        positive_blocks = training_programs[part.owners, cells]
        negative = torch.rand(
            len(part), 3, generator=generator, device=device
        ) * 1.998 - 0.999
        negative_cells = spec.cell_index(negative)
        negative_blocks = training_programs[part.owners, negative_cells]
        loss, components = model.loss(
            positive_blocks, part.centers, part.jewel_tokens,
            negative_blocks, negative, density_weight=0.1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        controls = oracle_control_metrics(
            model, validation_batch, validation_programs,
            validation_shuffled, null_token,
        )
        correct = controls["oracle block"]
        score = correct["token_nll_macro"] + 0.1 * correct["density_nce"]
        row = {
            "step": step,
            "selection_score": score,
            "controls": controls,
            "train_loss": float(loss.detach()),
            "train_token_nll": float(components["token_nll"].detach()),
            "train_density_nce": float(components["density_nce"].detach()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score * 0.999:
            best_score, best_step = score, step
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        (output / "progress.json").write_text(
            json.dumps(
                {"best_score": best_score, "best_step": best_step, "history": history},
                indent=2,
            ) + "\n"
        )
        model.train()
        if stale >= args.patience:
            print("source-disjoint validation plateau", step, stale, flush=True)
            break
    if best_state is None:
        raise RuntimeError("block oracle training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_controls = oracle_control_metrics(
        model, validation_batch, validation_programs,
        validation_shuffled, null_token,
    )

    direct_records = select_oracle_sources(training, [record.source_id for record in training])
    training_program_by_source = {
        record.source_id: training_programs[index]
        for index, record in enumerate(training)
    }
    direct_programs = torch.stack(
        [training_program_by_source[record.source_id] for record in direct_records]
    )
    direct_shuffled = torch.roll(direct_programs, shifts=-1, dims=0)
    direct_batch = make_batch(
        direct_records, physical_codebook=physical_codebook, device=device,
        rows_per_field=args.validation_jewels_per_field, seed=seed + 2000,
    )
    direct_controls = oracle_control_metrics(
        model, direct_batch, direct_programs, direct_shuffled, null_token
    )
    direct_rows, direct_generation = evaluate_generation(
        model, direct_records, direct_programs, direct_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 3000,
        image_path=output / "qualitative_direct.png",
    )
    heldout_records = _select_lowest_fit(validation)
    validation_program_by_path = {
        record.path: validation_programs[index]
        for index, record in enumerate(validation)
    }
    heldout_programs = torch.stack(
        [validation_program_by_path[record.path] for record in heldout_records]
    )
    heldout_shuffled = torch.roll(heldout_programs, shifts=-1, dims=0)
    heldout_rows, heldout_generation = evaluate_generation(
        model, heldout_records, heldout_programs, heldout_shuffled, null_token,
        physical_codebook=physical_codebook,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames, height=args.height, width=args.width,
        seed=seed + 4000,
        image_path=output / "qualitative_source_disjoint.png",
    )

    posterior_report = json.loads(Path(args.posterior_oracle_report).read_text())
    shared_scene_report = json.loads(Path(args.shared_scene_report).read_text())
    posterior_baseline = posterior_report["macro"]["posterior oracle"]
    direct_token = direct_controls["oracle block"]["token_nll_macro"]
    direct_histogram = direct_generation["oracle block"]["target_histogram_cosine"]
    token_improvement = (
        posterior_baseline["token_nll_macro"] - direct_token
    ) / posterior_baseline["token_nll_macro"]
    histogram_improvement = (
        direct_histogram - posterior_baseline["target_histogram_cosine"]
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
    passed = all(checks.values())
    torch.save(
        {
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "architecture": model.architecture(),
            "block_codebook": block_codebook.state_dict(),
            "physical_codebook": args.codebook,
            "training_sources": [record.source_id for record in training],
            "training_programs": training_programs.cpu(),
            "serialization_order": block_serialization_order(spec),
            "null_token": null_token,
            "best_step": best_step,
            "seed": seed,
        },
        output / "speaker.pt",
    )
    result = {
        "schema": "local-block-token-oracle-gate-v1",
        "protocol": {
            "roots": roots,
            "split_report": args.split_report,
            "training_sources": [record.source_id for record in training],
            "validation_sources": sorted({record.source_id for record in validation}),
            "training_fields": len(training),
            "validation_fields": len(validation),
            "block_shape": list(spec.shape),
            "blocks_per_program": spec.n_cells,
            "serialization": "time_major_xy_morton",
            "block_vocabulary_size": args.block_vocabulary_size,
            "block_descriptor_dim": block_codebook.descriptor_dim,
            "local_hist_shape": list(block_codebook.local_hist_shape),
            "train_jewels_per_source": args.train_jewels_per_source,
            "validation_jewels_per_field": args.validation_jewels_per_field,
            "steps_max": args.steps,
            "steps_completed": history[-1]["step"],
            "best_step": best_step,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "generation_jewels": args.generation_jewels,
            "oracle_uses_target_block_descriptors": True,
            "oracle_is_not_valid_prompt_inference": True,
            "emitted_centroids_are_continuous": True,
            "inference_inputs": ["ordered_block_token_program", "declared_random_seed"],
        },
        "block_vocabulary_fit": block_fit,
        "assignment_distance": {
            "training_mean": sum(training_assignment) / len(training_assignment),
            "source_disjoint_mean": sum(validation_assignment) / len(validation_assignment),
        },
        "history": history,
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
        "frozen_baselines": {
            "global_posterior_oracle": posterior_baseline,
            "global_text_prior": posterior_report["macro"]["text prior"],
            "global_prompt_blind": posterior_report["macro"]["prompt blind"],
            "shared_scene_correct_text": {
                "teacher_forced": shared_scene_report["teacher_forced_controls"]["correct"],
                "generation": shared_scene_report["generation_macro"]["correct"],
            },
            "shared_scene_shuffled_text": {
                "teacher_forced": shared_scene_report["teacher_forced_controls"]["shuffled"],
                "generation": shared_scene_report["generation_macro"]["shuffled"],
            },
            "shared_scene_null_text": {
                "teacher_forced": shared_scene_report["teacher_forced_controls"]["null"],
                "generation": shared_scene_report["generation_macro"]["null"],
            },
        },
        "improvement_over_global_posterior": {
            "token_nll_fraction": token_improvement,
            "histogram_cosine_absolute": histogram_improvement,
        },
        "gate": {
            "owns_primary_gate": args.block_vocabulary_size == 256,
            "checks": checks,
            "passed": passed,
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "vocabulary": args.block_vocabulary_size,
        "best_step": best_step,
        "direct_controls": direct_controls,
        "validation_controls": validation_controls,
        "direct_generation": direct_generation,
        "heldout_generation": heldout_generation,
        "improvement_over_global_posterior": result["improvement_over_global_posterior"],
        "gate": result["gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
