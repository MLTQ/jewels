"""Train and audit the factorized-text continuous-intensity Jewel caster."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import (
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.jewel_casting_language import histogram_cosine
from sol.prompt_jewel_caster import (
    ACTIVE_FACTORS,
    FactorizedPromptJewelCaster,
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.token_grid import GridSpec
from sol.train_prompt_jewel_caster import _metadata, frozen_text_embeddings


@dataclass(frozen=True)
class FactorPromptBatch:
    centers: torch.Tensor
    negative_centers: torch.Tensor
    tokens: torch.Tensor
    combinations: torch.Tensor

    def index(self, selected: torch.Tensor) -> "FactorPromptBatch":
        return FactorPromptBatch(
            self.centers[selected],
            self.negative_centers[selected],
            self.tokens[selected],
            self.combinations[selected],
        )

    def __len__(self) -> int:
        return int(len(self.centers))


def factor_control_metrics(
    model: FactorizedPromptJewelCaster,
    batch: FactorPromptBatch,
    validation_styles: torch.Tensor,
    validation_actions: torch.Tensor,
    *,
    chunk: int = 4096,
) -> dict:
    """Evaluate correct, cyclic-shuffled, and zero-text factorized controls."""
    prompt_count = len(validation_styles)
    output = {}
    with torch.no_grad():
        for arm in ("correct", "shuffled", "null"):
            token_sum = torch.zeros(len(ACTIVE_FACTORS), device=batch.centers.device)
            density_sum = 0.0
            count = 0
            for start in range(0, len(batch), chunk):
                selected = torch.arange(
                    start, min(start + chunk, len(batch)), device=batch.centers.device
                )
                part = batch.index(selected)
                if arm == "correct":
                    style = validation_styles[part.combinations]
                    action = validation_actions[part.combinations]
                elif arm == "shuffled":
                    shifted = (part.combinations + 1) % prompt_count
                    style = validation_styles[shifted]
                    action = validation_actions[shifted]
                else:
                    style = torch.zeros(
                        len(part), validation_styles.shape[1], device=batch.centers.device
                    )
                    action = torch.zeros_like(style)
                logits = model.token_logits(style, action, part.centers)
                for role in range(len(ACTIVE_FACTORS)):
                    token_sum[role] += F.cross_entropy(
                        logits[:, role], part.tokens[:, role], reduction="sum"
                    )
                positive = model.intensity_logits(style, action, part.centers)
                negative = model.intensity_logits(style, action, part.negative_centers)
                density_sum += float(
                    0.5
                    * (
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
                "density_nce": density_sum / count,
                "token_nll": {
                    name: float(token_sum[index] / count)
                    for index, name in enumerate(ACTIVE_FACTORS)
                },
                "token_nll_macro": float(token_sum.mean() / count),
            }
    return output


def exact_prompt_source_counts(
    training: list,
    validation: list,
    metadata_by_path: dict[str, dict],
) -> dict[str, int]:
    """Count source-disjoint exact style/action matches for each validation source."""
    training_keys = [
        (
            metadata_by_path[record.path]["style"],
            metadata_by_path[record.path]["source_prompt"],
        )
        for record in training
    ]
    output = {}
    for source in sorted({record.source_id for record in validation}):
        source_rows = [record for record in validation if record.source_id == source]
        keys = {
            (
                metadata_by_path[record.path]["style"],
                metadata_by_path[record.path]["source_prompt"],
            )
            for record in source_rows
        }
        if len(keys) != 1:
            raise ValueError(f"validation replicas disagree on prompt metadata: {source}")
        key = next(iter(keys))
        output[source] = sum(training_key == key for training_key in training_keys)
    return output


def select_prompt_splits(
    records: list,
    validation_sources: set[str],
    training_sources: set[str] | None = None,
) -> tuple[list, list]:
    """Select source-disjoint prompt splits, optionally restricting training sources."""
    validation = [record for record in records if record.source_id in validation_sources]
    training = [record for record in records if record.source_id not in validation_sources]
    if training_sources is not None:
        missing = training_sources - {record.source_id for record in training}
        if missing:
            raise ValueError(f"requested prompt training sources are missing: {sorted(missing)}")
        training = [record for record in training if record.source_id in training_sources]
    if not training:
        raise ValueError("factorized prompt training split is empty")
    if not validation:
        raise ValueError("factorized prompt validation split is empty")
    return training, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--training-source", action="append")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--train-jewels-per-source", type=int, default=8192)
    parser.add_argument("--validation-jewels-per-field", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    parser.add_argument("--minimum-exact-prompt-sources", type=int, default=0)
    args = parser.parse_args()
    if min(
        args.train_jewels_per_source,
        args.validation_jewels_per_field,
        args.steps,
        args.batch_size,
        args.eval_every,
        args.patience,
        args.generation_jewels,
        args.voxel_points,
    ) <= 0:
        raise ValueError("factorized prompt schedule values must be positive")
    if args.minimum_exact_prompt_sources < 0:
        raise ValueError("minimum exact-prompt sources must be non-negative")
    device = torch.device(args.device)
    seed = 20260901
    torch.manual_seed(seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    validation_sources = set(args.validation_source)
    records = load_field_records([Path(root) for root in args.root])
    training_sources = None if args.training_source is None else set(args.training_source)
    training, validation = select_prompt_splits(
        records, validation_sources, training_sources
    )
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("factorized prompt training permits one field per source")
    validation_counts = {
        source: sum(record.source_id == source for record in validation)
        for source in validation_sources
    }
    if any(count != 3 for count in validation_counts.values()):
        raise ValueError("factorized prompt validation requires three fits per source")
    metadata_by_path = {record.path: _metadata(record.path) for record in records}
    exact_prompt_counts = exact_prompt_source_counts(
        training, validation, metadata_by_path
    )
    if any(
        count < args.minimum_exact_prompt_sources
        for count in exact_prompt_counts.values()
    ):
        raise ValueError(
            "insufficient exact-prompt training sources: "
            f"required={args.minimum_exact_prompt_sources} observed={exact_prompt_counts}"
        )
    styles = sorted({row["style"] for row in metadata_by_path.values()})
    actions = sorted({row["source_prompt"] for row in metadata_by_path.values()})
    style_to_index = {value: index for index, value in enumerate(styles)}
    action_to_index = {value: index for index, value in enumerate(actions)}
    embeddings = frozen_text_embeddings(
        [f"{style} visual style" for style in styles] + actions,
        device=device,
        model_name=args.text_model,
    )
    style_embeddings = embeddings[: len(styles)]
    action_embeddings = embeddings[len(styles) :]
    validation_source_order = sorted(validation_sources)
    validation_combinations = []
    for source in validation_source_order:
        meta = metadata_by_path[
            next(record.path for record in validation if record.source_id == source)
        ]
        validation_combinations.append(
            (style_to_index[meta["style"]], action_to_index[meta["source_prompt"]])
        )
    validation_styles = torch.stack(
        [style_embeddings[style] for style, _ in validation_combinations]
    )
    validation_actions = torch.stack(
        [action_embeddings[action] for _, action in validation_combinations]
    )
    validation_combo_index = {
        source: index for index, source in enumerate(validation_source_order)
    }

    train_parts = []
    training_combinations = []
    for record in training:
        meta = metadata_by_path[record.path]
        combination = (
            style_to_index[meta["style"]], action_to_index[meta["source_prompt"]]
        )
        training_combinations.append(combination)
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.train_jewels_per_source, len(features))]
        sampled = features[selected]
        train_parts.append(
            FactorPromptBatch(
                centers=sampled[:, :3],
                negative_centers=torch.rand(
                    len(sampled), 3, generator=generator, device=device
                ) * 1.998 - 0.999,
                tokens=encode_active_jewel_tokens(sampled, codebook),
                combinations=torch.full(
                    (len(sampled),), len(training_combinations) - 1,
                    device=device, dtype=torch.long,
                ),
            )
        )
        print("sampled train", record.source_id, len(sampled), flush=True)
    train_batch = FactorPromptBatch(
        centers=torch.cat([part.centers for part in train_parts]),
        negative_centers=torch.cat([part.negative_centers for part in train_parts]),
        tokens=torch.cat([part.tokens for part in train_parts]),
        combinations=torch.cat([part.combinations for part in train_parts]),
    )
    train_style = torch.stack(
        [style_embeddings[style] for style, _ in training_combinations]
    )
    train_action = torch.stack(
        [action_embeddings[action] for _, action in training_combinations]
    )

    validation_parts = []
    for record in validation:
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(
            seed + 1000 + _seed_for(record)
        )
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.validation_jewels_per_field, len(features))]
        sampled = features[selected]
        validation_parts.append(
            FactorPromptBatch(
                centers=sampled[:, :3],
                negative_centers=torch.rand(
                    len(sampled), 3, generator=generator, device=device
                ) * 1.998 - 0.999,
                tokens=encode_active_jewel_tokens(sampled, codebook),
                combinations=torch.full(
                    (len(sampled),), validation_combo_index[record.source_id],
                    device=device, dtype=torch.long,
                ),
            )
        )
        print("sampled validation", record.source_id, record.fit_seed, len(sampled), flush=True)
    validation_batch = FactorPromptBatch(
        centers=torch.cat([part.centers for part in validation_parts]),
        negative_centers=torch.cat([part.negative_centers for part in validation_parts]),
        tokens=torch.cat([part.tokens for part in validation_parts]),
        combinations=torch.cat([part.combinations for part in validation_parts]),
    )

    model = FactorizedPromptJewelCaster(
        text_dim=style_embeddings.shape[1],
        vocabulary_size=codebook.vocabulary_size,
        hidden_dim=512,
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    dropout_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_score, best_step, best_state, stale = float("inf"), 0, None, 0
    history = []
    for step in range(1, args.steps + 1):
        selected = torch.randint(
            len(train_batch), (args.batch_size,),
            generator=sample_generator, device=device,
        )
        part = train_batch.index(selected)
        style = train_style[part.combinations].clone()
        action = train_action[part.combinations].clone()
        drop = torch.rand(
            len(part), generator=dropout_generator, device=device
        ) < 0.10
        style[drop], action[drop] = 0, 0
        negative = torch.rand(
            len(part), 3, generator=sample_generator, device=device
        ) * 1.998 - 0.999
        loss, _ = model.loss(
            style, action, part.centers, part.tokens, negative, density_weight=0.1
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        controls = factor_control_metrics(
            model, validation_batch, validation_styles, validation_actions
        )
        score = (
            controls["correct"]["token_nll_macro"]
            + 0.1 * controls["correct"]["density_nce"]
        )
        row = {"step": step, "selection_score": score, "controls": controls}
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
            print("validation plateau", step, stale, flush=True)
            break
    if best_state is None:
        raise RuntimeError("factorized prompt training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_controls = factor_control_metrics(
        model, validation_batch, validation_styles, validation_actions
    )

    target_records = [
        min(
            (record for record in validation if record.source_id == source),
            key=lambda record: record.fit_seed,
        )
        for source in validation_source_order
    ]
    target_histograms = []
    for record in target_records:
        features = record.features.to(device)
        target_histograms.append(
            active_cell_histogram(
                features[:, :3], encode_active_jewel_tokens(features, codebook),
                spec=spec, vocabulary_size=codebook.vocabulary_size,
            )
        )
    generation_rows = []
    generated_programs = {}
    correct_generation_histograms = []
    for prompt_index, record in enumerate(target_records):
        target_features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(
            seed + 10000 + prompt_index
        )
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2.0 - 1.0
        source_render = _render(target_features, points, record.background)
        arms = {}
        for arm in ("correct", "shuffled", "null"):
            if arm == "correct":
                style = validation_styles[prompt_index : prompt_index + 1]
                action = validation_actions[prompt_index : prompt_index + 1]
            elif arm == "shuffled":
                shifted = (prompt_index + 1) % len(target_records)
                style = validation_styles[shifted : shifted + 1]
                action = validation_actions[shifted : shifted + 1]
            else:
                style = torch.zeros_like(validation_styles[:1])
                action = torch.zeros_like(validation_actions[:1])
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 20000 + prompt_index
            )
            centers = model.sample_centers(
                style, action, args.generation_jewels,
                generator=arm_generator, proposal_multiplier=4,
            )
            tokens = model.sample_tokens(
                style, action, centers, generator=arm_generator,
                temperature=0.9, top_k=64,
            )
            features = active_tokens_to_features(centers, tokens, codebook)
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=codebook.vocabulary_size,
            )
            if arm == "correct":
                correct_generation_histograms.append(histogram)
            audit = _audit_candidate(
                target_features,
                features,
                reference_render=source_render,
                points=points,
                background=record.background,
                spec=spec,
            )
            arms[arm] = {
                "target_histogram_cosine": histogram_cosine(
                    target_histograms[prompt_index], histogram
                ),
                "audit": audit,
            }
            generated_programs[(record.source_id, arm)] = {
                "centers": centers.cpu(), "tokens": tokens.cpu()
            }
        generation_rows.append({"source_id": record.source_id, "arms": arms})
        print("generated", record.source_id, flush=True)

    retrieval = []
    for prompt_index, histogram in enumerate(correct_generation_histograms):
        similarities = [
            histogram_cosine(target, histogram) for target in target_histograms
        ]
        retrieval.append(
            {
                "prompt_index": prompt_index,
                "similarities": similarities,
                "retrieved_index": int(torch.tensor(similarities).argmax()),
                "correct": int(torch.tensor(similarities).argmax()) == prompt_index,
            }
        )

    def generation_macro(arm: str, path: tuple[str, ...]) -> float:
        values = []
        for row in generation_rows:
            value = row["arms"][arm]
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    generation_macro_report = {
        arm: {
            "target_histogram_cosine": generation_macro(
                arm, ("target_histogram_cosine",)
            ),
            "voxel_psnr_diagnostic": generation_macro(
                arm, ("audit", "voxel_psnr_to_continuous_source")
            ),
            "cell_center_lock_fraction": generation_macro(
                arm, ("audit", "center_irregularity", "cell_center_lock_fraction")
            ),
        }
        for arm in ("correct", "shuffled", "null")
    }
    correct = final_controls["correct"]
    better_control_token = min(
        final_controls["shuffled"]["token_nll_macro"],
        final_controls["null"]["token_nll_macro"],
    )
    token_improvement = (
        better_control_token - correct["token_nll_macro"]
    ) / better_control_token
    inference_audit = {
        "inputs": ["style_text", "action_text", "declared_random_seed"],
        "target_field": False,
        "target_centroids": False,
        "target_tokens": False,
        "source_pixels": False,
        "video_latent": False,
        "class_id": False,
        "source_identity": False,
    }
    all_finite = all(
        math.isfinite(row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"])
        for row in generation_rows
        for arm in ("correct", "shuffled", "null")
    )
    checks = {
        "correct_density_nce_beats_controls": (
            correct["density_nce"] < final_controls["shuffled"]["density_nce"]
            and correct["density_nce"] < final_controls["null"]["density_nce"]
        ),
        "correct_token_nll_beats_controls_all_roles": all(
            correct["token_nll"][name]
            < final_controls[arm]["token_nll"][name]
            for name in ACTIVE_FACTORS
            for arm in ("shuffled", "null")
        ),
        "correct_token_nll_improves_at_least_2pct": token_improvement >= 0.02,
        "free_run_histogram_margin_at_least_002": (
            generation_macro_report["correct"]["target_histogram_cosine"]
            - max(
                generation_macro_report["shuffled"]["target_histogram_cosine"],
                generation_macro_report["null"]["target_histogram_cosine"],
            )
            >= 0.02
        ),
        "free_run_top1_at_least_2of3": sum(row["correct"] for row in retrieval) >= 2,
        "generated_centers_not_grid_locked": (
            generation_macro_report["correct"]["cell_center_lock_fraction"] < 0.01
        ),
        "generated_renders_finite": all_finite,
        "prompt_only_inference": not any(
            value for key, value in inference_audit.items() if key != "inputs"
        ),
    }
    torch.save(
        {
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "architecture": model.architecture(),
            "text_model": args.text_model,
            "codebook": args.codebook,
            "styles": styles,
            "actions": actions,
            "best_step": best_step,
            "seed": seed,
        },
        output / "caster.pt",
    )
    torch.save(generated_programs, output / "generated_programs.pt")
    report = {
        "schema": "factorized-prompt-native-jewel-caster-gate-v1",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": validation_source_order,
            "train_jewels_per_source": args.train_jewels_per_source,
            "validation_jewels_per_field": args.validation_jewels_per_field,
            "generation_jewels": args.generation_jewels,
            "steps_max": args.steps,
            "steps_completed": history[-1]["step"],
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "seed": seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "text_model": args.text_model,
            "minimum_exact_prompt_sources": args.minimum_exact_prompt_sources,
            "training_source_filter": (
                None if training_sources is None else sorted(training_sources)
            ),
            "exact_prompt_training_sources": exact_prompt_counts,
            "styles": styles,
            "actions": actions,
        },
        "history": history,
        "best_step": best_step,
        "teacher_forced_controls": final_controls,
        "token_nll_improvement_fraction": token_improvement,
        "generation_macro": generation_macro_report,
        "generation_records": generation_rows,
        "retrieval": retrieval,
        "inference_audit": inference_audit,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate": report["gate"],
        "teacher_forced_controls": final_controls,
        "generation_macro": generation_macro_report,
        "retrieval": retrieval,
        "best_step": best_step,
    }, indent=2))


if __name__ == "__main__":
    main()
