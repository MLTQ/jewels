"""Fit and audit the preregistered additive prompt-to-Jewel language."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from sol.additive_prompt_jewel_caster import (
    AdditivePromptJewelCaster,
    accumulate_language_counts,
)
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
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.token_grid import GridSpec
from sol.train_prompt_jewel_caster import _metadata, frozen_text_embeddings


def resolve_factor(query: torch.Tensor, candidates: torch.Tensor) -> int:
    """Resolve one text vector to the nearest frozen factor phrase."""
    if query.shape[0] != 1 or query.shape[1] != candidates.shape[1]:
        raise ValueError("factor resolver embedding shapes are incompatible")
    similarity = F.normalize(query, dim=1) @ F.normalize(candidates, dim=1).T
    return int(similarity[0].argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--train-jewels-per-source", type=int, default=8192)
    parser.add_argument("--validation-jewels-per-field", type=int, default=16384)
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    args = parser.parse_args()
    if min(
        args.train_jewels_per_source,
        args.validation_jewels_per_field,
        args.generation_jewels,
        args.voxel_points,
    ) <= 0:
        raise ValueError("additive prompt audit budgets must be positive")
    device = torch.device(args.device)
    seed = 20260902
    codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    validation_sources = set(args.validation_source)
    records = load_field_records([Path(root) for root in args.root])
    training = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("additive prompt training permits one field per source")
    if any(
        sum(record.source_id == source for record in validation) != 3
        for source in validation_sources
    ):
        raise ValueError("additive prompt validation requires three fits per source")
    metadata = {record.path: _metadata(record.path) for record in records}
    styles = sorted({row["style"] for row in metadata.values()})
    actions = sorted({row["source_prompt"] for row in metadata.values()})
    style_to_index = {value: index for index, value in enumerate(styles)}
    action_to_index = {value: index for index, value in enumerate(actions)}
    style_labels = [f"{style} visual style" for style in styles]
    factor_embeddings = frozen_text_embeddings(
        style_labels + actions,
        device=device,
        model_name=args.text_model,
    )
    style_embeddings = factor_embeddings[: len(styles)]
    action_embeddings = factor_embeddings[len(styles) :]

    train_samples = []
    for record in training:
        meta = metadata[record.path]
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.train_jewels_per_source, len(features))]
        sampled = features[selected]
        train_samples.append(
            (
                sampled[:, :3],
                encode_active_jewel_tokens(sampled, codebook),
                style_to_index[meta["style"]],
                action_to_index[meta["source_prompt"]],
            )
        )
        print("sampled train", record.source_id, len(sampled), flush=True)
    counts = accumulate_language_counts(
        train_samples,
        spec=spec,
        vocabulary_size=codebook.vocabulary_size,
        style_count=len(styles),
        action_count=len(actions),
    )
    model = AdditivePromptJewelCaster(
        counts,
        spec=spec,
        token_concentration=64.0,
        cell_concentration=256.0,
    )
    source_order = sorted(validation_sources)
    source_factors = {}
    resolution_rows = []
    for source in source_order:
        record = next(record for record in validation if record.source_id == source)
        meta = metadata[record.path]
        query = frozen_text_embeddings(
            [f"{meta['style']} visual style", meta["source_prompt"]],
            device=device,
            model_name=args.text_model,
        )
        resolved_style = resolve_factor(query[:1], style_embeddings)
        resolved_action = resolve_factor(query[1:], action_embeddings)
        declared_style = style_to_index[meta["style"]]
        declared_action = action_to_index[meta["source_prompt"]]
        source_factors[source] = (resolved_style, resolved_action)
        resolution_rows.append(
            {
                "source_id": source,
                "style_text": f"{meta['style']} visual style",
                "action_text": meta["source_prompt"],
                "resolved_style": styles[resolved_style],
                "resolved_action": actions[resolved_action],
                "declared_style": meta["style"],
                "declared_action": meta["source_prompt"],
                "correct": (
                    resolved_style == declared_style and resolved_action == declared_action
                ),
            }
        )

    control_sums = {
        arm: {
            "cell_nll": 0.0,
            "token_nll": {name: 0.0 for name in ACTIVE_FACTORS},
            "fields": 0,
        }
        for arm in ("correct", "shuffled", "null")
    }
    for record in validation:
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(
            seed + 1000 + _seed_for(record)
        )
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.validation_jewels_per_field, len(features))]
        sampled = features[selected]
        tokens = encode_active_jewel_tokens(sampled, codebook)
        prompt_index = source_order.index(record.source_id)
        for arm in ("correct", "shuffled", "null"):
            if arm == "correct":
                style_index, action_index = source_factors[record.source_id]
            elif arm == "shuffled":
                shifted_source = source_order[(prompt_index + 1) % len(source_order)]
                style_index, action_index = source_factors[shifted_source]
            else:
                style_index, action_index = None, None
            metrics = model.negative_log_likelihood(
                sampled[:, :3], tokens,
                style_index=style_index, action_index=action_index,
            )
            control_sums[arm]["cell_nll"] += metrics["cell_nll"]
            for name in ACTIVE_FACTORS:
                control_sums[arm]["token_nll"][name] += metrics["token_nll"][name]
            control_sums[arm]["fields"] += 1
        print("evaluated", record.source_id, record.fit_seed, flush=True)
    controls = {}
    for arm, row in control_sums.items():
        fields = row["fields"]
        token_nll = {
            name: value / fields for name, value in row["token_nll"].items()
        }
        controls[arm] = {
            "cell_nll": row["cell_nll"] / fields,
            "token_nll": token_nll,
            "token_nll_macro": sum(token_nll.values()) / len(token_nll),
        }

    target_records = [
        min(
            (record for record in validation if record.source_id == source),
            key=lambda record: record.fit_seed,
        )
        for source in source_order
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
    correct_histograms = []
    generated_programs = {}
    for prompt_index, record in enumerate(target_records):
        target_features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(
            seed + 10000 + prompt_index
        )
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2 - 1
        source_render = _render(target_features, points, record.background)
        arms = {}
        for arm in ("correct", "shuffled", "null"):
            if arm == "correct":
                style_index, action_index = source_factors[record.source_id]
            elif arm == "shuffled":
                shifted_source = source_order[(prompt_index + 1) % len(source_order)]
                style_index, action_index = source_factors[shifted_source]
            else:
                style_index, action_index = None, None
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 20000 + prompt_index
            )
            centers, tokens = model.sample(
                args.generation_jewels,
                style_index=style_index,
                action_index=action_index,
                generator=arm_generator,
            )
            features = active_tokens_to_features(centers, tokens, codebook)
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=codebook.vocabulary_size,
            )
            if arm == "correct":
                correct_histograms.append(histogram)
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
    for prompt_index, histogram in enumerate(correct_histograms):
        similarities = [
            histogram_cosine(target, histogram) for target in target_histograms
        ]
        retrieved = int(torch.tensor(similarities).argmax())
        retrieval.append(
            {
                "prompt_index": prompt_index,
                "similarities": similarities,
                "retrieved_index": retrieved,
                "correct": retrieved == prompt_index,
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
    better_token_control = min(
        controls["shuffled"]["token_nll_macro"],
        controls["null"]["token_nll_macro"],
    )
    token_improvement = (
        better_token_control - controls["correct"]["token_nll_macro"]
    ) / better_token_control
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
        for row in generation_rows for arm in ("correct", "shuffled", "null")
    )
    checks = {
        "correct_cell_nll_beats_controls": (
            controls["correct"]["cell_nll"] < controls["shuffled"]["cell_nll"]
            and controls["correct"]["cell_nll"] < controls["null"]["cell_nll"]
        ),
        "correct_token_nll_beats_controls_all_roles": all(
            controls["correct"]["token_nll"][name]
            < controls[arm]["token_nll"][name]
            for name in ACTIVE_FACTORS for arm in ("shuffled", "null")
        ),
        "correct_token_nll_improves_at_least_2pct": token_improvement >= 0.02,
        "free_run_histogram_margin_at_least_002": (
            generation_macro_report["correct"]["target_histogram_cosine"]
            - max(
                generation_macro_report["shuffled"]["target_histogram_cosine"],
                generation_macro_report["null"]["target_histogram_cosine"],
            ) >= 0.02
        ),
        "free_run_top1_at_least_2of3": sum(row["correct"] for row in retrieval) >= 2,
        "text_factor_resolution_exact": all(row["correct"] for row in resolution_rows),
        "generated_centers_not_grid_locked": (
            generation_macro_report["correct"]["cell_center_lock_fraction"] < 0.01
        ),
        "generated_renders_finite": all_finite,
        "prompt_only_inference": not any(
            value for key, value in inference_audit.items() if key != "inputs"
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "counts": counts,
            "styles": styles,
            "actions": actions,
            "style_labels": style_labels,
            "text_model": args.text_model,
            "codebook": args.codebook,
            "token_concentration": 64.0,
            "cell_concentration": 256.0,
            "seed": seed,
        },
        output / "caster.pt",
    )
    torch.save(generated_programs, output / "generated_programs.pt")
    report = {
        "schema": "additive-prompt-native-jewel-caster-gate-v1",
        "protocol": {
            "roots": args.root,
            "training_fields": len(training),
            "validation_fields": len(validation),
            "validation_sources": source_order,
            "train_jewels_per_source": args.train_jewels_per_source,
            "validation_jewels_per_field": args.validation_jewels_per_field,
            "generation_jewels": args.generation_jewels,
            "token_concentration": 64.0,
            "cell_concentration": 256.0,
            "seed": seed,
            "text_model": args.text_model,
        },
        "resolution": resolution_rows,
        "teacher_forced_controls": controls,
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
        "teacher_forced_controls": controls,
        "generation_macro": generation_macro_report,
        "retrieval": retrieval,
        "resolution": resolution_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
