"""Train and gate a prompt speaker with one shared latent state per Jewel scene."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

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
from sol.scene_latent_prompt_jewel_caster import SceneLatentPromptJewelCaster
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import (
    exact_prompt_source_counts,
    select_prompt_splits,
)
from sol.train_prompt_jewel_caster import _metadata, frozen_text_embeddings


@dataclass(frozen=True)
class ScenePromptBatch:
    centers: torch.Tensor
    negative_centers: torch.Tensor
    tokens: torch.Tensor
    owners: torch.Tensor

    def index(self, selected: torch.Tensor) -> "ScenePromptBatch":
        return ScenePromptBatch(
            self.centers[selected], self.negative_centers[selected],
            self.tokens[selected], self.owners[selected],
        )

    def __len__(self) -> int:
        return int(len(self.centers))


def gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(q||p) for aligned diagonal Gaussian scene states."""
    if not (
        posterior_mean.shape
        == posterior_log_std.shape
        == prior_mean.shape
        == prior_log_std.shape
    ):
        raise ValueError("scene Gaussian parameters must share a shape")
    variance_ratio = torch.exp(2 * (posterior_log_std - prior_log_std))
    mean_term = (posterior_mean - prior_mean).square() * torch.exp(
        -2 * prior_log_std
    )
    return 0.5 * (
        variance_ratio + mean_term - 1 + 2 * (prior_log_std - posterior_log_std)
    ).mean()


def scene_control_metrics(
    model: SceneLatentPromptJewelCaster,
    batch: ScenePromptBatch,
    validation_styles: torch.Tensor,
    validation_actions: torch.Tensor,
    *,
    chunk: int = 4096,
) -> dict:
    """Evaluate correct, cyclic-shuffled, and zero-prompt prior-mean controls."""
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
                    style = validation_styles[part.owners]
                    action = validation_actions[part.owners]
                elif arm == "shuffled":
                    shifted = (part.owners + 1) % prompt_count
                    style = validation_styles[shifted]
                    action = validation_actions[shifted]
                else:
                    style = torch.zeros(
                        len(part), validation_styles.shape[1], device=batch.centers.device
                    )
                    action = torch.zeros_like(style)
                scene, _ = model.prior_parameters(style, action)
                logits = model.token_logits(style, action, scene, part.centers)
                for role in range(len(ACTIVE_FACTORS)):
                    token_sum[role] += F.cross_entropy(
                        logits[:, role], part.tokens[:, role], reduction="sum"
                    )
                positive = model.intensity_logits(style, action, scene, part.centers)
                negative = model.intensity_logits(
                    style, action, scene, part.negative_centers
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
                "density_nce": density_sum / count,
                "token_nll": {
                    name: float(token_sum[index] / count)
                    for index, name in enumerate(ACTIVE_FACTORS)
                },
                "token_nll_macro": float(token_sum.mean() / count),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--training-source", action="append", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--scene-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--prior-scene-probability", type=float, default=0.25)
    parser.add_argument("--train-jewels-per-source", type=int, default=8192)
    parser.add_argument("--validation-jewels-per-field", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    parser.add_argument("--minimum-exact-prompt-sources", type=int, default=2)
    args = parser.parse_args()
    if min(
        args.scene_dim, args.train_jewels_per_source,
        args.validation_jewels_per_field, args.steps, args.batch_size,
        args.eval_every, args.patience, args.generation_jewels,
        args.voxel_points,
    ) <= 0:
        raise ValueError("scene-latent prompt schedule values must be positive")
    if args.kl_weight < 0 or not 0 <= args.prior_scene_probability <= 1:
        raise ValueError("scene-latent regularization settings are invalid")
    device = torch.device(args.device)
    seed = 20260903
    torch.manual_seed(seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    validation_sources = set(args.validation_source)
    training_sources = set(args.training_source)
    records = load_field_records([Path(root) for root in args.root])
    training, validation = select_prompt_splits(
        records, validation_sources, training_sources
    )
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("scene-latent training permits one field per source")
    validation_counts = {
        source: sum(record.source_id == source for record in validation)
        for source in validation_sources
    }
    if any(count != 3 for count in validation_counts.values()):
        raise ValueError("scene-latent validation requires three fits per source")
    metadata_by_path = {record.path: _metadata(record.path) for record in records}
    exact_counts = exact_prompt_source_counts(training, validation, metadata_by_path)
    if any(count < args.minimum_exact_prompt_sources for count in exact_counts.values()):
        raise ValueError(
            "insufficient exact-prompt training sources: "
            f"required={args.minimum_exact_prompt_sources} observed={exact_counts}"
        )
    styles = sorted({row["style"] for row in metadata_by_path.values()})
    actions = sorted({row["source_prompt"] for row in metadata_by_path.values()})
    style_to_index = {value: index for index, value in enumerate(styles)}
    action_to_index = {value: index for index, value in enumerate(actions)}
    embeddings = frozen_text_embeddings(
        [f"{style} visual style" for style in styles] + actions,
        device=device, model_name=args.text_model,
    )
    style_embeddings = embeddings[: len(styles)]
    action_embeddings = embeddings[len(styles) :]

    training = sorted(training, key=lambda record: record.source_id)
    train_style = []
    train_action = []
    train_parts = []
    for owner, record in enumerate(training):
        meta = metadata_by_path[record.path]
        train_style.append(style_embeddings[style_to_index[meta["style"]]])
        train_action.append(action_embeddings[action_to_index[meta["source_prompt"]]])
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.train_jewels_per_source, len(features))]
        sampled = features[selected]
        train_parts.append(
            ScenePromptBatch(
                sampled[:, :3],
                torch.rand(len(sampled), 3, generator=generator, device=device) * 1.998 - 0.999,
                encode_active_jewel_tokens(sampled, codebook),
                torch.full((len(sampled),), owner, device=device, dtype=torch.long),
            )
        )
        print("sampled train", record.source_id, len(sampled), flush=True)
    train_style = torch.stack(train_style)
    train_action = torch.stack(train_action)
    train_batch = ScenePromptBatch(
        *(torch.cat([getattr(part, field) for part in train_parts])
          for field in ("centers", "negative_centers", "tokens", "owners"))
    )

    validation_order = sorted(validation_sources)
    validation_style = []
    validation_action = []
    validation_parts = []
    validation_owner = {source: index for index, source in enumerate(validation_order)}
    for source in validation_order:
        record = next(row for row in validation if row.source_id == source)
        meta = metadata_by_path[record.path]
        validation_style.append(style_embeddings[style_to_index[meta["style"]]])
        validation_action.append(action_embeddings[action_to_index[meta["source_prompt"]]])
    validation_style = torch.stack(validation_style)
    validation_action = torch.stack(validation_action)
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
            ScenePromptBatch(
                sampled[:, :3],
                torch.rand(len(sampled), 3, generator=generator, device=device) * 1.998 - 0.999,
                encode_active_jewel_tokens(sampled, codebook),
                torch.full(
                    (len(sampled),), validation_owner[record.source_id],
                    device=device, dtype=torch.long,
                ),
            )
        )
        print("sampled validation", record.source_id, record.fit_seed, len(sampled), flush=True)
    validation_batch = ScenePromptBatch(
        *(torch.cat([getattr(part, field) for part in validation_parts])
          for field in ("centers", "negative_centers", "tokens", "owners"))
    )

    model = SceneLatentPromptJewelCaster(
        text_dim=style_embeddings.shape[1],
        vocabulary_size=codebook.vocabulary_size,
        scene_dim=args.scene_dim,
        hidden_dim=512, depth=4,
    ).to(device)
    posterior_mean = nn.Embedding(len(training), args.scene_dim).to(device)
    posterior_log_std = nn.Embedding(len(training), args.scene_dim).to(device)
    nn.init.normal_(posterior_mean.weight, std=0.1)
    nn.init.constant_(posterior_log_std.weight, -1.0)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(posterior_mean.parameters())
        + list(posterior_log_std.parameters()),
        lr=args.learning_rate, weight_decay=1e-4,
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    dropout_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_score, best_step, best_state, stale = float("inf"), 0, None, 0
    history = []
    for step in range(1, args.steps + 1):
        selected = torch.randint(
            len(train_batch), (args.batch_size,), generator=sample_generator,
            device=device,
        )
        part = train_batch.index(selected)
        q_mean = posterior_mean.weight
        q_log_std = posterior_log_std.weight.clamp(-4.0, 1.0)
        p_mean, p_log_std = model.prior_parameters(train_style, train_action)
        epsilon = torch.randn(
            q_mean.shape, device=device, generator=sample_generator
        )
        q_scene = q_mean + q_log_std.exp() * epsilon
        p_scene = p_mean + p_log_std.exp() * epsilon
        use_prior = torch.rand(
            len(training), device=device, generator=sample_generator
        ) < args.prior_scene_probability
        scene_by_source = torch.where(use_prior[:, None], p_scene, q_scene)
        scene = scene_by_source[part.owners]
        style = train_style[part.owners].clone()
        action = train_action[part.owners].clone()
        drop = torch.rand(
            len(part), device=device, generator=dropout_generator
        ) < 0.10
        style[drop], action[drop] = 0, 0
        negative = torch.rand(
            len(part), 3, device=device, generator=sample_generator
        ) * 1.998 - 0.999
        reconstruction, components = model.loss(
            style, action, scene, part.centers, part.tokens, negative,
            density_weight=0.1,
        )
        kl = gaussian_kl(q_mean, q_log_std, p_mean, p_log_std)
        loss = reconstruction + args.kl_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(posterior_mean.parameters())
            + list(posterior_log_std.parameters()), 1.0
        )
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        controls = scene_control_metrics(
            model, validation_batch, validation_style, validation_action
        )
        score = controls["correct"]["token_nll_macro"] + 0.1 * controls["correct"]["density_nce"]
        row = {
            "step": step, "selection_score": score, "controls": controls,
            "train_reconstruction": float(reconstruction.detach()),
            "train_kl": float(kl.detach()),
            "train_token_nll": float(components["token_nll"].detach()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score * 0.999:
            best_score, best_step = score, step
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "posterior_mean": posterior_mean.weight.detach().cpu().clone(),
                "posterior_log_std": posterior_log_std.weight.detach().cpu().clone(),
            }
            stale = 0
        else:
            stale += 1
        (output / "progress.json").write_text(
            json.dumps({"best_score": best_score, "best_step": best_step, "history": history}, indent=2) + "\n"
        )
        model.train()
        if stale >= args.patience:
            print("validation plateau", step, stale, flush=True)
            break
    if best_state is None:
        raise RuntimeError("scene-latent prompt training produced no checkpoint")
    model.load_state_dict(best_state["model"])
    model.eval()
    final_controls = scene_control_metrics(
        model, validation_batch, validation_style, validation_action
    )

    target_records = [
        min((record for record in validation if record.source_id == source),
            key=lambda record: record.fit_seed)
        for source in validation_order
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
    correct_histograms = []
    for prompt_index, record in enumerate(target_records):
        target_features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + 10000 + prompt_index)
        points = torch.rand(args.voxel_points, 3, generator=generator, device=device) * 2 - 1
        source_render = _render(target_features, points, record.background)
        epsilon = torch.randn(
            1, args.scene_dim, generator=generator, device=device
        )
        arms = {}
        for arm in ("correct", "shuffled", "null"):
            if arm == "correct":
                style = validation_style[prompt_index : prompt_index + 1]
                action = validation_action[prompt_index : prompt_index + 1]
            elif arm == "shuffled":
                shifted = (prompt_index + 1) % len(target_records)
                style = validation_style[shifted : shifted + 1]
                action = validation_action[shifted : shifted + 1]
            else:
                style = torch.zeros_like(validation_style[:1])
                action = torch.zeros_like(validation_action[:1])
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 20000 + prompt_index
            )
            scene = model.sample_scene(
                style, action, generator=arm_generator, epsilon=epsilon
            )
            centers = model.sample_centers(
                style, action, scene, args.generation_jewels,
                generator=arm_generator, proposal_multiplier=4,
            )
            tokens = model.sample_tokens(
                style, action, scene, centers, generator=arm_generator,
                temperature=0.9, top_k=64,
            )
            features = active_tokens_to_features(centers, tokens, codebook)
            histogram = active_cell_histogram(
                centers, tokens, spec=spec, vocabulary_size=codebook.vocabulary_size,
            )
            if arm == "correct":
                correct_histograms.append(histogram)
            audit = _audit_candidate(
                target_features, features, reference_render=source_render,
                points=points, background=record.background, spec=spec,
            )
            arms[arm] = {
                "target_histogram_cosine": histogram_cosine(
                    target_histograms[prompt_index], histogram
                ),
                "audit": audit,
            }
            generated_programs[(record.source_id, arm)] = {
                "centers": centers.cpu(), "tokens": tokens.cpu(),
                "scene": scene.cpu(),
            }
        generation_rows.append({"source_id": record.source_id, "arms": arms})
        print("generated", record.source_id, flush=True)
    retrieval = []
    for prompt_index, histogram in enumerate(correct_histograms):
        similarities = [histogram_cosine(target, histogram) for target in target_histograms]
        retrieved = int(torch.tensor(similarities).argmax())
        retrieval.append({
            "prompt_index": prompt_index, "similarities": similarities,
            "retrieved_index": retrieved, "correct": retrieved == prompt_index,
        })

    def generation_macro(arm: str, path: tuple[str, ...]) -> float:
        values = []
        for row in generation_rows:
            value = row["arms"][arm]
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    generation = {
        arm: {
            "target_histogram_cosine": generation_macro(arm, ("target_histogram_cosine",)),
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
    better_control = min(
        final_controls["shuffled"]["token_nll_macro"],
        final_controls["null"]["token_nll_macro"],
    )
    token_improvement = (better_control - correct["token_nll_macro"]) / better_control
    checks = {
        "correct_density_nce_beats_controls": all(
            correct["density_nce"] < final_controls[arm]["density_nce"]
            for arm in ("shuffled", "null")
        ),
        "correct_token_nll_beats_controls_all_roles": all(
            correct["token_nll"][name] < final_controls[arm]["token_nll"][name]
            for name in ACTIVE_FACTORS for arm in ("shuffled", "null")
        ),
        "correct_token_nll_improves_at_least_2pct": token_improvement >= 0.02,
        "free_run_histogram_margin_at_least_002": (
            generation["correct"]["target_histogram_cosine"]
            - max(generation["shuffled"]["target_histogram_cosine"],
                  generation["null"]["target_histogram_cosine"])
            >= 0.02
        ),
        "free_run_top1_at_least_2of3": sum(row["correct"] for row in retrieval) >= 2,
        "generated_centers_not_grid_locked": generation["correct"]["cell_center_lock_fraction"] < 0.01,
        "generated_renders_finite": all(
            math.isfinite(row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"])
            for row in generation_rows for arm in ("correct", "shuffled", "null")
        ),
        "prompt_only_inference": True,
        "one_shared_scene_per_program": True,
    }
    torch.save({
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": model.architecture(), "text_model": args.text_model,
        "codebook": args.codebook, "styles": styles, "actions": actions,
        "best_step": best_step, "seed": seed,
        "training_posterior_mean": best_state["posterior_mean"],
        "training_posterior_log_std": best_state["posterior_log_std"],
    }, output / "caster.pt")
    torch.save(generated_programs, output / "generated_programs.pt")
    report = {
        "schema": "scene-latent-prompt-native-jewel-caster-gate-v1",
        "protocol": {
            "roots": args.root, "training_fields": len(training),
            "training_sources": [record.source_id for record in training],
            "validation_fields": len(validation),
            "validation_sources": validation_order,
            "exact_prompt_training_sources": exact_counts,
            "scene_dim": args.scene_dim, "kl_weight": args.kl_weight,
            "prior_scene_probability": args.prior_scene_probability,
            "steps_max": args.steps, "steps_completed": history[-1]["step"],
            "best_step": best_step, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "generation_jewels": args.generation_jewels,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_source_latents": True,
            "inference_inputs": ["style_text", "action_text", "declared_random_seed"],
        },
        "history": history, "best_step": best_step,
        "teacher_forced_controls": final_controls,
        "token_nll_improvement_fraction": token_improvement,
        "generation_macro": generation, "generation_records": generation_rows,
        "retrieval": retrieval,
        "inference_audit": {
            "target_field": False, "target_centroids": False,
            "target_tokens": False, "source_pixels": False,
            "video_latent": False, "class_id": False, "source_identity": False,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate": report["gate"], "teacher_forced_controls": final_controls,
        "generation_macro": generation, "retrieval": retrieval,
        "best_step": best_step,
    }, indent=2))


if __name__ == "__main__":
    main()
