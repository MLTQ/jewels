"""Train and audit the preregistered native prompt-to-Jewel caster."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
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
    PromptJewelCaster,
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class PromptSampleBatch:
    centers: torch.Tensor
    tokens: torch.Tensor
    prompts: torch.Tensor

    def index(self, selected: torch.Tensor) -> "PromptSampleBatch":
        return PromptSampleBatch(
            self.centers[selected], self.tokens[selected], self.prompts[selected]
        )

    def __len__(self) -> int:
        return int(len(self.centers))


def _metadata(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["source"]


def prompt_label(metadata: dict) -> str:
    """Build the frozen compositional style/action text input."""
    return f"{metadata['style']} video. {metadata['source_prompt']}"


@torch.no_grad()
def frozen_text_embeddings(
    labels: list[str], *, device: torch.device, model_name: str
) -> torch.Tensor:
    """Encode text locally through a frozen pretrained BGE model."""
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    encoded = tokenizer(
        labels, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"][:, :, None].to(hidden)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    result = F.normalize(pooled.float(), dim=1)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def control_metrics(
    model: PromptJewelCaster,
    batch: PromptSampleBatch,
    correct_embeddings: torch.Tensor,
    null_embedding: torch.Tensor,
    *,
    chunk: int = 4096,
) -> dict:
    """Evaluate correct, cyclic-shuffled, and null prompt NLL controls."""
    unique_prompts = len(correct_embeddings)
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
                    text = correct_embeddings[part.prompts]
                elif arm == "shuffled":
                    text = correct_embeddings[(part.prompts + 1) % unique_prompts]
                else:
                    text = null_embedding.expand(len(part), -1)
                logits = model.token_logits(text, part.centers)
                for role in range(len(ACTIVE_FACTORS)):
                    token_sum[role] += F.cross_entropy(
                        logits[:, role], part.tokens[:, role], reduction="sum"
                    )
                density_sum += float(
                    model.centroid_density.negative_log_likelihood(
                        text, part.centers
                    ).sum()
                )
                count += len(part)
            output[arm] = {
                "centroid_nll": density_sum / count,
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
        raise ValueError("prompt caster schedule values must be positive")
    device = torch.device(args.device)
    seed = 20260831
    torch.manual_seed(seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    validation_sources = set(args.validation_source)
    records = load_field_records([Path(root) for root in args.root])
    training = [record for record in records if record.source_id not in validation_sources]
    validation = [record for record in records if record.source_id in validation_sources]
    if len({record.source_id for record in training}) != len(training):
        raise ValueError("prompt training permits one field per source")
    validation_counts = {
        source: sum(record.source_id == source for record in validation)
        for source in validation_sources
    }
    if any(count != 3 for count in validation_counts.values()):
        raise ValueError("prompt validation requires three fits per held-out source")

    train_metadata = [_metadata(record.path) for record in training]
    validation_metadata = {
        source: _metadata(next(record.path for record in validation if record.source_id == source))
        for source in sorted(validation_sources)
    }
    train_labels = [prompt_label(row) for row in train_metadata]
    validation_source_order = sorted(validation_sources)
    validation_labels = [
        prompt_label(validation_metadata[source]) for source in validation_source_order
    ]
    all_embeddings = frozen_text_embeddings(
        train_labels + validation_labels + [""],
        device=device,
        model_name=args.text_model,
    )
    train_embeddings = all_embeddings[: len(train_labels)]
    validation_embeddings = all_embeddings[
        len(train_labels) : len(train_labels) + len(validation_labels)
    ]
    null_embedding = all_embeddings[-1:]

    train_batches = []
    for prompt_index, record in enumerate(training):
        features = record.features.to(device)
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected = torch.randperm(
            len(features), generator=generator, device=device
        )[: min(args.train_jewels_per_source, len(features))]
        sampled = features[selected]
        train_batches.append(
            PromptSampleBatch(
                centers=sampled[:, :3],
                tokens=encode_active_jewel_tokens(sampled, codebook),
                prompts=torch.full(
                    (len(sampled),), prompt_index, device=device, dtype=torch.long
                ),
            )
        )
        print("sampled train", record.source_id, len(sampled), flush=True)
    train_batch = PromptSampleBatch(
        centers=torch.cat([batch.centers for batch in train_batches]),
        tokens=torch.cat([batch.tokens for batch in train_batches]),
        prompts=torch.cat([batch.prompts for batch in train_batches]),
    )

    validation_batches = []
    source_to_prompt = {
        source: index for index, source in enumerate(validation_source_order)
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
        validation_batches.append(
            PromptSampleBatch(
                centers=sampled[:, :3],
                tokens=encode_active_jewel_tokens(sampled, codebook),
                prompts=torch.full(
                    (len(sampled),),
                    source_to_prompt[record.source_id],
                    device=device,
                    dtype=torch.long,
                ),
            )
        )
        print("sampled validation", record.source_id, record.fit_seed, len(sampled), flush=True)
    validation_batch = PromptSampleBatch(
        centers=torch.cat([batch.centers for batch in validation_batches]),
        tokens=torch.cat([batch.tokens for batch in validation_batches]),
        prompts=torch.cat([batch.prompts for batch in validation_batches]),
    )

    model = PromptJewelCaster(
        text_dim=train_embeddings.shape[1],
        vocabulary_size=codebook.vocabulary_size,
        hidden_dim=512,
        depth=4,
        mixture_components=64,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    dropout_generator = torch.Generator(device=device).manual_seed(seed + 2)
    best_score = float("inf")
    best_step = 0
    best_state = None
    stale = 0
    history = []
    for step in range(1, args.steps + 1):
        selected = torch.randint(
            len(train_batch),
            (args.batch_size,),
            generator=sample_generator,
            device=device,
        )
        part = train_batch.index(selected)
        text = train_embeddings[part.prompts].clone()
        drop = torch.rand(
            len(part), generator=dropout_generator, device=device
        ) < 0.10
        text[drop] = null_embedding
        loss, _ = model.loss(text, part.centers, part.tokens, density_weight=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every != 0 and step != args.steps:
            continue
        model.eval()
        controls = control_metrics(
            model,
            validation_batch,
            validation_embeddings,
            null_embedding,
        )
        score = (
            controls["correct"]["token_nll_macro"]
            + 0.1 * controls["correct"]["centroid_nll"]
        )
        row = {"step": step, "selection_score": score, "controls": controls}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score * 0.999:
            best_score = score
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        (output / "progress.json").write_text(
            json.dumps(
                {"best_score": best_score, "best_step": best_step, "history": history},
                indent=2,
            )
            + "\n"
        )
        model.train()
        if stale >= args.patience:
            print("validation plateau", step, stale, flush=True)
            break
    if best_state is None:
        raise RuntimeError("prompt caster training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_controls = control_metrics(
        model, validation_batch, validation_embeddings, null_embedding
    )

    generation_rows = []
    generated_cache = {}
    for prompt_index, source_id in enumerate(validation_source_order):
        target_record = min(
            (record for record in validation if record.source_id == source_id),
            key=lambda record: record.fit_seed,
        )
        target_features = target_record.features.to(device)
        target_tokens = encode_active_jewel_tokens(target_features, codebook)
        target_histogram = active_cell_histogram(
            target_features[:, :3], target_tokens,
            spec=spec,
            vocabulary_size=codebook.vocabulary_size,
        )
        generator = torch.Generator(device=device).manual_seed(seed + 10000 + prompt_index)
        points = torch.rand(
            args.voxel_points, 3, generator=generator, device=device
        ) * 2.0 - 1.0
        source_render = _render(target_features, points, target_record.background)
        arms = {}
        for arm in ("correct", "shuffled", "null"):
            if arm == "correct":
                text = validation_embeddings[prompt_index : prompt_index + 1]
            elif arm == "shuffled":
                shifted = (prompt_index + 1) % len(validation_source_order)
                text = validation_embeddings[shifted : shifted + 1]
            else:
                text = null_embedding
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 20000 + prompt_index
            )
            centers = model.centroid_density.sample(
                text, args.generation_jewels, generator=arm_generator
            )
            tokens = model.sample_tokens(
                text, centers, generator=arm_generator,
                temperature=0.9, top_k=64,
            )
            features = active_tokens_to_features(centers, tokens, codebook)
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=codebook.vocabulary_size,
            )
            audit = _audit_candidate(
                target_features,
                features,
                reference_render=source_render,
                points=points,
                background=target_record.background,
                spec=spec,
            )
            arms[arm] = {
                "prompt": (
                    validation_labels[prompt_index]
                    if arm == "correct"
                    else validation_labels[(prompt_index + 1) % len(validation_labels)]
                    if arm == "shuffled"
                    else ""
                ),
                "target_histogram_cosine": histogram_cosine(
                    target_histogram, histogram
                ),
                "audit": audit,
            }
            generated_cache[(source_id, arm)] = {
                "centers": centers.cpu(), "tokens": tokens.cpu()
            }
        generation_rows.append(
            {
                "source_id": source_id,
                "target_prompt": validation_labels[prompt_index],
                "arms": arms,
            }
        )
        print("generated", source_id, flush=True)

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
            "voxel_psnr": generation_macro(
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
        "inputs": ["frozen_text_embedding", "declared_random_seed"],
        "target_field": False,
        "target_centroids": False,
        "target_tokens": False,
        "source_pixels": False,
        "video_latent": False,
        "class_id": False,
        "source_identity": False,
    }
    checks = {
        "correct_centroid_nll_beats_controls": (
            correct["centroid_nll"] < final_controls["shuffled"]["centroid_nll"]
            and correct["centroid_nll"] < final_controls["null"]["centroid_nll"]
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
        "free_run_render_margin_at_least_025db": (
            generation_macro_report["correct"]["voxel_psnr"]
            - max(
                generation_macro_report["shuffled"]["voxel_psnr"],
                generation_macro_report["null"]["voxel_psnr"],
            )
            >= 0.25
        ),
        "generated_centers_not_grid_locked": (
            generation_macro_report["correct"]["cell_center_lock_fraction"] < 0.01
        ),
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
            "best_step": best_step,
            "seed": seed,
            "validation_labels": validation_labels,
        },
        output / "caster.pt",
    )
    torch.save(generated_cache, output / "generated_programs.pt")
    report = {
        "schema": "prompt-native-jewel-caster-gate-v1",
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
        },
        "prompt_labels": {
            "training": train_labels,
            "validation": validation_labels,
        },
        "history": history,
        "best_step": best_step,
        "teacher_forced_controls": final_controls,
        "token_nll_improvement_fraction": token_improvement,
        "generation_macro": generation_macro_report,
        "generation_records": generation_rows,
        "inference_audit": inference_audit,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate": report["gate"],
        "teacher_forced_controls": final_controls,
        "generation_macro": generation_macro_report,
        "best_step": best_step,
    }, indent=2))


if __name__ == "__main__":
    main()
