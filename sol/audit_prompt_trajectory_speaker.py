"""Audit prompt-only scene/trajectory/background Jewel programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from PIL import Image
import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import _render, center_irregularity, load_field_records
from sol.audit_scene_block_constellation_oracle import scene_key
from sol.block_token_language import BlockTokenCodebook
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.prompt_jewel_caster import active_tokens_to_features
from sol.prompt_trajectory_speaker import PromptTrajectorySpeaker
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.semantic_trajectory_realizer import fit_semantic_trajectory_realizer
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


CONDITIONS = ("correct prompt", "cyclic-shuffled prompt", "null prompt")
SEEDS = (20260914, 20260915, 20260916)


@torch.no_grad()
def clip_video_embedding(frames: torch.Tensor, model) -> torch.Tensor:
    """Embed a rendered three-frame video in the frozen CLIP image space."""
    resized = F.interpolate(
        frames.permute(0, 3, 1, 2), size=(224, 224),
        mode="bicubic", align_corners=False,
    )
    mean = resized.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = resized.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    encoded = model.encode_image((resized - mean) / std).float()
    return F.normalize(encoded.mean(dim=0), dim=0)


def semantic_summary(records: list[dict]) -> dict:
    """Aggregate the frozen retrieval and causal prompt metrics."""
    retrieval = sum(row["correct_top1"] for row in records)
    classes = sorted({row["scene_token"] for row in records})
    majority_classes = sum(
        sum(row["correct_top1"] for row in records if row["scene_token"] == scene) >= 2
        for scene in classes
    )
    correct = [row["correct_similarity"] for row in records]
    shuffled = [row["shuffled_generation_similarity"] for row in records]
    null = [row["null_generation_similarity"] for row in records]
    return {
        "programs": len(records),
        "correct_top1": retrieval,
        "correct_top1_fraction": retrieval / len(records),
        "classes_with_majority_retrieval": majority_classes,
        "correct_similarity_mean": statistics.mean(correct),
        "shuffled_generation_similarity_mean": statistics.mean(shuffled),
        "null_generation_similarity_mean": statistics.mean(null),
        "correct_minus_shuffled_generation": statistics.mean(
            a - b for a, b in zip(correct, shuffled)
        ),
        "correct_minus_null_generation": statistics.mean(
            a - b for a, b in zip(correct, null)
        ),
        "correct_beats_shuffled_generation": sum(
            a > b for a, b in zip(correct, shuffled)
        ),
        "correct_beats_null_generation": sum(a > b for a, b in zip(correct, null)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=216)
    args = parser.parse_args()
    if (
        args.generation_jewels != 72000 or args.frames != 49
        or args.height != 144 or args.width != 216
    ):
        raise ValueError("Gate 2b0 render/count settings are frozen")
    device = torch.device(args.device)
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if (
        int(architecture["block_vocabulary_size"]) != 1024
        or tuple(architecture["block_shape"]) != (16, 16, 8)
    ):
        raise ValueError("Gate 2b0 requires the frozen fine K=1024 language")
    block_codebook = BlockTokenCodebook.from_state_dict(
        checkpoint["block_codebook"], device
    )
    split_protocol = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split_protocol["roots"]])
    training, _ = select_prompt_splits(
        records,
        set(split_protocol["validation_sources"]),
        set(split_protocol["training_sources"]),
    )
    training = sorted(training, key=lambda record: record.source_id)
    if [record.source_id for record in training] != list(checkpoint["training_sources"]):
        raise ValueError("fine language and prompt speaker sources do not align")
    keys = sorted({scene_key(record) for record in training})
    prompts = tuple(key[1] for key in keys)
    key_to_scene = {key: index for index, key in enumerate(keys)}
    training_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in training],
        dtype=torch.long, device=device,
    )
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_semantic_trajectory_realizer(
        [record.features.to(device) for record in training],
        training_scenes,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=0.005,
    )
    scene_sources = tuple(
        tuple(torch.nonzero(training_scenes == scene).flatten().cpu().tolist())
        for scene in range(len(prompts))
    )
    speaker = PromptTrajectorySpeaker(prompts=prompts, scene_sources=scene_sources)

    import open_clip  # noqa: PLC0415
    clip = open_clip.create_model(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    ).to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = clip.encode_text(tokenizer(list(prompts) + [""]).to(device)).float()
    text = F.normalize(text, dim=1)

    frame_indices = torch.tensor([0, 24, 48], dtype=torch.long)
    points = frame_points(
        args.frames, frame_indices, args.height, args.width, device=device
    )
    audit_spec = GridSpec((8, 8, 4), slots_per_cell=1)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    image_rows = {seed: [] for seed in SEEDS}
    rows = []
    all_programs = []
    for scene, prompt in enumerate(prompts):
        label = keys[scene][0]
        for seed in SEEDS:
            programs = {
                "correct prompt": speaker.compile(prompt, seed),
                "cyclic-shuffled prompt": speaker.compile_shuffled(prompt, seed),
                "null prompt": speaker.compile_null(seed),
            }
            arms = {}
            for condition, program in programs.items():
                generator = torch.Generator(device=device).manual_seed(
                    seed + 100000 * scene
                )
                centers, tokens, realization = realizer.sample_rank_balanced_from_donors(
                    program.scene_token,
                    program.foreground_token,
                    program.background_token,
                    args.generation_jewels,
                    generator=generator,
                )
                features = active_tokens_to_features(centers, tokens, physical_codebook)
                background = training[program.background_token].background
                rendered = _render(features, points, background).reshape(
                    3, args.height, args.width, 3
                ).clamp(0, 1)
                embedding = clip_video_embedding(rendered, clip)
                named_realization = dict(realization)
                named_realization["foreground_source_id"] = training[
                    program.foreground_token
                ].source_id
                named_realization["background_source_id"] = training[
                    program.background_token
                ].source_id
                arms[condition] = {
                    "program": program.to_dict(),
                    "realization": named_realization,
                    "embedding": embedding.cpu(),
                    "finite_render": bool(torch.isfinite(rendered).all()),
                    "center_irregularity": center_irregularity(features, audit_spec),
                }
                all_programs.append(program.to_dict())
                image_rows[seed].append(_row([
                    _panel(
                        frame,
                        f"{label} / {condition} / t={int(index)}",
                    )
                    for frame, index in zip(rendered, frame_indices)
                ]))
            intended = text[scene].cpu()
            correct_embedding = arms["correct prompt"]["embedding"]
            similarities = correct_embedding @ text[:len(prompts)].cpu().T
            row = {
                "intended_prompt": prompt,
                "scene_token": scene,
                "seed": seed,
                "correct_similarity": float(correct_embedding @ intended),
                "correct_same_render_shuffled_text_similarity": float(
                    correct_embedding @ text[(scene + 1) % len(prompts)].cpu()
                ),
                "shuffled_generation_similarity": float(
                    arms["cyclic-shuffled prompt"]["embedding"] @ intended
                ),
                "null_generation_similarity": float(
                    arms["null prompt"]["embedding"] @ intended
                ),
                "correct_retrieved_scene": int(similarities.argmax()),
                "correct_top1": int(similarities.argmax()) == scene,
                "arms": {
                    condition: {
                        key: value for key, value in arm.items() if key != "embedding"
                    } for condition, arm in arms.items()
                },
            }
            rows.append(row)
            print("rendered prompt program", label, seed, flush=True)
    for seed, seed_rows in image_rows.items():
        sheet = Image.new(
            "RGB",
            (seed_rows[0].width,
             sum(row.height for row in seed_rows) + 3 * (len(seed_rows) - 1)),
            "white",
        )
        offset = 0
        for row in seed_rows:
            sheet.paste(row, (0, offset))
            offset += row.height + 3
        sheet.save(output / f"qualitative_seed{seed}.png")
    summary = semantic_summary(rows)
    realization_rows = [
        row["arms"][condition]["realization"]
        for row in rows for condition in CONDITIONS
    ]
    checks = {
        "correct_top1_at_least_6of9": summary["correct_top1"] >= 6,
        "majority_retrieval_in_at_least_2of3_classes": (
            summary["classes_with_majority_retrieval"] >= 2
        ),
        "correct_minus_shuffled_generation_at_least_002": (
            summary["correct_minus_shuffled_generation"] >= 0.02
        ),
        "correct_minus_null_generation_at_least_001": (
            summary["correct_minus_null_generation"] >= 0.01
        ),
        "correct_beats_shuffled_at_least_7of9": (
            summary["correct_beats_shuffled_generation"] >= 7
        ),
        "all_donors_distinct": all(
            item["foreground_training_field"] != item["background_training_field"]
            for item in realization_rows
        ),
        "all_programs_exactly_half_owned": all(
            item["foreground_fraction"] == 0.5
            and item["background_fraction"] == 0.5
            for item in realization_rows
        ),
        "all_counts_exact_without_adjustment": all(
            item["emitted_jewels"] == args.generation_jewels
            and item["adjustment_fraction"] == 0.0
            for item in realization_rows
        ),
        "all_renders_finite": all(
            row["arms"][condition]["finite_render"]
            for row in rows for condition in CONDITIONS
        ),
        "generated_centers_not_grid_locked": max(
            row["arms"][condition]["center_irregularity"][
                "cell_center_lock_fraction"
            ] for row in rows for condition in CONDITIONS
        ) < 0.01,
    }
    report = {
        "schema": "prompt-only-trajectory-token-speaker-v1",
        "protocol": {
            "prompts": prompts,
            "scene_keys": keys,
            "seeds": SEEDS,
            "training_sources": [record.source_id for record in training],
            "semantic_inputs": ["exact prompt text", "declared integer seed"],
            "target_video_input": False,
            "target_field_input": False,
            "target_block_program_input": False,
            "heldout_latent_input": False,
            "source_token_backed_vocabulary": True,
            "complete_video_retrieval": False,
            "open_vocabulary": False,
            "learned_language_model": False,
            "generation_jewels": args.generation_jewels,
            "render_shape": [args.frames, args.height, args.width],
            "rendered_frame_indices": frame_indices.tolist(),
            "semantic_evaluator": "OpenCLIP ViT-B-32 laion2b_s34b_b79k",
        },
        "fit": fit_report,
        "programs": all_programs,
        "summary": summary,
        "records": rows,
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_recognizability_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
