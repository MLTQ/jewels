"""Render and score programs sampled from the learned trajectory-token speaker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import _render, center_irregularity, load_field_records
from sol.audit_prompt_trajectory_speaker import clip_video_embedding, semantic_summary
from sol.audit_scene_block_constellation_oracle import scene_key
from sol.block_token_language import BlockTokenCodebook
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.learned_trajectory_speaker import LearnedTrajectorySpeaker
from sol.prompt_jewel_caster import active_tokens_to_features
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.semantic_trajectory_realizer import fit_semantic_trajectory_realizer
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


CONDITIONS = ("correct text", "cyclic-shuffled text", "null text")
SEEDS = (20260918, 20260919, 20260920)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--training-report", required=True)
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
        raise ValueError("Gate 2b1 render/count settings are frozen")
    device = torch.device(args.device)
    saved = torch.load(args.speaker, map_location="cpu", weights_only=False)
    if saved.get("schema") != "learned-trajectory-speaker-checkpoint-v1":
        raise ValueError("unsupported learned speaker checkpoint")
    model = LearnedTrajectorySpeaker(**saved["model_args"]).to(device).eval()
    model.load_state_dict(saved["model"])
    embeddings = {
        prompt: row.to(device) for prompt, row in saved["prompt_embeddings"].items()
    }
    exact_prompts = tuple(saved["exact_prompts"])
    paraphrases = saved["prompt_paraphrases"]
    evaluation_prompts = tuple(paraphrases[prompt]["evaluation"] for prompt in exact_prompts)

    block = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    block_codebook = BlockTokenCodebook.from_state_dict(block["block_codebook"], device)
    split_protocol = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split_protocol["roots"]])
    training, _ = select_prompt_splits(
        records,
        set(split_protocol["validation_sources"]),
        set(split_protocol["training_sources"]),
    )
    training = sorted(training, key=lambda record: record.source_id)
    if [record.source_id for record in training] != list(saved["training_sources"]):
        raise ValueError("learned speaker and Jewel source tokens do not align")
    keys = sorted({scene_key(record) for record in training})
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

    import open_clip  # noqa: PLC0415
    clip = open_clip.create_model(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    ).to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    semantic_text = clip.encode_text(tokenizer(list(evaluation_prompts) + [""]).to(device)).float()
    semantic_text = F.normalize(semantic_text, dim=1)

    frame_indices = torch.tensor([0, 24, 48], dtype=torch.long)
    points = frame_points(
        args.frames, frame_indices, args.height, args.width, device=device
    )
    audit_spec = GridSpec((8, 8, 4), slots_per_cell=1)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    image_rows = {seed: [] for seed in SEEDS}
    rows = []
    for intended_scene, intended_prompt in enumerate(evaluation_prompts):
        label = keys[intended_scene][0]
        for seed in SEEDS:
            condition_prompts = {
                "correct text": intended_prompt,
                "cyclic-shuffled text": evaluation_prompts[
                    (intended_scene + 1) % len(evaluation_prompts)
                ],
                "null text": "",
            }
            arms = {}
            for condition, prompt in condition_prompts.items():
                token_generator = torch.Generator(device=device).manual_seed(
                    seed + 100000 * intended_scene
                )
                program = model.sample(
                    embeddings[prompt][None],
                    generator=token_generator,
                    temperature=0.8,
                    top_k=6,
                )
                field_generator = torch.Generator(device=device).manual_seed(
                    seed + 500000 + 100000 * intended_scene
                )
                centers, tokens, realization = realizer.sample_rank_balanced_from_donors(
                    program.scene_token,
                    program.foreground_token,
                    program.background_token,
                    args.generation_jewels,
                    generator=field_generator,
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
                    "conditioning_text": prompt,
                    "program": {
                        "scene_token": program.scene_token,
                        "foreground_token": program.foreground_token,
                        "background_token": program.background_token,
                    },
                    "realization": named_realization,
                    "embedding": embedding.cpu(),
                    "donors_match_predicted_scene": bool(
                        int(training_scenes[program.foreground_token]) == program.scene_token
                        and int(training_scenes[program.background_token]) == program.scene_token
                    ),
                    "finite_render": bool(torch.isfinite(rendered).all()),
                    "center_irregularity": center_irregularity(features, audit_spec),
                }
                image_rows[seed].append(_row([
                    _panel(frame, f"{label} / {condition} / t={int(index)}")
                    for frame, index in zip(rendered, frame_indices)
                ]))
            intended = semantic_text[intended_scene].cpu()
            correct_embedding = arms["correct text"]["embedding"]
            similarities = correct_embedding @ semantic_text[:len(evaluation_prompts)].cpu().T
            rows.append({
                "intended_prompt": intended_prompt,
                "scene_token": intended_scene,
                "seed": seed,
                "correct_similarity": float(correct_embedding @ intended),
                "shuffled_generation_similarity": float(
                    arms["cyclic-shuffled text"]["embedding"] @ intended
                ),
                "null_generation_similarity": float(
                    arms["null text"]["embedding"] @ intended
                ),
                "correct_retrieved_scene": int(similarities.argmax()),
                "correct_top1": int(similarities.argmax()) == intended_scene,
                "arms": {
                    condition: {
                        key: value for key, value in arm.items() if key != "embedding"
                    } for condition, arm in arms.items()
                },
            })
            print("rendered learned program", label, seed, flush=True)
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
    correct_arms = [row["arms"]["correct text"] for row in rows]
    all_arms = [row["arms"][condition] for row in rows for condition in CONDITIONS]
    scene_consistent = sum(arm["donors_match_predicted_scene"] for arm in correct_arms)
    training_report = json.loads(Path(args.training_report).read_text())
    checks = {
        "training_token_gate_passed": training_report["gate"]["token_gate_passed"],
        "correct_program_donors_match_scene_at_least_8of9": scene_consistent >= 8,
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
            arm["program"]["foreground_token"] != arm["program"]["background_token"]
            for arm in all_arms
        ),
        "all_programs_exactly_half_owned": all(
            arm["realization"]["foreground_fraction"] == 0.5
            and arm["realization"]["background_fraction"] == 0.5
            for arm in all_arms
        ),
        "all_counts_exact_without_adjustment": all(
            arm["realization"]["emitted_jewels"] == args.generation_jewels
            and arm["realization"]["adjustment_fraction"] == 0.0
            for arm in all_arms
        ),
        "all_renders_finite": all(arm["finite_render"] for arm in all_arms),
        "generated_centers_not_grid_locked": max(
            arm["center_irregularity"]["cell_center_lock_fraction"]
            for arm in all_arms
        ) < 0.01,
    }
    report = {
        "schema": "learned-prompt-to-trajectory-speaker-audit-v1",
        "protocol": {
            "evaluation_prompts": evaluation_prompts,
            "seeds": SEEDS,
            "semantic_inputs": ["unseen prompt paraphrase", "declared integer seed"],
            "target_video_input": False,
            "target_field_input": False,
            "target_block_program_input": False,
            "heldout_latent_input": False,
            "source_token_logits_masked_by_scene": False,
            "repeated_background_token_masked": True,
            "source_token_backed_vocabulary": True,
            "open_vocabulary": False,
            "learned_program_speaker": True,
            "render_shape": [args.frames, args.height, args.width],
            "rendered_frame_indices": frame_indices.tolist(),
            "semantic_evaluator": "OpenCLIP ViT-B-32 laion2b_s34b_b79k",
        },
        "speaker_checkpoint": args.speaker,
        "speaker_best_step": saved["best_step"],
        "realizer_fit": fit_report,
        "summary": {**summary, "correct_scene_consistent_programs": scene_consistent},
        "records": rows,
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_recognizability_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
