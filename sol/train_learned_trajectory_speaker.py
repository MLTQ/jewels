"""Train the Gate 2b1 learned text-to-trajectory-token speaker."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import torch

from sol.audit_jewel_casting_language import load_field_records
from sol.audit_scene_block_constellation_oracle import scene_key
from sol.learned_trajectory_speaker import LearnedTrajectorySpeaker, trajectory_program_loss
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


PROMPT_PARAPHRASES = {
    "a ballerina spinning a pirouette in a studio": {
        "train": (
            "an anime ballerina performing a spinning pirouette in a dance studio",
            "a ballet dancer spins gracefully in a studio",
        ),
        "evaluation": "an animated ballet performer twirling inside a rehearsal room",
    },
    "a golden retriever catching a ball on grass": {
        "train": (
            "a cartoon golden retriever leaps to catch a ball on grass",
            "an animated dog chasing and catching a ball in a park",
        ),
        "evaluation": "a playful illustrated retriever catches a thrown ball outdoors",
    },
    "a welder joining steel with bright sparks": {
        "train": (
            "a 3D rendered welder joining metal while bright sparks fly",
            "a worker welding steel in a workshop filled with sparks",
        ),
        "evaluation": "a digital workshop scene of someone fusing metal with a shower of sparks",
    },
}


def build_program_examples(
    exact_prompts: tuple[str, ...],
    scene_sources: tuple[tuple[int, ...], ...],
) -> tuple[list[dict], list[dict]]:
    """Enumerate training prompts/pairs and held-out cyclic pairs/paraphrases."""
    if set(exact_prompts) != set(PROMPT_PARAPHRASES):
        raise ValueError("registered exact prompts do not match Gate 2b1 paraphrases")
    training, evaluation = [], []
    for scene, (prompt, sources) in enumerate(zip(exact_prompts, scene_sources)):
        if len(sources) != 6:
            raise ValueError("Gate 2b1 requires six source tokens per scene")
        heldout = {
            (sources[index], sources[(index + 1) % len(sources)])
            for index in range(len(sources))
        }
        train_prompts = (prompt,) + PROMPT_PARAPHRASES[prompt]["train"]
        for foreground in sources:
            for background in sources:
                if foreground == background:
                    continue
                row = {
                    "scene_token": scene,
                    "foreground_token": foreground,
                    "background_token": background,
                }
                if (foreground, background) in heldout:
                    evaluation.append({
                        **row,
                        "prompt": PROMPT_PARAPHRASES[prompt]["evaluation"],
                    })
                else:
                    training.extend({**row, "prompt": text} for text in train_prompts)
    return training, evaluation


@torch.no_grad()
def evaluate_conditions(
    model: LearnedTrajectorySpeaker,
    evaluation: list[dict],
    embeddings: dict[str, torch.Tensor],
    exact_prompts: tuple[str, ...],
) -> dict:
    """Score held-out program rows under correct, shuffled, and null text."""
    device = next(model.parameters()).device
    scenes = torch.tensor([row["scene_token"] for row in evaluation], device=device)
    foreground = torch.tensor([row["foreground_token"] for row in evaluation], device=device)
    background = torch.tensor([row["background_token"] for row in evaluation], device=device)
    condition_text = {
        "correct": torch.stack([embeddings[row["prompt"]] for row in evaluation]).to(device),
        "shuffled": torch.stack([
            embeddings[PROMPT_PARAPHRASES[
                exact_prompts[(row["scene_token"] + 1) % len(exact_prompts)]
            ]["evaluation"]]
            for row in evaluation
        ]).to(device),
        "null": torch.stack([embeddings[""] for _ in evaluation]).to(device),
    }
    output = {}
    for condition, text in condition_text.items():
        predictions = model(text, scenes, foreground)
        _, parts = trajectory_program_loss(predictions, scenes, foreground, background)
        output[condition] = {
            "token_nll": parts,
            "token_nll_macro": parts["total"] / 3,
            "scene_accuracy": float(
                (predictions["scene_logits"].argmax(dim=1) == scenes).float().mean()
            ),
        }
    return output


def _encode_prompts(prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    import open_clip  # noqa: PLC0415
    model = open_clip.create_model(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    ).to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        rows = model.encode_text(tokenizer(prompts).to(device)).float()
        rows = torch.nn.functional.normalize(rows, dim=1).cpu()
    return dict(zip(prompts, rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--hidden-dimension", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--minimum-steps", type=int, default=1000)
    parser.add_argument("--plateau-evaluations", type=int, default=10)
    args = parser.parse_args()
    frozen = (
        args.steps == 5000 and args.batch_size == 64
        and args.learning_rate == 0.002 and args.hidden_dimension == 256
        and args.eval_every == 100 and args.minimum_steps == 1000
        and args.plateau_evaluations == 10
    )
    if not frozen:
        raise ValueError("Gate 2b1 optimization settings are frozen")
    device = torch.device(args.device)
    seed = 20260917
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    block = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    split_protocol = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split_protocol["roots"]])
    training_records, _ = select_prompt_splits(
        records,
        set(split_protocol["validation_sources"]),
        set(split_protocol["training_sources"]),
    )
    training_records = sorted(training_records, key=lambda record: record.source_id)
    if [record.source_id for record in training_records] != list(block["training_sources"]):
        raise ValueError("Gate 2b1 sources do not align with the fine language")
    keys = sorted({scene_key(record) for record in training_records})
    exact_prompts = tuple(key[1] for key in keys)
    key_to_scene = {key: index for index, key in enumerate(keys)}
    scene_sources = tuple(
        tuple(
            index for index, record in enumerate(training_records)
            if key_to_scene[scene_key(record)] == scene
        )
        for scene in range(len(keys))
    )
    training, evaluation = build_program_examples(exact_prompts, scene_sources)
    prompt_list = sorted({row["prompt"] for row in training + evaluation} | {""})
    embeddings = _encode_prompts(prompt_list, device)
    text_dimension = len(next(iter(embeddings.values())))
    model_args = {
        "text_dimension": text_dimension,
        "hidden_dimension": args.hidden_dimension,
        "scene_tokens": len(keys),
        "source_tokens": len(training_records),
    }
    model = LearnedTrajectorySpeaker(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_text = torch.stack([embeddings[row["prompt"]] for row in training]).to(device)
    train_scene = torch.tensor([row["scene_token"] for row in training], device=device)
    train_foreground = torch.tensor([row["foreground_token"] for row in training], device=device)
    train_background = torch.tensor([row["background_token"] for row in training], device=device)
    null_embedding = embeddings[""].to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    progress = []
    best_nll = float("inf")
    best_step = 0
    best_state = None
    stale = 0
    stop_reason = "maximum_steps"
    for step in range(1, args.steps + 1):
        rows = torch.randint(
            len(training), (args.batch_size,), generator=generator, device=device
        )
        text = train_text[rows].clone()
        dropout = torch.rand(
            args.batch_size, generator=generator, device=device
        ) < 0.10
        text[dropout] = null_embedding
        scene = train_scene[rows]
        foreground = train_foreground[rows]
        background = train_background[rows]
        predictions = model(text, scene, foreground)
        loss, _ = trajectory_program_loss(
            predictions, scene, foreground, background
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0:
            model.eval()
            conditions = evaluate_conditions(
                model, evaluation, embeddings, exact_prompts
            )
            model.train()
            correct_nll = conditions["correct"]["token_nll_macro"]
            row = {
                "step": step,
                "training_loss": float(loss.detach()),
                "conditions": conditions,
            }
            progress.append(row)
            print(json.dumps(row), flush=True)
            if correct_nll < best_nll - 1e-4:
                best_nll = correct_nll
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if step >= args.minimum_steps and stale >= args.plateau_evaluations:
                stop_reason = "heldout_correct_nll_plateau"
                break
    if best_state is None:
        raise RuntimeError("training produced no evaluation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    best_conditions = evaluate_conditions(model, evaluation, embeddings, exact_prompts)
    correct = best_conditions["correct"]["token_nll_macro"]
    shuffled = best_conditions["shuffled"]["token_nll_macro"]
    null = best_conditions["null"]["token_nll_macro"]
    checks = {
        "correct_nll_at_least_20pct_below_shuffled": (shuffled - correct) / shuffled >= 0.20,
        "correct_nll_at_least_10pct_below_null": (null - correct) / null >= 0.10,
        "unseen_paraphrase_scene_top1_3of3": (
            best_conditions["correct"]["scene_accuracy"] == 1.0
        ),
        "stopped_after_frozen_minimum": progress[-1]["step"] >= args.minimum_steps,
        "all_metrics_finite": all(
            torch.isfinite(torch.tensor(condition["token_nll_macro"]))
            for condition in best_conditions.values()
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "learned-trajectory-speaker-checkpoint-v1",
        "model_args": model_args,
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "exact_prompts": exact_prompts,
        "prompt_paraphrases": PROMPT_PARAPHRASES,
        "prompt_embeddings": embeddings,
        "scene_keys": keys,
        "scene_sources": scene_sources,
        "training_sources": [record.source_id for record in training_records],
        "best_step": best_step,
        "seed": seed,
        "text_encoder": "OpenCLIP ViT-B-32 laion2b_s34b_b79k",
    }, output / "speaker.pt")
    (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    report = {
        "schema": "learned-trajectory-speaker-training-v1",
        "protocol": {
            "exact_prompts": exact_prompts,
            "prompt_paraphrases": PROMPT_PARAPHRASES,
            "scene_sources": scene_sources,
            "training_examples": len(training),
            "heldout_examples": len(evaluation),
            "heldout_pair_rule": "cyclic_next_source_within_scene",
            "text_dropout": 0.10,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "maximum_steps": args.steps,
            "minimum_steps": args.minimum_steps,
            "evaluation_interval": args.eval_every,
            "plateau_evaluations": args.plateau_evaluations,
            "plateau_minimum_delta": 1e-4,
            "seed": seed,
        },
        "model_args": model_args,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_step": best_step,
        "completed_step": progress[-1]["step"],
        "stop_reason": stop_reason,
        "best_conditions": best_conditions,
        "gate": {"checks": checks, "token_gate_passed": all(checks.values())},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
