"""Train the text-conditioned latent flow and gate it on text selectivity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from sol.latent_text_prior import (
    ARCHITECTURE,
    LatentStandardizer,
    LatentTextPrior,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-dropout", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pack(cells: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Concatenate cell features and flattened slot seeds into one token vector."""
    return torch.cat((cells, seed.reshape(*seed.shape[:-2], -1)), dim=-1)


def unpack(
    packed: torch.Tensor, cell_dim: int, slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cells = packed[..., :cell_dim]
    seed = packed[..., cell_dim:].reshape(*packed.shape[:-1], slots, 3)
    return cells, seed


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    records = cache["records"]
    slots = int(cache["encoder"]["slots_per_cell"])
    cell_dim = cache["cells"].shape[-1]

    packed = pack(cache["cells"], cache["seed"]).float()
    train_index = [i for i, r in enumerate(records) if r["split"] == "train"]
    validation_index = [i for i, r in enumerate(records) if r["split"] != "train"]
    if not train_index or not validation_index:
        raise ValueError("cache must contain train and validation windows")
    standardizer = LatentStandardizer.fit(packed[train_index])
    normalized = standardizer.normalize(packed).to(device)
    text_tokens = cache["text_tokens"].float().to(device)
    text_mask = cache["text_mask"].to(device)
    prompt_of = torch.tensor([r["prompt_index"] for r in records], device=device)

    model = LatentTextPrior(
        n_cells=normalized.shape[1],
        cell_dim=cell_dim,
        seed_dim=slots * 3,
        text_dim=int(cache["text_dim"]),
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    @torch.no_grad()
    def evaluate() -> dict:
        """Fixed-path velocity error under correct, shuffled, and null text."""
        model.eval()
        evaluation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        index = torch.tensor(validation_index, device=device)
        target = normalized[index]
        noise = torch.randn(
            target.shape, device=device, generator=evaluation_generator
        )
        flow_time = torch.rand(
            len(index), device=device, generator=evaluation_generator
        )
        noisy = (1 - flow_time[:, None, None]) * noise + flow_time[
            :, None, None
        ] * target
        expected = target - noise
        correct = prompt_of[index]
        shuffled = correct.roll(1)
        report = {}
        for name, prompts in (("correct", correct), ("shuffled", shuffled)):
            predicted = model(noisy, flow_time, text_tokens[prompts], text_mask[prompts])
            report[name] = float(torch.nn.functional.mse_loss(predicted, expected))
        predicted = model(noisy, flow_time, None, None)
        report["null"] = float(torch.nn.functional.mse_loss(predicted, expected))
        report["shuffled_minus_correct"] = report["shuffled"] - report["correct"]
        report["null_minus_correct"] = report["null"] - report["correct"]
        model.train()
        return report

    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"train={len(train_index)} validation={len(validation_index)} "
        f"cells={normalized.shape[1]} feature_dim={normalized.shape[2]} "
        f"model={parameters / 1e6:.2f}M",
        flush=True,
    )
    latest = None
    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = torch.tensor(
            [
                train_index[int(i)]
                for i in torch.randint(
                    0, len(train_index), (args.batch_size,), generator=generator,
                    device=device,
                )
            ],
            device=device,
        )
        target = normalized[batch]
        noise = torch.randn(target.shape, device=device, generator=generator)
        flow_time = torch.rand(len(batch), device=device, generator=generator)
        noisy = (1 - flow_time[:, None, None]) * noise + flow_time[
            :, None, None
        ] * target
        drop = (
            torch.rand(len(batch), device=device, generator=generator)
            < args.text_dropout
        )
        prompts = prompt_of[batch]
        tokens = text_tokens[prompts].clone()
        mask = text_mask[prompts].clone()
        if drop.any():
            tokens[drop] = 0.0
            mask[drop] = 0
            mask[drop, 0] = 1
        if step <= args.warmup:
            rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        predicted = model(noisy, flow_time, tokens, mask)
        loss = torch.nn.functional.mse_loss(predicted, target - noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            recent = history[-args.log_every :]
            record = {
                "step": step,
                "loss": sum(recent) / len(recent),
                "gradient_norm": float(gradient_norm),
                "lr": rate,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate()
            record = {"step": step, "evaluation": latest}
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "meta": {
                        "architecture": ARCHITECTURE,
                        "model_args": {
                            "n_cells": normalized.shape[1],
                            "cell_dim": cell_dim,
                            "seed_dim": slots * 3,
                            "text_dim": int(cache["text_dim"]),
                            "model_dim": args.model_dim,
                            "depth": args.depth,
                            "heads": args.heads,
                        },
                        "standardizer": standardizer.state_dict(),
                        "latents": args.latents,
                        "encoder": cache["encoder"],
                        "text_model": cache["text_model"],
                        "train_args": vars(args),
                        "latest_evaluation": latest,
                    },
                },
                output_dir / "prior.pt",
            )
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "parameters": parameters,
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
