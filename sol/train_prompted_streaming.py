"""Train prompt-conditioned continuation directly on canonical jewel births."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from sol.prompt_embeddings import load_prompt_cache
from sol.prompted_streaming_eval import evaluate_prompted_streaming
from sol.streaming_corpus import (
    PromptedContinuationCorpus,
    build_prompted_continuation_corpus,
    load_prompted_fields,
)
from sol.streaming_data import BirthTarget, rasterize_context
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class PreparedPromptedView:
    context_raster: torch.Tensor
    target: BirthTarget
    prompt_indices: tuple[int, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--cell-depth", type=int, default=2)
    parser.add_argument("--slot-depth", type=int, default=2)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--prefix-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--count-weight", type=float, default=0.25)
    parser.add_argument("--balance-count-loss", action="store_true")
    parser.add_argument("--text-dropout", type=float, default=0.15)
    parser.add_argument("--context-dropout", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _prepare(
    corpus: PromptedContinuationCorpus, device: torch.device
) -> list[PreparedPromptedView]:
    prepared = []
    for example in corpus.train:
        for view in example.dataset.views:
            context = rasterize_context(
                view.context_features,
                corpus.context_standardizer,
                prefix_frames=example.dataset.prefix_frames,
                stride_frames=example.dataset.stride_frames,
                grid_shape=example.dataset.grid_spec.shape,
            ).to(device)
            births = view.births
            target = BirthTarget(
                values=corpus.birth_standardizer.normalize(births.values).to(device),
                cell_indices=births.cell_indices.to(device),
                slot_indices=births.slot_indices.to(device),
                counts=births.counts.to(device),
                global_ids=births.global_ids.to(device),
                birth_frames=births.birth_frames.to(device),
            )
            prepared.append(
                PreparedPromptedView(context, target, example.train_prompt_indices)
            )
    return prepared


def _atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_json(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.lr <= 0 or args.count_weight < 0:
        raise ValueError("training arguments are outside their valid ranges")
    if not 0 <= args.text_dropout <= 1 or not 0 <= args.context_dropout <= 1:
        raise ValueError("condition dropout probabilities must be in [0,1]")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        torch.backends.cuda.enable_flash_sdp(False)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    spec = GridSpec(tuple(args.grid), args.slots)
    corpus = build_prompted_continuation_corpus(
        fields,
        prompt_cache.embeddings,
        prefix_frames=args.prefix_frames,
        stride_frames=args.stride_frames,
        support_sigma=args.support_sigma,
        grid_spec=spec,
    )
    prepared = _prepare(corpus, device)
    model_args = {
        "feature_dim": 22,
        "context_dim": 46,
        "model_dim": args.model_dim,
        "grid_spec": spec,
        "context_depth": args.context_depth,
        "cell_depth": args.cell_depth,
        "slot_depth": args.slot_depth,
        "context_mode": "local",
        "text_dim": int(prompt_cache.embeddings.shape[1]),
    }
    model = BirthContinuationModel(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "prompted_continuation.pt"
    log_path = output_dir / "train_log.jsonl"
    start_step = 0
    if args.resume and model_path.exists():
        saved = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"])

    def save(step: int, evaluation: dict | None) -> None:
        serializable_args = dict(model_args)
        serializable_args.pop("grid_spec")
        _atomic_save(
            model_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "meta": {
                    "architecture": "prompted_birth_continuation_v1",
                    "model_args": serializable_args,
                    "grid_shape": spec.shape,
                    "slots_per_cell": spec.slots_per_cell,
                    "manifest": args.manifest,
                    "manifest_sha256": prompt_cache.manifest_sha256,
                    "prompt_encoder": prompt_cache.encoder,
                    "training_sources": [example.source_id for example in corpus.train],
                    "validation_sources": [
                        example.source_id for example in corpus.validation
                    ],
                    "context_standardizer": corpus.context_standardizer.state_dict(),
                    "birth_standardizer": corpus.birth_standardizer.state_dict(),
                    "train_args": vars(args),
                    "latest_evaluation": evaluation,
                },
            },
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"train_sources={len(corpus.train)} validation_sources={len(corpus.validation)} "
        f"train_views={len(prepared)} model={parameters / 1e6:.2f}M "
        f"text_dim={model.text_dim} amp={use_amp}",
        flush=True,
    )
    losses = []
    latest = None
    interval_started = time.time()
    started = interval_started
    for step in range(start_step + 1, args.steps + 1):
        prepared_view = prepared[(step - 1) % len(prepared)]
        prompt_offset = ((step - 1) // len(prepared)) % len(
            prepared_view.prompt_indices
        )
        prompt_index = prepared_view.prompt_indices[prompt_offset]
        text = corpus.prompt_embeddings[prompt_index].to(device)
        drop_text = torch.tensor(
            [torch.rand((), generator=generator).item() < args.text_dropout],
            dtype=torch.bool,
            device=device,
        )
        drop_context = (
            torch.rand((), generator=generator).item() < args.context_dropout
        )
        context = (
            torch.zeros_like(prepared_view.context_raster)
            if drop_context
            else prepared_view.context_raster
        )
        if step <= args.warmup:
            learning_rate = args.lr * step / max(args.warmup, 1)
        else:
            progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
            learning_rate = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model.forward_training(
                context, prepared_view.target, text, drop_text
            )
            loss, terms = model.loss(
                output,
                prepared_view.target.values,
                prepared_view.target.counts,
                count_weight=args.count_weight,
                balance_count=args.balance_count_loss,
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            count = min(args.log_every, len(losses))
            record = {
                "step": step,
                "loss": sum(losses[-count:]) / count,
                "feature": float(terms["feature"]),
                "count": float(terms["count"]),
                "gradient_norm": float(gradient_norm),
                "lr": learning_rate,
                "seconds_per_step": (now - interval_started) / count,
            }
            interval_started = now
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            latest = evaluate_prompted_streaming(model, corpus, device=device).to_dict()
            record = {"step": step, "evaluation": latest}
            _append_json(log_path, record)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, latest)
    summary = {
        "steps": args.steps,
        "seconds": time.time() - started,
        "train_views": len(prepared),
        "latest_evaluation": latest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
