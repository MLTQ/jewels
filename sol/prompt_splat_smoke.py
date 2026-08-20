"""Test whether prompt identity survives video scaffold to jewel-field rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--style", default="photoreal")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--support-capacity", type=int, default=1024)
    args = parser.parse_args()
    device, output = torch.device(args.device), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    examples = sorted(
        (item for item in manifest["examples"]
         if item["split"] == "validation" and item["style"] == args.style),
        key=lambda item: item["class_id"],
    )
    if len(examples) < 3 or any(not item["source_prompt"] for item in examples):
        raise ValueError("prompt smoke needs at least three prompted validation examples")

    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = saved["meta"]
    encoder = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024), **meta["model_args"]
    ).to(device).eval()
    encoder.load_state_dict(saved["model"])

    import open_clip  # noqa: PLC0415
    clip = open_clip.create_model("ViT-B-32", pretrained="laion2b_s34b_b79k").to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = [item["source_prompt"] for item in examples]
    with torch.no_grad():
        text = clip.encode_text(tokenizer(prompts).to(device)).float()
        text = torch.nn.functional.normalize(text, dim=1)
        null = clip.encode_text(tokenizer([""]).to(device)).float()
        null = torch.nn.functional.normalize(null, dim=1)[0]
    records = []
    target_embeddings, render_embeddings = [], []
    for item in examples:
        video = load_video(item["video"], max_frames=item["frames"],
                           resize=(args.height, args.width), device="cpu")
        indices = torch.linspace(0, len(video) - 1, args.frames).long()
        with torch.no_grad():
            prediction = encoder(video.to(device))
            points = frame_points(len(video), indices, args.height, args.width,
                                  device=device)
            rendered = cholesky_render(
                prediction["centers"], prediction["cholesky"],
                prediction["colors"], prediction["color_grads"],
                prediction["logit_w"], points, prediction["background"],
                point_chunk=4096, cull_mode="support_tiled",
                support_capacity=args.support_capacity,
            ).reshape(args.frames, args.height, args.width, 3).clamp(0, 1)
            target = video[indices].to(device)

            def image_embedding(frames: torch.Tensor) -> torch.Tensor:
                resized = torch.nn.functional.interpolate(
                    frames.permute(0, 3, 1, 2), size=(224, 224),
                    mode="bicubic", align_corners=False,
                )
                mean = resized.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
                std = resized.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
                encoded = clip.encode_image((resized - mean) / std).float()
                return torch.nn.functional.normalize(encoded.mean(0), dim=0)

            target_embeddings.append(image_embedding(target).cpu())
            render_embeddings.append(image_embedding(rendered).cpu())
        print("embedded", item["class_name"], flush=True)

    target_embeddings = torch.stack(target_embeddings)
    render_embeddings = torch.stack(render_embeddings)
    text_cpu, null_cpu = text.cpu(), null.cpu()
    for index, item in enumerate(examples):
        wrong = (index + 1) % len(examples)
        records.append({
            "source_id": item["source_id"], "class_name": item["class_name"],
            "prompt": prompts[index], "shuffled_prompt": prompts[wrong],
            "target_correct": float(target_embeddings[index] @ text_cpu[index]),
            "render_correct": float(render_embeddings[index] @ text_cpu[index]),
            "render_shuffled": float(render_embeddings[index] @ text_cpu[wrong]),
            "render_null": float(render_embeddings[index] @ null_cpu),
        })
    summary = {
        "target_correct_mean": statistics.mean(row["target_correct"] for row in records),
        "render_correct_mean": statistics.mean(row["render_correct"] for row in records),
        "render_shuffled_mean": statistics.mean(row["render_shuffled"] for row in records),
        "render_null_mean": statistics.mean(row["render_null"] for row in records),
        "correct_minus_shuffled": statistics.mean(
            row["render_correct"] - row["render_shuffled"] for row in records),
        "correct_beats_shuffled_fraction": statistics.mean(
            row["render_correct"] > row["render_shuffled"] for row in records),
        "semantic_retention_ratio": statistics.mean(row["render_correct"] for row in records)
        / max(statistics.mean(row["target_correct"] for row in records), 1e-8),
    }
    report = {"schema": "prompt-splat-smoke-v1",
              "scope": "prompt identity through pretrained video scaffold -> encoder -> support renderer; not direct text-latent generation",
              "checkpoint": args.checkpoint, "style": args.style,
              "examples": len(examples), "frames_per_example": args.frames,
              "summary": summary, "records": records}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt  # noqa: PLC0415
    labels = ["correct", "shuffled", "null"]
    values = [summary["render_correct_mean"], summary["render_shuffled_mean"],
              summary["render_null_mean"]]
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, values)
    axis.bar_label(bars, fmt="%.3f")
    axis.set_ylabel("CLIP cosine similarity")
    axis.set_title("Prompt identity after video → splat bottleneck")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(output / "prompt_controls.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
