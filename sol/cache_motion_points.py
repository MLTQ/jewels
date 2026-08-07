"""Cache balanced motion-and-chroma source coordinates for tokenizer training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch


def _window_motion_points(
    frames: list[torch.Tensor], pool_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain the historical motion-only point-pool contract."""
    points, scores, _ = _window_saliency_points(
        frames, pool_size, chroma_fraction=0.0
    )
    return points, scores


def _window_saliency_points(
    frames: list[torch.Tensor],
    pool_size: int,
    *,
    chroma_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build temporally balanced, disjoint motion and saturated-color samples."""
    if not 0 <= chroma_fraction <= 1:
        raise ValueError("chroma fraction must be in [0,1]")
    video = torch.stack(frames).float().div_(255)
    frames_count, height, width, _ = video.shape
    median = video.median(dim=0).values
    residual = (video - median).abs().mean(dim=-1)
    temporal = torch.zeros_like(residual)
    temporal[1:] = (video[1:] - video[:-1]).abs().mean(dim=-1)
    motion_score = residual + 0.5 * temporal
    chroma_score = video.amax(dim=-1) - video.amin(dim=-1)
    per_frame = max(1, pool_size // frames_count)
    points, scores, kinds = [], [], []
    for frame_index in range(frames_count):
        count = min(per_frame, height * width)
        chroma_count = round(count * chroma_fraction)
        motion_count = count - chroma_count
        selected_indices = []
        selected_scores = []
        selected_kinds = []
        motion_flat = motion_score[frame_index].flatten()
        chroma_flat = chroma_score[frame_index].flatten()
        if motion_count:
            values, indices = motion_flat.topk(motion_count)
            selected_indices.append(indices)
            selected_scores.append(values)
            selected_kinds.append(torch.zeros_like(indices))
        if chroma_count:
            available = chroma_flat.clone()
            if selected_indices:
                available[selected_indices[0]] = -1
            values, indices = available.topk(chroma_count)
            selected_indices.append(indices)
            selected_scores.append(values)
            selected_kinds.append(torch.ones_like(indices))
        indices = torch.cat(selected_indices)
        values = torch.cat(selected_scores)
        v = indices // width
        u = indices % width
        points.append(
            torch.stack(
                [
                    u.float() * (2 / max(width - 1, 1)) - 1,
                    v.float() * (2 / max(height - 1, 1)) - 1,
                    torch.full_like(
                        u.float(), frame_index * (2 / max(frames_count - 1, 1)) - 1
                    ),
                ],
                dim=-1,
            )
        )
        scores.append(values)
        kinds.append(torch.cat(selected_kinds))
    return torch.cat(points), torch.cat(scores), torch.cat(kinds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--extra-corpus", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--chroma-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pool_size <= 0:
        raise ValueError("pool size must be positive")
    if not 0 <= args.chroma_fraction <= 1:
        raise ValueError("chroma fraction must be in [0,1]")
    corpora = [Path(args.corpus), *(Path(path) for path in args.extra_corpus)]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict]] = defaultdict(list)
    checkpoint_paths = sorted(
        path
        for corpus in corpora
        for path in corpus.glob("*_w*.pt")
        if not path.name.endswith(".recovery.pt")
    )
    names = [path.name for path in checkpoint_paths]
    if len(names) != len(set(names)):
        raise ValueError("corpus roots contain duplicate checkpoint filenames")
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        source = checkpoint.get("source", {})
        video_path = source.get("video")
        if not video_path:
            raise ValueError(f"missing source video in {checkpoint_path}")
        by_video[str(video_path)].append(
            {
                "name": checkpoint_path.stem,
                "start": int(source.get("start_frame", 0)),
                "shape": tuple(checkpoint["info"]["shape"]),
            }
        )

    records = []
    for video_path, windows in sorted(by_video.items()):
        pending = {window["start"]: window for window in windows}
        import av

        container = av.open(video_path)
        active: dict | None = None
        frames: list[torch.Tensor] = []
        for frame_index, frame in enumerate(container.decode(video=0)):
            if active is None and frame_index in pending:
                active = pending.pop(frame_index)
                frames = []
            if active is not None:
                _, height, width = active["shape"]
                resized = frame.reformat(width=width, height=height)
                frames.append(
                    torch.from_numpy(resized.to_ndarray(format="rgb24").copy())
                )
                if len(frames) == active["shape"][0]:
                    points, scores, kinds = _window_saliency_points(
                        frames,
                        args.pool_size,
                        chroma_fraction=args.chroma_fraction,
                    )
                    path = output / f"{active['name']}.motion.pt"
                    torch.save(
                        {
                            "schema": "jewel-source-saliency-v1",
                            "points": points,
                            "scores": scores,
                            "kinds": kinds,
                            "chroma_fraction": args.chroma_fraction,
                            "source_video": video_path,
                            "start_frame": active["start"],
                            "shape": active["shape"],
                        },
                        path,
                    )
                    record = {
                        "name": active["name"],
                        "points": len(points),
                        "motion_points": int((kinds == 0).sum()),
                        "chroma_points": int((kinds == 1).sum()),
                        "artifact": path.name,
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    active = None
                    frames = []
            if not pending and active is None:
                break
        container.close()
        if pending or active is not None:
            missing = sorted(pending)
            if active is not None:
                missing.append(active["start"])
            raise RuntimeError(f"video ended before windows {missing}: {video_path}")
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
