"""Batch-fit a directory of videos into a checkpoint corpus for stage 2.

    python cli/fit_corpus.py --videos 'corpora/avenue/training_videos/*.avi' \
        --out corpus/avenue --frames 64 --size 160

Each source video is cut into non-overlapping `--frames`-frame windows and
each window is fit independently -> one checkpoint per window. Resumable:
windows whose checkpoint already exists are skipped, so a killed run continues
where it left off. Progress and per-window metrics append to corpus_log.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.video_io import count_frames, load_video  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, required=True,
                    help="glob of source videos (quote it)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--frames", type=int, default=64, help="window length")
    ap.add_argument("--size", type=int, default=160,
                    help="short side; aspect preserved")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--num-init", type=int, default=3000)
    ap.add_argument("--max-primitives", type=int, default=10000)
    ap.add_argument("--voxels", type=int, default=65536)
    ap.add_argument("--t-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N windows (0 = all); for smoke runs")
    ap.add_argument("--windows-per-video", type=int, default=0,
                    help="cap windows taken from each source (0 = all). "
                         "One window per clip maximizes corpus diversity when "
                         "clips outnumber the fitting budget")
    args = ap.parse_args()

    _VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    videos = sorted(
        v for v in glob.glob(args.videos)
        if Path(v).is_dir() or v.lower().endswith(_VIDEO_EXTS)
    )
    if not videos:
        sys.exit(f"no videos match {args.videos!r}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "corpus_log.jsonl"

    # Enumerate (video, start_frame) windows up front so progress is knowable.
    windows: list[tuple[str, int]] = []
    for v in videos:
        n = count_frames(v)
        w = [(v, s) for s in range(0, n - args.frames + 1, args.frames)]
        if args.windows_per_video:
            w = w[: args.windows_per_video]
        windows += w
    print(f"{len(videos)} videos -> {len(windows)} windows of {args.frames} frames",
          flush=True)

    done = skipped = 0
    t0 = time.time()
    for i, (v, start) in enumerate(windows):
        if args.limit and done >= args.limit:
            break
        ckpt = outdir / f"{Path(v).stem}_w{start:06d}.pt"
        if ckpt.exists():
            skipped += 1
            continue

        vid = load_video(v, max_frames=args.frames, start_frame=start,
                         resize=args.size)
        if vid.shape[0] < args.frames:  # metadata overcounted; partial tail
            skipped += 1
            continue

        cfg = FitConfig(
            num_init=args.num_init, max_primitives=args.max_primitives,
            steps=args.steps, voxels_per_step=args.voxels,
            t_scale=args.t_scale, seed=args.seed,
        )
        field, info = fit_volume(vid, cfg, device=args.device, verbose=False)
        torch.save(
            {"state": field.state_dict(), "cfg": vars(cfg), "info": info,
             "source": {"video": v, "start_frame": start}},
            ckpt,
        )
        done += 1

        rec = {
            "ckpt": ckpt.name,
            "psnr": info["history"][-1]["psnr"],
            "n": info["n_final"],
            "seconds": round(info["seconds"], 1),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        rate = (time.time() - t0) / max(done, 1)
        remaining = len(windows) - i - 1
        print(f"[{i + 1}/{len(windows)}] {rec['ckpt']}  "
              f"psnr~{rec['psnr']:.2f}  N={rec['n']}  {rec['seconds']}s  "
              f"(~{rate * remaining / 3600:.1f}h left)", flush=True)

    print(f"done: {done} fit, {skipped} skipped, "
          f"{(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
