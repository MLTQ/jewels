"""Export full MP4s from the frozen prompt-to-Jewel generation proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol.prompt_video_runtime import PromptVideoPaths, PromptVideoRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--mode", choices=("exact", "learned"), default="exact")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runtime = PromptVideoRuntime(
        PromptVideoPaths.from_project_root(args.project_root), device=args.device
    )
    prompts = args.prompt or list(runtime.prompts)
    exports = []
    for prompt in prompts:
        video, metadata, row = runtime.generate_video(
            prompt, args.seed, mode=args.mode, output_dir=args.output_dir
        )
        exports.append({
            "video": str(video),
            "metadata": str(metadata),
            "prompt": row["prompt"],
            "mode": row["mode"],
            "seed": row["seed"],
            "scene": row["program_scene_label"],
        })
        print(f"exported {video}", flush=True)
    print(json.dumps({"schema": "prompt-video-exports-v1", "exports": exports}, indent=2))


if __name__ == "__main__":
    main()
