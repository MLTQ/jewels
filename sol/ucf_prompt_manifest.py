"""Build a balanced, leakage-safe UCF manifest for prompted jewel streaming."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess


SCHEMA = "ucf-prompt-streaming-v1"
_UCF_NAME = re.compile(r"^v_(?P<class_name>.+)_g(?P<group>\d+)_c(?P<clip>\d+)$")


@dataclass(frozen=True)
class PromptClassSpec:
    class_name: str
    label: str
    train_prompts: tuple[str, ...]
    evaluation_prompts: tuple[str, ...]


@dataclass(frozen=True)
class VideoCandidate:
    path: Path
    class_name: str
    group: int
    clip: int
    frame_count: int


PROMPT_SPECS = {
    spec.class_name: spec
    for spec in (
        PromptClassSpec(
            "Basketball",
            "playing basketball",
            (
                "a person playing basketball",
                "an indoor basketball game",
                "someone dribbling and shooting a basketball",
            ),
            ("a video of people playing basketball on a court",),
        ),
        PromptClassSpec(
            "HorseRiding",
            "riding a horse",
            (
                "a person riding a horse",
                "horse riding outdoors",
                "someone traveling on horseback",
            ),
            ("a video of a rider moving on a horse",),
        ),
        PromptClassSpec(
            "PlayingGuitar",
            "playing guitar",
            (
                "a person playing guitar",
                "a musician performing with a guitar",
                "someone strumming a guitar",
            ),
            ("a video of a guitarist playing music",),
        ),
        PromptClassSpec(
            "ApplyEyeMakeup",
            "applying eye makeup",
            (
                "a person applying eye makeup",
                "someone putting makeup around their eyes",
                "a close-up of applying eye cosmetics",
            ),
            ("a video of a person doing their eye makeup",),
        ),
    )
}


def parse_ucf_video(path: str | Path, frame_count: int) -> VideoCandidate:
    """Parse class, source group, and clip IDs from a canonical UCF filename."""
    path = Path(path)
    match = _UCF_NAME.match(path.stem)
    if match is None:
        raise ValueError(f"not a canonical UCF filename: {path.name}")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    class_name = match.group("class_name")
    if path.parent.name and path.parent.name != class_name:
        raise ValueError(
            f"directory class {path.parent.name!r} disagrees with {class_name!r}"
        )
    return VideoCandidate(
        path=path,
        class_name=class_name,
        group=int(match.group("group")),
        clip=int(match.group("clip")),
        frame_count=frame_count,
    )


def select_balanced_candidates(
    candidates: list[VideoCandidate],
    class_names: list[str] | tuple[str, ...],
    *,
    groups: tuple[int, ...] = (1, 2, 3, 4),
    frames: int = 96,
) -> list[VideoCandidate]:
    """Select the longest eligible clip for every requested class/source group."""
    if frames <= 0 or not groups or len(set(groups)) != len(groups):
        raise ValueError("frames and unique source groups are required")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class names must be non-empty and unique")
    selected = []
    for class_name in class_names:
        if class_name not in PROMPT_SPECS:
            raise ValueError(f"missing prompt specification for {class_name!r}")
        for group in groups:
            eligible = [
                candidate
                for candidate in candidates
                if candidate.class_name == class_name
                and candidate.group == group
                and candidate.frame_count >= frames
            ]
            if not eligible:
                raise ValueError(
                    f"no {frames}-frame clip for {class_name} source group {group:02d}"
                )
            selected.append(
                min(eligible, key=lambda item: (-item.frame_count, str(item.path)))
            )
    return selected


def _probe_frames(path: Path) -> int:
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def scan_candidates(root: str | Path, class_names: list[str]) -> list[VideoCandidate]:
    """Probe every AVI in requested UCF class directories."""
    root = Path(root)
    candidates = []
    for class_name in class_names:
        directory = root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"missing UCF class directory: {directory}")
        for path in sorted(directory.glob("*.avi")):
            candidates.append(parse_ucf_video(path, _probe_frames(path)))
    return candidates


def stage_candidates(
    selected: list[VideoCandidate], stage_dir: str | Path
) -> dict[Path, Path]:
    """Create an idempotent symlink set accepted by the existing corpus fitter."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    for candidate in selected:
        destination = stage_dir / candidate.path.name
        target = candidate.path.resolve()
        if destination.is_symlink():
            if destination.resolve() != target:
                raise ValueError(f"staging symlink targets the wrong video: {destination}")
        elif destination.exists():
            raise FileExistsError(f"refusing to replace staged path: {destination}")
        else:
            destination.symlink_to(target)
        staged[candidate.path] = destination
    return staged


def build_manifest(
    selected: list[VideoCandidate],
    class_names: list[str] | tuple[str, ...],
    *,
    frames: int = 96,
    validation_group: int = 4,
    staged_paths: dict[Path, Path] | None = None,
) -> dict:
    """Create the serialized prompt, split, fitting, and encoder contract."""
    class_ids = {name: index for index, name in enumerate(class_names)}
    examples = []
    for candidate in selected:
        spec = PROMPT_SPECS[candidate.class_name]
        fit_video = (
            staged_paths[candidate.path] if staged_paths is not None else candidate.path
        )
        examples.append(
            {
                "class_id": class_ids[candidate.class_name],
                "class_name": candidate.class_name,
                "label": spec.label,
                "source_group": candidate.group,
                "source_id": f"{candidate.class_name}_g{candidate.group:02d}",
                "clip_id": candidate.clip,
                "split": "validation" if candidate.group == validation_group else "train",
                "video": str(candidate.path),
                "fit_video": str(fit_video),
                "frame_count": candidate.frame_count,
                "start_frame": 0,
                "frames": frames,
                "train_prompts": list(spec.train_prompts),
                "evaluation_prompts": list(spec.evaluation_prompts),
            }
        )
    train_classes = {item["class_name"] for item in examples if item["split"] == "train"}
    validation_classes = {
        item["class_name"] for item in examples if item["split"] == "validation"
    }
    if train_classes != set(class_names) or validation_classes != set(class_names):
        raise ValueError("every class must occur in both train and validation splits")
    return {
        "schema": SCHEMA,
        "frames": frames,
        "validation_group": validation_group,
        "classes": [asdict(PROMPT_SPECS[name]) for name in class_names],
        "text_encoder": {
            "library": "open_clip",
            "model": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "pooling": "normalized_text_embedding",
            "condition_dropout": 0.15,
        },
        "fit_contract": {
            "size": 160,
            "num_init": 16000,
            "max_primitives": 120000,
            "steps": 9000,
            "voxels": 65536,
            "split_mode": "spatial",
            "recovery_every": 100,
            "windows_per_video": 1,
        },
        "examples": examples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage-dir")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--validation-group", type=int, default=4)
    parser.add_argument(
        "--classes", nargs="+", default=list(PROMPT_SPECS)
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = scan_candidates(args.root, args.classes)
    selected = select_balanced_candidates(
        candidates, args.classes, frames=args.frames
    )
    staged = stage_candidates(selected, args.stage_dir) if args.stage_dir else None
    manifest = build_manifest(
        selected,
        args.classes,
        frames=args.frames,
        validation_group=args.validation_group,
        staged_paths=staged,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {
        split: sum(item["split"] == split for item in manifest["examples"])
        for split in ("train", "validation")
    }
    print(
        json.dumps(
            {
                "manifest": str(output),
                "classes": args.classes,
                "examples": len(manifest["examples"]),
                "splits": counts,
                "minimum_frames": min(
                    item["frame_count"] for item in manifest["examples"]
                ),
                "stage_dir": args.stage_dir,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
