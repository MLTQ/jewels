"""Generate a resumable prompt-balanced LTX semantic-scaffold corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sol.ltx_scaffold import ScaffoldConfig, run_scaffold


SCHEMA = "jewels-ltx-corpus-v1"
SOURCE_SCHEMA = "ucf-prompt-streaming-v1"
DEFAULT_PROMPT_SUFFIX = (
    "Realistic video, one continuous shot, stable camera, natural motion, no cuts."
)


@dataclass(frozen=True)
class CorpusExample:
    """One source prompt and deterministic LTX sample identity."""

    class_id: int
    class_name: str
    prompt_role: str
    prompt_index: int
    source_prompt: str
    generation_prompt: str
    seed: int
    stem: str


@dataclass(frozen=True)
class CorpusRuntime:
    """Shared LTX paths, geometry, and memory policy for a corpus run."""

    ltx_root: Path
    output_dir: Path
    cuda_visible_device: str
    height: int = 512
    width: int = 768
    num_frames: int = 49
    frame_rate: float = 24.0
    quantization: str = "fp8-cast"
    offload: str = "cpu"
    max_batch_size: int = 1


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot make a filename from {value!r}")
    return slug


def _generation_prompt(source_prompt: str, suffix: str) -> str:
    source = source_prompt.strip().rstrip(".")
    suffix = suffix.strip()
    if not source:
        raise ValueError("source prompts must not be empty")
    return f"{source}. {suffix}" if suffix else f"{source}."


def build_plan(
    source_manifest: dict[str, object],
    *,
    seed_base: int = 42_000,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
) -> tuple[CorpusExample, ...]:
    """Flatten class prompt roles into a stable, balanced generation plan."""
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unsupported source prompt manifest schema")
    classes = source_manifest.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("source prompt manifest has no classes")
    if seed_base < 0:
        raise ValueError("seed_base must be non-negative")

    plan = []
    for class_id, raw_class in enumerate(classes):
        if not isinstance(raw_class, dict):
            raise ValueError("class specifications must be objects")
        class_name = raw_class.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("class_name must be a non-empty string")
        role_prompts = (
            ("train", raw_class.get("train_prompts")),
            ("evaluation", raw_class.get("evaluation_prompts")),
        )
        ordinal = 0
        for role, raw_prompts in role_prompts:
            if not isinstance(raw_prompts, list) or not raw_prompts:
                raise ValueError(f"{class_name} has no {role} prompts")
            for prompt_index, source_prompt in enumerate(raw_prompts):
                if not isinstance(source_prompt, str):
                    raise ValueError("prompts must be strings")
                seed = seed_base + class_id * 100 + ordinal
                stem = (
                    f"{class_id:02d}_{_slug(class_name)}_{role}_{prompt_index:02d}"
                    f"_seed{seed}"
                )
                plan.append(
                    CorpusExample(
                        class_id=class_id,
                        class_name=class_name,
                        prompt_role=role,
                        prompt_index=prompt_index,
                        source_prompt=source_prompt,
                        generation_prompt=_generation_prompt(
                            source_prompt, prompt_suffix
                        ),
                        seed=seed,
                        stem=stem,
                    )
                )
                ordinal += 1
    stems = [example.stem for example in plan]
    if len(stems) != len(set(stems)):
        raise ValueError("generation plan contains duplicate sample names")
    return tuple(plan)


def select_plan(
    plan: tuple[CorpusExample, ...], *, prompt_role: str = "all"
) -> tuple[CorpusExample, ...]:
    """Select one prompt split without changing corpus order or identities."""
    if prompt_role == "all":
        return plan
    if prompt_role not in {"train", "evaluation"}:
        raise ValueError(f"unsupported prompt role: {prompt_role}")
    selected = tuple(example for example in plan if example.prompt_role == prompt_role)
    if not selected:
        raise ValueError(f"generation plan has no {prompt_role} examples")
    return selected


def scaffold_config(example: CorpusExample, runtime: CorpusRuntime) -> ScaffoldConfig:
    """Bind one corpus identity to the pinned LTX scaffold runtime."""
    root = runtime.ltx_root
    return ScaffoldConfig(
        ltx_root=root,
        distilled_checkpoint=(
            root / "models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
        ),
        spatial_upsampler=(
            root / "models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ),
        gemma_root=root / "models/gemma-3-12b",
        prompt=example.generation_prompt,
        output=runtime.output_dir / f"{example.stem}.mp4",
        cuda_visible_device=runtime.cuda_visible_device,
        seed=example.seed,
        height=runtime.height,
        width=runtime.width,
        num_frames=runtime.num_frames,
        frame_rate=runtime.frame_rate,
        quantization=runtime.quantization,
        offload=runtime.offload,
        max_batch_size=runtime.max_batch_size,
    )


def _receipt_matches(config: ScaffoldConfig, receipt_path: Path) -> bool:
    if not config.output.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded = receipt.get("config")
    if receipt.get("status") != "complete" or not isinstance(recorded, dict):
        return False
    expected = {
        "prompt": config.prompt,
        "output": str(config.output),
        "seed": config.seed,
        "height": config.height,
        "width": config.width,
        "num_frames": config.num_frames,
        "frame_rate": config.frame_rate,
        "quantization": config.quantization,
        "offload": config.offload,
        "max_batch_size": config.max_batch_size,
    }
    return all(recorded.get(key) == value for key, value in expected.items())


def _receipt_summary(receipt_path: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text())
    gpu = receipt.get("gpu_monitor", {})
    probe = receipt.get("video_probe", {})
    return {
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "ltx_revision": receipt.get("ltx_revision"),
        "peak_memory_mib": gpu.get("peak_memory_mib"),
        "peak_above_baseline_mib": gpu.get("peak_above_baseline_mib"),
        "maximum_utilization_percent": gpu.get("maximum_utilization_percent"),
        "video_probe": probe,
    }


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus_manifest(
    source_manifest_path: Path,
    plan: tuple[CorpusExample, ...],
    runtime: CorpusRuntime,
) -> dict[str, object]:
    """Atomically snapshot completion state and receipt summaries."""
    entries = []
    complete = 0
    failed = 0
    for example in plan:
        config = scaffold_config(example, runtime)
        receipt_path = config.output.with_suffix(".json")
        status = "pending"
        summary = None
        if _receipt_matches(config, receipt_path):
            status = "complete"
            complete += 1
            summary = _receipt_summary(receipt_path)
        elif receipt_path.is_file():
            status = "failed"
            failed += 1
        entry = {
            **asdict(example),
            "output": str(config.output),
            "receipt": str(receipt_path),
            "status": status,
        }
        if summary is not None:
            entry["result"] = summary
        entries.append(entry)
    payload = {
        "schema": SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _source_digest(source_manifest_path),
        "runtime": {
            **asdict(runtime),
            "ltx_root": str(runtime.ltx_root),
            "output_dir": str(runtime.output_dir),
        },
        "summary": {
            "total": len(plan),
            "complete": complete,
            "failed": failed,
            "pending": len(plan) - complete - failed,
        },
        "examples": entries,
    }
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime.output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(manifest_path)
    return payload


def run_corpus(
    source_manifest_path: Path,
    plan: tuple[CorpusExample, ...],
    runtime: CorpusRuntime,
) -> dict[str, object]:
    """Run missing samples sequentially, checkpointing after every attempt."""
    write_corpus_manifest(source_manifest_path, plan, runtime)
    for index, example in enumerate(plan, start=1):
        config = scaffold_config(example, runtime)
        receipt_path = config.output.with_suffix(".json")
        if _receipt_matches(config, receipt_path):
            print(f"[{index}/{len(plan)}] skip complete {example.stem}", flush=True)
            continue
        print(f"[{index}/{len(plan)}] generate {example.stem}", flush=True)
        try:
            run_scaffold(config, receipt_path)
        finally:
            write_corpus_manifest(source_manifest_path, plan, runtime)
    return write_corpus_manifest(source_manifest_path, plan, runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-manifest", default="sol/results/prompt_smoke/manifest.json"
    )
    parser.add_argument(
        "--output-dir", default="/home/m/LTX-2/corpora/jewels_ucf_prompt_v1"
    )
    parser.add_argument("--ltx-root", default="/home/m/LTX-2")
    parser.add_argument("--cuda-visible-device", required=True)
    parser.add_argument("--seed-base", type=int, default=42_000)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument(
        "--quantization", choices=("fp8-cast", "fp8-scaled-mm"), default="fp8-cast"
    )
    parser.add_argument("--offload", choices=("none", "cpu", "disk"), default="cpu")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-role",
        choices=("all", "train", "evaluation"),
        default="all",
        help="generate all prompts or only one source-manifest split",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_manifest_path = Path(args.prompt_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text())
    plan = build_plan(
        source_manifest,
        seed_base=args.seed_base,
        prompt_suffix=args.prompt_suffix,
    )
    plan = select_plan(plan, prompt_role=args.prompt_role)
    if args.max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if args.max_examples:
        plan = plan[: args.max_examples]
    runtime = CorpusRuntime(
        ltx_root=Path(args.ltx_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        cuda_visible_device=args.cuda_visible_device,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        quantization=args.quantization,
        offload=args.offload,
        max_batch_size=args.max_batch_size,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "runtime": {
                        **asdict(runtime),
                        "ltx_root": str(runtime.ltx_root),
                        "output_dir": str(runtime.output_dir),
                    },
                    "examples": [asdict(example) for example in plan],
                },
                indent=2,
            )
        )
        return
    result = run_corpus(source_manifest_path, plan, runtime)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
