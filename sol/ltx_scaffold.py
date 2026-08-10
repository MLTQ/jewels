"""Run a reproducible LTX-2.3 semantic-scaffold generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ScaffoldConfig:
    """Inputs that define one LTX scaffold sample and its runtime contract."""

    ltx_root: Path
    distilled_checkpoint: Path
    spatial_upsampler: Path
    gemma_root: Path
    prompt: str
    output: Path
    cuda_visible_device: str
    seed: int = 42
    height: int = 512
    width: int = 768
    num_frames: int = 49
    frame_rate: float = 24.0
    quantization: str = "fp8-cast"
    offload: str = "cpu"
    max_batch_size: int = 1


def validate_config(config: ScaffoldConfig, *, require_assets: bool = True) -> None:
    """Reject invalid video geometry and missing pinned runtime assets."""
    if not config.prompt.strip():
        raise ValueError("prompt must not be empty")
    if config.height <= 0 or config.height % 64:
        raise ValueError("height must be a positive multiple of 64")
    if config.width <= 0 or config.width % 64:
        raise ValueError("width must be a positive multiple of 64")
    if config.num_frames <= 0 or (config.num_frames - 1) % 8:
        raise ValueError("num_frames must equal 8*K + 1")
    if config.frame_rate <= 0 or config.max_batch_size <= 0:
        raise ValueError("frame rate and max batch size must be positive")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
    if config.quantization not in {"fp8-cast", "fp8-scaled-mm"}:
        raise ValueError("unsupported quantization policy")
    if config.offload not in {"none", "cpu", "disk"}:
        raise ValueError("unsupported offload policy")
    if not config.cuda_visible_device:
        raise ValueError("cuda_visible_device must identify one GPU")
    if config.output.suffix.lower() != ".mp4":
        raise ValueError("output must use the .mp4 suffix")
    if not require_assets:
        return
    required = (
        config.ltx_root / ".venv/bin/python",
        config.distilled_checkpoint,
        config.spatial_upsampler,
        config.gemma_root / "model.safetensors.index.json",
        config.gemma_root / "tokenizer.model",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing LTX runtime assets: {missing}")


def build_command(config: ScaffoldConfig) -> list[str]:
    """Build the official distilled-pipeline command without shell quoting."""
    return [
        str(config.ltx_root / ".venv/bin/python"),
        "-m",
        "ltx_pipelines.distilled",
        "--distilled-checkpoint-path",
        str(config.distilled_checkpoint),
        "--spatial-upsampler-path",
        str(config.spatial_upsampler),
        "--gemma-root",
        str(config.gemma_root),
        "--prompt",
        config.prompt,
        "--output-path",
        str(config.output),
        "--seed",
        str(config.seed),
        "--height",
        str(config.height),
        "--width",
        str(config.width),
        "--num-frames",
        str(config.num_frames),
        "--frame-rate",
        str(config.frame_rate),
        "--quantization",
        config.quantization,
        "--offload",
        config.offload,
        "--max-batch-size",
        str(config.max_batch_size),
    ]


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _probe_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _parse_gpu_sample(output: str) -> tuple[int, int]:
    """Parse one no-header nvidia-smi memory/utilization row."""
    fields = [field.strip() for field in output.strip().split(",")]
    if len(fields) != 2:
        raise ValueError(f"unexpected nvidia-smi output: {output!r}")
    return int(fields[0]), int(fields[1])


def _read_gpu_sample(device: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_gpu_sample(result.stdout)


def run_scaffold(config: ScaffoldConfig, receipt_path: Path) -> dict[str, object]:
    """Launch LTX and persist success/failure provenance as JSON."""
    validate_config(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(config)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_device
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = datetime.now(timezone.utc)
    start_time = time.monotonic()
    gpu_monitor: dict[str, int | str | None] = {
        "device": config.cuda_visible_device,
        "baseline_memory_mib": None,
        "peak_memory_mib": None,
        "peak_above_baseline_mib": None,
        "maximum_utilization_percent": None,
        "samples": 0,
        "sample_errors": 0,
    }

    def sample_gpu() -> None:
        try:
            memory_mib, utilization = _read_gpu_sample(config.cuda_visible_device)
        except (OSError, subprocess.SubprocessError, ValueError):
            gpu_monitor["sample_errors"] = int(gpu_monitor["sample_errors"] or 0) + 1
            return
        if gpu_monitor["baseline_memory_mib"] is None:
            gpu_monitor["baseline_memory_mib"] = memory_mib
        current_peak = gpu_monitor["peak_memory_mib"]
        gpu_monitor["peak_memory_mib"] = max(
            memory_mib, int(current_peak) if current_peak is not None else memory_mib
        )
        current_utilization = gpu_monitor["maximum_utilization_percent"]
        gpu_monitor["maximum_utilization_percent"] = max(
            utilization,
            int(current_utilization) if current_utilization is not None else utilization,
        )
        gpu_monitor["samples"] = int(gpu_monitor["samples"] or 0) + 1

    sample_gpu()
    process = subprocess.Popen(
        command,
        cwd=config.ltx_root,
        env=environment,
    )
    while process.poll() is None:
        sample_gpu()
        time.sleep(1.0)
    returncode = process.wait()
    baseline_memory = gpu_monitor["baseline_memory_mib"]
    peak_memory = gpu_monitor["peak_memory_mib"]
    if baseline_memory is not None and peak_memory is not None:
        gpu_monitor["peak_above_baseline_mib"] = int(peak_memory) - int(
            baseline_memory
        )
    receipt: dict[str, object] = {
        "status": "complete" if returncode == 0 else "failed",
        "returncode": returncode,
        "started_at": started.isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "ltx_revision": _git_revision(config.ltx_root),
        "config": {
            **asdict(config),
            "ltx_root": str(config.ltx_root),
            "distilled_checkpoint": str(config.distilled_checkpoint),
            "spatial_upsampler": str(config.spatial_upsampler),
            "gemma_root": str(config.gemma_root),
            "output": str(config.output),
        },
        "command": command,
        "gpu_monitor": gpu_monitor,
    }
    if returncode == 0:
        if not config.output.exists():
            raise FileNotFoundError("LTX returned success without creating its output")
        receipt["video_probe"] = _probe_video(config.output)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltx-root", default="/home/m/LTX-2")
    parser.add_argument("--distilled-checkpoint")
    parser.add_argument("--spatial-upsampler")
    parser.add_argument("--gemma-root")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--cuda-visible-device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument(
        "--quantization", choices=("fp8-cast", "fp8-scaled-mm"), default="fp8-cast"
    )
    parser.add_argument("--offload", choices=("none", "cpu", "disk"), default="cpu")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.ltx_root).resolve()
    output = Path(args.output).resolve()
    config = ScaffoldConfig(
        ltx_root=root,
        distilled_checkpoint=Path(
            args.distilled_checkpoint
            or root / "models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
        ).resolve(),
        spatial_upsampler=Path(
            args.spatial_upsampler
            or root / "models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ).resolve(),
        gemma_root=Path(args.gemma_root or root / "models/gemma-3-12b").resolve(),
        prompt=args.prompt,
        output=output,
        cuda_visible_device=args.cuda_visible_device,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        quantization=args.quantization,
        offload=args.offload,
        max_batch_size=args.max_batch_size,
    )
    validate_config(config, require_assets=not args.dry_run)
    receipt = Path(args.receipt).resolve() if args.receipt else output.with_suffix(".json")
    if args.dry_run:
        print(json.dumps({"command": build_command(config)}, indent=2))
        return
    run_scaffold(config, receipt)
    print(receipt)


if __name__ == "__main__":
    main()
