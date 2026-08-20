"""Run a resumable manifest of temporal-tilt ablations and aggregate it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sol.aggregate_temporal_tilt import aggregate_reports
from sol.temporal_tilt_ablation import write_report


SCRIPT = Path(__file__).with_name("temporal_tilt_ablation.py")


def validate_manifest(manifest: dict) -> None:
    """Validate the bounded replication manifest before spending GPU compute."""
    if manifest.get("schema") != "temporal-tilt-replication-manifest-v1":
        raise ValueError("unsupported replication manifest schema")
    protocol = manifest.get("protocol", {})
    required = {
        "frames",
        "size",
        "steps",
        "constraints",
        "seeds",
        "num_init",
        "max_primitives",
        "voxels",
        "support_capacity",
        "support_point_chunk",
        "adapt_every",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"manifest protocol is missing: {missing}")
    sources = manifest.get("sources", [])
    if len(sources) < 1:
        raise ValueError("manifest must contain at least one source")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source ids must be unique")


def build_command(
    manifest: dict,
    source: dict,
    *,
    output_dir: Path,
    device: str,
) -> list[str]:
    """Build one dependency-free per-source fitter invocation."""
    protocol = manifest["protocol"]
    command = [
        sys.executable,
        str(SCRIPT),
        "--video",
        source["path"],
        "--frames",
        str(protocol["frames"]),
        "--size",
        str(protocol["size"]),
        "--start-frame",
        str(source.get("start_frame", 0)),
        "--steps",
        *(str(value) for value in protocol["steps"]),
        "--constraints",
        *(str(value) for value in protocol["constraints"]),
        "--seeds",
        *(str(value) for value in protocol["seeds"]),
        "--num-init",
        str(protocol["num_init"]),
        "--max-primitives",
        str(protocol["max_primitives"]),
        "--voxels",
        str(protocol["voxels"]),
        "--support-capacity",
        str(protocol["support_capacity"]),
        "--support-point-chunk",
        str(protocol["support_point_chunk"]),
        "--adapt-every",
        str(protocol["adapt_every"]),
        "--device",
        device,
        "--out",
        str(output_dir / source["id"]),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for source in manifest["sources"]:
        print(f"=== source {source['id']} ===", flush=True)
        subprocess.run(
            build_command(
                manifest,
                source,
                output_dir=output_dir,
                device=args.device,
            ),
            check=True,
        )
        reports.append(
            json.loads((output_dir / source["id"] / "report.json").read_text())
        )

    aggregate = aggregate_reports(reports)
    aggregate["manifest"] = {
        "path": str(manifest_path.resolve()),
        "contents": manifest,
    }
    write_report(output_dir / "report.json", aggregate)
    stats = aggregate["aggregate"]["paired_psnr_delta_db"]
    print(
        f"completed {aggregate['pair_count']} pairs: "
        f"{stats['mean']:.3f} dB "
        f"95% CI [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
