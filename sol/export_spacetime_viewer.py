"""Export a fitted PrimitiveField to the browser spacetime-viewer schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


STPRIM_ROOT = Path(__file__).resolve().parent.parent / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import quat_to_rotmat  # noqa: E402


SCHEMA = "spacetime-jewel-viewer-v1"
REQUIRED_STATE = ("mu", "log_scale", "quat", "color", "color_grad", "logit_w")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded(values: torch.Tensor, decimals: int = 6) -> list[float]:
    scale = 10**decimals
    return (
        torch.round(values.detach().float().cpu().reshape(-1) * scale)
        .div(scale)
        .tolist()
    )


def build_viewer_payload(
    checkpoint: dict[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Convert one fitted checkpoint into compact browser-ready arrays."""
    state = checkpoint.get("state")
    if not isinstance(state, dict) or any(key not in state for key in REQUIRED_STATE):
        raise ValueError(f"checkpoint state must contain {REQUIRED_STATE}")
    count = int(state["mu"].shape[0])
    if count <= 0 or state["mu"].shape != (count, 3):
        raise ValueError("checkpoint must contain at least one three-coordinate Jewel")
    if any(int(state[key].shape[0]) != count for key in REQUIRED_STATE):
        raise ValueError("all Jewel parameter arrays must have the same first dimension")

    scales = state["log_scale"].float().clamp(-8.0, 2.0).exp()
    quaternion_wxyz = state["quat"].float()
    quaternion_wxyz = quaternion_wxyz / quaternion_wxyz.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    rotations = quat_to_rotmat(quaternion_wxyz)
    covariance = rotations @ torch.diag_embed(scales.square()) @ rotations.transpose(1, 2)
    time_variance = covariance[:, 2, 2].clamp_min(1e-10)
    cross = covariance[:, :2, 2]
    velocity = cross / time_variance[:, None]
    slice_covariance = covariance[:, :2, :2] - (
        cross[:, :, None] * cross[:, None, :] / time_variance[:, None, None]
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(slice_covariance.float())
    slice_roots = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(1e-10).sqrt())
    weights = torch.sigmoid(state["logit_w"].float())
    color_energy = state["color"].float().square().mean(dim=1).sqrt()
    importance = weights * scales.prod(dim=1) * color_energy.clamp_min(1e-4)

    # Three.js stores quaternions as x,y,z,w while the research checkpoint is w,x,y,z.
    quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]
    info = checkpoint.get("info", {})
    config = checkpoint.get("cfg", {})
    shape = [int(value) for value in info.get("shape", (1, 1, 1))]
    background = info.get("background", (0.0, 0.0, 0.0))
    if len(shape) != 3 or len(background) != 3:
        raise ValueError("checkpoint info must declare shape (T,H,W) and RGB background")

    return {
        "schema": SCHEMA,
        "source": {
            "checkpoint": source_name,
            "sha256": source_sha256,
            "fit_mode": config.get("mode", "unknown"),
            "training_steps": int(config.get("steps", 0)),
            "fit_seconds": float(info.get("seconds", 0.0)),
        },
        "field": {
            "count": count,
            "shape": shape,
            "t_scale": float(config.get("t_scale", 1.0)),
            "background": [float(value) for value in background],
        },
        "arrays": {
            "centers": _rounded(state["mu"]),
            "scales": _rounded(scales),
            "quaternions": _rounded(quaternion_xyzw),
            "colors": _rounded(state["color"]),
            "color_gradients": _rounded(state["color_grad"]),
            "weights": _rounded(weights),
            "time_sigmas": _rounded(time_variance.sqrt()),
            "slice_velocities": _rounded(velocity),
            "slice_roots": _rounded(slice_roots),
            "importance": _rounded(importance, decimals=9),
        },
        "display_contract": {
            "volume": "all centroids plus importance-ranked two-sigma covariance shells",
            "slice": "all Jewels, conditional Gaussian slice, P1 color, positive additive browser preview",
            "evaluation": "use the PyTorch support-complete renderer for quantitative results",
        },
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "sol/results/jewel_explainer_series_v1/assets/singer_field_additive_seed0.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "sol/spacetime_viewer/public/data/singer-field.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload = build_viewer_payload(
        checkpoint,
        source_name=args.checkpoint.name,
        source_sha256=_sha256(args.checkpoint),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "jewels": payload["field"]["count"],
                "frames": payload["field"]["shape"][0],
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
