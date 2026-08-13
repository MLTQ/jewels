"""Load architecture-compatible model weights with explicit corpus provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sol.token_grid import GridSpec


def load_compatible_model_weights(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device,
    architecture: str,
    model_args: dict[str, Any],
    grid_spec: GridSpec,
    destination_manifest_sha256: str,
    allow_cross_manifest: bool,
) -> dict[str, Any]:
    """Load model-only state after validating architecture, rank, and ownership."""
    path = Path(checkpoint)
    saved = torch.load(path, map_location=map_location, weights_only=False)
    meta = saved.get("meta", {})
    if meta.get("architecture") != architecture:
        raise ValueError("initialization checkpoint architecture differs")
    if meta.get("model_args") != model_args:
        raise ValueError("initialization checkpoint model arguments differ")
    if tuple(meta.get("grid_shape", ())) != grid_spec.shape:
        raise ValueError("initialization checkpoint grid differs")
    if int(meta.get("slots_per_cell", 0)) != grid_spec.slots_per_cell:
        raise ValueError("initialization checkpoint rank capacity differs")
    source_manifest_sha256 = meta.get("manifest_sha256")
    if not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64:
        raise ValueError("initialization checkpoint lacks manifest provenance")
    if (
        not allow_cross_manifest
        and source_manifest_sha256 != destination_manifest_sha256
    ):
        raise ValueError("initialization checkpoint owns a different manifest")
    if "model" not in saved:
        raise ValueError("initialization checkpoint lacks model weights")
    model.load_state_dict(saved["model"])
    return {
        "path": str(path),
        "mode": (
            "cross_manifest_transfer"
            if source_manifest_sha256 != destination_manifest_sha256
            else "same_manifest_initialization"
        ),
        "source_architecture": architecture,
        "source_manifest_sha256": source_manifest_sha256,
        "destination_manifest_sha256": destination_manifest_sha256,
        "source_step": int(saved.get("step", 0)),
        "optimizer_restored": False,
    }
