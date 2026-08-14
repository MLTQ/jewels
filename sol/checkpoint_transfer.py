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


def load_augmented_model_weights(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device,
    architecture: str,
    model_args: dict[str, Any],
    added_model_args: dict[str, Any],
    added_state_prefixes: tuple[str, ...],
    grid_spec: GridSpec,
    destination_manifest_sha256: str,
) -> dict[str, Any]:
    """Load every shared weight while retaining new zero-residual modules."""
    if not added_model_args or not added_state_prefixes:
        raise ValueError("architecture augmentation must declare arguments and state prefixes")
    if any(model_args.get(name) != value for name, value in added_model_args.items()):
        raise ValueError("added model arguments do not match the destination model")
    source_model_args = {
        name: value for name, value in model_args.items() if name not in added_model_args
    }
    path = Path(checkpoint)
    saved = torch.load(path, map_location=map_location, weights_only=False)
    meta = saved.get("meta", {})
    if meta.get("architecture") != architecture:
        raise ValueError("augmentation checkpoint architecture differs")
    if meta.get("model_args") != source_model_args:
        raise ValueError("augmentation checkpoint base model arguments differ")
    if tuple(meta.get("grid_shape", ())) != grid_spec.shape:
        raise ValueError("augmentation checkpoint grid differs")
    if int(meta.get("slots_per_cell", 0)) != grid_spec.slots_per_cell:
        raise ValueError("augmentation checkpoint rank capacity differs")
    source_manifest_sha256 = meta.get("manifest_sha256")
    if source_manifest_sha256 != destination_manifest_sha256:
        raise ValueError("augmentation checkpoint owns a different manifest")
    if "model" not in saved:
        raise ValueError("augmentation checkpoint lacks model weights")
    incompatible = model.load_state_dict(saved["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("augmentation checkpoint contains unexpected model weights")
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(added_state_prefixes)
    ]
    if invalid_missing:
        raise ValueError("augmentation checkpoint is missing shared model weights")
    if not incompatible.missing_keys:
        raise ValueError("augmentation did not introduce any new model weights")
    return {
        "path": str(path),
        "mode": "same_manifest_architecture_augmentation",
        "source_architecture": architecture,
        "source_manifest_sha256": source_manifest_sha256,
        "destination_manifest_sha256": destination_manifest_sha256,
        "source_step": int(saved.get("step", 0)),
        "added_model_args": added_model_args,
        "added_state_prefixes": list(added_state_prefixes),
        "new_parameter_tensors": len(incompatible.missing_keys),
        "optimizer_restored": False,
    }
