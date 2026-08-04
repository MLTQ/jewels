"""Durable checkpoint helpers for long-running primitive fits.

The fitter owns the meaning of its recovery state. This module owns the
filesystem contract: recovery payloads are plain tensor trees and are replaced
atomically so a process dying during ``torch.save`` cannot destroy the last
good checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


RECOVERY_SCHEMA = "stprim-fit-recovery-v1"


def tensors_to_cpu(value: Any) -> Any:
    """Clone every tensor in a nested state tree onto CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)
    return value


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a torch payload via a sibling temporary file, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
