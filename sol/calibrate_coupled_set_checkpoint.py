"""Create an explicitly calibrated coupled-set checkpoint for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def calibrated_checkpoint(saved: dict, strength: float, source: str) -> dict:
    """Scale only each set block's zero-origin residual projection."""
    if not 0 <= strength <= 1:
        raise ValueError("coupled-set strength must lie inside [0,1]")
    meta = saved.get("meta", {})
    if meta.get("architecture") != "scaffold_birth_mark_flow_v1":
        raise ValueError("checkpoint is not a scaffold mark flow")
    if int(meta.get("model_args", {}).get("set_depth", 0)) <= 0:
        raise ValueError("checkpoint does not contain coupled set blocks")
    if meta.get("coupled_set_calibration") is not None:
        raise ValueError("checkpoint has already been calibrated")
    state = {name: value.clone() for name, value in saved["model"].items()}
    selected = [
        name
        for name in state
        if name.startswith("set_blocks.")
        and (name.endswith("row_update.3.weight") or name.endswith("row_update.3.bias"))
    ]
    expected = 2 * int(meta["model_args"]["set_depth"])
    if len(selected) != expected:
        raise ValueError("coupled-set residual projection state is incomplete")
    for name in selected:
        state[name].mul_(strength)
    calibrated = dict(saved)
    calibrated["model"] = state
    calibrated["optimizer"] = None
    calibrated["scaler"] = None
    calibrated["meta"] = dict(meta)
    calibrated["meta"]["coupled_set_calibration"] = {
        "source_checkpoint": source,
        "strength": strength,
        "scaled_state_keys": selected,
        "optimizer_restored": False,
    }
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated_checkpoint(saved, args.strength, args.checkpoint), output)


if __name__ == "__main__":
    main()
