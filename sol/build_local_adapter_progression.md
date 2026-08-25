# `build_local_adapter_progression.py`

## Purpose

Turns a labeled exact convergence audit into pitch-readable image evidence. It removes the unrelated
lattice column, preserves identical held-out frames across checkpoints, and adds exact LPIPS/PSNR
milestones above the visual progression.

## Components

### `build_progression`
- **Does**: Validates the qualitative sheet against report dimensions and candidate labels.
- **Does**: Emits one progression strip per held-out style plus a stacked headline sheet.
- **Does**: Labels each checkpoint with its exact shared-audit LPIPS and PSNR.
- **Rationale**: A compute narrative is credible only when the scene, frame, renderer, and ownership
  remain fixed while optimizer updates change.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Pitch report | Audit protocol contains one display label per candidate checkpoint | Label or report schema |
| Visual comparison | Input columns remain target, lattice, all candidates, teacher | Audit sheet order |
| Style strips | One audit row exists per unique style | Validation-source ownership |
