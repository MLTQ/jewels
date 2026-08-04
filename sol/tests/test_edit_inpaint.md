# `test_edit_inpaint.py`

## Purpose

Protects the interactive-editing invariants: selection changes only intended centers, dirty cells
cover the edit neighborhood, moved jewels remain explicit constraints, and clean latents never drift.

## Components

### `EditAndInpaintTests`
- **Does**: Covers translation, edit planning, exact clamping under conditional flow updates, and
  explicit dirty-mask dispatch to mask-aware priors.
- **Interacts with**: `geometry.py`, `edit.py`, `inpaint.py`, and `token_grid.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future editor | Moved jewels are preserved separately from generated repair | Edit-plan semantics |
| Future prior | Dirty-only sampling cannot alter clean latent cells | Clamp invariant |
| Mask-aware prior | Sampler forwards one clean/dirty value per raster cell | Mask dispatch |
