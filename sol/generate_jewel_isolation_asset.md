# `generate_jewel_isolation_asset.py`

## Purpose

Builds the evidence clip used in episode 2 of the Jewel explainer. The clip starts with a complete
fitted reconstruction, marks four actual fitted primitives, fades every other contribution away,
tracks those same primitives through all 64 source frames, and restores the complete field.

## Components

### `select_visible_moving_jewels`

- Filters for on-screen centers, useful temporal extent, and nonzero conditional motion.
- Ranks a 192-candidate pool by contribution measured from the actual Gaussian renderer on a
  coarse spacetime grid. Opacity or primitive size alone is not treated as proof of visibility.
- Selects one strong candidate at each of four evenly spaced time anchors and enforces modest label
  spacing. This keeps the 64-frame isolation pass populated while honestly showing each Jewel's
  local lifespan.

### `covariance_and_velocity`

- Converts fitted widths and rotations into the full covariance matrix.
- Uses the conditional Gaussian mean
  `mu_uv + covariance_uv,t / covariance_t,t * (t - mu_t)` to follow the center of each visible
  frame slice. This is the same derivative interpretation used by the project’s stronger motion
  analysis, rather than a guessed screen animation.

### `_isolated_display`

- Renders only the four selected fitted primitives with exact all-primitive evaluation.
- Mattes their positive RGB contributions over eggshell and applies one declared exposure gain so
  tiny contributions survive video scaling. The gain changes presentation brightness only; center,
  covariance, time evolution, fitted color direction, and strength-derived alpha all come from the
  checkpoint.

## Outputs

- `actual_jewel_isolation.mp4` — 108-frame explanatory sequence.
- `actual_jewel_isolation.json` — selected field rows, covariances, conditional velocities,
  selection scores, checkpoint hash, render policies, and a frame-by-frame timeline.
- `actual_jewel_isolation_contact.png` — five checkpoints spanning full, identified, isolated,
  evolving, and restored views.
- `singer_field_additive_seed0.pt` — the 6,471-row fitted checkpoint copied beside the generated
  assets; its SHA-256 is frozen in the JSON.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `jewel_explainer_scenes.py` | Four consistently colored numeric labels and a 108-frame clip | Label count/order or timeline stages |
| `render_jewel_explainer.py` | Assets under `sol/results/jewel_explainer_series_v1/assets/` | Output names |
| Visual audit | Metadata identifies exact checkpoint rows and discloses display gain | Selection/render provenance |

## Notes

- The selected checkpoint predates support-complete culling, so the full reconstruction deliberately
  uses its checkpointed legacy 64-neighbor renderer. The four-jewel isolated view is rendered exactly.
- The fitted field is evidence, not decoration: no hand-authored Jewel geometry is substituted.
