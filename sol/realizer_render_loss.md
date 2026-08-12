# `realizer_render_loss.py`

## Purpose

Aligns stochastic mark training with the video actually rendered by the jewel field instead of
letting normalized 22-D feature error alone decide visual quality.

## Components

### `estimate_target_marks`

- **Does**: Converts a rectified-flow velocity at an intermediate noisy state into its implied clean
  22-D endpoint.
- **Rationale**: Render supervision must score a denoised mark estimate, not the velocity vector or
  the deliberately corrupted path state.

### `realizer_render_loss`

- **Does**: Samples fresh contiguous spatiotemporal patches in one future stride, renders predicted
  and target births over the same fixed carried field/background, and combines RGB, spatial/temporal
  edge, opponent-chroma, and patchwise SSIM losses.
- **Interacts with**: `render_exact` and frontier-local jewel coordinates.
- **Rationale**: Contiguous patches expose geometry, rare color, motion, and local structure that
  independent uniform points or feature MSE can average away.
- **Efficiency**: Births render directly in frontier-local time; only the fixed carried field uses
  global coordinates. This avoids a differentiable eigendecomposition merely to change time units.
- **Boundary anchor**: `anchor_frontier=True` fixes the first sampled patch at local frame zero so
  low-contribution support tails cannot hide a blank initial frame or continuation seam.
- **Saliency sampling**: A declared patch fraction may be drawn from scaffold cells scored by
  fitted-background deviation, adjacent-time motion, rare chroma, and spatial color boundaries.
  Uniform patches remain in the mix to protect whole-scene fidelity.
- **Foreground/motion terms**: Target-render saliency weights foreground RGB and thin boundaries.
  Motion loss emphasizes temporal differences and their spatial boundary; stability loss emphasizes
  temporal error where the target is quiet, explicitly penalizing stochastic flicker.

### `scaffold_saliency_weights`

- **Does**: Converts the canonical cell-RGB guide into non-negative foreground/motion/chroma/edge
  sampling weights without inspecting fitted jewel features.
- **Rationale**: The low-resolution teacher already locates the actor/action, so it can allocate
  scarce differentiable render queries while preserving the inference ownership boundary.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Mark-flow trainer | Predicted and target births have identical target-owned row topology | Row ownership |
| Renderer | Local query time is `(frame-frontier)/stride` | Time convention |
| Visual objective | Fixed carried jewels and background are identical for both renders | Base composition |
| Streaming fine-tune | Optional anchored patch begins at the current frontier | Patch sampling |
| Motion-aware fine-tune | Guide grid order matches the realizer grid | Saliency addressing |

## Notes

- The structural term is differentiable spatiotemporal SSIM over each sampled patch; it requires no
  downloaded perceptual network and is reproducible on the allocated GPUs.
- Render updates can be subsampled by the trainer; their loss is frequency-corrected there.
