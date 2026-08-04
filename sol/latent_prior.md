# `latent_prior.py`

## Purpose

Defines the final text-to-video generation locus: a conditional rectified-flow model over a small,
canonical raster of jewel latents. This same velocity interface plugs directly into dirty-cell
inpainting.

## Components

### `RasterFlowPrior`
- **Does**: Predicts latent velocity from noisy raster cells, flow time, and an optional text
  embedding.
- **Interacts with**: `OccupancyAwareEncoder` output and `masked_flow_inpaint` callable contract.
- **Rationale**: Raster cell positions are canonical, so positional embeddings are meaningful and
  raw-jewel ordering gauge is absent.

### `ConditionalBlock`
- **Does**: Applies cell self-attention and MLP updates modulated by text/time via adaLN-Zero.

### `flow_matching_loss`
- **Does**: Trains noise-to-latent rectified flow with classifier-free condition dropout.

### `flow_matching_objective`
- **Does**: Scores caller-supplied noise, time, and condition-drop masks.
- **Rationale**: Held-out conditional, shuffled, and unconditional comparisons must share the exact
  same flow paths.

### `masked_flow_matching_loss` / `masked_flow_matching_objective`
- **Does**: Keep clean cells at their target values for the full flow path and score velocity only
  inside dirty cells; pass the mask to models that declare mask conditioning.
- **Rationale**: This exactly matches clamped editor inference; full-generation training never
  exposes the model to a clean-context/noisy-hole state.

### `sample_flow`
- **Does**: Euler-integrates normalized raster noise with optional classifier-free guidance.
- **Interacts with**: Frozen latent normalization in `latent_data.py` and the tokenizer decoder.

### `timestep_embedding`
- **Does**: Maps normalized integration time to sinusoidal model features.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Text-to-video training/evaluation | `(B,C,D)` velocity and explicit fixed-path scoring | Forward signature |
| `masked_flow_inpaint` | Model callable accepts nullable text condition | Conditioning semantics |
| Masked-repair fine-tuning | Clean context is fixed and loss is dirty-only | Flow-path construction |
| Future checkpoints | `n_cells`, latent width, text encoder identity, and model args are serialized | Architecture defaults |

## Notes

- `text_condition` may come from CLIP, T5, or a multimodal adapter, but one encoder identity and
  normalization must be frozen corpus-wide.
- For edit repair, augment the text condition with protected moved-jewel summaries or add dedicated
  cross-attention. Text alone cannot communicate the hard geometric constraint.
