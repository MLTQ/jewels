# `amortized_encoder.py`

## Purpose

The dense-intermediate pivot's core module (`jewels-2a5`): a feed-forward video-to-jewel-field
encoder that replaces generative set sampling with amortized fitting. It converts the failing
problem (jointly-coherent mark hallucination) into the working one (regression against video
with the fitter's own loss), supervised by unlimited teacher-generated (video, field) pairs.

## Components

### `encode` / `decode`
- **Does**: `encode` maps a window to the generatable latent — `cells` (per-cell trunk features,
  `(n_cells, model_dim)`) and `seed` (per-slot RGB sampled on the fixed lattice,
  `(n_cells, slots, 3)`); `decode` turns any such latent into jewel parameters with no video in
  scope. `forward` is exactly `decode(encode(video))`, verified bit-exact by test.
- **Rationale**: Both latent parts are video-derived, so a text-conditioned generator must emit
  both (Stage 1). The `seed` tensor is effectively a coarse RGB volume on the slot lattice,
  which makes the generation target a small structured volume rather than an opaque code.

### `VideoToJewelEncoder`
- **Does**: One 3D-conv trunk pools a `(T,H,W,3)` window to the birth-cell grid; a zero-init
  linear head emits a fixed budget of jewels per cell (centers cell-anchored via bounded tanh,
  precision-Cholesky scales biased to sensible extents, opacity biased dim) plus one global
  background. At init the field renders to approximately the background.
- **Rationale**: Fixed per-cell budget sidesteps count prediction entirely; near-zero-opacity
  slots are free to act as pruning. Assignment ambiguity is irrelevant because supervision is
  in render space.

### `cholesky_render`
- **Does**: The training-path renderer — identical math to `sol.render.render_exact` (additive
  alpha, P1 color, background) but parameterized by the precision Cholesky factor so
  `torch.linalg.eigh` never enters the gradient graph.
- **Interacts with**: Unit-verified equivalent to `render_exact` after
  `cholesky_to_log_covariance` conversion.

### `cholesky_to_log_covariance` / `canonical_features`
- **Does**: No-grad conversion to the canonical 22-D layout so saved fields flow through every
  existing tool (`render_exact`, `perceptual_eval`, edit machinery) unchanged.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_amortized_encoder.py` | Prediction dict keys and shapes | Output contract |
| `perceptual_eval.py` | Saved fields carry canonical `features` + `background` | Feature layout |
| jewels-sb0 gate | Init-near-background start; render-space supervision only | Objective change |

## Notes

- v0 is single-window and carry-free; persistence conditioning is the planned v1 once the
  fidelity gate passes.
