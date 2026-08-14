# `train_scaffold_mark_flow.py`

## Purpose

Trains one stochastic 1,024-rank jewel-mark realizer across the initial window and every later
complete stride, removing the continuation-only 512-rank/fitted-prefix limitation.

## Components

### `PreparedScaffoldMarkView`

- **Does**: Keeps one normalized variable-cardinality target, causal context, aligned cell-RGB
  scaffold, topology, and owned prompt rows resident on the selected device.
- **Saliency state**: Precomputes mean-normalized scaffold cell importance from foreground,
  motion, chroma, and boundaries for optional per-jewel feature weighting.

### `_feature_objective`

- **Does**: Applies scaffold-cell saliency to each owned jewel's rectified-flow velocity MSE and
  normalizes by the realized row weights.
- **Lifecycle-safe mode**: `spatial-appearance` weights spatial centers/covariance, RGB/gradients,
  and opacity while keeping temporal center and time-coupled covariance dimensions uniform.
- **Rationale**: This retains the stable dense feature objective while shifting model capacity
  toward actor/action cells, avoiding the high-variance sparse render gradients rejected by the
  first motion-aware ablation.

### `_load_guides` / `_prepare`

- **Does**: Aligns every source video stride with the shared grid and preserves explicit empty
  context at frontier zero.
- **Interacts with**: `scaffold_mark_data.py` and `video_guide.py`.
- **Render state**: Keeps the exact fitted carried field/background only as supervised training
  targets; neither is serialized into or supplied to autonomous inference.

### `_single_background_initialization`

- **Does**: Initializes an optional three-value learned background from the first scaffold stride's
  mean RGB and rejects any corpus containing more than one physical training source.
- **Rationale**: The single-field capacity gate can learn decomposition gauge without initializing
  from the privileged fitted background.

### `main`

- **Does**: Cycles every train stride, trains rectified flow with text/context/guide dropout,
  evaluates fixed held-out controls, and saves resumable provenance plus train-only standardizers.
- **Rationale**: Cycling guarantees that high-rank initial topology is not under-sampled relative
  to the five continuation strides per 96-frame UCF field.
- **Fine-tune path**: `--initialize-from` retains the same-manifest guard, while the explicit
  `--transfer-from` path imports model-only weights from a compatible corpus and records both
  manifest digests. `--initial-repeat` and optional frontier-anchored render patches support a
  bounded visual correction without restarting the feature-trained flow.
- **Coupled-set augmentation**: `--set-depth 1 --augment-from ...` adds a zero-residual learned
  cell/neighborhood set block while loading every shared tensor from the selected same-manifest
  base. `--freeze-base-on-augment` trains only the new block, making the first experiment an exact
  architectural attribution rather than an unconstrained full-model fine-tune. That frozen mode
  rejects `--learn-single-background`, which would otherwise add an unrelated trainable color
  parameter.
- **Selection snapshots**: `--snapshot-every N` retains immutable numbered checkpoints while the
  ordinary latest checkpoint remains resumable. This prevents a late coupled-set regression from
  erasing the best predeclared evaluation screen.
- **Contribution correction**: An optional loss matches per-jewel alpha, cell alpha mass, and soft
  5%-visible counts exactly at local frontier time zero, addressing weak lifecycle tails directly.
- **Single-field background**: `--learn-single-background` optimizes a sigmoid-bounded RGB
  parameter only through rendered supervision and serializes both its causal initialization
  contract and final value. It cannot be enabled for multi-source training. Its optimizer group
  has no weight decay, avoiding an otherwise hidden bias toward mid-gray; model weights retain
  ordinary AdamW decay.
- **Precision safety**: Feature-only training retains CUDA mixed precision. Any run with rendered
  supervision automatically uses full-precision backpropagation because loss scaling through the
  differentiable covariance renderer can produce infinite gradients and silently skip updates.
  Gradient clipping includes the optional background as well as model parameters.
- **Saliency correction**: Optional scaffold-derived sampling mixes uniform patches with cells rich
  in foreground, motion, rare chroma, and spatial boundaries. Separate rendered terms emphasize
  foreground RGB, moving boundaries, and quiet-region temporal stability.
- **Censored initial boundary**: Projected denoised marks may retain support before frame zero only
  for initial views, matching the corpus convention that clips already-active jewels into the
  first observed time cell. Continuation views retain strict local birth projection.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous mark rollout | Checkpoint architecture is `scaffold_birth_mark_flow_v1` | Save schema |
| Initial topology | Grid has capacity for all observed 1,024-rank cells | Grid/rank basis |
| Leakage-safe evaluation | Feature statistics and optimization rows use train sources only | Split policy |
| Scaffold controls | Null guide/context branches receive explicit dropout training | Dropout semantics |
| Causal renderer | Background contract is first-window scaffold RGB mean | Metadata value |
| Render fine-tune | Fitted background/carry are target-only and never inference inputs | Ownership |
| Boundary correction | Initial repeats and anchored patches are checkpointed | Train args |
| Contribution density | Frontier threshold matches the density audit and is checkpointed | Threshold |
| Motion-aware ablation | Saliency fraction and all component weights are checkpointed | Loss policy |
| Salient feature ablation | Cell saliency is guide-owned and never changes mark topology | Ownership |
| Cross-corpus adaptation | Transfer validates the exact architecture/grid/rank basis and starts a fresh optimizer | Initialization policy |
| Coupled-set spike | Augmentation may omit only `set_blocks.*`; base freezing is checkpointed | Partial-load policy |
| Checkpoint selection | Numbered snapshots carry the same state/provenance as latest | Save cadence |
| Single-field memorization | Learned background starts from causal guide RGB, not fitted metadata | Gauge ownership |

## Notes

- This first gate retains the selected cell-RGB guide rather than pre-emptively adding multiscale
  attention. A visual failure can then be attributed before expanding the architecture.
- Render updates are cadence-corrected so `render_weight` retains its expected scale.
- Set depth zero omits the new constructor arguments from metadata, preserving exact compatibility
  with historical v1 checkpoints.
