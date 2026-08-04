# `train_dense_autoencoder.py`

## Purpose

Runs the deterministic tokenizer gate on 45k-jewel fitted windows using the sparse variable-count
decoder. It can train on multiple domains with deterministic domain-balanced sampling while
retaining exact source-video holdouts. Defaults fit the allocated 8 GB RTX 2070 SUPER without
worst-cell slot padding.

## Components

### `main`

- **Does**: Loads/splits dense fitted windows, audits 512-rank capacity, prepares compact targets,
  trains sparse feature/count plus sampled-render losses, evaluates held-out renders, and checkpoints.
- **Mixed-domain mode**: `--extra-corpus` adds roots, `--sampling domain-balanced` round-robins
  domains and fits equal-domain normalization statistics, and `--validation-source-ids` pins the
  scientific holdout instead of allowing a new domain to change a seeded split.
- **Exposure control**: when comparing balanced joint training with single-domain runs, multiply the
  joint step budget by the number of domains. With two domains and batch one, 6,000 joint steps give
  each domain the same 3,000 updates as its single-domain control.
- **Encoder comparison**: `--encoder-mode rank` binds canonical rank to each feature before local
  pooling; the default `pooled` mode remains checkpoint-compatible with earlier runs.
- **Interacts with**: `sparse_autoencoder.py`, `corpus.py`, `evaluation.py`, and `render.py`.

### `_prepare_examples` / `_batch`

- **Does**: Precompute canonical compact targets on CPU and transfer only selected 45k-jewel batches.
- **Optional data**: Loads and batches fixed-size source-video motion pools when configured.

### `_sampled_render_loss`

- **Does**: Renders exactly the 45k occupied predictions and targets at fresh points. By default the
  points are uniform; `--motion-render-fraction` replaces a declared fraction with candidates having
  the largest fitted-target change across `t ± motion_time_delta`.
- **Rationale**: Sparse decoding removes padded existence fields without weakening the visual loss.
- **Rationale for motion sampling**: Uniform volume loss is dominated by static architecture. Target
  temporal differences provide a segmentation-free importance signal for pedestrians and other
  moving detail while the retained uniform fraction protects global reconstruction.
- **Source-video pools**: `--motion-points-dir` loads precomputed per-window coordinates from
  `cache_motion_points.py`. These are preferred over online fitted-field candidates because small
  figures cannot be missed by random proposals and static fit shimmer cannot outrank real motion.

### `_atomic_checkpoint`

- **Does**: Preserves resumable model/optimizer/scaler state and marks the sparse architecture in meta.

### Memorization control

- **Does**: `--overfit-name` intentionally uses one named window for both training and evaluation.
- **Rationale**: This is a diagnostic—not a generalization result—that separates insufficient codec
  capacity from a corpus objective that rewards averaging rare foreground detail.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense prior cache | Checkpoint exposes encoder, normalizer, grid, and sparse architecture ID | Meta schema |
| Research comparison | Whole-source holdout and exact sampled renderer match prior tokenizer gates | Protocol |
| Joint-domain comparison | Domain alternation and normalizer weighting remain independent of corpus size | Sampling policy |
| 2070S | Actual decode work is 45k ranks, not 442,368 padded capacity slots | Scaling behavior |

## Notes

- Defaults use a 128-D raster latent: 990k raw jewel values to 110,592 latent values, about 8.95×.
- The first run is a feasibility/performance gate; visual comparison decides whether to scale steps.
