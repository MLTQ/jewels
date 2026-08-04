# `train_autoencoder.py`

## Purpose

Runs the first real research gate on fitted corpus checkpoints: train a structured autoencoder and
measure rendered reconstruction on entirely held-out source videos. Defaults target the allocated
8 GB RTX 2070S.

## Components

### `main`
- **Does**: Loads/splits the corpus, audits capacity, fits train-only normalization, prepares compact
  targets, records an untrained render baseline, trains with structural plus sampled-render loss,
  evaluates rendered PSNR, and writes atomic resumable checkpoints.
- **Interacts with**: `corpus.py`, `autoencoder.py`, `evaluation.py`, and `token_grid.py`.

### `_prepare_examples` / `_batch`
- **Does**: Precompute compact canonical targets on CPU and transfer only the sampled batch.
- **Rationale**: Dense empty target slots would consume unnecessary host and GPU memory.

### `_sampled_render_loss`
- **Does**: Denormalizes target-corresponding decoded slots and minimizes exact-render RGB MSE at
  fresh uniform spacetime points.
- **Rationale**: The first 1,000-step pilot recovered 97.8% of jewel count while held-out render PSNR
  plateaued at 11.72 dB; equal feature weighting is therefore not aligned with visual fidelity.
- **Boundary**: Existence/count losses still govern which slots survive decoding. Rendering every
  possible empty slot would be unnecessarily expensive and would make soft existence semantics part
  of the renderer.

### `_atomic_checkpoint`
- **Does**: Writes to a temporary file before replacement so interrupted saves are never mistaken for
  resumable checkpoints.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future latent-prior trainer | Checkpoint contains encoder weights, grid, normalizer, and split | Meta schema |
| Research log | JSONL includes baseline, structural/render losses, and source-balanced evaluations | Log schema |
| 2070S run | Default model/grid fits within 8 GB and uses `cuda:1` | Memory-affecting defaults |

## Notes

- Defaults use `12×12×6`, 80 slots: measured maximum occupancy is 67 on the 6,471-jewel Avenue
  corpus. The code still audits every example and aborts if a future corpus exceeds capacity.
- Dense 45k-jewel training needs a sparse/hierarchical decoder; do not merely raise slots on this
  dense decoder until profiling proves the memory budget.
- Set `--render-weight 0` for the structural-only ablation. The default uses 32 new points per
  example and weight 0.1.
