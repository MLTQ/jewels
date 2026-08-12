# `render_prompted_mark_flow.py`

## Purpose

Performs the decisive oracle-topology visual gate for stochastic birth marks. It compares the new
flow with the prior deterministic regressor under identical held-out prefixes, prompts, target
topology, carried state, renderer, and frame grid.

## Components

### `main`
- **Does**: Samples correct, shuffled, and text-only mark flows from matched noise; applies the hard
  topology projection; renders them beside fitted, carried-only, deterministic, raw-flow, and
  projected-target controls; writes GIF/contact/JSON artifacts.
- **Interacts with**: `BirthMarkFlowModel`, `BirthContinuationModel`, the washout metrics, and the
  exact jewel renderer.
- **Rationale**: Target counts/cells/ranks isolate whether stochastic mark generation restores
  contrast and coherent geometry before any topology model is justified.
- **Oracle-guide mode**: Cell-raster and multiscale-token checkpoints replace the text-only panel
  with a same-noise zero-guide ablation and show raw/projected guided samples plus a shuffled-text
  guide control. Multiscale sampling geometry comes from checkpointed trainer arguments.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Architecture decision | Flow and deterministic baseline use the same topology/corpus | Checkpoint checks |
| Prompt control | Correct and shuffled conditions begin from identical Gaussian noise | Seed policy |
| Projection audit | A projected-target panel reveals any distortion caused by hard constraints | Panel set |
| Persistent continuation | Every candidate shares bit-identical carried jewels | Merge semantics |

## Notes

- Stochastic samples need not maximize paired PSNR, but they must recover substantially more target
  detail energy than the deterministic conditional mean.
- The symmetric local/global covariance transform runs on CPU to avoid oversized CUDA eigensolver
  workspace on consumer GPUs.
