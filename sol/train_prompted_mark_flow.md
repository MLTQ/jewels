# `train_prompted_mark_flow.py`

## Purpose

Trains the bounded oracle-topology experiment selected after the washout decomposition rejected a
count-only fix. The model learns a stochastic flow over direct 22-D birth marks while target
cells/ranks/counts remain fixed.

## Components

### `PreparedMarkFlowView`
- **Does**: Keeps one train-only-normalized target set, exact topology, local prefix raster, and
  owned prompt rows resident on the target device.

### `main`
- **Does**: Loads the leakage-safe 12/4 prompt corpus, trains Gaussian-noise-to-mark rectified flow,
  evaluates fixed-path prompt controls, and saves resumable provenance.
- **Text dropout**: Trains the null branch required for classifier-free controls.
- **Context dropout**: Exposes the model to text-only one-to-many generation instead of silently
  assuming a zero prefix at inference.
- **Interacts with**: `BirthMarkFlowModel`, `prompted_mark_flow_eval.py`, and
  `streaming_corpus.py`.
- **Oracle guide control**: `--oracle-video-guide` decodes each true future stride at low
  resolution and aligns it with the birth grid. This tests whether a semantic video scaffold can
  drive coherent jewel realization; it is not an inference-time claim.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Oracle mark renderer | Checkpoint stores architecture, grid, standardizers, and corpus identity | Save schema |
| Research decision | Topology never enters the learned objective in this experiment | Target ownership |
| Recovery | Optimizer, scaler, and exact completed step are restorable | Checkpoint fields |
| Oracle guide | Original source frames and fitted fields use identical frame indices | Alignment |

## Notes

- Empty-birth views are excluded because they define a topology event, not a mark distribution.
- A visual improvement here licenses a stochastic topology model; failure redirects effort toward
  larger/pretrained video supervision rather than a more elaborate count head.
