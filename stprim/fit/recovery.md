# recovery.py

## Purpose

Filesystem-safe persistence for an in-progress primitive fit. The fitting loop defines and
validates the state; this module keeps serialization mechanics out of the optimizer code.

## Components

### `RECOVERY_SCHEMA`

- **Does**: version the on-disk recovery envelope used by `cli/fit_corpus.py`

### `tensors_to_cpu(value)`

- **Does**: recursively clone a tensor tree to CPU so a saved recovery state is an exact,
  device-independent snapshot rather than a live view of parameters that continue changing

### `atomic_torch_save(payload, path)`

- **Does**: save to `.<name>.tmp` beside the target, then atomically replace the target
- **Why**: power loss or process termination during serialization leaves the prior recovery
  checkpoint intact

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | `tensors_to_cpu` preserves nested dict/list/tuple structure | Tensor-tree conversion semantics |
| `cli/fit_corpus.py` | same-directory atomic replacement and `RECOVERY_SCHEMA` | File naming or schema value |

## Notes

- Payloads intentionally contain only tensors and Python scalar/container types. Recovery files
  are local trusted artifacts, but this also keeps them compatible with restricted torch loading.
- An interrupted write can leave a hidden `.tmp`; the last complete recovery remains valid.
