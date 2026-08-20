# `run_encoder_convergence.py`

## Purpose

Runs the corrected encoder data/convergence experiment as a resumable matrix of three nested
training sizes and three independent seeds.

## Components

### `validate_nested_manifests`
- Refuses to run unless every smaller training identity list is an exact prefix of the next and
  every curve point has the identical ordered validation identities.
- Records the validation identity hash in the protocol artifact.

### `main`
- Freezes renderer, finite support, training/evaluation cadence, stopping rule, and sample counts
  in `protocol.json` before launching training.
- Runs each size/seed pair sequentially on the declared device, skips only runs with completed
  summaries, and appends completed results to the protocol after each run.
- Optionally resumes every arm from a matching first-stage root while freezing a new learning rate
  and warmup schedule in the continuation protocol.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Curve aggregation | Outputs live under `n<size>/seed<seed>` and protocol embeds every summary | Directory/report schema |
| Scientific claim | Size is the only data variable; epochs and convergence rule match | Manifest validation or command construction |
