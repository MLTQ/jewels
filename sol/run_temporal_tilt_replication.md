# `run_temporal_tilt_replication.py`

## Purpose

Executes a frozen multi-source temporal-tilt replication manifest sequentially on one accelerator,
then produces the decision-grade aggregate report. Per-source fitting remains independently
resumable.

## Components

### `validate_manifest(manifest)`
- **Does**: rejects unknown schemas, incomplete protocols, empty source sets, and duplicate IDs
  before GPU work begins.

### `build_command(manifest, source, output_dir, device)`
- **Does**: constructs a shell-free invocation of `temporal_tilt_ablation.py` for one source.
- **Rationale**: exact argument materialization is retained in the manifest and avoids quoting or
  shell-expansion ambiguity in long unattended runs.

### `main()`
- **Does**: runs sources sequentially, relies on seed-qualified inner checkpoints for resume, loads
  completed v2 reports, aggregates them, and embeds the full manifest in the final report.
- **Interacts with**: `aggregate_temporal_tilt.py` and `temporal_tilt_ablation.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Replication runs | `temporal-tilt-replication-manifest-v1` | Manifest schema or flag mapping |
| Aggregate evidence | `<out>/report.json` plus `<out>/<source-id>/report.json` | Output layout |
