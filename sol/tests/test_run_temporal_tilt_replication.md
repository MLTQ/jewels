# `test_run_temporal_tilt_replication.py`

## Purpose

Protects the frozen manifest runner before it launches a long GPU replication.

## Components

### `manifest()`
- Builds a complete minimal replication-manifest fixture.

### `TemporalTiltReplicationRunnerTests`
- Confirms protocol, seed list, source path, device, and output directory become explicit arguments.
- Confirms duplicate source IDs fail before any subprocess is launched.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/run_temporal_tilt_replication.py` | Strict validation and deterministic command construction | Manifest or CLI mapping |
