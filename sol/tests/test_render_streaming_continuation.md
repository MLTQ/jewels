# `test_render_streaming_continuation.py`

## Purpose

Protects the global-time and spatial-coordinate contract used by continuation visualizations.

## Components

### `RenderStreamingContinuationTests`

- **Does**: verifies point count, selected global frame times, and full normalized spatial extent
- **Interacts with**: `render_streaming_continuation.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Visual comparison | All control fields render at identical global coordinates | Grid ordering |
