# `prompt_video_demo.py`

## Purpose

Provides a small dependency-free LAN interface for the prompt-to-Jewel proof. The page accepts a
prompt, seed, and speaker mode, then plays and downloads the full rendered MP4.

## Components

### `parse_byte_range`

- **Does**: Parses single HTTP byte ranges so browsers can play and seek generated MP4s.

### `demo_html`

- **Does**: Returns the complete responsive UI, including canonical examples and plain-language
  proof limitations.

### `DemoApplication`

- **Does**: Serializes GPU requests and delegates prompt generation to `PromptVideoRuntime`.
- **Rationale**: The small demo has one GPU and should not overlap resident renderer work.

### `make_handler`

- **Does**: Serves the UI, status JSON, generation API, metadata JSON, and range-aware MP4 files.
- **Rationale**: Uses only the Python standard library because the GPU environment does not carry a
  web framework.
- **Does**: Rejects oversized, non-object, or invalid JSON requests before invoking GPU work.

### `main`

- **Does**: Loads the frozen runtime and serves on the configured LAN address and port.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Browser client | `POST /api/generate` returns video URL and metadata | API schema |
| Video element | `/videos/*.mp4` supports `Range` | Range behavior |
| Demo operator | Defaults to port 7860 and project-relative artifacts | CLI defaults |

## Notes

- The service has no authentication and is intended only for the trusted local network.
- Exact and learned modes are deliberately described differently in both the UI and metadata.
