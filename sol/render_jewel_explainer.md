# `render_jewel_explainer.py`

## Purpose

Renders the six-part technical series from frozen episode specifications. It synthesizes original
local narration, times animation to real speech, embeds exact proof assets, exports subtitles and
scripts, and muxes browser-compatible MP4s.

## Components

### Media helpers

- **Does**: Run ffmpeg/say commands, probe durations, format SRT time, decode retained proof MP4
  frames, and concatenate padded narration chunks.

### `subtitle_rows` / `write_srt`

- **Does**: Allocate sentence captions proportionally within each real spoken shot duration and
  write sidecar subtitles.

### `synthesize_narration`

- **Does**: Generates one Daniel-voice narration chunk per shot at the declared rate and appends a
  fixed visual breathing pause.

### `load_assets`

- **Does**: Loads committed causal evidence images and decodes actual generated Jewel MP4 frames for
  animation inserts.

### `render_silent_video` / `mux_episode`

- **Does**: Streams 1280×720 RGB frames to H.264, then combines video, AAC narration, and mov_text
  subtitles with fast-start metadata.

### `render_episode`

- **Does**: Produces MP4, SRT, narration script, poster, duration, and provenance for one chapter.

### `validate_specs` / `main`

- **Does**: Enforces six numbered seven-shot episodes, registered visuals, existing claim sources,
  valid timing, episode selection, inventory JSON, and six-poster contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Production command | `--episode N` renders one pilot; no selector renders all six | CLI semantics |
| Series audit | Inventory records media contract, durations, sources, scripts, and subtitles | JSON schema |
| Video players | H.264/yuv420p video, AAC mono narration, English mov_text subtitles | Mux contract |
| Visual QA | Posters and contact sheet derive from the same scene renderer | Asset paths |

## Notes

- macOS `say` must run outside the restricted sandbox to emit audio; no network speech service or
  borrowed narration is used.
