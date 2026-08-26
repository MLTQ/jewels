# `render_jewel_explainer.py`

## Purpose

Renders the six-part technical series from frozen episode specifications. It synthesizes consistent
reference-conditioned narration through Pharaoh Qwen3-TTS or a macOS fallback, times animation to
real speech, embeds exact proof assets, exports subtitles and scripts, and muxes compatible MP4s.

## Components

### Media helpers

- **Does**: Run ffmpeg/say commands, probe durations, format SRT time, decode retained proof MP4
  frames, normalize narration loudness, and concatenate padded narration chunks.

### `subtitle_rows` / `write_srt`

- **Does**: Allocate sentence captions proportionally within each real spoken shot duration and
  write sidecar subtitles.

### `synthesize_narration`

- **Does**: Uploads the selected original officer reference once per episode, generates one
  deterministic ICL VoiceClone take per shot, targets -18 LUFS, and appends a breathing pause.
- **Fallbacks**: Retains Qwen CustomVoice and macOS `say` for controlled comparisons.
- **Interacts with**: `QwenTTSClient` in `qwen_tts_client.py` for asynchronous remote jobs.

### `narration_duration_bounds` / `qwen_token_ceiling` / `synthesize_qwen_shot`

- **Does**: Estimates duration from a 145-WPM baseline, converts the maximum accepted duration into
  Qwen's approximately 12.5-Hz audio-token ceiling, rejects cap hits and temporal outliers, and
  retries with a deterministic alternate seed.
- **Rationale**: A 2,048-token CustomVoice request once produced a 164.5-second runaway take for a
  normal sixty-seven-word shot; the bound makes that failure structurally impossible.

### `load_assets`

- **Does**: Loads committed causal evidence images and decodes actual generated Jewel MP4 frames for
  animation inserts.

### `focus_evidence_asset`

- **Does**: Reflows the two portrait audit sheets into horizontal, claim-specific montages. The
  persistent-owner shot shows target/oracle pairs for all three scenes; the causal-control shot
  shows the three exact correct-versus-shuffled pairs.
- **Rationale**: Containing an entire 1,000–1,500-pixel-tall audit sheet in a 370-pixel video panel
  made the evidence technically present but visually unreadable.

### `render_silent_video` / `mux_episode`

- **Does**: Streams 1280×720 RGB frames to H.264, then combines video, AAC narration, and mov_text
  subtitles with fast-start metadata.

### `render_episode`

- **Does**: Produces MP4, SRT, narration script, poster, duration, and provenance for one chapter.

### `merge_episode_records`

- **Does**: Replaces newly rendered episode records while preserving untouched episodes already in
  the output inventory.
- **Rationale**: A visually revised pilot can be promoted into the complete artifact set without
  resynthesizing its accepted narration or leaving a one-episode inventory behind.

### `validate_specs` / `main`

- **Does**: Enforces six numbered seven-shot episodes, registered visuals, existing claim sources,
  valid timing, episode selection, inventory JSON, and six-poster contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Production command | `--episode N` renders or replaces selected episodes; no selector renders all six; Qwen clone is default | CLI semantics |
| Series audit | Inventory records media contract, durations, sources, scripts, and subtitles | JSON schema |
| Video players | H.264/yuv420p video, AAC mono narration, English mov_text subtitles | Mux contract |
| Visual QA | Posters and contact sheet derive from the same scene renderer | Asset paths |

## Notes

- Qwen production uses the Pharaoh Base model on port 18001 and records its reference hash,
  transcript, ICL mode, accepted seeds, rejected attempts, timing bounds, and sampling settings.
- The selected reference is an original warm American first-officer archetype produced by
  VoiceDesign; it is not an imitation of a real actor or fictional character performance.
- `--tts-backend qwen-custom` and `--tts-backend say` remain comparison fallbacks.
- When direct LAN access is restricted, point `--qwen-url` at a temporary local SSH tunnel.
