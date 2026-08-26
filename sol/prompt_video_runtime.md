# `prompt_video_runtime.py`

## Purpose

Loads the frozen Gate 2b0/2b1 artifacts once and exposes one reproducible path from prompt plus seed
to a 72,000-Jewel field, all 49 rendered frames, and a browser-compatible MP4.

## Components

### `PromptVideoPaths`

- **Does**: Names the fine block language, source split, physical codebook, and optional learned
  speaker artifacts.
- **Rationale**: Keeps machine-specific locations out of the generation logic.

### `GeneratedJewelField`

- **Does**: Carries renderable features, background, and complete prompt/program provenance.

### `normalize_prompt` / `video_basename` / `realization_seed`

- **Does**: Validate UI input and derive stable cache-safe filenames from prompt, seed, mode, and
  runtime version.
- **Does**: Preserves the frozen Gate 2b0 exact and Gate 2b1 learned field-RNG offsets so demo
  exports reproduce the audited generation conditions.

### `ffmpeg_command` / `encode_mp4`

- **Does**: Encodes float RGB frames to H.264/yuv420p MP4 with fast-start metadata.

### `PromptVideoRuntime`

- **Does**: Reconstructs the frozen source-backed grammar, compiles either the proven exact prompt
  path or experimental learned path, casts 72,000 irregular Jewels, and renders 49 frames in
  bounded GPU batches.
- **Interacts with**: `PromptTrajectorySpeaker`, `LearnedTrajectorySpeaker`,
  `SemanticTrajectoryRealizer`, and the support-tiled production renderer.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `prompt_video_demo.py` | Resident runtime and `generate_video` return paths/metadata | Method signature |
| `render_prompt_video.py` | Canonical prompts and stable artifact defaults | Prompt/path schema |
| Pitch evidence | Exactly 72k Jewels, 49×144×216 render, full provenance JSON | Frozen proof contract |

## Notes

- Exact mode accepts only the three registered Gate 2b0 prompts.
- Learned mode embeds arbitrary text but can still emit only the learned three-scene, 18-source
  macro vocabulary. It is intentionally described as extrapolation, not open-vocabulary T2V.
