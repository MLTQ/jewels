# `render_prompt_video.py`

## Purpose

Command-line exporter for full 49-frame videos from the frozen prompt-to-Jewel runtime. With no
explicit prompt it exports all three canonical Gate 2b0 prompts at one declared seed.

## Components

### `main`

- **Does**: Loads `PromptVideoRuntime`, selects canonical or requested prompts, writes MP4/JSON
  pairs, and prints a machine-readable export inventory.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Demo evidence job | No `--prompt` means all canonical prompts | Selection rule |
| Artifact collector | Final stdout contains `prompt-video-exports-v1` JSON | Output schema |
| Runtime | Uses exactly its frozen render and provenance contract | Runtime API |
