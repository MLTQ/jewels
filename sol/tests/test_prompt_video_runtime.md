# `test_prompt_video_runtime.py`

## Purpose

Protects the dependency-free contracts around prompt validation, deterministic artifact identity,
and browser-compatible MP4 encoding without loading GPU artifacts.

## Components

### `PromptVideoRuntimeTests`

- **Does**: Verifies whitespace normalization and limits, prompt/mode/seed filename ownership, and
  the H.264/yuv420p/fast-start ffmpeg command.
- **Does**: Locks the exact-versus-learned realization seed offsets to their frozen audits.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Test discovery | Uses `unittest` without GPU setup | Test class/module names |
| Demo cache | Same normalized request has the same filename | Hash inputs/version |
