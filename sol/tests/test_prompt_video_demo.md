# `test_prompt_video_demo.py`

## Purpose

Protects the prompt demo's seekable-video protocol and the explicit distinction between proven
exact generation and experimental learned wording.

## Components

### `PromptVideoDemoTests`

- **Does**: Covers normal, open-ended, suffix, invalid, and multi-range requests; checks that the UI
  contains canonical prompts and honest mode language.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Browser playback | Single `Range` requests map to inclusive offsets | Parser semantics |
| Scientific claim | UI retains proof and limitation labels | User-facing copy |
