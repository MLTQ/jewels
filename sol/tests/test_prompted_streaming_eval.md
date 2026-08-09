# `test_prompted_streaming_eval.py`

## Purpose

Protects the direct-jewel prompt-control evaluation schema on a multi-class held-out corpus.

## Components

### `PromptedStreamingEvaluationTests`
- **Does**: Verifies finite correct/shuffled/null metrics for both full-prefix and text-only modes.
- **Interacts with**: `prompted_streaming_eval.py`, `streaming_corpus.py`, and
  `streaming_model.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt research gate | Both context families expose all three text controls | Report keys |
