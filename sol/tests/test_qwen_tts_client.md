# `test_qwen_tts_client.py`

## Purpose

Protects the remote narration boundary without requiring a live GPU or Pharaoh service during the
ordinary test suite.

## Coverage

- Validates typed CustomVoice and VoiceClone inputs and the mandatory server-local `output_path`
  contract.
- Simulates asynchronous submission, running and complete job states, one-time file download, and
  atomic local placement.
- Exercises reference upload, ICL transcript propagation, and clone-result retrieval.
- Confirms deterministic seed propagation and useful server-error reporting.

## Rationale

The mocked transport exercises the actual URL and JSON behavior while keeping tests deterministic,
fast, and independent of LAN availability or model load state. A real Qwen sample render remains the
artifact-level integration gate.
