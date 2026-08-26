# `qwen_tts_client.py`

## Purpose

Provides the explainer renderer with a small dependency-free client for Pharaoh's asynchronous
Qwen3-TTS service. It owns CustomVoice and VoiceClone request validation, reference upload,
submission, polling, failure handling, and safe remote WAV retrieval.

## Components

### `QwenCustomVoiceRequest`

- **Does**: Defines the deterministic custom-voice payload: text, speaker, instruction, language,
  seed, temperature, top-p, and token ceiling.
- **Contract**: `payload()` always sends an empty `output_path`, causing the remote server to use its
  disposable output area and expose the result through `/files/{job_id}`.

### `QwenVoiceCloneRequest`

- **Does**: Defines a deterministic reference-conditioned generation using a server-local audio
  path, exact reference transcript, ICL mode, and bounded sampling controls.
- **Rationale**: A selected VoiceDesign reference has a more suitable identity than the fixed
  CustomVoice speakers; cloning keeps that identity stable across forty-two independent shots.

### `QwenTTSClient`

- **Does**: Checks service health, uploads clone references, submits CustomVoice or VoiceClone jobs,
  polls `/jobs/{job_id}`, and atomically downloads the completed WAV.
- **Interacts with**: Pharaoh `inference/tts_server.py` on port 18001 and
  `synthesize_narration()` in `render_jewel_explainer.py`.
- **Rationale**: Uses only Python's standard library so media production does not inherit the local
  Torch/Qwen training environment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Explainer renderer | Completed remote jobs become non-empty local WAV files | Endpoint or polling semantics |
| Renderer tests | Typed requests reject invalid sampling and always use remote output mode | Payload keys/defaults |
| Pharaoh server | JSON follows `CustomVoiceParams` or `VoiceCloneParams`; file is retrieved once | API route names |

## Notes

- `/files/{job_id}` deletes the server-side WAV after the response is sent, so failed local writes
  are surfaced and never treated as completed narration.
- Reference uploads use a content-derived stable filename. Pharaoh currently retains uploads, so a
  later render overwrites the same small file instead of accumulating per-shot copies.
- The service may be reached directly at `192.168.0.202:18001` or through a local SSH tunnel.
