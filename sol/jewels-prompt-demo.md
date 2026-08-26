# `jewels-prompt-demo.service`

## Purpose

Defines the prompt-to-Jewel browser proof user service for the LAN GPU machine. The service was
disabled after the demo so it cannot reclaim a GPU without an explicit restart.

## Components

### Service process

- **Does**: Starts `sol.prompt_video_demo` from the GPU worktree through the shared LTX-2 Python
  environment; the environment supplies dependencies but the demo does not load the LTX-2 model.
- **Does**: Binds port 7860 on the LAN and stores generated MP4/JSON pairs beneath the proof results
  directory.
- **Rationale**: A user service survives terminal disconnects and restarts after transient failures.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| LAN demo | `http://192.168.0.202:7860` | Host, port, bind address |
| Runtime | GPU worktree and LTX virtual environment paths | Machine layout |
| Evidence | Outputs remain under `jewel_prompt_demo_v1/generated` | Output path |

## Notes

- This is a machine-specific deployment unit, not a portable production service.
- The UI intentionally has no authentication and must remain confined to the trusted LAN.
- In this Python environment `cuda:0` maps to the RTX 4090, despite `nvidia-smi` listing the 4090 as
  physical index 1. The unit must remain disabled until it is moved to a dedicated environment and
  the 8 GB 2070 path is verified.
