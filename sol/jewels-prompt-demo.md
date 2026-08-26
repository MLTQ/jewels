# `jewels-prompt-demo.service`

## Purpose

Runs the prompt-to-Jewel browser proof as a persistent user service on the LAN GPU machine.

## Components

### Service process

- **Does**: Starts `sol.prompt_video_demo` from the GPU worktree on the free RTX 2070 SUPER.
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
