# Execution note

## Hardware and runtime

The long runs executed on `m@192.168.0.202` using the existing project environment.

| Run | GPU | Updates | Wall time |
|---|---|---:|---:|
| raw local seed 0 | RTX 4090 | 12,000 | 1,652 s |
| raw continuation | RTX 4090 | 4,000 | 470 s |
| derivative x32 seed 0 | RTX 2070 | 12,000 | 3,667 s |
| derivative x32 seed 1 | RTX 4090 | 12,000 | 1,700 s |

Audit work ran from the same frozen data and renderer configuration. Checkpoint binaries remain on
the GPU host; this result directory stores logs, summaries, exact JSON reports, and rendered image
evidence without duplicating large model files.

## Monitoring

Training logs were polled throughout the run. Fixed validation was evaluated at regular intervals;
exact support-rendered LPIPS/PSNR audits were run at the registered checkpoints. No run was stopped
because it was merely fast. Raw training was explicitly continued to 16k, and the selected
derivative arm was repeated from a fresh optimizer seed to 12k.

## Reproducibility record

- Seed-0 and seed-1 derivative checkpoints share architecture, data, objective, and schedule.
- Audit display labels are stored in each report while stable candidate keys remain unchanged.
- Both summaries independently verify bitwise-exact frozen geometry and base parameters.
- Qualitative sheets use fixed clip/frame ownership; optimizer progress is the only changing input.
