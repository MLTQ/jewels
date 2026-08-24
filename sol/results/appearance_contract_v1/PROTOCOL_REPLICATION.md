# Residual-control seed replication v1 — frozen protocol

## Trigger

The eligible residual-control path improved jointly under equal continuation: exact PSNR
`19.35551 -> 19.66478 dB`, LPIPS `0.74760 -> 0.73288`, SSIM `0.80897 -> 0.82256`, and layout
PSNR `22.41838 -> 23.04958`. Occupancy `0.98458`, active fraction `0.62815`, and mixed tilt
`0.52115` pass. The raw-response continuation is not replicated because sampled occupancy
`0.98517` exceeded the frozen `0.985` gate.

## Registered replication

- Replicate residual-control at continuation seeds 1 and 2 from the same bounded total-1,200 source
  used by seed 0. This tests optimization/sampling robustness of the appearance expansion; it is not
  claimed as independent upstream-encoder initialization.
- For each seed, zero-expand the bounded source and run the same 600-step residual-control tranche,
  then continue its checkpoint for the same second 600-step tranche with the same seed.
- Both tranches retain the seed-0 residual-control data, AdamW schedule, LR `1e-4`, render/teacher/
  student samples, structural objectives, support settings, and zero local/responsibility weights.
- All runs use explicitly mapped `CUDA_VISIBLE_DEVICES=1`, which PyTorch identifies as the physical
  RTX 2070 Super. The spare llama service remains runtime-masked only for the experiment.

## Decision rules

At total step 1,200, each seed must pass sampled occupancy `<=0.985`, active fraction `<=0.70`,
mixed tilt `>=0.25`, and finite training. Every eligible final seed is exact-audited together with
seed 0 in one process.

The representation result is replicated only if every eligible exact seed beats both the bounded
control (`18.50932 / 0.75795`) and bounded midpoint (`18.50301 / 0.75308`) on PSNR and lower LPIPS,
with all structure gates retained. Mean and range are reported; a favorable mean cannot hide an
individual failure. Absolute promotion remains mean exact PSNR `>=20 dB` and LPIPS `<=0.40`.

Visual review must inspect all seed columns for shared contrast/boundary recovery and for unstable
negative speckles. Passing this replication supports a compute-scaling feasibility claim for the
irregular reconstruction mechanism; it does not by itself prove a text-conditioned prior.
