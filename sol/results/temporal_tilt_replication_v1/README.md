# Temporal-tilt multi-source replication v1

## Decision

The replicated representation-causality gate passes. Across three real motion classes and three
paired seeds per class, freely tilted spacetime Gaussians beat axis-aligned anisotropic Gaussians in
all 9/9 comparisons at exactly matched primitive counts, parameter bytes, optimizer compute, and
support-correct rendering.

Under `sol/EVIDENCE_POLICY.md`, this is decision-grade evidence for prioritizing time-distorted
Gaussian splats as the stage-1 representation. It is not yet evidence for promptability.

## Frozen protocol

- Sources: UCF PlayingGuitar, PoleVault, and SalsaSpin; source paths, byte sizes, and SHA-256 hashes
  are embedded in `report.json`.
- First 16 frames, resized to 80×107.
- Seeds 0, 1, and 2; free and axis-aligned arms paired within each seed.
- 900 steps, 8,192 sampled voxels/step, 500 initial and 2,000 maximum primitives.
- Five-sigma support-complete rendering, capacity 1,024, query chunk 512.
- Every arm ends with 791 primitives and 72,784 raw parameter bytes.
- Runtime: PyTorch 2.12.1+cu130 on the UUID-pinned RTX 4090
  `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.

## Results

| Source | Seed deltas, free − axis | Mean | Paired 95% CI | Wins |
|---|---|---:|---:|---:|
| PlayingGuitar | +0.778, +0.859, +0.939 dB | +0.858 dB | [+0.658, +1.059] | 3/3 |
| PoleVault | +0.330, +0.511, +0.523 dB | +0.455 dB | [+0.185, +0.724] | 3/3 |
| SalsaSpin | +0.924, +0.996, +1.089 dB | +1.003 dB | [+0.797, +1.209] | 3/3 |
| **Aggregate** | nine paired runs | **+0.772 dB** | **[+0.573, +0.971]** | **9/9** |

The axis-aligned controls have 0.000 mixed spacetime tilt. Free fields have mean median mixed tilt
0.346. The gain is not confined to aggregate PSNR:

- Axis-aligned global RGB MAE is worse by 0.00292, 95% CI [0.00210, 0.00374].
- In target-defined top-20% motion regions it is worse by 0.00509, 95% CI
  [0.00325, 0.00692].
- Free quality-at-budget averages 33.30 versus 32.32 dB per 1,000 primitives and 361.86 versus
  351.25 dB per parameter MB. These ratios are descriptive; the causal statistic is the paired
  PSNR difference at exactly equal budgets.

Every predeclared gate in `report.json` is true: source/pair counts, 7/9 consistency, ≥0.5 dB macro
advantage, confidence interval excluding zero, matched counts/bytes, and successful tilt removal.

## Bounds

- Clips are short and all come from UCF-101; longer shots, camera motion, and broader domains remain
  generalization tests.
- PSNR and local RGB error do not replace perceptual or human evaluation.
- This proves that time distortion earns reconstruction quality per splat. It does not prove that a
  text-conditioned model can generate those fields.

The aggregate machine report is `report.json`; the three source-level v2 reports are retained in
the adjacent subdirectories. Eighteen checkpoints remain on the GPU host under
`/home/m/jewels-codex-support/runs/temporal_tilt_replication_v1`.
