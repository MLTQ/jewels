# `benchmark_encoder_support.py`

## Purpose

Measures whether the support-complete renderer makes a corrected encoder convergence curve
practical, using a mature encoder field rather than synthetic geometry.

## Components

### `fixed_points`
- Produces deterministic video voxel coordinates so renderer arms receive identical work.

### `main`
- Loads one existing encoder checkpoint and real generated-video window.
- Compares five-sigma tiled output with the all-center infinite-Gaussian oracle.
- Times complete encoder forward, render loss, and backward passes after warmup and records peak
  allocated GPU memory, then plots the matched time and memory bars.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Encoder convergence protocol | Both arms use the same model, video, points, target, and gradient workload | Timing workload |
| Feasibility report | JSON records raw samples, medians, memory, GPU, and output discrepancy | Report schema |
