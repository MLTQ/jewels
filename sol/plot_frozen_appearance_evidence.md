# `plot_frozen_appearance_evidence.py`

## Purpose

Builds one labeled figure from the registered frozen-appearance reports. It separates the positive
compute/replication result from the negative perceptual-objective ablation and partial stability win.

## Components

### `collect_evidence`

- **Does**: Maps generic audit arm labels to source, render, perceptual, and stabilized semantics;
  extracts the seed-0 compute curve, three final seeds, exact ablation deltas, temporal deviation,
  and logged rendered-range fraction.

### `plot_evidence`

- **Does**: Draws four panels for compute scaling, exact replication, dominated objective deltas,
  and stability/fidelity separation.
- **Rationale**: A single favorable mean would hide both the three-seed threshold result and the
  rejected perceptual formulation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Frozen-appearance report | Registered result directory layout and audit schema | Path mapping |
| Pitch evidence | 20 dB threshold and lower-is-better LPIPS remain explicit | Axis semantics |
