# `audit_scene_block_constellation_oracle.py`

## Purpose

Runs Gate 2a5, the first explicit two-level native Jewel utterance: a text-owned scene token chooses
a coherent template family and target-derived local block tokens choose complete constellations.
It also runs Gate 2a7's corrected same-address realization when explicitly requested.

## Components

### `scene_key` / `control_conditions`

- **Does**: Maps exact registered style/action labels to three scene tokens and owns correct,
  shuffled-scene, shuffled-block, and prompt-blind conditions.

### `hierarchical_control_metrics`

- **Does**: Measures active-role likelihood while separately disrupting global and local syntax.

### `evaluate_generation`

- **Does**: Casts exact-72k continuous Jewel programs for all four arms, audits structural/render
  metrics and count adjustment, and writes qualitative time-progress sheets.
- **Does**: Separates internal routing-grid irregularity from the fixed 8x8x4 evaluation histogram
  when finer routing is audited downstream.

### Gate assembly

- **Does**: Fits templates only on six videos per scene token, evaluates three direct and nine
  source-disjoint fields, compares with the frozen global posterior, and keeps qualitative scene
  coherence as an explicit advancement condition.
- **Does**: Accepts a fixed comparison-grid override for fine-routing ablations and independently
  gates centroid locking on the internal routing grid.
- **Does**: Can enforce Gate 2a7's `(scene, address, token)` lookup while retaining identical causal
  arms and thresholds.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a5 protocol | Three semantic plus one null scene token and four fixed arms | Control semantics |
| Gate 2b | Scene token followed by 256 time-major Morton block tokens | Hierarchical syntax |
| Scientific review | Text supplies scene class; target supplies only local oracle program | Leakage ownership |
