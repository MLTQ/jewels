# `audit_irregular_encoder.py`

## Purpose

Runs the decisive matched audit for the irregular-field gate: frozen high-fidelity lattice encoder,
replicated sparse structural candidates, and source-matched support-correct fitted ceilings.

## Components

### `load_baseline` / `load_candidate` / `load_teacher`
- **Does**: Restores each architecture under its recorded metadata and rejects a candidate without
  the versioned irregular architecture identity.

### `structure`
- **Does**: Extends the standard field-structure battery with active fraction and mixed spacetime
  tilt while retaining opacity-filtered occupancy and anisotropy semantics.

### `main`
- **Does**: Selects only validation sources with independently fitted teachers; support-renders the
  same seven frames for every arm and scores PSNR, SSIM, layout, and LPIPS.
- **Does**: Uses a declared bounded point chunk for every renderer so the audit remains mathematically
  identical on accelerators sharing memory with another workload.
- **Does**: Aggregates all candidate seeds for structure, preserves seed/source records, and writes a
  four-arm qualitative sheet plus a pitch-readable comparison graph.
- **Does**: Writes `field_layout.png`, a geometry-only comparison of active center locations in
  matched XY and spacetime XT slices. This separates actual lattice regularity from blur caused by
  overly broad covariance or weak appearance reconstruction.
- **Does**: Reports each seed independently and applies the thresholds frozen in
  `results/irregular_encoder_v1/PROTOCOL.md`; a favorable mean cannot hide a failed seed.
- **Does**: Draws frozen gate thresholds on the comparison chart and expands the occupancy axis so
  a small but real departure from a near-one lattice score remains visible.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Feasibility gate | Lattice, candidates, and teacher share exact source/frame ownership | Audit protocol |
| Prompt-prior decision | Candidate architecture is `structural_jewel_encoder_v2` | Architecture ID |
| Visual review | Sheet columns remain target/lattice/irregular/teacher | Layout |
| Center-layout review | Only jewels above 2% opacity and within the declared coordinate band are plotted | Plot semantics |
