# `render_scaffold_topology_realizer.py`

## Purpose

Runs the first visual continuation gate with both learned topology and generated marks. It keeps the
selected UCF-trained raster-guided mark flow frozen and replaces fitted birth cells/counts/ranks
with the scaffold-topology head's output on held-out LTX clips.

## Components

### `_macro_average` / `_density_report`

- **Does**: Compute equal-source render aggregates and contribution-aware density over the emitted
  continuation frames.
- **Rationale**: Raw birth count and exact fitted-rank overlap are not visual density measures.

### `main`

- **Does**: Reconstructs the frozen realizer's exact train-only standardizers, predicts learned,
  cross-class-shuffled, and null topology, synthesizes every learned rank, merges generated marks
  with bit-identical carried jewels, renders matched controls, and writes per-source GIF/contact
  sheets plus a JSON report.
- **Panels**: Fitted target, carried only, oracle topology with generated marks, learned topology
  with generated marks, and shuffled-scaffold topology with the correct mark guide.
- **Control isolation**: The shuffled panel changes only the topology head's scaffold. The frozen
  mark realizer still receives the correct future raster and text, so differences isolate topology.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Leakage-safe LTX gate | Topology checkpoint owns the merged manifest; flow training sources exactly match manifest train rows | Ownership checks |
| Frozen realizer | Cell-RGB guide only, checkpoint standardizers byte-exact, topology fits 512 ranks | Realizer contract |
| Causal topology control | Shuffled guide comes from another class while carry/target/mark guide stay fixed | Control policy |
| Persistent state | Every combined field begins with unchanged carried features | Merge order |
| Visual review | Five panel names and aggregate contact sheet remain stable | Artifact schema |

## Notes

- The current LTX fields provide one 32-frame-prefix/16-frame continuation view. This tests
  autonomous topology/mark coupling, not initial generation or multi-window free-running marks.
- Oracle-topology flow is a realizer upper control, not the fitted target: its marks are sampled by
  the same frozen model.
