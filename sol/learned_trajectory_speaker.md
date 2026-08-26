# `learned_trajectory_speaker.py`

## Purpose

Defines the small autoregressive model that learns text-to-scene, scene/text-to-foreground, and
text/scene/foreground-to-background token emission for Gate 2b1.

## Components

### `LearnedTrajectorySpeaker`

- **Does**: Projects a frozen text embedding into a shared hidden state, predicts a semantic scene,
  then predicts foreground and background source-level trajectory tokens autoregressively.
- **Does**: Samples scene by argmax and source tokens from top-six temperature-0.8 categorical
  distributions, masking only exact donor repetition.
- **Rationale**: Source tokens are not masked by scene, so prompt-to-scene ownership and valid donor
  selection must be learned from program examples.

### `trajectory_program_loss`

- **Does**: Sums scene, foreground, and background cross-entropy and reports each component.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2b1 trainer | Normalized text rows, dense scene IDs, global source-token IDs | Input schema |
| Gate 2b1 audit | Architecture arguments and state are checkpointed | Save schema |
| Grammar | Background token cannot equal foreground at sampling | Program validity |
