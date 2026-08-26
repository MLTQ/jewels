# `train_learned_trajectory_speaker.py`

## Purpose

Trains Gate 2b1's small autoregressive program speaker on exact prompts plus authored paraphrases and
evaluates held-out donor pairs under unseen correct, cyclic-shuffled, and learned-null text.

## Components

### `PROMPT_PARAPHRASES`

- **Does**: Freezes two training paraphrases and one evaluation paraphrase per exact scene prompt.

### `build_program_examples`

- **Does**: Enumerates ordered same-scene donor pairs while reserving cyclic pairs for evaluation.
- **Rationale**: The held-out set changes both wording and foreground/background combination.

### `evaluate_conditions`

- **Does**: Scores identical held-out programs under correct, cyclic-shuffled, and empty text and
  reports per-token NLL plus scene accuracy.

### `main`

- **Does**: Runs the frozen optimizer/stopping contract, records every evaluation, keeps the best
  correct-NLL checkpoint, and writes the token-gate report.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2b1 protocol | Prompt strings, pair split, optimizer, and stopping rule stay frozen | Evidence comparability |
| Learned-speaker audit | Checkpoint includes model args, prompts, embeddings, sources, and best step | Save schema |
| Null control | Ten-percent empty-text dropout is active during training | Control validity |
