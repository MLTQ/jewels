# `prompt_trajectory_speaker.py`

## Purpose

Compiles an exact registered text prompt plus declared seed into a finite scene/foreground/background
Jewel program. It is the smallest prompt-only realization of the proposed volume-speaking grammar.

## Components

### `PromptTrajectoryProgram`

- **Does**: Records prompt ownership, semantic scene token, two distinct source-level trajectory
  tokens, and seed.

### `PromptTrajectorySpeaker`

- **Does**: Maps exact registered prompts to semantic tokens and uses a deterministic per-scene
  permutation to choose foreground/background tokens.
- **Does**: Compiles cyclic-shuffled and prompt-free null controls without target data.
- **Rationale**: This isolates prompt-to-program causality before replacing the source-backed
  vocabulary with a learned large vocabulary.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2b0 audit | Three exact prompts and six dense donor rows per scene | Vocabulary ownership |
| Reproducibility | Prompt plus integer seed uniquely determines the program | RNG mapping |
| Claim scope | Compiler is finite/template-backed, not a trained open-vocabulary LLM | Model description |
