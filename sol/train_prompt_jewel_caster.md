# `train_prompt_jewel_caster.py`

## Purpose

Trains and audits Gate 1: text directly drives a continuous centroid point process and the passing
three-token Jewel vocabulary. Correct, cyclic-shuffled, and null prompts are compared both with
teacher-forced likelihood and fully prompt-only 72k-Jewel generation.

## Components

### `PromptSampleBatch` / `prompt_label` / `frozen_text_embeddings`
- **Does**: Owns sampled point/token targets and creates frozen normalized BGE embeddings from the
  exact style/action metadata string.

### `control_metrics`
- **Does**: Computes per-role token and centroid NLL for correct, shuffled, and null prompts over the
  fixed source-disjoint validation sample.

### Plateau-controlled training
- **Does**: Balances 8,192 Jewels per training source, applies 10% null-prompt dropout, retains the
  best correct-prompt validation checkpoint, and stops only at the registered plateau.

### Prompt-only generation and gate
- **Does**: Samples 72k centroids and all token IDs from text/seed only, decodes/render them, compares
  target histogram and render controls, audits grid locking and inference inputs, and saves model,
  generated programs, and report.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1 protocol | 33/9 compositional split and BGE conditioning strings | Prompt/data ownership |
| Scientific review | Shuffled/null controls share fixed target samples and declared generation seeds | Control ownership |
| Pitch artifacts | `caster.pt`, `generated_programs.pt`, and report schema v1 | Artifact schema |
