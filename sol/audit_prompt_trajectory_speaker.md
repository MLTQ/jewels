# `audit_prompt_trajectory_speaker.py`

## Purpose

Runs Gate 2b0: text plus declared randomness compiles directly to a scene/trajectory/background
Jewel program, with no target video, target field, or target-derived local program.

## Components

### `clip_video_embedding`

- **Does**: Mean-pools three rendered frames in the frozen OpenCLIP ViT-B/32 image space and returns
  one unit video embedding.

### `semantic_summary`

- **Does**: Computes intended-prompt top-1 retrieval, correct-versus-shuffled/null generation
  margins, per-class majority retrieval, and pairwise win counts.

### `main`

- **Does**: Fits training-only semantic paths, compiles three prompts at three seeds, renders matched
  correct/shuffled/null programs, and writes per-seed qualitative sheets plus the complete report.
- **Rationale**: This is the first inference path whose only semantic inputs are prompt text and
  declared randomness.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2b0 protocol | Three exact prompts, seeds 20260914–16, frames 0/24/48 | Evaluation set |
| Semantic score | OpenCLIP ViT-B/32 LAION2B weights and three-frame mean | Embedding identity |
| Claim scope | Programs remain source-token-backed and finite-vocabulary | Model description |
