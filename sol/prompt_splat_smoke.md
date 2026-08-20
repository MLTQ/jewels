# `prompt_splat_smoke.py`

## Purpose

Tests the cheapest end-to-end promptability question available before training a text-to-latent
generator: does prompt identity survive the already-pretrained text-to-video scaffold, the
video-to-jewel encoder, and support-correct jewel rendering?

## Components

### `main`
- Selects the 12 held-out action prompts in one fixed visual style.
- Encodes seven evenly spaced scaffold frames and their matching splat render with frozen
  OpenCLIP, then compares each render with its correct prompt, a deterministic shuffled prompt,
  and the null prompt.
- Reports paired margin/win rate and the fraction of source-video semantic alignment retained by
  the splat bottleneck, with a three-control bar chart.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Promptability report | Correct/shuffled/null conditions use the same rendered fields | Control construction |
| Claim scope | This is explicitly a bottleneck-retention test, not direct text-to-jewel generation | Report scope string |
