# `test_prompt_trajectory_speaker.py`

## Purpose

Verifies deterministic prompt-only compilation and matched shuffled/null causal controls.

## Coverage

- exact prompt-to-scene mapping;
- deterministic distinct donor tokens;
- cyclic-shuffled and seed-only null scene ownership;
- rejection of unregistered text.
