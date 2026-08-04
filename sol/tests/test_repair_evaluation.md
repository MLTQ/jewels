# `test_repair_evaluation.py`

## Purpose

Protects reproducible cuboid volume/layout and verifies that repair metrics observe the exact
clean-cell clamp.

## Coverage

- Fixed extents produce the expected dirty volume for every batch item.
- Evaluation reports zero clean error and the correct dirty fraction.
