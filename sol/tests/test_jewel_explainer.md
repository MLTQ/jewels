# `test_jewel_explainer.py`

## Purpose

Protects the reproducible contracts behind the six narrated Jewel explainers without requiring a
full media render in the ordinary test suite.

## Coverage

- Verifies quintic easing bounds, monotonicity, endpoints, and invalid reveal intervals.
- Validates all six episodes, forty-two shots, registered scene types, and evidence-source paths.
- Rejects Unicode modifier glyphs that the local Pillow/font stack would render as tofu boxes.
- Smoke-renders every shot type to a 1280×720 RGB frame, including graceful missing-asset states.
- Checks proportional subtitle timing and exact SRT timestamp formatting.

## Rationale

Speech synthesis and H.264 encoding remain an explicit artifact-production gate because they are
slow and platform-dependent. These tests isolate scene and timing regressions early; the render
inventory and media probes verify the final encoded outputs.
