# `test_jewel_explainer.py`

## Purpose

Protects the reproducible contracts behind the six narrated Jewel explainers without requiring a
full media render in the ordinary test suite.

## Coverage

- Verifies quintic easing bounds, monotonicity, endpoints, and invalid reveal intervals.
- Validates all six episodes, forty-two shots, registered scene types, and evidence-source paths.
- Confirms episode 2 alone selects the eggshell palette and maps ordinary foreground copy to black.
- Proves the four video-frame corners advance together by the drawn diagonal time vector, rather
  than sliding a sheet along the screen's horizontal axis.
- Proves spacetime samples remain hidden before the moving frame crosses their time coordinate,
  reveal monotonically behind it, and reject invalid reveal feather widths.
- Samples the near-frame edge during the sweep to ensure the black frame is composited over the
  teal slice rather than hidden underneath it.
- Rejects Unicode modifier glyphs that the local Pillow/font stack would render as tofu boxes.
- Enforces the audience contract: narration is at most eighty words per shot, captions are at most
  eighteen words, and public copy does not expose internal abbreviations such as `JRGB` or `NLL`.
- Smoke-renders every shot type to a 1280×720 RGB frame, including graceful missing-asset states.
- Compares the rendered header with clean episode chrome to prove animated diagrams cannot enter
  the protected 170-pixel title zone.
- Checks proportional subtitle timing and exact SRT timestamp formatting.
- Checks that a selected-episode rerender replaces only that inventory record and preserves the
  other completed episodes.
- Checks that tall audit sheets are reflowed to the exact horizontal montage dimensions expected by
  the evidence scenes while already-wide evidence remains unchanged.
- Verifies prose-length duration bounds and the derived Qwen token ceiling prevent multi-minute
  runaway takes for ordinary seventy-seven-word shots.
- Hashes the committed singer checkpoint and checks that the real-Jewel asset names four unique
  field rows, carries 3×3 covariances, spans the declared 64-frame source, and exports 108
  explanatory frames.

## Rationale

Speech synthesis and H.264 encoding remain an explicit artifact-production gate because they are
slow and platform-dependent. These tests isolate scene and timing regressions early; the render
inventory and media probes verify the final encoded outputs.
