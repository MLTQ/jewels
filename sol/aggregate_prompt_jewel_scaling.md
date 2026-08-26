# `aggregate_prompt_jewel_scaling.py`

## Purpose

Combines the preregistered additive prompt-caster reports into the nested 9/33/57-video scaling
result and a pitch-readable evidence graph. It reports failures as measured; it does not retune
the speaker or thresholds after seeing the curve.

## Components

### `summarize_point`

- **Does**: Validates the additive report schema and extracts correct, shuffled, and prompt-blind
  NLLs; free-running histogram similarity; retrieval accuracy; and the absolute gate result.
- **Does**: Defines a positive NLL margin as `control NLL - correct-prompt NLL`, so positive values
  mean that prompt conditioning helped.

### `aggregate`

- **Does**: Sorts by unique training-field count and applies the frozen monotonic checks to
  correct-versus-shuffled centroid margin, token margin, correct free-run match, and retrieval.

### `plot`

- **Does**: Produces four panels for centroid advantage, token advantage, independently sampled
  field match, and prompt retrieval. Null-prior lines remain visible rather than being omitted.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt scaling protocol | Additive gate-v1 reports at unique nested data sizes | Input schema |
| Pitch evidence | Positive NLL advantage means correct prompt beats its control | Sign convention |
| Reproducibility | JSON summary and PNG are derived from the same report object | Output schema |
