# `aggregate_scene_prompt_scaling.py`

## Purpose

Aggregates Gate 1h's frozen two/four/six-video exact-prompt curve for the shared-scene native Jewel
speaker.

## Components

### `aggregate`

- **Does**: Extracts correct/shuffled/null token and free-run controls, correct-minus-null margins,
  retrieval accuracy, and each report-owned absolute gate.
- **Does**: Calls scaling positive only when token margin, histogram margin, and retrieval are all
  nondecreasing across the registered nested data points.

### `plot`

- **Does**: Keeps zero, the 0.02 histogram margin, and the 2/3 retrieval gate visible alongside the
  data curves.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1h | `COUNT=PATH` reports at nested source counts | CLI/schema |
| Pitch evidence | Failed endpoints and nonmonotonic scaling remain visible | Verdict semantics |
