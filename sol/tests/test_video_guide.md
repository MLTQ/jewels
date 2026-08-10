# `test_video_guide.py`

## Purpose

Protects the axis convention at the semantic-video to jewel-grid boundary.

## Components

### `VideoGuideTests`
- **Does**: Uses coordinate-coded RGB values to verify exact `(t,v,u)` to `(u,v,t)` conversion.
- **Interacts with**: `video_guide.py` and `GridSpec`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Oracle semantic-guide gate | Each frame pixel conditions the intended jewel cell | Flatten order |
