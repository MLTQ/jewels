# `jewel_explainer_episodes.py`

## Purpose

Freezes the six-episode technical narrative, shot captions, visual specifications, and local claim
sources. It is the editorial source of truth for narration, subtitles, and animation content.

## Components

### `Shot`

- **Does**: Couples one narrated technical claim to a concise on-screen caption, reusable visual
  kind, and visual payload.

### `Episode`

- **Does**: Names and orders one seven-shot chapter and records the code/report files supporting its
  claims.

### `EPISODES`

- **Does**: Defines six chapters: prompt compilation, Gaussian spacetime geometry, physical
  vocabulary, coherence/trajectories, evidence gates, and the scaling program.
- **Rationale**: Claims explicitly retain failures and scope boundaries; narration cannot silently
  promote the learned near-pass or source-backed vocabulary.

### `episode_by_number`

- **Does**: Resolves a stable one-based episode identifier for CLI rendering.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `render_jewel_explainer.py` | Six unique episodes with ordered shots and narration | Dataclass/schema |
| `jewel_explainer_scenes.py` | Every `Shot.visual` has a registered renderer | Visual names/payloads |
| Series audit | Every episode records existing local evidence paths | Source ownership |

## Notes

- “Three-dimensional spacetime Gaussian” is used deliberately: `(u,v,t)` is the Gaussian domain;
  RGB is attached appearance, not a fourth geometric axis.
