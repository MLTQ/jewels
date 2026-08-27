# `jewel_explainer_episodes.py`

## Purpose

Freezes the six-episode plain-language narrative, shot captions, visual specifications, and local
claim sources. It is the editorial source of truth for narration, subtitles, and animation content.

## Components

### `Shot`

- **Does**: Couples one narrated technical claim to a concise on-screen caption, reusable visual
  kind, and visual payload.

### `Episode`

- **Does**: Names and orders one seven-shot chapter and records the code/report files supporting its
  claims, plus its presentation theme.

### `EPISODES`

- **Does**: Defines six chapters: prompt-to-video flow, Gaussian spacetime geometry, physical
  vocabulary, coherence/trajectories, evidence gates, and the scaling program.
- **Rationale**: Claims explicitly retain failures and scope boundaries; narration cannot silently
  promote the learned near-pass or source-backed vocabulary.
- **Audience contract**: Introduces necessary technical terms on first use, prefers ordinary words
  afterward, uses analogies for geometry and hierarchy, and keeps each shot to one main idea.

### `episode_by_number`

- **Does**: Resolves a stable one-based episode identifier for CLI rendering.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `render_jewel_explainer.py` | Six unique episodes with ordered shots and narration | Dataclass/schema |
| `jewel_explainer_scenes.py` | Every `Shot.visual` has a registered renderer | Visual names/payloads |
| Series audit | Every episode records existing local evidence paths | Source ownership |

## Notes

- Episode 2 uses an eggshell background and black copy. It defines a Gaussian as a soft bell-shaped
  fade, spacetime as left-right/up-down/time, covariance as the shape-and-tilt table, the color
  Jacobian as a local color-change table, additive rendering as overlapping light, and support as
  the region where a Jewel can matter.
- Its second shot identifies four actual rows from a 6,471-Jewel fitted singer field, fades away all
  other rendered contributions, follows those rows through 64 frames, and discloses the uniform
  visibility gain used in the isolated view.
- `JRGB` is intentionally absent from public labels. The earlier shorthand meant the three-by-three
  color Jacobian, but the full term or the plain phrase “color-change table” is clearer.
- “Three-dimensional spacetime Gaussian” remains precise: `(u,v,t)` is the Gaussian domain; RGB is
  attached appearance, not a fourth geometric axis.
