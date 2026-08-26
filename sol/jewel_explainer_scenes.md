# `jewel_explainer_scenes.py`

## Purpose

Maps evidence-backed shot specifications to animated mathematical diagrams. Reusable scene types
keep six long episodes visually consistent while allowing their data, equations, and assets to vary.

## Components

### Scene renderers

- **Does**: Draw pipelines, excluded-input audits, typed programs, spacetime tubes, factor
  vocabularies, canonical feature vectors, equations, support indexes, comparisons, evidence
  images, metric bars, plateau curves, future-stage flows, and actual Jewel video playback.
- **Interacts with**: `JewelCanvas` in `jewel_explainer_style.py` and shot payloads in
  `jewel_explainer_episodes.py`.

### `SCENE_RENDERERS`

- **Does**: Registers every allowed `Shot.visual` name for validation and dispatch.

### `draw_shot`

- **Does**: Creates one 1280×720 frame with episode chrome, the selected animated diagram, key
  caption, and episode progress.
- **Layout guarantee**: Renders the animated diagram onto a separate canvas and composites only
  rows 170 through 629. This makes the title and footer protected safe zones even when a scene
  accidentally draws outside its intended bounds.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `render_jewel_explainer.py` | PIL RGB image per progress sample | Signature/frame mode |
| Episode specs | Every visual name exists and required payload fields are consumed | Registry/payload schema |
| Visual QA | Key equations, evidence images, and captions remain inside safe areas | Layout constants |

## Notes

- Actual proof MP4 frames are decoded as assets and placed inside the vector diagrams; they are not
  recreated or simulated.
- `CONTENT_TOP` and `CONTENT_BOTTOM` are the single safe-area contract shared by every scene.
