# Episode 1: How a prompt becomes a video

A direct tour from words to moving pixels

## Claim sources

- `sol/prompt_trajectory_speaker.py`
- `sol/semantic_trajectory_realizer.py`
- `sol/prompt_video_runtime.py`
- `sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md`

## 1. Start with the whole idea

Our prototype takes a short text prompt and a random seed. It turns them into a small plan, expands that plan into seventy-two thousand colored shapes called Jewels, and renders forty-nine frames. No finished video is supplied at generation time. The path really runs from words to a plan to a field of Jewels to moving pixels.

**On screen:** prompt + seed → short plan → Jewel field → video

## 2. What goes in—and what does not

A system can appear creative while quietly copying most of its answer. We block that shortcut. Generation receives only the prompt and the declared random seed. It does not receive an input video, a fitted version of the target, a hidden target plan, or a saved target code. Some building blocks still come from training examples; that is an important limit, but the test video itself never enters the process.

**On screen:** Only text and a seed enter generation; no target video is hidden inside.

## 3. The prompt becomes a recipe

Think of the first stage as a tiny recipe writer. It recognizes one of the three prompts the prototype currently knows, then uses the seed to choose two different training-owned ingredients. One ingredient will supply the moving subject and the other will supply the surroundings. The same prompt and seed always produce the same recipe, which makes every result reproducible.

**On screen:** The prompt chooses the scene; the seed chooses two ingredients.

## 4. The recipe stays small

The recipe does not list seventy-two thousand Jewels one by one. It names a scene, a subject ingredient, a setting ingredient, and a seed. Those names act like instructions for whole regions of the video, not individual pixels. This hierarchy matters: the short recipe keeps the subject consistent across time, while the many Jewels provide the visual detail.

**On screen:** A few persistent instructions control many local details.

## 5. Build a tube through time

Imagine drawing the subject's path through a stack of video frames. The path forms a tube: a connected region that moves through space and time. We take the thirty-six thousand Jewels closest to that tube from the subject ingredient. We take thirty-six thousand far-away Jewels from the setting ingredient. The result has an exact size and a clear division of responsibility.

**On screen:** 36,000 subject Jewels + 36,000 setting Jewels = one field

## 6. Place the Jewels irregularly

Each Jewel has a continuous position, a shape choice, a base-color choice, and a color-change choice. Continuous means its center can land anywhere inside the volume, not only at fixed grid points. A small random nudge prevents repeated positions from lining up. The output is therefore an irregular cloud of soft colored shapes rather than a blocky three-dimensional pixel grid.

**On screen:** continuous center + shape + color + color change → one Jewel

## 7. What this proves

The renderer turns the field into a normal video. In nine matched tests, the intended prompt was identified eight times, and correct prompts beat wrong or empty prompts in every comparison. That is real evidence that text controls the generated field. It is not yet a general text-to-video model: the vocabulary contains only three prompts and eighteen training-owned ingredients. We have proved the route, not the scale.

**On screen:** The route works; the vocabulary is still deliberately tiny.
