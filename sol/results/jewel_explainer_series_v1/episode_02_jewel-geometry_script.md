# Episode 2: What is a spacetime Gaussian Jewel?

A soft shape that exists across position and time

## Claim sources

- `stprim/core/params.py`
- `stprim/prior/featurize.py`
- `stprim/models/render.py`
- `stprim/models/tiled_support.py`
- `sol/generate_jewel_isolation_asset.py`
- `sol/results/jewel_explainer_series_v1/assets/actual_jewel_isolation.json`

## 1. A soft blob through time

Picture a video as a stack of transparent sheets, one sheet per frame. A Jewel is a soft three-dimensional blob placed inside that stack. Its three directions are left-to-right, up-and-down, and time. Mathematicians call its smooth bell-shaped fade a Gaussian: it is strongest at the center and gradually fades away. A visible frame is simply one slice through the blob.

**On screen:** A Jewel is a soft Gaussian blob in left-right, up-down, and time.

## 2. Meet four actual Jewels

This is a real fitted video of a singer at a microphone, built from six thousand four hundred seventy-one Jewels. We mark four strong, moving Jewels, fade away every other contribution, and follow those same four through all sixty-four frames. Their centers move and their visible cross-sections change as time passes. We brighten the isolated view equally so the small contributions remain easy to see; their fitted positions, shapes, and colors are unchanged.

**On screen:** Full fitted video, then four actual Jewels, then full fitted video.

## 3. Five kinds of information

A Jewel stores five things. Its center says where and when it lives. Its shape says how wide it is and which way it tilts. Its base color says what it looks like at the center. A color-change table says how that color varies nearby. Finally, a strength value says how much it contributes. The technical name for the shape table is covariance. Together these use twenty-two numbers.

**On screen:** center | shape | base color | nearby color change | strength

## 4. Tilt creates motion

Imagine pushing a cucumber through that stack of frame sheets at an angle. Each sheet cuts the cucumber at a slightly different horizontal position, so the slice appears to move. A tilted Jewel works the same way. If it stretches mostly through time, it persists. If it tilts across both space and time, its visible position moves from frame to frame. Motion is built into the shape itself.

**On screen:** A tilted spacetime shape becomes motion when sliced into frames.

## 5. Color can vary nearby

The base color describes the exact center. A small three-by-three table describes how red, green, and blue change when we move left, up, or forward in time. That table is called a color Jacobian. We will use the full name instead of unexplained shorthand. Think of it as three tiny color slopes, including a slope through time.

**On screen:** color Jacobian = how red, green, and blue change across space and time

## 6. Many Jewels paint one pixel

At each pixel, every nearby Jewel paints a small amount of color. Strong, close Jewels contribute more; weak or distant Jewels contribute less. We simply add those contributions to a learned background color. This is called an additive renderer because the contributions are added, like overlapping pools of light. No Jewel must win exclusive ownership of the pixel.

**On screen:** background + all nearby Jewel contributions = the final pixel

## 7. Check only where a Jewel matters

A Jewel fades forever in theory, but after five of its own widths the remaining effect is tiny. We call the region where it can matter its support. The renderer divides the volume into boxes so it can quickly find every Jewel whose support reaches a pixel. This is a filing system, not a placement grid: Jewel centers remain irregular, and long tilted shapes are not accidentally missed.

**On screen:** A box index speeds up lookup without snapping Jewels to a grid.
