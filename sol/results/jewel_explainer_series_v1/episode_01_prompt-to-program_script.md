# Episode 1: From words to a spacetime program

The exact inference path, without an input video

## Claim sources

- `sol/prompt_trajectory_speaker.py`
- `sol/semantic_trajectory_realizer.py`
- `sol/prompt_video_runtime.py`
- `sol/results/jewel_casting_language_v0/PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md`

## 1. The claim, stated precisely

The current result is a bounded existence proof, not a general text-to-video model. Given one of three registered prompts and an integer seed, the system emits a finite native program, expands that program into exactly seventy-two thousand irregular Jewels, and renders forty-nine video frames. The important point is causal direction. At inference, information flows from text to program to a continuous spacetime field to pixels. It does not flow from a target video backward into an encoding.

**On screen:** prompt + seed → typed program → 72,000 Jewels → 49 rendered frames

## 2. What the generator is forbidden to see

A reconstruction system can look generative while secretly receiving most of the answer. So Gate two-b-zero explicitly removes four channels: no input video, no fitted target field, no target-derived block program, and no held-out latent. The only semantic input is the exact prompt string. The only stochastic input is the declared integer seed. Training-owned source tokens remain in the vocabulary; that is a limitation we will return to, but no test target participates in compilation or casting.

**On screen:** No video, target field, target program, or hidden latent enters inference.

## 3. The exact compiler

The exact speaker normalizes whitespace, looks up the prompt's dense scene index, and then creates a deterministic random generator from the declared seed plus one-thousand-and-nine times the scene. A seeded permutation of the six source tokens owned by that scene chooses two distinct entries. The first becomes the foreground trajectory token; the second becomes the background token. Therefore prompt plus seed uniquely determines the program, while changing the seed changes the donor pair without consulting any pixels.

**On screen:** scene = lookup(prompt); order = randperm(seed + 1009 * scene)

## 4. One three-token utterance

A compiled utterance contains a semantic scene token, a foreground token, a background token, the seed, and an audit condition. Scene is not a pixel class. It selects a persistent semantic path through the entire spacetime window. Foreground and background are not individual Gaussians either. They name coherent, source-backed macro programs containing tens of thousands of physical Jewel tokens. This hierarchy is the central design decision: the small program owns global coherence; local Jewels own rendering detail.

**On screen:** scene / foreground / background are persistent program tokens—not pixels.

## 5. Casting through a moving tube

For every time slab, the realizer stores a two-dimensional semantic path. It computes each donor Jewel's squared distance from that path. The foreground contribution is the thirty-six thousand closest Jewels from the foreground donor. The background contribution is the thirty-six thousand farthest Jewels from the background donor. This rank-balanced construction is deliberately count-exact. It avoids assuming that two valid fields have the same density inside one fixed radius, while forcing both donors to own exactly half of the final field.

**On screen:** closest 36,000 foreground + farthest 36,000 background = 72,000

## 6. Physical tokens become Jewels

Each selected row carries a continuous centroid and three active physical token identifiers: covariance, surface color, and color gradient. Every active vocabulary contains one-thousand-and-twenty-four prototypes. Decoding substitutes the selected prototype values into the canonical twenty-two-dimensional feature layout while leaving the centroid continuous. A small Gaussian jitter is applied and clamped strictly inside normalized volume bounds. The result is an irregular set, not a Cartesian output grid.

**On screen:** continuous μ + covariance token + surface token + gradient token → one Jewel

## 7. A bounded but genuine generation path

The support-correct renderer evaluates that field on every requested frame grid and encodes the result as an H.264 video. The proof passes because text controls which coherent programs are cast: across nine exact programs, intended prompt retrieval is eight of nine, and correct prompt programs beat both cyclic-shuffled and null generations in all nine matched cases. But the vocabulary is still three prompts and eighteen source-backed macro tokens. This proves the mechanism, not the final scale.

**On screen:** The path is genuinely generative; the vocabulary is deliberately tiny.
